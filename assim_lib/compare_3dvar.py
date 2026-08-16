from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from dataclasses import replace
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

from .config import TrainingConfig, load_json, merge_config_overrides, resolve_path
from .data import THREEDVAR_MAIN_200D_SPLITS, build_dataset, validate_temporal_split_protocol
from .evaluate import (
    INTERVAL_LEVELS,
    _inference_autocast_dtype,
    _sample_target,
    apply_sampler_normalization,
    denormalize_and_clip,
    generate_ensemble,
)
from .metrics import ensemble_crps, interval_coverage
from .model_io import load_sampler
from .runtime import get_device

DETERMINISTIC_METRICS = (
    "background_mae",
    "analysis_mean_mae",
    "mae_delta",
    "background_rmse",
    "analysis_mean_rmse",
    "rmse_delta",
    "background_mse",
    "analysis_mean_mse",
    "mse_delta",
    "background_iiee",
    "analysis_mean_iiee",
    "iiee_delta",
    *(
        f"analysis_member_{metric}_{statistic}"
        for metric in ("mae", "rmse", "mse", "iiee")
        for statistic in ("min", "mean", "max")
    ),
)

LEGACY_OVERLAPPING_SPLITS = {
    "train": THREEDVAR_MAIN_200D_SPLITS["train"],
    "valid": {
        "back_start_day": "2022-01-01",
        "back_end_day": "2022-12-31",
        "obs_start_day": "2023-01-01",
        "obs_end_day": "2023-12-31",
    },
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _prepare_output_dir(path: str | Path) -> Path:
    output_dir = Path(path).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Comparison output directory is not empty: {output_dir}. "
            "Use a new path so benchmark results cannot be mixed."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _mask_hash(mask: np.ndarray) -> str:
    packed = np.packbits(np.asarray(mask, dtype=np.uint8).reshape(-1))
    return hashlib.sha256(packed.tobytes()).hexdigest()


def _case_indices(
    dataset,
    start_date: date,
    end_date: date,
    target_hour: int,
    case_stride_days: int = 1,
) -> list[int]:
    if not hasattr(dataset, "obs_data") or not hasattr(dataset, "_num_days"):
        raise TypeError("3D-Var comparison requires M2MForecastDataset date metadata")
    hours_per_day = int(getattr(dataset, "hours_per_day", 1))
    hour_mode = str(getattr(dataset, "hour_mode", "fixed"))
    if hour_mode == "all":
        if not 0 <= target_hour < hours_per_day:
            raise ValueError(f"target_hour={target_hour} is outside [0, {hours_per_day})")
        hour_offset = target_hour
    else:
        configured_hour = int(getattr(dataset, "hour_index", target_hour))
        if configured_hour != target_hour:
            raise ValueError(
                f"Fixed dataset hour {configured_hour} does not match comparison hour {target_hour}"
            )
        hour_offset = 0

    by_date: dict[date, int] = {}
    obs_shift = int(getattr(dataset, "obs_shift", 0))
    for day_index in range(dataset._num_days()):
        target = dataset.obs_data[day_index + obs_shift].date
        if start_date <= target <= end_date:
            by_date[target] = day_index * hours_per_day + hour_offset

    if case_stride_days <= 0:
        raise ValueError("case_stride_days must be positive")
    expected_dates = [
        start_date + timedelta(days=offset)
        for offset in range(0, (end_date - start_date).days + 1, case_stride_days)
    ]
    missing = [target.isoformat() for target in expected_dates if target not in by_date]
    if missing:
        raise ValueError(f"Missing {len(missing)} benchmark target dates: {missing[:10]}")
    return [by_date[target] for target in expected_dates]


def _validate_protocol(data_config: dict[str, Any], comparison: dict[str, Any]) -> None:
    validate_temporal_split_protocol(data_config)
    fields = list(data_config.get("fields", []))
    if "siconc" not in fields:
        raise ValueError("3D-Var comparison requires the siconc field")
    siconc_channel = fields.index("siconc")
    if list(data_config.get("observed_channels", [])) != [siconc_channel]:
        raise ValueError("Exact colleague protocol must start with siconc as the only observed field")
    if int(data_config.get("assimilation_range", 0)) != 3:
        raise ValueError("Exact colleague protocol requires assimilation_range=3")
    if data_config.get("background_strategy") != "indexed":
        raise ValueError("Exact colleague protocol requires background_strategy='indexed'")
    if data_config.get("hour_mode") != "all":
        raise ValueError("Exact colleague protocol requires hour_mode='all'")
    if int(comparison.get("target_hour", -1)) != 23:
        raise ValueError("Exact colleague protocol requires target_hour=23")
    dataset_split = comparison.get("dataset_split")
    if dataset_split not in {"valid", "test"}:
        raise ValueError("Comparison dataset_split must be 'valid' or 'test'")

    mask_config = data_config.get("observation_mask", {})
    required = {
        "kind": "sral_tracks",
        "sral_transform_index": 1,
        "synthetic_probability": 0.0,
        "empty_probability": 0.0,
        "require_finite_model_values": False,
    }
    mismatches = {
        key: (mask_config.get(key), expected)
        for key, expected in required.items()
        if mask_config.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Observation policy does not match the 3D-Var protocol: {mismatches}")

    start = date.fromisoformat(str(comparison["start_date"]))
    end = date.fromisoformat(str(comparison["end_date"]))
    stride_days = int(comparison.get("case_stride_days", 1))
    if stride_days <= 0:
        raise ValueError("case_stride_days must be positive")
    selected_days = len(range(0, (end - start).days + 1, stride_days))
    expected_cases = int(comparison.get("expected_num_cases", selected_days))
    if selected_days != expected_cases:
        raise ValueError(
            "Date range and case_stride_days do not produce expected_num_cases: "
            f"selected={selected_days}, expected={expected_cases}"
        )
    split_config = data_config[dataset_split]
    split_start = date.fromisoformat(str(split_config["obs_start_day"]))
    split_end = date.fromisoformat(str(split_config["obs_end_day"]))
    if start < split_start or end > split_end:
        raise ValueError(
            f"Comparison dates {start}..{end} are outside {dataset_split} "
            f"target dates {split_start}..{split_end}"
        )


def _checkpoint_split_kind(run_metadata: dict[str, Any]) -> str:
    training_data_config = run_metadata.get("data_config")
    if not isinstance(training_data_config, dict):
        raise ValueError("Checkpoint metadata does not contain its training data_config")
    if training_data_config.get("split_protocol") == "3dvar_main_200d":
        try:
            validate_temporal_split_protocol(training_data_config)
        except ValueError as error:
            raise ValueError("Checkpoint declares an invalid 3D-Var temporal split") from error
        return "disjoint_2022_validation"

    legacy_matches = all(
        isinstance(training_data_config.get(split), dict)
        and all(training_data_config[split].get(key) == value for key, value in expected.items())
        for split, expected in LEGACY_OVERLAPPING_SPLITS.items()
    )
    if legacy_matches:
        return "legacy_overlapping_2023_validation"
    raise ValueError(
        "Checkpoint metadata contains an unknown temporal split. Use a run trained with the "
        "current m2m_2f_1y.json or the recognized legacy run."
    )


def _validate_checkpoint_training_split(run_metadata: dict[str, Any], checkpoint_name: str) -> str:
    split_kind = _checkpoint_split_kind(run_metadata)
    if split_kind == "legacy_overlapping_2023_validation" and "last" not in checkpoint_name.lower():
        raise ValueError(
            "The legacy run used 2023 for validation, so only a fixed last-epoch checkpoint "
            "is admissible. Use ema_last_model.pth, not a best checkpoint."
        )
    return split_kind


def _apply_runtime_args(args: argparse.Namespace, comparison: dict[str, Any]) -> argparse.Namespace:
    defaults = {
        "checkpoint_name": "auto",
        "output_dir": None,
        "start_date": "2023-01-01",
        "end_date": "2023-07-19",
        "target_hour": 23,
        "expected_num_cases": 200,
        "case_stride_days": 1,
        "ensemble_size": 15,
        "sample_batch_size": 5,
        "num_timesteps": 25,
        "method": "dopri5",
        "rtol": 1e-5,
        "atol": 1e-6,
        "inference_precision": "float32",
        "seed": 1234,
        "initial_noise_scale": 1.0,
        "field_protocol": "siconc-only",
        "conditioning_mode": "full",
        "track_weight": 0.5,
        "save_ensembles": False,
    }
    for key, default in defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, comparison.get(key, default))
    return args


def _select_experiment_mode(
    experiment: dict[str, Any], requested_mode: str | None
) -> tuple[dict[str, Any], str | None]:
    modes = experiment.get("modes")
    if not modes:
        if requested_mode is not None:
            raise ValueError("--mode was provided, but this config does not define modes")
        return experiment, None
    if not isinstance(modes, dict):
        raise TypeError("Experiment modes must be a JSON object")
    mode_name = requested_mode or experiment.get("default_mode")
    if mode_name not in modes:
        raise ValueError(f"Unknown mode={mode_name!r}; available modes: {sorted(modes)}")
    mode = modes[mode_name]
    if not isinstance(mode, dict):
        raise TypeError(f"Mode {mode_name!r} must be a JSON object")

    selected = dict(experiment)
    selected["data_overrides"] = merge_config_overrides(
        experiment.get("data_overrides", {}), mode.get("data_overrides")
    )
    selected["comparison"] = merge_config_overrides(experiment.get("comparison", {}), mode.get("comparison"))
    if "training" in mode:
        selected["training"] = merge_config_overrides(experiment.get("training", {}), mode.get("training"))
    return selected, str(mode_name)


def _configure_conditioning(training: TrainingConfig, args: argparse.Namespace) -> TrainingConfig:
    if args.conditioning_mode == "full":
        return replace(
            training,
            sample_cfg_mode="none",
            sample_cfg_background_scale=1.0,
            sample_cfg_observation_scale=1.0,
        )
    track_weight = float(args.track_weight)
    if not 0.0 <= track_weight <= 1.0:
        raise ValueError(f"track_weight must be within [0, 1], got {track_weight}")
    return replace(
        training,
        sample_cfg_mode="independent",
        sample_cfg_background_scale=1.0 - track_weight,
        sample_cfg_observation_scale=track_weight,
    )


def _condition_tensors(
    item: dict[str, Any],
    *,
    siconc_channel: int,
    field_protocol: str,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    background = item["background"].clone()
    background_mask = torch.ones_like(background)
    obs_values = item["obs_values"].clone()
    obs_mask = item["obs_mask"].clone()

    if field_protocol == "siconc-only":
        hidden_channels = [channel for channel in range(background.shape[0]) if channel != siconc_channel]
        if hidden_channels:
            background[hidden_channels] = 0.0
            background_mask[hidden_channels] = 0.0
            obs_values[hidden_channels] = 0.0
            obs_mask[hidden_channels] = 0.0
    elif field_protocol != "native-two-field":
        raise ValueError(f"Unknown field_protocol={field_protocol!r}")

    return {
        "background": background.unsqueeze(0).to(device, dtype=torch.float32),
        "background_mask": background_mask.unsqueeze(0).to(device, dtype=torch.float32),
        "obs_values": obs_values.unsqueeze(0).to(device, dtype=torch.float32),
        "obs_mask": obs_mask.unsqueeze(0).to(device, dtype=torch.float32),
        "water_mask": item["water_mask"].unsqueeze(0).to(device, dtype=torch.float32),
        "valid_mask": item["valid_mask"].unsqueeze(0).to(device, dtype=torch.float32),
    }


def _comparison_metrics(
    *,
    ensemble: np.ndarray,
    truth: np.ndarray,
    background: np.ndarray,
    mask: np.ndarray,
    edge_threshold: float = 0.15,
) -> dict[str, Any]:
    selector = (
        np.asarray(mask, dtype=bool)
        & np.isfinite(truth)
        & np.isfinite(background)
        & np.all(np.isfinite(ensemble), axis=0)
    )
    count = int(selector.sum())
    if count == 0:
        return {"evaluated_points": 0}

    target = np.asarray(truth, dtype=np.float64)[selector]
    baseline = np.asarray(background, dtype=np.float64)[selector]
    members = np.asarray(ensemble, dtype=np.float64)[:, selector]
    analysis = members.mean(axis=0)
    background_error = baseline - target
    analysis_error = analysis - target

    background_mae = float(np.mean(np.abs(background_error)))
    analysis_mae = float(np.mean(np.abs(analysis_error)))
    background_mse = float(np.mean(background_error**2))
    analysis_mse = float(np.mean(analysis_error**2))
    background_rmse = float(np.sqrt(background_mse))
    analysis_rmse = float(np.sqrt(analysis_mse))
    background_iiee = float(np.mean((baseline > edge_threshold) != (target > edge_threshold)))
    analysis_iiee = float(np.mean((analysis > edge_threshold) != (target > edge_threshold)))

    member_error = members - target[None, :]
    member_metrics = {
        "mae": np.mean(np.abs(member_error), axis=1),
        "rmse": np.sqrt(np.mean(member_error**2, axis=1)),
        "mse": np.mean(member_error**2, axis=1),
        "iiee": np.mean(
            (members > edge_threshold) != (target[None, :] > edge_threshold),
            axis=1,
        ),
    }

    crps = float(np.mean(ensemble_crps(members, target)))
    variance = members.var(axis=0, ddof=1 if members.shape[0] > 1 else 0)
    spread = float(np.sqrt(np.mean(variance)))
    coverage = interval_coverage(members, target, levels=INTERVAL_LEVELS)
    row: dict[str, Any] = {
        "evaluated_points": count,
        "background_mae": background_mae,
        "analysis_mean_mae": analysis_mae,
        "mae_delta": analysis_mae - background_mae,
        "background_rmse": background_rmse,
        "analysis_mean_rmse": analysis_rmse,
        "rmse_delta": analysis_rmse - background_rmse,
        "background_mse": background_mse,
        "analysis_mean_mse": analysis_mse,
        "mse_delta": analysis_mse - background_mse,
        "background_iiee": background_iiee,
        "analysis_mean_iiee": analysis_iiee,
        "iiee_delta": analysis_iiee - background_iiee,
        "analysis_crps": crps,
        "analysis_spread": spread,
        "analysis_spread_skill_ratio": spread / analysis_rmse if analysis_rmse > 0.0 else float("nan"),
    }
    for metric, values in member_metrics.items():
        row[f"analysis_member_{metric}_min"] = float(np.min(values))
        row[f"analysis_member_{metric}_mean"] = float(np.mean(values))
        row[f"analysis_member_{metric}_max"] = float(np.max(values))
    for level in INTERVAL_LEVELS:
        row[f"analysis_coverage_{round(100 * level):d}"] = float(np.mean(coverage[f"{level:g}"]))
    return row


def _aggregate_case_means(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    metric_keys = [
        *DETERMINISTIC_METRICS,
        "analysis_crps",
        "analysis_spread",
        "analysis_spread_skill_ratio",
        *[f"analysis_coverage_{round(100 * level):d}" for level in INTERVAL_LEVELS],
    ]
    for region in ("full", "track_imitation"):
        selected = [row for row in rows if row["region"] == region]
        aggregate: dict[str, Any] = {
            "region": region,
            "num_cases": len(selected),
            "cases_with_evaluated_points": sum(int(row["evaluated_points"] > 0) for row in selected),
            "evaluated_points_case_mean": float(
                np.mean([float(row["evaluated_points"]) for row in selected])
            ),
        }
        for key in metric_keys:
            values = np.asarray([float(row.get(key, float("nan"))) for row in selected])
            finite = np.isfinite(values)
            aggregate[key] = float(np.mean(values))
            aggregate[f"{key}_finite_case_mean"] = (
                float(np.mean(values[finite])) if np.any(finite) else float("nan")
            )
            aggregate[f"{key}_finite_cases"] = int(finite.sum())
        output.append(aggregate)
    return output


def _track_days(dataset, target_date: date) -> list[dict[str, Any]]:
    result = []
    for offset in range(int(dataset.assimilation_range)):
        track_date = target_date - timedelta(days=offset)
        paths = dataset.sral_records.get(track_date, [])
        result.append(
            {
                "offset": offset,
                "date": track_date.isoformat(),
                "num_sral_files": len(paths),
                "sral_files": [str(path) for path in paths],
            }
        )
    return result


class RunStatus:
    def __init__(self, output_dir: Path, total_cases: int):
        self.path = output_dir / "run_status.json"
        self.total_cases = total_cases
        self.started_at = datetime.now()
        self.started = time.monotonic()
        self.write("running", 0)

    def write(self, status: str, completed_cases: int, **last: Any) -> None:
        elapsed = time.monotonic() - self.started
        rate = completed_cases / elapsed if elapsed > 0.0 else 0.0
        remaining = (self.total_cases - completed_cases) / rate if rate > 0.0 else None
        payload = {
            "status": status,
            "started_at": self.started_at.isoformat(),
            "updated_at": datetime.now().isoformat(),
            "completed_cases": completed_cases,
            "total_cases": self.total_cases,
            "progress_percent": 100.0 * completed_cases / max(self.total_cases, 1),
            "elapsed_hours": elapsed / 3600.0,
            "estimated_remaining_hours": remaining / 3600.0 if remaining is not None else None,
            "last": _json_safe(last),
        }
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    experiment_path = Path(args.config).expanduser().resolve()
    experiment = load_json(experiment_path)
    experiment, selected_mode = _select_experiment_mode(experiment, args.mode)
    args = _apply_runtime_args(args, experiment.get("comparison", {}))
    if not np.isfinite(float(args.initial_noise_scale)) or float(args.initial_noise_scale) <= 0.0:
        raise ValueError("initial_noise_scale must be finite and positive")
    if not args.run_dir:
        raise ValueError("--run-dir is required")

    data_config_path = resolve_path(experiment["data_config"], experiment_path.parent).resolve()
    model_config_path = resolve_path(experiment["model_config"], experiment_path.parent).resolve()
    data_config = merge_config_overrides(load_json(data_config_path), experiment.get("data_overrides"))
    comparison = {
        **experiment.get("comparison", {}),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "target_hour": args.target_hour,
        "expected_num_cases": args.expected_num_cases,
        "case_stride_days": args.case_stride_days,
    }
    _validate_protocol(data_config, comparison)

    fields = list(data_config["fields"])
    siconc_channel = fields.index("siconc")
    if args.field_protocol == "siconc-only":
        data_config["observed_channels"] = [siconc_channel]
    else:
        data_config["observed_channels"] = list(range(len(fields)))

    dataset_split = str(comparison["dataset_split"])
    dataset = build_dataset(data_config, split=dataset_split)
    start_date = date.fromisoformat(str(args.start_date))
    end_date = date.fromisoformat(str(args.end_date))
    case_indices = _case_indices(
        dataset,
        start_date,
        end_date,
        int(args.target_hour),
        int(args.case_stride_days),
    )
    if len(case_indices) != int(args.expected_num_cases):
        raise ValueError(f"Selected {len(case_indices)} cases, expected {args.expected_num_cases}")

    run_dir = Path(args.run_dir).expanduser().resolve()
    run_metadata_path = run_dir / "metadata.json"
    if not run_metadata_path.is_file():
        raise FileNotFoundError(run_metadata_path)
    run_metadata = load_json(run_metadata_path)
    split_kind = _checkpoint_split_kind(run_metadata)
    if args.checkpoint_name == "auto":
        args.checkpoint_name = (
            "ema_best_model.pth" if split_kind == "disjoint_2022_validation" else "ema_last_model.pth"
        )
    split_kind = _validate_checkpoint_training_split(run_metadata, args.checkpoint_name)
    checkpoint_path = run_dir / args.checkpoint_name
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    raw_model_config = load_json(model_config_path)
    model_config = {**raw_model_config, **experiment.get("training", {})}
    training = _configure_conditioning(TrainingConfig.from_dict(model_config), args)
    device = torch.device(args.device or get_device())
    sampler = load_sampler(str(run_dir), args.checkpoint_name, model_config, device=device)
    means, stds = apply_sampler_normalization(dataset, sampler, data_config)
    autocast_dtype, precision = _inference_autocast_dtype(str(args.inference_precision), training, device)

    output_dir = _prepare_output_dir(
        args.output_dir
        or run_dir
        / "evaluation"
        / f"3dvar_comparison_{args.field_protocol}_{args.conditioning_mode}_{start_date}_{end_date}"
    )
    samples_dir = output_dir / "samples"
    if args.save_ensembles:
        samples_dir.mkdir()

    per_case_rows: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    status = RunStatus(output_dir, len(case_indices))
    total_samples = len(case_indices) * int(args.ensemble_size)
    sample_target = _sample_target(training)

    with tqdm(total=total_samples, desc="3D-Var protocol comparison", unit="sample") as progress:
        try:
            for case_order, dataset_index in enumerate(case_indices):
                item = dataset[dataset_index]
                target_date = date.fromisoformat(str(item["meta"]["target_date"]))
                expected_target = start_date + timedelta(days=case_order * int(args.case_stride_days))
                if target_date != expected_target:
                    raise AssertionError(
                        f"Case order {case_order}: target={target_date}, expected={expected_target}"
                    )
                if int(item["meta"]["hour"]) != int(args.target_hour):
                    raise AssertionError(f"Wrong target hour in {item['meta']}")
                background_date = date.fromisoformat(str(item["meta"]["background_date"]))
                day_index, _ = dataset._resolve_index(dataset_index)
                expected_background = dataset.back_data[day_index].date
                if background_date != expected_background:
                    raise AssertionError(
                        f"Indexed background mismatch: background={background_date}, "
                        f"expected={expected_background}, target={target_date}"
                    )
                expected_offset = (target_date - expected_background).days
                if int(item["meta"]["background_offset_days"]) != expected_offset:
                    raise AssertionError(
                        "Background offset metadata does not match the indexed records: "
                        f"metadata={item['meta']['background_offset_days']}, "
                        f"expected={expected_offset}"
                    )
                if item["meta"].get("mask_kind") != "sral_tracks":
                    raise AssertionError(
                        f"Expected real sral_tracks, got mask_kind={item['meta'].get('mask_kind')!r}"
                    )

                expected_condition_tensor = dataset.sral_spatial_mask(
                    target_date,
                    day_offsets=range(dataset.assimilation_range),
                    transform_index=1,
                )
                expected_condition_mask = (
                    np.zeros(dataset.image_size, dtype=bool)
                    if expected_condition_tensor is None
                    else expected_condition_tensor.detach().cpu().numpy() > 0.5
                )
                actual_condition_mask = item["obs_mask"][siconc_channel].detach().cpu().numpy() > 0.5
                if not np.array_equal(actual_condition_mask, expected_condition_mask):
                    differing = int(np.count_nonzero(actual_condition_mask != expected_condition_mask))
                    raise AssertionError(
                        f"Conditioning mask differs from d,d-1,d-2 rule-1 SRAL union "
                        f"at {differing} pixels for {target_date}"
                    )

                tensors = _condition_tensors(
                    item,
                    siconc_channel=siconc_channel,
                    field_protocol=args.field_protocol,
                    device=device,
                )
                if args.field_protocol == "siconc-only":
                    other = [channel for channel in range(len(fields)) if channel != siconc_channel]
                    if other and (
                        torch.any(tensors["obs_mask"][:, other] > 0)
                        or torch.any(tensors["background_mask"][:, other] > 0)
                    ):
                        raise AssertionError("Non-siconc information leaked into strict comparison")

                progress.set_postfix(case=f"{case_order + 1}/{len(case_indices)}")
                initial_noise_hashes: list[str] = []
                scaled_initial_noise_hashes: list[str] = []
                generated = generate_ensemble(
                    sampler=sampler,
                    background=tensors["background"],
                    background_mask=tensors["background_mask"],
                    obs_values=tensors["obs_values"],
                    obs_mask=tensors["obs_mask"],
                    water_mask=tensors["water_mask"],
                    valid_mask=tensors["valid_mask"],
                    config=training,
                    ensemble_size=int(args.ensemble_size),
                    sample_batch_size=int(args.sample_batch_size),
                    num_timesteps=int(args.num_timesteps),
                    method=str(args.method),
                    device=device,
                    seed=int(args.seed),
                    case_order=case_order,
                    sample_target=sample_target,
                    rtol=float(args.rtol),
                    atol=float(args.atol),
                    autocast_dtype=autocast_dtype,
                    progress=progress,
                    initial_noise_hashes=initial_noise_hashes,
                    initial_noise_scale=float(args.initial_noise_scale),
                    scaled_initial_noise_hashes=scaled_initial_noise_hashes,
                )
                if len(initial_noise_hashes) != int(args.ensemble_size):
                    raise AssertionError("Initial-noise accounting is incomplete")
                if len(scaled_initial_noise_hashes) != int(args.ensemble_size):
                    raise AssertionError("Scaled initial-noise accounting is incomplete")

                ensemble = denormalize_and_clip(generated, means, stds, siconc_channel)[:, siconc_channel]
                truth = None
                if not args.sealed_raw_only:
                    truth = denormalize_and_clip(item["truth"].unsqueeze(0), means, stds, siconc_channel)[
                        0, siconc_channel
                    ]
                background = denormalize_and_clip(
                    item["background"].unsqueeze(0), means, stds, siconc_channel
                )[0, siconc_channel]
                valid = item["valid_mask"][siconc_channel].detach().cpu().numpy() > 0.5
                conditioning_mask = actual_condition_mask
                next_track_date = target_date + timedelta(days=1)
                next_mask_tensor = dataset.sral_spatial_mask(
                    next_track_date,
                    day_offsets=(0,),
                    transform_index=1,
                )
                next_track_mask = (
                    np.zeros_like(valid)
                    if next_mask_tensor is None
                    else next_mask_tensor.detach().cpu().numpy() > 0.5
                )
                next_track_mask &= valid

                common = {
                    "case_order": case_order,
                    "dataset_index": dataset_index,
                    "background_date": item["meta"]["background_date"],
                    "target_date": target_date.isoformat(),
                    "hour": int(args.target_hour),
                    "conditioning_obs_count": int(conditioning_mask.sum()),
                    "track_imitation_date": next_track_date.isoformat(),
                    "track_imitation_count": int(next_track_mask.sum()),
                }
                if args.sealed_raw_only:
                    per_case_rows.append(
                        {
                            **common,
                            "region": "raw_sampling_only",
                            "outcome_metrics_computed": False,
                        }
                    )
                else:
                    assert truth is not None
                    for region, mask in (
                        ("full", valid),
                        ("track_imitation", next_track_mask),
                    ):
                        per_case_rows.append(
                            {
                                **common,
                                "region": region,
                                **_comparison_metrics(
                                    ensemble=ensemble,
                                    truth=truth,
                                    background=background,
                                    mask=mask,
                                ),
                            }
                        )

                track_days = _track_days(dataset, target_date)
                case_metadata = {
                    **common,
                    "conditioning_track_days": track_days,
                    "conditioning_mask_sha256": _mask_hash(conditioning_mask),
                    "track_imitation_mask_sha256": _mask_hash(next_track_mask),
                    "mask_kind": item["meta"].get("mask_kind"),
                    "sral_files_used": item["meta"].get("sral_files_used"),
                    "empty_obs_days": item["meta"].get("empty_obs_days"),
                    "initial_noise_sha256": initial_noise_hashes,
                    "scaled_initial_noise_sha256": scaled_initial_noise_hashes,
                }
                cases.append(case_metadata)

                if args.save_ensembles:
                    obs_values = denormalize_and_clip(
                        item["obs_values"].unsqueeze(0), means, stds, siconc_channel
                    )[0, siconc_channel]
                    sample_path = samples_dir / f"{case_order:03d}_{target_date.isoformat()}_h23.npz"
                    if args.sealed_raw_only:
                        np.savez_compressed(
                            sample_path,
                            analysis_ensemble=ensemble.astype(np.float32),
                            valid_mask=valid,
                        )
                    else:
                        assert truth is not None
                        np.savez_compressed(
                            sample_path,
                            analysis_ensemble=ensemble.astype(np.float32),
                            analysis_mean=ensemble.mean(axis=0).astype(np.float32),
                            truth=truth.astype(np.float32),
                            background=background.astype(np.float32),
                            obs_values=obs_values.astype(np.float32),
                            conditioning_mask=conditioning_mask,
                            track_imitation_mask=next_track_mask,
                            valid_mask=valid,
                        )

                _write_csv(output_dir / "per_case_metrics.csv", per_case_rows)
                status.write(
                    "running",
                    case_order + 1,
                    target_date=target_date,
                    conditioning_obs_count=int(conditioning_mask.sum()),
                    track_imitation_count=int(next_track_mask.sum()),
                )
        except Exception as error:
            status.write("failed", len(cases), error=str(error))
            raise

    aggregate_rows = (
        [
            {
                "region": "raw_sampling_only",
                "num_cases": len(per_case_rows),
                "outcome_metrics_computed": False,
            }
        ]
        if args.sealed_raw_only
        else _aggregate_case_means(per_case_rows)
    )
    _write_csv(output_dir / "per_case_metrics.csv", per_case_rows)
    _write_csv(output_dir / "aggregate_case_mean_metrics.csv", aggregate_rows)
    (output_dir / "aggregate_case_mean_metrics.json").write_text(
        json.dumps(_json_safe(aggregate_rows), indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "cases.json").write_text(
        json.dumps(_json_safe(cases), indent=2) + "\n",
        encoding="utf-8",
    )

    sampler_metadata = getattr(sampler, "metadata", {}) or {}
    metadata = {
        "protocol": (
            "sealed external chronological holdout raw sampling"
            if args.sealed_raw_only
            else "LiubovAnt/3d_var model-to-model main_200d"
        ),
        "colleague_reference": {
            "branch": "origin/vae_exp",
            "config": "config/assimilation/model2model/main_200d/var_100_main_200d.yaml",
            "data_config": "config/data/model/var_200d.yaml",
            "clearml_project": "m2m_main_200d",
            "clearml_task": "3d_var_baseline_var_100_main_200d",
        },
        "experiment_config": str(experiment_path),
        "experiment_mode": selected_mode,
        "checkpoint": str(checkpoint_path),
        "dataset_split": dataset_split,
        "temporal_split_protocol": (
            "external_chronological_holdout" if args.sealed_raw_only else "3dvar_main_200d"
        ),
        "temporal_splits": None if args.sealed_raw_only else THREEDVAR_MAIN_200D_SPLITS,
        "checkpoint_training_split_kind": split_kind,
        "checkpoint_selection": (
            "fixed final epoch; legacy 2023 validation was not used for checkpoint selection"
            if split_kind == "legacy_overlapping_2023_validation"
            else (
                "selected using the disjoint 2022 validation split"
                if "best" in checkpoint_path.name.lower()
                else "fixed final training epoch"
            )
        ),
        "point_estimator": "mean of the generated ensemble",
        "start_date": start_date,
        "end_date": end_date,
        "num_cases": len(case_indices),
        "case_stride_days": int(args.case_stride_days),
        "target_hour": int(args.target_hour),
        "assimilation_range": int(dataset.assimilation_range),
        "observation_source": "M2M values on real SRAL footprints",
        "sral_transform_index": 1,
        "observed_fields": [fields[index] for index in dataset.observed_channels],
        "field_protocol": args.field_protocol,
        "strict_information_note": (
            "siconc-only hides both observation and background information for every other field"
            if args.field_protocol == "siconc-only"
            else "native-two-field exposes the model's normal multivariate conditions"
        ),
        "conditioning_mode": args.conditioning_mode,
        "cfg_mode": training.sample_cfg_mode,
        "cfg_background_scale": training.sample_cfg_background_scale,
        "cfg_observation_scale": training.sample_cfg_observation_scale,
        "ensemble_size": int(args.ensemble_size),
        "sample_batch_size": int(args.sample_batch_size),
        "num_timesteps": int(args.num_timesteps),
        "method": args.method,
        "rtol": float(args.rtol),
        "atol": float(args.atol),
        "inference_precision": precision,
        "seed": int(args.seed),
        "initial_noise_scale": float(args.initial_noise_scale),
        "normalization_means": means,
        "normalization_stds": stds,
        "normalization_source": (
            "checkpoint_metadata"
            if sampler_metadata.get("normalization_means") is not None
            else "data_config"
        ),
        "aggregation": (
            "Arithmetic mean of per-day metrics, matching the colleague ClearML logger. "
            "Columns ending in _finite_case_mean are diagnostic nan-safe alternatives."
        ),
        "track_imitation": (
            "Same-day dense M2M truth evaluated only on the next day's rule-1 SRAL footprint"
        ),
        "edge_threshold": 0.15,
        "save_ensembles": bool(args.save_ensembles),
        "sealed_raw_only": bool(args.sealed_raw_only),
        "outcome_metrics_computed": not bool(args.sealed_raw_only),
        "truth_saved_with_raw_samples": not bool(args.sealed_raw_only),
        "cases_file": "cases.json",
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(_json_safe(metadata), indent=2) + "\n",
        encoding="utf-8",
    )
    status.write("completed", len(case_indices), target_date=end_date)
    result = {
        "output_dir": str(output_dir),
        "num_cases": len(case_indices),
        "per_case_metrics": str(output_dir / "per_case_metrics.csv"),
        "aggregate_metrics": str(output_dir / "aggregate_case_mean_metrics.csv"),
        "metadata": str(output_dir / "metadata.json"),
    }
    print(json.dumps(result, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate flow matching under the LiubovAnt/3d_var model-to-model protocol."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", default=None)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint-name", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--target-hour", type=int, default=None)
    parser.add_argument("--expected-num-cases", type=int, default=None)
    parser.add_argument("--case-stride-days", type=int, default=None)
    parser.add_argument("--ensemble-size", type=int, default=None)
    parser.add_argument("--sample-batch-size", type=int, default=None)
    parser.add_argument("--num-timesteps", type=int, default=None)
    parser.add_argument("--method", choices=("euler", "dopri5"), default=None)
    parser.add_argument("--rtol", type=float, default=None)
    parser.add_argument("--atol", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--inference-precision",
        choices=("auto", "float32", "float16", "bfloat16"),
        default=None,
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--initial-noise-scale", type=float, default=None)
    parser.add_argument(
        "--field-protocol",
        choices=("siconc-only", "native-two-field"),
        default=None,
    )
    parser.add_argument(
        "--conditioning-mode",
        choices=("full", "independent-balance"),
        default=None,
    )
    parser.add_argument("--track-weight", type=float, default=None)
    parser.add_argument(
        "--save-ensembles",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--sealed-raw-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
