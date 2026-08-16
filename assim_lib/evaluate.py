from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import warnings
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

from .clearml_tracking import ClearMLTracker
from .config import TrainingConfig, load_json, merge_config_overrides, resolve_path
from .data import build_dataset
from .metrics import ensemble_crps, interval_coverage
from .model_io import load_sampler
from .runtime import get_device
from .transforms import channel_denormalize

INTERVAL_LEVELS = (0.5, 0.8, 0.9, 0.95)


def denormalize_and_clip(
    tensor: torch.Tensor,
    means: list[float],
    stds: list[float],
    concentration_channel: int | None,
) -> np.ndarray:
    """Convert a normalized tensor to physical units and constrain concentration."""
    physical = channel_denormalize(tensor.detach().cpu().float(), means, stds).numpy()
    if concentration_channel is not None:
        physical[..., concentration_channel, :, :] = np.clip(
            physical[..., concentration_channel, :, :], 0.0, 1.0
        )
    return physical


def validation_case_indices(dataset, stride_days: int, max_cases: int | None = None) -> list[int]:
    """Apply the same daily case selection and target-hour shift used during validation."""
    limit = int(max_cases) if max_cases is not None and max_cases > 0 else len(dataset)
    if hasattr(dataset, "strided_case_indices"):
        indices = list(dataset.strided_case_indices(limit, stride_days))
    else:
        indices = list(range(min(limit, len(dataset))))
    hours_per_day = int(getattr(dataset, "hours_per_day", 1))
    if hours_per_day > 1:
        hour = min(max(0, int(getattr(dataset, "hour_index", 0))), hours_per_day - 1)
        indices = [(idx // hours_per_day) * hours_per_day + hour for idx in indices]
    return indices


def _safe_skill(analysis: float, background: float) -> float:
    return 1.0 - analysis / background if background > 0.0 else float("nan")


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0.0 else float("nan")


def _region_masks(valid_mask: np.ndarray, obs_mask: np.ndarray, channel: int) -> dict[str, np.ndarray]:
    valid = np.asarray(valid_mask[channel], dtype=bool)
    observed = valid & np.asarray(obs_mask[channel], dtype=bool)
    return {
        "full": valid,
        "observed": observed,
        "unobserved": valid & ~observed,
    }


class PhysicalMetricAccumulator:
    """Accumulates deterministic-baseline and ensemble metrics in physical units."""

    def __init__(self, interval_levels: tuple[float, ...] = INTERVAL_LEVELS):
        self.interval_levels = interval_levels
        self._totals: dict[tuple[str, str], defaultdict[str, float]] = {}

    def add(
        self,
        field: str,
        region: str,
        ensemble: np.ndarray,
        truth: np.ndarray,
        background: np.ndarray,
        mask: np.ndarray,
    ) -> dict[str, Any] | None:
        selector = np.asarray(mask, dtype=bool)
        count = int(selector.sum())
        if count == 0:
            return None

        target = np.asarray(truth, dtype=np.float64)[selector]
        baseline = np.asarray(background, dtype=np.float64)[selector]
        members = np.asarray(ensemble, dtype=np.float64)[:, selector]
        mean = members.mean(axis=0)
        mean_error = mean - target
        background_error = baseline - target
        member_error = members - target[None, :]
        crps = ensemble_crps(members, target)
        variance = members.var(axis=0, ddof=1 if members.shape[0] > 1 else 0)
        coverage = interval_coverage(members, target, levels=self.interval_levels)

        analysis_mae = float(np.mean(np.abs(mean_error)))
        analysis_rmse = float(np.sqrt(np.mean(mean_error**2)))
        background_mae = float(np.mean(np.abs(background_error)))
        background_rmse = float(np.sqrt(np.mean(background_error**2)))
        analysis_crps = float(np.mean(crps))
        spread = float(np.sqrt(np.mean(variance)))
        row: dict[str, Any] = {
            "field": field,
            "region": region,
            "evaluated_points": count,
            "analysis_mean_mae": analysis_mae,
            "analysis_mean_rmse": analysis_rmse,
            "analysis_mean_bias": float(np.mean(mean_error)),
            "analysis_member_mae_pooled": float(np.mean(np.abs(member_error))),
            "analysis_member_rmse_pooled": float(np.sqrt(np.mean(member_error**2))),
            "background_mae": background_mae,
            "background_rmse": background_rmse,
            "background_bias": float(np.mean(background_error)),
            "analysis_mae_skill": _safe_skill(analysis_mae, background_mae),
            "analysis_rmse_skill": _safe_skill(analysis_rmse, background_rmse),
            "analysis_crps": analysis_crps,
            "background_crps": background_mae,
            "analysis_crps_skill": _safe_skill(analysis_crps, background_mae),
            "analysis_spread": spread,
            "analysis_spread_skill_ratio": _safe_ratio(spread, analysis_rmse),
        }
        for level in self.interval_levels:
            key = f"analysis_coverage_{round(100 * level):d}"
            row[key] = float(np.mean(coverage[f"{level:g}"]))

        totals = self._totals.setdefault((field, region), defaultdict(float))
        totals["evaluated_points"] += count
        totals["analysis_abs"] += float(np.abs(mean_error).sum())
        totals["analysis_sq"] += float((mean_error**2).sum())
        totals["analysis_bias"] += float(mean_error.sum())
        totals["member_abs"] += float(np.abs(member_error).sum())
        totals["member_sq"] += float((member_error**2).sum())
        totals["member_points"] += float(member_error.size)
        totals["background_abs"] += float(np.abs(background_error).sum())
        totals["background_sq"] += float((background_error**2).sum())
        totals["background_bias"] += float(background_error.sum())
        totals["crps"] += float(crps.sum())
        totals["variance"] += float(variance.sum())
        for level in self.interval_levels:
            totals[f"coverage_{round(100 * level):d}"] += float(coverage[f"{level:g}"].sum())
        return row

    def finalize(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for (field, region), totals in sorted(self._totals.items()):
            count = int(totals["evaluated_points"])
            analysis_mae = totals["analysis_abs"] / count
            analysis_rmse = math.sqrt(totals["analysis_sq"] / count)
            background_mae = totals["background_abs"] / count
            background_rmse = math.sqrt(totals["background_sq"] / count)
            crps = totals["crps"] / count
            spread = math.sqrt(totals["variance"] / count)
            row: dict[str, Any] = {
                "field": field,
                "region": region,
                "evaluated_points": count,
                "analysis_mean_mae": analysis_mae,
                "analysis_mean_rmse": analysis_rmse,
                "analysis_mean_bias": totals["analysis_bias"] / count,
                "analysis_member_mae_pooled": totals["member_abs"] / totals["member_points"],
                "analysis_member_rmse_pooled": math.sqrt(totals["member_sq"] / totals["member_points"]),
                "background_mae": background_mae,
                "background_rmse": background_rmse,
                "background_bias": totals["background_bias"] / count,
                "analysis_mae_skill": _safe_skill(analysis_mae, background_mae),
                "analysis_rmse_skill": _safe_skill(analysis_rmse, background_rmse),
                "analysis_crps": crps,
                "background_crps": background_mae,
                "analysis_crps_skill": _safe_skill(crps, background_mae),
                "analysis_spread": spread,
                "analysis_spread_skill_ratio": _safe_ratio(spread, analysis_rmse),
            }
            for level in self.interval_levels:
                pct = round(100 * level)
                row[f"analysis_coverage_{pct:d}"] = totals[f"coverage_{pct:d}"] / count
            rows.append(row)
        return rows


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _sample_target(config: TrainingConfig) -> str:
    return "residual" if config.training_objective == "residual_flow" else "state"


def _seed_member(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _inference_autocast_dtype(
    requested_precision: str,
    config: TrainingConfig,
    device: torch.device,
) -> tuple[torch.dtype | None, str]:
    precision = config.mixed_precision if requested_precision == "auto" else requested_precision
    precision = str(precision).lower()
    if precision in ("no", "float32", "fp32") or device.type != "cuda":
        return None, "float32"
    if precision in ("bf16", "bfloat16"):
        return torch.bfloat16, "bfloat16"
    if precision in ("fp16", "float16"):
        return torch.float16, "float16"
    raise ValueError(f"Unknown inference precision: {precision!r}")


def _apply_evaluation_config(args: argparse.Namespace, experiment: dict) -> argparse.Namespace:
    config = experiment.get("evaluation", {})
    defaults = {
        "checkpoint_name": "auto",
        "output_dir": "",
        "split": "valid",
        "stride_days": 15,
        "ensemble_size": 15,
        "sample_batch_size": 16,
        "max_cases": None,
        "num_timesteps": None,
        "method": "euler",
        "rtol": None,
        "atol": None,
        "device": "",
        "inference_precision": "auto",
        "seed": 1234,
        "save_ensembles": False,
        "cfg_mode": None,
        "cfg_background_scale": None,
        "cfg_observation_scale": None,
    }
    for name, default in defaults.items():
        if getattr(args, name, None) is None:
            setattr(args, name, config.get(name, default))
    if args.run_dir is None:
        args.run_dir = config.get("run_dir")
    return args


def _start_clearml(
    experiment: dict,
    args: argparse.Namespace,
    data_config: dict,
    model_config: dict,
) -> ClearMLTracker | None:
    config = experiment.get("clearml", {})
    enabled = (
        bool(config.get("enabled", False))
        if getattr(args, "clearml_enabled", None) is None
        else bool(args.clearml_enabled)
    )
    if not enabled:
        return None
    tracker = ClearMLTracker(
        project_name=config.get("project_name", experiment.get("project_name", "assim_evaluation")),
        task_name=config.get("task_name", experiment.get("task_name", "physical_evaluation")),
        tags=config.get("tags", []),
        output_uri=config.get("output_uri"),
        env_path=config.get("env_path"),
    )
    tracker.connect("evaluation_config", _json_safe(experiment.get("evaluation", {})))
    tracker.connect("experiment_config", _json_safe(experiment))
    tracker.connect("data_config", _json_safe(data_config))
    tracker.connect("model_config", _json_safe(model_config))
    tracker.connect(
        "evaluation_runtime",
        _json_safe(
            {
                "run_dir": args.run_dir,
                "checkpoint_name": args.checkpoint_name,
                "output_dir": args.output_dir,
                "split": args.split,
                "stride_days": args.stride_days,
                "ensemble_size": args.ensemble_size,
                "sample_batch_size": args.sample_batch_size,
                "max_cases": args.max_cases,
                "num_timesteps": args.num_timesteps,
                "method": args.method,
                "rtol": args.rtol,
                "atol": args.atol,
                "device": args.device,
                "inference_precision": args.inference_precision,
                "seed": args.seed,
                "cfg_mode": args.cfg_mode,
                "cfg_background_scale": args.cfg_background_scale,
                "cfg_observation_scale": args.cfg_observation_scale,
                "save_ensembles": args.save_ensembles,
                "clearml_enabled": enabled,
            }
        ),
    )
    return tracker


def _publish_clearml_results(
    tracker: ClearMLTracker,
    output_dir: Path,
    aggregate_rows: list[dict[str, Any]],
    metadata: dict[str, Any],
    upload_samples: bool,
) -> None:
    tracker.report_single_value("num_cases", float(metadata["num_cases"]))
    for row in aggregate_rows:
        series = f"{row['field']}/{row['region']}"
        for metric, value in row.items():
            if metric in ("field", "region") or not isinstance(value, (int, float)):
                continue
            number = float(value)
            if math.isfinite(number):
                tracker.report_scalar(f"physical/{metric}", series, number, 0)
    tracker.upload_artifact("evaluation_metadata", output_dir / "metadata.json")
    tracker.upload_artifact("aggregate_metrics_csv", output_dir / "aggregate_metrics.csv")
    tracker.upload_artifact("aggregate_metrics_json", output_dir / "aggregate_metrics.json")
    tracker.upload_artifact("per_case_metrics", output_dir / "per_case_metrics.csv")
    if upload_samples:
        tracker.upload_artifact("physical_samples", output_dir / "samples")


def generate_ensemble(
    sampler,
    background: torch.Tensor,
    obs_values: torch.Tensor,
    obs_mask: torch.Tensor,
    water_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    config: TrainingConfig,
    ensemble_size: int,
    sample_batch_size: int,
    num_timesteps: int,
    method: str,
    device: torch.device,
    seed: int,
    case_order: int,
    sample_target: str,
    rtol: float = 1e-3,
    atol: float = 1e-4,
    autocast_dtype: torch.dtype | None = None,
    progress=None,
    background_mask: torch.Tensor | None = None,
    initial_noise_hashes: list[str] | None = None,
    initial_noise_scale: float = 1.0,
    scaled_initial_noise_hashes: list[str] | None = None,
) -> torch.Tensor:
    """Generate one conditional ensemble in GPU-friendly member batches."""
    initial_noise_scale = float(initial_noise_scale)
    if not math.isfinite(initial_noise_scale) or initial_noise_scale <= 0.0:
        raise ValueError("initial_noise_scale must be finite and positive")
    samples: list[torch.Tensor] = []
    for start in range(0, ensemble_size, sample_batch_size):
        batch_size = min(sample_batch_size, ensemble_size - start)
        initial_noise = None
        if config.sample_start_mode in ("noise", "background"):
            noise = []
            for member in range(start, start + batch_size):
                _seed_member(seed + case_order * ensemble_size + member)
                member_noise = torch.randn_like(background)
                if initial_noise_hashes is not None or scaled_initial_noise_hashes is not None:
                    base_canonical = (
                        member_noise.detach().cpu().to(dtype=torch.float32).contiguous().numpy()
                    )
                    if initial_noise_hashes is not None:
                        initial_noise_hashes.append(
                            hashlib.sha256(base_canonical.tobytes()).hexdigest()
                        )
                member_noise = member_noise * initial_noise_scale
                if scaled_initial_noise_hashes is not None:
                    scaled_canonical = (
                        member_noise.detach().cpu().to(dtype=torch.float32).contiguous().numpy()
                    )
                    scaled_initial_noise_hashes.append(
                        hashlib.sha256(scaled_canonical.tobytes()).hexdigest()
                    )
                noise.append(member_noise)
            initial_noise = torch.cat(noise, dim=0)

        autocast_context = (
            torch.autocast(device_type=device.type, dtype=autocast_dtype)
            if autocast_dtype is not None
            else nullcontext()
        )
        with autocast_context:
            sample = sampler.sample_conditioned(
                background=background.expand(batch_size, -1, -1, -1),
                background_mask=(
                    background_mask.expand(batch_size, -1, -1, -1) if background_mask is not None else None
                ),
                obs_values=obs_values.expand(batch_size, -1, -1, -1),
                obs_mask=obs_mask.expand(batch_size, -1, -1, -1),
                water_mask=water_mask.expand(batch_size, -1, -1, -1),
                size=tuple(config.image_size),
                num_timesteps=num_timesteps,
                device=device,
                method=method,
                rtol=rtol,
                atol=atol,
                start_mode=config.sample_start_mode,
                start_noise_level=config.sample_start_noise_level,
                enforce_observations=config.sample_enforce_observations,
                valid_mask=valid_mask.expand(batch_size, -1, -1, -1),
                obs_guidance_scale=config.sample_obs_guidance_scale,
                obs_guidance_eps=config.sample_obs_guidance_eps,
                initial_noise=initial_noise,
                sample_target=sample_target,
                memory_efficient_euler=method == "euler",
                cfg_mode=config.sample_cfg_mode,
                cfg_background_scale=config.sample_cfg_background_scale,
                cfg_observation_scale=config.sample_cfg_observation_scale,
            )
        samples.append(sample.cpu())
        if progress is not None:
            progress.update(batch_size)
    return torch.cat(samples, dim=0)


def apply_sampler_normalization(dataset, sampler, data_config: dict) -> tuple[list[float], list[float]]:
    configured_means = list(getattr(dataset, "means", data_config.get("means", [])))
    configured_stds = list(getattr(dataset, "stds", data_config.get("stds", [])))
    metadata = getattr(sampler, "metadata", {}) or {}
    means = metadata.get("normalization_means")
    stds = metadata.get("normalization_stds")
    if means is None or stds is None:
        return configured_means, configured_stds
    means = list(means)
    stds = list(stds)
    if configured_means != means or configured_stds != stds:
        warnings.warn(
            "Evaluation data_config normalization differs from checkpoint metadata; "
            "using checkpoint normalization.",
            stacklevel=2,
        )
    dataset.means = means
    dataset.stds = stds
    if hasattr(dataset, "config"):
        dataset.config["means"] = means
        dataset.config["stds"] = stds
    data_config["means"] = means
    data_config["stds"] = stds
    return means, stds


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    experiment_path = Path(args.config).resolve()
    experiment = load_json(experiment_path)
    args = _apply_evaluation_config(args, experiment)
    if not args.run_dir:
        raise ValueError("--run-dir is required unless evaluation.run_dir is provided in the config.")
    if args.stride_days <= 0:
        raise ValueError("--stride-days must be positive.")
    if args.ensemble_size <= 0:
        raise ValueError("--ensemble-size must be positive.")
    if args.sample_batch_size <= 0:
        raise ValueError("--sample-batch-size must be positive.")
    if args.num_timesteps is not None and args.num_timesteps <= 0:
        raise ValueError("--num-timesteps must be positive.")

    data_config_path = resolve_path(experiment["data_config"], experiment_path.parent).resolve()
    model_config_path = resolve_path(experiment["model_config"], experiment_path.parent).resolve()
    data_config = merge_config_overrides(load_json(data_config_path), experiment.get("data_overrides"))
    raw_model_config = load_json(model_config_path)
    model_config = {**raw_model_config, **experiment.get("training", {})}
    training = TrainingConfig.from_dict(model_config)
    scale_override_requested = args.cfg_background_scale is not None or args.cfg_observation_scale is not None
    if args.cfg_mode is not None:
        training.sample_cfg_mode = str(args.cfg_mode)
    elif scale_override_requested and training.sample_cfg_mode == "none":
        training.sample_cfg_mode = "background_delta"
    if args.cfg_background_scale is not None:
        training.sample_cfg_background_scale = float(args.cfg_background_scale)
    if args.cfg_observation_scale is not None:
        training.sample_cfg_observation_scale = float(args.cfg_observation_scale)
    device = torch.device(args.device or get_device())
    autocast_dtype, inference_precision = _inference_autocast_dtype(
        args.inference_precision, training, device
    )

    dataset = build_dataset(data_config, split=args.split)
    case_indices = validation_case_indices(dataset, args.stride_days, args.max_cases)
    if not case_indices:
        raise ValueError("No evaluation cases were selected.")

    run_dir = Path(args.run_dir).resolve()
    checkpoint_path = run_dir / args.checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    sampler = load_sampler(str(run_dir), args.checkpoint_name, model_config, device=device)
    means, stds = apply_sampler_normalization(dataset, sampler, data_config)
    sampler_metadata = getattr(sampler, "metadata", {}) or {}
    normalization_source = (
        "checkpoint_metadata"
        if sampler_metadata.get("normalization_means") is not None
        and sampler_metadata.get("normalization_stds") is not None
        else "data_config"
    )

    timesteps = args.num_timesteps
    if timesteps is None:
        timesteps = training.metric_num_timesteps or training.num_sample_timesteps
    rtol = float(training.sample_rtol if args.rtol is None else args.rtol)
    atol = float(training.sample_atol if args.atol is None else args.atol)
    args.rtol = rtol
    args.atol = atol
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else run_dir / "evaluation" / f"{args.split}_physical_{args.stride_days}d_{args.ensemble_size}ens"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = output_dir / "samples"
    if args.save_ensembles:
        samples_dir.mkdir(parents=True, exist_ok=True)
    clearml = _start_clearml(experiment, args, data_config, raw_model_config)

    fields = list(getattr(dataset, "config", data_config).get("fields", data_config.get("fields", [])))
    concentration_channel = fields.index("siconc") if "siconc" in fields else None
    if training.sample_start_mode == "bridge" and args.ensemble_size > 1:
        warnings.warn(
            "Bridge sampling is deterministic for a fixed condition in the current sampler; "
            "ensemble members may be identical.",
            stacklevel=2,
        )

    accumulator = PhysicalMetricAccumulator()
    per_case_rows: list[dict[str, Any]] = []
    case_metadata: list[dict[str, Any]] = []
    sample_target = _sample_target(training)
    total_samples = len(case_indices) * args.ensemble_size
    with tqdm(total=total_samples, desc="sampling physical evaluation", unit="sample") as progress:
        for case_order, case_index in enumerate(case_indices):
            item = dataset[case_index]
            background = item["background"].unsqueeze(0).to(device)
            obs_values = item["obs_values"].unsqueeze(0).to(device)
            obs_mask = item["obs_mask"].unsqueeze(0).to(device)
            water_mask = item["water_mask"].unsqueeze(0).to(device)
            valid_mask_tensor = item["valid_mask"].unsqueeze(0).to(device)

            progress.set_postfix(case=f"{case_order + 1}/{len(case_indices)}")
            generated = generate_ensemble(
                sampler=sampler,
                background=background,
                obs_values=obs_values,
                obs_mask=obs_mask,
                water_mask=water_mask,
                valid_mask=valid_mask_tensor,
                config=training,
                ensemble_size=args.ensemble_size,
                sample_batch_size=args.sample_batch_size,
                num_timesteps=timesteps,
                method=args.method,
                device=device,
                seed=args.seed,
                case_order=case_order,
                sample_target=sample_target,
                rtol=rtol,
                atol=atol,
                autocast_dtype=autocast_dtype,
                progress=progress,
            )

            ensemble = denormalize_and_clip(generated, means, stds, concentration_channel)
            truth = denormalize_and_clip(item["truth"].unsqueeze(0), means, stds, concentration_channel)[0]
            background_physical = denormalize_and_clip(
                item["background"].unsqueeze(0), means, stds, concentration_channel
            )[0]
            obs_values_physical = denormalize_and_clip(
                item["obs_values"].unsqueeze(0), means, stds, concentration_channel
            )[0]
            valid_mask = item["valid_mask"].detach().cpu().numpy() > 0.5
            obs_mask_physical = item["obs_mask"].detach().cpu().numpy() > 0.5

            metadata = {"case_order": case_order, "dataset_index": case_index, **item.get("meta", {})}
            case_metadata.append(_json_safe(metadata))
            for channel, field in enumerate(fields):
                for region, mask in _region_masks(valid_mask, obs_mask_physical, channel).items():
                    row = accumulator.add(
                        field,
                        region,
                        ensemble[:, channel],
                        truth[channel],
                        background_physical[channel],
                        mask,
                    )
                    if row is not None:
                        per_case_rows.append({**metadata, **row})

            if args.save_ensembles:
                case_id = str(item.get("meta", {}).get("case_id", f"index_{case_index:06d}"))
                np.savez_compressed(
                    samples_dir / f"{case_order:04d}_{case_id}.npz",
                    analysis_ensemble=ensemble.astype(np.float32),
                    truth=truth.astype(np.float32),
                    background=background_physical.astype(np.float32),
                    obs_values=obs_values_physical.astype(np.float32),
                    obs_mask=obs_mask_physical,
                    valid_mask=valid_mask,
                    fields=np.asarray(fields),
                )

    aggregate_rows = accumulator.finalize()
    _write_csv(output_dir / "per_case_metrics.csv", per_case_rows)
    _write_csv(output_dir / "aggregate_metrics.csv", aggregate_rows)
    (output_dir / "aggregate_metrics.json").write_text(
        json.dumps(_json_safe(aggregate_rows), indent=2) + "\n"
    )
    condition_note = (
        "Condition is produced by the same dataset split and observation_mask policy as validation. "
        "For sral_tracks this uses forecast-field values on the selected SRAL footprint; "
        "empty_probability and synthetic_probability remain active."
    )
    metadata = {
        "experiment_config": str(experiment_path),
        "data_config": str(data_config_path),
        "model_config": str(model_config_path),
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "device": str(device),
        "case_indices": case_indices,
        "num_cases": len(case_indices),
        "stride_days": args.stride_days,
        "ensemble_size": args.ensemble_size,
        "sample_batch_size": args.sample_batch_size,
        "effective_sample_batch_size": min(args.sample_batch_size, args.ensemble_size),
        "inference_precision": inference_precision,
        "memory_efficient_euler": args.method == "euler",
        "num_timesteps": timesteps,
        "sampling_method": args.method,
        "sampling_rtol": rtol,
        "sampling_atol": atol,
        "sample_target": sample_target,
        "sample_start_mode": training.sample_start_mode,
        "sample_cfg_mode": training.sample_cfg_mode,
        "sample_cfg_background_scale": training.sample_cfg_background_scale,
        "sample_cfg_observation_scale": training.sample_cfg_observation_scale,
        "values_space": "physical",
        "normalization_means": means,
        "normalization_stds": stds,
        "normalization_source": normalization_source,
        "concentration_channel": concentration_channel,
        "concentration_clipping": "[0, 1]" if concentration_channel is not None else None,
        "condition_note": condition_note,
        "clearml_enabled": clearml is not None,
        "cases": case_metadata,
    }
    (output_dir / "metadata.json").write_text(json.dumps(_json_safe(metadata), indent=2) + "\n")
    if clearml is not None:
        _publish_clearml_results(
            clearml,
            output_dir,
            aggregate_rows,
            metadata,
            upload_samples=bool(experiment.get("clearml", {}).get("upload_samples", False)),
        )
        clearml.close()
    result = {
        "output_dir": str(output_dir),
        "aggregate_metrics_path": str(output_dir / "aggregate_metrics.json"),
        "per_case_metrics_path": str(output_dir / "per_case_metrics.csv"),
        "num_cases": len(case_indices),
    }
    print(json.dumps(result, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a conditioned model checkpoint in physical units.")
    parser.add_argument(
        "--config", required=True, help="Path to an evaluation or training experiment JSON config."
    )
    parser.add_argument("--run-dir", default=None, help="Directory containing the trained checkpoint.")
    parser.add_argument("--checkpoint-name", default=None, help="Checkpoint filename.")
    parser.add_argument("--output-dir", default=None, help="Directory for evaluation outputs.")
    parser.add_argument(
        "--split",
        default=None,
        choices=("train", "valid", "test"),
        help="Dataset split.",
    )
    parser.add_argument(
        "--stride-days", type=int, default=None, help="Spacing between evaluated target days."
    )
    parser.add_argument("--ensemble-size", type=int, default=None, help="Samples generated for each case.")
    parser.add_argument(
        "--sample-batch-size", type=int, default=None, help="Maximum members sampled per model batch."
    )
    parser.add_argument(
        "--max-cases", type=int, default=None, help="Optional maximum number of selected cases."
    )
    parser.add_argument("--num-timesteps", type=int, default=None, help="Override sampling time steps.")
    parser.add_argument("--method", default=None, choices=("euler", "dopri5"), help="ODE solver method.")
    parser.add_argument(
        "--rtol", type=float, default=None, help="Relative tolerance for adaptive ODE solvers."
    )
    parser.add_argument(
        "--atol", type=float, default=None, help="Absolute tolerance for adaptive ODE solvers."
    )
    parser.add_argument(
        "--device", default=None, help="Torch device; defaults to repository device selection."
    )
    parser.add_argument(
        "--inference-precision",
        default=None,
        choices=("auto", "float32", "float16", "bfloat16"),
        help="CUDA autocast precision; auto follows mixed_precision from the training config.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed used to generate ensemble members.")
    parser.add_argument(
        "--cfg-mode",
        default=None,
        choices=("none", "independent", "background_delta"),
        help=(
            "CFG-like conditioning control. 'independent' combines none/background-only/"
            "observation-only predictions; 'background_delta' adds an observation correction "
            "to the background-conditioned prediction."
        ),
    )
    parser.add_argument(
        "--cfg-background-scale",
        type=float,
        default=None,
        help="Scale for background CFG influence. 1.0 keeps the learned baseline.",
    )
    parser.add_argument(
        "--cfg-observation-scale",
        type=float,
        default=None,
        help="Scale for observation CFG influence. 1.0 keeps the learned baseline.",
    )
    parser.add_argument(
        "--save-ensembles",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Store physical arrays per selected case.",
    )
    parser.add_argument(
        "--clearml",
        dest="clearml_enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable ClearML; by default follows clearml.enabled in the config.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
