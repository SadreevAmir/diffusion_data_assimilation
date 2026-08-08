from __future__ import annotations

import argparse
import csv
import json
import math
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

from .config import TrainingConfig, load_json, merge_config_overrides, resolve_path
from .data import build_dataset
from .evaluate import (
    PhysicalMetricAccumulator,
    apply_sampler_normalization,
    denormalize_and_clip,
    generate_ensemble,
)
from .model_io import load_sampler
from .runtime import get_device


@dataclass(frozen=True)
class BlendSetting:
    """Convex background-only/track-only velocity blend."""

    track_weight: float
    background_weight: float
    both_weight: float = 0.0

    @classmethod
    def from_track_weight(cls, track_weight: float) -> BlendSetting:
        track_weight = float(track_weight)
        if not 0.0 <= track_weight <= 1.0:
            raise ValueError(f"track weight must be within [0, 1], got {track_weight}")
        return cls(
            track_weight=track_weight,
            background_weight=1.0 - track_weight,
            both_weight=0.0,
        )

    @property
    def label(self) -> str:
        return f"bg={self.background_weight:.2f}, track={self.track_weight:.2f}, both=0"

    @property
    def slug(self) -> str:
        return f"bg_{self.background_weight:.3f}_track_{self.track_weight:.3f}"

    def sampler_overrides(self) -> dict[str, float | str]:
        # independent CFG:
        # v_none + w_bg*(v_bg-v_none) + w_track*(v_track-v_none)
        # With w_bg+w_track=1 this is exactly w_bg*v_bg+w_track*v_track.
        return {
            "sample_cfg_mode": "independent",
            "sample_cfg_background_scale": self.background_weight,
            "sample_cfg_observation_scale": self.track_weight,
        }


@dataclass
class ExperimentContext:
    config_path: Path
    experiment: dict[str, Any]
    data_config: dict[str, Any]
    model_config: dict[str, Any]
    training: TrainingConfig
    dataset: Any
    sampler: Any
    means: list[float]
    stds: list[float]
    fields: list[str]
    device: torch.device


class RunMonitor:
    def __init__(self, output_dir: Path, total_samples: int, description: str):
        self.path = output_dir / "run_status.json"
        self.total_samples = max(int(total_samples), 1)
        self.completed_samples = 0
        self.description = description
        self.started_at = datetime.now()
        self.started_monotonic = time.monotonic()
        self.last: dict[str, Any] = {}
        self._write("running")

    def advance(self, samples: int, **last: Any) -> None:
        self.completed_samples += int(samples)
        self.last = _json_safe(last)
        self._write("running")

    def finish(self) -> None:
        self.completed_samples = self.total_samples
        self._write("completed")

    def fail(self, error: str) -> None:
        self.last = {**self.last, "error": str(error)}
        self._write("failed")

    def _write(self, status: str) -> None:
        elapsed_seconds = time.monotonic() - self.started_monotonic
        rate = self.completed_samples / elapsed_seconds if elapsed_seconds > 0 else 0.0
        remaining_samples = max(self.total_samples - self.completed_samples, 0)
        remaining_seconds = remaining_samples / rate if rate > 0 else None
        payload = {
            "status": status,
            "description": self.description,
            "started_at": self.started_at.isoformat(),
            "updated_at": datetime.now().isoformat(),
            "completed_samples": self.completed_samples,
            "total_samples": self.total_samples,
            "progress_percent": 100.0 * self.completed_samples / self.total_samples,
            "elapsed_minutes": elapsed_seconds / 60.0,
            "estimated_remaining_minutes": (
                remaining_seconds / 60.0 if remaining_seconds is not None else None
            ),
            "last_completed": self.last,
        }
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(_json_safe(payload), indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.path)


def blend_settings(track_weights: list[float]) -> list[BlendSetting]:
    if not track_weights:
        raise ValueError("At least one track weight is required")
    settings = [BlendSetting.from_track_weight(value) for value in track_weights]
    if len({setting.track_weight for setting in settings}) != len(settings):
        raise ValueError("Track weights must be unique")
    return settings


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


def _prepare_output_dir(path: str | Path) -> Path:
    output_dir = Path(path).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. "
            "Choose a new directory so experiment results are not overwritten."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _load_context(args: argparse.Namespace) -> ExperimentContext:
    config_path = Path(args.config).expanduser().resolve()
    experiment = load_json(config_path)
    data_config_path = resolve_path(experiment["data_config"], config_path.parent).resolve()
    model_config_path = resolve_path(experiment["model_config"], config_path.parent).resolve()
    data_config = merge_config_overrides(load_json(data_config_path), experiment.get("data_overrides"))
    model_config = {
        **load_json(model_config_path),
        **experiment.get("training", {}),
    }
    training = TrainingConfig.from_dict(model_config)
    device = torch.device(args.device or get_device())
    dataset = build_dataset(deepcopy(data_config), split=args.split)
    sampler = load_sampler(args.run_dir, args.checkpoint_name, model_config, device=device)
    means, stds = apply_sampler_normalization(dataset, sampler, data_config)
    fields = list(getattr(dataset, "config", data_config).get("fields", data_config.get("fields", [])))
    if not fields:
        fields = [f"channel_{index}" for index in range(training.out_channels)]
    return ExperimentContext(
        config_path=config_path,
        experiment=experiment,
        data_config=data_config,
        model_config=model_config,
        training=training,
        dataset=dataset,
        sampler=sampler,
        means=list(means),
        stds=list(stds),
        fields=fields,
        device=device,
    )


def _target_date(dataset, day_index: int) -> date | None:
    if not hasattr(dataset, "obs_data"):
        return None
    obs_shift = int(getattr(dataset, "obs_shift", 0))
    return dataset.obs_data[day_index + obs_shift].date


def month_case_indices(
    dataset,
    *,
    month: int,
    year: int | None,
    max_cases: int,
    require_observations: bool = False,
) -> list[int]:
    """Select evenly spaced cases from a calendar month at the configured target hour."""
    if max_cases <= 0:
        raise ValueError("max_cases must be positive")
    if int(month) != 12:
        raise ValueError("CFG balance experiments are restricted to December (month=12)")

    hours_per_day = int(getattr(dataset, "hours_per_day", 1))
    hour = min(max(int(getattr(dataset, "hour_index", 0)), 0), hours_per_day - 1)
    if hasattr(dataset, "_num_days") and hasattr(dataset, "obs_data"):
        candidates: list[tuple[int, date]] = []
        for day_index in range(dataset._num_days()):
            target = _target_date(dataset, day_index)
            if target is None or target.month != month:
                continue
            if year is not None and target.year != year:
                continue
            candidates.append((day_index * hours_per_day + hour, target))
        if year is None and candidates:
            latest_year = max(target.year for _, target in candidates)
            candidates = [(index, target) for index, target in candidates if target.year == latest_year]
        indices = [index for index, _ in candidates]
    else:
        indices = list(range(len(dataset)))

    if require_observations:
        indices = [index for index in indices if bool(torch.any(dataset[index]["obs_mask"] > 0))]
    if not indices:
        raise ValueError(f"No December dataset cases selected for year={year}")

    count = min(max_cases, len(indices))
    positions = np.linspace(0, len(indices) - 1, num=count, dtype=np.int64)
    return [indices[int(position)] for position in positions]


def force_synthetic_track_count(dataset, track_count: int) -> None:
    """Make dataset observations use exactly ``track_count`` generated strips."""
    track_count = int(track_count)
    if track_count <= 0:
        raise ValueError("track_count must be positive")
    if getattr(dataset, "name", "") == "M2MForecastDataset":
        dataset.config["observation_mask"] = {
            "kind": "sral_tracks",
            "sral_transform_index": 11,
            "synthetic_probability": 1.0,
            "empty_probability": 0.0,
            "synthetic": {
                "kind": "generated_track",
                "n_tracks_range": [track_count, track_count],
            },
        }
    else:
        dataset.config["observation_mask"] = {
            "kind": "generated_track",
            "n_tracks_range": [track_count, track_count],
        }


def _case_metadata(item: dict[str, Any], dataset_index: int, case_order: int) -> dict[str, Any]:
    metadata = item.get("meta", {})
    if not isinstance(metadata, dict):
        metadata = {}
    return _json_safe(
        {
            "case_order": case_order,
            "dataset_index": dataset_index,
            **metadata,
        }
    )


def _physical_item(
    item: dict[str, Any],
    means: list[float],
    stds: list[float],
    concentration_channel: int | None,
) -> dict[str, np.ndarray]:
    output = {}
    for key in ("truth", "background", "obs_values"):
        output[key] = denormalize_and_clip(
            item[key].unsqueeze(0),
            means,
            stds,
            concentration_channel,
        )[0]
    output["obs_mask"] = item["obs_mask"].detach().cpu().numpy() > 0.5
    output["valid_mask"] = item["valid_mask"].detach().cpu().numpy() > 0.5
    output["water_mask"] = item["water_mask"].detach().cpu().numpy() > 0.5
    return output


def _sample_balance(
    context: ExperimentContext,
    item: dict[str, Any],
    *,
    setting: BlendSetting,
    case_order: int,
    ensemble_size: int,
    sample_batch_size: int,
    num_timesteps: int,
    method: str,
    seed: int,
    progress=None,
) -> torch.Tensor:
    config = replace(context.training, **setting.sampler_overrides())
    sample_target = "residual" if config.training_objective == "residual_flow" else "state"
    tensors = {
        key: item[key].unsqueeze(0).to(context.device, dtype=torch.float32)
        for key in ("background", "obs_values", "obs_mask", "water_mask", "valid_mask")
    }
    return generate_ensemble(
        sampler=context.sampler,
        background=tensors["background"],
        obs_values=tensors["obs_values"],
        obs_mask=tensors["obs_mask"],
        water_mask=tensors["water_mask"],
        valid_mask=tensors["valid_mask"],
        config=config,
        ensemble_size=ensemble_size,
        sample_batch_size=sample_batch_size,
        num_timesteps=num_timesteps,
        method=method,
        device=context.device,
        seed=seed,
        case_order=case_order,
        sample_target=sample_target,
        rtol=float(config.sample_rtol),
        atol=float(config.sample_atol),
        progress=progress,
    )


def _add_metrics(
    accumulator: PhysicalMetricAccumulator,
    per_case_rows: list[dict[str, Any]],
    *,
    fields: list[str],
    ensemble: np.ndarray,
    physical: dict[str, np.ndarray],
    labels: dict[str, Any],
) -> None:
    for channel, field in enumerate(fields):
        valid = physical["valid_mask"][channel]
        observed = valid & physical["obs_mask"][channel]
        regions = {
            "full": valid,
            "observed": observed,
            "unobserved": valid & ~observed,
        }
        for region, mask in regions.items():
            row = accumulator.add(
                field=field,
                region=region,
                ensemble=ensemble[:, channel],
                truth=physical["truth"][channel],
                background=physical["background"][channel],
                mask=mask,
            )
            if row is not None:
                per_case_rows.append({**labels, **row})


def _display_limits(
    channel: int,
    physical: dict[str, np.ndarray],
    ensembles: list[np.ndarray],
) -> tuple[float | None, float | None]:
    if channel == 0:
        return 0.0, 1.0
    valid = physical["valid_mask"][channel]
    values = [
        physical["background"][channel][valid],
        physical["truth"][channel][valid],
        *[ensemble[:, channel][:, valid].reshape(-1) for ensemble in ensembles],
    ]
    finite = np.concatenate([value[np.isfinite(value)] for value in values if value.size])
    if finite.size == 0:
        return None, None
    low, high = np.nanpercentile(finite, [1.0, 99.0])
    if low == high:
        return None, None
    return float(low), float(high)


def _plot_december_case(
    output_path: Path,
    *,
    field: str,
    channel: int,
    metadata: dict[str, Any],
    physical: dict[str, np.ndarray],
    settings: list[BlendSetting],
    ensembles: list[np.ndarray],
) -> None:
    import matplotlib.pyplot as plt

    ensemble_size = ensembles[0].shape[0]
    ncols = 4 + ensemble_size
    fig, axes = plt.subplots(
        len(settings),
        ncols,
        figsize=(3.2 * ncols, 3.0 * len(settings)),
        squeeze=False,
        constrained_layout=True,
    )
    vmin, vmax = _display_limits(channel, physical, ensembles)
    cmap = "Blues_r" if channel == 0 else "viridis"
    valid = physical["valid_mask"][channel]
    condition = np.where(
        physical["obs_mask"][channel] & valid,
        physical["obs_values"][channel],
        np.nan,
    )
    common = [
        ("background", np.where(valid, physical["background"][channel], np.nan)),
        ("track", condition),
        ("truth", np.where(valid, physical["truth"][channel], np.nan)),
    ]
    images = []
    for row, (setting, ensemble) in enumerate(zip(settings, ensembles, strict=True)):
        panels = [
            *common,
            ("ensemble mean", np.where(valid, ensemble[:, channel].mean(axis=0), np.nan)),
            *[
                (f"sample {member + 1}", np.where(valid, ensemble[member, channel], np.nan))
                for member in range(ensemble_size)
            ],
        ]
        for col, (title, values) in enumerate(panels):
            image = axes[row, col].imshow(
                np.ma.masked_invalid(values),
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )
            images.append(image)
            if row == 0:
                axes[row, col].set_title(title)
            if col == 0:
                axes[row, col].set_ylabel(setting.label)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
    case_label = metadata.get("case_id", metadata.get("target_date", metadata["dataset_index"]))
    fig.suptitle(f"{field}: December CFG balance, case={case_label}", fontsize=15)
    fig.colorbar(images[-1], ax=axes.ravel().tolist(), fraction=0.012, pad=0.01)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _save_case_npz(
    path: Path,
    *,
    ensemble: np.ndarray,
    physical: dict[str, np.ndarray],
    fields: list[str],
    setting: BlendSetting,
    metadata: dict[str, Any],
    track_count: int | None = None,
) -> None:
    np.savez_compressed(
        path,
        analysis_ensemble=ensemble.astype(np.float32),
        truth=physical["truth"].astype(np.float32),
        background=physical["background"].astype(np.float32),
        obs_values=physical["obs_values"].astype(np.float32),
        obs_mask=physical["obs_mask"],
        valid_mask=physical["valid_mask"],
        water_mask=physical["water_mask"],
        fields=np.asarray(fields),
        background_weight=np.float32(setting.background_weight),
        track_weight=np.float32(setting.track_weight),
        both_weight=np.float32(0.0),
        track_count=np.int32(-1 if track_count is None else track_count),
        metadata_json=np.asarray(json.dumps(_json_safe(metadata))),
    )


def _base_metadata(
    context: ExperimentContext,
    args: argparse.Namespace,
    settings: list[BlendSetting],
    case_indices: list[int],
) -> dict[str, Any]:
    return {
        "config": str(context.config_path),
        "run_dir": str(Path(args.run_dir).expanduser().resolve()),
        "checkpoint_name": args.checkpoint_name,
        "split": args.split,
        "device": str(context.device),
        "year": args.year,
        "month": args.month,
        "case_indices": case_indices,
        "ensemble_size": args.ensemble_size,
        "sample_batch_size": args.sample_batch_size,
        "num_timesteps": args.num_timesteps,
        "method": args.method,
        "seed": args.seed,
        "normalization_means": context.means,
        "normalization_stds": context.stds,
        "formula": (
            "v = background_weight * v_background_only "
            "+ track_weight * v_track_only + 0 * v_both"
        ),
        "cfg_implementation": (
            "independent CFG with background_weight + track_weight = 1; "
            "the unconditional coefficient cancels exactly"
        ),
        "settings": [asdict(setting) for setting in settings],
    }


def run_december(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = _prepare_output_dir(args.output_dir)
    context = _load_context(args)
    settings = blend_settings(args.track_weights)
    if args.track_source == "synthetic":
        force_synthetic_track_count(context.dataset, args.synthetic_track_count)
    case_indices = month_case_indices(
        context.dataset,
        month=args.month,
        year=args.year,
        max_cases=args.num_cases,
        require_observations=True,
    )
    concentration_channel = context.fields.index("siconc") if "siconc" in context.fields else None
    accumulators = {setting: PhysicalMetricAccumulator() for setting in settings}
    per_case_rows: list[dict[str, Any]] = []
    case_metadata = []
    samples_dir = output_dir / "samples"
    figures_dir = output_dir / "figures"
    samples_dir.mkdir()
    figures_dir.mkdir()
    total_samples = len(case_indices) * len(settings) * args.ensemble_size
    monitor = RunMonitor(output_dir, total_samples, "short December CFG visual balance")
    progress = tqdm(total=total_samples, desc="December CFG visual", unit="sample")

    try:
        for case_order, dataset_index in enumerate(case_indices):
            item = context.dataset[dataset_index]
            metadata = _case_metadata(item, dataset_index, case_order)
            case_metadata.append(metadata)
            physical = _physical_item(item, context.means, context.stds, concentration_channel)
            case_ensembles = []
            for setting in settings:
                progress.set_postfix(case=case_order + 1, track_weight=setting.track_weight)
                generated = _sample_balance(
                    context,
                    item,
                    setting=setting,
                    case_order=case_order,
                    ensemble_size=args.ensemble_size,
                    sample_batch_size=args.sample_batch_size,
                    num_timesteps=args.num_timesteps,
                    method=args.method,
                    seed=args.seed,
                    progress=progress,
                )
                ensemble = denormalize_and_clip(
                    generated,
                    context.means,
                    context.stds,
                    concentration_channel,
                )
                case_ensembles.append(ensemble)
                labels = {
                    **metadata,
                    "background_weight": setting.background_weight,
                    "track_weight": setting.track_weight,
                    "both_weight": 0.0,
                }
                _add_metrics(
                    accumulators[setting],
                    per_case_rows,
                    fields=context.fields,
                    ensemble=ensemble,
                    physical=physical,
                    labels=labels,
                )
                case_id = str(metadata.get("case_id", f"index_{dataset_index:06d}"))
                _save_case_npz(
                    samples_dir / f"{case_order:02d}_{case_id}_{setting.slug}.npz",
                    ensemble=ensemble,
                    physical=physical,
                    fields=context.fields,
                    setting=setting,
                    metadata=metadata,
                )
                monitor.advance(
                    args.ensemble_size,
                    case_order=case_order,
                    case_id=case_id,
                    background_weight=setting.background_weight,
                    track_weight=setting.track_weight,
                )
                _write_csv(output_dir / "per_case_metrics.csv", per_case_rows)

            case_id = str(metadata.get("case_id", f"index_{dataset_index:06d}"))
            for channel, field in enumerate(context.fields):
                _plot_december_case(
                    figures_dir / f"{case_order:02d}_{case_id}_{field}.png",
                    field=field,
                    channel=channel,
                    metadata=metadata,
                    physical=physical,
                    settings=settings,
                    ensembles=case_ensembles,
                )
    except Exception as error:
        monitor.fail(str(error))
        raise
    finally:
        progress.close()

    aggregate_rows = []
    for setting in settings:
        for row in accumulators[setting].finalize():
            aggregate_rows.append(
                {
                    "background_weight": setting.background_weight,
                    "track_weight": setting.track_weight,
                    "both_weight": 0.0,
                    **row,
                }
            )
    _write_csv(output_dir / "aggregate_metrics.csv", aggregate_rows)
    _write_csv(output_dir / "per_case_metrics.csv", per_case_rows)
    metadata_payload = {
        **_base_metadata(context, args, settings, case_indices),
        "experiment": "december_visual_balance",
        "track_source": args.track_source,
        "synthetic_track_count": (
            args.synthetic_track_count if args.track_source == "synthetic" else None
        ),
        "cases": case_metadata,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(_json_safe(metadata_payload), indent=2) + "\n",
        encoding="utf-8",
    )
    monitor.finish()
    result = {
        "output_dir": str(output_dir),
        "figures_dir": str(figures_dir),
        "samples_dir": str(samples_dir),
        "num_cases": len(case_indices),
        "num_settings": len(settings),
    }
    print(json.dumps(result, indent=2))
    return result


def _plot_track_masks(
    path: Path,
    masks: list[tuple[int, np.ndarray]],
    *,
    case_label: str,
) -> None:
    import matplotlib.pyplot as plt

    ncols = min(5, len(masks))
    nrows = math.ceil(len(masks) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.2 * ncols, 3.0 * nrows),
        squeeze=False,
        constrained_layout=True,
    )
    for axis, (count, mask) in zip(axes.ravel(), masks, strict=False):
        axis.imshow(mask, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
        axis.set_title(f"{count} synthetic track(s)")
        axis.axis("off")
    for axis in axes.ravel()[len(masks) :]:
        axis.axis("off")
    fig.suptitle(f"Synthetic observation masks, case={case_label}")
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_quality_surfaces(
    output_dir: Path,
    rows: list[dict[str, Any]],
    *,
    fields: list[str],
    track_counts: list[int],
    settings: list[BlendSetting],
) -> list[str]:
    import matplotlib.pyplot as plt

    saved = []
    track_weights = [setting.track_weight for setting in settings]
    x_grid, y_grid = np.meshgrid(track_counts, track_weights, indexing="ij")
    for region in ("full", "observed", "unobserved"):
        fig = plt.figure(figsize=(8 * len(fields), 6), constrained_layout=True)
        for field_index, field in enumerate(fields):
            axis = fig.add_subplot(1, len(fields), field_index + 1, projection="3d")
            lookup = {
                (int(row["track_count"]), float(row["track_weight"])): row
                for row in rows
                if row["field"] == field and row["region"] == region
            }
            z_grid = np.full(x_grid.shape, np.nan, dtype=np.float64)
            for count_index, count in enumerate(track_counts):
                for weight_index, weight in enumerate(track_weights):
                    row = lookup.get((count, weight))
                    if row is not None:
                        z_grid[count_index, weight_index] = float(row["analysis_rmse_skill"])
            if len(track_counts) >= 2 and len(track_weights) >= 2:
                surface = axis.plot_surface(
                    x_grid,
                    y_grid,
                    z_grid,
                    cmap="viridis",
                    edgecolor="none",
                    antialiased=True,
                )
                fig.colorbar(surface, ax=axis, shrink=0.65, pad=0.08)
            else:
                axis.scatter(x_grid.ravel(), y_grid.ravel(), z_grid.ravel(), c=z_grid.ravel())
            axis.set_xlabel("synthetic track count")
            axis.set_ylabel("track weight\n(background weight = 1 - track weight)")
            axis.set_zlabel("RMSE skill vs background")
            axis.set_title(f"{field}, region={region}")
            axis.view_init(elev=28, azim=-135)
        fig.suptitle(
            "Quality surface: background-only/track-only blend; both coefficient = 0",
            fontsize=15,
        )
        path = output_dir / f"quality_surface_{region}_rmse_skill.png"
        fig.savefig(path, dpi=190, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(path))
    return saved


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _calibrated_december_indices(
    context: ExperimentContext,
    args: argparse.Namespace,
    settings: list[BlendSetting],
    track_counts: list[int],
    output_dir: Path,
) -> tuple[list[int], dict[str, Any] | None]:
    available = month_case_indices(
        context.dataset,
        month=12,
        year=args.year,
        max_cases=10_000,
        require_observations=True,
    )
    if args.target_hours is None:
        count = min(args.num_cases, len(available))
        positions = np.linspace(0, len(available) - 1, num=count, dtype=np.int64)
        return [available[int(position)] for position in positions], None

    if float(args.target_hours) <= 0.0:
        raise ValueError("target_hours must be positive")
    middle_count = track_counts[len(track_counts) // 2]
    middle_setting = min(settings, key=lambda setting: abs(setting.track_weight - 0.5))
    force_synthetic_track_count(context.dataset, middle_count)
    probe_index = available[len(available) // 2]
    probe_item = context.dataset[probe_index]
    print(
        "[cfg_experiments] calibrating runtime on one December case: "
        f"tracks={middle_count}, {middle_setting.label}, ensemble={args.ensemble_size}, "
        f"method={args.method}, timesteps={args.num_timesteps}",
        flush=True,
    )
    _synchronize(context.device)
    started = time.monotonic()
    _sample_balance(
        context,
        probe_item,
        setting=middle_setting,
        case_order=0,
        ensemble_size=args.ensemble_size,
        sample_batch_size=args.sample_batch_size,
        num_timesteps=args.num_timesteps,
        method=args.method,
        seed=args.seed,
    )
    _synchronize(context.device)
    probe_seconds = time.monotonic() - started
    safety_factor = 1.05
    seconds_per_case = probe_seconds * len(track_counts) * len(settings) * safety_factor
    requested_cases = max(1, int(round(float(args.target_hours) * 3600.0 / seconds_per_case)))
    selected_cases = min(requested_cases, len(available))
    positions = np.linspace(0, len(available) - 1, num=selected_cases, dtype=np.int64)
    selected = [available[int(position)] for position in positions]
    projected_hours = seconds_per_case * selected_cases / 3600.0
    calibration = {
        "target_hours": float(args.target_hours),
        "probe_seconds": probe_seconds,
        "probe_track_count": middle_count,
        "probe_setting": asdict(middle_setting),
        "safety_factor": safety_factor,
        "available_december_cases": len(available),
        "requested_case_count": requested_cases,
        "selected_case_count": selected_cases,
        "projected_sweep_hours": projected_hours,
        "note": (
            "Projection is based on one real GPU sampling cell and includes a 5% safety factor. "
            "Adaptive dopri5 work can vary by condition."
        ),
    }
    (output_dir / "runtime_calibration.json").write_text(
        json.dumps(_json_safe(calibration), indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "[cfg_experiments] runtime calibration: "
        f"probe={probe_seconds:.1f}s, December cases={selected_cases}, "
        f"projected sweep={projected_hours:.2f}h",
        flush=True,
    )
    return selected, calibration


def run_track_sweep(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = _prepare_output_dir(args.output_dir)
    context = _load_context(args)
    settings = blend_settings(args.track_weights)
    track_counts = list(range(args.min_tracks, args.max_tracks + 1))
    if not track_counts or track_counts[0] <= 0:
        raise ValueError("Track range must contain positive values")

    force_synthetic_track_count(context.dataset, track_counts[0])
    case_indices, runtime_calibration = _calibrated_december_indices(
        context,
        args,
        settings,
        track_counts,
        output_dir,
    )
    concentration_channel = context.fields.index("siconc") if "siconc" in context.fields else None
    aggregate_rows: list[dict[str, Any]] = []
    per_case_rows: list[dict[str, Any]] = []
    track_masks = []
    samples_dir = output_dir / "samples"
    if args.save_samples:
        samples_dir.mkdir()
    total_samples = (
        len(track_counts) * len(settings) * len(case_indices) * int(args.ensemble_size)
    )
    monitor = RunMonitor(output_dir, total_samples, "December synthetic-track CFG quality sweep")
    progress = tqdm(total=total_samples, desc="December track/CFG sweep", unit="sample")

    try:
        for track_count in track_counts:
            force_synthetic_track_count(context.dataset, track_count)
            items = [context.dataset[index] for index in case_indices]
            first_mask = items[0]["obs_mask"][0].detach().cpu().numpy()
            track_masks.append((track_count, first_mask))
            for setting in settings:
                accumulator = PhysicalMetricAccumulator()
                for case_order, (dataset_index, item) in enumerate(
                    zip(case_indices, items, strict=True)
                ):
                    metadata = _case_metadata(item, dataset_index, case_order)
                    physical = _physical_item(item, context.means, context.stds, concentration_channel)
                    progress.set_postfix(
                        tracks=track_count,
                        track_weight=setting.track_weight,
                        case=case_order + 1,
                    )
                    generated = _sample_balance(
                        context,
                        item,
                        setting=setting,
                        case_order=case_order,
                        ensemble_size=args.ensemble_size,
                        sample_batch_size=args.sample_batch_size,
                        num_timesteps=args.num_timesteps,
                        method=args.method,
                        seed=args.seed,
                        progress=progress,
                    )
                    ensemble = denormalize_and_clip(
                        generated,
                        context.means,
                        context.stds,
                        concentration_channel,
                    )
                    labels = {
                        **metadata,
                        "track_count": track_count,
                        "background_weight": setting.background_weight,
                        "track_weight": setting.track_weight,
                        "both_weight": 0.0,
                        "observed_fraction": float(physical["obs_mask"].sum())
                        / max(float(physical["valid_mask"].sum()), 1.0),
                    }
                    _add_metrics(
                        accumulator,
                        per_case_rows,
                        fields=context.fields,
                        ensemble=ensemble,
                        physical=physical,
                        labels=labels,
                    )
                    if args.save_samples:
                        case_id = str(metadata.get("case_id", f"index_{dataset_index:06d}"))
                        _save_case_npz(
                            samples_dir
                            / f"tracks_{track_count:02d}_{setting.slug}_{case_order:02d}_{case_id}.npz",
                            ensemble=ensemble,
                            physical=physical,
                            fields=context.fields,
                            setting=setting,
                            metadata=metadata,
                            track_count=track_count,
                        )
                    monitor.advance(
                        args.ensemble_size,
                        track_count=track_count,
                        case_order=case_order,
                        case_id=metadata.get("case_id"),
                        background_weight=setting.background_weight,
                        track_weight=setting.track_weight,
                    )
                observed_fractions = [
                    row["observed_fraction"]
                    for row in per_case_rows
                    if row.get("track_count") == track_count
                    and row.get("track_weight") == setting.track_weight
                ]
                for row in accumulator.finalize():
                    aggregate_rows.append(
                        {
                            "track_count": track_count,
                            "background_weight": setting.background_weight,
                            "track_weight": setting.track_weight,
                            "both_weight": 0.0,
                            "mean_observed_fraction": (
                                float(np.mean(observed_fractions))
                                if observed_fractions
                                else float("nan")
                            ),
                            **row,
                        }
                    )
                _write_csv(output_dir / "surface_metrics.csv", aggregate_rows)
                _write_csv(output_dir / "per_case_metrics.csv", per_case_rows)
    except Exception as error:
        monitor.fail(str(error))
        raise
    finally:
        progress.close()

    _write_csv(output_dir / "surface_metrics.csv", aggregate_rows)
    _write_csv(output_dir / "per_case_metrics.csv", per_case_rows)
    case_label = str(_case_metadata(context.dataset[case_indices[0]], case_indices[0], 0).get("case_id"))
    _plot_track_masks(output_dir / "synthetic_track_masks.png", track_masks, case_label=case_label)
    surface_paths = _plot_quality_surfaces(
        output_dir,
        aggregate_rows,
        fields=context.fields,
        track_counts=track_counts,
        settings=settings,
    )
    metadata_payload = {
        **_base_metadata(context, args, settings, case_indices),
        "experiment": "synthetic_track_count_balance_surface",
        "track_counts": track_counts,
        "runtime_calibration": runtime_calibration,
        "save_samples": bool(args.save_samples),
        "quality_metric": "analysis_rmse_skill",
        "surface_regions": ["full", "observed", "unobserved"],
        "surface_paths": surface_paths,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(_json_safe(metadata_payload), indent=2) + "\n",
        encoding="utf-8",
    )
    monitor.finish()
    result = {
        "output_dir": str(output_dir),
        "surface_metrics": str(output_dir / "surface_metrics.csv"),
        "surface_plots": surface_paths,
        "num_cases": len(case_indices),
        "num_track_counts": len(track_counts),
        "num_settings": len(settings),
    }
    print(json.dumps(result, indent=2))
    return result


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="Evaluation experiment JSON.")
    parser.add_argument("--run-dir", required=True, help="Trained run containing the checkpoint.")
    parser.add_argument("--checkpoint-name", default="ema_best_model.pth")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=("train", "valid"), default="valid")
    parser.add_argument("--year", type=int, default=None, help="Defaults to the latest matching year.")
    parser.add_argument(
        "--month",
        type=int,
        choices=(12,),
        default=12,
        help="Experiments are intentionally restricted to December.",
    )
    parser.add_argument("--num-cases", type=int, default=3)
    parser.add_argument("--ensemble-size", type=int, default=3)
    parser.add_argument("--sample-batch-size", type=int, default=3)
    parser.add_argument("--num-timesteps", type=int, default=10)
    parser.add_argument("--method", choices=("euler", "dopri5"), default="euler")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--track-weights",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
        help="Track-only weights; background weight is always 1-track_weight.",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run controlled background-only/track-only CFG experiments. "
            "The fully-conditioned branch always has coefficient zero."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    december = subparsers.add_parser(
        "december",
        help="Short visual December experiment with several samples per balance.",
    )
    _add_common_arguments(december)
    december.add_argument(
        "--track-source",
        choices=("configured", "synthetic"),
        default="configured",
        help="Use experiment observations or force a fixed number of generated strips.",
    )
    december.add_argument("--synthetic-track-count", type=int, default=5)
    december.set_defaults(func=run_december)

    sweep = subparsers.add_parser(
        "track-sweep",
        help="Sweep synthetic track count and CFG balance and plot 3D quality surfaces.",
    )
    _add_common_arguments(sweep)
    sweep.add_argument("--min-tracks", type=int, default=1)
    sweep.add_argument("--max-tracks", type=int, default=10)
    sweep.add_argument(
        "--target-hours",
        type=float,
        default=None,
        help=(
            "Calibrate one real sampling cell and automatically select the number "
            "of December cases for this approximate sweep duration."
        ),
    )
    sweep.add_argument(
        "--save-samples",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save every generated physical ensemble; disabled by default to limit disk usage.",
    )
    sweep.set_defaults(func=run_track_sweep)
    return parser.parse_args()


def main() -> dict[str, Any]:
    args = parse_args()
    return args.func(args)


if __name__ == "__main__":
    main()
