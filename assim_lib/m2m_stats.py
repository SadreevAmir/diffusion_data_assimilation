from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import timedelta
from functools import partial
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from .config import load_json
from .forecast import ForecastRecord, as_date, forecast_records, records_in_range


@dataclass
class ChannelStats:
    counts: np.ndarray
    means: np.ndarray
    m2: np.ndarray

    @classmethod
    def empty(cls, channels: int) -> ChannelStats:
        return cls(
            counts=np.zeros(channels, dtype=np.int64),
            means=np.zeros(channels, dtype=np.float64),
            m2=np.zeros(channels, dtype=np.float64),
        )

    def merge(self, other: ChannelStats) -> None:
        if self.counts.shape != other.counts.shape:
            raise ValueError(
                f"Cannot merge channel stats with shapes {self.counts.shape} and {other.counts.shape}"
            )
        if np.any(other.counts < 0) or np.any(self.counts > np.iinfo(np.int64).max - other.counts):
            raise OverflowError("Channel value counts exceed int64 capacity")

        total_counts = self.counts + other.counts
        new_only = (self.counts == 0) & (other.counts > 0)
        both = (self.counts > 0) & (other.counts > 0)

        self.means[new_only] = other.means[new_only]
        self.m2[new_only] = other.m2[new_only]
        if np.any(both):
            left = self.counts[both].astype(np.float64)
            right = other.counts[both].astype(np.float64)
            total = total_counts[both].astype(np.float64)
            delta = other.means[both] - self.means[both]
            try:
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    self.means[both] += delta * right / total
                    self.m2[both] += other.m2[both] + delta * delta * left * right / total
            except FloatingPointError as exc:
                raise FloatingPointError("Merging M2M channel statistics exceeded float64 range") from exc

        self.counts = total_counts

    def population_variance(self) -> np.ndarray:
        return np.divide(
            self.m2,
            self.counts,
            out=np.full_like(self.m2, np.nan, dtype=np.float64),
            where=self.counts > 0,
        )


def _merged_split_config(config: dict, split: str) -> dict:
    split_config = config.get(split)
    if not isinstance(split_config, dict):
        raise KeyError(f"Missing split config: {split}")
    return {**config, **split_config}


def _paired_records(config: dict, split: str):
    merged = _merged_split_config(config, split)
    preds_dir = Path(merged["dataset_dir"]) / "preds"
    lead_time = merged.get("lead_time_hours")
    records = forecast_records(preds_dir, None if lead_time is None else int(lead_time))
    background = records_in_range(
        records,
        as_date(merged["back_start_day"]),
        as_date(merged["back_end_day"]),
    )

    assimilation_range = int(merged.get("assimilation_range", 1))
    obs_shift = max(0, assimilation_range - 1)
    target_pool = records_in_range(
        records,
        as_date(merged["obs_start_day"]) - timedelta(days=obs_shift),
        as_date(merged["obs_end_day"]) + timedelta(days=1),
    )
    if len(target_pool) < len(background) + obs_shift:
        raise ValueError("Observation/target date range is too short for background records")

    target = target_pool[obs_shift : obs_shift + len(background)]
    if not background or not target:
        raise ValueError(f"No paired M2M records found for split={split}")
    return merged, target


def _water_mask(config: dict, channels: int) -> np.ndarray | None:
    mask_path = config.get("mask_path")
    if not mask_path:
        return None

    raw = np.asarray(np.load(mask_path), dtype=np.float32)
    if config.get("mask_true_is_invalid", True):
        raw = 1.0 - raw
    if raw.ndim == 2:
        raw = np.broadcast_to(raw[None, ...], (channels, *raw.shape))
    elif raw.ndim == 3 and raw.shape[0] == 1 and channels > 1:
        raw = np.broadcast_to(raw, (channels, *raw.shape[-2:]))
    if raw.ndim != 3 or raw.shape[0] != channels:
        raise ValueError(f"Expected land mask with {channels} channels or one 2D mask, got {raw.shape}")

    out_height, out_width = tuple(int(v) for v in config.get("image_size", [320, 256]))
    if raw.shape[-2] > out_height or raw.shape[-1] > out_width:
        raise ValueError(f"Cannot fit mask shape {raw.shape[-2:]} into image_size {(out_height, out_width)}")
    mask = np.zeros((channels, out_height, out_width), dtype=bool)
    mask[:, : raw.shape[-2], : raw.shape[-1]] = raw > 0
    return mask


def _stats_from_block(
    block: np.ndarray,
    water_mask: np.ndarray | None,
    source_path: Path,
) -> ChannelStats:
    block = np.asarray(block, dtype=np.float64)
    if block.ndim not in (3, 4):
        raise ValueError(f"Expected selected block [C,H,W] or [C,T,H,W], got {block.shape} in {source_path}")
    valid = np.isfinite(block)
    if water_mask is not None:
        height, width = block.shape[-2:]
        if height > water_mask.shape[-2] or width > water_mask.shape[-1]:
            raise ValueError(
                f"Field shape {(height, width)} is larger than mask shape {water_mask.shape[-2:]}"
            )
        cropped = water_mask[:, :height, :width]
        broadcast = cropped[:, None, :, :] if block.ndim == 4 else cropped
        valid &= broadcast

    values = np.where(valid, block, 0.0)
    reduce_dims = tuple(range(1, block.ndim))
    counts = np.count_nonzero(valid, axis=reduce_dims).astype(np.int64, copy=False)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            sums = np.sum(values, axis=reduce_dims, dtype=np.float64)
            means = np.divide(
                sums,
                counts,
                out=np.zeros(block.shape[0], dtype=np.float64),
                where=counts > 0,
            )
            mean_view_shape = (block.shape[0],) + (1,) * (block.ndim - 1)
            centered = values - means.reshape(mean_view_shape)
            centered[~valid] = 0.0
            m2 = np.sum(centered * centered, axis=reduce_dims, dtype=np.float64)
    except FloatingPointError as exc:
        raise FloatingPointError(f"Reducing {source_path} exceeded float64 range") from exc
    return ChannelStats(counts=counts, means=means, m2=m2)


def _partial_stats_for_record(
    record: ForecastRecord,
    indices: np.ndarray,
    hours: list[int],
    water_mask: np.ndarray | None,
) -> ChannelStats:
    field = np.load(record.path, mmap_mode="r")
    if np.any(indices < 0) or int(indices.max()) >= field.shape[0]:
        raise IndexError(
            f"Configured indices {indices.tolist()} exceed {field.shape[0]} channels in {record.path}"
        )

    if field.ndim == 4:
        n_hours = field.shape[1]
        for hour in hours:
            if not -n_hours <= hour < n_hours:
                raise IndexError(f"hour={hour} is outside the {n_hours} time slices in {record.path}")
        selected_channels = np.array(field[indices], copy=True)  # [C, T, H, W]
        if len(hours) == n_hours and list(hours) == list(range(n_hours)):
            block = selected_channels
        else:
            block = selected_channels[:, list(hours)]
    elif field.ndim == 3:
        block = np.array(field[indices], copy=True)  # [C, H, W]
    else:
        raise ValueError(f"Expected forecast array [V,T,H,W] or [V,H,W], got {field.shape} in {record.path}")

    return _stats_from_block(block, water_mask, record.path)


def _default_workers() -> int:
    return min(8, os.cpu_count() or 1)


def compute_m2m_channel_stats(
    config: dict,
    split: str = "train",
    use_water_mask: bool = True,
    workers: int | None = None,
    hour_mode: str | None = None,
) -> dict:
    if config.get("dataset_name") != "M2MForecastDataset":
        raise ValueError("M2M channel statistics require dataset_name=M2MForecastDataset")

    merged, records = _paired_records(config, split)
    indices = np.asarray([int(value) for value in merged["indices"]], dtype=np.int64)
    if indices.size == 0:
        raise ValueError("indices must select at least one channel")
    fields = list(merged.get("fields", [f"channel_{index}" for index in indices]))
    if len(fields) != int(indices.size):
        raise ValueError("fields and indices must have the same length")

    hour_index = int(merged.get("target_hour_index", 23))
    resolved_hour_mode = hour_mode if hour_mode is not None else str(merged.get("hour_mode", "fixed"))
    if resolved_hour_mode not in ("fixed", "all"):
        raise ValueError(f"Unknown hour_mode={resolved_hour_mode!r}; expected 'fixed' or 'all'")
    hours = list(range(24)) if resolved_hour_mode == "all" else [hour_index]
    water_mask = _water_mask(merged, int(indices.size)) if use_water_mask else None
    workers = _default_workers() if workers is None else int(workers)
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")

    stats = ChannelStats.empty(int(indices.size))
    worker = partial(_partial_stats_for_record, indices=indices, hours=hours, water_mask=water_mask)
    desc = f"m2m_stats {split} hour_mode={resolved_hour_mode}"
    progress = tqdm(total=len(records), desc=desc, unit="file")
    try:
        if workers == 1:
            for record_stats in map(worker, records):
                stats.merge(record_stats)
                progress.update(1)
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                for record_stats in executor.map(worker, records):
                    stats.merge(record_stats)
                    progress.update(1)
    finally:
        progress.close()

    if np.any(stats.counts == 0):
        raise ValueError(f"At least one selected channel has no valid values: counts={stats.counts.tolist()}")

    unique_files = {record.path for record in records}
    variances = stats.population_variance()
    stds = np.sqrt(variances)
    if np.any(~np.isfinite(stds)) or np.any(stds <= 0.0):
        raise ValueError(f"At least one selected channel has unusable std values: std={stds.tolist()}")

    return {
        "dataset_name": merged["dataset_name"],
        "dataset_dir": merged["dataset_dir"],
        "split": split,
        "record_set": "truth",
        "lead_time_hours": merged.get("lead_time_hours"),
        "hour_mode": resolved_hour_mode,
        "hours": hours,
        "target_hour_index": hour_index,
        "use_water_mask": bool(use_water_mask and water_mask is not None),
        "mask_path": merged.get("mask_path") if use_water_mask else None,
        "fields": fields,
        "indices": indices.tolist(),
        "workers": workers,
        "records_seen": len(records),
        "unique_files_seen": len(unique_files),
        "counts": stats.counts.tolist(),
        "mean": stats.means.tolist(),
        "variance": variances.tolist(),
        "std": stds.tolist(),
        "data_config_values": {
            "means": stats.means.tolist(),
            "stds": stds.tolist(),
        },
    }


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def update_data_config(path: str | Path, payload: dict) -> None:
    config = load_json(path)
    config["means"] = payload["mean"]
    config["stds"] = payload["std"]
    write_json(path, config)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute M2M normalization statistics for configured channels."
    )
    parser.add_argument("--config", required=True, help="Path to an M2M data JSON config")
    parser.add_argument("--output", help="Optional JSON path for full statistics and provenance")
    parser.add_argument("--split", default="train", help="Split config used to select record dates")
    parser.add_argument(
        "--workers",
        type=int,
        default=_default_workers(),
        help="Number of threads used for per-file partial statistics",
    )
    parser.add_argument(
        "--include-land",
        action="store_true",
        help="Ignore mask_path and include all finite pixels in the statistics",
    )
    parser.add_argument(
        "--update-config",
        action="store_true",
        help="Write computed mean/std values into --config after the computation succeeds",
    )
    parser.add_argument(
        "--hour-mode",
        choices=("fixed", "all", "auto"),
        default="auto",
        help=(
            "'fixed' uses target_hour_index, 'all' iterates 24 hours per file, "
            "'auto' reads hour_mode from the config"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config)
    hour_mode_override = None if args.hour_mode == "auto" else args.hour_mode
    payload = compute_m2m_channel_stats(
        load_json(config_path),
        split=args.split,
        use_water_mask=not args.include_land,
        workers=args.workers,
        hour_mode=hour_mode_override,
    )
    if args.output:
        write_json(args.output, payload)
    if args.update_config:
        update_data_config(config_path, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
