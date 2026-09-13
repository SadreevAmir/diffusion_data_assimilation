"""CPU-only conditioning statistics for the heldout-2021 IDEA-F1 screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np

from .config import load_json
from .data import M2MForecastDataset
from .forecast import field_at_hour


MODE = "direct_dynamics_mixed_support_train2016_2020_stats_v1"


def apply_verified_conditioning_stats(
    data_config: dict[str, Any], spec: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind a data config to the immutable train-2016-2020 stats artifact."""
    path = Path(spec["path"])
    if not path.is_file():
        raise ValueError("conditioning stats artifact is missing")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != spec.get("sha256"):
        raise ValueError("conditioning stats artifact SHA mismatch")
    stats = load_json(path)
    if (
        stats.get("schema_version") != MODE
        or stats.get("source_split") != "train_years_2016_2020"
        or stats.get("heldout_year_accessed") is not False
        or stats.get("test_2023_accessed") is not False
    ):
        raise ValueError("conditioning stats provenance is not heldout-safe")
    updated = dict(data_config)
    updated["means"] = stats["state"]["means"]
    updated["stds"] = stats["state"]["stds"]
    updated["dynamic_forcing_stats"] = {
        "source_split": "train",
        "source_years": [2016, 2017, 2018, 2019, 2020],
        "heldout_year": 2021,
        "indices": stats["forcing"]["indices"],
        "means": stats["forcing"]["means"],
        "stds": stats["forcing"]["stds"],
        "finite_value_count_per_field": stats["forcing"]["finite_value_count_per_field"],
    }
    return updated, {"path": str(path), "sha256": digest, "schema_version": MODE}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _moments(total: np.ndarray, square: np.ndarray, count: np.ndarray) -> tuple[list[float], list[float]]:
    if np.any(count < 2):
        raise ValueError("conditioning statistic has fewer than two values")
    mean = total / count
    variance = np.maximum(square / count - mean * mean, 0.0)
    std = np.sqrt(variance)
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)) or np.any(std <= 0):
        raise ValueError("conditioning normalization is degenerate")
    return mean.tolist(), std.tolist()


def run(config_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    config, output_path = load_json(config_path), Path(output_path)
    if config.get("mode") != MODE or config.get("resource_kind") != "cpu":
        raise ValueError("unreviewed stats mode/resource")
    if config.get("train_years") != [2016, 2017, 2018, 2019, 2020] or config.get("heldout_year") != 2021:
        raise ValueError("stats require train 2016-2020 and heldout 2021")
    if bool(config.get("test_2023_access_allowed", True)):
        raise ValueError("test-2023 must remain sealed")
    data_config = dict(load_json(config["dataset_config"]))
    data_config["dynamic_forcing_stats"] = config["bootstrap_dynamic_forcing_stats"]
    dataset = M2MForecastDataset(data_config, split="train")
    if dataset.hour_mode != "fixed" or dataset.hour_index != 23:
        raise ValueError("IDEA-F1 stats require fixed archive slice 23")
    valid = dataset.base_valid_mask[0].numpy() > 0
    state_total = np.zeros(2, dtype=np.float64)
    state_square = np.zeros(2, dtype=np.float64)
    state_count = np.zeros(2, dtype=np.int64)
    forcing_indices = tuple(dataset.dynamic_forcing_indices)
    forcing_total = np.zeros(len(forcing_indices), dtype=np.float64)
    forcing_square = np.zeros(len(forcing_indices), dtype=np.float64)
    forcing_count = np.zeros(len(forcing_indices), dtype=np.int64)
    dates = []
    for _, anchor in dataset.calendar_pairs:
        if anchor.date.year not in config["train_years"]:
            continue
        if (anchor.date + timedelta(days=max(dataset.trajectory_lead_days))).year not in config["train_years"]:
            continue
        raw_state = np.asarray(field_at_hour(anchor.path, 23, [0, 1], copy=False))
        if np.any(np.isinf(raw_state)):
            raise ValueError(f"Inf in state conditioning {anchor.path}")
        for channel, values in enumerate(raw_state):
            selected = np.where(np.isfinite(values), values, 0.0)[valid[: values.shape[0], : values.shape[1]]]
            selected = np.asarray(selected, dtype=np.float64)
            state_total[channel] += selected.sum(dtype=np.float64)
            state_square[channel] += np.square(selected).sum(dtype=np.float64)
            state_count[channel] += selected.size
        raw_forcing = np.asarray(field_at_hour(anchor.path, 23, list(forcing_indices), copy=False))
        if np.any(np.isinf(raw_forcing)):
            raise ValueError(f"Inf in forcing conditioning {anchor.path}")
        for channel, values in enumerate(raw_forcing):
            ocean = valid[: values.shape[0], : values.shape[1]]
            selected = np.asarray(values, dtype=np.float64)[ocean & np.isfinite(values)]
            forcing_total[channel] += selected.sum(dtype=np.float64)
            forcing_square[channel] += np.square(selected).sum(dtype=np.float64)
            forcing_count[channel] += selected.size
        dates.append(anchor.date.isoformat())
    if not dates or any(date.startswith("2021-") for date in dates):
        raise RuntimeError("train-only date inventory is invalid")
    state_means, state_stds = _moments(state_total, state_square, state_count)
    forcing_means, forcing_stds = _moments(forcing_total, forcing_square, forcing_count)
    result = {
        "schema_version": MODE,
        "status": "complete",
        "resource_kind": "cpu",
        "source_split": "train_years_2016_2020",
        "heldout_year_accessed": False,
        "test_2023_accessed": False,
        "target_slice_index": 23,
        "window_rule": "d0 through d9 entirely within 2016-2020",
        "anchor_count": len(dates),
        "anchor_dates_sha256": hashlib.sha256("\n".join(dates).encode()).hexdigest(),
        "state": {"indices": [0, 1], "means": state_means, "stds": state_stds,
                  "value_count_per_field": state_count.tolist(), "nan_as_open_water_zero": True},
        "forcing": {"indices": list(forcing_indices), "means": forcing_means, "stds": forcing_stds,
                    "finite_value_count_per_field": forcing_count.tolist()},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("output")
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output), indent=2))


if __name__ == "__main__":
    main()
