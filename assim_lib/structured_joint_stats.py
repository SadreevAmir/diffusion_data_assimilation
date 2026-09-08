"""Build immutable train-only statistics for the structured SIC/SIT codec.

This is deliberately a CPU streaming preflight.  It reads only the target
fields selected by the configured training pairs and never initializes CUDA.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import timedelta
from pathlib import Path

import numpy as np

from .config import load_json
from .data import M2MForecastDataset
from .forecast import field_at_hour
from .structured_archive_audit import validate_bound_archive_audit
from .structured_joint_state import (
    INACTIVE_FILLER_LAW,
    INTERIOR_VALUE_LAW,
    SCHEMA_VERSION,
    canonical_mapping_sha256,
    dequantized_logit_moments,
    validate_structured_state_stats,
)

_SIC_LOGIT_RANGE = (-24.0, 24.0)
_SIT_LOG_RANGE = (-24.0, 8.0)
_HISTOGRAM_BINS = 131_072


class _StreamingMomentHistogram:
    def __init__(self, lower: float, upper: float, bins: int = _HISTOGRAM_BINS):
        self.lower = float(lower)
        self.upper = float(upper)
        self.edges = np.linspace(self.lower, self.upper, int(bins) + 1, dtype=np.float64)
        self.counts = np.zeros(int(bins), dtype=np.int64)
        self.count = 0
        self.total = 0.0
        self.total_square = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if values.size == 0:
            return
        if not np.all(np.isfinite(values)):
            raise ValueError("structured-state statistic contains a non-finite value")
        minimum = float(values.min())
        maximum = float(values.max())
        if minimum < self.lower or maximum > self.upper:
            raise ValueError(
                f"statistic range [{minimum}, {maximum}] exceeds fixed histogram "
                f"range [{self.lower}, {self.upper}]"
            )
        self.counts += np.histogram(values, bins=self.edges)[0]
        self.count += int(values.size)
        self.total += float(values.sum(dtype=np.float64))
        self.total_square += float(np.square(values).sum(dtype=np.float64))
        self.minimum = min(self.minimum, minimum)
        self.maximum = max(self.maximum, maximum)

    def moments(self) -> tuple[float, float]:
        if self.count < 2:
            raise ValueError("at least two active values are required for structured statistics")
        mean = self.total / self.count
        variance = max(self.total_square / self.count - mean * mean, 0.0)
        std = math.sqrt(variance)
        if not math.isfinite(std) or std <= 0.0:
            raise ValueError("structured statistic has zero or invalid variance")
        return mean, std


def build_structured_state_stats(data_config: dict, data_config_path: str | Path) -> dict:
    """Stream the exact configured train pairs and construct the codec contract."""
    archive_audit = validate_bound_archive_audit(data_config, Path(data_config_path).resolve())
    if "structured_sic_cap" not in data_config:
        raise ValueError("data config must explicitly declare structured_sic_cap")
    cap = float(data_config["structured_sic_cap"])
    if not 0.0 < cap <= 1.0:
        raise ValueError("structured_sic_cap must lie in (0, 1]")

    dataset = M2MForecastDataset(data_config, split="train")
    dataset.validate_structured_sral_audit_contract(archive_audit)
    if dataset.indices != [0, 1]:
        raise ValueError("structured statistics require data indices [0, 1] (SIC, SIT)")
    if dataset.background_strategy != "calendar_year_ago":
        raise ValueError("structured statistics require exact calendar-year train pairs")

    sic_interior = _StreamingMomentHistogram(*_SIC_LOGIT_RANGE)
    sit_positive = _StreamingMomentHistogram(*_SIT_LOG_RANGE)
    valid_count = 0
    ice_count = 0
    cap_count = 0
    sample_count = 0
    joint_nan_count = 0
    physical_total = np.zeros(2, dtype=np.float64)
    physical_total_square = np.zeros(2, dtype=np.float64)
    distinct_target_paths: set[str] = set()
    forcing_indices = tuple(dataset.dynamic_forcing_indices)
    forcing_total = np.zeros(len(forcing_indices), dtype=np.float64)
    forcing_total_square = np.zeros(len(forcing_indices), dtype=np.float64)
    forcing_count = np.zeros(len(forcing_indices), dtype=np.int64)
    hours = range(24) if dataset.hour_mode == "all" else (dataset.hour_index,)
    base_valid = dataset.base_valid_mask[0].detach().cpu().numpy() > 0
    tolerance = max(np.finfo(np.float32).eps * max(cap, 1.0) * 2.0, 1e-7)

    for _, anchor_record in dataset.calendar_pairs:
        if forcing_indices:
            for hour in hours:
                raw_forcing = np.asarray(field_at_hour(anchor_record.path, hour, forcing_indices, copy=False))
                if np.any(np.isinf(raw_forcing)):
                    raise ValueError(f"infinite dynamic forcing value in {anchor_record.path}")
                ocean = base_valid[: raw_forcing.shape[-2], : raw_forcing.shape[-1]]
                for channel, values in enumerate(raw_forcing):
                    selected = np.asarray(values, dtype=np.float64)[ocean & np.isfinite(values)]
                    if selected.size == 0:
                        continue
                    forcing_total[channel] += selected.sum(dtype=np.float64)
                    forcing_total_square[channel] += np.square(selected).sum(dtype=np.float64)
                    forcing_count[channel] += selected.size
        for lead_days in dataset.trajectory_lead_days:
            target_record = dataset.records_by_date[anchor_record.date + timedelta(days=lead_days)]
            distinct_target_paths.add(str(target_record.path))
            for hour in hours:
                raw = field_at_hour(target_record.path, hour, dataset.indices, copy=False)
                raw = np.asarray(raw)
                if raw.ndim != 3 or raw.shape[0] != 2:
                    raise ValueError(f"expected target [2,H,W], got {raw.shape} in {target_record.path}")
                expected_dtype = data_config.get("source_array_dtype")
                if expected_dtype is not None and raw.dtype.name != str(expected_dtype):
                    raise ValueError(
                        f"target dtype {raw.dtype.name} != declared source_array_dtype "
                        f"{expected_dtype!r} in {target_record.path}"
                    )
                if raw.shape[-2] > base_valid.shape[-2] or raw.shape[-1] > base_valid.shape[-1]:
                    raise ValueError(
                        f"target shape {raw.shape[-2:]} exceeds valid mask {base_valid.shape[-2:]}"
                    )
                valid = base_valid[: raw.shape[-2], : raw.shape[-1]]
                sic_raw = raw[0]
                sit_raw = raw[1]
                if np.any(np.isinf(sic_raw)) or np.any(np.isinf(sit_raw)):
                    raise ValueError(f"infinite SIC/SIT value in {target_record.path}")
                sic_nan = np.isnan(sic_raw) & valid
                sit_nan = np.isnan(sit_raw) & valid
                if not np.array_equal(sic_nan, sit_nan):
                    raise ValueError(
                        f"SIC/SIT NaN patterns differ on the water domain in {target_record.path}"
                    )
                sic = np.where(np.isfinite(sic_raw), sic_raw, 0.0).astype(np.float64, copy=False)
                sit = np.where(np.isfinite(sit_raw), sit_raw, 0.0).astype(np.float64, copy=False)
                sic_valid = sic[valid]
                sit_valid = sit[valid]
                physical_values = (sic_valid, sit_valid)
                for channel, values in enumerate(physical_values):
                    physical_total[channel] += values.sum(dtype=np.float64)
                    physical_total_square[channel] += np.square(values).sum(dtype=np.float64)
                if np.any(sic_valid < 0.0) or np.any(sic_valid > cap + tolerance):
                    raise ValueError(f"SIC outside [0, {cap}] in {target_record.path}")
                if np.any(sit_valid < 0.0):
                    raise ValueError(f"negative SIT in {target_record.path}")
                ice = sic_valid > 0.0
                if not np.array_equal(ice, sit_valid > 0.0):
                    raise ValueError(f"SIC/SIT joint occurrence invariant failed in {target_record.path}")
                capped = ice & (np.abs(sic_valid - cap) <= tolerance)
                interior = ice & ~capped
                if np.any(sic_valid[interior] >= cap):
                    raise ValueError(f"non-cap SIC reaches cap in {target_record.path}")

                if np.any(interior):
                    ratio = sic_valid[interior] / cap
                    sic_interior.update(np.log(ratio) - np.log1p(-ratio))
                if np.any(ice):
                    sit_positive.update(np.log(sit_valid[ice]))
                valid_count += int(valid.sum())
                ice_count += int(ice.sum())
                cap_count += int(capped.sum())
                joint_nan_count += int(sic_nan.sum())
                sample_count += 1

    if not 0 < ice_count < valid_count or not 0 < cap_count < ice_count:
        raise ValueError("training data must contain open water, ice, capped ice, and non-capped ice")
    occurrence_probability = ice_count / valid_count
    cap_probability = cap_count / ice_count
    occurrence_mean, occurrence_std = dequantized_logit_moments(occurrence_probability)
    cap_mean, cap_std = dequantized_logit_moments(cap_probability)
    sic_mean, sic_std = sic_interior.moments()
    sit_mean, sit_std = sit_positive.moments()
    conditioning_means = physical_total / valid_count
    conditioning_variances = np.maximum(
        physical_total_square / valid_count - np.square(conditioning_means), 0.0
    )
    conditioning_stds = np.sqrt(conditioning_variances)
    if not np.all(np.isfinite(conditioning_stds)) or np.any(conditioning_stds <= 0.0):
        raise ValueError("train-only physical conditioning normalization is degenerate")
    normalized_data_config = dict(data_config)
    normalized_data_config["means"] = conditioning_means.tolist()
    normalized_data_config["stds"] = conditioning_stds.tolist()
    dynamic_forcing_stats = None
    if forcing_indices:
        if np.any(forcing_count < 2):
            raise ValueError("dynamic forcing has fewer than two finite train values")
        forcing_means = forcing_total / forcing_count
        forcing_variances = np.maximum(forcing_total_square / forcing_count - np.square(forcing_means), 0.0)
        forcing_stds = np.sqrt(forcing_variances)
        if not np.all(np.isfinite(forcing_stds)) or np.any(forcing_stds <= 0.0):
            raise ValueError("train-only dynamic forcing normalization is degenerate")
        dynamic_forcing_stats = {
            "source_split": "train",
            "indices": list(forcing_indices),
            "means": forcing_means.tolist(),
            "stds": forcing_stds.tolist(),
            "finite_value_count_per_field": forcing_count.tolist(),
        }
        normalized_data_config["dynamic_forcing_stats"] = dynamic_forcing_stats
    stats = {
        "schema_version": SCHEMA_VERSION,
        "source_split": "train",
        "inactive_filler_law": INACTIVE_FILLER_LAW,
        "interior_value_law": INTERIOR_VALUE_LAW,
        "data_config_sha256": canonical_mapping_sha256(normalized_data_config),
        "pair_manifest_sha256": dataset.provenance()["pair_manifest_sha256"],
        "conditioning_normalization": {
            "source_split": "train",
            "fields": ["siconc", "sithic"],
            "open_water_nan_as_physical_zero": True,
            "estimator": "population_moments_over_all_valid_ocean_trajectory_values",
            "means": conditioning_means.tolist(),
            "stds": conditioning_stds.tolist(),
            "value_count_per_field": valid_count,
            "data_config_update_required": {
                "means": conditioning_means.tolist(),
                "stds": conditioning_stds.tolist(),
                "dynamic_forcing_stats": dynamic_forcing_stats,
            },
        },
        "dynamic_forcing_normalization": dynamic_forcing_stats,
        "sic_cap": cap,
        "occurrence_probability": occurrence_probability,
        "cap_probability_given_ice": cap_probability,
        "occurrence_logit_mean": occurrence_mean,
        "occurrence_logit_std": occurrence_std,
        "cap_logit_mean": cap_mean,
        "cap_logit_std": cap_std,
        "sic_interior_logit_mean": sic_mean,
        "sic_interior_logit_std": sic_std,
        "sit_positive_log_mean": sit_mean,
        "sit_positive_log_std": sit_std,
        "audit": {
            "sample_count": sample_count,
            "anchor_count": len(dataset),
            "trajectory_steps": len(dataset.trajectory_lead_days),
            "trajectory_lead_days": list(dataset.trajectory_lead_days),
            "distinct_target_record_count": len(distinct_target_paths),
            "valid_value_count": valid_count,
            "ice_value_count": ice_count,
            "cap_value_count": cap_count,
            "joint_nan_value_count": joint_nan_count,
            "source_array_dtype": data_config.get("source_array_dtype"),
            "nan_semantics": data_config.get("model_nan_semantics"),
            "hour_mode": dataset.hour_mode,
            "target_slice_index": dataset.hour_index,
            "trajectory_semantics": data_config.get("trajectory_semantics"),
            "utc_time_coordinate_verified": data_config.get("utc_time_coordinate_verified"),
            "histogram_bins": _HISTOGRAM_BINS,
            "sic_interior_logit_range": list(_SIC_LOGIT_RANGE),
            "sit_positive_log_range": list(_SIT_LOG_RANGE),
        },
    }
    validate_structured_state_stats(stats)
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_config_path = Path(args.data_config).expanduser().resolve()
    data_config = load_json(data_config_path)
    stats = build_structured_state_stats(data_config, data_config_path)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"output": str(output.resolve()), "audit": stats["audit"]}, indent=2))


if __name__ == "__main__":
    main()
