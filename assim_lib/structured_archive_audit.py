"""Full-archive CPU audit for structured SIC/SIT trajectory training.

The audit decides missingness from the static land mask and observed archive
patterns.  It never imputes dynamic features and never initializes CUDA.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from .config import load_json
from .data import _combine_sral_files, _sral_transform
from .forecast import as_date, forecast_records, parse_date
from .structured_joint_state import canonical_mapping_sha256

VARIABLES = {
    0: "siconc",
    1: "sithic",
    6: "current_grid_x_archive_units",
    7: "current_grid_y_archive_units",
    13: "wind_grid_x_archive_units",
    14: "wind_grid_y_archive_units",
}

AUDIT_SCHEMA_VERSION = "structured_archive_semantics_audit_v3"
TRAJECTORY_SEMANTICS = "consecutive_daily_archive_snapshots"
ALLOWED_TRAJECTORY_SEMANTICS = {
    TRAJECTORY_SEMANTICS: (0, 1, 2, 3),
    "analysis_snapshot_only": (0,),
    "state_only_forecast_snapshots_d_plus_3_6_9": (3, 6, 9),
}
TIME_CLAIM_POLICY = "archive_date_only_no_utc_or_operational_lead_claim"
FORBIDDEN_TIME_LABELS = ["operational_lead", "23:00_UTC", "24h_issue_time"]


def _previous_calendar_date(value: date) -> date | None:
    try:
        return value.replace(year=value.year - 1)
    except ValueError:
        return None


def _trajectory_date_audit(config: dict, records) -> dict:
    """Prove the exact date law used by every retained trajectory anchor."""
    records_by_date = {record.date: record for record in records}
    duplicate_date_count = len(records) - len(records_by_date)
    lead_days = tuple(
        int(value)
        for value in config.get(
            "trajectory_lead_days",
            range(int(config.get("future_horizon_days", 0)) + 1),
        )
    )
    horizon = max(lead_days) if lead_days else -1
    needs_background = config.get("background_strategy") == "calendar_year_ago"
    split_results = {}
    all_verified = (
        duplicate_date_count == 0
        and bool(lead_days)
        and tuple(sorted(set(lead_days))) == lead_days
        and all(value >= 0 for value in lead_days)
    )

    for split_name in ("train", "valid", "test"):
        split = config[split_name]
        background_start = as_date(split["back_start_day"])
        background_end = as_date(split["back_end_day"])
        target_start = as_date(split["obs_start_day"])
        target_end = as_date(split["obs_end_day"])
        target_records = sorted(value for value in records_by_date if target_start <= value <= target_end)
        contained_candidates = [
            value for value in target_records if value + timedelta(days=horizon) <= target_end
        ]
        eligible = []
        missing_target_trajectory = 0
        missing_exact_background_trajectory = 0
        for anchor in contained_candidates:
            target_dates = [anchor + timedelta(days=lead) for lead in lead_days]
            background_target_dates = (
                target_dates
                if config.get("conditioning_layout") == "structured_sic_sit_trajectory_v1"
                else [anchor]
            )
            background_dates = [_previous_calendar_date(value) for value in background_target_dates]
            targets_ok = all(value in records_by_date for value in target_dates)
            backgrounds_ok = not needs_background or all(
                value is not None and background_start <= value <= background_end and value in records_by_date
                for value in background_dates
            )
            if not targets_ok:
                missing_target_trajectory += 1
            if not backgrounds_ok:
                missing_exact_background_trajectory += 1
            if targets_ok and backgrounds_ok:
                eligible.append(anchor)

        split_verified = bool(eligible) and all(
            target_start <= anchor and anchor + timedelta(days=horizon) <= target_end for anchor in eligible
        )
        all_verified = all_verified and split_verified
        split_results[split_name] = {
            "target_record_count": len(target_records),
            "contained_anchor_candidate_count": len(contained_candidates),
            "eligible_anchor_count": len(eligible),
            "missing_target_trajectory_count": missing_target_trajectory,
            "missing_exact_previous_calendar_background_trajectory_count": (
                missing_exact_background_trajectory
            ),
            "first_eligible_anchor": eligible[0].isoformat() if eligible else None,
            "last_eligible_anchor": eligible[-1].isoformat() if eligible else None,
            "all_retained_targets_match_declared_lead_days": split_verified,
            "all_retained_trajectories_stay_inside_split": split_verified,
            "all_required_backgrounds_are_exact_previous_calendar_dates": split_verified,
        }

    return {
        "status": "verified" if all_verified else "rejected",
        "trajectory_horizon_days": horizon,
        "trajectory_lead_days": list(lead_days),
        "duplicate_archive_date_count": duplicate_date_count,
        "lead_labels": [
            "analysis_snapshot_d" if lead == 0 else f"forecast_archive_day_d_plus_{lead}"
            for lead in lead_days
        ],
        "splits": split_results,
    }


def validate_trajectory_time_contract(config: dict, audit: dict) -> None:
    """Admit the narrow archive-date law without making an unproved UTC claim."""
    semantics = config.get("trajectory_semantics")
    expected_leads = ALLOWED_TRAJECTORY_SEMANTICS.get(semantics)
    if expected_leads is None:
        raise ValueError(f"unknown structured trajectory semantics: {semantics!r}")
    actual_leads = tuple(
        int(value)
        for value in config.get(
            "trajectory_lead_days",
            range(int(config.get("future_horizon_days", 0)) + 1),
        )
    )
    if actual_leads != expected_leads:
        raise ValueError(
            f"trajectory semantics {semantics!r} requires lead days {expected_leads}, got {actual_leads}"
        )
    expected = {
        "trajectory_semantics": semantics,
        "target_slice_index": 23,
        "utc_time_coordinate_verified": False,
        "time_claim_policy": TIME_CLAIM_POLICY,
        "forbidden_time_labels": FORBIDDEN_TIME_LABELS,
    }
    actual = {key: config.get(key) for key in expected}
    mismatches = {
        key: (actual[key], expected_value)
        for key, expected_value in expected.items()
        if actual[key] != expected_value
    }
    if mismatches:
        raise ValueError(f"unproved or incompatible archive-time contract: {mismatches}")
    if audit.get("trajectory_date_audit", {}).get("status") != "verified":
        raise ValueError("consecutive archive-date trajectory audit did not pass")
    if audit.get("trajectory_semantics") != semantics:
        raise ValueError("archive audit trajectory semantics do not match the launch contract")
    if int(audit.get("target_slice_index", -1)) != 23:
        raise ValueError("archive audit did not bind target_slice_index=23")
    if audit.get("utc_time_coordinate_verified") is not False:
        raise ValueError("this run is admitted only without a UTC time-coordinate claim")
    if audit.get("time_claim_policy") != TIME_CLAIM_POLICY:
        raise ValueError("archive audit does not enforce the narrow time-claim policy")


def _protocol_payload(config: dict) -> dict:
    # Numeric normalization is estimated later from train only and bound by the
    # v3 statistics artifact.  It does not alter raw archive/mask semantics, so
    # excluding it avoids a circular audit -> stats -> config -> audit workflow.
    return {
        key: value
        for key, value in config.items()
        if key
        not in {
            "archive_semantics_audit_path",
            "archive_semantics_audit_sha256",
            "means",
            "stds",
            "dynamic_forcing_stats",
        }
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _static_mask_provenance(config: dict) -> tuple[np.ndarray, dict]:
    path = Path(config["mask_path"])
    if not path.is_file():
        raise ValueError(f"static land mask is missing: {path}")
    raw = np.load(path, mmap_mode="r")
    if raw.ndim != 2:
        raise ValueError(f"land mask must be 2D, got {raw.shape}")
    if not np.all(np.isfinite(raw)):
        raise ValueError("static land mask must be finite everywhere")
    if not np.all((raw == 0) | (raw == 1)):
        raise ValueError("static land mask must be exactly binary {0,1}")
    ocean = raw == (0 if bool(config.get("mask_true_is_invalid", True)) else 1)
    if not np.any(ocean):
        raise ValueError("static land mask contains no valid ocean cells")
    provenance = {
        "declared_path": str(config["mask_path"]),
        "resolved_path": str(path.resolve()),
        "content_sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
        "array_dtype": raw.dtype.name,
        "array_shape": [int(value) for value in raw.shape],
        "mask_true_is_invalid": bool(config.get("mask_true_is_invalid", True)),
        "valid_ocean_count": int(ocean.sum()),
    }
    return np.asarray(raw, dtype=np.float32), provenance


def _candidate_sral_lag_dates(config: dict, records) -> dict[str, dict[int, set[date]]]:
    records_by_date = {record.date: record for record in records}
    lead_days = tuple(
        int(value)
        for value in config.get(
            "trajectory_lead_days",
            range(int(config["future_horizon_days"]) + 1),
        )
    )
    uses_lag_background = config.get("conditioning_layout") == "structured_sic_sit_trajectory_v1"
    candidates = {split_name: {lag: set() for lag in range(3)} for split_name in ("train", "valid", "test")}
    for split_name in ("train", "valid", "test"):
        split = config[split_name]
        background_start = as_date(split["back_start_day"])
        background_end = as_date(split["back_end_day"])
        target_start = as_date(split["obs_start_day"])
        target_end = as_date(split["obs_end_day"])
        for anchor in sorted(records_by_date):
            if not target_start <= anchor <= target_end:
                continue
            target_dates = [anchor + timedelta(days=lead) for lead in lead_days]
            if max(target_dates) > target_end or not all(value in records_by_date for value in target_dates):
                continue
            background_dates = [_previous_calendar_date(anchor)]
            if not all(
                value is not None and background_start <= value <= background_end and value in records_by_date
                for value in background_dates
            ):
                continue
            for lag in range(3):
                value_date = anchor - timedelta(days=lag)
                background_date = _previous_calendar_date(value_date)
                if value_date in records_by_date and (
                    not uses_lag_background or background_date in records_by_date
                ):
                    candidates[split_name][lag].add(value_date)
    return candidates


def _pixel_coverage_summary(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "median": None, "p90": None, "max": None, "total": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": int(array.min()),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "max": int(array.max()),
        "total": int(array.sum()),
    }


def _sral_provenance(config: dict, records, ocean_full: np.ndarray) -> dict:
    if config.get("observation_mask", {}).get("kind") == "none":
        return {
            "used_by_model": False,
            "selection": "none",
            "availability_by_split_and_lag": {},
        }
    root = Path(config["sral_dir"])
    if not root.is_dir():
        raise ValueError(f"SRAL geometry directory is missing: {root}")
    files = sorted(path for path in root.glob("*.npy") if path.is_file())
    if not files:
        raise ValueError(f"SRAL geometry directory contains no .npy files: {root}")
    digest = hashlib.sha256(b"structured_sral_content_manifest_v1\0")
    shape_counts: dict[str, int] = defaultdict(int)
    dtype_counts: dict[str, int] = defaultdict(int)
    date_counts: dict[date, int] = defaultdict(int)
    date_finite_valid_geometry: dict[date, int] = defaultdict(int)
    paths_by_date: dict[date, list[Path]] = defaultdict(list)
    finite_valid_geometry_count = 0
    transform_index = int(config["observation_mask"]["sral_transform_index"])
    for path in files:
        array = np.load(path, mmap_mode="r")
        parsed = parse_date(path.name)
        if parsed is None:
            raise ValueError(f"SRAL filename has no unambiguous date: {path}")
        date_counts[parsed] += 1
        paths_by_date[parsed].append(path)
        if array.ndim != 3:
            raise ValueError(f"SRAL source must be [C,H,W], got {array.shape} in {path}")
        if transform_index >= array.shape[0]:
            raise ValueError(
                f"SRAL transform index {transform_index} exceeds {array.shape[0]} channels in {path}"
            )
        if np.any(np.isinf(array)):
            raise ValueError(f"SRAL source contains Inf: {path}")
        shape_counts[str(tuple(int(value) for value in array.shape))] += 1
        dtype_counts[array.dtype.name] += 1
        relative = path.relative_to(root).as_posix()
        file_sha = _sha256_file(path)
        digest.update(f"{relative}\0{path.stat().st_size}\0{file_sha}\n".encode())
    if len(shape_counts) != 1:
        raise ValueError(f"SRAL archive shape is inconsistent: {dict(shape_counts)}")
    for parsed, paths in sorted(paths_by_date.items()):
        combined = _combine_sral_files(paths)
        if combined is None:
            raise ValueError(f"SRAL combination unexpectedly empty for {parsed}")
        geometry = _sral_transform(combined, transform_index)
        height, width = geometry.shape
        if height > ocean_full.shape[0] or width > ocean_full.shape[1]:
            raise ValueError(f"SRAL geometry {geometry.shape} exceeds static mask {ocean_full.shape}")
        valid_geometry = int((np.isfinite(geometry) & ocean_full[:height, :width]).sum())
        if valid_geometry <= 0:
            raise ValueError(
                f"SRAL date {parsed} has files but no usable geometry after runtime combine/transform"
            )
        finite_valid_geometry_count += valid_geometry
        date_finite_valid_geometry[parsed] = valid_geometry
    if finite_valid_geometry_count <= 0:
        raise ValueError("SRAL archive has no finite geometry on the valid ocean domain")
    candidate_dates = _candidate_sral_lag_dates(config, records)
    availability_by_split_and_lag = {}
    all_candidates: set[date] = set()
    all_missing: set[date] = set()
    for split_name, split_lags in candidate_dates.items():
        availability_by_split_and_lag[split_name] = {}
        for lag, candidates in split_lags.items():
            available = candidates & set(date_counts)
            usable = {value for value in available if date_finite_valid_geometry[value] > 0}
            missing = candidates - usable
            if candidates and not usable:
                raise ValueError(
                    f"SRAL observation law is empty for split={split_name} lag={lag}: "
                    f"{len(candidates)} candidate dates and zero usable dates"
                )
            all_candidates.update(candidates)
            all_missing.update(missing)
            availability_by_split_and_lag[split_name][f"lag{lag}"] = {
                "candidate_date_count": len(candidates),
                "available_file_date_count": len(available),
                "usable_nonempty_geometry_date_count": len(usable),
                "missing_date_count": len(missing),
                "missing_fraction": len(missing) / len(candidates) if candidates else None,
                "available_file_count": sum(date_counts[value] for value in available),
                "first_candidate_date": min(candidates).isoformat() if candidates else None,
                "last_candidate_date": max(candidates).isoformat() if candidates else None,
                "first_usable_date": min(usable).isoformat() if usable else None,
                "last_usable_date": max(usable).isoformat() if usable else None,
                "candidate_dates": [value.isoformat() for value in sorted(candidates)],
                "available_file_dates": [value.isoformat() for value in sorted(available)],
                "usable_nonempty_geometry_dates": [value.isoformat() for value in sorted(usable)],
                "missing_dates": [value.isoformat() for value in sorted(missing)],
                "pixel_coverage_distribution": _pixel_coverage_summary(
                    [date_finite_valid_geometry[value] for value in sorted(usable)]
                ),
            }
    return {
        "declared_root": str(config["sral_dir"]),
        "resolved_root": str(root.resolve()),
        "file_count": len(files),
        "first_file": files[0].name,
        "last_file": files[-1].name,
        "content_manifest_sha256": digest.hexdigest(),
        "array_dtype_counts": dict(sorted(dtype_counts.items())),
        "array_shape_counts": dict(sorted(shape_counts.items())),
        "dated_record_count": len(date_counts),
        "candidate_lag_date_count": len(all_candidates),
        "missing_candidate_lag_date_count": len(all_missing),
        "availability_by_split_and_lag": availability_by_split_and_lag,
        "finite_valid_geometry_count": finite_valid_geometry_count,
        "transform_index": transform_index,
        "selection": "all_top_level_npy_files_loaded_by_M2MForecastDataset",
    }


def _manifest_lines(records) -> list[str]:
    lines = []
    for record in records:
        stat = record.path.stat()
        lines.append(f"{record.date.isoformat()},{record.path.name},{stat.st_size},{stat.st_mtime_ns}\n")
    return lines


def _manifest_sha256(records) -> str:
    return hashlib.sha256("".join(_manifest_lines(records)).encode("utf-8")).hexdigest()


def _update_forecast_content_hash(
    digest,
    record,
    array: np.ndarray,
    time_index: int,
) -> None:
    digest.update(
        (
            f"{record.date.isoformat()}\0{record.path.name}\0{array.dtype.name}\0"
            f"{tuple(int(value) for value in array.shape)}\0{time_index}\n"
        ).encode()
    )
    for index in VARIABLES:
        values = np.ascontiguousarray(array[index, time_index])
        digest.update(f"variable={index}\0".encode("ascii"))
        digest.update(values.tobytes(order="C"))


def _forecast_content_sha256(records, time_index: int) -> str:
    digest = hashlib.sha256(b"structured_forecast_audited_content_v1\0")
    for record in records:
        array = np.load(record.path, mmap_mode="r")
        if (
            array.ndim != 4
            or array.shape[0] <= max(VARIABLES)
            or not -array.shape[1] <= time_index < array.shape[1]
        ):
            raise ValueError(f"unexpected archive shape {array.shape} in {record.path}")
        _update_forecast_content_hash(digest, record, array, time_index)
    return digest.hexdigest()


def _empty_counts() -> dict[str, int | float | None]:
    return {
        "ocean_finite": 0,
        "ocean_nan": 0,
        "ocean_inf": 0,
        "ocean_zero": 0,
        "land_finite": 0,
        "land_nan": 0,
        "land_inf": 0,
        "finite_min": None,
        "finite_max": None,
    }


def _selected_records_and_ranges(config: dict, records):
    ranges = []
    for split_name in ("train", "valid", "test"):
        split = config[split_name]
        ranges.append(
            (
                split_name,
                as_date(split["back_start_day"]),
                as_date(split["back_end_day"]),
                as_date(split["obs_start_day"]),
                as_date(split["obs_end_day"]),
            )
        )
    selected = [
        record
        for record in records
        if any(
            back_start <= record.date <= back_end or target_start <= record.date <= target_end
            for _, back_start, back_end, target_start, target_end in ranges
        )
    ]
    return selected, ranges


def _update_counts(counts: dict, values: np.ndarray, ocean: np.ndarray) -> None:
    finite = np.isfinite(values)
    nan = np.isnan(values)
    inf = np.isinf(values)
    for region, mask in (("ocean", ocean), ("land", ~ocean)):
        counts[f"{region}_finite"] += int((finite & mask).sum())
        counts[f"{region}_nan"] += int((nan & mask).sum())
        counts[f"{region}_inf"] += int((inf & mask).sum())
    counts["ocean_zero"] += int((finite & ocean & (values == 0)).sum())
    finite_values = values[finite]
    if finite_values.size:
        minimum = float(finite_values.min())
        maximum = float(finite_values.max())
        counts["finite_min"] = minimum if counts["finite_min"] is None else min(counts["finite_min"], minimum)
        counts["finite_max"] = maximum if counts["finite_max"] is None else max(counts["finite_max"], maximum)


def build_archive_semantics_audit(config: dict) -> dict:
    records = forecast_records(
        Path(config["dataset_dir"]) / "preds",
        int(config["lead_time_hours"]),
    )
    selected, ranges = _selected_records_and_ranges(config, records)
    if not selected:
        raise ValueError("archive audit selected no records")

    raw_land, static_mask_provenance = _static_mask_provenance(config)
    ocean_full = raw_land <= 0 if bool(config.get("mask_true_is_invalid", True)) else raw_land > 0
    sral_provenance = _sral_provenance(config, records, ocean_full)
    time_index = int(config["target_slice_index"])
    trajectory_date_audit = _trajectory_date_audit(config, records)
    per_variable = {name: _empty_counts() for name in VARIABLES.values()}
    per_role = defaultdict(lambda: {name: _empty_counts() for name in VARIABLES.values()})
    dtype_counts: dict[str, int] = defaultdict(int)
    shape_counts: dict[str, int] = defaultdict(int)
    pair = {
        "ocean_sic_sit_nan_mismatch": 0,
        "ocean_joint_nan": 0,
        "ocean_occurrence_mismatch_after_joint_nan_to_zero": 0,
        "ocean_sic_outside_declared_support_after_joint_nan_to_zero": 0,
        "ocean_negative_sit_after_joint_nan_to_zero": 0,
    }
    cap = float(config["structured_sic_cap"])
    tolerance = max(np.finfo(np.float32).eps * max(cap, 1.0) * 2.0, 1e-7)
    forecast_content_digest = hashlib.sha256(b"structured_forecast_audited_content_v1\0")

    for record in selected:
        array = np.load(record.path, mmap_mode="r")
        dtype_counts[array.dtype.name] += 1
        shape_counts[str(tuple(int(value) for value in array.shape))] += 1
        if (
            array.ndim != 4
            or array.shape[0] <= max(VARIABLES)
            or not -array.shape[1] <= time_index < array.shape[1]
        ):
            raise ValueError(f"unexpected archive shape {array.shape} in {record.path}")
        _update_forecast_content_hash(forecast_content_digest, record, array, time_index)
        height, width = array.shape[-2:]
        if height > ocean_full.shape[0] or width > ocean_full.shape[1]:
            raise ValueError(f"archive field {array.shape[-2:]} exceeds land mask {ocean_full.shape}")
        ocean = ocean_full[:height, :width]
        roles = []
        for split_name, back_start, back_end, target_start, target_end in ranges:
            if back_start <= record.date <= back_end:
                roles.append(f"{split_name}_background")
            if target_start <= record.date <= target_end:
                roles.append(f"{split_name}_target")
        for index, name in VARIABLES.items():
            values = np.asarray(array[index, time_index])
            _update_counts(per_variable[name], values, ocean)
            for role in roles:
                _update_counts(per_role[role][name], values, ocean)

        sic_raw = np.asarray(array[0, time_index])
        sit_raw = np.asarray(array[1, time_index])
        sic_nan = np.isnan(sic_raw) & ocean
        sit_nan = np.isnan(sit_raw) & ocean
        pair["ocean_sic_sit_nan_mismatch"] += int(np.logical_xor(sic_nan, sit_nan).sum())
        joint_nan = sic_nan & sit_nan
        pair["ocean_joint_nan"] += int(joint_nan.sum())
        sic = np.where(joint_nan, 0.0, sic_raw)
        sit = np.where(joint_nan, 0.0, sit_raw)
        pair["ocean_occurrence_mismatch_after_joint_nan_to_zero"] += int(
            (((sic > 0) != (sit > 0)) & ocean).sum()
        )
        pair["ocean_sic_outside_declared_support_after_joint_nan_to_zero"] += int(
            (((sic < 0) | (sic > cap + tolerance)) & ocean).sum()
        )
        pair["ocean_negative_sit_after_joint_nan_to_zero"] += int(((sit < 0) & ocean).sum())

    declared_dtype = str(config.get("source_array_dtype"))
    data_semantics_verified = (
        set(dtype_counts) == {declared_dtype}
        and pair["ocean_sic_sit_nan_mismatch"] == 0
        and pair["ocean_occurrence_mismatch_after_joint_nan_to_zero"] == 0
        and pair["ocean_sic_outside_declared_support_after_joint_nan_to_zero"] == 0
        and pair["ocean_negative_sit_after_joint_nan_to_zero"] == 0
        and all(
            int(values["ocean_inf"]) == 0 and int(values["land_inf"]) == 0 for values in per_variable.values()
        )
        and trajectory_date_audit["status"] == "verified"
        and config.get("trajectory_semantics") in ALLOWED_TRAJECTORY_SEMANTICS
        and time_index == 23
        and config.get("utc_time_coordinate_verified") is False
        and config.get("time_claim_policy") == TIME_CLAIM_POLICY
        and config.get("forbidden_time_labels") == FORBIDDEN_TIME_LABELS
        and tuple(config.get("dynamic_forcing_indices", ())) in {(), (6, 7, 13, 14)}
    )
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "status": "data_semantics_verified" if data_semantics_verified else "rejected",
        "data_protocol_sha256": canonical_mapping_sha256(_protocol_payload(config)),
        "forecast_metadata_manifest_sha256": _manifest_sha256(selected),
        "forecast_audited_content_sha256": forecast_content_digest.hexdigest(),
        "forecast_content_scope": "variables_0_1_6_7_13_14_at_target_slice_index",
        "forecast_declared_root": str(Path(config["dataset_dir"]) / "preds"),
        "forecast_resolved_root": str((Path(config["dataset_dir"]) / "preds").resolve()),
        "static_mask_provenance": static_mask_provenance,
        "sral_provenance": sral_provenance,
        "record_count": len(selected),
        "first_record_date": selected[0].date.isoformat(),
        "last_record_date": selected[-1].date.isoformat(),
        "trajectory_semantics": config.get("trajectory_semantics"),
        "target_slice_index": time_index,
        "utc_time_coordinate_verified": config.get("utc_time_coordinate_verified"),
        "time_claim_policy": config.get("time_claim_policy"),
        "forbidden_time_labels": config.get("forbidden_time_labels"),
        "trajectory_date_audit": trajectory_date_audit,
        "source_array_dtype_declared": declared_dtype,
        "dtype_record_counts": dict(sorted(dtype_counts.items())),
        "shape_record_counts": dict(sorted(shape_counts.items())),
        "variable_mapping": {str(index): name for index, name in VARIABLES.items()},
        "dynamic_feature_policy": {
            "candidate_indices": list(config.get("candidate_dynamic_indices", [])),
            "enabled_indices": list(config.get("dynamic_forcing_indices", [])),
            "nan_rule": "missing_with_explicit_mask; no imputation",
            "physical_units_claim": (
                "none; transformed signed grid displacement requires separate sign/orientation audit"
            ),
        },
        "sic_sit_missingness_rule": (
            "static mask excludes land; paired SIC/SIT NaN on valid ocean is physical open water zero; "
            "Inf or mismatched missingness rejects"
        ),
        "sic_sit_pair_checks": pair,
        "per_variable": per_variable,
        "per_split_role": dict(per_role),
    }


def validate_bound_archive_audit(config: dict, data_config_path: str | Path) -> dict:
    data_config_path = Path(data_config_path).resolve()
    audit_path = Path(config.get("archive_semantics_audit_path", ""))
    if not audit_path.is_absolute():
        audit_path = data_config_path.parent / audit_path
    if not audit_path.is_file():
        raise ValueError(f"full-dataset archive semantics audit is missing: {audit_path}")
    raw = audit_path.read_bytes()
    actual_sha = hashlib.sha256(raw).hexdigest()
    expected_sha = config.get("archive_semantics_audit_sha256")
    if actual_sha != expected_sha:
        raise ValueError(f"archive semantics audit hash mismatch: expected {expected_sha}, got {actual_sha}")
    audit = json.loads(raw)
    if audit.get("schema_version") != AUDIT_SCHEMA_VERSION:
        raise ValueError(f"archive semantics audit schema is stale; expected {AUDIT_SCHEMA_VERSION}")
    if audit.get("status") != "data_semantics_verified":
        raise ValueError("archive semantics audit did not verify the data")
    expected_protocol = canonical_mapping_sha256(_protocol_payload(config))
    if audit.get("data_protocol_sha256") != expected_protocol:
        raise ValueError("archive semantics audit was built for a different data protocol")
    records = forecast_records(Path(config["dataset_dir"]) / "preds", int(config["lead_time_hours"]))
    selected, _ = _selected_records_and_ranges(config, records)
    if not selected:
        raise ValueError("forecast archive selection became empty after the full-dataset audit")
    if (
        len(selected) != int(audit.get("record_count", -1))
        or selected[0].date.isoformat() != audit.get("first_record_date")
        or selected[-1].date.isoformat() != audit.get("last_record_date")
    ):
        raise ValueError("forecast archive record inventory drifted after the full-dataset audit")
    if _manifest_sha256(selected) != audit.get("forecast_metadata_manifest_sha256"):
        raise ValueError("forecast archive metadata manifest drifted after the full-dataset audit")
    if _forecast_content_sha256(selected, int(config["target_slice_index"])) != audit.get(
        "forecast_audited_content_sha256"
    ):
        raise ValueError("forecast audited content drifted after the full-dataset audit")
    _, static_mask_provenance = _static_mask_provenance(config)
    if static_mask_provenance != audit.get("static_mask_provenance"):
        raise ValueError("static land mask content/provenance drifted after the full-dataset audit")
    raw_land, _ = _static_mask_provenance(config)
    ocean_full = raw_land <= 0 if bool(config.get("mask_true_is_invalid", True)) else raw_land > 0
    if _sral_provenance(config, records, ocean_full) != audit.get("sral_provenance"):
        raise ValueError("SRAL geometry content/provenance drifted after the full-dataset audit")
    validate_trajectory_time_contract(config, audit)
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(args.data_config)
    audit = build_archive_semantics_audit(config)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), "status": audit["status"]}, indent=2))


if __name__ == "__main__":
    main()
