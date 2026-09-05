"""Real-data, CPU-only integrity and orientation audit for E1."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import date, timedelta
from pathlib import Path
import struct
from typing import Any, Callable
import zlib

import numpy as np
import torch

from .config import load_json
from .data import build_dataset
from .forecast import field_at_hour
from .transforms import prepare_model_field

MODE = "occurrence_intensity_e1_real_data_audit"
CONFIG_KEYS = {
    "mode", "project_name", "task_name", "clearml", "resource_kind",
    "dataset_config", "dataset_split", "case_manifest", "lags_days",
    "sral_transform_index", "observed_channel", "artifact_policy", "selected_artifacts",
}
OUTPUT_FILES = ("run_status.json", "metadata.json", "per_case_audit.json")
FROZEN_CASES = [
    {"case_id": "2022-01-01_h23", "target_date": "2022-01-01", "hour": 23, "coverage_slot": "winter_early"},
    {"case_id": "2022-01-26_h23", "target_date": "2022-01-26", "hour": 23, "coverage_slot": "winter_late"},
    {"case_id": "2022-02-25_h23", "target_date": "2022-02-25", "hour": 23, "coverage_slot": "spring_early"},
    {"case_id": "2022-03-22_h23", "target_date": "2022-03-22", "hour": 23, "coverage_slot": "spring_late"},
    {"case_id": "2022-04-21_h23", "target_date": "2022-04-21", "hour": 23, "coverage_slot": "melt_early"},
    {"case_id": "2022-05-16_h23", "target_date": "2022-05-16", "hour": 23, "coverage_slot": "melt_mid"},
    {"case_id": "2022-06-15_h23", "target_date": "2022-06-15", "hour": 23, "coverage_slot": "melt_late"},
    {"case_id": "2022-07-15_h23", "target_date": "2022-07-15", "hour": 23, "coverage_slot": "summer_early"},
]
FROZEN_LANDMARKS = [
    {"name": "western_arctic_grid_landmark", "row": 32, "column": 32},
    {"name": "greenland_sector_grid_landmark", "row": 287, "column": 223},
]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Any) -> None:
    """Publish one compact JSON artifact without exposing a partial document."""
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def validate_audit_config(config: dict[str, Any]) -> None:
    if set(config) != CONFIG_KEYS:
        raise ValueError("E1 data-audit config keys must be exact")
    expected = {
        "mode": MODE,
        "project_name": "generative-sea-ice-da",
        "task_name": "occurrence-intensity-e1-real-data-audit",
        "clearml": {"enabled": True},
        "resource_kind": "server_cpu",
        "dataset_config": "config/data/m2m_2f_1y.json",
        "dataset_split": "valid",
        "case_manifest": "paper/OCCURRENCE_INTENSITY_E1_REAL_CASES.json",
        "lags_days": [0, 1, 2],
        "sral_transform_index": 11,
        "observed_channel": 0,
        "artifact_policy": "selected_artifacts",
        "selected_artifacts": [
            "run_status.json", "metadata.json", "per_case_audit.json", "panels/*.png"
        ],
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"invalid frozen E1 data-audit field: {key}")


def validate_case_manifest(manifest: dict[str, Any]) -> None:
    if set(manifest) != {"selection_rule", "truth_values_consulted", "cases", "orientation_landmarks"}:
        raise ValueError("E1 real-case manifest keys must be exact")
    if manifest["selection_rule"] != "eight_fixed_temporal_coverage_slots_from_forecast_and_sral_metadata_only":
        raise ValueError("unreviewed E1 case-selection rule")
    if manifest["truth_values_consulted"] is not False:
        raise ValueError("E1 cases must be frozen without consulting truth values")
    cases = manifest["cases"]
    if not isinstance(cases, list) or len(cases) != 8:
        raise ValueError("E1 real-case manifest must contain exactly eight cases")
    if cases != FROZEN_CASES:
        raise ValueError("E1 real-case identities or coverage slots differ from the frozen selection")
    if len({case.get("case_id") for case in cases}) != 8:
        raise ValueError("E1 case identifiers must be unique")
    for case in cases:
        if set(case) != {"case_id", "target_date", "hour", "coverage_slot"}:
            raise ValueError("E1 case keys must be exact")
        parsed = date.fromisoformat(case["target_date"])
        if case["case_id"] != f"{parsed.isoformat()}_h{int(case['hour']):02d}":
            raise ValueError("case_id must bind target_date and hour")
        if int(case["hour"]) != 23 or not case["coverage_slot"]:
            raise ValueError("invalid case hour or metadata coverage slot")
    landmarks = manifest["orientation_landmarks"]
    if not isinstance(landmarks, list) or len(landmarks) != 2:
        raise ValueError("exactly two orientation landmarks are required")
    if landmarks != FROZEN_LANDMARKS:
        raise ValueError("E1 orientation landmarks differ from the frozen geographic anchors")
    for landmark in landmarks:
        if set(landmark) != {"name", "row", "column"}:
            raise ValueError("orientation landmark keys must be exact")
        if not landmark["name"] or min(int(landmark["row"]), int(landmark["column"])) < 0:
            raise ValueError("invalid orientation landmark")


def _png(path: Path, image: np.ndarray) -> None:
    if image.ndim != 2 or not np.all(np.isfinite(image)):
        raise ValueError("panel must be a finite two-dimensional image")
    pixels = np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)
    height, width = pixels.shape
    scanlines = b"".join(b"\x00" + pixels[row].tobytes() for row in range(height))
    def chunk(kind: bytes, payload: bytes) -> bytes:
        body = kind + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(scanlines)) + chunk(b"IEND", b"")
    )


def _panel(fields: list[torch.Tensor]) -> np.ndarray:
    arrays = []
    for field in fields:
        array = field.detach().cpu().to(torch.float32).numpy()
        finite = np.isfinite(array)
        if not np.any(finite):
            array = np.zeros_like(array)
        else:
            lo, hi = np.nanmin(array), np.nanmax(array)
            array = (array - lo) / (hi - lo) if hi > lo else np.zeros_like(array)
        arrays.append(np.nan_to_num(array, nan=0.0))
    separator = np.ones((arrays[0].shape[0], 4), dtype=np.float32)
    return np.concatenate([item for pair in zip(arrays, [separator] * len(arrays)) for item in pair][:-1], axis=1)


def _dataset_index(dataset: Any, case: dict[str, Any]) -> int:
    for day_idx in range(dataset._num_days()):
        target = (dataset.calendar_pairs[day_idx][1] if dataset.background_strategy == "calendar_year_ago"
                  else dataset.obs_data[day_idx + dataset.obs_shift])
        if target.date.isoformat() == case["target_date"]:
            hour_offset = int(case["hour"]) if dataset.hour_mode == "all" else 0
            return day_idx * dataset.hours_per_day + hour_offset
    raise ValueError(f"predeclared case is absent from dataset metadata: {case['case_id']}")


def _lag_background(dataset: Any, target_date: date, hour: int, lag: int) -> tuple[torch.Tensor, Path]:
    record = dataset.records_by_date.get(target_date - timedelta(days=lag))
    if record is None:
        raise ValueError(f"missing lag-specific background for lag={lag}")
    field, finite = prepare_model_field(
        field_at_hour(record.path, hour, dataset.indices), None, dataset.means, dataset.stds,
        dataset.padding_values, dataset.image_size,
    )
    if not torch.all(torch.isfinite(field)) or not torch.all(finite[0][dataset.base_valid_mask[0] > 0]):
        raise ValueError(f"non-finite lag-specific background for lag={lag}")
    return field[0], Path(record.path)


def run_real_data_audit(
    config_path: str | Path, output_dir: str | Path, *,
    dataset_builder: Callable[[dict[str, Any], str], Any] = build_dataset,
) -> dict[str, Any]:
    config_path, output = Path(config_path), Path(output_dir)
    config = load_json(config_path)
    validate_audit_config(config)
    manifest_path = Path(config["case_manifest"])
    manifest = load_json(manifest_path)
    validate_case_manifest(manifest)
    output.mkdir(parents=True, exist_ok=True)
    # Invalidate a completion marker from an earlier attempt before dataset
    # construction, source hashing, or truth access can fail.  Reusing an
    # output directory must never expose stale success for the current attempt.
    _write_json_atomic(output / "run_status.json", {
        "status": "running", "mode": MODE, "resource_kind": "server_cpu", "cases": 8,
        "engineering_only": True, "calibration_claim": False,
    })
    data_config = load_json(config["dataset_config"])
    # Primary audit is deterministic and real-only; no truth-derived synthetic masks are permitted.
    data_config = dict(data_config)
    data_config["hour_mode"] = "all"
    data_config["assimilation_range"] = 3
    data_config["observation_mask"] = {
        "kind": "sral_tracks", "sral_transform_index": config["sral_transform_index"],
        "synthetic_probability": 0.0, "empty_probability": 0.0,
        "require_finite_model_values": True,
    }
    dataset = dataset_builder(data_config, config["dataset_split"])

    valid_domain = (
        dataset.base_valid_mask[config["observed_channel"]]
        .detach().cpu().to(torch.uint8).contiguous().numpy()
    )
    valid_domain_sha256 = hashlib.sha256(valid_domain.tobytes(order="C")).hexdigest()

    # Resolve every identity and seal source metadata before __getitem__ may read truth.
    resolved = [(case, _dataset_index(dataset, case)) for case in manifest["cases"]]
    source_inventory = []
    for case, _ in resolved:
        target = date.fromisoformat(case["target_date"])
        sources = []
        for lag in config["lags_days"]:
            source_date = target - timedelta(days=lag)
            record = dataset.records_by_date.get(source_date)
            if record is None:
                raise ValueError(f"missing pre-truth source metadata for {case['case_id']} lag={lag}")
            if record.date != source_date:
                raise ValueError(
                    f"forecast record date differs from requested lag for {case['case_id']} lag={lag}"
                )
            sral_paths = dataset.sral_records.get(source_date, [])
            if not sral_paths:
                raise ValueError(
                    f"missing pre-truth SRAL source metadata for {case['case_id']} lag={lag}"
                )
            sources.append({
                "lag": lag, "date": record.date.isoformat(),
                "forecast_path": str(record.path),
                "forecast_sha256": sha256_file(record.path),
                "sral_paths": [str(path) for path in sral_paths],
                "sral_sha256": [sha256_file(path) for path in sral_paths],
            })
        source_inventory.append({"case_id": case["case_id"], "sources": sources})
    inventory_bytes = json.dumps(source_inventory, sort_keys=True, separators=(",", ":")).encode()

    panels_dir = output / "panels"
    panels_dir.mkdir(exist_ok=True)
    per_case = []
    for (case, index), sealed_case in zip(resolved, source_inventory):
        item = dataset[index]
        if item["meta"]["case_id"] != case["case_id"]:
            raise ValueError("dataset case identity changed after metadata seal")
        sealed_truth_path = Path(sealed_case["sources"][0]["forecast_path"])
        actual_truth_path = Path(item["meta"].get("target_path", ""))
        if not actual_truth_path.is_file() or actual_truth_path.resolve() != sealed_truth_path.resolve():
            raise ValueError("dataset truth source differs from the pre-truth source seal")
        target = date.fromisoformat(case["target_date"])
        truth = item["truth"][config["observed_channel"]]
        valid = item["valid_mask"][config["observed_channel"]].bool()
        if not torch.all(torch.isfinite(truth[valid])):
            raise ValueError("truth is non-finite inside the valid domain")
        lag_masks, backgrounds, background_sources = [], [], []
        lag_rows = []
        for lag in config["lags_days"]:
            mask = dataset.sral_spatial_mask(
                target, day_offsets=[lag], transform_index=config["sral_transform_index"]
            )
            if mask is None:
                raise ValueError(
                    f"missing real SRAL footprint for {case['case_id']} lag={lag}"
                )
            mask = mask.to(torch.float32)
            if (
                not torch.any(mask > 0)
                or not torch.all((mask == 0) | (mask == 1))
                or not torch.all(mask[~valid] == 0)
            ):
                raise ValueError("SRAL footprint violates exact binary/valid-domain semantics")
            background, background_path = _lag_background(dataset, target, int(case["hour"]), lag)
            observed = mask.bool()
            # The E1 observation value at each lag is the co-registered forecast
            # value on that lag's exact real footprint. Missing pixels remain NaN
            # until the NaN-safe channel constructor applies torch.where.
            lag_values = background
            finite_on = bool(torch.all(torch.isfinite(lag_values[observed])))
            nan_outside = torch.full_like(lag_values, math.nan)
            nan_outside[observed] = lag_values[observed]
            nan_safe = bool(torch.all(torch.isnan(nan_outside[~observed])))
            lag_rows.append({
                "lag_days": lag, "source_date": (target - timedelta(days=lag)).isoformat(),
                "footprint_pixels": int(mask.sum()), "finite_on_observed": finite_on,
                "nan_outside_observed": nan_safe, "geometry_provenance": "real_sral_footprint",
                "value_provenance": "forecast_value_at_real_sral_footprint",
                "future_date_leakage": bool(target - timedelta(days=lag) > target),
            })
            lag_masks.append(mask)
            backgrounds.append(background)
            background_sources.append(str(background_path))
        landmark_values = []
        for landmark in manifest["orientation_landmarks"]:
            row, column = int(landmark["row"]), int(landmark["column"])
            if row >= truth.shape[0] or column >= truth.shape[1]:
                raise ValueError("orientation landmark lies outside the real grid")
            landmark_values.append({
                "name": landmark["name"], "row": row, "column": column,
                "valid_domain": bool(valid[row, column]),
                "lag_mask_values": [int(mask[row, column]) for mask in lag_masks],
            })
        panel_name = f"{case['case_id']}_truth_background_lag_masks.png"
        panel_fields = [truth]
        for background, mask in zip(backgrounds, lag_masks):
            panel_fields.extend((background, mask))
        panel_path = panels_dir / panel_name
        _png(panel_path, _panel(panel_fields))
        per_case.append({
            "case_id": case["case_id"], "coverage_slot": case["coverage_slot"],
            "target_date": case["target_date"], "hour": case["hour"],
            "truth_finite_on_valid_domain": True,
            "lag_specific_backgrounds": len(set(background_sources)) == 3,
            "co_registered_shape": list(truth.shape), "orientation_landmarks": landmark_values,
            "lags": lag_rows, "panel": f"panels/{panel_name}",
            "panel_sha256": sha256_file(panel_path),
        })
    status = {
        "status": "completed", "mode": MODE, "resource_kind": "server_cpu", "cases": 8,
        "engineering_only": True, "calibration_claim": False,
    }
    metadata = {
        "config_sha256": sha256_file(config_path), "case_manifest_sha256": sha256_file(manifest_path),
        "dataset_config_sha256": sha256_file(config["dataset_config"]),
        "valid_domain_sha256": valid_domain_sha256,
        "source_inventory_sha256": hashlib.sha256(inventory_bytes).hexdigest(),
        "source_inventory": source_inventory,
        "dataset_provenance": dataset.provenance(), "truth_values_consulted_for_selection": False,
        "lags_days": [0, 1, 2], "sral_footprint_semantics": "finite_after_transform_11_and_valid_domain",
        "panel_layout": [
            "truth", "background_lag0", "mask_lag0", "background_lag1", "mask_lag1",
            "background_lag2", "mask_lag2",
        ],
    }
    # Evidence must be durable before the completion marker becomes visible.
    # Each replace is atomic within the output directory, so readers never see
    # a partially serialized compact JSON document.
    for name, payload in (
        ("metadata.json", metadata),
        ("per_case_audit.json", per_case),
        ("run_status.json", status),
    ):
        _write_json_atomic(output / name, payload)
    return {"run_status": status, "metadata": metadata, "per_case_audit": per_case}
