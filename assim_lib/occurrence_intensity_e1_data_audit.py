"""Real-data, CPU-only integrity and orientation audit for E1."""

from __future__ import annotations

import hashlib
import json
import math
import struct
import zlib
from collections.abc import Callable
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import load_json
from .data import build_dataset, previous_calendar_date
from .forecast import field_at_hour
from .structured_sic import make_lagged_observation_channels
from .transforms import channel_denormalize, prepare_model_field

MODE = "occurrence_intensity_e1_real_data_audit"
CONFIG_KEYS = {
    "mode", "project_name", "task_name", "clearml", "resource_kind",
    "dataset_config", "dataset_split", "case_manifest", "lags_days",
    "sral_transform_index", "observed_channel", "artifact_policy", "selected_artifacts",
}
OUTPUT_FILES = ("run_status.json", "metadata.json", "per_case_audit.json")
FROZEN_CASES = [
    {"case_id": "2022-01-02_h23", "target_date": "2022-01-02", "hour": 23, "coverage_slot": "winter_early"},
    {"case_id": "2022-01-26_h23", "target_date": "2022-01-26", "hour": 23, "coverage_slot": "winter_late"},
    {"case_id": "2022-02-25_h23", "target_date": "2022-02-25", "hour": 23, "coverage_slot": "spring_early"},
    {"case_id": "2022-03-22_h23", "target_date": "2022-03-22", "hour": 23, "coverage_slot": "spring_late"},
    {"case_id": "2022-04-21_h23", "target_date": "2022-04-21", "hour": 23, "coverage_slot": "melt_early"},
    {"case_id": "2022-05-16_h23", "target_date": "2022-05-16", "hour": 23, "coverage_slot": "melt_mid"},
    {"case_id": "2022-06-15_h23", "target_date": "2022-06-15", "hour": 23, "coverage_slot": "melt_late"},
    {"case_id": "2022-07-15_h23", "target_date": "2022-07-15", "hour": 23, "coverage_slot": "summer_early"},
]
FROZEN_SELECTION_VERSION = "e1_real_cases_v2"
FROZEN_SELECTION_RULE = (
    "eight_fixed_temporal_coverage_slots_with_complete_lag_source_metadata_only"
)
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


def _reject_output_symlinks(output: Path) -> None:
    """Reject redirected entries before the producer writes any artifact."""
    if any(path.is_symlink() for path in output.rglob("*")):
        raise ValueError("E1 data-audit output tree must not contain symbolic links")


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
    if set(manifest) != {
        "selection_version", "selection_rule", "truth_values_consulted", "cases",
        "orientation_landmarks",
    }:
        raise ValueError("E1 real-case manifest keys must be exact")
    if manifest["selection_version"] != FROZEN_SELECTION_VERSION:
        raise ValueError("unreviewed E1 case-selection version")
    if manifest["selection_rule"] != FROZEN_SELECTION_RULE:
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


def _mark_landmarks(array: np.ndarray, landmarks: list[dict[str, Any]]) -> np.ndarray:
    """Add deterministic high-contrast crosses without changing panel geometry."""
    marked = array.copy()
    height, width = marked.shape
    for landmark in landmarks:
        row, column = int(landmark["row"]), int(landmark["column"])
        if not (0 <= row < height and 0 <= column < width):
            raise ValueError("orientation landmark lies outside panel field")
        for offset in range(-3, 4):
            if 0 <= column + offset < width:
                marked[row, column + offset] = 1.0 if offset % 2 == 0 else 0.0
            if 0 <= row + offset < height:
                marked[row + offset, column] = 1.0 if offset % 2 == 0 else 0.0
    return marked


def _panel(fields: list[torch.Tensor], landmarks: list[dict[str, Any]]) -> np.ndarray:
    arrays = []
    for field in fields:
        array = field.detach().cpu().to(torch.float32).numpy()
        finite = np.isfinite(array)
        if not np.any(finite):
            array = np.zeros_like(array)
        else:
            lo, hi = np.nanmin(array), np.nanmax(array)
            array = (array - lo) / (hi - lo) if hi > lo else np.zeros_like(array)
        arrays.append(_mark_landmarks(np.nan_to_num(array, nan=0.0), landmarks))
    separator = np.ones((arrays[0].shape[0], 4), dtype=np.float32)
    interleaved = [item for pair in zip(arrays, [separator] * len(arrays)) for item in pair]
    return np.concatenate(interleaved[:-1], axis=1)


def _dataset_index(dataset: Any, case: dict[str, Any]) -> int:
    for day_idx in range(dataset._num_days()):
        target = (dataset.calendar_pairs[day_idx][1] if dataset.background_strategy == "calendar_year_ago"
                  else dataset.obs_data[day_idx + dataset.obs_shift])
        if target.date.isoformat() == case["target_date"]:
            hour_offset = int(case["hour"]) if dataset.hour_mode == "all" else 0
            return day_idx * dataset.hours_per_day + hour_offset
    raise ValueError(f"predeclared case is absent from dataset metadata: {case['case_id']}")


def _physical_sic_at_date(
    dataset: Any, source_date: date, hour: int, observed_channel: int,
) -> tuple[torch.Tensor, torch.Tensor, Path]:
    """Load physical SIC plus its raw-finite mask, preserving zero padding."""
    record = dataset.records_by_date.get(source_date)
    if record is None:
        raise ValueError(f"missing model forecast for {source_date.isoformat()}")
    source_index = dataset.indices[observed_channel]
    mean = dataset.means[observed_channel]
    std = dataset.stds[observed_channel]
    padding = dataset.padding_values[observed_channel]
    normalized, finite = prepare_model_field(
        field_at_hour(record.path, hour, [source_index]), None, [mean], [std],
        [padding], dataset.image_size,
    )
    physical = channel_denormalize(normalized, [mean], [std])[0]
    raw_finite = finite[0].bool()
    valid = dataset.base_valid_mask[observed_channel] > 0
    if not torch.all(torch.isfinite(physical[valid])):
        raise ValueError(f"non-finite padded SIC forecast for {source_date.isoformat()}")
    finite_valid = valid & raw_finite
    if not torch.all((physical[finite_valid] >= 0) & (physical[finite_valid] <= 1)):
        raise ValueError(f"SIC forecast is outside physical [0,1] for {source_date.isoformat()}")
    physical_padding = torch.as_tensor(padding, dtype=physical.dtype, device=physical.device)
    if not torch.all(physical[valid & ~raw_finite] == physical_padding):
        raise ValueError(f"raw-missing SIC did not map to physical padding for {source_date.isoformat()}")
    return physical, raw_finite, Path(record.path)


def _lag_forecast_pair(
    dataset: Any, target_date: date, hour: int, lag: int, observed_channel: int,
) -> tuple[torch.Tensor, torch.Tensor, Path, date, torch.Tensor, torch.Tensor, Path]:
    """Return physical ``y_{t-k}`` and exact previous-calendar-year ``b_{t-k}``."""
    value_date = target_date - timedelta(days=lag)
    background_date = previous_calendar_date(value_date)
    if background_date is None:
        raise ValueError(f"no previous-calendar-year date for lag={lag}")
    value, value_finite, value_path = _physical_sic_at_date(
        dataset, value_date, hour, observed_channel
    )
    background, background_finite, background_path = _physical_sic_at_date(
        dataset, background_date, hour, observed_channel
    )
    return (
        value, value_finite, value_path, background_date,
        background, background_finite, background_path,
    )


def _verify_runtime_sources(dataset: Any, target: date, sealed_source: dict[str, Any]) -> None:
    """Bind post-truth runtime reads to the corresponding pre-truth source seal."""
    lag = int(sealed_source["lag"])
    value_date = target - timedelta(days=lag)
    background_date = previous_calendar_date(value_date)
    if background_date is None:
        raise ValueError(f"no previous-calendar-year date for lag={lag}")
    for role, source_date in (("value", value_date), ("background", background_date)):
        record = dataset.records_by_date.get(source_date)
        if record is None:
            raise ValueError(f"missing sealed runtime {role} forecast source for lag={lag}")
        forecast_path = Path(record.path)
        if (
            record.date != source_date
            or sealed_source[f"{role}_date"] != source_date.isoformat()
            or forecast_path.resolve() != Path(sealed_source[f"{role}_forecast_path"]).resolve()
            or sha256_file(forecast_path) != sealed_source[f"{role}_forecast_sha256"]
        ):
            raise ValueError(
                f"runtime forecast source differs from pre-truth seal for lag={lag} role={role}"
            )

    runtime_sral_paths = [Path(path) for path in dataset.sral_records.get(value_date, [])]
    sealed_sral_paths = [Path(path) for path in sealed_source["sral_paths"]]
    if (
        len(runtime_sral_paths) != len(sealed_sral_paths)
        or any(
            actual.resolve() != sealed.resolve()
            for actual, sealed in zip(runtime_sral_paths, sealed_sral_paths)
        )
        or [sha256_file(path) for path in runtime_sral_paths] != sealed_source["sral_sha256"]
    ):
        raise ValueError(f"runtime SRAL sources differ from pre-truth seal for lag={lag}")


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
    if output.is_symlink():
        raise ValueError("E1 data-audit output directory must not be a symbolic link")
    output.mkdir(parents=True, exist_ok=True)
    if not output.is_dir():
        raise ValueError("E1 data-audit output path must be a directory")
    _reject_output_symlinks(output)
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
    data_config["observed_channels"] = [0]
    data_config["observation_mask"] = {
        "kind": "sral_tracks", "sral_transform_index": config["sral_transform_index"],
        "synthetic_probability": 0.0, "empty_probability": 0.0,
        "require_finite_model_values": True,
    }
    dataset = dataset_builder(data_config, config["dataset_split"])
    if list(dataset.observed_channels) != [config["observed_channel"]]:
        raise ValueError("E1 data audit must observe SIC channel 0 only")

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
            value_date = target - timedelta(days=lag)
            background_date = previous_calendar_date(value_date)
            if background_date is None:
                raise ValueError(f"no previous-calendar-year date for {case['case_id']} lag={lag}")
            value_record = dataset.records_by_date.get(value_date)
            background_record = dataset.records_by_date.get(background_date)
            if value_record is None or background_record is None:
                raise ValueError(f"missing pre-truth forecast metadata for {case['case_id']} lag={lag}")
            if value_record.date != value_date or background_record.date != background_date:
                raise ValueError(
                    f"forecast record date differs from requested lag for {case['case_id']} lag={lag}"
                )
            sral_paths = dataset.sral_records.get(value_date, [])
            if not sral_paths:
                raise ValueError(
                    f"missing pre-truth SRAL source metadata for {case['case_id']} lag={lag}"
                )
            sources.append({
                "lag": lag,
                "value_date": value_record.date.isoformat(),
                "value_forecast_path": str(value_record.path),
                "value_forecast_sha256": sha256_file(value_record.path),
                "background_date": background_record.date.isoformat(),
                "background_forecast_path": str(background_record.path),
                "background_forecast_sha256": sha256_file(background_record.path),
                "sral_paths": [str(path) for path in sral_paths],
                "sral_sha256": [sha256_file(path) for path in sral_paths],
            })
        source_inventory.append({"case_id": case["case_id"], "sources": sources})
    inventory_bytes = json.dumps(source_inventory, sort_keys=True, separators=(",", ":")).encode()

    panels_dir = output / "panels"
    if panels_dir.is_symlink():
        raise ValueError("E1 data-audit panels directory must not be a symbolic link")
    panels_dir.mkdir(exist_ok=True)
    if not panels_dir.is_dir():
        raise ValueError("E1 data-audit panels path must be a directory")
    per_case = []
    for (case, index), sealed_case in zip(resolved, source_inventory):
        item = dataset[index]
        if item["meta"]["case_id"] != case["case_id"]:
            raise ValueError("dataset case identity changed after metadata seal")
        sealed_truth_path = Path(sealed_case["sources"][0]["value_forecast_path"])
        actual_truth_path = Path(item["meta"].get("target_path", ""))
        if not actual_truth_path.is_file() or actual_truth_path.resolve() != sealed_truth_path.resolve():
            raise ValueError("dataset truth source differs from the pre-truth source seal")
        target = date.fromisoformat(case["target_date"])
        truth_normalized = item["truth"][config["observed_channel"]]
        truth = channel_denormalize(
            truth_normalized.unsqueeze(0),
            [dataset.means[config["observed_channel"]]],
            [dataset.stds[config["observed_channel"]]],
        )[0]
        valid = item["valid_mask"][config["observed_channel"]].bool()
        sealed_valid = dataset.base_valid_mask[config["observed_channel"]].bool()
        if valid.shape != sealed_valid.shape or not torch.equal(valid.cpu(), sealed_valid.cpu()):
            raise ValueError("dataset valid domain differs from the pre-truth domain seal")
        if truth.shape != valid.shape:
            raise ValueError("truth and sealed valid domain are not co-registered")
        if not torch.all(torch.isfinite(truth[valid])):
            raise ValueError("truth is non-finite inside the valid domain")
        if not torch.all((truth[valid] >= 0) & (truth[valid] <= 1)):
            raise ValueError("truth SIC is outside physical [0,1]")
        lag_masks, backgrounds, background_sources, lag_values_with_nan = [], [], [], []
        lag_rows = []
        for lag, sealed_source in zip(config["lags_days"], sealed_case["sources"]):
            _verify_runtime_sources(dataset, target, sealed_source)
            mask = dataset.sral_spatial_mask(
                target, day_offsets=[lag], transform_index=config["sral_transform_index"]
            )
            if mask is None:
                raise ValueError(
                    f"missing real SRAL footprint for {case['case_id']} lag={lag}"
                )
            sral_mask = mask.to(torch.float32)
            if (
                not torch.any(sral_mask > 0)
                or not torch.all((sral_mask == 0) | (sral_mask == 1))
                or not torch.all(sral_mask[~valid] == 0)
            ):
                raise ValueError("SRAL footprint violates exact binary/valid-domain semantics")
            (
                lag_values, value_raw_finite, value_path, background_date,
                background, background_raw_finite, background_path,
            ) = _lag_forecast_pair(
                dataset, target, int(case["hour"]), lag, config["observed_channel"]
            )
            # Match require_finite_model_values: SRAL supplies geometry, while
            # only raw-finite target-year M2M values become observations.
            mask = sral_mask * value_raw_finite.to(sral_mask.dtype)
            if not torch.any(mask > 0):
                raise ValueError(
                    f"real SRAL footprint has no raw-finite target-year SIC for "
                    f"{case['case_id']} lag={lag}"
                )
            observed = mask.bool()
            # The E1 observation value at each lag is the co-registered forecast
            # value on that lag's exact real footprint. Missing pixels remain NaN
            # until the NaN-safe channel constructor applies torch.where.
            finite_on = bool(torch.all(torch.isfinite(lag_values[observed])))
            nan_outside = torch.full_like(lag_values, math.nan)
            nan_outside[observed] = lag_values[observed]
            lag_rows.append({
                "lag_days": lag,
                "value_source_date": (target - timedelta(days=lag)).isoformat(),
                "background_source_date": background_date.isoformat(),
                "sral_footprint_pixels": int(sral_mask.sum()),
                "footprint_pixels": int(mask.sum()), "finite_on_observed": finite_on,
                "target_raw_nonfinite_sral_pixels": int(
                    (sral_mask.bool() & ~value_raw_finite).sum().item()
                ),
                "background_raw_finite_observed_pixels": int(
                    background_raw_finite[observed].sum().item()
                ),
                "background_padding_observed_pixels": int(
                    (~background_raw_finite[observed]).sum().item()
                ),
                "background_raw_finite_required_for_innovation": False,
                "nan_outside_observed": False, "geometry_provenance": "real_sral_footprint",
                "value_provenance": "target_year_m2m_forecast_at_real_sral_footprint",
                "occurrence_encoder_value_space": "physical_0_1",
                "innovation_nonzero_pixels": int(
                    torch.count_nonzero((lag_values - background)[observed]).item()
                ),
                "future_date_leakage": bool(target - timedelta(days=lag) > target),
            })
            lag_masks.append(mask)
            backgrounds.append(background)
            lag_values_with_nan.append(nan_outside)
            background_sources.append(str(background_path))
            if Path(value_path).resolve() == Path(background_path).resolve():
                raise ValueError("E1 value and background forecasts must be distinct sources")
        masks_tensor = torch.stack(lag_masks).unsqueeze(0)
        backgrounds_tensor = torch.stack(backgrounds).unsqueeze(0)
        values_tensor = torch.stack(lag_values_with_nan).unsqueeze(0)
        channels = make_lagged_observation_channels(
            backgrounds_tensor,
            values_tensor,
            masks_tensor,
            torch.tensor([[0, 1, 2]], dtype=backgrounds_tensor.dtype),
            torch.zeros((1, 3), dtype=backgrounds_tensor.dtype),
            torch.zeros_like(values_tensor),
        ).reshape(1, 3, 8, *truth.shape)
        if not torch.all(torch.isfinite(channels)):
            raise ValueError("E1 observation channels leak non-finite values")
        for lag_index, observed in enumerate(masks_tensor[0].bool()):
            emitted_values = channels[0, lag_index, 1]
            if (
                not torch.equal(emitted_values[observed], values_tensor[0, lag_index][observed])
                or not torch.all(emitted_values[~observed] == 0)
            ):
                raise ValueError("E1 observation channels violate NaN-safe masking")
            emitted_innovation = channels[0, lag_index, 0]
            expected_innovation = values_tensor[0, lag_index] - backgrounds_tensor[0, lag_index]
            if (
                not torch.equal(emitted_innovation[observed], expected_innovation[observed])
                or not torch.all(emitted_innovation[~observed] == 0)
            ):
                raise ValueError("E1 innovation channels violate the lagged forecast protocol")
            lag_rows[lag_index]["nan_outside_observed"] = True
        if sum(row["innovation_nonzero_pixels"] for row in lag_rows) == 0:
            raise ValueError("E1 lagged forecast innovations are identically zero for the case")
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
        _png(panel_path, _panel(panel_fields, manifest["orientation_landmarks"]))
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
        "case_selection_version": FROZEN_SELECTION_VERSION,
        "observed_channels": [0], "occurrence_encoder_value_space": "physical_0_1",
        "background_missing_sic_policy": "physical_zero_padding_allowed_and_counted",
        "lags_days": [0, 1, 2],
        "sral_footprint_semantics": (
            "finite_after_transform_11_valid_domain_and_target_raw_finite_sic"
        ),
        "panel_layout": [
            "truth", "background_lag0", "mask_lag0", "background_lag1", "mask_lag1",
            "background_lag2", "mask_lag2",
        ],
        "panel_landmark_overlay": {
            "applied_to_each_tile": True,
            "cross_radius_pixels": 3,
            "alternating_values": [1.0, 0.0],
            "landmarks": manifest["orientation_landmarks"],
        },
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
