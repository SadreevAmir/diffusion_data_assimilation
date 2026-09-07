#!/usr/bin/env python3
"""Fail-closed validator for the compact E1 real-data audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import zlib
from datetime import date, timedelta
from pathlib import Path
from typing import Any

MODE = "occurrence_intensity_e1_real_data_audit"
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
FROZEN_LANDMARKS = [
    {"name": "western_arctic_grid_landmark", "row": 32, "column": 32},
    {"name": "greenland_sector_grid_landmark", "row": 287, "column": 223},
]
FROZEN_CONFIG = {
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
        "run_status.json", "metadata.json", "per_case_audit.json", "panels/*.png",
    ],
}

FROZEN_SELECTION_RULE = (
    "eight_fixed_temporal_coverage_slots_with_complete_lag_source_metadata_only"
)


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_manifest_contract(manifest: Any) -> None:
    """Independently enforce pre-truth case selection in compact evidence."""
    if not isinstance(manifest, dict) or set(manifest) != {
        "selection_version", "selection_rule", "truth_values_consulted", "cases",
        "orientation_landmarks",
    }:
        raise ValueError("case manifest keys must match the frozen selection contract")
    if manifest["selection_version"] != FROZEN_SELECTION_VERSION:
        raise ValueError("case manifest uses an unreviewed selection version")
    if manifest["selection_rule"] != FROZEN_SELECTION_RULE:
        raise ValueError("case manifest uses an unreviewed selection rule")
    if manifest["truth_values_consulted"] is not False:
        raise ValueError("case manifest does not prove pre-truth selection")
    if manifest["cases"] != FROZEN_CASES:
        raise ValueError("manifest cases differ from the frozen eight-case selection")
    if manifest["orientation_landmarks"] != FROZEN_LANDMARKS:
        raise ValueError("manifest landmarks differ from the frozen geographic anchors")


def _png_dimensions(path: Path) -> tuple[int, int]:
    header = path.read_bytes()[:24]
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError("missing or invalid compact orientation panel")
    return struct.unpack(">II", header[16:24])


def _png_grayscale_pixels(path: Path) -> tuple[int, int, bytes]:
    """Decode the runner's frozen 8-bit grayscale/filter-0 PNG contract."""
    payload = path.read_bytes()
    if payload[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("missing or invalid compact orientation panel")
    offset, width, height, compressed = 8, None, None, bytearray()
    chunk_kinds = []
    seen_ihdr = False
    seen_iend = False
    while offset < len(payload):
        if offset + 12 > len(payload):
            raise ValueError("truncated compact orientation panel")
        length = struct.unpack(">I", payload[offset:offset + 4])[0]
        kind = payload[offset + 4:offset + 8]
        chunk_kinds.append(kind)
        end = offset + 12 + length
        if end > len(payload):
            raise ValueError("truncated compact orientation panel chunk")
        chunk = payload[offset + 8:offset + 8 + length]
        expected_crc = struct.unpack(">I", payload[offset + 8 + length:end])[0]
        if zlib.crc32(kind + chunk) != expected_crc:
            raise ValueError("orientation panel contains a PNG chunk with invalid CRC")
        if kind == b"IHDR":
            if seen_ihdr or offset != 8:
                raise ValueError("orientation panel has invalid IHDR placement")
            if len(chunk) != 13 or chunk[8:] != bytes((8, 0, 0, 0, 0)):
                raise ValueError("orientation panel must use frozen 8-bit grayscale encoding")
            width, height = struct.unpack(">II", chunk[:8])
            seen_ihdr = True
        elif kind == b"IDAT":
            if not seen_ihdr or seen_iend:
                raise ValueError("orientation panel has invalid IDAT placement")
            compressed.extend(chunk)
        elif kind == b"IEND":
            if length != 0 or seen_iend:
                raise ValueError("orientation panel has invalid IEND chunk")
            seen_iend = True
            offset = end
            break
        offset = end
    if not seen_iend or offset != len(payload):
        raise ValueError("orientation panel has missing or trailing PNG content")
    if chunk_kinds != [b"IHDR", b"IDAT", b"IEND"]:
        raise ValueError("orientation panel differs from the frozen PNG chunk sequence")
    if width is None or height is None or not compressed:
        raise ValueError("orientation panel is missing required PNG chunks")
    try:
        decoder = zlib.decompressobj()
        scanlines = decoder.decompress(bytes(compressed)) + decoder.flush()
    except zlib.error as error:
        raise ValueError("orientation panel has invalid compressed pixels") from error
    if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
        raise ValueError("orientation panel has an incomplete or concatenated zlib stream")
    stride = width + 1
    if len(scanlines) != height * stride:
        raise ValueError("orientation panel pixel payload has invalid size")
    rows = []
    for row in range(height):
        scanline = scanlines[row * stride:(row + 1) * stride]
        if scanline[0] != 0:
            raise ValueError("orientation panel uses an unfrozen PNG row filter")
        rows.append(scanline[1:])
    return width, height, b"".join(rows)


def _validate_landmark_overlay(path: Path) -> None:
    width, height, pixels = _png_grayscale_pixels(path)
    tile_width, separator_width, tile_count = 256, 4, 7
    if (width, height) != (tile_count * tile_width + (tile_count - 1) * separator_width, 320):
        raise ValueError("compact orientation panel has invalid geometry")
    for tile in range(tile_count):
        tile_start = tile * (tile_width + separator_width)
        for landmark in FROZEN_LANDMARKS:
            row, column = landmark["row"], landmark["column"]
            for delta in range(-3, 4):
                expected = 255 if delta % 2 == 0 else 0
                horizontal = row * width + tile_start + column + delta
                vertical = (row + delta) * width + tile_start + column
                if pixels[horizontal] != expected or pixels[vertical] != expected:
                    raise ValueError("orientation landmark overlay pixels differ from the frozen contract")


def _validate_mask_footprints(path: Path, audit_case: dict[str, Any]) -> None:
    """Bind reported footprint sizes to the three rendered binary mask tiles."""
    width, height, pixels = _png_grayscale_pixels(path)
    tile_width, separator_width = 256, 4
    overlay_pixels = set()
    for landmark in FROZEN_LANDMARKS:
        row, column = landmark["row"], landmark["column"]
        for delta in range(-3, 4):
            overlay_pixels.add((row, column + delta))
            overlay_pixels.add((row + delta, column))
    for lag_index, lag in enumerate(audit_case["lags"]):
        tile = 2 + 2 * lag_index
        tile_start = tile * (tile_width + separator_width)
        visible_ones = 0
        for row in range(height):
            for column in range(tile_width):
                if (row, column) in overlay_pixels:
                    continue
                value = pixels[row * width + tile_start + column]
                if value not in (0, 255):
                    raise ValueError("rendered lag mask contains non-binary pixels")
                visible_ones += value == 255
        footprint = lag["footprint_pixels"]
        if not visible_ones <= footprint <= visible_ones + len(overlay_pixels):
            raise ValueError("reported lag footprint differs from rendered mask pixels")


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_compact_audit(config_path: Path, output: Path) -> str:
    config = _load(config_path)
    if config != FROZEN_CONFIG:
        raise ValueError("config differs from the complete frozen audit contract")
    expected_files = {
        "run_status.json", "metadata.json", "per_case_audit.json",
        *{
            f"panels/{case['case_id']}_truth_background_lag_masks.png"
            for case in FROZEN_CASES
        },
    }
    actual_files = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if actual_files != expected_files:
        raise ValueError("compact artifact inventory must match the frozen selection exactly")
    if any(path.is_symlink() for path in output.rglob("*")):
        raise ValueError("compact artifacts must be regular files, not symbolic links")
    status = _load(output / "run_status.json")
    metadata = _load(output / "metadata.json")
    cases = _load(output / "per_case_audit.json")
    if status != {
        "status": "completed", "mode": MODE, "resource_kind": "server_cpu", "cases": 8,
        "engineering_only": True, "calibration_claim": False,
    }:
        raise ValueError("run_status does not describe the frozen engineering-only audit")
    expected_metadata = {
        "config_sha256", "case_manifest_sha256", "dataset_config_sha256",
        "valid_domain_sha256", "source_inventory_sha256",
        "source_inventory", "dataset_provenance", "truth_values_consulted_for_selection", "lags_days",
        "sral_footprint_semantics", "panel_layout", "panel_landmark_overlay",
        "observed_channels", "occurrence_encoder_value_space",
        "case_selection_version", "background_missing_sic_policy",
    }
    if set(metadata) != expected_metadata:
        raise ValueError("metadata keys must be exact")
    if metadata["config_sha256"] != hashlib.sha256(config_path.read_bytes()).hexdigest():
        raise ValueError("metadata does not bind the frozen config")
    manifest_path = Path(config["case_manifest"])
    if metadata["case_manifest_sha256"] != hashlib.sha256(manifest_path.read_bytes()).hexdigest():
        raise ValueError("metadata does not bind the frozen case manifest")
    dataset_config_path = Path(config["dataset_config"])
    if metadata["dataset_config_sha256"] != hashlib.sha256(dataset_config_path.read_bytes()).hexdigest():
        raise ValueError("metadata does not bind the frozen dataset config")
    if not _is_sha256(metadata["valid_domain_sha256"]):
        raise ValueError("metadata does not bind a canonical valid-domain mask")
    if metadata["truth_values_consulted_for_selection"] is not False:
        raise ValueError("truth-dependent case selection is forbidden")
    if metadata["case_selection_version"] != FROZEN_SELECTION_VERSION:
        raise ValueError("metadata does not identify the reviewed case selection")
    if metadata["observed_channels"] != [0]:
        raise ValueError("E1 data audit must observe SIC channel 0 only")
    if metadata["occurrence_encoder_value_space"] != "physical_0_1":
        raise ValueError("E1 occurrence encoder did not receive physical SIC")
    if metadata["background_missing_sic_policy"] != "physical_zero_padding_allowed_and_counted":
        raise ValueError("E1 background missing-SIC policy drift")
    if metadata["sral_footprint_semantics"] != (
        "finite_after_transform_11_valid_domain_and_target_raw_finite_sic"
    ):
        raise ValueError("E1 target raw-finite footprint policy drift")
    if metadata["lags_days"] != [0, 1, 2] or metadata["panel_layout"] != [
        "truth", "background_lag0", "mask_lag0", "background_lag1", "mask_lag1",
        "background_lag2", "mask_lag2",
    ]:
        raise ValueError("lag or panel layout drift")
    if metadata["panel_landmark_overlay"] != {
        "applied_to_each_tile": True,
        "cross_radius_pixels": 3,
        "alternating_values": [1.0, 0.0],
        "landmarks": FROZEN_LANDMARKS,
    }:
        raise ValueError("orientation landmark overlay differs from the frozen contract")
    digest = metadata["source_inventory_sha256"]
    if not _is_sha256(digest):
        raise ValueError("invalid source inventory digest")
    inventory_bytes = json.dumps(
        metadata["source_inventory"], sort_keys=True, separators=(",", ":")
    ).encode()
    if hashlib.sha256(inventory_bytes).hexdigest() != digest:
        raise ValueError("source inventory digest does not bind the compact inventory")
    inventory = metadata["source_inventory"]
    if not isinstance(inventory, list) or len(inventory) != 8:
        raise ValueError("source inventory must cover all eight cases")
    if not isinstance(cases, list) or len(cases) != 8:
        raise ValueError("per-case audit must contain exactly eight cases")
    manifest = _load(manifest_path)
    _validate_manifest_contract(manifest)
    manifest_cases = manifest["cases"]
    manifest_landmarks = manifest["orientation_landmarks"]
    expected_ids = [case["case_id"] for case in manifest_cases]
    expected_panels = {
        f"panels/{case_id}_truth_background_lag_masks.png" for case_id in expected_ids
    }
    actual_panels = {
        path.relative_to(output).as_posix() for path in (output / "panels").glob("*.png")
    }
    if actual_panels != expected_panels:
        raise ValueError("compact orientation panel inventory must match the frozen cases exactly")
    if [case.get("case_id") for case in cases] != expected_ids:
        raise ValueError("audited identities differ from the frozen manifest")
    if [entry.get("case_id") for entry in inventory] != expected_ids:
        raise ValueError("source inventory identities differ from the frozen manifest")
    for inventory_entry, audit_case in zip(inventory, cases):
        if set(inventory_entry) != {"case_id", "sources"}:
            raise ValueError("source inventory case keys must be exact")
        sources = inventory_entry["sources"]
        if not isinstance(sources, list) or len(sources) != 3:
            raise ValueError("source inventory must contain exactly three lag sources per case")
        audit_lags = audit_case.get("lags")
        if not isinstance(audit_lags, list) or len(audit_lags) != 3:
            raise ValueError("per-case audit must contain exactly three lag rows")
        target_date = date.fromisoformat(audit_case.get("target_date", ""))
        for source, audit_lag, expected_lag in zip(sources, audit_lags, (0, 1, 2)):
            if set(source) != {
                "lag", "value_date", "value_forecast_path", "value_forecast_sha256",
                "background_date", "background_forecast_path", "background_forecast_sha256",
                "sral_paths", "sral_sha256",
            }:
                raise ValueError("source inventory lag keys must be exact")
            expected_value_date = target_date - timedelta(days=expected_lag)
            try:
                expected_background_date = expected_value_date.replace(
                    year=expected_value_date.year - 1
                )
            except ValueError as error:
                raise ValueError("lag has no exact previous-calendar-year background date") from error
            if (
                not isinstance(audit_lag, dict)
                or source["lag"] != expected_lag
                or source["value_date"] != expected_value_date.isoformat()
                or source["background_date"] != expected_background_date.isoformat()
                or source["value_date"] != audit_lag.get("value_source_date")
                or source["background_date"] != audit_lag.get("background_source_date")
            ):
                raise ValueError("source inventory lag/date differs from the per-case audit")
            for role in ("value", "background"):
                forecast_hash = source[f"{role}_forecast_sha256"]
                forecast_path_text = source[f"{role}_forecast_path"]
                if not _is_sha256(forecast_hash):
                    raise ValueError(f"invalid {role} forecast source hash")
                if not isinstance(forecast_path_text, str) or not forecast_path_text:
                    raise ValueError(f"{role} forecast source path must be explicit")
                forecast_path = Path(forecast_path_text)
                if (
                    not forecast_path.is_file()
                    or hashlib.sha256(forecast_path.read_bytes()).hexdigest() != forecast_hash
                ):
                    raise ValueError(f"{role} forecast source path does not match its sealed hash")
            if Path(source["value_forecast_path"]).resolve() == Path(
                source["background_forecast_path"]
            ).resolve():
                raise ValueError("value and background forecasts must be distinct sources")
            sral_paths = source["sral_paths"]
            sral_hashes = source["sral_sha256"]
            if (
                not isinstance(sral_paths, list)
                or not sral_paths
                or not all(isinstance(item, str) and item for item in sral_paths)
                or not isinstance(sral_hashes, list)
                or not sral_hashes
                or not all(_is_sha256(item) for item in sral_hashes)
                or len(sral_paths) != len(sral_hashes)
            ):
                raise ValueError("each lag must bind explicit real SRAL paths one-to-one with hashes")
            for sral_path_text, sral_hash in zip(sral_paths, sral_hashes):
                sral_path = Path(sral_path_text)
                if not sral_path.is_file() or hashlib.sha256(sral_path.read_bytes()).hexdigest() != sral_hash:
                    raise ValueError("SRAL source path does not match its sealed hash")
    for case, manifest_case in zip(cases, manifest_cases):
        if set(case) != {
            "case_id", "coverage_slot", "target_date", "hour", "truth_finite_on_valid_domain",
            "lag_specific_backgrounds", "co_registered_shape", "orientation_landmarks", "lags", "panel",
            "panel_sha256",
        }:
            raise ValueError("per-case keys must be exact")
        for identity_key in ("case_id", "coverage_slot", "target_date", "hour"):
            if case[identity_key] != manifest_case[identity_key]:
                raise ValueError("per-case identity metadata differs from the frozen manifest")
        if case["truth_finite_on_valid_domain"] is not True or case["lag_specific_backgrounds"] is not True:
            raise ValueError("truth finiteness or lag-specific background invariant failed")
        if case["co_registered_shape"] != [320, 256] or len(case["orientation_landmarks"]) != 2:
            raise ValueError("co-registration/orientation contract failed")
        for observed, expected in zip(case["orientation_landmarks"], manifest_landmarks):
            if set(observed) != {"name", "row", "column", "valid_domain", "lag_mask_values"}:
                raise ValueError("orientation landmark keys must be exact")
            if any(observed[key] != expected[key] for key in ("name", "row", "column")):
                raise ValueError("orientation landmarks differ from the frozen geographic anchors")
            if not isinstance(observed["valid_domain"], bool):
                raise ValueError("orientation landmark valid-domain state must be boolean")
            if (
                not isinstance(observed["lag_mask_values"], list)
                or len(observed["lag_mask_values"]) != 3
                or any(value not in (0, 1) for value in observed["lag_mask_values"])
            ):
                raise ValueError("orientation landmark lag-mask states must be three binary values")
        if [lag.get("lag_days") for lag in case["lags"]] != [0, 1, 2]:
            raise ValueError("lag order changed")
        for lag in case["lags"]:
            if set(lag) != {
                "lag_days", "value_source_date", "background_source_date",
                "sral_footprint_pixels", "footprint_pixels", "finite_on_observed",
                "target_raw_nonfinite_sral_pixels",
                "background_raw_finite_observed_pixels",
                "background_padding_observed_pixels",
                "background_raw_finite_required_for_innovation",
                "nan_outside_observed", "geometry_provenance", "value_provenance",
                "occurrence_encoder_value_space", "innovation_nonzero_pixels",
                "future_date_leakage",
            }:
                raise ValueError("lag audit keys must be exact")
            if lag["finite_on_observed"] is not True or lag["nan_outside_observed"] is not True:
                raise ValueError("NaN-safe observed-pixel invariant failed")
            counts = (
                lag["sral_footprint_pixels"], lag["footprint_pixels"],
                lag["target_raw_nonfinite_sral_pixels"],
                lag["background_raw_finite_observed_pixels"],
                lag["background_padding_observed_pixels"],
            )
            if any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in counts
            ):
                raise ValueError("raw-finite audit counts must be non-negative integers")
            if (
                lag["sral_footprint_pixels"]
                != lag["footprint_pixels"] + lag["target_raw_nonfinite_sral_pixels"]
            ):
                raise ValueError("target raw-finite footprint accounting is inconsistent")
            if (
                lag["footprint_pixels"]
                != lag["background_raw_finite_observed_pixels"]
                + lag["background_padding_observed_pixels"]
            ):
                raise ValueError("background raw-finite footprint accounting is inconsistent")
            if lag["background_raw_finite_required_for_innovation"] is not False:
                raise ValueError("background raw-finite innovation policy drift")
            if lag["future_date_leakage"] is not False:
                raise ValueError("future-date leakage detected")
            if (
                lag["geometry_provenance"] != "real_sral_footprint"
                or lag["value_provenance"] != "target_year_m2m_forecast_at_real_sral_footprint"
                or lag["occurrence_encoder_value_space"] != "physical_0_1"
            ):
                raise ValueError("geometry/value provenance drift")
            if (
                isinstance(lag["innovation_nonzero_pixels"], bool)
                or not isinstance(lag["innovation_nonzero_pixels"], int)
                or lag["innovation_nonzero_pixels"] < 0
            ):
                raise ValueError("invalid innovation nonzero-pixel evidence")
            if (
                isinstance(lag["footprint_pixels"], bool)
                or not isinstance(lag["footprint_pixels"], int)
                or lag["footprint_pixels"] <= 0
            ):
                raise ValueError("real lag footprint must be non-empty")
        if sum(lag["innovation_nonzero_pixels"] for lag in case["lags"]) == 0:
            raise ValueError("lagged forecast innovations are identically zero")
        expected_panel = f"panels/{case['case_id']}_truth_background_lag_masks.png"
        if case["panel"] != expected_panel:
            raise ValueError("compact orientation panel path is not bound to its frozen case")
        panel = output / expected_panel
        if not panel.is_file() or _png_dimensions(panel) != (1816, 320):
            raise ValueError("compact orientation panel has invalid geometry")
        if not _is_sha256(case["panel_sha256"]):
            raise ValueError("compact orientation panel hash is invalid")
        if hashlib.sha256(panel.read_bytes()).hexdigest() != case["panel_sha256"]:
            raise ValueError("compact orientation panel differs from its sealed hash")
        _validate_landmark_overlay(panel)
        _validate_mask_footprints(panel, case)
    return "E1_REAL_DATA_AUDIT_PASS"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps({"decision": validate_compact_audit(args.config, args.output_dir)}, sort_keys=True))


if __name__ == "__main__":
    main()
