#!/usr/bin/env python3
"""Fail-closed validator for the compact E1 real-data audit."""

from __future__ import annotations

import argparse
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import struct
from typing import Any

MODE = "occurrence_intensity_e1_real_data_audit"


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _png_dimensions(path: Path) -> tuple[int, int]:
    header = path.read_bytes()[:24]
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError("missing or invalid compact orientation panel")
    return struct.unpack(">II", header[16:24])


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_compact_audit(config_path: Path, output: Path) -> str:
    config = _load(config_path)
    status = _load(output / "run_status.json")
    metadata = _load(output / "metadata.json")
    cases = _load(output / "per_case_audit.json")
    if status != {
        "status": "completed", "mode": MODE, "resource_kind": "server_cpu", "cases": 8,
        "engineering_only": True, "calibration_claim": False,
    }:
        raise ValueError("run_status does not describe the frozen engineering-only audit")
    expected_metadata = {
        "config_sha256", "case_manifest_sha256", "source_inventory_sha256",
        "source_inventory", "dataset_provenance", "truth_values_consulted_for_selection", "lags_days",
        "sral_footprint_semantics", "panel_layout",
    }
    if set(metadata) != expected_metadata:
        raise ValueError("metadata keys must be exact")
    if metadata["config_sha256"] != hashlib.sha256(config_path.read_bytes()).hexdigest():
        raise ValueError("metadata does not bind the frozen config")
    manifest_path = Path(config["case_manifest"])
    if metadata["case_manifest_sha256"] != hashlib.sha256(manifest_path.read_bytes()).hexdigest():
        raise ValueError("metadata does not bind the frozen case manifest")
    if metadata["truth_values_consulted_for_selection"] is not False:
        raise ValueError("truth-dependent case selection is forbidden")
    if metadata["lags_days"] != [0, 1, 2] or metadata["panel_layout"] != [
        "truth", "background_lag0", "mask_lag0", "background_lag1", "mask_lag1",
        "background_lag2", "mask_lag2",
    ]:
        raise ValueError("lag or panel layout drift")
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
    manifest_cases = _load(manifest_path)["cases"]
    expected_ids = [case["case_id"] for case in manifest_cases]
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
            if set(source) != {"lag", "date", "forecast_sha256", "sral_sha256"}:
                raise ValueError("source inventory lag keys must be exact")
            expected_date = (target_date - timedelta(days=expected_lag)).isoformat()
            if (
                not isinstance(audit_lag, dict)
                or source["lag"] != expected_lag
                or source["date"] != expected_date
                or source["date"] != audit_lag.get("source_date")
            ):
                raise ValueError("source inventory lag/date differs from the per-case audit")
            if not _is_sha256(source["forecast_sha256"]):
                raise ValueError("invalid forecast source hash")
            sral_hashes = source["sral_sha256"]
            if (
                not isinstance(sral_hashes, list)
                or not sral_hashes
                or not all(_is_sha256(item) for item in sral_hashes)
            ):
                raise ValueError("each lag must bind at least one real SRAL source hash")
    for case, manifest_case in zip(cases, manifest_cases):
        if set(case) != {
            "case_id", "coverage_slot", "target_date", "hour", "truth_finite_on_valid_domain",
            "lag_specific_backgrounds", "co_registered_shape", "orientation_landmarks", "lags", "panel",
        }:
            raise ValueError("per-case keys must be exact")
        for identity_key in ("case_id", "coverage_slot", "target_date", "hour"):
            if case[identity_key] != manifest_case[identity_key]:
                raise ValueError("per-case identity metadata differs from the frozen manifest")
        if case["truth_finite_on_valid_domain"] is not True or case["lag_specific_backgrounds"] is not True:
            raise ValueError("truth finiteness or lag-specific background invariant failed")
        if case["co_registered_shape"] != [320, 256] or len(case["orientation_landmarks"]) != 2:
            raise ValueError("co-registration/orientation contract failed")
        if [lag.get("lag_days") for lag in case["lags"]] != [0, 1, 2]:
            raise ValueError("lag order changed")
        for lag in case["lags"]:
            if set(lag) != {
                "lag_days", "source_date", "footprint_pixels", "finite_on_observed",
                "nan_outside_observed", "geometry_provenance", "value_provenance",
                "future_date_leakage",
            }:
                raise ValueError("lag audit keys must be exact")
            if lag["finite_on_observed"] is not True or lag["nan_outside_observed"] is not True:
                raise ValueError("NaN-safe observed-pixel invariant failed")
            if lag["future_date_leakage"] is not False:
                raise ValueError("future-date leakage detected")
            if lag["geometry_provenance"] != "real_sral_footprint" or lag["value_provenance"] != "forecast_value_at_real_sral_footprint":
                raise ValueError("geometry/value provenance drift")
            if (
                isinstance(lag["footprint_pixels"], bool)
                or not isinstance(lag["footprint_pixels"], int)
                or lag["footprint_pixels"] <= 0
            ):
                raise ValueError("real lag footprint must be non-empty")
        panel = output / case["panel"]
        if not panel.is_file() or _png_dimensions(panel) != (1816, 320):
            raise ValueError("compact orientation panel has invalid geometry")
    return "E1_REAL_DATA_AUDIT_PASS"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps({"decision": validate_compact_audit(args.config, args.output_dir)}, sort_keys=True))


if __name__ == "__main__":
    main()
