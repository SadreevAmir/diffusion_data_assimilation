#!/usr/bin/env python3
"""Atomic, fail-closed adapter for a probabilistic-DA compact bundle."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from .probabilistic_da_contract_oracle import (
        CASE_COLUMNS,
        EXPECTED_ARTIFACTS,
        evaluate_decision,
        validate_case_rows,
        validate_manifest,
    )
except ImportError:  # Preserve direct-script use at the reviewed boundary.
    from probabilistic_da_contract_oracle import (
        CASE_COLUMNS,
        EXPECTED_ARTIFACTS,
        evaluate_decision,
        validate_case_rows,
        validate_manifest,
    )


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain one JSON object")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _case_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(CASE_COLUMNS):
            raise ValueError("case metric CSV columns must be exact and ordered")
        rows = list(reader)
    parsed = []
    for row in rows:
        try:
            parsed.append({
                **row,
                "case_index": int(row["case_index"]),
                "fold": int(row["fold"]),
                **{metric: float(row[metric]) for metric in CASE_COLUMNS[4:]},
            })
        except (TypeError, ValueError) as error:
            raise ValueError("case metric CSV contains an invalid typed value") from error
    return parsed


def validate_result_directory(root: Path) -> str:
    """Validate all three files as one bundle and return its recomputed outcome."""
    if not root.is_dir() or {path.name for path in root.iterdir()} != set(EXPECTED_ARTIFACTS):
        raise ValueError("result directory must contain exactly three compact artifacts")

    summary_path = root / "probabilistic_da_summary.json"
    case_path = root / "probabilistic_da_per_case.csv"
    manifest = _json_object(root / "probabilistic_da_manifest.json")
    validate_manifest(manifest)

    expected_hashes = manifest["artifact_hashes"]
    observed_hashes = {
        summary_path.name: _sha256(summary_path),
        case_path.name: _sha256(case_path),
    }
    if observed_hashes != expected_hashes:
        raise ValueError("compact artifact hash mismatch")

    validate_case_rows(_case_rows(case_path))
    summary = _json_object(summary_path)
    if set(summary) != {
        "case_count", "letkf_fair_crps", "raw_learned_joint_fair_crps",
        "rank_uniformity_pass", "letkf_mean_rmse", "var3d_mean_rmse", "outcome",
    }:
        raise ValueError("summary keys must be exact")
    claimed = summary.pop("outcome")
    if claimed not in {"PROBABILISTIC_DA_USEFUL", "PROBABILISTIC_DA_NEGATIVE"}:
        raise ValueError("summary outcome is not a frozen decision label")
    recomputed = evaluate_decision(summary)
    if claimed != recomputed:
        raise ValueError("summary outcome disagrees with recomputed decision")
    return recomputed
