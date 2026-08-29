#!/usr/bin/env python3
"""Fail-closed joint admission for the two coverage/occurrence gate records."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


FAMILIES = (
    "proper_score",
    "reliability",
    "boundary",
    "spatial_physical",
    "operational",
)
EXPECTED = {
    "threshold": {
        "experiment_id": "joint_coverage_occurrence_threshold_valid",
        "mechanism_id": "coverage_spread_occurrence_calibration",
        "variant_id": "purged_trace_ice_threshold",
    },
    "joint_rank": {
        "experiment_id": "joint_coverage_occurrence_joint_rank_valid",
        "mechanism_id": "joint_coverage_occurrence_rank_calibration",
        "variant_id": "purged_joint_scale_threshold_minimax",
    },
}
SOURCE = "joint_full_condition_validation_2022"


def _load(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: top-level JSON must be an object")
    return value, hashlib.sha256(raw).hexdigest()


def _require(record: dict, key: str, expected: object, label: str) -> None:
    actual = record.get(key)
    if actual != expected:
        raise ValueError(f"{label}.{key}: expected {expected!r}, got {actual!r}")


def admit(path: Path, label: str) -> dict:
    record, digest = _load(path)
    identity = record.get("identity", record)
    gate = record.get("gate", record.get("gate_decision"))
    if not isinstance(identity, dict) or not isinstance(gate, dict):
        raise ValueError(f"{label}: identity and gate objects are required")

    for key, expected in EXPECTED[label].items():
        _require(identity, key, expected, label)
    _require(identity, "source_experiment", SOURCE, label)
    _require(identity, "cases", 40, label)
    _require(identity, "ensemble_size", 10, label)

    decisions = {}
    for family in FAMILIES:
        value = gate.get(family)
        if type(value) is not bool:  # bool only: integers must fail closed.
            raise ValueError(f"{label}.gate.{family}: Boolean required")
        decisions[family] = value
    overall = gate.get("overall_eligible")
    if type(overall) is not bool:
        raise ValueError(f"{label}.gate.overall_eligible: Boolean required")
    if overall != all(decisions.values()):
        raise ValueError(f"{label}: overall_eligible is not the family conjunction")

    return {
        **EXPECTED[label],
        "source_experiment": SOURCE,
        "cases": 40,
        "ensemble_size": 10,
        "sha256": digest,
        "gate": {**decisions, "overall_eligible": overall},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=Path, required=True)
    parser.add_argument("--joint-rank", type=Path, required=True)
    args = parser.parse_args()
    admitted = {
        "admission": "GO",
        "results": [
            admit(args.threshold, "threshold"),
            admit(args.joint_rank, "joint_rank"),
        ],
    }
    print(json.dumps(admitted, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
