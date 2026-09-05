#!/usr/bin/env python3
"""Fail-closed semantic validation of one compact E2 comparison result."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "occurrence-intensity-e2-compact-v1"
DECISIONS = {"TEMPORAL_MECHANISM_USEFUL", "TEMPORAL_MECHANISM_NEGATIVE", "E2_INVALID"}
INVARIANTS = (
    "finite", "range", "exact_mask", "provenance", "orientation",
    "masked_leakage", "zero_hard_clips", "lag_operator_parity",
)
OPERATIONAL_CHECKS = (
    "complete_case_vectors", "identity_match", "inventory_match",
    "code_digest_match", "ten_member_schedule_match",
)
METRICS = ("innovation_rmse", "analysis_fair_crps", "offtrack_anomaly_energy")
TOP_LEVEL_KEYS = {
    "schema_version", "e1_experiment_id", "e2_experiment_id", "inventory_sha256",
    "e1_code_sha256", "e2_code_sha256", "case_ids", "lag_order", "ensemble_size",
    "casewise", "aggregates", "invariants", "operational_checks",
    "secondary_diagnostics", "decision",
}


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{label} must be one lowercase SHA-256 digest")
    return value


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be finite numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite numeric")
    return result


def _vector(value: object, size: int, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"{label} must contain exactly one value per case")
    return [_finite(item, f"{label}[{index}]") for index, item in enumerate(value)]


def _exact_bools(value: object, keys: tuple[str, ...], label: str) -> dict[str, bool]:
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f"{label} keys must be exact")
    if any(not isinstance(item, bool) for item in value.values()):
        raise ValueError(f"{label} values must be Boolean")
    return value


def _mean(values: list[float]) -> float:
    return math.fsum(values) / len(values)


def validate_result(result: Mapping[str, Any]) -> str:
    if set(result) != TOP_LEVEL_KEYS:
        raise ValueError("compact result keys must be exact")
    if result["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported schema_version")
    for identity in ("e1_experiment_id", "e2_experiment_id"):
        if not isinstance(result[identity], str) or not result[identity].strip():
            raise ValueError(f"{identity} must be non-empty")
    if result["e1_experiment_id"] == result["e2_experiment_id"]:
        raise ValueError("E1 and E2 identities must differ")
    for digest in ("inventory_sha256", "e1_code_sha256", "e2_code_sha256"):
        _digest(result[digest], digest)
    case_ids = result["case_ids"]
    if not isinstance(case_ids, list) or not case_ids or any(not isinstance(x, str) or not x for x in case_ids):
        raise ValueError("case_ids must be a non-empty string list")
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("case_ids must be unique and ordered")
    if result["lag_order"] != ["current", "one_day_old", "two_days_old"]:
        raise ValueError("lag_order must be exact")
    if result["ensemble_size"] != 10:
        raise ValueError("ensemble_size must equal ten")

    casewise = result["casewise"]
    if not isinstance(casewise, dict) or set(casewise) != set(METRICS):
        raise ValueError("casewise metric keys must be exact")
    vectors: dict[str, tuple[list[float], list[float]]] = {}
    for metric in METRICS:
        pair = casewise[metric]
        if not isinstance(pair, dict) or set(pair) != {"e1", "e2"}:
            raise ValueError(f"casewise.{metric} keys must be exact")
        before = _vector(pair["e1"], len(case_ids), f"casewise.{metric}.e1")
        after = _vector(pair["e2"], len(case_ids), f"casewise.{metric}.e2")
        if any(value < 0 for value in before + after):
            raise ValueError(f"casewise.{metric} values must be non-negative")
        vectors[metric] = before, after

    aggregates = result["aggregates"]
    if not isinstance(aggregates, dict) or set(aggregates) != set(METRICS):
        raise ValueError("aggregate metric keys must be exact")
    means: dict[str, tuple[float, float]] = {}
    for metric, (before, after) in vectors.items():
        summary = aggregates[metric]
        if not isinstance(summary, dict) or set(summary) != {"e1_mean", "e2_mean", "relative_change"}:
            raise ValueError(f"aggregates.{metric} keys must be exact")
        e1_mean = _finite(summary["e1_mean"], f"aggregates.{metric}.e1_mean")
        e2_mean = _finite(summary["e2_mean"], f"aggregates.{metric}.e2_mean")
        if e1_mean <= 0:
            raise ValueError(f"aggregates.{metric}.e1_mean must be positive")
        relative = _finite(summary["relative_change"], f"aggregates.{metric}.relative_change")
        expected = (e2_mean - e1_mean) / e1_mean
        if not math.isclose(e1_mean, _mean(before), rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"aggregates.{metric}.e1_mean disagrees with casewise values")
        if not math.isclose(e2_mean, _mean(after), rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"aggregates.{metric}.e2_mean disagrees with casewise values")
        if not math.isclose(relative, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"aggregates.{metric}.relative_change disagrees with means")
        means[metric] = e1_mean, e2_mean

    invariants = _exact_bools(result["invariants"], INVARIANTS, "invariants")
    operational = _exact_bools(result["operational_checks"], OPERATIONAL_CHECKS, "operational_checks")
    diagnostics = result["secondary_diagnostics"]
    diagnostic_keys = {"analysis_crps", "absolute_rank_adequacy", "attainable_coverage", "boundary", "spatial_physical"}
    if not isinstance(diagnostics, dict) or set(diagnostics) != diagnostic_keys:
        raise ValueError("secondary_diagnostics keys must be exact")
    if any(not isinstance(value, dict) or not value for value in diagnostics.values()):
        raise ValueError("each secondary diagnostic must be a non-empty object")

    all_evidence_valid = all(invariants.values()) and all(operational.values())
    if all_evidence_valid:
        useful = (
            means["innovation_rmse"][1] <= 0.90 * means["innovation_rmse"][0]
            and means["analysis_fair_crps"][1] <= 1.02 * means["analysis_fair_crps"][0]
            and means["offtrack_anomaly_energy"][1] <= 1.05 * means["offtrack_anomaly_energy"][0]
        )
        expected_decision = "TEMPORAL_MECHANISM_USEFUL" if useful else "TEMPORAL_MECHANISM_NEGATIVE"
    else:
        expected_decision = "E2_INVALID"
    if result["decision"] not in DECISIONS or result["decision"] != expected_decision:
        raise ValueError(f"decision must equal recomputed {expected_decision}")
    return expected_decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    value = json.loads(args.result.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("compact result must be a JSON object")
    decision = validate_result(value)
    print(json.dumps({"schema": SCHEMA_VERSION, "decision": decision}, sort_keys=True))


if __name__ == "__main__":
    main()
