#!/usr/bin/env python3
"""Fail-closed schema and cross-file parity for casewise-selector results."""

from __future__ import annotations

import argparse
import json
import math
from datetime import date, timedelta
from pathlib import Path


CASES = 40
MEMBERS = 10
HOLDOUT_SIZE = 8
PURGE = 3
PREDICTORS = 6
ACTION_MARGIN = 0.002
ACTIONS = ("raw", "projected_spread")
FAMILIES = ("proper_score", "reliability", "boundary", "spatial_physical", "operational")
FILES = {
    "cases": "case_selection.json",
    "aggregate": "aggregate_selection.json",
    "uncertainty": "paired_uncertainty.json",
    "gate": "gate_decision.json",
}
EXPECTED_CASE_IDS = tuple(
    (date(2022, 1, 1) + timedelta(days=5 * index)).isoformat()
    for index in range(CASES)
)


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be finite numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite numeric")
    return result


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{label} must be one lowercase SHA-256 digest")
    return value


def _vector(value: object, length: int, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{label} must contain exactly {length} values")
    return [_finite(item, f"{label}[{index}]") for index, item in enumerate(value)]


def expected_training_case_ids(fold: int) -> list[str]:
    start = fold * HOLDOUT_SIZE
    stop = start + HOLDOUT_SIZE
    excluded_start = max(0, start - PURGE)
    excluded_stop = min(CASES, stop + PURGE)
    return [
        case_id for index, case_id in enumerate(EXPECTED_CASE_IDS)
        if not excluded_start <= index < excluded_stop
    ]


def _load(directory: Path) -> dict[str, object]:
    observed = {path.name for path in directory.iterdir() if path.is_file()}
    if observed != set(FILES.values()):
        raise ValueError("compact directory must contain exactly the four frozen files")
    return {
        key: json.loads((directory / filename).read_text(encoding="utf-8"))
        for key, filename in FILES.items()
    }


def validate_directory(directory: Path) -> None:
    payload = _load(directory)
    cases_doc = payload["cases"]
    if not isinstance(cases_doc, dict) or set(cases_doc) != {"schema_version", "cases"}:
        raise ValueError("case_selection.json has wrong schema")
    if cases_doc["schema_version"] != "casewise-safety-selection-v1":
        raise ValueError("case selection schema_version is unsupported")
    cases = cases_doc["cases"]
    if not isinstance(cases, list) or len(cases) != CASES:
        raise ValueError("case selection must contain exactly forty cases")

    keys = {
        "case_id", "fold", "training_case_ids", "descriptors", "training_mean",
        "training_scale", "raw_coefficients", "projected_spread_coefficients",
        "raw_intercept", "projected_spread_intercept", "raw_predicted_loss",
        "projected_spread_predicted_loss", "selected_action", "action_margin",
        "raw_source_sha256", "projected_spread_source_sha256",
        "analysis_fair_crps_delta", "analysis_crps_delta", "bitwise_copy_pass",
    }
    action_counts = {action: 0 for action in ACTIONS}
    fair_deltas: list[float] = []
    crps_deltas: list[float] = []
    copy_failures = 0
    source_pair: tuple[str, str] | None = None
    for index, case in enumerate(cases):
        if not isinstance(case, dict) or set(case) != keys:
            raise ValueError(f"case[{index}] has wrong schema")
        case_id = case["case_id"]
        if case_id != EXPECTED_CASE_IDS[index]:
            raise ValueError("case_id order must match the frozen stride-five envelope")
        fold = case["fold"]
        if isinstance(fold, bool) or not isinstance(fold, int) or fold != index // HOLDOUT_SIZE:
            raise ValueError(f"{case_id}.fold disagrees with contiguous holdout membership")
        if case["training_case_ids"] != expected_training_case_ids(fold):
            raise ValueError(f"{case_id}.training_case_ids disagrees with non-circular purge")
        for name in ("descriptors", "training_mean", "training_scale", "raw_coefficients", "projected_spread_coefficients"):
            vector = _vector(case[name], PREDICTORS, f"{case_id}.{name}")
            if name == "training_scale" and any(value <= 0 for value in vector):
                raise ValueError(f"{case_id}.training_scale must be positive")
        raw_loss = _finite(case["raw_predicted_loss"], f"{case_id}.raw_predicted_loss")
        projected_loss = _finite(case["projected_spread_predicted_loss"], f"{case_id}.projected_spread_predicted_loss")
        for name in ("raw_intercept", "projected_spread_intercept"):
            _finite(case[name], f"{case_id}.{name}")
        margin = _finite(case["action_margin"], f"{case_id}.action_margin")
        if not math.isclose(margin, raw_loss - projected_loss, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"{case_id}.action_margin disagrees with predicted losses")
        expected_action = "projected_spread" if margin >= ACTION_MARGIN else "raw"
        if case["selected_action"] != expected_action:
            raise ValueError(f"{case_id}.selected_action disagrees with frozen margin rule")
        pair = (
            _digest(case["raw_source_sha256"], f"{case_id}.raw_source_sha256"),
            _digest(case["projected_spread_source_sha256"], f"{case_id}.projected_spread_source_sha256"),
        )
        if source_pair is not None and pair != source_pair:
            raise ValueError("source hashes must be identical across all cases")
        source_pair = pair
        if not isinstance(case["bitwise_copy_pass"], bool):
            raise ValueError("bitwise_copy_pass must be Boolean")
        copy_failures += not case["bitwise_copy_pass"]
        action_counts[expected_action] += 1
        fair_deltas.append(_finite(case["analysis_fair_crps_delta"], f"{case_id}.analysis_fair_crps_delta"))
        crps_deltas.append(_finite(case["analysis_crps_delta"], f"{case_id}.analysis_crps_delta"))

    aggregate = payload["aggregate"]
    aggregate_keys = {
        "schema_version", "num_cases", "ensemble_size", "action_counts",
        "raw_analysis_fair_crps", "candidate_analysis_fair_crps",
        "raw_analysis_crps", "candidate_analysis_crps", "bitwise_copy_failures",
        "non_degenerate_policy", "raw_source_sha256", "projected_spread_source_sha256",
    }
    if not isinstance(aggregate, dict) or set(aggregate) != aggregate_keys:
        raise ValueError("aggregate_selection.json has wrong schema")
    if aggregate["schema_version"] != "casewise-safety-aggregate-v1":
        raise ValueError("aggregate schema_version is unsupported")
    if aggregate["num_cases"] != CASES or aggregate["ensemble_size"] != MEMBERS:
        raise ValueError("aggregate envelope disagrees with frozen contract")
    if aggregate["action_counts"] != action_counts:
        raise ValueError("aggregate action_counts disagree with per-case selections")
    non_degenerate = all(action_counts[action] > 0 for action in ACTIONS)
    if aggregate["non_degenerate_policy"] is not non_degenerate:
        raise ValueError("non_degenerate_policy disagrees with action counts")
    if aggregate["bitwise_copy_failures"] != copy_failures:
        raise ValueError("bitwise_copy_failures disagrees with per-case records")
    if source_pair != (aggregate["raw_source_sha256"], aggregate["projected_spread_source_sha256"]):
        raise ValueError("aggregate source hashes disagree with per-case records")
    raw_fair = _finite(aggregate["raw_analysis_fair_crps"], "raw_analysis_fair_crps")
    candidate_fair = _finite(aggregate["candidate_analysis_fair_crps"], "candidate_analysis_fair_crps")
    raw_crps = _finite(aggregate["raw_analysis_crps"], "raw_analysis_crps")
    candidate_crps = _finite(aggregate["candidate_analysis_crps"], "candidate_analysis_crps")
    if not math.isclose(candidate_fair - raw_fair, sum(fair_deltas) / CASES, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("aggregate fair-CRPS delta disagrees with cases")
    if not math.isclose(candidate_crps - raw_crps, sum(crps_deltas) / CASES, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("aggregate CRPS delta disagrees with cases")

    uncertainty = payload["uncertainty"]
    if not isinstance(uncertainty, dict) or set(uncertainty) != {"schema_version", "analysis_fair_crps", "analysis_crps"}:
        raise ValueError("paired_uncertainty.json has wrong schema")
    if uncertainty["schema_version"] != "casewise-safety-paired-uncertainty-v1":
        raise ValueError("paired uncertainty schema_version is unsupported")
    for metric, deltas in (("analysis_fair_crps", fair_deltas), ("analysis_crps", crps_deltas)):
        summary = uncertainty[metric]
        if not isinstance(summary, dict) or set(summary) != {"point_delta", "date_interval", "four_case_block_interval"}:
            raise ValueError(f"{metric} uncertainty has wrong schema")
        point = _finite(summary["point_delta"], f"{metric}.point_delta")
        if not math.isclose(point, sum(deltas) / CASES, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"{metric}.point_delta disagrees with cases")
        for interval_name in ("date_interval", "four_case_block_interval"):
            interval = _vector(summary[interval_name], 2, f"{metric}.{interval_name}")
            if interval[0] > interval[1] or not interval[0] <= point <= interval[1]:
                raise ValueError(f"{metric}.{interval_name} must be ordered and contain point_delta")

    fair_interval = uncertainty["analysis_fair_crps"]["date_interval"]
    proper_pass = (
        raw_fair > 0 and (raw_fair - candidate_fair) / raw_fair >= 0.03
        and fair_interval[1] < 0 and candidate_crps <= 1.01 * raw_crps
    )
    gate = payload["gate"]
    if not isinstance(gate, dict) or set(gate) != {"schema_version", "families", "overall_eligible"}:
        raise ValueError("gate_decision.json has wrong schema")
    families = gate["families"]
    if gate["schema_version"] != "casewise-safety-no-compensation-gate-v1" or not isinstance(families, dict) or set(families) != set(FAMILIES):
        raise ValueError("gate family schema is not frozen")
    if any(not isinstance(value, bool) for value in families.values()):
        raise ValueError("gate family decisions must be Boolean")
    if families["proper_score"] is not proper_pass:
        raise ValueError("proper_score gate disagrees with frozen thresholds")
    expected_operational = copy_failures == 0 and non_degenerate
    if families["operational"] is not expected_operational:
        raise ValueError("operational gate disagrees with copy and policy checks")
    if not isinstance(gate["overall_eligible"], bool) or gate["overall_eligible"] != all(families.values()):
        raise ValueError("overall_eligible must equal the conjunction of all families")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate casewise-selector compact outputs.")
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    validate_directory(args.directory)
    print("casewise_safety_selector_compact_outputs=PASS")


if __name__ == "__main__":
    main()
