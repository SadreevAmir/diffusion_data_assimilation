#!/usr/bin/env python3
"""Fail-closed cross-file parity for score-aware compact result directories."""

from __future__ import annotations

import argparse
from datetime import date, timedelta
import hashlib
import json
import math
from pathlib import Path

try:
    from .score_aware_raw_reweighting_reference import MEMBERS, select_raw_scenarios
except ImportError:
    from score_aware_raw_reweighting_reference import MEMBERS, select_raw_scenarios


EXPECTED_CASES = 40
CASE_START = date(2022, 1, 1)
CASE_STRIDE_DAYS = 5
HOLDOUT_SIZE = 8
PURGE_CASES = 3
EXPECTED_CASE_IDS = tuple(
    (CASE_START + timedelta(days=CASE_STRIDE_DAYS * index)).isoformat()
    for index in range(EXPECTED_CASES)
)
FILES = {
    "cases": "case_selection.json",
    "aggregate": "aggregate_selection.json",
    "uncertainty": "paired_uncertainty.json",
    "gate": "gate_decision.json",
}
FAMILIES = ("proper_score", "reliability", "boundary", "spatial_physical", "operational")


def directory_sha256(directory: Path) -> str:
    """Bind the exact four-file directory with an unambiguous byte framing."""
    observed = {path.name for path in directory.iterdir() if path.is_file()}
    if observed != set(FILES.values()):
        raise ValueError("compact directory must contain exactly the four frozen files")
    digest = hashlib.sha256()
    for filename in sorted(FILES.values()):
        name = filename.encode("utf-8")
        payload = (directory / filename).read_bytes()
        digest.update(len(name).to_bytes(4, "big"))
        digest.update(name)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _load_exact(directory: Path) -> dict[str, object]:
    observed = {path.name for path in directory.iterdir() if path.is_file()}
    if observed != set(FILES.values()):
        raise ValueError("compact directory must contain exactly the four frozen files")
    return {
        name: json.loads((directory / filename).read_text(encoding="utf-8"))
        for name, filename in FILES.items()
    }


def _close(actual: object, expected: float, label: str) -> None:
    if isinstance(actual, bool) or not isinstance(actual, (int, float)):
        raise ValueError(f"{label} must be numeric")
    if not math.isfinite(float(actual)) or not math.isclose(
        float(actual), expected, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise ValueError(f"{label} disagrees with recomputed value")


def expected_training_case_ids(fold: int) -> list[str]:
    """Return the exact retained cases for a contiguous holdout and non-circular purge."""
    holdout_start = fold * HOLDOUT_SIZE
    holdout_stop = holdout_start + HOLDOUT_SIZE
    excluded_start = max(0, holdout_start - PURGE_CASES)
    excluded_stop = min(EXPECTED_CASES, holdout_stop + PURGE_CASES)
    return [
        case_id
        for index, case_id in enumerate(EXPECTED_CASE_IDS)
        if not excluded_start <= index < excluded_stop
    ]


def validate_directory(directory: Path) -> None:
    payload = _load_exact(directory)
    cases_doc = payload["cases"]
    if not isinstance(cases_doc, dict) or set(cases_doc) != {"schema_version", "cases"}:
        raise ValueError("case_selection.json has wrong schema")
    if cases_doc["schema_version"] != "score-aware-case-selection-v1":
        raise ValueError("case selection schema_version is unsupported")
    cases = cases_doc["cases"]
    if not isinstance(cases, list) or len(cases) != EXPECTED_CASES:
        raise ValueError("case selection must contain exactly forty cases")

    expected_case_keys = {
        "case_id", "fold", "training_case_ids", "predicted_risks", "normalized_weights",
        "source_raw_member_indices", "source_multiplicities",
        "unique_selected_raw_members", "effective_sample_size",
        "bitwise_copy_pass", "mask_invariants_pass", "analysis_fair_crps_delta",
        "analysis_crps_delta",
    }
    ids: set[str] = set()
    fold_counts = [0] * 5
    multiplicity_totals = [0] * MEMBERS
    ess_values: list[float] = []
    unique_values: list[int] = []
    copy_failures = mask_failures = 0
    fair_deltas: list[float] = []
    crps_deltas: list[float] = []
    for index, case in enumerate(cases):
        if not isinstance(case, dict) or set(case) != expected_case_keys:
            raise ValueError(f"case[{index}] has wrong schema")
        case_id = case["case_id"]
        if not isinstance(case_id, str) or not case_id or case_id in ids:
            raise ValueError("case_id values must be unique non-empty strings")
        ids.add(case_id)
        if case_id != EXPECTED_CASE_IDS[index]:
            raise ValueError("case_id order must match the frozen stride-five case envelope")
        if not isinstance(case["fold"], int) or isinstance(case["fold"], bool) or not 0 <= case["fold"] < 5:
            raise ValueError("fold must be an integer from zero through four")
        expected_fold = index // HOLDOUT_SIZE
        if case["fold"] != expected_fold:
            raise ValueError(f"{case_id}.fold disagrees with frozen contiguous holdout membership")
        if case["training_case_ids"] != expected_training_case_ids(expected_fold):
            raise ValueError(f"{case_id}.training_case_ids disagrees with frozen non-circular purge")
        fold_counts[case["fold"]] += 1
        expected = select_raw_scenarios(case["predicted_risks"])
        for key in ("normalized_weights", "source_raw_member_indices", "source_multiplicities"):
            if case[key] != expected[key]:
                raise ValueError(f"{case_id}.{key} disagrees with frozen selection")
        if case["unique_selected_raw_members"] != expected["unique_selected_raw_members"]:
            raise ValueError(f"{case_id}.unique_selected_raw_members disagrees with multiplicities")
        _close(case["effective_sample_size"], expected["effective_sample_size"], f"{case_id}.effective_sample_size")
        if not isinstance(case["bitwise_copy_pass"], bool) or not isinstance(case["mask_invariants_pass"], bool):
            raise ValueError("copy and mask invariant flags must be Boolean")
        for metric, destination in (("analysis_fair_crps_delta", fair_deltas), ("analysis_crps_delta", crps_deltas)):
            value = case[metric]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"{case_id}.{metric} must be finite")
            destination.append(float(value))
        copy_failures += not case["bitwise_copy_pass"]
        mask_failures += not case["mask_invariants_pass"]
        for member, count in enumerate(expected["source_multiplicities"]):
            multiplicity_totals[member] += count
        ess_values.append(expected["effective_sample_size"])
        unique_values.append(expected["unique_selected_raw_members"])

    if fold_counts != [8] * 5:
        raise ValueError("fold allocation must contain exactly eight cases in each of five folds")

    aggregate = payload["aggregate"]
    aggregate_keys = {"schema_version", "num_cases", "source_multiplicity_totals", "mean_unique_selected_raw_members", "mean_effective_sample_size", "bitwise_copy_failures", "mask_invariant_failures", "all_invariants_pass"}
    if not isinstance(aggregate, dict) or set(aggregate) != aggregate_keys or aggregate["schema_version"] != "score-aware-selection-aggregate-v1":
        raise ValueError("aggregate_selection.json has wrong schema")
    if aggregate["num_cases"] != EXPECTED_CASES or aggregate["source_multiplicity_totals"] != multiplicity_totals:
        raise ValueError("aggregate counts disagree with per-case selections")
    _close(aggregate["mean_unique_selected_raw_members"], sum(unique_values) / EXPECTED_CASES, "mean_unique_selected_raw_members")
    _close(aggregate["mean_effective_sample_size"], sum(ess_values) / EXPECTED_CASES, "mean_effective_sample_size")

    expected_pass = copy_failures == mask_failures == 0
    if (aggregate["bitwise_copy_failures"], aggregate["mask_invariant_failures"], aggregate["all_invariants_pass"]) != (copy_failures, mask_failures, expected_pass):
        raise ValueError("copy/mask aggregate disagrees with per-case invariants")

    uncertainty = payload["uncertainty"]
    if not isinstance(uncertainty, dict) or set(uncertainty) != {"schema_version", "analysis_fair_crps", "analysis_crps"} or uncertainty["schema_version"] != "score-aware-paired-uncertainty-v1":
        raise ValueError("paired_uncertainty.json has wrong schema")
    for metric, deltas in (("analysis_fair_crps", fair_deltas), ("analysis_crps", crps_deltas)):
        summary = uncertainty[metric]
        if not isinstance(summary, dict) or set(summary) != {"point_delta", "date_interval", "four_case_block_interval"}:
            raise ValueError(f"{metric} uncertainty has wrong schema")
        _close(summary["point_delta"], sum(deltas) / EXPECTED_CASES, f"{metric}.point_delta")
        for interval_name in ("date_interval", "four_case_block_interval"):
            interval = summary[interval_name]
            if not isinstance(interval, list) or len(interval) != 2:
                raise ValueError(f"{metric}.{interval_name} must contain two bounds")
            low, high = interval
            if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in interval) or low > high or not low <= summary["point_delta"] <= high:
                raise ValueError(f"{metric}.{interval_name} must be finite, ordered and contain the point estimate")

    gate = payload["gate"]
    if not isinstance(gate, dict) or set(gate) != {"schema_version", "families", "overall_eligible"}:
        raise ValueError("gate_decision.json has wrong schema")
    families = gate["families"]
    if gate["schema_version"] != "score-aware-no-compensation-gate-v1" or not isinstance(families, dict) or set(families) != set(FAMILIES):
        raise ValueError("gate family schema is not the frozen no-compensation gate")
    if any(not isinstance(value, bool) for value in families.values()):
        raise ValueError("gate family decisions must be Boolean")
    if families["operational"] != expected_pass:
        raise ValueError("operational gate disagrees with copy/mask invariants")
    if not isinstance(gate["overall_eligible"], bool) or gate["overall_eligible"] != all(families.values()):
        raise ValueError("overall_eligible must equal the conjunction of all families")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate score-aware compact directory parity.")
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    validate_directory(args.directory)
    print("score_aware_compact_outputs=PASS")


if __name__ == "__main__":
    main()
