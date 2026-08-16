#!/usr/bin/env python3
"""Dependency-free review oracle for the frozen guidance-mixture runner.

This is not an experiment entry point and reads no project data.  It gives the
trusted runner review an executable definition of member ordering, common-noise
pairing, compact accounting, and fail-closed gate reporting.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence


SCALE_PAIRS = (
    (0.5, 0.5),
    (0.625, 0.375),
    (0.75, 0.25),
    (0.875, 0.125),
    (1.0, 0.0),
)
MEMBERS_PER_PAIR = 2
EXPECTED_CASES = 40
EXPECTED_MEMBERS = 10
SOURCE_EXPERIMENT = "joint_full_condition_validation_2022"
ARTIFACT_POLICY = "summary_only"
MANDATORY_FAMILIES = (
    "proper_scores",
    "finite_ensemble_reliability",
    "boundary_behaviour",
    "spatial_physical_preservation",
    "operational_validity",
)


def identifier_hash(values: Sequence[str]) -> str:
    """Hash an ordered identifier sequence with unambiguous framing."""
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def frozen_member_plan(seed_ids: Sequence[str], noise_ids: Sequence[str]) -> tuple[dict[str, object], ...]:
    """Build the only admissible ordered ten-member sampling plan."""
    if len(seed_ids) != MEMBERS_PER_PAIR or len(noise_ids) != MEMBERS_PER_PAIR:
        raise ValueError("exactly two frozen seed and noise identifiers are required")
    if list(seed_ids) != sorted(seed_ids) or len(set(seed_ids)) != MEMBERS_PER_PAIR:
        raise ValueError("seed identifiers must be distinct and ascending")
    if len(set(noise_ids)) != MEMBERS_PER_PAIR:
        raise ValueError("noise identifiers must be distinct")
    return tuple(
        {
            "member_index": pair_index * MEMBERS_PER_PAIR + seed_index,
            "track_scale": track,
            "background_scale": background,
            "seed_id": seed_ids[seed_index],
            "noise_id": noise_ids[seed_index],
        }
        for pair_index, (track, background) in enumerate(SCALE_PAIRS)
        for seed_index in range(MEMBERS_PER_PAIR)
    )


def validate_runner_interface(parameters: Mapping[str, object], artifact_policy: str) -> None:
    """Reject runtime knobs or retrieval beyond the frozen one-parameter interface."""
    if dict(parameters) != {"source_experiment": SOURCE_EXPERIMENT}:
        raise ValueError("runner must expose only the frozen source_experiment")
    if artifact_policy != ARTIFACT_POLICY:
        raise ValueError("runner retrieval must remain summary_only")


def compact_operational_accounting(
    plan: Sequence[Mapping[str, object]],
    completed_cases: int,
    fallback_member_count: int = 0,
) -> dict[str, object]:
    """Validate runner records and return the frozen compact accounting."""
    if completed_cases != EXPECTED_CASES:
        raise ValueError("all forty cases must complete")
    if len(plan) != EXPECTED_MEMBERS:
        raise ValueError("every case must have exactly ten members")
    if fallback_member_count != 0:
        raise ValueError("fallback member substitution is forbidden")
    expected = frozen_member_plan(
        [str(plan[index]["seed_id"]) for index in range(MEMBERS_PER_PAIR)],
        [str(plan[index]["noise_id"]) for index in range(MEMBERS_PER_PAIR)],
    )
    if tuple(dict(record) for record in plan) != expected:
        raise ValueError("member order, allocation, scale, seed, or common-noise pairing drifted")
    seed_ids = [str(record["seed_id"]) for record in plan]
    noise_ids = [str(record["noise_id"]) for record in plan]
    return {
        "completed_cases": completed_cases,
        "ensemble_size": len(plan),
        "per_scale_member_counts": [MEMBERS_PER_PAIR] * len(SCALE_PAIRS),
        "ordered_seed_identifier_hash": identifier_hash(seed_ids),
        "ordered_noise_identifier_hash": identifier_hash(noise_ids),
        "exact_common_noise_pairing": all(
            noise_ids[offset : offset + MEMBERS_PER_PAIR] == noise_ids[:MEMBERS_PER_PAIR]
            for offset in range(0, EXPECTED_MEMBERS, MEMBERS_PER_PAIR)
        ),
        "fallback_member_count": fallback_member_count,
    }


def compact_run_accounting(
    case_plans: Sequence[Sequence[Mapping[str, object]]],
    fallback_member_count: int,
) -> dict[str, object]:
    """Validate the ordered plan for every case, then emit one compact report."""
    if len(case_plans) != EXPECTED_CASES:
        raise ValueError("runner must report one member plan for each of forty cases")
    reports = [
        compact_operational_accounting(plan, EXPECTED_CASES, fallback_member_count)
        for plan in case_plans
    ]
    reference = reports[0]
    if any(report != reference for report in reports[1:]):
        raise ValueError("compact operational accounting differs between cases")
    return {**reference, "validated_case_plan_count": len(case_plans)}


def validate_compact_gate(gate: Mapping[str, object]) -> None:
    """Reject incomplete or compensating gate summaries before publication use."""
    families = gate.get("families")
    family_pass = gate.get("family_pass")
    if not isinstance(families, Mapping) or not isinstance(family_pass, Mapping):
        raise ValueError("gate must report families and family_pass mappings")
    if set(families) != set(MANDATORY_FAMILIES) or set(family_pass) != set(MANDATORY_FAMILIES):
        raise ValueError("gate family set is incomplete or unexpected")
    if gate.get("no_compensation_across_families") is not True:
        raise ValueError("no-compensation gate is not asserted")
    expected_eligible = all(value is True for value in family_pass.values())
    if gate.get("overall_eligible") is not expected_eligible:
        raise ValueError("overall_eligible is inconsistent with mandatory family passes")


def _self_test() -> None:
    validate_runner_interface(
        {"source_experiment": SOURCE_EXPERIMENT},
        ARTIFACT_POLICY,
    )
    plan = frozen_member_plan(("seed-000", "seed-001"), ("noise-000", "noise-001"))
    report = compact_operational_accounting(plan, EXPECTED_CASES)
    assert report["ensemble_size"] == EXPECTED_MEMBERS
    assert report["per_scale_member_counts"] == [2, 2, 2, 2, 2]
    assert report["exact_common_noise_pairing"] is True
    run_report = compact_run_accounting([plan] * EXPECTED_CASES, fallback_member_count=0)
    assert run_report["validated_case_plan_count"] == EXPECTED_CASES
    gate = {
        "families": {name: {} for name in MANDATORY_FAMILIES},
        "family_pass": {name: True for name in MANDATORY_FAMILIES},
        "no_compensation_across_families": True,
        "overall_eligible": True,
    }
    validate_compact_gate(gate)
    mutated = [dict(record) for record in plan]
    mutated[3]["noise_id"] = "noise-substitute"
    try:
        compact_operational_accounting(mutated, EXPECTED_CASES)
    except ValueError as exc:
        assert "pairing drifted" in str(exc)
    else:
        raise AssertionError("common-noise mutation did not fail closed")
    case_plans = [plan] * EXPECTED_CASES
    case_plans[17] = mutated
    try:
        compact_run_accounting(case_plans, fallback_member_count=0)
    except ValueError as exc:
        assert "pairing drifted" in str(exc)
    else:
        raise AssertionError("single-case member-plan mutation did not fail closed")
    try:
        compact_run_accounting([plan] * EXPECTED_CASES, fallback_member_count=1)
    except ValueError as exc:
        assert "fallback" in str(exc)
    else:
        raise AssertionError("fallback substitution did not fail closed")
    try:
        validate_runner_interface(
            {"source_experiment": SOURCE_EXPERIMENT, "track_scale": 0.75},
            ARTIFACT_POLICY,
        )
    except ValueError as exc:
        assert "only" in str(exc)
    else:
        raise AssertionError("extra runtime parameter did not fail closed")
    gate["overall_eligible"] = False
    try:
        validate_compact_gate(gate)
    except ValueError as exc:
        assert "inconsistent" in str(exc)
    else:
        raise AssertionError("gate inconsistency did not fail closed")


if __name__ == "__main__":
    _self_test()
    print("guidance-mixture reference checks: PASS")
