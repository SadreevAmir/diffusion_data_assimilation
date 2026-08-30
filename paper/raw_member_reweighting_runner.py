#!/usr/bin/env python3
"""Pure trusted-runner surface for frozen raw-member rank reweighting.

This module deliberately contains no controller or filesystem plumbing.  The
server adapter owns sealed-data loading and the unchanged full gate; this file
owns the complete deterministic candidate construction and its fail-closed
provenance diagnostics.
"""

from __future__ import annotations

import math

try:
    from .raw_member_reweighting_reference import (
        analog_rank_probabilities,
        construct_selection,
        rank_positions_to_member_indices,
        selection_diagnostics,
        systematic_rank_positions,
    )
except ImportError:  # pragma: no cover - admission loads this file by path
    from paper.raw_member_reweighting_reference import (
        analog_rank_probabilities,
        construct_selection,
        rank_positions_to_member_indices,
        selection_diagnostics,
        systematic_rank_positions,
    )


CASES = 40
MEMBERS = 10
FOLDS = 5
HOLDOUT = 8
PURGE = 3
FEATURES = 6
SOURCE_EXPERIMENT = "joint_full_condition_validation_2022"
ARTIFACT_POLICY = "summary_only"


def purged_folds():
    """Return the frozen five contiguous holdouts and non-circular purge."""
    folds = []
    universe = range(CASES)
    for fold in range(FOLDS):
        start = fold * HOLDOUT
        holdout = tuple(range(start, start + HOLDOUT))
        training = tuple(
            index for index in universe
            if index not in holdout
            and not (max(0, start - PURGE) <= index < start)
            and not (start + HOLDOUT <= index < min(CASES, start + HOLDOUT + PURGE))
        )
        folds.append((holdout, training))
    return tuple(folds)


def validate_interface(parameters, artifact_policy):
    if parameters != {"source_experiment": SOURCE_EXPERIMENT}:
        raise ValueError("runner accepts only the frozen source_experiment")
    if artifact_policy != ARTIFACT_POLICY:
        raise ValueError("runner accepts only summary_only")


def _finite_vector(values, length, context):
    values = list(values)
    if len(values) != length:
        raise ValueError(f"{context} must contain exactly {length} values")
    result = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{context} must be numeric")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"{context} must be finite")
        result.append(value)
    return result


def standardize_and_select_analogs(target_features, training_features):
    """Select ten analog indices using training-only population scaling."""
    target = _finite_vector(target_features, FEATURES, "target features")
    if not isinstance(training_features, dict) or len(training_features) < MEMBERS:
        raise ValueError("training feature map must contain at least ten cases")
    checked = {
        index: _finite_vector(values, FEATURES, "training features")
        for index, values in training_features.items()
    }
    if any(isinstance(index, bool) or not isinstance(index, int) for index in checked):
        raise ValueError("training indices must be integers")
    means = [sum(row[j] for row in checked.values()) / len(checked) for j in range(FEATURES)]
    scales = []
    for j in range(FEATURES):
        scale = math.sqrt(sum((row[j] - means[j]) ** 2 for row in checked.values()) / len(checked))
        if not math.isfinite(scale) or scale == 0.0:
            raise ValueError("training population scale must be finite and positive")
        scales.append(scale)
    distances = []
    for index, row in checked.items():
        distance = sum(((row[j] - target[j]) / scales[j]) ** 2 for j in range(FEATURES))
        distances.append((distance, index))
    return [index for _, index in sorted(distances)[:MEMBERS]]


def _shape(field):
    if not isinstance(field, list) or not field or not isinstance(field[0], list) or not field[0]:
        raise ValueError("member field must be a non-empty rectangular 2-D list")
    shape = len(field), len(field[0])
    if any(not isinstance(row, list) or len(row) != shape[1] for row in field):
        raise ValueError("member field must be rectangular")
    for row in field:
        _finite_vector(row, shape[1], "member field row")
    return shape


def construct_case(analog_ranks, raw_member_case_means, raw_members):
    """Copy selected complete raw fields and return compact exact-copy evidence."""
    if not isinstance(raw_members, list) or len(raw_members) != MEMBERS:
        raise ValueError("raw ensemble must contain exactly ten members")
    shapes = [_shape(field) for field in raw_members]
    if len(set(shapes)) != 1:
        raise ValueError("raw members must share one shape")
    selection = construct_selection(analog_ranks, raw_member_case_means)
    indices = selection["source_raw_member_indices"]
    candidate = [raw_members[index] for index in indices]
    exact_copy = all(candidate[j] == raw_members[indices[j]] for j in range(MEMBERS))
    thresholds = (0.0, 0.15, 1.0, 0.999)
    mask_invariants = {}
    for threshold in thresholds:
        name = {0.0: "exact_zero", 0.15: "established_ice", 1.0: "exact_one", 0.999: "near_one"}[threshold]
        if threshold in (0.0, 1.0):
            compare = lambda value, t=threshold: value == t
        else:
            compare = lambda value, t=threshold: value >= t
        mask_invariants[name] = all(
            [[compare(value) for value in row] for row in candidate[j]]
            == [[compare(value) for value in row] for row in raw_members[indices[j]]]
            for j in range(MEMBERS)
        )
    mask_invariants["positive_ice"] = all(
        [[value > 0.0 for value in row] for row in candidate[j]]
        == [[value > 0.0 for value in row] for row in raw_members[indices[j]]]
        for j in range(MEMBERS)
    )
    if not exact_copy or not all(mask_invariants.values()):
        raise ValueError("source-copy or mask invariant failed")
    diagnostics = {
        **selection,
        "exact_source_copy": exact_copy,
        "mask_invariants": mask_invariants,
    }
    return candidate, diagnostics


def synthetic_result():
    """Return an outcome-agnostic compact dry-run record with no project data."""
    ranks = [(index + 0.5) / MEMBERS for index in range(MEMBERS)]
    means = [float(index) for index in range(MEMBERS)]
    raw = [[[index / MEMBERS, (index + 1) / MEMBERS]] for index in range(MEMBERS)]
    _, diagnostics = construct_case(ranks, means, raw)
    return {
        "schema_version": "raw-member-reweighting-synthetic-v1",
        "status": "PASS",
        "decision_bearing": False,
        "project_data_metrics_emitted": False,
        "folds": [
            {"holdout": list(holdout), "training": list(training)}
            for holdout, training in purged_folds()
        ],
        "construction": diagnostics,
    }


if __name__ == "__main__":
    assert synthetic_result()["status"] == "PASS"
    print("raw_member_reweighting_runner=PASS")
