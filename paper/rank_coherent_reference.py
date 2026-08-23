#!/usr/bin/env python3
"""Dependency-free review oracle for rank-coherent anomaly transport.

This is not an experiment entry point and reads no project data.  It makes the
frozen fold, ordering, anomaly-selection, and bounded mean-preservation rules
executable for an independent trusted-runner review.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


EXPECTED_CASES = 40
EXPECTED_MEMBERS = 10
HOLDOUT_SIZE = 8
PURGE = 3
ALPHAS = (0.0, 0.5, 0.75, 1.0, 1.25)
SOURCE_EXPERIMENT = "joint_full_condition_validation_2022"
ARTIFACT_POLICY = "summary_only"


def purged_folds() -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    """Return five contiguous holdouts and their non-circular purged training sets."""
    folds = []
    for start in range(0, EXPECTED_CASES, HOLDOUT_SIZE):
        stop = start + HOLDOUT_SIZE
        holdout = tuple(range(start, stop))
        excluded_start = max(0, start - PURGE)
        excluded_stop = min(EXPECTED_CASES, stop + PURGE)
        training = tuple(
            index
            for index in range(EXPECTED_CASES)
            if not excluded_start <= index < excluded_stop
        )
        folds.append((holdout, training))
    return tuple(folds)


def validate_runner_interface(
    parameters: Mapping[str, object], artifact_policy: str
) -> None:
    """Reject runtime knobs or retrieval beyond the frozen interface."""
    if dict(parameters) != {"source_experiment": SOURCE_EXPERIMENT}:
        raise ValueError("runner must expose only the frozen source_experiment")
    if artifact_policy != ARTIFACT_POLICY:
        raise ValueError("runner retrieval must remain summary_only")


def select_rank_stratified_fields(
    analog_case_indices: Sequence[int],
    analog_truth_ranks: Sequence[float],
    anomaly_case_means: Sequence[Sequence[float]],
    anomaly_fields: Sequence[Sequence[Sequence[Sequence[float]]]],
) -> tuple[Sequence[Sequence[float]], ...]:
    """Take order statistic j from analog date j after scalar-rank sorting.

    Inputs contain exactly ten already-selected analog dates in neighbor order.
    Case-rank ties use the ordered case index; member-mean ties use member index.
    Complete fields are returned by reference and are never spatially shuffled.
    """
    lengths = {
        len(analog_case_indices),
        len(analog_truth_ranks),
        len(anomaly_case_means),
        len(anomaly_fields),
    }
    if lengths != {EXPECTED_MEMBERS}:
        raise ValueError("exactly ten selected analog dates are required")
    if len(set(analog_case_indices)) != EXPECTED_MEMBERS:
        raise ValueError("selected analog dates must be distinct")
    if not all(math.isfinite(rank) for rank in analog_truth_ranks):
        raise ValueError("analog truth ranks must be finite")
    if any(len(means) != EXPECTED_MEMBERS for means in anomaly_case_means):
        raise ValueError("every analog date must contain ten anomaly means")
    if any(len(fields) != EXPECTED_MEMBERS for fields in anomaly_fields):
        raise ValueError("every analog date must contain ten anomaly fields")

    date_order = sorted(
        range(EXPECTED_MEMBERS),
        key=lambda position: (
            analog_truth_ranks[position],
            analog_case_indices[position],
        ),
    )
    selected = []
    for order_statistic, date_position in enumerate(date_order):
        means = anomaly_case_means[date_position]
        if not all(math.isfinite(value) for value in means):
            raise ValueError("anomaly case means must be finite")
        member_order = sorted(
            range(EXPECTED_MEMBERS), key=lambda member: (means[member], member)
        )
        selected.append(anomaly_fields[date_position][member_order[order_statistic]])
    return tuple(selected)


def pair_with_heldout_member_order(
    heldout_anomaly_means: Sequence[float],
    selected_fields: Sequence[Sequence[Sequence[float]]],
) -> tuple[Sequence[Sequence[float]], ...]:
    """Place borrowed fields into held-out raw-member order-statistic slots."""
    if len(heldout_anomaly_means) != EXPECTED_MEMBERS:
        raise ValueError("held-out ensemble must contain ten anomaly means")
    if len(selected_fields) != EXPECTED_MEMBERS:
        raise ValueError("exactly ten selected fields are required")
    if not all(math.isfinite(value) for value in heldout_anomaly_means):
        raise ValueError("held-out anomaly means must be finite")
    member_order = sorted(
        range(EXPECTED_MEMBERS),
        key=lambda member: (heldout_anomaly_means[member], member),
    )
    paired: list[Sequence[Sequence[float]] | None] = [None] * EXPECTED_MEMBERS
    for order_statistic, member in enumerate(member_order):
        paired[member] = selected_fields[order_statistic]
    if any(field is None for field in paired):
        raise AssertionError("incomplete held-out member pairing")
    return tuple(field for field in paired if field is not None)


def capped_simplex_projection(
    provisional: Sequence[float], target_mean: float, tolerance: float = 1e-13
) -> tuple[float, ...]:
    """Euclidean projection onto [0,1]^M with an exact target member mean."""
    if len(provisional) != EXPECTED_MEMBERS:
        raise ValueError("projection requires ten members")
    if not all(math.isfinite(value) for value in provisional):
        raise ValueError("projection input must be finite")
    if not math.isfinite(target_mean) or not 0.0 <= target_mean <= 1.0:
        raise ValueError("target mean must be finite and bounded")
    target_sum = EXPECTED_MEMBERS * target_mean
    lower = min(provisional) - 1.0
    upper = max(provisional)
    for _ in range(100):
        shift = 0.5 * (lower + upper)
        current = sum(min(1.0, max(0.0, value - shift)) for value in provisional)
        if current > target_sum:
            lower = shift
        else:
            upper = shift
        if upper - lower <= tolerance:
            break
    shift = 0.5 * (lower + upper)
    projected = tuple(min(1.0, max(0.0, value - shift)) for value in provisional)
    if abs(sum(projected) / EXPECTED_MEMBERS - target_mean) > 1e-10:
        raise ValueError("projection failed the frozen mean-preservation tolerance")
    return projected


def select_alpha(training_scores: Mapping[float, float], feasible: Mapping[float, bool]) -> tuple[float, bool]:
    """Select minimum fair CRPS among feasible frozen alphas, tie to smaller."""
    if set(training_scores) != set(ALPHAS) or set(feasible) != set(ALPHAS):
        raise ValueError("alpha accounting must contain the exact frozen set")
    if not all(math.isfinite(score) for score in training_scores.values()):
        raise ValueError("training fair CRPS must be finite")
    candidates = [alpha for alpha in ALPHAS if feasible[alpha]]
    if not candidates:
        raise ValueError("alpha=0.0 must provide the null feasible construction")
    positive = [alpha for alpha in candidates if alpha > 0.0]
    if not positive:
        return 0.0, True
    selected = min(candidates, key=lambda alpha: (training_scores[alpha], alpha))
    return selected, False


def _self_test() -> None:
    folds = purged_folds()
    assert len(folds) == 5
    assert folds[0][0] == tuple(range(8)) and folds[-1][0] == tuple(range(32, 40))
    for holdout, training in folds:
        assert not set(holdout) & set(training)
        assert all(min(abs(case - held) for held in holdout) > PURGE for case in training)

    fields = [
        [[[1000 * case + 10 * member + row + col] for col in range(2)] for row in range(2)]
        for case in range(10)
        for member in range(10)
    ]
    nested_fields = [fields[case * 10 : (case + 1) * 10] for case in range(10)]
    means = [[float(member) for member in range(10)] for _ in range(10)]
    selected = select_rank_stratified_fields(
        list(reversed(range(10))), [0.5] * 10, means, nested_fields
    )
    assert selected[0] is nested_fields[9][0]
    assert selected[9] is nested_fields[0][9]
    paired = pair_with_heldout_member_order(list(reversed(range(10))), selected)
    assert paired[9] is selected[0] and paired[0] is selected[9]

    projected = capped_simplex_projection(
        (-0.5, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5), 0.5
    )
    assert min(projected) == 0.0 and max(projected) == 1.0
    assert abs(sum(projected) / 10 - 0.5) <= 1e-10
    selected_alpha, no_positive = select_alpha(
        {alpha: abs(alpha - 1.0) for alpha in ALPHAS},
        {alpha: alpha <= 1.0 for alpha in ALPHAS},
    )
    assert selected_alpha == 1.0 and no_positive is False
    assert select_alpha(
        {alpha: alpha for alpha in ALPHAS},
        {alpha: alpha == 0.0 for alpha in ALPHAS},
    ) == (0.0, True)
    validate_runner_interface({"source_experiment": SOURCE_EXPERIMENT}, ARTIFACT_POLICY)
    try:
        validate_runner_interface(
            {"source_experiment": SOURCE_EXPERIMENT, "alpha": 1.0},
            ARTIFACT_POLICY,
        )
    except ValueError as exc:
        assert "only" in str(exc)
    else:
        raise AssertionError("runtime alpha did not fail closed")


if __name__ == "__main__":
    _self_test()
    print("rank-coherent reference checks: PASS")
