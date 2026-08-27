"""Dependency-free oracle for the frozen raw-member reweighting contract."""

from __future__ import annotations

import math


RANKS = 10
SMOOTHING = 0.5


def analog_rank_probabilities(normalized_ranks):
    """Return the frozen ten-bin Jeffreys-smoothed analog probabilities."""
    ranks = list(normalized_ranks)
    if len(ranks) != RANKS:
        raise ValueError("exactly ten analog ranks are required")
    counts = [0] * RANKS
    for value in ranks:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("ranks must be numeric")
        value = float(value)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("ranks must be finite and in [0,1]")
        counts[min(RANKS - 1, math.floor(RANKS * value))] += 1
    denominator = RANKS + RANKS * SMOOTHING
    probabilities = [(count + SMOOTHING) / denominator for count in counts]
    if abs(sum(probabilities) - 1.0) > 1e-12:
        raise AssertionError("probability normalization drift")
    return counts, probabilities


def systematic_rank_positions(probabilities):
    """Map the ten frozen midpoint positions through the discrete inverse CDF."""
    probabilities = list(probabilities)
    if len(probabilities) != RANKS:
        raise ValueError("exactly ten rank probabilities are required")
    cumulative = []
    total = 0.0
    for value in probabilities:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("probabilities must be numeric")
        value = float(value)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("probabilities must be finite and non-negative")
        total += value
        cumulative.append(total)
    if abs(total - 1.0) > 1e-12:
        raise ValueError("probabilities must sum to one")
    positions = []
    for member in range(RANKS):
        target = (member + 0.5) / RANKS
        position = next((index for index, bound in enumerate(cumulative)
                         if target <= bound + 1e-15), None)
        if position is None:
            raise AssertionError("inverse CDF left a position unassigned")
        positions.append(position)
    return positions


def selection_diagnostics(positions):
    """Return frozen multiplicities, unique count and effective sample size."""
    positions = list(positions)
    if len(positions) != RANKS or any(
        isinstance(value, bool) or not isinstance(value, int)
        or value < 0 or value >= RANKS for value in positions
    ):
        raise ValueError("positions must be ten integer raw-member indices")
    multiplicities = [positions.count(index) for index in range(RANKS)]
    ess = 1.0 / sum((count / RANKS) ** 2 for count in multiplicities)
    return {
        "source_multiplicities": multiplicities,
        "unique_selected_raw_members": sum(count > 0 for count in multiplicities),
        "effective_sample_size": ess,
    }


def rank_positions_to_member_indices(raw_member_case_means, positions):
    """Resolve selected rank positions using the frozen mean/index ordering."""
    means = list(raw_member_case_means)
    if len(means) != RANKS:
        raise ValueError("exactly ten raw-member case means are required")
    checked_means = []
    for value in means:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("raw-member case means must be numeric")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("raw-member case means must be finite")
        checked_means.append(value)
    checked_positions = list(positions)
    selection_diagnostics(checked_positions)
    ordered_members = sorted(range(RANKS), key=lambda index: (checked_means[index], index))
    return [ordered_members[position] for position in checked_positions]


def construct_selection(normalized_ranks, raw_member_case_means=None):
    counts, probabilities = analog_rank_probabilities(normalized_ranks)
    positions = systematic_rank_positions(probabilities)
    source_indices = positions
    if raw_member_case_means is not None:
        source_indices = rank_positions_to_member_indices(raw_member_case_means, positions)
    return {
        "analog_rank_bin_counts": counts,
        "analog_rank_probabilities": probabilities,
        "selected_rank_positions": positions,
        "source_raw_member_indices": source_indices,
        **selection_diagnostics(source_indices),
    }


if __name__ == "__main__":
    uniform = construct_selection([(index + 0.5) / RANKS for index in range(RANKS)])
    assert uniform["source_raw_member_indices"] == list(range(RANKS))
    assert abs(uniform["effective_sample_size"] - RANKS) < 1e-12
    print("raw_member_reweighting_reference=PASS")
