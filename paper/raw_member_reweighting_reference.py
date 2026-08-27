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


def construct_selection(normalized_ranks):
    counts, probabilities = analog_rank_probabilities(normalized_ranks)
    positions = systematic_rank_positions(probabilities)
    return {
        "analog_rank_bin_counts": counts,
        "analog_rank_probabilities": probabilities,
        "source_raw_member_indices": positions,
        **selection_diagnostics(positions),
    }


if __name__ == "__main__":
    uniform = construct_selection([(index + 0.5) / RANKS for index in range(RANKS)])
    assert uniform["source_raw_member_indices"] == list(range(RANKS))
    assert abs(uniform["effective_sample_size"] - RANKS) < 1e-12
    print("raw_member_reweighting_reference=PASS")
