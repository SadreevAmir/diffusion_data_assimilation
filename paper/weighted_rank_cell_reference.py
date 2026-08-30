"""Reference weighted extension of the existing finite-ensemble rank gate."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence


def weighted_rank_cell(
    members: Sequence[float],
    truth: float,
    weights: Sequence[float],
    *,
    tie_draw: int,
) -> tuple[float, list[float]]:
    """Return a weighted rank coordinate and its mass-preserving M+1-bin count.

    ``tie_draw`` is the already-seeded integer draw in ``[0, K_eq]``.  At equal
    weights this is exactly the integer rank used by ``assim_lib.metrics``.
    """
    member_count = len(members)
    if member_count == 0 or len(weights) != member_count:
        raise ValueError("members and weights must have the same positive length")
    if isinstance(truth, bool) or not math.isfinite(truth):
        raise ValueError("truth must be finite")
    if any(isinstance(value, bool) or not math.isfinite(value) for value in members):
        raise ValueError("members must be finite")
    if any(isinstance(weight, bool) or not math.isfinite(weight) or weight < 0 for weight in weights):
        raise ValueError("weights must be finite and non-negative")
    if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("weights must sum to one")

    below_mass = sum(weight for value, weight in zip(members, weights) if value < truth)
    equal_weights = [weight for value, weight in zip(members, weights) if value == truth]
    equal_count = len(equal_weights)
    if isinstance(tie_draw, bool) or not isinstance(tie_draw, int) or not 0 <= tie_draw <= equal_count:
        raise ValueError("tie_draw must be an integer in [0, K_eq]")

    if equal_count:
        coordinate = member_count * (
            below_mass + (tie_draw / equal_count) * sum(equal_weights)
        )
    else:
        coordinate = member_count * below_mass
    coordinate = min(float(member_count), max(0.0, coordinate))
    nearest = round(coordinate)
    if math.isclose(coordinate, nearest, rel_tol=0.0, abs_tol=1e-12):
        coordinate = float(nearest)

    lower = min(member_count, int(math.floor(coordinate)))
    upper = min(member_count, int(math.ceil(coordinate)))
    counts = [0.0] * (member_count + 1)
    if lower == upper:
        counts[lower] = 1.0
    else:
        counts[lower] = upper - coordinate
        counts[upper] = coordinate - lower
    return coordinate, counts


def seeded_tie_draw(equal_count: int, *, seed: int) -> int:
    """Small stdlib oracle for the inclusive tie-cell draw."""
    if isinstance(equal_count, bool) or not isinstance(equal_count, int) or equal_count < 0:
        raise ValueError("equal_count must be a non-negative integer")
    return random.Random(seed).randrange(equal_count + 1)
