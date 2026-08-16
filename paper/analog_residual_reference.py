#!/usr/bin/env python3
"""Dependency-free review oracle for the frozen analog-residual method.

This is not an experiment entry point and reads no project data. Running it
executes deterministic synthetic checks of the future server runner's logic.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


ICE_THRESHOLD = 0.15
NEIGHBOR_COUNT = 10
FEATURE_COUNT = 6


def _weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    pairs = [(v, w) for v, w in zip(values, weights, strict=True) if math.isfinite(v) and math.isfinite(w) and w > 0]
    if not pairs:
        raise ValueError("no positively weighted finite values")
    return sum(v * w for v, w in pairs) / sum(w for _, w in pairs)


def weighted_semivariogram(field: Sequence[Sequence[float]], weights: Sequence[Sequence[float]], lag: int) -> float:
    """Mean half-squared increments over horizontal and vertical valid pairs."""
    rows, cols = len(field), len(field[0])
    if rows != len(weights) or any(len(row) != cols for row in [*field, *weights]):
        raise ValueError("field and weights must be equally shaped rectangular grids")
    if lag <= 0 or lag >= min(rows, cols):
        raise ValueError("lag must fit both spatial dimensions")
    increments: list[float] = []
    pair_weights: list[float] = []
    for row in range(rows):
        for col in range(cols - lag):
            increments.append(0.5 * (field[row][col + lag] - field[row][col]) ** 2)
            pair_weights.append(0.5 * (weights[row][col + lag] + weights[row][col]))
    for row in range(rows - lag):
        for col in range(cols):
            increments.append(0.5 * (field[row + lag][col] - field[row][col]) ** 2)
            pair_weights.append(0.5 * (weights[row + lag][col] + weights[row][col]))
    return _weighted_mean(increments, pair_weights)


def forecast_features(raw_mean: Sequence[Sequence[float]], weights: Sequence[Sequence[float]]) -> tuple[float, ...]:
    """Return the six forecast-only features in frozen order."""
    flat = [value for row in raw_mean for value in row]
    flat_weights = [value for row in weights for value in row]
    mean = _weighted_mean(flat, flat_weights)
    variance = _weighted_mean([(value - mean) ** 2 for value in flat], flat_weights)
    return (
        _weighted_mean(flat, flat_weights),
        _weighted_mean([float(value >= ICE_THRESHOLD) for value in flat], flat_weights),
        mean,
        math.sqrt(variance),
        weighted_semivariogram(raw_mean, weights, 1),
        weighted_semivariogram(raw_mean, weights, 4),
    )


def select_neighbors(held_out: Sequence[float], training: Sequence[Sequence[float]], ordered_case_indices: Sequence[int]) -> list[int]:
    """Select ten cases after training-only population standardization."""
    if len(held_out) != FEATURE_COUNT or any(len(row) != FEATURE_COUNT for row in training):
        raise ValueError("features must have six columns")
    if len(training) < NEIGHBOR_COUNT:
        raise ValueError("fewer than ten eligible training dates")
    if len(ordered_case_indices) != len(training):
        raise ValueError("ordered_case_indices has incompatible length")
    if not all(math.isfinite(value) for row in [held_out, *training] for value in row):
        raise ValueError("non-finite feature")
    columns = list(zip(*training, strict=True))
    centers = [sum(column) / len(column) for column in columns]
    scales = [math.sqrt(sum((v - c) ** 2 for v in column) / len(column)) for column, c in zip(columns, centers, strict=True)]
    if any(scale == 0 for scale in scales):
        raise ValueError("zero training population standard deviation")
    distances = [sum(((v - target) / scale) ** 2 for v, target, scale in zip(row, held_out, scales, strict=True)) for row in training]
    return sorted(range(len(training)), key=lambda pos: (distances[pos], ordered_case_indices[pos]))[:NEIGHBOR_COUNT]


def construct_candidate(raw_mean: Sequence[Sequence[float]], selected_residuals: Sequence[Sequence[Sequence[float]]]) -> tuple[list[list[list[float]]], dict[str, float]]:
    """Apply each of ten full residual fields once and report accounting."""
    rows, cols = len(raw_mean), len(raw_mean[0])
    if len(selected_residuals) != NEIGHBOR_COUNT or any(len(field) != rows or any(len(row) != cols for row in field) for field in selected_residuals):
        raise ValueError("selected_residuals must contain ten full fields")
    unbounded = [[[raw_mean[i][j] + residual[i][j] for j in range(cols)] for i in range(rows)] for residual in selected_residuals]
    candidate = [[[min(1.0, max(0.0, value)) for value in row] for row in field] for field in unbounded]
    total = NEIGHBOR_COUNT * rows * cols
    displacement = max(abs(sum(field[i][j] for field in candidate) / NEIGHBOR_COUNT - raw_mean[i][j]) for i in range(rows) for j in range(cols))
    report = {
        "lower_clipping_mass": sum(v < 0 for field in unbounded for row in field for v in row) / total,
        "upper_clipping_mass": sum(v > 1 for field in unbounded for row in field for v in row) / total,
        "maximum_ensemble_mean_displacement": displacement,
    }
    return candidate, report


def _self_test() -> None:
    grid = [[(row * 6 + col) / 35 for col in range(6)] for row in range(6)]
    weights = [[1.0] * 6 for _ in range(6)]
    assert len(forecast_features(grid, weights)) == FEATURE_COUNT
    training = [[float(row + col) for col in range(6)] for row in range(12)]
    assert len(select_neighbors([5.5 + col for col in range(6)], training, range(12))) == 10
    assert select_neighbors(training[5], training, list(reversed(range(12))))[0] == 5
    residuals = [[[(-0.8 + member * 1.6 / 9)] * 6 for _ in range(6)] for member in range(10)]
    candidate, report = construct_candidate(grid, residuals)
    assert min(v for field in candidate for row in field for v in row) == 0
    assert max(v for field in candidate for row in field for v in row) == 1
    assert report["lower_clipping_mass"] > 0 and report["upper_clipping_mass"] > 0
    try:
        select_neighbors([0.0] * 6, [[1.0] * 6 for _ in range(10)], range(10))
    except ValueError as exc:
        assert "zero training" in str(exc)
    else:
        raise AssertionError("zero-variance fold did not fail closed")


if __name__ == "__main__":
    _self_test()
    print("analog-residual reference checks: PASS")
