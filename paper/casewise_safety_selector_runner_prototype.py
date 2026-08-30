"""Deterministic prototype for the frozen purged casewise safety selector.

This module is a local review target only.  It reads no artifacts, chooses no
runtime settings and cannot launch an experiment.
"""

from __future__ import annotations

import math

try:
    import numpy as np
except ModuleNotFoundError:  # Keep constants and non-numeric checks inspectable.
    np = None


CASES = 40
HOLDOUT_SIZE = 8
PURGE = 3
PREDICTORS = 6
RIDGE = 1.0
SVD_RELATIVE_CUTOFF = 1e-12
ACTION_MARGIN = 0.002
ACTIONS = ("raw", "projected_spread")

_DIAGNOSTICS = (
    "fair_crps", "rank_abs_error", "inner_coverage_error", "boundary_error",
    "iiee_ratio", "edge_ratio", "variogram_ratio",
)
_LIMITS = ("rank_limit", "inner_limit", "boundary_limit")


def _finite_number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be finite numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite numeric")
    return result


def build_purged_folds(case_ids):
    identifiers = tuple(case_ids)
    if len(identifiers) != CASES or len(set(identifiers)) != CASES:
        raise ValueError("case_ids must contain forty unique non-empty strings")
    if any(not isinstance(identifier, str) or identifier == "" for identifier in identifiers):
        raise ValueError("case_ids must contain forty unique non-empty strings")
    records = []
    for fold in range(CASES // HOLDOUT_SIZE):
        first = fold * HOLDOUT_SIZE
        last = first + HOLDOUT_SIZE
        forbidden = range(max(0, first - PURGE), min(CASES, last + PURGE))
        forbidden_indices = frozenset(forbidden)
        records.append({
            "fold": fold,
            "holdout_case_ids": identifiers[first:last],
            "training_case_ids": tuple(
                identifier for index, identifier in enumerate(identifiers)
                if index not in forbidden_indices
            ),
        })
    return tuple(records)


def scalar_loss(diagnostics, limits):
    if not isinstance(diagnostics, dict) or set(diagnostics) != set(_DIAGNOSTICS):
        raise ValueError("diagnostics must have the exact frozen schema")
    if not isinstance(limits, dict) or set(limits) != set(_LIMITS):
        raise ValueError("limits must have the exact frozen schema")
    d = {key: _finite_number(diagnostics[key], key) for key in _DIAGNOSTICS}
    threshold = {key: _finite_number(limits[key], key) for key in _LIMITS}
    penalties = (
        4.0 * max(0.0, d["rank_abs_error"] - threshold["rank_limit"]),
        4.0 * max(0.0, d["inner_coverage_error"] - threshold["inner_limit"]),
        4.0 * max(0.0, d["boundary_error"] - threshold["boundary_limit"]),
        2.0 * max(0.0, d["iiee_ratio"] - 1.02),
        2.0 * max(0.0, d["edge_ratio"] - 1.02),
        2.0 * max(0.0, d["variogram_ratio"] - 1.02),
    )
    return d["fair_crps"] + sum(penalties)


def fit_ridge(training_descriptors, training_losses):
    if np is None:
        raise RuntimeError("numpy is required for the ridge prototype")
    matrix = np.asarray(training_descriptors)
    response = np.asarray(training_losses)
    if matrix.dtype.kind not in "iuf" or matrix.ndim != 2 or matrix.shape[1] != PREDICTORS:
        raise ValueError("training_descriptors must be a numeric six-column matrix")
    if response.dtype.kind not in "iuf" or response.ndim != 1 or response.shape[0] != matrix.shape[0]:
        raise ValueError("training_losses must contain one numeric value per row")
    matrix = matrix.astype(float, copy=False)
    response = response.astype(float, copy=False)
    if matrix.shape[0] == 0 or not np.isfinite(matrix).all() or not np.isfinite(response).all():
        raise ValueError("training inputs must be non-empty and finite")
    center = np.mean(matrix, axis=0)
    scale = np.std(matrix, axis=0, ddof=0)
    if not np.isfinite(scale).all() or np.any(scale == 0.0):
        raise ValueError("every training descriptor must have finite non-zero scale")
    standardized = (matrix - center) / scale
    intercept = float(np.mean(response))
    left, singular_values, right_t = np.linalg.svd(standardized, full_matrices=False)
    largest = singular_values.max(initial=0.0)
    retained = singular_values > SVD_RELATIVE_CUTOFF * largest
    shrinkage = np.zeros_like(singular_values)
    shrinkage[retained] = (
        singular_values[retained] /
        (singular_values[retained] * singular_values[retained] + RIDGE)
    )
    coefficients = right_t.T @ (shrinkage * (left.T @ (response - intercept)))
    if not np.isfinite(coefficients).all():
        raise ValueError("ridge coefficients must be finite")
    return {
        "training_mean": center,
        "training_scale": scale,
        "intercept": intercept,
        "coefficients": coefficients,
    }


def predict_loss(model, descriptor):
    if np is None:
        raise RuntimeError("numpy is required for the ridge prototype")
    vector = np.asarray(descriptor)
    if vector.dtype.kind not in "iuf" or vector.shape != (PREDICTORS,):
        raise ValueError("descriptor must contain six numeric values")
    try:
        center = np.asarray(model["training_mean"], dtype=float)
        scale = np.asarray(model["training_scale"], dtype=float)
        coefficients = np.asarray(model["coefficients"], dtype=float)
        intercept = float(model["intercept"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("model is incomplete or non-numeric") from error
    vector = vector.astype(float, copy=False)
    if any(item.shape != (PREDICTORS,) for item in (center, scale, coefficients)):
        raise ValueError("model vectors must contain six values")
    if not all(np.isfinite(item).all() for item in (vector, center, scale, coefficients)):
        raise ValueError("prediction inputs must be finite")
    if np.any(scale <= 0.0) or not math.isfinite(intercept):
        raise ValueError("model scale and intercept must be valid")
    prediction = intercept + np.dot((vector - center) / scale, coefficients)
    if not math.isfinite(float(prediction)):
        raise ValueError("predicted loss must be finite")
    return float(prediction)


def select_action(raw_predicted_loss, projected_predicted_loss):
    raw = _finite_number(raw_predicted_loss, "raw_predicted_loss")
    projected = _finite_number(projected_predicted_loss, "projected_predicted_loss")
    margin = raw - projected
    return {
        "selected_action": "projected_spread" if margin >= ACTION_MARGIN else "raw",
        "action_margin": margin,
    }
