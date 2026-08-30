"""Dependency-light oracle for the frozen purged casewise safety selector."""

from __future__ import annotations

import math

try:
    import numpy as np
except ModuleNotFoundError:  # The contract remains inspectable without NumPy.
    np = None


CASES = 40
HOLDOUT_SIZE = 8
PURGE = 3
PREDICTORS = 6
RIDGE = 1.0
SVD_RELATIVE_CUTOFF = 1e-12
ACTION_MARGIN = 0.002
ACTIONS = ("raw", "projected_spread")


def build_purged_folds(case_ids):
    identifiers = tuple(case_ids)
    if (len(identifiers) != CASES or len(set(identifiers)) != CASES or
            any(not isinstance(value, str) or not value for value in identifiers)):
        raise ValueError("case_ids must contain forty unique non-empty strings")
    folds = []
    for fold, start in enumerate(range(0, CASES, HOLDOUT_SIZE)):
        stop = start + HOLDOUT_SIZE
        excluded_start = max(0, start - PURGE)
        excluded_stop = min(CASES, stop + PURGE)
        folds.append({
            "fold": fold,
            "holdout_case_ids": identifiers[start:stop],
            "training_case_ids": tuple(
                value for index, value in enumerate(identifiers)
                if not excluded_start <= index < excluded_stop
            ),
        })
    return tuple(folds)


def scalar_loss(diagnostics, limits):
    """Evaluate the literal frozen action loss from case-level diagnostics."""
    required = {
        "fair_crps", "rank_abs_error", "inner_coverage_error", "boundary_error",
        "iiee_ratio", "edge_ratio", "variogram_ratio",
    }
    limit_keys = {"rank_limit", "inner_limit", "boundary_limit"}
    if not isinstance(diagnostics, dict) or set(diagnostics) != required:
        raise ValueError("diagnostics must have the exact frozen schema")
    if not isinstance(limits, dict) or set(limits) != limit_keys:
        raise ValueError("limits must have the exact frozen schema")
    values = tuple(diagnostics.values()) + tuple(limits.values())
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or
           not math.isfinite(float(value)) for value in values):
        raise ValueError("diagnostics and limits must be finite numeric values")
    return (
        float(diagnostics["fair_crps"])
        + 4.0 * max(0.0, diagnostics["rank_abs_error"] - limits["rank_limit"])
        + 4.0 * max(0.0, diagnostics["inner_coverage_error"] - limits["inner_limit"])
        + 4.0 * max(0.0, diagnostics["boundary_error"] - limits["boundary_limit"])
        + 2.0 * max(0.0, diagnostics["iiee_ratio"] - 1.02)
        + 2.0 * max(0.0, diagnostics["edge_ratio"] - 1.02)
        + 2.0 * max(0.0, diagnostics["variogram_ratio"] - 1.02)
    )


def fit_ridge(training_descriptors, training_losses):
    """Fit training-only standardization and unpenalized-intercept ridge."""
    if np is None:
        raise RuntimeError("numpy is required for the ridge oracle")
    design = np.asarray(training_descriptors)
    targets = np.asarray(training_losses)
    if design.dtype.kind not in "iuf" or design.ndim != 2 or design.shape[1] != PREDICTORS:
        raise ValueError("training_descriptors must be a numeric six-column matrix")
    if targets.dtype.kind not in "iuf" or targets.ndim != 1 or len(targets) != len(design):
        raise ValueError("training_losses must contain one numeric value per row")
    design = design.astype(float, copy=False)
    targets = targets.astype(float, copy=False)
    if len(design) == 0 or not np.isfinite(design).all() or not np.isfinite(targets).all():
        raise ValueError("training inputs must be non-empty and finite")
    mean = design.mean(axis=0)
    scale = design.std(axis=0, ddof=0)
    if not np.isfinite(scale).all() or np.any(scale == 0.0):
        raise ValueError("every training descriptor must have finite non-zero scale")
    standardized = (design - mean) / scale
    intercept = float(targets.mean())
    u, singular, vt = np.linalg.svd(standardized, full_matrices=False)
    cutoff = SVD_RELATIVE_CUTOFF * singular.max(initial=0.0)
    factors = np.where(singular > cutoff, singular / (singular**2 + RIDGE), 0.0)
    coefficients = vt.T @ (factors * (u.T @ (targets - intercept)))
    if not np.isfinite(coefficients).all():
        raise ValueError("ridge coefficients must be finite")
    return {"training_mean": mean, "training_scale": scale,
            "intercept": intercept, "coefficients": coefficients}


def predict_loss(model, descriptor):
    if np is None:
        raise RuntimeError("numpy is required for the ridge oracle")
    vector = np.asarray(descriptor)
    if vector.dtype.kind not in "iuf" or vector.shape != (PREDICTORS,):
        raise ValueError("descriptor must contain six numeric values")
    vector = vector.astype(float, copy=False)
    try:
        mean = np.asarray(model["training_mean"], dtype=float)
        scale = np.asarray(model["training_scale"], dtype=float)
        coefficients = np.asarray(model["coefficients"], dtype=float)
        intercept = float(model["intercept"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("model is incomplete or non-numeric") from error
    if any(value.shape != (PREDICTORS,) for value in (mean, scale, coefficients)):
        raise ValueError("model vectors must contain six values")
    if not all(np.isfinite(value).all() for value in (vector, mean, scale, coefficients)):
        raise ValueError("prediction inputs must be finite")
    if np.any(scale <= 0.0) or not math.isfinite(intercept):
        raise ValueError("model scale and intercept must be valid")
    result = intercept + ((vector - mean) / scale) @ coefficients
    if not math.isfinite(float(result)):
        raise ValueError("predicted loss must be finite")
    return float(result)


def select_action(raw_predicted_loss, projected_predicted_loss):
    values = (raw_predicted_loss, projected_predicted_loss)
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or
           not math.isfinite(float(value)) for value in values):
        raise ValueError("predicted losses must be finite numeric values")
    margin = float(raw_predicted_loss) - float(projected_predicted_loss)
    return {
        "selected_action": "projected_spread" if margin >= ACTION_MARGIN else "raw",
        "action_margin": margin,
    }
