"""Dependency-light oracle for frozen score-aware raw-scenario reweighting."""

from __future__ import annotations

import math

try:
    import numpy as np
except ModuleNotFoundError:  # Selection remains independently auditable.
    np = None


MEMBERS = 10
PREDICTORS = 13
RIDGE = 1.0
SVD_RELATIVE_CUTOFF = 1e-12
CASES = 40
HOLDOUT_SIZE = 8
PURGE = 3


def build_purged_folds(case_ids):
    """Bind positional contiguous holdouts and purge membership to exact identifiers."""
    identifiers = tuple(case_ids)
    if (
        len(identifiers) != CASES
        or any(not isinstance(case_id, str) or not case_id for case_id in identifiers)
        or len(set(identifiers)) != CASES
    ):
        raise ValueError("case_ids must contain forty unique non-empty strings")
    folds = []
    for fold_index, start in enumerate(range(0, CASES, HOLDOUT_SIZE)):
        stop = start + HOLDOUT_SIZE
        excluded_start = max(0, start - PURGE)
        excluded_stop = min(CASES, stop + PURGE)
        folds.append(
            {
                "fold": fold_index,
                "holdout_case_ids": identifiers[start:stop],
                "training_case_ids": tuple(
                    case_id
                    for index, case_id in enumerate(identifiers)
                    if not excluded_start <= index < excluded_stop
                ),
            }
        )
    return tuple(folds)


def _finite_matrix(values, *, columns, name):
    if np is None:
        raise RuntimeError("numpy is required for the ridge oracle")
    array = np.asarray(values)
    if array.dtype.kind not in "iuf" or array.ndim != 2 or array.shape[1] != columns:
        raise ValueError(f"{name} must be a numeric matrix with {columns} columns")
    array = array.astype(float, copy=False)
    if array.shape[0] == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be non-empty and finite")
    return array


def fit_fold_ridge(training_predictors, training_targets):
    """Fit the exact training-only standardization and unpenalized-intercept ridge."""
    design = _finite_matrix(
        training_predictors, columns=PREDICTORS, name="training_predictors"
    )
    targets = np.asarray(training_targets)
    if targets.dtype.kind not in "iuf" or targets.ndim != 1 or len(targets) != len(design):
        raise ValueError("training_targets must be one numeric value per design row")
    targets = targets.astype(float, copy=False)
    if not np.isfinite(targets).all():
        raise ValueError("training_targets must be finite")

    mean = design.mean(axis=0)
    scale = design.std(axis=0, ddof=0)
    if not np.isfinite(scale).all() or np.any(scale == 0.0):
        raise ValueError("every training predictor must have finite non-zero scale")
    standardized = (design - mean) / scale
    intercept = float(targets.mean())
    centered_targets = targets - intercept
    u, singular, vt = np.linalg.svd(standardized, full_matrices=False)
    if not (np.isfinite(u).all() and np.isfinite(singular).all() and np.isfinite(vt).all()):
        raise ValueError("SVD must be finite")
    cutoff = SVD_RELATIVE_CUTOFF * singular.max(initial=0.0)
    factors = np.where(singular > cutoff, singular / (singular**2 + RIDGE), 0.0)
    coefficients = vt.T @ (factors * (u.T @ centered_targets))
    if not np.isfinite(coefficients).all() or not math.isfinite(intercept):
        raise ValueError("fitted ridge model must be finite")
    return {
        "training_predictor_mean": mean,
        "training_predictor_scale": scale,
        "intercept": intercept,
        "coefficients": coefficients,
    }


def predict_member_risks(model, heldout_predictors):
    """Predict ten held-out raw-member risks with frozen training statistics."""
    design = _finite_matrix(
        heldout_predictors, columns=PREDICTORS, name="heldout_predictors"
    )
    if len(design) != MEMBERS:
        raise ValueError("heldout_predictors must contain exactly ten members")
    try:
        mean = np.asarray(model["training_predictor_mean"], dtype=float)
        scale = np.asarray(model["training_predictor_scale"], dtype=float)
        coefficients = np.asarray(model["coefficients"], dtype=float)
        intercept = float(model["intercept"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("model is incomplete or non-numeric") from error
    if any(value.shape != (PREDICTORS,) for value in (mean, scale, coefficients)):
        raise ValueError("model vectors must have thirteen entries")
    if not all(np.isfinite(value).all() for value in (mean, scale, coefficients)):
        raise ValueError("model vectors must be finite")
    if np.any(scale <= 0.0) or not math.isfinite(intercept):
        raise ValueError("model scale and intercept must be valid")
    risks = intercept + ((design - mean) / scale) @ coefficients
    if not np.isfinite(risks).all():
        raise ValueError("predicted risks must be finite")
    return risks


def select_raw_scenarios(predicted_risks):
    """Apply stable softmax weights and risk-ordered systematic inverse CDF."""
    risks = list(predicted_risks)
    if len(risks) != MEMBERS or any(
        isinstance(value, bool) or not isinstance(value, (int, float)) for value in risks
    ):
        raise ValueError("predicted_risks must contain ten numeric values")
    risks = [float(value) for value in risks]
    if not all(math.isfinite(value) for value in risks):
        raise ValueError("predicted_risks must be finite")
    minimum = min(risks)
    unnormalized = [math.exp(-(value - minimum)) for value in risks]
    total = sum(unnormalized)
    weights = [value / total for value in unnormalized]
    if not all(math.isfinite(value) for value in weights) or abs(sum(weights) - 1.0) > 1e-12:
        raise ValueError("risk weights must be finite and sum to one")

    ordered_indices = sorted(range(MEMBERS), key=lambda index: (risks[index], index))
    cumulative = []
    running = 0.0
    for index in ordered_indices:
        running += weights[index]
        cumulative.append(running)
    selected = []
    for position in ((index + 0.5) / MEMBERS for index in range(MEMBERS)):
        ordered_position = next(
            (index for index, bound in enumerate(cumulative) if position <= bound), MEMBERS
        )
        if ordered_position >= MEMBERS:
            raise AssertionError("inverse CDF left a probability position unassigned")
        selected.append(ordered_indices[ordered_position])
    multiplicities = [selected.count(index) for index in range(MEMBERS)]
    frequencies = [count / MEMBERS for count in multiplicities]
    return {
        "predicted_risks": risks,
        "normalized_weights": weights,
        "risk_ordered_raw_member_indices": ordered_indices,
        "source_raw_member_indices": selected,
        "source_multiplicities": multiplicities,
        "unique_selected_raw_members": sum(count > 0 for count in multiplicities),
        "effective_sample_size": 1.0 / sum(value**2 for value in frequencies),
    }


if __name__ == "__main__":
    result = select_raw_scenarios([0.0] * MEMBERS)
    assert result["source_raw_member_indices"] == list(range(MEMBERS))
    assert abs(result["effective_sample_size"] - MEMBERS) < 1e-12
    print("score_aware_raw_reweighting_reference=PASS")
