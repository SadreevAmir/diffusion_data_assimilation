#!/usr/bin/env python3
"""Pure runner surface for frozen score-aware raw-scenario reweighting.

No controller, filesystem, metric, or project-data loading lives here.  A future
server adapter supplies sealed fold inputs and evaluates the unchanged gate;
this module owns the deterministic fit, selection, and exact-copy invariants.
"""

from __future__ import annotations

import math

try:
    from .score_aware_raw_reweighting_reference import (
        MEMBERS,
        PREDICTORS,
        build_purged_folds,
        fit_fold_ridge,
        predict_member_risks,
        select_raw_scenarios,
    )
except ImportError:  # pragma: no cover - direct script/admission path loading
    from score_aware_raw_reweighting_reference import (
        MEMBERS,
        PREDICTORS,
        build_purged_folds,
        fit_fold_ridge,
        predict_member_risks,
        select_raw_scenarios,
    )


SOURCE_EXPERIMENT = "joint_full_condition_validation_2022"
ARTIFACT_POLICY = "summary_only"


def validate_interface(parameters, artifact_policy):
    """Reject every runtime degree of freedom excluded by the frozen contract."""
    if parameters != {"source_experiment": SOURCE_EXPERIMENT}:
        raise ValueError("runner accepts only the frozen source_experiment")
    if artifact_policy != ARTIFACT_POLICY:
        raise ValueError("runner accepts only summary_only")


def _checked_members(raw_members):
    try:
        import numpy as np
    except ModuleNotFoundError as error:  # pragma: no cover
        raise RuntimeError("numpy is required for candidate construction") from error
    members = np.asarray(raw_members)
    if members.dtype.kind not in "iuf" or members.ndim != 3 or len(members) != MEMBERS:
        raise ValueError("raw_members must be ten numeric two-dimensional fields")
    members = members.astype(float, copy=False)
    if not np.isfinite(members).all():
        raise ValueError("raw_members must be finite")
    return members


def _mask_equal(candidate, source):
    import numpy as np

    return {
        "exact_zero": bool(np.array_equal(candidate == 0.0, source == 0.0)),
        "positive_ice": bool(np.array_equal(candidate > 0.0, source > 0.0)),
        "established_ice": bool(np.array_equal(candidate >= 0.15, source >= 0.15)),
        "exact_one": bool(np.array_equal(candidate == 1.0, source == 1.0)),
        "near_one": bool(np.array_equal(candidate >= 0.999, source >= 0.999)),
    }


def construct_case(training_predictors, training_targets, heldout_predictors, raw_members):
    """Fit one frozen fold model and copy the selected complete raw scenarios."""
    import numpy as np

    members = _checked_members(raw_members)
    model = fit_fold_ridge(training_predictors, training_targets)
    risks = predict_member_risks(model, heldout_predictors)
    selection = select_raw_scenarios(risks.tolist())
    indices = selection["source_raw_member_indices"]
    candidate = members[indices].copy()
    exact_copy = all(np.array_equal(candidate[j], members[index]) for j, index in enumerate(indices))
    masks = [_mask_equal(candidate[j], members[index]) for j, index in enumerate(indices)]
    if not exact_copy or not all(all(record.values()) for record in masks):
        raise ValueError("source-copy or member-mask invariant failed")
    return candidate, {
        **selection,
        "predicted_risks": risks.tolist(),
        "training_predictor_mean": model["training_predictor_mean"].tolist(),
        "training_predictor_scale": model["training_predictor_scale"].tolist(),
        "intercept": float(model["intercept"]),
        "coefficients": model["coefficients"].tolist(),
        "exact_source_copy": exact_copy,
        "member_mask_invariants": masks,
    }


def synthetic_result():
    """Exercise the full construction with deterministic synthetic arrays only."""
    import numpy as np

    rows = np.arange(1, 41, dtype=float)[:, None]
    columns = np.arange(1, PREDICTORS + 1, dtype=float)[None, :]
    training = np.sin(rows / columns) + rows * columns / 1000.0
    targets = 0.2 + training @ np.linspace(-0.1, 0.1, PREDICTORS)
    heldout = training[:MEMBERS] + np.linspace(0.0, 0.2, MEMBERS)[:, None]
    raw = np.arange(MEMBERS * 5 * 6, dtype=float).reshape(MEMBERS, 5, 6)
    raw /= raw.max()
    candidate, diagnostics = construct_case(training, targets, heldout, raw)
    case_ids = tuple(f"case-{index:02d}" for index in range(40))
    folds = build_purged_folds(case_ids)
    return {
        "schema_version": "score-aware-raw-reweighting-synthetic-v1",
        "status": "PASS",
        "decision_bearing": False,
        "project_data_metrics_emitted": False,
        "fold_count": len(folds),
        "candidate_members": len(candidate),
        "construction": diagnostics,
    }


if __name__ == "__main__":
    result = synthetic_result()
    assert result["status"] == "PASS" and math.isclose(
        sum(result["construction"]["normalized_weights"]), 1.0, abs_tol=1e-12
    )
    print("score_aware_raw_reweighting_runner=PASS")
