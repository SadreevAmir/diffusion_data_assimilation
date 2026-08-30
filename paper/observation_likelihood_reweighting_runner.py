#!/usr/bin/env python3
"""Pure copy-only construction for frozen observation-likelihood reweighting."""

from __future__ import annotations

import math

CASES, MEMBERS, FOLDS, HOLDOUT, PURGE = 40, 10, 5, 8, 3
SOURCE_EXPERIMENT = "joint_full_condition_validation_2022"
ARTIFACT_POLICY = "summary_only"


def validate_interface(parameters, artifact_policy):
    if parameters != {"source_experiment": SOURCE_EXPERIMENT}:
        raise ValueError("runner accepts only the frozen source_experiment")
    if artifact_policy != ARTIFACT_POLICY:
        raise ValueError("runner accepts only summary_only")


def purged_folds():
    result = []
    for fold in range(FOLDS):
        start, stop = fold * HOLDOUT, (fold + 1) * HOLDOUT
        excluded_start, excluded_stop = max(0, start - PURGE), min(CASES, stop + PURGE)
        result.append((tuple(range(start, stop)), tuple(i for i in range(CASES) if not excluded_start <= i < excluded_stop)))
    return tuple(result)


def training_bandwidth(innovations):
    values = sorted(abs(float(value)) for value in innovations)
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("training innovations must be finite and non-empty")
    middle = len(values) // 2
    median = values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2
    bandwidth = 1.4826 * median
    if not math.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("training bandwidth must be finite and positive")
    return bandwidth


def likelihood_weights(masked_reconstructions, observations, bandwidth):
    if not math.isfinite(float(bandwidth)) or bandwidth <= 0:
        raise ValueError("bandwidth must be finite and positive")
    observations = [float(value) for value in observations]
    if not observations or any(not math.isfinite(value) for value in observations):
        raise ValueError("observations must be finite and non-empty")
    rows = [list(map(float, row)) for row in masked_reconstructions]
    if len(rows) != MEMBERS or any(len(row) != len(observations) for row in rows):
        raise ValueError("masked reconstructions have wrong shape")
    if any(not math.isfinite(value) for row in rows for value in row):
        raise ValueError("masked reconstructions must be finite")
    scores = [sum(-0.5 * ((y - x) / bandwidth) ** 2 - math.log(bandwidth) - 0.5 * math.log(2 * math.pi) for x, y in zip(row, observations)) for row in rows]
    anchor = max(scores)
    masses = [math.exp(score - anchor) for score in scores]
    total = sum(masses)
    weights = [mass / total for mass in masses]
    ess = 1.0 / sum(weight * weight for weight in weights)
    if abs(sum(weights) - 1.0) > 1e-12 or not math.isfinite(ess) or ess <= 1.0:
        raise ValueError("invalid likelihood weights")
    return weights, ess


def construct_case(raw_members, masked_reconstructions, observations, bandwidth):
    if not isinstance(raw_members, list) or len(raw_members) != MEMBERS:
        raise ValueError("raw ensemble must contain exactly ten members")
    weights, ess = likelihood_weights(masked_reconstructions, observations, bandwidth)
    return list(raw_members), {"weights": weights, "effective_sample_size": ess, "exact_source_copy": True}
