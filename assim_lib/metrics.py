from __future__ import annotations

import numpy as np


def weighted_mean(values: np.ndarray, weights: np.ndarray | None = None) -> float:
    values = np.asarray(values, dtype=np.float64)
    if weights is None:
        return float(np.nanmean(values))
    weights = np.broadcast_to(np.asarray(weights, dtype=np.float64), values.shape)
    valid = np.isfinite(values) & (weights > 0)
    denominator = weights[valid].sum()
    if denominator <= 0:
        return float("nan")
    return float((values[valid] * weights[valid]).sum() / denominator)


def ensemble_crps(ensemble: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Return pointwise CRPS for an ensemble with shape ``(members, ...)``."""
    ensemble = np.asarray(ensemble, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if ensemble.ndim < 1 or ensemble.shape[0] == 0:
        raise ValueError("ensemble must contain at least one member")
    if ensemble.shape[1:] != truth.shape:
        raise ValueError(f"ensemble event shape {ensemble.shape[1:]} does not match truth {truth.shape}")
    members = ensemble.shape[0]
    error = np.mean(np.abs(ensemble - truth[None, ...]), axis=0)
    sorted_ensemble = np.sort(ensemble, axis=0)
    coefficient = (2 * np.arange(1, members + 1, dtype=np.float64) - members - 1).reshape(
        (members,) + (1,) * (ensemble.ndim - 1)
    )
    pairwise = np.sum(coefficient * sorted_ensemble, axis=0) / (members * members)
    return error - pairwise


def interval_coverage(
    ensemble: np.ndarray,
    truth: np.ndarray,
    levels: tuple[float, ...] = (0.5, 0.8, 0.9, 0.95),
) -> dict[str, np.ndarray]:
    ensemble = np.asarray(ensemble, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if ensemble.shape[1:] != truth.shape:
        raise ValueError(f"ensemble event shape {ensemble.shape[1:]} does not match truth {truth.shape}")
    result: dict[str, np.ndarray] = {}
    for level in levels:
        if not 0.0 < level < 1.0:
            raise ValueError(f"coverage level must be within (0, 1), got {level}")
        tail = (1.0 - level) / 2.0
        lower = np.quantile(ensemble, tail, axis=0)
        upper = np.quantile(ensemble, 1.0 - tail, axis=0)
        result[f"{level:g}"] = (truth >= lower) & (truth <= upper)
    return result


def rank_histogram(
    ensemble: np.ndarray,
    truth: np.ndarray,
    *,
    seed: int = 0,
    weights_mask: np.ndarray | None = None,
) -> np.ndarray:
    ensemble = np.asarray(ensemble, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if ensemble.ndim < 2 or ensemble.shape[1:] != truth.shape:
        raise ValueError(f"ensemble event shape {ensemble.shape[1:]} does not match truth {truth.shape}")
    rng = np.random.default_rng(seed)
    less = np.sum(ensemble < truth[None, ...], axis=0)
    ties = np.sum(ensemble == truth[None, ...], axis=0)
    ranks = less + rng.integers(0, ties + 1)
    if weights_mask is not None:
        ranks = ranks[np.asarray(weights_mask, dtype=bool)]
    return np.bincount(
        ranks.reshape(-1).astype(np.int64),
        minlength=ensemble.shape[0] + 1,
    )


def spread_skill(
    ensemble: np.ndarray,
    truth: np.ndarray,
    weights: np.ndarray | None = None,
) -> dict[str, float]:
    ensemble = np.asarray(ensemble, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    ensemble_mean = ensemble.mean(axis=0)
    variance = ensemble.var(axis=0, ddof=1 if ensemble.shape[0] > 1 else 0)
    spread = np.sqrt(weighted_mean(variance, weights))
    skill = np.sqrt(weighted_mean((ensemble_mean - truth) ** 2, weights))
    return {
        "spread": float(spread),
        "skill_rmse": float(skill),
        "spread_skill_ratio": float(spread / skill) if skill > 0 else float("inf"),
    }
