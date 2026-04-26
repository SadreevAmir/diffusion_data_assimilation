from __future__ import annotations

import numpy as np


def _valid_weights(valid_mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    if valid_mask is None:
        return np.ones(shape, dtype=np.float64)
    weights = np.asarray(valid_mask, dtype=np.float64)
    if weights.ndim == 3:
        weights = weights[0]
    if weights.shape != shape:
        raise ValueError(f"weights shape {weights.shape} does not match {shape}")
    return weights


def weighted_mean(values: np.ndarray, weights: np.ndarray | None = None) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if weights is None:
        return float(np.nanmean(arr))
    w = np.broadcast_to(np.asarray(weights, dtype=np.float64), arr.shape)
    ok = np.isfinite(arr) & (w > 0)
    denom = w[ok].sum()
    return float((arr[ok] * w[ok]).sum() / denom) if denom > 0 else float("nan")


def ensemble_crps(ensemble: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Pointwise CRPS for ensemble shape (M, ...), using sorted O(M log M) formula."""
    ens = np.asarray(ensemble, dtype=np.float64)
    y = np.asarray(truth, dtype=np.float64)
    if ens.ndim < 1:
        raise ValueError("ensemble must have a member axis")
    m = ens.shape[0]
    if m == 0:
        raise ValueError("ensemble is empty")
    term1 = np.mean(np.abs(ens - y[None, ...]), axis=0)
    xs = np.sort(ens, axis=0)
    coeff = (2 * np.arange(1, m + 1, dtype=np.float64) - m - 1).reshape((m,) + (1,) * (ens.ndim - 1))
    pair_term = np.sum(coeff * xs, axis=0) / (m * m)
    return term1 - pair_term


def rank_histogram(
    ensemble: np.ndarray,
    truth: np.ndarray,
    seed: int = 0,
    weights_mask: np.ndarray | None = None,
) -> np.ndarray:
    ens = np.asarray(ensemble, dtype=np.float64)
    y = np.asarray(truth, dtype=np.float64)
    if ens.ndim < 2:
        raise ValueError("Expected ensemble shape (M, ...)")
    if ens.shape[1:] != y.shape:
        raise ValueError(f"ensemble event shape {ens.shape[1:]} does not match truth {y.shape}")
    rng = np.random.default_rng(seed)
    less = np.sum(ens < y[None, ...], axis=0)
    ties = np.sum(ens == y[None, ...], axis=0)
    ranks = less + rng.integers(0, ties + 1)
    if weights_mask is not None:
        ranks = ranks[np.asarray(weights_mask, dtype=bool)]
    return np.bincount(ranks.reshape(-1).astype(np.int64), minlength=ens.shape[0] + 1)


def interval_coverage(ensemble: np.ndarray, truth: np.ndarray, levels=(0.5, 0.8, 0.9, 0.95)) -> dict[str, np.ndarray]:
    ens = np.asarray(ensemble, dtype=np.float64)
    y = np.asarray(truth, dtype=np.float64)
    out: dict[str, np.ndarray] = {}
    for level in levels:
        lo_q = (1.0 - level) / 2.0
        hi_q = 1.0 - lo_q
        lo = np.quantile(ens, lo_q, axis=0)
        hi = np.quantile(ens, hi_q, axis=0)
        out[f"{level:g}"] = (y >= lo) & (y <= hi)
    return out


def spread_skill(ensemble: np.ndarray, truth: np.ndarray, weights: np.ndarray | None = None) -> dict[str, float]:
    ens = np.asarray(ensemble, dtype=np.float64)
    y = np.asarray(truth, dtype=np.float64)
    mean = ens.mean(axis=0)
    variance = ens.var(axis=0, ddof=1 if ens.shape[0] > 1 else 0)
    sq_error = (mean - y) ** 2
    spread = np.sqrt(weighted_mean(variance, weights))
    skill = np.sqrt(weighted_mean(sq_error, weights))
    return {
        "spread": float(spread),
        "skill_rmse": float(skill),
        "spread_skill_ratio": float(spread / skill) if skill > 0 else float("inf"),
    }


def binned_spread_skill(
    ensemble: np.ndarray,
    truth: np.ndarray,
    n_bins: int = 10,
    weights_mask: np.ndarray | None = None,
) -> list[dict[str, float]]:
    ens = np.asarray(ensemble, dtype=np.float64)
    y = np.asarray(truth, dtype=np.float64)
    mean = ens.mean(axis=0)
    spread = np.sqrt(ens.var(axis=0, ddof=1 if ens.shape[0] > 1 else 0))
    error2 = (mean - y) ** 2
    if weights_mask is not None:
        selector = np.asarray(weights_mask, dtype=bool)
        spread = spread[selector]
        error2 = error2[selector]
    spread = spread.reshape(-1)
    error2 = error2.reshape(-1)
    if spread.size == 0:
        return []
    edges = np.quantile(spread, np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(edges)
    if edges.size < 2:
        return [{"bin_lo": float(spread.min()), "bin_hi": float(spread.max()), "count": float(spread.size), "mean_spread": float(spread.mean()), "rmse": float(np.sqrt(error2.mean()))}]
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (spread >= lo) & (spread <= hi if hi == edges[-1] else spread < hi)
        if not np.any(sel):
            continue
        rows.append({
            "bin_lo": float(lo),
            "bin_hi": float(hi),
            "count": float(sel.sum()),
            "mean_spread": float(spread[sel].mean()),
            "rmse": float(np.sqrt(error2[sel].mean())),
        })
    return rows


def energy_score(ensemble: np.ndarray, truth: np.ndarray, weights: np.ndarray | None = None) -> float:
    ens = np.asarray(ensemble, dtype=np.float64).reshape(ensemble.shape[0], -1)
    y = np.asarray(truth, dtype=np.float64).reshape(-1)
    if weights is not None:
        w = np.sqrt(np.broadcast_to(np.asarray(weights, dtype=np.float64), truth.shape).reshape(-1))
        ens = ens * w[None, :]
        y = y * w
    term1 = np.linalg.norm(ens - y[None, :], axis=1).mean()
    diffs = ens[:, None, :] - ens[None, :, :]
    term2 = 0.5 * np.linalg.norm(diffs, axis=2).mean()
    return float(term1 - term2)


def ice_summaries(ensemble_mean: np.ndarray, truth: np.ndarray, valid_mask: np.ndarray | None = None, conc_index: int = 0, thick_index: int = 1) -> dict[str, float]:
    mask = _valid_weights(valid_mask, truth.shape[-2:]) > 0
    out: dict[str, float] = {}
    if truth.shape[0] > conc_index:
        pred_ice = ensemble_mean[conc_index] >= 0.15
        true_ice = truth[conc_index] >= 0.15
        out["ice_extent_error_cells"] = float((pred_ice & mask).sum() - (true_ice & mask).sum())
        out["ice_area_error_sum"] = float(((ensemble_mean[conc_index] - truth[conc_index]) * mask).sum())
        intersection = (pred_ice & true_ice & mask).sum()
        union = ((pred_ice | true_ice) & mask).sum()
        out["ice_edge_proxy_iou_error"] = float(1.0 - (intersection / max(1, union)))
    if truth.shape[0] > thick_index:
        out["integrated_thickness_error"] = float(((ensemble_mean[thick_index] - truth[thick_index]) * mask).sum())
    return out
