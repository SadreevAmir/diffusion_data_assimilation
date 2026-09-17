#!/usr/bin/env python3
"""Null distribution of gate statistics for an exchangeable (perfectly calibrated) ensemble.

For each train date (2020, hour 23) the truth SIC is encoded into the atom-aware
latent (water -> -gamma*distance, cap -> censoring).  M+1 exchangeable fields are
produced by iid smooth random warps (displacement fields) plus iid smooth
level-set perturbations of the latent, then decoded (atoms preserved).  Copy 0
is declared "truth", copies 1..M the ensemble.  By exchangeability the ensemble
is calibrated by construction, so the statistics below are what a perfect
M-member ensemble attains on this archive: rank TV (pooled and by ensemble-mean
class), 80% coverage, spread/RMSE, exact-zero and cap mass errors, IIEE of the
mean, member variogram ratios.  Gate tolerances must not be stricter than the
q95 of these null distributions.  Train-only; no model involved.
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates

ROOT = Path("/mnt/sciml/data_assimilation")
PREDS = ROOT / "da_arctic_2015-2024_v0.1/preds"
WATER = ~np.load(ROOT / "land_mask.npy")
CAP = 0.9970703125
GAMMA, R_MAX, DELTA_CAP = 0.02, 40.0, 0.01


def sic(d: date) -> np.ndarray:
    a = np.asarray(np.load(PREDS / f"ocean+atmosphere_24_{d.isoformat()}.npy", mmap_mode="r")[0, 23], dtype=np.float32)
    a = np.where(np.isnan(a), 0.0, a)
    return np.where(WATER, a, 0.0).astype(np.float64)


def enc(a):
    z = a.copy(); ice = (a > 0) & WATER
    d = distance_transform_edt(~ice) if ice.any() else np.full(a.shape, R_MAX)
    z[~ice] = -GAMMA * np.minimum(d[~ice], R_MAX)
    z[a >= CAP] = CAP + DELTA_CAP
    return np.where(WATER, z, 0.0)


def dec(z):
    return np.where(WATER, np.clip(z, 0.0, CAP), 0.0)


def smooth_field(rng, shape, L):
    f = gaussian_filter(rng.standard_normal(shape), L, mode="reflect")
    return f / max(f.std(), 1e-9)


def warp(z, rng, amp_px, L, level_amp):
    h, w = z.shape
    dy = amp_px * smooth_field(rng, z.shape, L); dx = amp_px * smooth_field(rng, z.shape, L)
    yy, xx = np.mgrid[0:h, 0:w]
    zw = map_coordinates(z, [yy + dy, xx + dx], order=1, mode="nearest")
    zw = zw + level_amp * smooth_field(rng, z.shape, L)
    return np.where(WATER, zw, 0.0)


def fair_crps(m, y):
    M = m.shape[0]; t1 = np.abs(m - y[None]).mean(0); t2 = np.zeros_like(y)
    for i in range(M): t2 += np.abs(m[i][None] - m).sum(0)
    return float((t1 - t2 / (2 * M * (M - 1))).mean())


def rank_tv(m, y, rng):
    M = m.shape[0]; below = (m < y[None]).sum(0); ties = (m == y[None]).sum(0)
    r = below + np.floor(rng.random(y.shape) * (ties + 1)).astype(int)
    h = np.bincount(r, minlength=M + 1).astype(float); h /= h.sum()
    return float(0.5 * np.abs(h - 1 / (M + 1)).sum())


def variogram(f, lag, mask):
    a = (f[:, lag:] - f[:, :-lag]) ** 2; m = mask[:, lag:] & mask[:, :-lag]
    b = (f[lag:, :] - f[:-lag, :]) ** 2; mv = mask[lag:, :] & mask[:-lag, :]
    return float((a[m].sum() + b[mv].sum()) / max(m.sum() + mv.sum(), 1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--members", type=int, default=10)
    p.add_argument("--amp-px", type=float, default=4.0)
    p.add_argument("--corr-px", type=float, default=20.0)
    p.add_argument("--level-amp", type=float, default=0.08)
    p.add_argument("--seed", type=int, default=20260918)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    dates = [date(2020, m, d) for m in range(1, 13) for d in (4, 14, 24) if (PREDS / f"ocean+atmosphere_24_{date(2020, m, d).isoformat()}.npy").is_file()]
    stats = {k: [] for k in ("fair_crps", "rmse", "spread", "spread_over_rmse", "rank_tv", "cov80", "zero_mass_err", "cap_mass_err", "iiee_mean", "vario_ratio_lag1", "vario_ratio_lag4", "vario_ratio_lag16", "rank_tv_mean_class_ice", "rank_tv_mean_class_open", "rank_tv_mean_class_mixed")}
    for d in dates:
        z = enc(sic(d))
        fields = np.stack([dec(warp(z, rng, a.amp_px, a.corr_px, a.level_amp)) for _ in range(a.members + 1)])
        y, m = fields[0], fields[1:]
        w = WATER
        mean = m.mean(0); sd = m.std(0, ddof=1)
        err = (mean - y)[w]
        stats["fair_crps"].append(fair_crps(m[:, w], y[w]))
        stats["rmse"].append(float(np.sqrt(np.mean(err**2)))); stats["spread"].append(float(np.sqrt(np.mean(sd[w] ** 2))))
        stats["spread_over_rmse"].append(stats["spread"][-1] / max(stats["rmse"][-1], 1e-9))
        stats["rank_tv"].append(rank_tv(m[:, w], y[w], rng))
        lo, hi = np.quantile(m[:, w], [0.1, 0.9], axis=0); stats["cov80"].append(float(np.mean((y[w] >= lo) & (y[w] <= hi))))
        stats["zero_mass_err"].append(float(np.mean(m[:, w] == 0) - np.mean(y[w] == 0)))
        stats["cap_mass_err"].append(float(np.mean(m[:, w] >= CAP) - np.mean(y[w] >= CAP)))
        stats["iiee_mean"].append(float(np.mean(((mean > 0.15) != (y > 0.15))[w])))
        for lag in (1, 4, 16):
            vt = variogram(y, lag, w); vm = np.mean([variogram(m[i], lag, w) for i in range(a.members)])
            stats[f"vario_ratio_lag{lag}"].append(vm / max(vt, 1e-12))
        for name, sel in (("ice", mean >= 0.95), ("open", mean <= 0.0), ("mixed", (mean > 0.0) & (mean < 0.95))):
            s = sel & w
            if s.sum() > 500:
                stats[f"rank_tv_mean_class_{name}"].append(rank_tv(m[:, s], y[s], rng))
    out = {"config": vars(a) | {"n_dates": len(dates)}, "null": {}}
    for k, v in stats.items():
        if v:
            v = np.array(v); out["null"][k] = {"mean": float(v.mean()), "q05": float(np.quantile(v, .05)), "q95": float(np.quantile(v, .95))}
    a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text(json.dumps(out, indent=2, default=str))
    for k, s in out["null"].items():
        print(f"{k:28s} mean {s['mean']:.4f}  q05 {s['q05']:.4f}  q95 {s['q95']:.4f}")


if __name__ == "__main__":
    main()
