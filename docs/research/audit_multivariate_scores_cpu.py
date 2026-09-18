#!/usr/bin/env python3
"""Multivariate (joint-structure) proper scores for saved ensembles, held-out dates.

For each --set LABEL GLOB (layer npz: members/raw/truth/water; compare_3dvar npz: analysis_ensemble/truth/valid_mask):
  * fair energy score on random 16x16 water patches (ES_16), fair adjustment for M members
  * variogram score of order 0.5 (Scheuerer & Hamill 2015) on random pixel pairs at lags 1, 2, 4, 8, 16, 32 px
  * fair CRPS of block means at scales 1, 4, 16, 64 px (aggregation CRPS: does the ensemble get regional totals right?)
  * spectrum ratio: mean member power spectrum / truth power spectrum in 3 wavenumber bands (texture check)
Per-date values, means, and paired 'dates improved' vs the first set. No truth stratification.
"""
from __future__ import annotations
import argparse, glob, json
from pathlib import Path
import numpy as np

LAGS = (1, 2, 4, 8, 16, 32)
SCALES = (1, 4, 16, 64)
PATCH = 16


def load(f):
    z = np.load(f)
    if "analysis_ensemble" in z:
        return z["analysis_ensemble"].astype(np.float64), z["truth"].astype(np.float64), z["valid_mask"]
    return z["members"].astype(np.float64), z["truth"].astype(np.float64), z["water"]


def fair_crps_1d(m, y):  # m: (M, n), y: (n,)
    M = m.shape[0]
    t1 = np.abs(m - y[None]).mean(0)
    ms = np.sort(m, 0); idx = np.arange(M)
    t2 = (ms * (2 * idx - M + 1)[:, None]).sum(0) * 2 / (M * (M - 1))
    return float(np.mean(t1 - 0.5 * t2))


def block_means(a, s, w):
    """Mean over s x s blocks of water pixels; returns (..., nb) for blocks with >= half water."""
    H, W = a.shape[-2:]; Hs, Ws = H // s * s, W // s * s
    aw = np.where(w, a, 0.0)[..., :Hs, :Ws].reshape(*a.shape[:-2], Hs // s, s, Ws // s, s).sum((-3, -1))
    cnt = w[:Hs, :Ws].reshape(Hs // s, s, Ws // s, s).sum((1, 3))
    ok = cnt >= max(1, s * s // 2)
    return (aw / np.maximum(cnt, 1))[..., ok]


def energy_patches(m, y, w, rng, n_patches=64):
    M = m.shape[0]; H, W = y.shape; out = []
    for _ in range(n_patches * 4):
        i = rng.integers(0, H - PATCH); j = rng.integers(0, W - PATCH)
        ww = w[i:i + PATCH, j:j + PATCH]
        if ww.mean() < 0.5:
            continue
        mp = m[:, i:i + PATCH, j:j + PATCH][:, ww]; yp = y[i:i + PATCH, j:j + PATCH][ww]
        d1 = np.linalg.norm(mp - yp[None], axis=1).mean()
        d2 = np.linalg.norm(mp[:, None] - mp[None], axis=2).sum() / (M * (M - 1))
        out.append(d1 - 0.5 * d2)
        if len(out) >= n_patches:
            break
    return float(np.mean(out))


def variogram_score(m, y, w, rng, n_pairs=20000, p=0.5):
    M = m.shape[0]; H, W = y.shape; res = {}
    for lag in LAGS:
        ii = rng.integers(0, H, n_pairs); jj = rng.integers(0, W - lag, n_pairs)
        ang = rng.integers(0, 2, n_pairs)  # horizontal or vertical pair
        i2 = np.where(ang == 0, ii, np.minimum(ii + lag, H - 1)); j2 = np.where(ang == 0, jj + lag, jj)
        ok = w[ii, jj] & w[i2, j2]
        ii, jj, i2, j2 = ii[ok], jj[ok], i2[ok], j2[ok]
        vy = np.abs(y[ii, jj] - y[i2, j2]) ** p
        vm = (np.abs(m[:, ii, jj] - m[:, i2, j2]) ** p).mean(0)
        res[lag] = float(np.mean((vy - vm) ** 2))
    return res


def spectrum_ratio(m, y, w):
    def ps(a):
        a = np.where(w, a, 0.0); F = np.abs(np.fft.fft2(a - a[w].mean() * w)) ** 2
        ky = np.fft.fftfreq(a.shape[0])[:, None]; kx = np.fft.fftfreq(a.shape[1])[None]
        k = np.sqrt(ky**2 + kx**2)
        return np.array([F[(k >= lo) & (k < hi)].mean() for lo, hi in ((0.0, 0.03), (0.03, 0.12), (0.12, 0.5))])
    pt = ps(y); pm = np.mean([ps(mi) for mi in m], 0)
    return (pm / np.maximum(pt, 1e-12)).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", action="append", nargs=2, metavar=("LABEL", "GLOB"), required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    per = {}
    for label, pattern in a.set:
        rng = np.random.default_rng(a.seed)
        rows = {}
        for f in sorted(glob.glob(pattern)):
            m, y, w = load(f); d = Path(f).stem.split("_")[-1] if "layered" in f else Path(f).stem.split("_")[1]
            r = {"es16": energy_patches(m, y, w, rng), "vs": variogram_score(m, y, w, rng), "spec": spectrum_ratio(m, y, w)}
            for s in SCALES:
                bm = block_means(m, s, w); by = block_means(y, s, w)
                r[f"crps_b{s}"] = fair_crps_1d(bm.reshape(m.shape[0], -1), by.ravel())
            rows[d] = r
        per[label] = rows
        print(label, "dates", len(rows), flush=True)
    labels = [l for l, _ in a.set]; ref = labels[0]; dates = sorted(per[ref])
    summary = {}
    print(f"{'metric':14s} " + " ".join(f"{l:>22s}" for l in labels))
    def line(name, get):
        vals = {l: np.array([get(per[l][d]) for d in dates]) for l in labels}
        imp = {l: int((vals[l] < vals[ref]).sum()) for l in labels}
        summary[name] = {l: {"mean": float(vals[l].mean()), "dates_improved_vs_ref": imp[l]} for l in labels}
        print(f"{name:14s} " + " ".join(f"{vals[l].mean():14.5f} ({imp[l]:2d}/{len(dates)})" for l in labels))
    line("ES patch16", lambda r: r["es16"])
    for lag in LAGS:
        line(f"VS0.5 lag{lag}", lambda r, lag=lag: r["vs"][lag])
    for s in SCALES:
        line(f"CRPS block{s}", lambda r, s=s: r[f"crps_b{s}"])
    for b, nm in enumerate(("spec low", "spec mid", "spec high")):
        vals = {l: float(np.mean([per[l][d]["spec"][b] for d in dates])) for l in labels}
        summary[nm] = vals
        print(f"{nm:14s} " + " ".join(f"{vals[l]:22.3f}" for l in labels) + "   (member/truth power ratio; 1 = same texture)")
    a.output.write_text(json.dumps({"summary": summary, "per_date": per}, indent=1))


if __name__ == "__main__":
    main()
