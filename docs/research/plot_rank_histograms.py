#!/usr/bin/env python3
"""Rank histograms (randomized ties) for saved ensembles: pooled and by distance-to-track strata.

Inputs: one or more (label, samples_dir | npz-glob) pairs. Works with compare_3dvar
samples (analysis_ensemble/truth/conditioning_mask/valid_mask) and with layer
outputs (members/truth/cond/water). Produces one PNG per label plus a
combined pooled figure and a JSON with histogram values and TV distances.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

STRATA = [(0, 2, "≤2 px"), (3, 10, "3–10 px"), (11, 40, "11–40 px"), (41, 10_000, ">40 px")]


def load(f):
    z = np.load(f)
    if "analysis_ensemble" in z:
        return z["analysis_ensemble"].astype(np.float64), z["truth"].astype(np.float64), z["conditioning_mask"] & z["valid_mask"], z["valid_mask"]
    return z["members"].astype(np.float64), z["truth"].astype(np.float64), z["cond"] & z["water"], z["water"]


def ranks(m, y, rng):
    M = m.shape[0]
    below = (m < y[None]).sum(0); ties = (m == y[None]).sum(0)
    return below + np.floor(rng.random(y.shape) * (ties + 1)).astype(int), M


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--set", action="append", nargs=2, metavar=("LABEL", "GLOB"), required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20220815)
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    rng = np.random.default_rng(a.seed)
    summary = {}
    pooled = {}
    for label, pattern in a.set:
        files = sorted(glob.glob(pattern))
        hist_all = None; hist_strata = {}; n_cases = 0
        for f in files:
            m, y, cond, water = load(f)
            if not cond.any():
                continue
            d = distance_transform_edt(~cond)
            r, M = ranks(m[:, water], y[water], rng)
            h = np.bincount(r, minlength=M + 1).astype(float)
            hist_all = h if hist_all is None else hist_all + h
            dw = d[water]
            for lo, hi, name in STRATA:
                sel = (dw >= lo) & (dw <= hi)
                if sel.sum() < 50:
                    continue
                hs = np.bincount(r[sel], minlength=M + 1).astype(float)
                hist_strata[name] = hs if name not in hist_strata else hist_strata[name] + hs
            n_cases += 1
        if hist_all is None:
            print("no data for", label); continue
        M = len(hist_all) - 1
        norm = lambda h: h / h.sum()
        tv = lambda h: float(0.5 * np.abs(norm(h) - 1 / (M + 1)).sum())
        summary[label] = {"n_cases": n_cases, "members": M, "pooled_tv": tv(hist_all), "pooled_hist": norm(hist_all).tolist(),
                          "strata_tv": {k: tv(v) for k, v in hist_strata.items()}}
        pooled[label] = norm(hist_all)
        fig, axes = plt.subplots(1, 1 + len(hist_strata), figsize=(3.4 * (1 + len(hist_strata)), 3.4))
        axes = np.atleast_1d(axes)
        axes[0].bar(range(M + 1), norm(hist_all), color="tab:blue"); axes[0].axhline(1 / (M + 1), color="k", ls="--", lw=1)
        axes[0].set_title(f"{label}\npooled, TV={tv(hist_all):.3f}, n={n_cases}", fontsize=9); axes[0].set_xlabel("rank of truth among members")
        for ax, (name, h) in zip(axes[1:], hist_strata.items()):
            ax.bar(range(M + 1), norm(h), color="tab:orange"); ax.axhline(1 / (M + 1), color="k", ls="--", lw=1)
            ax.set_title(f"{name}, TV={tv(h):.3f}", fontsize=9)
        fig.tight_layout(); out = a.output_dir / f"rank_hist_{label.replace(' ', '_').replace('/', '_')}.png"; fig.savefig(out, dpi=110); plt.close(fig)
        print("saved", out, "pooled TV", round(tv(hist_all), 4))
    if pooled:
        fig, ax = plt.subplots(figsize=(8, 4))
        labels = list(pooled); M = len(next(iter(pooled.values()))) - 1; X = np.arange(M + 1); w = 0.8 / len(labels)
        for i, l in enumerate(labels):
            ax.bar(X + (i - (len(labels) - 1) / 2) * w, pooled[l], w, label=f"{l} (TV {summary[l]['pooled_tv']:.3f})")
        ax.axhline(1 / (M + 1), color="k", ls="--", lw=1, label="uniform"); ax.set_xlabel("rank of truth among members (randomized ties)"); ax.set_ylabel("frequency")
        ax.set_title("Pooled rank histograms, dev-2022, all water pixels"); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(a.output_dir / "rank_hist_pooled_all.png", dpi=120)
    (a.output_dir / "rank_histograms.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
