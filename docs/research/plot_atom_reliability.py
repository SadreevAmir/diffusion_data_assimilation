#!/usr/bin/env python3
"""Reliability diagrams for the atoms (exact zero, cap) and for P(SIC>0.15) — no truth stratification.

For each pixel-date, the ensemble gives p = share of members in the event; the
diagram bins p and plots the observed frequency of the event in truth. Inputs
are saved ensembles (compare_3dvar npz or layer npz).
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

CAP = 0.9970703125
BINS = np.linspace(0, 1, 11)


def load(f):
    z = np.load(f)
    if "analysis_ensemble" in z:
        return z["analysis_ensemble"].astype(np.float32), z["truth"].astype(np.float32), z["valid_mask"]
    return z["members"].astype(np.float32), z["truth"].astype(np.float32), z["water"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--set", action="append", nargs=2, metavar=("LABEL", "GLOB"), required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    events = {"exact zero (SIC = 0)": lambda x: x <= 0.0, "ice presence (SIC > 0.15)": lambda x: x > 0.15, "cap (SIC = cap)": lambda x: x >= CAP}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    summary = {}
    for label, pattern in a.set:
        acc = {e: [np.zeros(len(BINS) - 1), np.zeros(len(BINS) - 1)] for e in events}  # sums of truth, counts
        brier = {e: [] for e in events}
        n = 0
        for f in sorted(glob.glob(pattern)):
            m, y, w = load(f); n += 1
            for e, fn in events.items():
                pe = fn(m).mean(0)[w]; ye = fn(y)[w].astype(float)
                idx = np.clip(np.digitize(pe, BINS) - 1, 0, len(BINS) - 2)
                acc[e][0] += np.bincount(idx, weights=ye, minlength=len(BINS) - 1)
                acc[e][1] += np.bincount(idx, minlength=len(BINS) - 1)
                M = m.shape[0]
                brier[e].append(float(np.mean((pe - ye) ** 2 - pe * (1 - pe) / (M - 1))))  # fair Brier
        summary[label] = {}
        for ax, (e, (s, c)) in zip(axes, acc.items()):
            freq = np.where(c > 0, s / np.maximum(c, 1), np.nan); centers = 0.5 * (BINS[1:] + BINS[:-1])
            ax.plot(centers, freq, marker="o", label=f"{label} (fair Brier {np.mean(brier[e]):.4f})")
            summary[label][e] = {"forecast_prob": centers.tolist(), "observed_freq": freq.tolist(), "counts": c.tolist(), "fair_brier": float(np.mean(brier[e]))}
        print(label, "dates", n)
    for ax, e in zip(axes, events):
        ax.plot([0, 1], [0, 1], "k--", lw=1); ax.set_xlabel("ensemble probability"); ax.set_ylabel("observed frequency"); ax.set_title(e); ax.grid(alpha=.3); ax.legend(fontsize=7)
    fig.suptitle("Atom / event reliability diagrams (dev-2022, 40 dates, all water pixels; no truth stratification)", fontsize=10)
    fig.tight_layout(); fig.savefig(a.output_dir / "fig_atom_reliability.png", dpi=120)
    (a.output_dir / "atom_reliability.json").write_text(json.dumps(summary, indent=2))
    print("saved")


if __name__ == "__main__":
    main()
