#!/usr/bin/env python3
"""Visual companion to audit_face_dependence_train_cpu.py (train-only, CPU).

Fits the local product-law face kernel on a reduced anchor set (visual use
only) and renders truth / d0 / product samples with land in grey, plus zooms
on consolidated pack, the ice edge and open water.  Saves the fields as npz.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_face_dependence_train_cpu as A  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--anchor", type=str, default="2020-02-04")
    p.add_argument("--lead", type=int, default=3)
    p.add_argument("--factor", type=int, default=4)
    p.add_argument("--fit-days", type=int, nargs="+", default=(6, 20))
    p.add_argument("--seed", type=int, default=20260917)
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    fit = A.anchors_for((2016, 2017, 2018, 2019), tuple(a.fit_days))
    kernel = A.fit_local_kernel(fit, a.factor, rng)
    anchor = date.fromisoformat(a.anchor)
    initial = A.sic(anchor)
    future = A.sic(anchor + timedelta(days=a.lead))
    fid, water = A.child_features(future, initial, a.factor)
    fut = A.pad_to(future, a.factor)
    ini = A.pad_to(initial, a.factor)
    samples = np.stack([A.sample_product(kernel, fid, water, rng) for _ in range(4)])
    np.savez_compressed(a.output_dir / f"panel_{a.anchor}_lead{a.lead}_f{a.factor}.npz", initial=ini, truth=fut, samples=samples, water=water)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def show(ax, f, title, vmin=0.0, vmax=1.0, win=None):
        img = np.where(water, f, np.nan)
        if win is not None:
            (y0, y1), (x0, x1) = win
            img = img[y0:y1, x0:x1]
        ax.imshow(img, vmin=vmin, vmax=vmax, cmap="viridis", interpolation="nearest")
        ax.set_facecolor("0.6")
        ax.set_title(title, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    # choose zoom windows from truth: pack (>=.95 dense), edge (mixed), open (==0 dense)
    def best_window(score_field, size=48):
        best, bw = -1, None
        for y in range(0, fut.shape[0] - size, 8):
            for x in range(0, fut.shape[1] - size, 8):
                s = score_field[y : y + size, x : x + size].mean()
                if s > best:
                    best, bw = s, ((y, y + size), (x, x + size))
        return bw

    pack = best_window(((fut >= 0.95) & water).astype(float))
    edge = best_window((((fut > 0.15) & (fut < 0.85)) & water).astype(float))
    fields = [("d0 initial", ini), (f"truth +{a.lead}", fut), ("product sample 1", samples[0]), ("product sample 2", samples[1]), ("product sample 3", samples[2])]
    fig, axes = plt.subplots(3, 5, figsize=(22, 13))
    for ax, (n, f) in zip(axes[0], fields):
        show(ax, f, f"{n}  {a.anchor}")
    for ax, (n, f) in zip(axes[1], fields):
        show(ax, f, f"pack zoom {n} (0.9..1)", vmin=0.9, vmax=1.0, win=pack)
    for ax, (n, f) in zip(axes[2], fields):
        show(ax, f, f"edge zoom {n} (0..1)", win=edge)
    fig.suptitle(f"Local product-law face kernel under TRUE parents, factor {a.factor}, lead +{a.lead}; fit anchors={len(fit)}", fontsize=11)
    fig.tight_layout()
    out = a.output_dir / f"panel_{a.anchor}_lead{a.lead}_f{a.factor}.png"
    fig.savefig(out, dpi=100)
    print("saved", out)


if __name__ == "__main__":
    main()
