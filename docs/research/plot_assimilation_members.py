#!/usr/bin/env python3
"""All-member panels for saved assimilation ensembles (dev-2022, 10 members).

Model: conditional CFM `m2m_2f_diffusion_balanced_50m/run_20260607_184239`,
siconc-only 3D-Var protocol; inputs = previous-year background + SRAL tracks
(d, d-1, d-2).  Panels show truth, background, observations on the track,
ensemble mean, spread, |mean-truth| and every member, with land in grey and a
zoom around the densest track region.  Diagnostic figure only.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--samples-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--dates", nargs="+", default=("2022-01-16", "2022-04-11", "2022-07-15"))
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for d in a.dates:
        files = sorted(a.samples_dir.glob(f"*_{d}_h23.npz"))
        if not files:
            print("missing", d)
            continue
        z = np.load(files[0])
        ens, truth, bg, obs = z["analysis_ensemble"], z["truth"], z["background"], z["obs_values"]
        track, valid = z["conditioning_mask"], z["valid_mask"]
        mean, sd = ens.mean(0), ens.std(0, ddof=1)
        # zoom window: 96x96 with the most track pixels
        best, win = -1, (0, 0)
        for y in range(0, truth.shape[0] - 96, 8):
            for x in range(0, truth.shape[1] - 96, 8):
                s = track[y : y + 96, x : x + 96].sum()
                if s > best:
                    best, win = s, (y, x)
        y0, x0 = win

        def show(ax, f, title, vmin=0, vmax=1, cmap="viridis", zoom=False, overlay_track=False):
            img = np.where(valid, f, np.nan)
            if zoom:
                img = img[y0 : y0 + 96, x0 : x0 + 96]
            ax.imshow(img, vmin=vmin, vmax=vmax, cmap=cmap, interpolation="nearest")
            if overlay_track:
                t = track[y0 : y0 + 96, x0 : x0 + 96] if zoom else track
                yy, xx = np.nonzero(t)
                ax.scatter(xx, yy, s=1.5 if zoom else 0.3, c="red", marker="s")
            ax.set_facecolor("0.6")
            ax.set_title(title, fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])

        for zoom in (False, True):
            fig, axes = plt.subplots(4, 4, figsize=(16, 19))
            ax = axes.ravel()
            show(ax[0], truth, f"truth {d}", overlay_track=True, zoom=zoom)
            show(ax[1], bg, "background (prev. year)", zoom=zoom)
            show(ax[2], mean, "ensemble mean", overlay_track=True, zoom=zoom)
            show(ax[3], sd, "ensemble sd (0..0.3)", vmin=0, vmax=0.3, cmap="magma", zoom=zoom, overlay_track=True)
            show(ax[4], np.abs(mean - truth), "|mean - truth| (0..0.5)", vmin=0, vmax=0.5, cmap="magma", zoom=zoom, overlay_track=True)
            show(ax[5], (truth == 0).astype(float), "truth exact-zero mask", cmap="gray", zoom=zoom)
            for m in range(10):
                show(ax[6 + m], ens[m], f"member {m}", zoom=zoom)
            fig.suptitle(f"Assimilation ensemble, learned full condition, {d}, 10 members{' — zoom on track' if zoom else ''}; red = assimilated track pixels", fontsize=10)
            fig.tight_layout()
            out = a.output_dir / f"members_{d}{'_zoom' if zoom else ''}.png"
            fig.savefig(out, dpi=90)
            plt.close(fig)
            print("saved", out)


if __name__ == "__main__":
    main()
