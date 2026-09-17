#!/usr/bin/env python3
"""Panels: truth / background / raw members / OOF-layer members for saved layer outputs."""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--members-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--label", default="layer")
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for f in sorted(glob.glob(str(a.members_dir / "layered_*.npz"))):
        z = np.load(f)
        lay, raw, truth, bg, water = z["members"], z["raw"], z["truth"], z["background"], z["water"]
        day_d = z["day_d"]
        d = Path(f).stem.replace("layered_", "")
        # zoom on the 96x96 window with most day-d track pixels that also contains an ice edge
        best, win = -1, (0, 0)
        for y in range(0, truth.shape[0] - 96, 8):
            for x in range(0, truth.shape[1] - 96, 8):
                sub = truth[y : y + 96, x : x + 96]
                if not (0.05 < sub[water[y : y + 96, x : x + 96]].mean() < 0.95):
                    continue
                s = day_d[y : y + 96, x : x + 96].sum()
                if s > best:
                    best, win = s, (y, x)
        y0, x0 = win

        def show(ax, fld, title, zoom, vmin=0, vmax=1, cmap="viridis", track=False):
            img = np.where(water, fld, np.nan)
            if zoom:
                img = img[y0 : y0 + 96, x0 : x0 + 96]
            ax.imshow(img, vmin=vmin, vmax=vmax, cmap=cmap, interpolation="nearest")
            if track:
                t = day_d[y0 : y0 + 96, x0 : x0 + 96] if zoom else day_d
                yy, xx = np.nonzero(t)
                ax.scatter(xx, yy, s=1.5 if zoom else 0.3, c="red", marker="s")
            ax.set_facecolor("0.6"); ax.set_title(title, fontsize=8); ax.set_xticks([]); ax.set_yticks([])

        for zoom in (False, True):
            fig, axes = plt.subplots(4, 6, figsize=(22, 16))
            ax = axes.ravel()
            show(ax[0], truth, f"truth {d}", zoom, track=True)
            show(ax[1], bg, "background", zoom)
            show(ax[2], raw.mean(0), "raw mean", zoom, track=True)
            show(ax[3], lay.mean(0), f"{a.label} mean", zoom, track=True)
            show(ax[4], raw.std(0), "raw sd (0..0.3)", zoom, 0, 0.3, "magma")
            show(ax[5], lay.std(0), f"{a.label} sd (0..0.3)", zoom, 0, 0.3, "magma")
            for m in range(6):
                show(ax[6 + m], raw[m], f"raw member {m}", zoom)
            for m in range(6):
                show(ax[12 + m], lay[m], f"{a.label} member {m}", zoom)
            show(ax[18], (truth == 0).astype(float), "truth exact-zero", zoom, cmap="gray")
            show(ax[19], (raw == 0).mean(0), "raw P(exact 0)", zoom, cmap="gray")
            show(ax[20], (lay == 0).mean(0), f"{a.label} P(exact 0)", zoom, cmap="gray")
            show(ax[21], (truth >= 0.9970703125).astype(float), "truth cap", zoom, cmap="gray")
            show(ax[22], (raw >= 0.9970703125).mean(0), "raw P(cap)", zoom, cmap="gray")
            show(ax[23], (lay >= 0.9970703125).mean(0), f"{a.label} P(cap)", zoom, cmap="gray")
            fig.suptitle(f"{d}: raw learned-full ensemble vs OOF dispersion layer ({a.label}); red = day-d track", fontsize=10)
            fig.tight_layout()
            out = a.output_dir / f"{a.label}_{d}{'_zoom' if zoom else ''}.png"
            fig.savefig(out, dpi=85); plt.close(fig)
            print("saved", out)


if __name__ == "__main__":
    main()
