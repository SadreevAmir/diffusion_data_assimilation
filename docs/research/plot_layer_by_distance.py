#!/usr/bin/env python3
"""Raw vs layered metrics by distance-to-track stratum, one panel row per layer JSON (oof_dispersion_layer_v*.py output)."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

METRICS = [("rmse", "RMSE of mean"), ("fair_crps", "fair CRPS"), ("spread_over_rmse", "spread / RMSE"), ("rank_tv", "rank-hist TV"), ("member_zero", "exact-zero share")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--set", action="append", nargs=2, metavar=("LABEL", "JSON"), required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(a.set), len(METRICS), figsize=(3.3 * len(METRICS), 3.0 * len(a.set)), squeeze=False)
    for row, (label, path) in enumerate(a.set):
        res = json.load(open(path))
        strata = [k for k in res["raw"] if k != "all_water"]
        x = np.arange(len(strata))
        for col, (key, title) in enumerate(METRICS):
            ax = axes[row, col]
            r = [res["raw"][s].get(key, np.nan) for s in strata]; l = [res["layered"][s].get(key, np.nan) for s in strata]
            ax.bar(x - 0.2, r, 0.4, label="raw", color="tab:gray"); ax.bar(x + 0.2, l, 0.4, label="layer", color="tab:blue")
            if key == "member_zero":
                ax.plot(x, [res["raw"][s]["truth_zero"] for s in strata], "k_", ms=14, mew=2, label="truth")
            if key == "spread_over_rmse":
                ax.axhline(1.0, color="k", ls="--", lw=1)
            ax.set_xticks(x); ax.set_xticklabels(strata, rotation=45, ha="right", fontsize=7); ax.set_title(f"{label}: {title}", fontsize=8); ax.grid(alpha=.3, axis="y")
            if col == 0:
                ax.legend(fontsize=7)
    fig.tight_layout(); a.output.parent.mkdir(parents=True, exist_ok=True); fig.savefig(a.output, dpi=120); print("saved", a.output)


if __name__ == "__main__":
    main()
