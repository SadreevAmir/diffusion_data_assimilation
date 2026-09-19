#!/usr/bin/env python3
"""P2 figure: ordinary kernel score / CRPS of the 64-member ensemble vs number of key groups L, measured vs predicted R(Qbar)+V/L+W/M."""
import json
from pathlib import Path
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT = Path(__file__).resolve().parents[3] / "autoresearch_results" / "invariance_audit_v1" / "p2_correction"
fig, ax = plt.subplots(1, 3, figsize=(15, 4)); col = {"K1": "tab:red", "K0": "tab:blue", "Kfresh": "tab:orange"}; ls = {0: "-", 1: "--"}
for b in (0, 1):
    for arm in ("K1", "K0", "Kfresh"):
        p = OUT / f"block{b}_{arm}" / "summary.json"
        if not p.exists():
            continue
        s = json.loads(p.read_text()); L = [t["L"] for t in s["table"]]
        for a, key, pk in ((ax[0], "kernel_measured", "kernel_predicted"), (ax[1], "crps_measured", "crps_predicted")):
            a.errorbar(L, [t[key] for t in s["table"]], yerr=[t["kernel_se_over_cats"] if key.startswith("kernel") else 0 for t in s["table"]], color=col[arm], ls=ls[b], marker="o", ms=4, label=f"{arm} b{b} measured")
            a.plot(L, [t[pk] for t in s["table"]], color=col[arm], ls=ls[b], marker="x", ms=6, alpha=.5, label=f"{arm} b{b} predicted R̄+V/L+W/M")
        if arm == "K1":
            ax[2].scatter(s["per_cat_pred_gain_kernel"], s["per_cat_gain_kernel"], color=col[arm], marker="o" if b == 0 else "s", label=f"K1 b{b}: per category")
for a, t in zip(ax, ["ordinary kernel score of the 64-member ensemble", "ordinary marginal CRPS (64 members)", "gain S(L=1) − S(L=64) per category (kernel)"]):
    a.set_title(t, fontsize=10)
ax[0].set_xscale("log", base=2); ax[1].set_xscale("log", base=2); ax[0].set_xlabel("L key groups (m = 64/L members per key)"); ax[1].set_xlabel("L")
ax[2].plot([0, .03], [0, .03], "k-", lw=.6); ax[2].set_xlabel("predicted (1 − 1/M)·V (label-free)"); ax[2].set_ylabel("measured")
ax[0].legend(fontsize=6, ncol=2); ax[2].legend(fontsize=7); fig.tight_layout(); fig.savefig(OUT / "p2_curves.png", dpi=130); print("saved")
