#!/usr/bin/env python3
"""Curves for the P1 pilot: joint regret, mean MSE (corrected), marginal CRPS, sd ratio and label-free V per step, per seed block."""
import csv, json
from pathlib import Path
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[3]; import os; OUT = ROOT / "autoresearch_results" / "invariance_audit_v1" / ("p1_pilot" if os.environ.get("P1_STAGE", "pilot") == "pilot" else "p1_confirm")
tab = [{k: (float(v) if k != "arm" else v) for k, v in r.items()} for r in csv.DictReader((OUT / "p1_checkpoint_metrics.csv").open())]
V = {}
BLOCKS = (0, 1) if os.environ.get("P1_STAGE", "pilot") == "pilot" else (10, 11, 12, 13, 14); ARMS = ("K0", "K1", "Kfresh") if os.environ.get("P1_STAGE", "pilot") == "pilot" else ("K0", "K1", "Kfresh", "Krho05", "Krho09")
for b in BLOCKS:
    for arm in ARMS:
        p = OUT / f"block{b}" / f"key_sensitivity_{arm}.csv"
        if p.exists():
            V[(b, arm)] = [(float(r["step"]), float(r["V_kernel"]), float(r["V_kernel_se"])) for r in csv.DictReader(p.open()) if r["cat"] == "-1"]
col = {"K0": "tab:blue", "K1": "tab:red", "Kfresh": "tab:orange", "Krho05": "tab:green", "Krho09": "tab:purple"}; ls = {b: ["-", "--", ":", "-.", (0, (1, 1))][i] for i, b in enumerate(BLOCKS)}
fig, ax = plt.subplots(1, 5, figsize=(20, 3.8))
for (name, key) in zip(["joint regret MMD² (oracle GMM)", "mean MSE (corrected)", "marginal CRPS", "sd ratio (coord 1)", "label-free V (kernel)"], ["mmd2_joint", "mean_mse_corrected", "crps_marginal", "sd1_ratio", None]):
    a = ax[["mmd2_joint", "mean_mse_corrected", "crps_marginal", "sd1_ratio", None].index(key)]
    for b in BLOCKS:
        for arm in ARMS:
            if key is None:
                if (b, arm) in V:
                    s, v, e = zip(*V[(b, arm)]); a.errorbar(np.array(s) + 1, v, yerr=e, color=col[arm], ls=ls[b], marker="o", ms=3, label=f"{arm} b{b}")
            else:
                rows = sorted([r for r in tab if r["block"] == b and r["arm"] == arm], key=lambda r: r["step"])
                a.plot([r["step"] + 1 for r in rows], [r[key] for r in rows], color=col[arm], ls=ls[b], marker="o", ms=3, label=f"{arm} b{b}")
    a.set_xscale("log"); a.set_title(name, fontsize=10); a.set_xlabel("step + 1")
    if key in ("mmd2_joint", "mean_mse_corrected"):
        a.set_yscale("log")
    if key == "sd1_ratio":
        a.axhline(1, color="k", lw=.5)
    if key is None:
        a.axhline(0, color="k", lw=.5)
ax[0].legend(fontsize=7, ncol=2); fig.tight_layout(); fig.savefig(OUT / "p1_curves.png", dpi=130); print("saved", OUT / "p1_curves.png")
