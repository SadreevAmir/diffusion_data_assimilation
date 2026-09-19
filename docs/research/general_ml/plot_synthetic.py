#!/usr/bin/env python3
"""Figures for the synthetic studies: (1) spread/sigma_true vs steps by N and k; (2) noisy-partial-obs heat-map;
(3) copula error vs marginal collapse."""
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

D = Path("docs/research/general_ml"); F = Path("docs/research/figures"); F.mkdir(exist_ok=True)
# 1
r = json.load(open(D / "synth_realizations_per_condition.json"))
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for name, run in r.items():
    st = [c["step"] for c in run["curve"]]; ax[0].plot(st, [c["oos"]["spread_over_true"] for c in run["curve"]], marker=".", label=f"{name} (N={run['N']}, k={run['k']})")
    ax[1].plot(st, [c["oos"]["rmse_mean"] for c in run["curve"]], marker=".", label=name)
ax[0].axhline(1, color="k", ls="--", lw=1); ax[0].set_ylabel("spread / true sigma (OOS)"); ax[0].set_xlabel("training steps"); ax[0].legend(fontsize=6); ax[0].set_title("Density of conditions governs collapse")
ax[1].set_ylabel("RMSE of ensemble mean (OOS)"); ax[1].set_xlabel("training steps"); ax[1].set_title("Skill"); fig.tight_layout(); fig.savefig(F / "fig_synth_realizations.png", dpi=120)
# 2
r = json.load(open(D / "synth_noisy_partial_obs_v3.json"))
Ns = sorted({v["N"] for v in r.values()}); Ss = sorted({v["s_obs"] for v in r.values()})
M = np.full((len(Ns), len(Ss)), np.nan); R = np.full_like(M, np.nan); NN = {}
for v in r.values():
    i, j = Ns.index(v["N"]), Ss.index(v["s_obs"]); c = v["curve"][-1]["oos"]; M[i, j] = c["spread_over_true"]; R[i, j] = c["rmse_mean"] / c["rmse_optimal"]; NN[v["N"]] = v.get("nn_any", np.nan)
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for k, (Z, title) in enumerate(((M, "spread / true posterior sigma (20k steps)"), (R, "RMSE / optimal posterior RMSE"))):
    im = ax[k].imshow(Z, cmap="viridis", vmin=0 if k == 0 else 1, vmax=1.05 if k == 0 else 1.5, origin="lower"); ax[k].set_xticks(range(len(Ss))); ax[k].set_xticklabels([str(s) for s in Ss]); ax[k].set_yticks(range(len(Ns))); ax[k].set_yticklabels([f"N={n} (d_NN={NN[n]:.2f})" for n in Ns])
    ax[k].set_xlabel("observation noise sigma (fresh per step)"); ax[k].set_title(title, fontsize=9)
    for i in range(len(Ns)):
        for j in range(len(Ss)):
            ax[k].text(j, i, f"{Z[i,j]:.2f}", ha="center", va="center", color="w", fontsize=9)
    fig.colorbar(im, ax=ax[k])
fig.tight_layout(); fig.savefig(F / "fig_synth_noise_bandwidth.png", dpi=120)
# 3
r = json.load(open(D / "synth_copula_check.json"))
fig, ax = plt.subplots(figsize=(6, 4))
for name, run in r.items():
    ax.plot([c["oos"]["spread_over_true"] for c in run["curve"]], [c["oos"]["copula_corr_vs_true"] for c in run["curve"]], marker="o", label=f"N={run['N']}")
ax.set_xlabel("spread / true sigma (collapse →)"); ax.set_ylabel("corr(member correlation matrix, true K)"); ax.set_ylim(0.85, 1.01); ax.axhline(1, color="k", ls="--", lw=1); ax.legend(); ax.set_title("Collapse of the marginals, not of the copula")
fig.tight_layout(); fig.savefig(F / "fig_synth_copula.png", dpi=120); print("saved 3 figures")
