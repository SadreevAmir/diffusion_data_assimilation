#!/usr/bin/env python3
"""Hybrid (user question, 21.09): background enters the MEAN, the tracks-only generator (arm D) provides the spread structure.
Point prior mu0 = OOF-weighted combination of the background and the arm-D ensemble mean per cell (weights w_cell fitted on
training folds by least squares of truth on [bg, D_mean] in the v19 latent, per distance x mean cell), then the v19 machinery:
kriging update by track innovations, OOF cell marginals, coupling from arm-D anomalies (mixed with the stationary field).
Rows compared on identical dates/masks: D raw (M=10), D + v19 (own mean), hybrid (bg-informed mean + D anomalies), and
'hybrid mean only' (no marginal recalibration). Ordinary and U CRPS, RMSE, spread. Usage: --samples-d DIR --experiment ... --output JSON"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from docs.research import oof_dispersion_layer_v19 as L
from docs.research.audit_layer_sampling_law_cpu import crps_plain, crps_fair


def folds(n, k=5):
    fs = int(np.ceil(n / k))
    for j in range(k):
        held = list(range(j * fs, min(n, (j + 1) * fs))); purge = set(held) | {h - 1 for h in held} | {h + 1 for h in held}
        yield held, [i for i in range(n) if i not in purge]


def fit_blend(train):
    """Per-cell least-squares weights (a, b, c): zt ≈ a + b*enc(bg) + c*mu_D in the latent, cells = v19 cell_index(dist, mu_D)."""
    n_cells = 7 * L.NMU; X = [[] for _ in range(n_cells)]; Y = [[] for _ in range(n_cells)]
    for c in train:
        w = c["water"]; zb = L.enc(c["bg"], w); cell = c["cell"]
        for k in np.unique(cell[w]):
            sel = w & (cell == k); X[k].append(np.stack([np.ones(sel.sum()), zb[sel], c["mu"][sel]], 1)); Y[k].append(c["zt"][sel])
    coef = np.zeros((n_cells, 3)); coef[:, 2] = 1.0
    for k in range(n_cells):
        if X[k] and sum(len(x) for x in X[k]) > 200:
            A = np.concatenate(X[k]); b = np.concatenate(Y[k]); coef[k] = np.linalg.lstsq(A, b, rcond=None)[0]
    return coef


def apply_blend(c, coef):
    w = c["water"]; zb = L.enc(c["bg"], w); cell = c["cell"]; mu = c["mu"].copy()
    sel = w; k = cell[sel]; mu[sel] = coef[k, 0] + coef[k, 1] * zb[sel] + coef[k, 2] * c["mu"][sel]
    return mu


def scores(m, y, w):
    mean = m.mean(0); sd = m.std(0, ddof=1)
    return {"crps_ordinary": crps_plain(m[:, w], y[w]), "crps_U": crps_fair(m[:, w], y[w]), "rmse": float(np.sqrt(np.mean((mean[w] - y[w]) ** 2))), "spread": float(np.sqrt(np.mean(sd[w] ** 2)))}


def main():
    p = argparse.ArgumentParser(); p.add_argument("--samples-d", type=Path, required=True); p.add_argument("--experiment", type=Path, required=True); p.add_argument("--mode", default="validation_strict_full"); p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(); rng = np.random.default_rng(20260921)
    cases = L.load_cases(a.samples_d, a.experiment, a.mode)
    for c in cases:
        L.prepare(c)
    n = len(cases); per = {}
    for held, tr in folds(n):
        train = [cases[i] for i in tr]
        params = L.fit(train)  # v19 with D's own mean
        coef = fit_blend(train)
        # hybrid: replace mu by the blended mean, refit the layer's residual law on the same training folds with blended mu
        htrain = []
        for c in train:
            h = dict(c); h["mu"] = apply_blend(c, coef); h["resid"] = h["zt"] - h["mu"]; h["cell"] = L.cell_index(h["dist_d"], h["mu"]); htrain.append(h)
        hparams = L.fit(htrain)
        for i in held:
            c = cases[i]; w = c["water"]; y = c["truth"]; d = c["date"].isoformat(); per[d] = {}
            per[d]["D_raw_M10"] = scores(c["ens"], y, w)
            per[d]["D_v19_M10"] = scores(L.simulate(c, params, 10, rng), y, w)
            h = dict(c); h["mu"] = apply_blend(c, coef); h["resid"] = h["zt"] - h["mu"]; h["cell"] = L.cell_index(h["dist_d"], h["mu"])
            per[d]["hybrid_bgmean_Danom_v19_M10"] = scores(L.simulate(h, hparams, 10, rng), y, w)
            mu2 = L.kriged_mean(h, hparams["ell1"], hparams["s_track"], None); shift = L.dec(mu2) - c["ens"].mean(0); upd = np.clip(c["ens"] + shift[None], 0.0, L.CAP); pin = c["day_d"] & w; upd[:, pin] = np.clip(c["obs"][pin], 0.0, L.CAP)
            per[d]["hybrid_mean_update_only_M10"] = scores(np.where(w[None], upd, 0.0), y, w)
            per[d]["blend_mean_rmse"] = float(np.sqrt(np.mean((L.dec(h["mu"])[w] - y[w]) ** 2))); per[d]["D_mean_rmse"] = float(np.sqrt(np.mean((c["ens"].mean(0)[w] - y[w]) ** 2))); per[d]["bg_rmse"] = float(np.sqrt(np.mean((c["bg"][w] - y[w]) ** 2)))
            print(d, {k: (round(v["crps_ordinary"], 4) if isinstance(v, dict) else round(v, 4)) for k, v in per[d].items()}, flush=True)
    dates = sorted(per); rows = [k for k in per[dates[0]] if isinstance(per[dates[0]][k], dict)]
    summary = {r: {k: float(np.mean([per[d][r][k] for d in dates])) for k in ("crps_ordinary", "crps_U", "rmse", "spread")} for r in rows}
    summary["means"] = {k: float(np.mean([per[d][k] for d in dates])) for k in ("blend_mean_rmse", "D_mean_rmse", "bg_rmse")}
    a.output.write_text(json.dumps({"summary": summary, "per_date": per}, indent=1)); print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
