#!/usr/bin/env python3
"""Ablations of the layer (review §5), cross-fitted like v15, all on the same held-out dates:

  raw            : generator ensemble as is
  mean_only      : generator members shifted so that their mean equals the kriged mean (day-d obs pinned); no
                   recalibration of spread / atoms  -> how much of the gain is the mean correction?
  With IDENTICAL cell marginals (same OOF quantiles, same kriged mean), only the dependence differs:
  dep_stationary : score = independent long-scale stationary field (lambda = 1)
  dep_generator  : score = standardised anomaly pattern of the generator for THIS date (lambda = 0)
  dep_other_date : score = standardised anomaly pattern of the generator for ANOTHER date (the held-out date shifted by
                   +/- 25 days inside the same fold; same season, different state)
  dep_mixture    : v15 (lambda = 0.5)
Scores: fair CRPS, patch energy score (16x16), variogram p=1/2 at lags 1/4/16, block-mean CRPS at 16 px, rank TV.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from scipy.special import ndtr
from docs.research import oof_dispersion_layer_v15 as L
from docs.research.audit_multivariate_scores_cpu import energy_patches, variogram_score, block_means, fair_crps_1d


def simulate_dep(c, params, rng, mode, other=None):
    w = c["water"]; Q = params["Q"]; ell = params["ell"]
    mu2 = L.kriged_mean(c, params["ell1"], params["s_track"], None); pin = c["day_d"] & w; cell = L.cell_index(c["dist_d"], mu2)
    M = c["anom"].shape[0]; out = np.zeros((M, *w.shape))
    for m in range(M):
        g = L.smooth_unit(rng.standard_normal(w.shape), {"a": 0.0, "l1": ell["l1"], "l2": ell["l2"]}, w, rng.standard_normal(w.shape))
        if mode == "dep_stationary":
            score = g
        elif mode == "dep_generator":
            score = c["anom"][m]
        elif mode == "dep_other_date":
            score = other["anom"][m]
        else:
            score = np.sqrt(0.5) * c["anom"][m] + np.sqrt(0.5) * g
        u = np.clip(ndtr(np.clip(score[w], -6, 6)), L.U_CLIP, 1 - L.U_CLIP); r = L.Finv_cell(u, cell[w], Q)
        zm = np.zeros(w.shape); zm[w] = mu2[w] + r; zm[pin] = c["zobs"][pin]; out[m] = np.where(w, L.dec(zm), 0.0)
    return out


def mean_only(c, params):
    w = c["water"]; mu2 = L.kriged_mean(c, params["ell1"], params["s_track"], None); pin = c["day_d"] & w
    target = L.dec(mu2); shift = target - c["ens"].mean(0)
    out = np.clip(c["ens"] + shift[None], 0.0, L.CAP); out[:, pin] = np.clip(c["obs"][pin], 0.0, L.CAP)
    return np.where(w[None], out, 0.0)


def score_set(m, y, w, rng):
    M = m.shape[0]
    r = dict(fair_crps=fair_crps_1d(m[:, w], y[w]), es16=energy_patches(m, y, w, rng), vs=variogram_score(m, y, w, rng, n_pairs=10000))
    bm = block_means(m, 16, w); by = block_means(y, 16, w); r["crps_block16"] = fair_crps_1d(bm.reshape(M, -1), by.ravel())
    below = (m[:, w] < y[w][None]).sum(0); ties = (m[:, w] == y[w][None]).sum(0)
    rk = below + np.floor(rng.random(below.shape) * (ties + 1)).astype(int); h = np.bincount(rk, minlength=M + 1) / rk.size
    r["rank_tv"] = float(0.5 * np.abs(h - 1 / (M + 1)).sum()); r["spread_over_rmse"] = float(np.sqrt(m[:, w].var(0, ddof=1).mean()) / np.sqrt(np.mean((m[:, w].mean(0) - y[w]) ** 2)))
    return r


def main():
    p = argparse.ArgumentParser(); p.add_argument("--samples-dir", type=Path, required=True); p.add_argument("--experiment", type=Path, required=True)
    p.add_argument("--mode", default="validation_strict_full"); p.add_argument("--output", type=Path, required=True); p.add_argument("--folds", type=int, default=5); p.add_argument("--seed", type=int, default=20260919)
    a = p.parse_args(); rng = np.random.default_rng(a.seed)
    cases = L.load_cases(a.samples_dir, a.experiment, a.mode)
    for c in cases:
        L.prepare(c)
    n = len(cases); fold_size = int(np.ceil(n / a.folds)); variants = ["raw", "mean_only", "dep_stationary", "dep_generator", "dep_other_date", "dep_mixture"]
    per = {v: {} for v in variants}
    for k in range(a.folds):
        held = list(range(k * fold_size, min(n, (k + 1) * fold_size))); purge = set(held) | {h - 1 for h in held} | {h + 1 for h in held}
        train = [cases[i] for i in range(n) if i not in purge]; params = L.fit(train)
        for i in held:
            c = cases[i]; w = c["water"]; y = c["truth"]
            other = cases[i + 5] if i + 5 < n else cases[i - 5]  # +-25 days (stride 5 days): same season, different state
            ens = {"raw": c["ens"], "mean_only": mean_only(c, params)}
            for v in variants[2:]:
                ens[v] = simulate_dep(c, params, rng, v, other=other)
            for v in variants:
                per[v][c["date"].isoformat()] = score_set(ens[v], y, w, rng)
            print(f"{c['date']}: " + " ".join(f"{v}={per[v][c['date'].isoformat()]['fair_crps']:.4f}" for v in variants), flush=True)
    dates = sorted(per["raw"]); summary = {}
    print(f"{'variant':16s} {'fairCRPS':>9s} {'ES16':>7s} {'VS1':>7s} {'VS4':>7s} {'VS16':>7s} {'CRPSb16':>8s} {'rankTV':>7s} {'s/r':>5s}  dates<raw")
    for v in variants:
        g = lambda key, f=lambda r: r: float(np.mean([f(per[v][d])[key] if key in f(per[v][d]) else np.nan for d in dates]))
        row = dict(fair_crps=g("fair_crps"), es16=g("es16"), vs1=float(np.mean([per[v][d]["vs"][1] for d in dates])), vs4=float(np.mean([per[v][d]["vs"][4] for d in dates])), vs16=float(np.mean([per[v][d]["vs"][16] for d in dates])), crps_block16=g("crps_block16"), rank_tv=g("rank_tv"), spread_over_rmse=g("spread_over_rmse"), dates_better_than_raw=int(sum(per[v][d]["fair_crps"] < per["raw"][d]["fair_crps"] for d in dates)))
        summary[v] = row
        print(f"{v:16s} {row['fair_crps']:9.4f} {row['es16']:7.3f} {row['vs1']:7.4f} {row['vs4']:7.4f} {row['vs16']:7.4f} {row['crps_block16']:8.4f} {row['rank_tv']:7.3f} {row['spread_over_rmse']:5.2f}  {row['dates_better_than_raw']}/{len(dates)}")
    a.output.write_text(json.dumps(dict(summary=summary, per_date=per), indent=1, default=float))


if __name__ == "__main__":
    main()
