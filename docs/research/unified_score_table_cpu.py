#!/usr/bin/env python3
"""Review-4 §5: ONE score table with the evaluation object and ensemble size stated per row, on identical dates / masks /
observations / weights (all_water, 40 dev dates 2022, native protocol). For every row: ordinary (plain, 1/M²) CRPS of the
declared M-member empirical forecast, U-formula (1/(M(M-1))) CRPS, RMSE of the ensemble mean, spread. Rows:
  raw_M10                : 10 generator members (conditionally iid)
  mean_update_only_M10   : generator members shifted so that their mean = v19 kriged mean, day-d obs pinned (ablation)
  v19_M10                : frozen layer v19 (cross-fit 5x8, purge +-1)
  v19marg_stationary_M10 : v19 members re-coupled per pixel by a stationary field (identical sorted multisets; P6.2 'stationary')
  v9stat_on_<pred>_M10   : v9-type stationary layer (fold-fitted residual law) on a point predictor: background / CFM mean
                           (legacy balanced) / CFM mean arm A / UNet-MSE arm R  (marginal engine differs from v19; stated)
Old (v19-fair) numbers are kept in the audit table of the report; this script does not modify them.
Usage (server CPU container, worktree root): python3 -m docs.research.unified_score_table_cpu --output <json>
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from docs.research import oof_dispersion_layer_v19 as L19
from docs.research import oof_dispersion_layer_v9 as L9
from docs.research.ablation_dependence_cpu import mean_only
from docs.research.audit_layer_sampling_law_cpu import crps_plain, crps_fair
from docs.research.p6_copula_only_cpu import rank_assign

EXP = Path("config/experiments/experiment_m2m_flow_modes.json"); MODE = "validation_strict_full"
SRC = {"legacy_balanced": "/home/autoresearch_results/balanced_native2f_dev2022/result/samples", "armA_cfm": "/home/autoresearch_results/eval_armA_ep10_native2f_dev2022/result/samples", "armR_unet_mse": "/home/autoresearch_results/eval_armR_unet_mse_dev2022/result/samples"}


def scores(m, y, w):
    mean = m.mean(0); sd = m.std(0, ddof=1)
    return {"M": int(m.shape[0]), "crps_ordinary": crps_plain(m[:, w], y[w]), "crps_U": crps_fair(m[:, w], y[w]) if m.shape[0] > 1 else None, "rmse_mean": float(np.sqrt(np.mean((mean[w] - y[w]) ** 2))), "spread": float(np.sqrt(np.mean(sd[w] ** 2)))}


def folds(n, k=5):
    fs = int(np.ceil(n / k))
    for j in range(k):
        held = list(range(j * fs, min(n, (j + 1) * fs))); purge = set(held) | {h - 1 for h in held} | {h + 1 for h in held}
        yield held, [i for i in range(n) if i not in purge]


def main():
    p = argparse.ArgumentParser(); p.add_argument("--output", type=Path, required=True); p.add_argument("--seed", type=int, default=20260921); a = p.parse_args(); rng = np.random.default_rng(a.seed)
    per_date = {}
    # --- v19 rows on the legacy balanced samples
    cases = L19.load_cases(Path(SRC["legacy_balanced"]), EXP, MODE)
    for c in cases:
        L19.prepare(c)
    n = len(cases)
    for held, tr in folds(n):
        params = L19.fit([cases[i] for i in tr])
        for i in held:
            c = cases[i]; w = c["water"]; y = c["truth"]; d = c["date"].isoformat(); per_date.setdefault(d, {})
            per_date[d]["raw_M10"] = scores(c["ens"], y, w)
            per_date[d]["mean_update_only_M10"] = scores(mean_only(c, params), y, w)
            out = L19.simulate(c, params, 10, rng); per_date[d]["v19_M10"] = scores(out, y, w)
            S = np.sort(out, axis=0); g = np.stack([L19.smooth_unit(rng.standard_normal(w.shape), {"a": 0.0, "l1": params["ell"]["l1"], "l2": params["ell"]["l2"]}, w, rng.standard_normal(w.shape)) for _ in range(10)])
            per_date[d]["v19marg_stationary_M10"] = scores(rank_assign(S, g), y, w)
            bg = np.repeat(c["bg"][None], 10, 0); per_date[d]["background_M1"] = {"M": 1, "crps_ordinary": crps_plain(bg[:1][:, w], y[w]), "crps_U": None, "rmse_mean": float(np.sqrt(np.mean((c["bg"][w] - y[w]) ** 2))), "spread": 0.0}
        print(f"v19 fold done ({len(held)} dates)", flush=True)
    # --- v9-type stationary layer on point predictors
    for label, src in (("background", SRC["legacy_balanced"]), ("cfm_mean_legacy", SRC["legacy_balanced"]), ("cfm_mean_armA", SRC["armA_cfm"]), ("unet_mse_armR", SRC["armR_unet_mse"])):
        cs = L9.load_cases(Path(src), EXP, MODE)
        for c in cs:
            point = c["bg"] if label == "background" else c["ens"].mean(0); c["ens"] = np.repeat(point[None], 10, 0); L9.prepare(c)
        for held, tr in folds(len(cs)):
            params = L9.fit([cs[i] for i in tr])
            for i in held:
                c = cs[i]; d = c["date"].isoformat(); per_date[d][f"point_{label}_M1"] = {"M": 1, "crps_ordinary": crps_plain(c["ens"][:1][:, c["water"]], c["truth"][c["water"]]), "crps_U": None, "rmse_mean": float(np.sqrt(np.mean((c["ens"][0][c["water"]] - c["truth"][c["water"]]) ** 2))), "spread": 0.0}
                per_date[d][f"v9stat_on_{label}_M10"] = scores(L9.simulate(c, params, 10, rng), c["truth"], c["water"])
        print(f"v9 rows done: {label}", flush=True)
    rows = sorted({k for v in per_date.values() for k in v}); dates = sorted(per_date)
    table = {r: {k: (float(np.mean([per_date[d][r][k] for d in dates])) if per_date[dates[0]][r][k] is not None else None) for k in ("M", "crps_ordinary", "crps_U", "rmse_mean", "spread")} for r in rows}
    for r in rows:
        table[r]["n_dates"] = len(dates); table[r]["dates_better_than_raw_ordinary"] = int(sum(per_date[d][r]["crps_ordinary"] < per_date[d]["raw_M10"]["crps_ordinary"] for d in dates))
    a.output.write_text(json.dumps({"table": table, "per_date": per_date, "estimands": {"crps_ordinary": "plain CRPS of the declared M-member empirical forecast (1/M^2 pairwise), all_water, mean over dates", "crps_U": "U-formula (1/(M(M-1))) on the same samples; unbiased for the underlying law only if members are conditionally iid (raw); for post-processed dependent members the bias sign is not fixed"}}, indent=1))
    print(f"{'row':32s} {'M':>3s} {'ordinary':>9s} {'U':>8s} {'RMSE':>7s} {'spread':>7s} {'dates<raw':>9s}")
    for r in rows:
        t = table[r]; print(f"{r:32s} {t['M']:3.0f} {t['crps_ordinary']:9.4f} {t['crps_U'] if t['crps_U'] is None else round(t['crps_U'], 4)!s:>8s} {t['rmse_mean']:7.4f} {t['spread']:7.4f} {t['dates_better_than_raw_ordinary']:9d}")


if __name__ == "__main__":
    main()
