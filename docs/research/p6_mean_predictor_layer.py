#!/usr/bin/env python3
"""P6.1 (Astra pack v1): the same DA update + the same fold-fitted residual law + the same STATIONARY dependence (v9 layer
machinery) on top of different point predictors: background, UNet-MSE (arm R), CFM ensemble mean.  The generator's own
dependence (v19 mixture copula) is the separate last row of the table, computed elsewhere.
--mean-from bg       : point predictor = background (as ablation_no_generator_layer.py)
--mean-from ens_mean : point predictor = mean of the members in --samples-dir (arm R: 1 member; CFM: 10 members)
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from docs.research import oof_dispersion_layer_v9 as L


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples-dir", type=Path, required=True); p.add_argument("--experiment", type=Path, required=True)
    p.add_argument("--mode", default="validation_strict_full"); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mean-from", choices=["bg", "ens_mean"], default="ens_mean"); p.add_argument("--label", default="predictor")
    p.add_argument("--members", type=int, default=10); p.add_argument("--folds", type=int, default=5); p.add_argument("--seed", type=int, default=20260918)
    a = p.parse_args(); rng = np.random.default_rng(a.seed)
    cases = L.load_cases(a.samples_dir, a.experiment, a.mode)
    for c in cases:
        c["ens_generator"] = c["ens"]
        point = c["bg"] if a.mean_from == "bg" else c["ens"].mean(0)
        c["ens"] = np.repeat(point[None], a.members, 0)  # point predictor as a zero-spread 'ensemble'
        L.prepare(c)
    n = len(cases); fold_size = int(np.ceil(n / a.folds)); layered = {}; fold_params = []
    for k in range(a.folds):
        held = list(range(k * fold_size, min(n, (k + 1) * fold_size))); purge = set(held) | {h - 1 for h in held} | {h + 1 for h in held}
        train = [cases[i] for i in range(n) if i not in purge]; params = L.fit(train)
        fold_params.append({"fold": k, "ell": params["ell"], "ell1": params["ell1"], "s_track": params["s_track"]})
        for i in held:
            layered[cases[i]["date"]] = L.simulate(cases[i], params, a.members, rng)
        print(f"fold {k}: kernel={params['ell']}", flush=True)
    point = {c["date"]: c["ens"] for c in cases}
    pd_pt, pd_lay = {}, {}
    res = {"label": a.label, "mean_from": a.mean_from, "point": L.evaluate(cases, point, "point", rng, pd_pt), "point_layer": L.evaluate(cases, layered, "layered", rng, pd_lay), "fold_params": fold_params, "per_date": {"point_layer": pd_lay}}
    a.output.write_text(json.dumps(res, indent=2, default=float))
    print(f"{'stratum':12s} {'RMSE point/point+layer':>24s} {'fCRPS point+layer':>18s} {'s/r':>6s} {'rankTV':>7s} {'zero t/lay':>11s}")
    for name in res["point"]:
        b, l = res["point"][name], res["point_layer"][name]
        print(f"{name:12s} {b['rmse']:11.4f}/{l['rmse']:7.4f} {l['fair_crps']:18.4f} {l['spread_over_rmse']:6.2f} {l['rank_tv']:7.3f} {l.get('truth_zero', float('nan')):5.2f}/{l.get('member_zero', float('nan')):5.2f}")


if __name__ == "__main__":
    main()
