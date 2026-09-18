#!/usr/bin/env python3
"""Ablation: the OOF layer WITHOUT the generator. The point analysis is the background itself (last year's field),
the kriging update by the track innovations and the OOF residual law with the stationary Gaussian copula (v9) are
applied on top. Cross-fit exactly as the v9 layer; compared with raw generator and generator + v9/v13 numbers.
If this is close to generator + layer, the generative model is not needed as a point estimator either."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from docs.research import oof_dispersion_layer_v9 as L


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples-dir", type=Path, required=True); p.add_argument("--experiment", type=Path, required=True)
    p.add_argument("--mode", default="validation_strict_full"); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--members", type=int, default=10); p.add_argument("--folds", type=int, default=5); p.add_argument("--seed", type=int, default=20260918)
    a = p.parse_args(); rng = np.random.default_rng(a.seed)
    cases = L.load_cases(a.samples_dir, a.experiment, a.mode)
    for c in cases:
        c["ens_generator"] = c["ens"]
        c["ens"] = np.repeat(c["bg"][None], a.members, 0)  # background as the 'ensemble': mean = background, no spread
        L.prepare(c)
    n = len(cases); fold_size = int(np.ceil(n / a.folds)); layered = {}; fold_params = []
    for k in range(a.folds):
        held = list(range(k * fold_size, min(n, (k + 1) * fold_size))); purge = set(held) | {h - 1 for h in held} | {h + 1 for h in held}
        train = [cases[i] for i in range(n) if i not in purge]; params = L.fit(train)
        fold_params.append({"fold": k, "ell": params["ell"], "ell1": params["ell1"], "s_track": params["s_track"]})
        for i in held:
            layered[cases[i]["date"]] = L.simulate(cases[i], params, a.members, rng)
        print(f"fold {k}: kernel={params['ell']}", flush=True)
    bg = {c["date"]: c["ens"] for c in cases}; gen = {c["date"]: c["ens_generator"] for c in cases}
    pd_bg, pd_lay, pd_gen = {}, {}, {}
    res = {"background": L.evaluate(cases, bg, "background", rng, pd_bg), "background_layer": L.evaluate(cases, layered, "layered", rng, pd_lay), "generator_raw": L.evaluate(cases, gen, "generator", rng, pd_gen), "fold_params": fold_params}
    dates = sorted(pd_lay)
    res["per_date"] = {"background_layer": pd_lay, "generator_raw": pd_gen}
    res["dates_layer_better_than_generator_raw"] = int(sum(pd_lay[d]["fair_crps"] < pd_gen[d]["fair_crps"] for d in dates))
    a.output.write_text(json.dumps(res, indent=2, default=float))
    print(f"{'stratum':12s} {'RMSE bg/bg+layer/gen':>24s} {'fCRPS bg+layer/gen':>20s} {'s/r layer':>10s} {'rankTV layer/gen':>17s}")
    for name in res["background"]:
        b, l, g = res["background"][name], res["background_layer"][name], res["generator_raw"][name]
        print(f"{name:12s} {b['rmse']:7.4f}/{l['rmse']:7.4f}/{g['rmse']:7.4f} {l['fair_crps']:9.4f}/{g['fair_crps']:7.4f} {l['spread_over_rmse']:10.2f} {l['rank_tv']:8.3f}/{g['rank_tv']:6.3f}")
    print("dates where background+layer beats raw generator (CRPS):", res["dates_layer_better_than_generator_raw"], "/", len(dates))


if __name__ == "__main__":
    main()
