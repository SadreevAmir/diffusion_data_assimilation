#!/usr/bin/env python3
"""v13 layer fitted jointly on several years of saved ensembles (each with its own experiment file), cross-fit with
per-year contiguous folds (purge ±1 within a year) and evaluated per year and pooled. Tests whether the winter
weakness of the single-year law (analysis-doc §12.5) is cured by more regimes in the training folds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import importlib


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--set", action="append", nargs=3, metavar=("LABEL", "SAMPLES_DIR", "EXPERIMENT"), required=True)
    p.add_argument("--mode", default="validation_strict_full")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260918)
    p.add_argument("--layer", default="oof_dispersion_layer_v13", help="layer module under docs.research")
    a = p.parse_args()
    global L
    L = importlib.import_module(f"docs.research.{a.layer}")
    rng = np.random.default_rng(a.seed)
    years = {}
    for label, sd, ex in a.set:
        cases = L.load_cases(Path(sd), Path(ex), a.mode)
        for c in cases:
            L.prepare(c); c["year_label"] = label
        years[label] = cases
    layered = {}
    fold_params = []
    for k in range(a.folds):
        held, purge = [], set()
        for label, cases in years.items():
            n = len(cases); fs = int(np.ceil(n / a.folds))
            h = list(range(k * fs, min(n, (k + 1) * fs)))
            held += [(label, i) for i in h]
            purge |= {(label, j) for i in h for j in (i - 1, i, i + 1)}
        train = [years[l][i] for l in years for i in range(len(years[l])) if (l, i) not in purge]
        params = L.fit(train)
        fold_params.append({"fold": k, "ell": params["ell"], "ell1": params["ell1"], "s_track": params["s_track"], "n_train": len(train)})
        for label, i in held:
            c = years[label][i]
            layered[(label, c["date"])] = L.simulate(c, params, 10, rng)
        print(f"fold {k}: n_train={len(train)} kernel={params['ell']}", flush=True)
    res = {"fold_params": fold_params, "per_year": {}}
    all_cases = []
    for label, cases in years.items():
        raw = {c["date"]: c["ens"] for c in cases}; lay = {c["date"]: layered[(label, c["date"])] for c in cases}
        pd_raw, pd_lay = {}, {}
        r = {"raw": L.evaluate(cases, raw, "raw", rng, pd_raw), "layered": L.evaluate(cases, lay, "layered", rng, pd_lay)}
        dates = sorted(pd_raw)
        delta = np.array([pd_lay[d]["fair_crps"] - pd_raw[d]["fair_crps"] for d in dates])
        r["paired"] = {"mean_delta": float(delta.mean()), "dates_improved": int((delta < 0).sum()), "n_dates": len(delta)}
        r["per_date"] = {"raw": pd_raw, "layered": pd_lay}
        res["per_year"][label] = r
        aw = r["raw"]["all_water"]; lw = r["layered"]["all_water"]
        print(f"{label}: CRPS {aw['fair_crps']:.4f} -> {lw['fair_crps']:.4f} ({r['paired']['dates_improved']}/{len(delta)}), RMSE {aw['rmse']:.4f} -> {lw['rmse']:.4f}, s/r {lw['spread_over_rmse']:.2f}, rankTV {aw['rank_tv']:.3f} -> {lw['rank_tv']:.3f}, zeros {aw['truth_zero']:.2f}/{lw['member_zero']:.2f}", flush=True)
        for name in r["raw"]:
            print(f"   {name:12s} RMSE {r['raw'][name]['rmse']:.4f}/{r['layered'][name]['rmse']:.4f}  CRPS {r['raw'][name]['fair_crps']:.4f}/{r['layered'][name]['fair_crps']:.4f}  s/r {r['layered'][name]['spread_over_rmse']:.2f}")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(res, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating,)) else str(o)))


if __name__ == "__main__":
    main()
