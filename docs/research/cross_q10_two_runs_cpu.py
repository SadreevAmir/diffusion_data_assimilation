#!/usr/bin/env python3
"""Review-5 §2.3: cross-score of the FULL 10-member layered procedure (Q10) from two independent raw generator runs
(different sampler seeds), each processed independently by the frozen v19 layer (cross-fit). Also the ordinary CRPS of each
10-member run and the raw cross-score. Usage: --samples-a DIR --samples-b DIR --experiment ... --output JSON"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from docs.research import oof_dispersion_layer_v19 as L
from docs.research.audit_layer_sampling_law_cpu import crps_plain, crps_fair, crps_cross


def folds(n, k=5):
    fs = int(np.ceil(n / k))
    for j in range(k):
        held = list(range(j * fs, min(n, (j + 1) * fs))); purge = set(held) | {h - 1 for h in held} | {h + 1 for h in held}
        yield held, [i for i in range(n) if i not in purge]


def main():
    p = argparse.ArgumentParser(); p.add_argument("--samples-a", type=Path, required=True); p.add_argument("--samples-b", type=Path, required=True); p.add_argument("--experiment", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(); rows = []
    A = L.load_cases(a.samples_a, a.experiment, "validation_strict_full"); B = L.load_cases(a.samples_b, a.experiment, "validation_strict_full")
    for c in A + B:
        L.prepare(c)
    n = len(A); rngA = np.random.default_rng(11); rngB = np.random.default_rng(22)
    for held, tr in folds(n):
        pA = L.fit([A[i] for i in tr]); pB = L.fit([B[i] for i in tr])  # parameters fitted on each run's own training folds
        for i in held:
            ca, cb = A[i], B[i]; w = ca["water"]; y = ca["truth"]; assert ca["date"] == cb["date"]
            la = L.simulate(ca, pA, 10, rngA); lb = L.simulate(cb, pB, 10, rngB)
            rows.append({"date": ca["date"].isoformat(), "rawA_ordinary": crps_plain(ca["ens"][:, w], y[w]), "rawB_ordinary": crps_plain(cb["ens"][:, w], y[w]), "raw_U_A": crps_fair(ca["ens"][:, w], y[w]), "raw_cross_AB": crps_cross(ca["ens"][:, w], cb["ens"][:, w], y[w]),
                         "layA_ordinary": crps_plain(la[:, w], y[w]), "layB_ordinary": crps_plain(lb[:, w], y[w]), "lay_U_A": crps_fair(la[:, w], y[w]), "lay_U_B": crps_fair(lb[:, w], y[w]), "lay_cross_Q10": crps_cross(la[:, w], lb[:, w], y[w]),
                         "rawA_rmse": float(np.sqrt(np.mean((ca["ens"].mean(0)[w] - y[w]) ** 2))), "rawB_rmse": float(np.sqrt(np.mean((cb["ens"].mean(0)[w] - y[w]) ** 2)))})
            print(rows[-1], flush=True)
    agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != "date"}; agg["n_dates"] = len(rows)
    a.output.write_text(json.dumps({"summary": agg, "per_date": rows, "estimands": {"lay_cross_Q10": "unbiased CRPS of the random-member law of the FULL 10-member layered procedure (two independent raw runs, independently processed)", "layA_ordinary": "ordinary CRPS of the declared 10-member forecast (run A)", "lay_U_A": "U-formula on dependent layered members (bias sign not fixed)"}}, indent=1))
    print(json.dumps(agg, indent=1))


if __name__ == "__main__":
    main()
