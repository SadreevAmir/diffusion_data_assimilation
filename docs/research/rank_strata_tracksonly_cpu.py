#!/usr/bin/env python3
"""Rank-histogram diagnosis by truth stratum x distance to the day-d track for saved ensembles (user question 21.09:
'absence of information should show as many different realisations'). Per stratum: n, RMSE of mean, spread, s/r,
mean bias (mean - truth), rank-TV (M+1 bins, vs uniform), share of extreme ranks (truth below all / above all members),
CRPS. Strata: truth == 0 / (0,.15] / MIZ (.15,.8) / pack [.8,1) / truth == 1 (cap); distance to day-d observed pixels
0 / 1-5 / 6-20 / 21-40 / >40 px. Usage: --sets LABEL DIR ... --output out.json"""
from __future__ import annotations
import argparse, glob, json
from pathlib import Path
import numpy as np
from scipy.ndimage import distance_transform_edt
from assim_lib.metrics import ensemble_crps


def main():
    p = argparse.ArgumentParser(); p.add_argument("--sets", nargs="+", required=True); p.add_argument("--output", type=Path, required=True); a = p.parse_args()
    sets = dict(zip(a.sets[0::2], a.sets[1::2])); out = {}
    for label, d in sets.items():
        acc = {}
        for f in sorted(glob.glob(str(Path(d) / "*.npz"))):
            z = np.load(f); ens = z["analysis_ensemble"].astype(np.float64); y = z["truth"].astype(np.float64); w = z["valid_mask"].astype(bool); obs = z["conditioning_mask"].astype(bool) & w
            M = ens.shape[0]; mean = ens.mean(0); sd = ens.std(0, ddof=1)
            dist = distance_transform_edt(~obs) if obs.any() else np.full(y.shape, 1e4)
            rank = (ens < y[None]).sum(0) + 0.5 * (ens == y[None]).sum(0)  # 0..M
            T = {"truth0": y == 0, "truth_0_15": (y > 0) & (y <= .15), "miz": (y > .15) & (y < .8), "pack_lt1": (y >= .8) & (y < 1), "truth1": y >= 1, "all": np.ones_like(w)}
            D = {"d0": dist == 0, "d1_5": (dist > 0) & (dist <= 5), "d6_20": (dist > 5) & (dist <= 20), "d21_40": (dist > 20) & (dist <= 40), "dgt40": dist > 40, "all": np.ones_like(w)}
            for tn, ts in T.items():
                for dn, ds in D.items():
                    sel = w & ts & ds
                    if sel.sum() < 100:
                        continue
                    r = acc.setdefault(f"{tn}|{dn}", {"n": [], "rmse": [], "spread": [], "bias": [], "tv": [], "below_all": [], "above_all": [], "crps": []})
                    h = np.bincount(np.floor(rank[sel]).astype(int).clip(0, M), minlength=M + 1) / sel.sum()
                    r["n"].append(int(sel.sum())); r["rmse"].append(float(np.sqrt(np.mean((mean[sel] - y[sel]) ** 2)))); r["spread"].append(float(np.sqrt(np.mean(sd[sel] ** 2)))); r["bias"].append(float(np.mean(mean[sel] - y[sel])))
                    r["tv"].append(float(0.5 * np.abs(h - 1 / (M + 1)).sum())); r["below_all"].append(float(np.mean(rank[sel] == 0))); r["above_all"].append(float(np.mean(rank[sel] == M))); r["crps"].append(float(np.mean(ensemble_crps(ens[:, sel], y[sel]))))
        out[label] = {k: {m: float(np.mean(v)) for m, v in r.items()} | {"s_over_r": float(np.mean(r["spread"]) / max(np.mean(r["rmse"]), 1e-9)), "n_dates": len(r["n"])} for k, r in acc.items()}
        print(f"== {label}")
        for k, r in out[label].items():
            print(f"  {k:20s} n={r['n']:7.0f} rmse {r['rmse']:.3f} s/r {r['s_over_r']:.2f} bias {r['bias']:+.3f} TV {r['tv']:.3f} below {r['below_all']:.2f} above {r['above_all']:.2f} crps {r['crps']:.4f}")
    a.output.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
