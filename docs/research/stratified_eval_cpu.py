#!/usr/bin/env python3
"""Review-3 §7: stratified evaluation of saved compare_3dvar ensembles on identical dates.
Strata (all within the water mask): truth class (open water = truth 0 exactly / ice-free ≤ .15 / MIZ (.15,.8) / pack ≥ .8),
edge band (|truth − .15| crossing within 5 px), distance to the day-d observed pixels (0 / 1–5 / 6–20 / > 20 px),
observed vs unobserved (conditioning mask). Per stratum: RMSE of the mean, spread, spread/RMSE, plain CRPS, IIEE(.15).
Usage: python3 -m docs.research.stratified_eval_cpu --sets LABEL /path/to/result/samples ... --output out.json
"""
from __future__ import annotations
import argparse, json, glob
from pathlib import Path
import numpy as np
from scipy.ndimage import distance_transform_edt, binary_dilation
from assim_lib.metrics import ensemble_crps

EDGE = 0.15


def strata_for(z):
    truth = z["truth"].astype(np.float64); w = z["valid_mask"].astype(bool); obs = z["conditioning_mask"].astype(bool) & w
    d_obs = distance_transform_edt(~obs) if obs.any() else np.full(truth.shape, 1e4)
    ice = truth > EDGE; edge = (binary_dilation(ice, iterations=5) & ~ice) | (binary_dilation(~ice, iterations=5) & ice)
    S = {"all_water": w, "truth_zero": w & (truth == 0), "ice_free_le15": w & (truth > 0) & (truth <= EDGE), "miz_15_80": w & (truth > EDGE) & (truth < 0.8), "pack_ge80": w & (truth >= 0.8),
         "edge_band_5px": w & edge, "observed_px": obs, "unobserved_px": w & ~obs, "dist_obs_1_5": w & ~obs & (d_obs <= 5), "dist_obs_6_20": w & ~obs & (d_obs > 5) & (d_obs <= 20), "dist_obs_gt20": w & ~obs & (d_obs > 20)}
    return S


def main():
    p = argparse.ArgumentParser(); p.add_argument("--sets", nargs="+", required=True, help="LABEL DIR pairs"); p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(); sets = dict(zip(a.sets[0::2], a.sets[1::2])); out = {}
    for label, d in sets.items():
        acc = {}
        for f in sorted(glob.glob(str(Path(d) / "*.npz"))):
            z = np.load(f); ens = z["analysis_ensemble"].astype(np.float64); truth = z["truth"].astype(np.float64); mean = ens.mean(0); sd = ens.std(0, ddof=1)
            for name, sel in strata_for(z).items():
                if sel.sum() < 50:
                    continue
                r = acc.setdefault(name, {"n": [], "rmse": [], "spread": [], "crps": [], "iiee": []})
                r["n"].append(int(sel.sum())); r["rmse"].append(float(np.sqrt(np.mean((mean[sel] - truth[sel]) ** 2)))); r["spread"].append(float(np.sqrt(np.mean(sd[sel] ** 2))))
                r["crps"].append(float(np.mean(ensemble_crps(ens[:, sel], truth[sel])))); r["iiee"].append(float(np.mean((mean[sel] > EDGE) != (truth[sel] > EDGE))))
        out[label] = {name: {k: float(np.mean(v)) for k, v in r.items()} | {"spread_over_rmse": float(np.mean(r["spread"]) / max(np.mean(r["rmse"]), 1e-9)), "n_dates": len(r["rmse"])} for name, r in acc.items()}
        print(f"== {label}")
        for name, r in out[label].items():
            print(f"  {name:16s} n={r['n']:7.0f} RMSE {r['rmse']:.4f} spread {r['spread']:.4f} s/r {r['spread_over_rmse']:.2f} CRPS {r['crps']:.4f} IIEE {r['iiee']:.4f}")
    a.output.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
