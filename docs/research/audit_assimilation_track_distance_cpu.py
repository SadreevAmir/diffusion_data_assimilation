#!/usr/bin/env python3
"""Distance-to-track calibration diagnostic for saved assimilation ensembles.

For each saved dev-2022 case (10 members, learned full conditioning) the
analysis ensemble is stratified by pixel distance to the assimilated SRAL
track pixels (conditioning_mask).  Observations are exact model-truth values
on the track, so a correct posterior must be nearly degenerate on the track
and widen with distance.  Reported per stratum: ensemble-mean RMSE, spread,
spread/RMSE, fair CRPS, randomized-rank total variation, nominal 80 % interval
coverage, truth/member exact-zero mass, member mass above 0.999, background
RMSE, and on-track |mean - observation|.  Development-2022 data only; this is
a diagnostic of an existing ensemble, not a new method.
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

STRATA = [(0, 0, "on_track"), (1, 2, "d1_2"), (3, 5, "d3_5"), (6, 10, "d6_10"), (11, 20, "d11_20"), (21, 40, "d21_40"), (41, 10_000, "d_gt_40")]


def fair_crps(members: np.ndarray, y: np.ndarray) -> float:
    m = members.shape[0]
    t1 = np.abs(members - y[None]).mean(axis=0)
    t2 = np.zeros_like(y)
    for i in range(m):
        t2 += np.abs(members[i][None] - members).sum(axis=0)
    return float((t1 - t2 / (2 * m * (m - 1))).mean())


def rank_tv(members: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> float:
    m = members.shape[0]
    below = (members < y[None]).sum(axis=0)
    ties = (members == y[None]).sum(axis=0)
    rank = below + np.floor(rng.random(y.shape) * (ties + 1)).astype(np.int64)
    hist = np.bincount(rank, minlength=m + 1).astype(np.float64)
    hist /= hist.sum()
    return float(0.5 * np.abs(hist - 1.0 / (m + 1)).sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20220815)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    files = sorted(glob.glob(str(args.samples_dir / "*.npz")))
    acc: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for f in files:
        d = np.load(f)
        ens = d["analysis_ensemble"].astype(np.float64)
        truth = d["truth"].astype(np.float64)
        bg = d["background"].astype(np.float64)
        obs = d["obs_values"].astype(np.float64)
        track = d["conditioning_mask"]
        valid = d["valid_mask"]
        if not track.any():
            continue
        dist = distance_transform_edt(~track)
        mean = ens.mean(axis=0)
        sd = ens.std(axis=0, ddof=1)
        for lo, hi, name in STRATA:
            sel = valid & (dist >= lo) & (dist <= hi)
            n = int(sel.sum())
            if n < 50:
                continue
            m = ens[:, sel]
            y = truth[sel]
            a = acc[name]
            a["n_pixels"].append(n)
            err = mean[sel] - y
            a["rmse_mean"].append(float(np.sqrt(np.mean(err**2))))
            a["rmse_background"].append(float(np.sqrt(np.mean((bg[sel] - y) ** 2))))
            a["spread"].append(float(np.sqrt(np.mean(sd[sel] ** 2))))
            a["fair_crps"].append(fair_crps(m, y))
            a["rank_tv"].append(rank_tv(m, y, rng))
            lo_q, hi_q = np.quantile(m, [0.1, 0.9], axis=0)
            a["coverage_80"].append(float(np.mean((y >= lo_q) & (y <= hi_q))))
            a["truth_exact_zero"].append(float(np.mean(y == 0)))
            a["member_exact_zero"].append(float(np.mean(m == 0)))
            a["member_mass_ge_0p999"].append(float(np.mean(m >= 0.999)))
            a["truth_mass_ge_0p999"].append(float(np.mean(y >= 0.999)))
            if name == "on_track":
                o = obs[sel]
                a["abs_mean_minus_obs"].append(float(np.mean(np.abs(mean[sel] - o))))
                a["abs_obs_minus_truth"].append(float(np.mean(np.abs(o - y))))
                a["member_sd_on_track"].append(float(np.mean(sd[sel])))
    out = {"num_cases": len(files), "strata": {}}
    for _, _, name in STRATA:
        if name in acc:
            s = {k: float(np.mean(v)) for k, v in acc[name].items()}
            s["spread_over_rmse"] = s["spread"] / s["rmse_mean"] if s["rmse_mean"] > 0 else float("nan")
            s["num_cases"] = len(acc[name]["n_pixels"])
            out["strata"][name] = s
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    hdr = f"{'stratum':10s} {'npix':>7s} {'bgRMSE':>7s} {'RMSE':>7s} {'spread':>7s} {'s/r':>5s} {'fCRPS':>7s} {'rankTV':>6s} {'cov80':>6s} {'t0':>5s} {'m0':>5s} {'m>.999':>6s} {'t>.999':>6s}"
    print(hdr)
    for name, s in out["strata"].items():
        print(f"{name:10s} {s['n_pixels']:7.0f} {s['rmse_background']:7.4f} {s['rmse_mean']:7.4f} {s['spread']:7.4f} {s['spread_over_rmse']:5.2f} {s['fair_crps']:7.4f} {s['rank_tv']:6.3f} {s['coverage_80']:6.3f} {s['truth_exact_zero']:5.2f} {s['member_exact_zero']:5.2f} {s['member_mass_ge_0p999']:6.3f} {s['truth_mass_ge_0p999']:6.3f}")
    if "on_track" in out["strata"]:
        s = out["strata"]["on_track"]
        print(f"on-track |mean-obs|={s['abs_mean_minus_obs']:.4f}  |obs-truth|={s['abs_obs_minus_truth']:.4f}  member sd={s['member_sd_on_track']:.4f}")


if __name__ == "__main__":
    main()
