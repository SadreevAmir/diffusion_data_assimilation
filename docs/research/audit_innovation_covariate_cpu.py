#!/usr/bin/env python3
"""Does per-date ensemble error follow an operationally available covariate (day-d innovation magnitude)?

For every saved layered case (npz with members/raw/truth/background/day_d/cond/water):
  innov = mean |truth - background| over day-d track pixels  (obs = truth in OSSE, so this is |obs - bg|: known at analysis time)
  per-date RMSE / spread / cov80 / rank-TV of raw and layered ensembles, all water and >40 px from tracks.
Reports Spearman correlations of RMSE, spread, spread/RMSE with innov. If RMSE tracks innov but spread does not,
the residual law needs an innovation covariate.
"""
from __future__ import annotations
import argparse, glob, json
from pathlib import Path
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.stats import spearmanr


def stats(m, y, w, rng):
    M = m.shape[0]; mm = m.mean(0)
    rmse = float(np.sqrt(np.mean((mm[w] - y[w]) ** 2)))
    spread = float(np.sqrt(np.mean(m[:, w].var(0, ddof=1))))
    lo, hi = np.quantile(m[:, w], [.1, .9], axis=0)
    cov = float(np.mean((y[w] >= lo) & (y[w] <= hi)))
    below = (m[:, w] < y[w][None]).sum(0); ties = (m[:, w] == y[w][None]).sum(0)
    r = below + np.floor(rng.random(below.shape) * (ties + 1)).astype(int)
    h = np.bincount(r, minlength=M + 1) / r.size
    tv = float(0.5 * np.abs(h - 1 / (M + 1)).sum())
    return dict(rmse=rmse, spread=spread, sr=spread / max(rmse, 1e-9), cov80=cov, rank_tv=tv)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--members-glob", required=True); p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(); rng = np.random.default_rng(0)
    rows = []
    for f in sorted(glob.glob(a.members_glob)):
        z = np.load(f); w = z["water"]; y = z["truth"].astype(np.float64); bg = z["background"].astype(np.float64)
        dd = z["day_d"] & w; cond = z["cond"] & w
        d = distance_transform_edt(~cond)
        far = w & (d > 40)
        innov = float(np.mean(np.abs(y[dd] - bg[dd]))) if dd.any() else np.nan
        innov_all = float(np.mean(np.abs(y[cond] - bg[cond])))
        bg_rmse = float(np.sqrt(np.mean((bg[w] - y[w]) ** 2)))
        row = dict(date=Path(f).stem.replace("layered_", ""), innov_day_d=innov, innov_all_tracks=innov_all, bg_rmse=bg_rmse, n_day_d=int(dd.sum()))
        for name, m in (("raw", z["raw"].astype(np.float64)), ("lay", z["members"].astype(np.float64))):
            for reg, mask in (("all", w), ("far", far)):
                for k, v in stats(m, y, mask, rng).items():
                    row[f"{name}_{reg}_{k}"] = v
        rows.append(row)
    keys = [k for k in rows[0] if k != "date"]
    innov = np.array([r["innov_day_d"] for r in rows])
    corr = {}
    for cov_name, cov in (("innov_day_d", innov), ("innov_all_tracks", np.array([r["innov_all_tracks"] for r in rows]))):
        corr[cov_name] = {k: float(spearmanr(cov, [r[k] for r in rows]).statistic) for k in keys if k not in ("innov_day_d", "innov_all_tracks", "n_day_d")}
    out = dict(rows=rows, spearman=corr)
    a.output.write_text(json.dumps(out, indent=2))
    print(f"{'metric':22s} rho(innov_d) rho(innov_all)")
    for k in corr["innov_day_d"]:
        print(f"{k:22s} {corr['innov_day_d'][k]:+.2f}       {corr['innov_all_tracks'][k]:+.2f}")
    print("innov_day_d range", innov.min(), innov.max(), "bg_rmse range", min(r["bg_rmse"] for r in rows), max(r["bg_rmse"] for r in rows))


if __name__ == "__main__":
    main()
