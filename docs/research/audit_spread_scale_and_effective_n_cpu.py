#!/usr/bin/env python3
"""Two CPU audits for the memorisation question (assimilation model, dev-2022 + train).

Part 1 (saved dev-2022 ensembles): decompose ensemble-mean error and ensemble
spread by spatial scale (native, 4x4 blocks, 16x16 blocks, and the detail
left after removing the 16x16 block mean).  Reports spread/RMSE per scale.
If the deficit is uniform across scales, contraction is a global property;
if it concentrates at large scales, the missing uncertainty is the
large-scale (interannual) mode with very few independent realisations.

Part 2 (train archive 2016-2021, hour slices): redundancy of the training
set.  (a) RMS difference between hour-0 and hour-23 SIC of the same day
versus day-to-day and year-to-year differences; (b) fraction of the
target-minus-background anomaly variance explained by a per-(year, month)
mean over the domain (a proxy for the "which year is it" mode).
Train-only reads; truth used only as the measured object.
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np

ROOT = Path("/mnt/sciml/data_assimilation")
PREDS = ROOT / "da_arctic_2015-2024_v0.1/preds"
WATER = ~np.load(ROOT / "land_mask.npy")


def block_mean(f: np.ndarray, w: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    h, wd = f.shape
    ph, pw = (-h) % k, (-wd) % k
    fp = np.pad(f * w, ((0, ph), (0, pw)))
    wp = np.pad(w.astype(float), ((0, ph), (0, pw)))
    hb, wb = fp.shape[0] // k, fp.shape[1] // k
    s = fp.reshape(hb, k, wb, k).sum(axis=(1, 3))
    c = wp.reshape(hb, k, wb, k).sum(axis=(1, 3))
    m = np.where(c > 0, s / np.maximum(c, 1), 0.0)
    lift = np.repeat(np.repeat(m, k, 0), k, 1)[:h, :wd]
    return m, lift


def part1(samples_dir: Path) -> dict:
    acc = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(str(samples_dir / "*.npz"))):
        z = np.load(f)
        ens = z["analysis_ensemble"].astype(np.float64)
        y = z["truth"].astype(np.float64)
        v = z["valid_mask"]
        wf = v.astype(float)
        mean = ens.mean(0)
        for name, k in (("native", 1), ("block4_80x64", 4), ("block16_20x16", 16)):
            if k == 1:
                em, ym, members = mean, y, ens
                sel = v
            else:
                _, em = block_mean(mean, wf, k)
                _, ym = block_mean(y, wf, k)
                members = np.stack([block_mean(m, wf, k)[1] for m in ens])
                sel = v
            err = (em - ym)[sel]
            sp = members.std(0, ddof=1)[sel]
            acc[name]["mse"].append(float(np.mean(err**2)))
            acc[name]["var"].append(float(np.mean(sp**2)))
        # detail = native minus 16x16 lift
        _, l_mean = block_mean(mean, wf, 16)
        _, l_y = block_mean(y, wf, 16)
        det_members = np.stack([m - block_mean(m, wf, 16)[1] for m in ens])
        err = ((mean - l_mean) - (y - l_y))[v]
        sp = det_members.std(0, ddof=1)[v]
        acc["detail_below_16"]["mse"].append(float(np.mean(err**2)))
        acc["detail_below_16"]["var"].append(float(np.mean(sp**2)))
    out = {}
    for name, d in acc.items():
        mse, var = float(np.mean(d["mse"])), float(np.mean(d["var"]))
        out[name] = {"rmse_mean": mse**0.5, "spread": var**0.5, "spread_over_rmse": (var / mse) ** 0.5, "mse_share_of_native": None}
    nat = out["native"]["rmse_mean"] ** 2
    for name in out:
        out[name]["mse_share_of_native"] = (out[name]["rmse_mean"] ** 2) / nat
    return out


def sic(d: date, hour: int) -> np.ndarray:
    a = np.asarray(np.load(PREDS / f"ocean+atmosphere_24_{d.isoformat()}.npy", mmap_mode="r")[0, hour], dtype=np.float32)
    a = np.where(np.isnan(a), 0.0, a)
    return np.where(WATER, a, 0.0)


def part2(years=(2016, 2017, 2018, 2019, 2020, 2021), days=(5, 15, 25)) -> dict:
    w = WATER
    rms_hour, rms_day, rms_year, rms_bg_anom = [], [], [], []
    anomalies = defaultdict(list)  # (year, month) -> domain-mean anomaly (Y - bg)
    anomaly_fields = []
    keys = []
    for y in years:
        for mth in range(1, 13):
            for dd in days:
                try:
                    d = date(y, mth, dd)
                except ValueError:
                    continue
                d_prev = d - timedelta(days=1)
                try:
                    bg_date = date(y - 1, mth, dd)
                except ValueError:
                    continue
                paths = [PREDS / f"ocean+atmosphere_24_{x.isoformat()}.npy" for x in (d, d_prev, bg_date)]
                if not all(p.is_file() for p in paths):
                    continue
                a23 = sic(d, 23)
                a0 = sic(d, 0)
                p23 = sic(d_prev, 23)
                b23 = sic(bg_date, 23)
                rms_hour.append(float(np.sqrt(np.mean((a23 - a0)[w] ** 2))))
                rms_day.append(float(np.sqrt(np.mean((a23 - p23)[w] ** 2))))
                rms_year.append(float(np.sqrt(np.mean((a23 - b23)[w] ** 2))))
                anom = (a23 - b23)
                anomalies[(y, mth)].append(float(anom[w].mean()))
                _, lift = block_mean(anom, w.astype(float), 16)
                anomaly_fields.append(lift[w])
                keys.append((y, mth))
    A = np.stack(anomaly_fields)  # n x pixels (16x16-block anomaly, large scale)
    # variance explained by (year, month) group mean of the large-scale anomaly field
    total = float(np.mean((A - A.mean(0)) ** 2))
    groups = defaultdict(list)
    for i, k in enumerate(keys):
        groups[k].append(i)
    within = 0.0
    n = 0
    for k, idx in groups.items():
        g = A[idx]
        within += float(((g - g.mean(0)) ** 2).sum())
        n += g.size
    within /= n
    # variance explained by month-only group mean (climatology) for comparison
    gm = defaultdict(list)
    for i, k in enumerate(keys):
        gm[k[1]].append(i)
    within_m = 0.0
    n = 0
    for k, idx in gm.items():
        g = A[idx]
        within_m += float(((g - g.mean(0)) ** 2).sum())
        n += g.size
    within_m /= n
    return {
        "n_dates": len(rms_hour),
        "rms_diff_hour0_vs_hour23_same_day": float(np.mean(rms_hour)),
        "rms_diff_day_to_day_hour23": float(np.mean(rms_day)),
        "rms_diff_year_to_year_same_date_hour23 (background error)": float(np.mean(rms_year)),
        "largescale_anomaly_var_explained_by_month_climatology": 1 - within_m / total,
        "largescale_anomaly_var_explained_by_year_month_mean": 1 - within / total,
        "note": "Y-bg anomaly at 16x16-block scale; (year,month) groups have ~3 dates each (days 5/15/25), so the year-month share is an upper bound of a low-dimensional 'year regime' mode.",
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--samples-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    out = {"part1_scale_decomposition_dev2022": part1(a.samples_dir), "part2_train_redundancy": part2()}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
