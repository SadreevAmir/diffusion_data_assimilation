#!/usr/bin/env python3
"""Does the condition (background = last year's field) have neighbours in the training set? For each dev-2022 date:
distance from its background to the nearest TRAINING background (2015-2020 fields, all dates), compared with the
scale of what must be predicted, RMS(truth - background). If the nearest neighbour is as far as the target
variability, kernel smoothing across conditions cannot recover p(y|x) (synthetic: calibration needs NN distance
<< scale of variation). Also reports the nearest neighbour's calendar offset (same season?) and year."""
from __future__ import annotations
import argparse, json
from datetime import date, timedelta
from pathlib import Path
import numpy as np
from assim_lib.config import load_json, merge_config_overrides, resolve_path
from assim_lib.data import build_dataset


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--experiment", type=Path, required=True); ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--coarse", type=int, default=8, help="block size for the coarse (large-scale) distance")
    a = ap.parse_args()
    e = load_json(a.experiment); dc = merge_config_overrides(load_json(resolve_path(e["data_config"], a.experiment.parent)), e.get("data_overrides"))
    dc["hour_mode"] = "subset"; dc["hour_subset"] = [23]
    tr = build_dataset(dc, split="train"); va = build_dataset(dc, split="valid")
    def bg_of(ds, i):
        it = ds[i]; return it["background"][0].numpy(), it["truth"][0].numpy(), it["valid_mask"][0].numpy() > 0, it["meta"]["target_date"]
    def coarse(x, w, s):
        H, W = x.shape; Hs, Ws = H // s * s, W // s * s
        xw = np.where(w, x, 0.0)[:Hs, :Ws].reshape(Hs // s, s, Ws // s, s).sum((1, 3)); cnt = w[:Hs, :Ws].reshape(Hs // s, s, Ws // s, s).sum((1, 3))
        return np.where(cnt > 0, xw / np.maximum(cnt, 1), 0.0), cnt > 0
    train_bg = []; train_dates = []
    for i in range(len(tr)):
        b, t, w, d = bg_of(tr, i); cb, cw = coarse(b, w, a.coarse); train_bg.append(cb[cw]); train_dates.append(d)
    train_bg = np.stack(train_bg); print("train backgrounds", train_bg.shape, flush=True)
    rows = []
    idxs = list(range(0, len(va), max(1, len(va) // 40)))[:40]
    for i in idxs:
        b, t, w, d = bg_of(va, i); cb, cw = coarse(b, w, a.coarse); ct, _ = coarse(t, w, a.coarse)
        dist = np.sqrt(np.mean((train_bg - cb[cw][None]) ** 2, 1)); j = int(np.argmin(dist))
        anom = float(np.sqrt(np.mean((ct[cw] - cb[cw]) ** 2)))
        dd = date.fromisoformat(d); nd = date.fromisoformat(train_dates[j]); doy = abs((dd.replace(year=2000) - nd.replace(year=2000)).days); doy = min(doy, 366 - doy)
        rows.append(dict(date=d, nn_date=train_dates[j], nn_dist=float(dist[j]), nn_doy_offset=int(doy), anomaly_rms=anom, ratio=float(dist[j] / max(anom, 1e-6)), n_within_anomaly=int((dist < anom).sum())))
        print(f"{d}: NN {train_dates[j]} dist {dist[j]:.3f} | truth-bg anomaly RMS {anom:.3f} | ratio {dist[j]/max(anom,1e-6):.2f} | train dates closer than anomaly scale: {(dist < anom).sum()}", flush=True)
    r = np.array([x["ratio"] for x in rows]); n = np.array([x["n_within_anomaly"] for x in rows])
    print(f"median NN-distance / anomaly-scale = {np.median(r):.2f}; median #train dates within anomaly scale = {np.median(n):.0f}")
    a.output.write_text(json.dumps({"coarse": a.coarse, "rows": rows, "median_ratio": float(np.median(r)), "median_n_within": float(np.median(n))}, indent=1))


if __name__ == "__main__":
    main()
