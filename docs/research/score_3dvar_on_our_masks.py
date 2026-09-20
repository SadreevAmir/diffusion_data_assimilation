#!/usr/bin/env python3
"""Re-score the colleagues' 3D-Var analyses (my copy of their pipeline, local logger) with OUR metric definitions on OUR
masks, so that the deterministic 3D-Var row is comparable with our compare_3dvar numbers.

Their fields are on the native 311x225 grid; ours are padded to 320x256 (top-left placement, see transforms.pad_to_size).
For each day of the main_200d protocol we take truth / water(valid) mask / track_imitation mask / background from our saved
test ensembles (test_armC_ep10_main_200d/samples), pad their analysis and background, verify that their background equals
ours on the water mask (protocol identity check), and compute MAE / RMSE / MSE / IIEE(0.15) exactly as compare_3dvar does
(per-day values on the 'full' water mask and on the next-day track footprint, then arithmetic mean over days).
Usage (CPU container, worktree root): python3 -m docs.research.score_3dvar_on_our_masks --their /home/threedvar_out/var_100_main_200d --ours /home/autoresearch_results/test_armC_ep10_main_200d/samples --out <json>
"""
from __future__ import annotations
import argparse, csv, json, re
from pathlib import Path
import numpy as np

EDGE = 0.15


def pad(a: np.ndarray, shape=(320, 256), fill=0.0) -> np.ndarray:
    out = np.full(shape, fill, dtype=np.float64); out[: a.shape[0], : a.shape[1]] = a; return out


def region_metrics(analysis, background, truth, sel):
    t, b, x = truth[sel], background[sel], analysis[sel]
    return {"n": int(sel.sum()), "background_mae": float(np.mean(np.abs(b - t))), "analysis_mae": float(np.mean(np.abs(x - t))), "background_rmse": float(np.sqrt(np.mean((b - t) ** 2))), "analysis_rmse": float(np.sqrt(np.mean((x - t) ** 2))),
            "background_mse": float(np.mean((b - t) ** 2)), "analysis_mse": float(np.mean((x - t) ** 2)), "background_iiee": float(np.mean((b > EDGE) != (t > EDGE))), "analysis_iiee": float(np.mean((x > EDGE) != (t > EDGE)))}


def main():
    p = argparse.ArgumentParser(); p.add_argument("--their", type=Path, required=True); p.add_argument("--ours", type=Path, required=True); p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(); rows = []; bg_diffs = []
    ours = {re.search(r"(\d{4}-\d{2}-\d{2})", f.name).group(1): f for f in sorted(a.ours.glob("*.npz"))}
    for d in sorted(a.their.glob("run_*")):
        back_date = d.name.split("_", 1)[1]; target = f"2023{back_date[4:]}"  # protocol: background = same calendar date one year earlier
        if target not in ours:
            print("no counterpart for", target); continue
        z = np.load(ours[target]); truth = z["truth"].astype(np.float64); water = z["valid_mask"].astype(bool); track = z["track_imitation_mask"].astype(bool) & water; our_bg = z["background"].astype(np.float64)
        x = np.load(d / "analysis_siconc.npy"); b = np.load(d / "background_siconc.npy")
        x = pad(np.nan_to_num(np.asarray(x, dtype=np.float64).reshape(x.shape[-2:]), nan=0.0)); b = pad(np.nan_to_num(np.asarray(b, dtype=np.float64).reshape(b.shape[-2:]), nan=0.0))
        x = np.clip(x, 0, 1); bg_diffs.append(float(np.max(np.abs(b[water] - our_bg[water]))))
        r = {"target_date": target, "background_date": back_date, "bg_max_abs_diff_vs_ours": bg_diffs[-1]}
        for region, sel in (("full", water), ("track_imitation", track)):
            for k, v in region_metrics(x, our_bg, truth, sel).items():
                r[f"{region}_{k}"] = v
        rows.append(r)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.with_suffix(".csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if isinstance(rows[0][k], float)}; agg["n_days"] = len(rows); agg["bg_max_abs_diff_vs_ours_max"] = float(max(bg_diffs))
    a.out.write_text(json.dumps(agg, indent=1))
    print(f"days {len(rows)}; background identity max|diff| = {max(bg_diffs):.4g}")
    for region in ("full", "track_imitation"):
        print(f"{region:16s} bg RMSE {agg[region+'_background_rmse']:.4f} -> 3D-Var {agg[region+'_analysis_rmse']:.4f} | MAE {agg[region+'_background_mae']:.4f} -> {agg[region+'_analysis_mae']:.4f} | IIEE {agg[region+'_background_iiee']:.4f} -> {agg[region+'_analysis_iiee']:.4f}")


if __name__ == "__main__":
    main()
