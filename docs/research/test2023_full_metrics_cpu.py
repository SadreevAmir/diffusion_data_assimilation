#!/usr/bin/env python3
"""Review-5 §3: full metric set on test-2023 (main_200d) for arm C + frozen v19 layer and for arm C + the observation-update
block alone, computed from the saved arm-C ensembles with the same point-map rule (ensemble mean), threshold (.15), masks
(full water; track_imitation = next-day footprint from the saved npz) and aggregation (mean over days) as compare_3dvar.
Also answers review-5 §2.2: the 'mean_update_only' block = common shift of all members in the PHYSICAL variable to the v19
kriged mean, then clipping to [0, CAP] and pinning of day-d observations; pairwise member differences are invariant before
clipping/pinning; we report the fraction of pixels touched by clipping/pinning.
Run in the server CPU container from the worktree root.
"""
from __future__ import annotations
import argparse, csv, glob, json, re
from pathlib import Path
import numpy as np
from docs.research import oof_dispersion_layer_v19 as L
from docs.research.ablation_dependence_cpu import mean_only
from docs.research.audit_layer_sampling_law_cpu import crps_plain, crps_fair

EDGE = 0.15


def region_metrics(point, bg, truth, sel):
    t, b, x = truth[sel], bg[sel], point[sel]
    return {"n": int(sel.sum()), "bg_mae": float(np.mean(np.abs(b - t))), "mae": float(np.mean(np.abs(x - t))), "bg_rmse": float(np.sqrt(np.mean((b - t) ** 2))), "rmse": float(np.sqrt(np.mean((x - t) ** 2))),
            "bg_iiee": float(np.mean((b > EDGE) != (t > EDGE))), "iiee": float(np.mean((x > EDGE) != (t > EDGE)))}


def main():
    p = argparse.ArgumentParser(); p.add_argument("--dev-samples", type=Path, required=True); p.add_argument("--test-samples", type=Path, required=True); p.add_argument("--experiment", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True); p.add_argument("--seed", type=int, default=20260919); a = p.parse_args(); rng = np.random.default_rng(a.seed)
    dev = L.load_cases(a.dev_samples, a.experiment, "validation_strict_full")
    for c in dev:
        L.prepare(c)
    params = L.fit(dev)
    test = L.load_cases(a.test_samples, a.experiment, "test_strict_full", split="test")
    track_masks = {re.search(r"(\d{4}-\d{2}-\d{2})", f).group(1): np.load(f)["track_imitation_mask"].astype(bool) for f in glob.glob(str(a.test_samples / "*.npz"))}
    rows = []
    for c in test:
        L.prepare(c); d = c["date"].isoformat(); w = c["water"]; y = c["truth"]; bg = c["bg"]; tim = track_masks[d] & w
        lay = L.simulate(c, params, 10, rng)
        # observation-update block: common physical shift (invariant pairwise differences) then clip + pin
        mu2 = L.kriged_mean(c, params["ell1"], params["s_track"], None); shift = L.dec(mu2) - c["ens"].mean(0)
        shifted = c["ens"] + shift[None]; clipped = np.clip(shifted, 0.0, L.CAP); pin = c["day_d"] & w
        upd = mean_only(c, params)
        sd_before = shifted.std(0, ddof=1); sd_after = upd.std(0, ddof=1)
        r = {"date": d, "frac_pixels_clipped": float(np.mean(((shifted < 0) | (shifted > L.CAP)).any(0)[w])), "frac_pixels_pinned": float(pin[w].mean()),
             "pairwise_sd_ratio_before_clip": float(np.sqrt(np.mean(sd_before[w] ** 2)) / max(np.sqrt(np.mean(c["ens"].std(0, ddof=1)[w] ** 2)), 1e-9)), "pairwise_sd_ratio_after_clip_pin": float(np.sqrt(np.mean(sd_after[w] ** 2)) / max(np.sqrt(np.mean(c["ens"].std(0, ddof=1)[w] ** 2)), 1e-9))}
        for name, ens in (("raw_M5", c["ens"]), ("update_only_M5", upd), ("v19_M10", lay)):
            point = ens.mean(0)
            for region, sel in (("full", w), ("track_imitation", tim)):
                for k, v in region_metrics(point, bg, y, sel).items():
                    r[f"{name}_{region}_{k}"] = v
            r[f"{name}_full_crps_ordinary"] = crps_plain(ens[:, w], y[w]); r[f"{name}_full_crps_U"] = crps_fair(ens[:, w], y[w]); r[f"{name}_M"] = int(ens.shape[0])
        rows.append(r)
        if len(rows) % 20 == 0:
            print(f"{d}: v19 rmse {r['v19_M10_full_rmse']:.4f} upd {r['update_only_M5_full_rmse']:.4f} raw {r['raw_M5_full_rmse']:.4f}", flush=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.with_suffix(".csv").open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if isinstance(rows[0][k], float)}; agg["n_days"] = len(rows)
    a.output.write_text(json.dumps(agg, indent=1))
    for name in ("raw_M5", "update_only_M5", "v19_M10"):
        print(f"{name:16s} full: RMSE {agg[name+'_full_rmse']:.4f} MAE {agg[name+'_full_mae']:.4f} IIEE {agg[name+'_full_iiee']:.4f} | track_im: RMSE {agg[name+'_track_imitation_rmse']:.4f} MAE {agg[name+'_track_imitation_mae']:.4f} IIEE {agg[name+'_track_imitation_iiee']:.4f} | CRPS ord {agg[name+'_full_crps_ordinary']:.4f} U {agg[name+'_full_crps_U']:.4f}")
    print(f"update block: pixels clipped {agg['frac_pixels_clipped']:.4f}, pinned {agg['frac_pixels_pinned']:.4f}; sd ratio before clip {agg['pairwise_sd_ratio_before_clip']:.4f}, after clip+pin {agg['pairwise_sd_ratio_after_clip_pin']:.4f}")


if __name__ == "__main__":
    main()
