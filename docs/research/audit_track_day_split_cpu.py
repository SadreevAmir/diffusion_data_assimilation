#!/usr/bin/env python3
"""Split the on-track diagnostic by footprint age (day d vs d-1/d-2).

Observation values on day-d footprints equal the target-day truth exactly;
on d-1/d-2 footprints they are the truth of those earlier days.  A correct
posterior must therefore be (near-)degenerate only on day-d pixels.  This
script rebuilds the day-d track mask from the SRAL records through the same
dataset class used by the comparison protocol and reports, per stratum,
ensemble spread, |mean-obs|, |obs-truth|, exact-zero shares and fair CRPS.
Development-2022 saved ensembles only.
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import torch

from assim_lib.config import load_json, merge_config_overrides, resolve_path
from assim_lib.data import build_dataset


def fair_crps(members: np.ndarray, y: np.ndarray) -> float:
    m = members.shape[0]
    t1 = np.abs(members - y[None]).mean(axis=0)
    t2 = np.zeros_like(y)
    for i in range(m):
        t2 += np.abs(members[i][None] - members).sum(axis=0)
    return float((t1 - t2 / (2 * m * (m - 1))).mean())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", type=Path, required=True)
    p.add_argument("--mode", default="validation_strict_full")
    p.add_argument("--samples-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    experiment = load_json(a.experiment)
    mode = experiment["modes"][a.mode] if "modes" in experiment else {}
    data_config = merge_config_overrides(load_json(resolve_path(experiment["data_config"], a.experiment.parent)), experiment.get("data_overrides"))
    data_config = merge_config_overrides(data_config, mode.get("data_overrides"))
    fields = list(data_config["fields"])
    data_config["observed_channels"] = [fields.index("siconc")] if "siconc" in fields else [0]
    dataset = build_dataset(data_config, split="valid")
    valid_mask = dataset.valid_mask if hasattr(dataset, "valid_mask") else None
    acc = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(str(a.samples_dir / "*.npz"))):
        d = date.fromisoformat(Path(f).stem.split("_")[1])
        z = np.load(f)
        ens = z["analysis_ensemble"].astype(np.float64)
        truth = z["truth"].astype(np.float64)
        obs = z["obs_values"].astype(np.float64)
        cond = z["conditioning_mask"]
        valid = z["valid_mask"]
        vm = torch.as_tensor(valid[None].astype(np.float32)) if valid_mask is None else valid_mask
        saved_range = dataset.assimilation_range
        dataset.assimilation_range = 1
        _, day_mask, _ = dataset._sral_track_observations(d, dataset.hour_index, vm)
        dataset.assimilation_range = saved_range
        day_d = day_mask[0].numpy() > 0
        day_d = day_d[: cond.shape[0], : cond.shape[1]] & valid
        stale = cond & ~day_d & valid
        fresh = cond & day_d
        mean = ens.mean(0)
        sd = ens.std(0, ddof=1)
        for name, sel in (("day_d", fresh), ("stale_d1_d2", stale)):
            n = int(sel.sum())
            if n < 20:
                continue
            r = acc[name]
            r["n_pixels"].append(n)
            r["abs_mean_minus_obs"].append(float(np.abs(mean[sel] - obs[sel]).mean()))
            r["abs_obs_minus_truth"].append(float(np.abs(obs[sel] - truth[sel]).mean()))
            r["abs_mean_minus_truth"].append(float(np.abs(mean[sel] - truth[sel]).mean()))
            r["member_sd"].append(float(sd[sel].mean()))
            r["rmse_mean"].append(float(np.sqrt(np.mean((mean[sel] - truth[sel]) ** 2))))
            r["spread"].append(float(np.sqrt(np.mean(sd[sel] ** 2))))
            r["fair_crps"].append(fair_crps(ens[:, sel], truth[sel]))
            r["truth_exact_zero"].append(float(np.mean(truth[sel] == 0)))
            r["obs_exact_zero"].append(float(np.mean(obs[sel] == 0)))
            r["member_exact_zero"].append(float(np.mean(ens[:, sel] == 0)))
            hi = sel & (obs >= 0.99)
            if hi.sum() >= 20:
                r["hi_obs_mean_member"].append(float(mean[hi].mean()))
                r["hi_obs_abs_mean_minus_obs"].append(float(np.abs(mean[hi] - obs[hi]).mean()))
                r["hi_obs_n"].append(int(hi.sum()))
    out = {}
    for name, r in acc.items():
        s = {k: float(np.mean(v)) for k, v in r.items()}
        s["spread_over_rmse"] = s["spread"] / s["rmse_mean"]
        s["num_cases"] = len(r["n_pixels"])
        out[name] = s
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2))
    for name, s in out.items():
        print(f"== {name}: cases={s['num_cases']} npix/case={s['n_pixels']:.0f}")
        for k in ("abs_obs_minus_truth", "abs_mean_minus_obs", "abs_mean_minus_truth", "member_sd", "spread_over_rmse", "fair_crps", "truth_exact_zero", "obs_exact_zero", "member_exact_zero", "hi_obs_mean_member", "hi_obs_abs_mean_minus_obs"):
            if k in s:
                print(f"   {k:28s} {s[k]:.4f}")


if __name__ == "__main__":
    main()
