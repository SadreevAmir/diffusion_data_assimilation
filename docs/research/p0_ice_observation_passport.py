#!/usr/bin/env python3
"""Review-2 §5: observation passport of the ice arms on ACTUALLY CONSTRUCTED batches.

For each (config, split) draws `--n` dataset items and records: mask kind (real / library / synthetic), number of observed
pixels, how many of them lie on open water (model truth non-finite -> NaN = ice-free), in the MIZ (.15 < truth < .8) or in
pack ice (truth >= .8), the fraction of footprint pixels dropped because the model value is non-finite (require_finite policy),
the observation noise law from the config, and how NaN model values are encoded (padding value). Truth classes are computed on
the physical (de-normalised) truth field. Output: JSON passport per config + a markdown table.
Run inside the server CPU container from the worktree root: python3 -m docs.research.p0_ice_observation_passport --n 12
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, torch
from assim_lib.config import load_json, merge_config_overrides, resolve_path
from assim_lib.data import build_dataset, _sral_transform, _combine_sral_files
from assim_lib.transforms import pad_to_size
from assim_lib.forecast import field_at_hour
from datetime import timedelta

CONFIGS = {  # label: (experiment json, split)
    "train_A_clean":   ("config/experiments/train_claude_retrain_A_clean.json", "train"),
    "train_C_fielddrop": ("config/experiments/train_claude_retrain_C_fielddrop.json", "train"),
    "train_D_tracksonly": ("config/experiments/train_claude_retrain_D_tracksonly.json", "train"),
    "train_F_libgeom": ("config/experiments/train_claude_retrain_F_tracksonly_libgeom.json", "train"),
    "train_R_unet_mse": ("config/experiments/train_claude_retrain_R_unet_mse.json", "train"),
    "eval_native_protocol": ("config/experiments/experiment_m2m_flow_modes.json", "valid"),
    "eval_tracksonly_D": ("config/experiments/armD_tracksonly_flow_modes.json", "valid"),
    "eval_tracksonly_LIBGEOM": ("config/experiments/armD_tracksonly_LIBGEOM_eval_modes.json", "valid"),
    "eval_tracksonly_TRAINMATCHED": ("config/experiments/armD_tracksonly_TRAINMATCHED_eval_modes.json", "valid"),
}


def passport(label, exp_path, split, n, seed):
    exp_path = Path(exp_path)
    if not exp_path.exists():
        return {"label": label, "status": "CONFIG_NOT_FOUND", "path": str(exp_path)}
    exp = load_json(exp_path); data_config = merge_config_overrides(load_json(resolve_path(exp["data_config"], exp_path.parent)), exp.get("data_overrides"))
    ds = build_dataset(data_config, split=split); rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(ds), size=min(n, len(ds)), replace=False))
    means, stds = ds.means, ds.stds; sic = ds.observed_channels[0] if ds.observed_channels else 0
    rows = []
    for i in idx:
        it = ds[int(i)]; om = it["obs_mask"][sic].numpy() > 0.5; truth = it["truth"][sic].numpy() * stds[sic] + means[sic]
        # physical truth: is the raw model field finite here? reconstruct from target file for the open-water test
        target_date = it["meta"]["target_date"]; hour = it["meta"]["hour"]
        raw = field_at_hour(Path(it["meta"]["target_path"]), hour, [ds.indices[sic]])[0]
        raw = pad_to_size(torch.as_tensor(np.asarray(raw, dtype=np.float32))[None], ds.image_size, fill_value=float("nan"))[0].numpy()
        finite = np.isfinite(raw); water = it["water_mask"][0].numpy() > 0.5
        n_obs = int(om.sum()); n_open = int((om & ~finite).sum()); n_miz = int((om & finite & (raw > 0.15) & (raw < 0.8)).sum()); n_pack = int((om & finite & (raw >= 0.8)).sum()); n_low = int((om & finite & (raw <= 0.15)).sum())
        # footprint union of the real d,d-1,d-2 tracks for this date (rule-1) to estimate what the policy dropped
        mk = it["meta"].get("mask_kind"); dropped = None
        try:
            from datetime import date as _d
            td = _d.fromisoformat(target_date); union = np.zeros(ds.image_size, bool)
            for off in range(ds.assimilation_range):
                comb = _combine_sral_files(ds.sral_records.get(td - timedelta(days=off), []))
                if comb is not None:
                    m = np.isfinite(_sral_transform(comb, 1)); union |= pad_to_size(torch.as_tensor(m.astype(np.float32))[None], ds.image_size, fill_value=0.0)[0].numpy() > 0.5
            union &= water; dropped = int((union & ~finite).sum()); union_n = int(union.sum())
        except Exception as e:
            union_n = None
        rows.append({"index": int(i), "target_date": target_date, "hour": hour, "mask_kind": mk, "n_obs": n_obs, "n_obs_open_water": n_open, "n_obs_low_ice": n_low, "n_obs_miz": n_miz, "n_obs_pack": n_pack,
                     "real_union_pixels": union_n, "real_union_nonfinite_pixels": dropped, "obs_value_at_open_water_mean_physical": float((it["obs_values"][sic].numpy()[om & ~finite] * stds[sic] + means[sic]).mean()) if n_open else None})
    om_cfg = data_config.get("observation_mask", {}); noise = data_config.get("observation_noise", {})
    return {"label": label, "config": str(exp_path), "split": split, "n_items": len(rows), "observation_mask_config": om_cfg, "observation_noise_config": noise, "padding_value_sic": ds.padding_values[sic], "hour_mode": ds.hour_mode,
            "summary": {k: float(np.mean([r[k] for r in rows if r[k] is not None])) if any(r[k] is not None for r in rows) else None for k in ("n_obs", "n_obs_open_water", "n_obs_low_ice", "n_obs_miz", "n_obs_pack", "real_union_pixels", "real_union_nonfinite_pixels")},
            "mask_kinds": sorted({str(r["mask_kind"]) for r in rows}), "items": rows}


def main():
    p = argparse.ArgumentParser(); p.add_argument("--n", type=int, default=12); p.add_argument("--seed", type=int, default=7); p.add_argument("--out", default="/home/autoresearch_results/assimilation_track_distance_cpu_v1/p0_ice_observation_passport.json")
    a = p.parse_args(); out = {}
    for label, (path, split) in CONFIGS.items():
        try:
            out[label] = passport(label, path, split, a.n, a.seed)
        except Exception as e:
            out[label] = {"label": label, "status": f"ERROR: {type(e).__name__}: {e}"}
        r = out[label]; print(label, r.get("status", ""), r.get("mask_kinds"), r.get("summary"), "noise:", r.get("observation_noise_config", {}).get("sigma_physical_range") if isinstance(r.get("observation_noise_config"), dict) else None, "require_finite:", (r.get("observation_mask_config") or {}).get("require_finite_model_values", "default(True)"), flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
