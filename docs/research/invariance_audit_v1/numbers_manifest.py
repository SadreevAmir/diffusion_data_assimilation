#!/usr/bin/env python3
"""Machine-readable manifest of the numbers used in ml_paper_v3_ru.md / applied_paper_v2_ru.md (review-5 §9).
Local (synthetic P1) numbers are RECOMPUTED from the result files here and written with the file sha256; ice numbers
live in server files: their entries carry the file path, the JSON key and the value quoted in the draft, and the sha256 is
filled when the server is reachable (--server-root) else 'PENDING_SERVER'.
"""
from __future__ import annotations
import argparse, csv, hashlib, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]; RES = ROOT / "autoresearch_results/invariance_audit_v1"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.exists() else "PENDING_SERVER"


def rows(p: Path):
    with p.open() as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--server-root", type=Path, default=None); ap.add_argument("--output", type=Path, default=ROOT / "docs/research/invariance_audit_v1/numbers_manifest.json"); a = ap.parse_args()
    man = {"local": [], "server": []}
    # --- synthetic P1 confirm: per-arm values at 20k, paired block differences 20k-10k (blocks 10-14)
    cd = RES / "confirm_decomposition/confirm_decomposition_summary.csv"; R = rows(cd)
    def get(arm, step, key):
        return {int(r["block"]): float(r[key]) for r in R if r["arm"] == arm and int(r["step"]) == step and r[key] != ""}
    for arm in ("K0", "K1", "Kfresh", "Krho05", "Krho09"):
        for key, label in (("crps_raw", "conditional (raw) score"), ("crps_mix64", "key-mixture score L=64"), ("V", "nuisance-key sensitivity V"), ("mmd2_raw", "MMD² raw"), ("mmd2_mixture", "MMD² mixture")):
            v20 = get(arm, 20000, key); v10 = get(arm, 10000, key)
            if not v20:
                continue
            b = sorted(set(v20) & set(v10)); d = np.array([v20[i] - v10[i] for i in b])
            man["local"].append({"quantity": f"{arm} {label}", "file": str(cd.relative_to(ROOT)), "key": key, "value_20k_mean_blocks": float(np.mean(list(v20.values()))), "se_blocks": float(np.std(list(v20.values()), ddof=1) / np.sqrt(len(v20))) if len(v20) > 1 else None, "n_blocks": len(v20),
                                 "paired_20k_minus_10k_mean": float(d.mean()) if len(b) else None, "paired_20k_minus_10k_se": float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else None, "sha256_16": sha(cd)})
    # K1-K0 contrast at 20k from primary contrasts
    pc = RES / "p1_confirm/p1_primary_contrasts.csv"
    for r in rows(pc):
        if int(r["step"]) == 20000 and r["contrast"] in ("K1-K0", "Kfresh-K0"):
            man["local"].append({"quantity": f"{r['contrast']} {r['metric']} at 20k (block {r['block']})", "file": str(pc.relative_to(ROOT)), "key": f"{r['contrast']}/{r['metric']}", "value": float(r["mean_delta"]), "se_over_conditions": float(r["se_over_conditions"]), "sha256_16": sha(pc)})
    # memory long-form at 20k
    ml = RES / "confirm_decomposition/memory_long_form.csv"; M = [r for r in rows(ml) if int(r["step"]) == 20000]
    for arm in sorted({r["arm"] for r in M}):
        for key in ("contrast_perm_owner_minus_own", "loss_gap_fresh_minus_stored", "contrast_stored_minus_new"):
            v = np.array([float(r[key]) for r in M if r["arm"] == arm])
            man["local"].append({"quantity": f"{arm} memory {key} at 20k", "file": str(ml.relative_to(ROOT)), "key": key, "value_mean_blocks": float(v.mean()), "se_blocks": float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else None, "n_blocks": int(len(v)), "sha256_16": sha(ml)})
    pa = RES / "confirm_decomposition/pilot_size_audit_b10.json"; P = json.loads(pa.read_text())
    man["local"].append({"quantity": "pilot-size audit block 10: (K1-K0) c256 vs c1024", "file": str(pa.relative_to(ROOT)), "key": "K1.c256-K0.c256 / K1.c1024-K0.c1024", "value": [P["K1"]["c256"] - P["K0"]["c256"], P["K1"]["c1024"] - P["K0"]["c1024"]], "oracle_artifact_256_1024": [P["K1"]["o_c256"], P["K1"]["o_c1024"]], "sha256_16": sha(pa)})
    # --- server files: values quoted in the drafts (filled/verified when reachable)
    S = a.server_root; A = "assimilation_track_distance_cpu_v1"
    server_entries = [
        (f"{A}/audit_sampling_law_v19.json", "layer_v19 ordinary/U/cross_Q5 CRPS dev-2022", ["ordinary", "U", "cross_Q5"], [0.0372, 0.0337, 0.0331]),
        (f"{A}/unified_score_table_native2f.json", "unified table: raw / v19 / mean-update / R+layer / A+layer ordinary CRPS", ["raw", "v19", "update_only", "R_layer", "A_layer"], [0.0451, 0.0375, 0.0383, 0.0500, 0.0396]),
        (f"{A}/test2023_full_metrics_armC.json", "test-2023 arm C raw / update / v19 full RMSE, MAE, IIEE", ["raw_M5_full_rmse", "update_only_M5_full_rmse", "v19_M10_full_rmse", "v19_M10_full_mae", "v19_M10_full_iiee", "frac_pixels_clipped", "frac_pixels_pinned"], [0.2033, 0.1434, 0.1450, 0.0614, 0.0695, 0.598, 0.013]),
        ("threedvar_repro/main_200d_ourmasks.json", "3D-Var var_100 on our masks, test-2023: RMSE/MAE/IIEE full and track_imitation", ["var_100.full.rmse", "var_100.full.mae", "var_100.full.iiee"], [0.1632, 0.0617, 0.0742]),
        ("threedvar_repro/paired_daily_contrasts.json", "block-bootstrap paired contrasts (v19 − 3D-Var etc.)", ["v19_minus_3dvar.rmse.mean/ci28", "update_minus_3dvar.iiee.mean/ci28"], [[-0.018, [-0.022, -0.013]], [0.032, [0.010, 0.058]]]),
        ("threedvar_repro/convergence_2d.json", "3D-Var loss trajectory, 2 days", ["rel_change_last20"], ["0.03–0.05 %"]),
        ("threedvar_repro/observation_artifacts.json", "observation set hash shared by all rows", ["sha256"], ["c1bcede865b7…"]),
        (f"{A}/ice_matrix_stratified_fixedref.json", "T9 fixed-reference strata", ["REF_dist_obs_gt20 D/F rmse deployment & trainmatched"], [[0.334, 0.400], [0.436, 0.458]]),
        (f"{A}/p6_copula_only_native2f.json", "P6.2 es16 stationary vs gen_same, variogram lag1", ["summary.stationary.es16_fair", "summary.gen_same.es16_fair"], [0.684, 0.689]),
        (f"{A}/p0_ice_observation_passport.json", "deployment obs open-water share", ["open_water_share"], [0.83]),
        (f"{A}/cross_q10_native2f.json", "cross_Q10 two independent legacy runs", ["lay_cross_Q10", "lay_U_A", "lay_U_B", "layA_ordinary", "layB_ordinary"], [0.0330, 0.0338, 0.0334, 0.0373, 0.0369]),
        (f"{A}/hybrid_bgmean_Dspread_dev2022.json", "hybrid bg-mean + D-spread (rejected diagnostic)", ["hybrid_bgmean_Danom_v19_M10.crps_ordinary", "D_v19_M10.crps_ordinary"], [0.0429, 0.0456]),
        (f"{A}/legacy_test2023_ordinary_crps.json", "legacy ordinary CRPS test-2023", ["summary.ordinary", "summary.U"], [0.0481, 0.0414]),
    ]
    for rel, q, keys, vals in server_entries:
        man["server"].append({"quantity": q, "file": f"$R/{rel}", "keys": keys, "values_quoted_in_draft": vals, "sha256_16": sha(S / rel) if S else "PENDING_SERVER"})
    a.output.write_text(json.dumps(man, indent=1, ensure_ascii=False)); print(f"local {len(man['local'])} entries, server {len(man['server'])} -> {a.output}")
    for e in man["local"][:12]:
        print(e["quantity"], {k: (round(v, 4) if isinstance(v, float) else v) for k, v in e.items() if k.startswith(("value", "paired", "se"))})


if __name__ == "__main__":
    main()
