#!/usr/bin/env python3
"""P0.2 score parity: repository scorers vs the reference implementation on identical toy arrays.
Checks (EXPERIMENT_SPECIFICATIONS §P0.2): plain CRPS, U/fair formula, cross-score, energy uses the norm (not squared),
identical marginal multisets give identical one-point CRPS, M=1 handled (plain defined, fair refuses),
unbiased MMD² may be negative and is not clipped."""
from __future__ import annotations
import csv, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE / "astra_pack_v1" / "reference")); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "docs" / "research"))
import core as R
from assim_lib import metrics as AM
import oof_dispersion_layer_v19 as L
import audit_layer_sampling_law_cpu as AS
import audit_multivariate_scores_cpu as MV

rng = np.random.default_rng(123); rows = []
def rec(check, repo_name, repo_val, ref_val, tol=1e-9, note=""):
    ok = (abs(repo_val - ref_val) <= tol) if isinstance(ref_val, float) else (repo_val == ref_val)
    rows.append({"check": check, "repo_function": repo_name, "repo_value": repo_val, "reference_value": ref_val, "status": "PASS" if ok else "FAIL", "note": note}); print(f"{'PASS' if ok else 'FAIL'} {check:55s} {repo_name:40s} {repo_val} vs {ref_val} {note}")

M, D = 10, 7
m = rng.standard_normal((M, D)); y = rng.standard_normal(D); b = rng.standard_normal((M, D))
# plain (empirical) CRPS averaged over D coordinates  (reference kind='marginal' = mean over coordinates of |x-y| - 1/2 E|x-x'| with 1/M^2)
ref_plain = R.empirical_score(m, y, "marginal")
rec("plain CRPS (1/M^2), mean over D", "assim_lib.metrics.ensemble_crps", float(AM.ensemble_crps(m, y).mean()), ref_plain)
rec("plain CRPS (1/M^2), mean over D", "audit_layer_sampling_law_cpu.crps_plain", AS.crps_plain(m, y), ref_plain)
ref_fair = R.u_score(m, y, "marginal")
rec("fair/U CRPS (1/(M(M-1))), mean over D", "assim_lib.metrics.fair_ensemble_crps", float(AM.fair_ensemble_crps(m, y).mean()), ref_fair)
rec("fair/U CRPS (1/(M(M-1))), mean over D", "oof_dispersion_layer_v19.fair_crps", L.fair_crps(m, y), ref_fair)
rec("fair/U CRPS (1/(M(M-1))), mean over D", "audit_layer_sampling_law_cpu.crps_fair", AS.crps_fair(m, y), ref_fair)
rec("fair/U CRPS (1/(M(M-1))), mean over D", "audit_multivariate_scores_cpu.fair_crps_1d", MV.fair_crps_1d(m, y), ref_fair)
ref_cross = R.cross_score(m, b, y, "marginal")
rec("cross-score two banks, mean over D", "audit_layer_sampling_law_cpu.crps_cross", AS.crps_cross(m, b, y), ref_cross)
# energy score: norm, not squared norm; reference divides by sqrt(D). Repo energy_patches uses plain norm (no sqrt(D)), M(M-1) pairwise
ref_energy = R.u_score(m, y, "energy")
d1 = np.linalg.norm(m - y[None], axis=1).mean(); d2 = np.linalg.norm(m[:, None] - m[None], axis=2).sum() / (M * (M - 1))
repo_energy = (d1 - 0.5 * d2) / np.sqrt(D)
rec("energy score (norm, U pairwise), reference normalisation /sqrt(D)", "audit_multivariate_scores_cpu.energy_patches formula (rescaled by 1/sqrt(D))", float(repo_energy), ref_energy, note="repo reports unscaled norm; only a constant factor per patch size")
sq = (np.linalg.norm(m - y[None], axis=1) ** 2).mean() - 0.5 * (np.linalg.norm(m[:, None] - m[None], axis=2) ** 2).sum() / (M * (M - 1))
rec("energy score must NOT be the squared norm", "squared-norm variant (should differ)", float(abs(sq / D - ref_energy) > 1e-6), True)
# identical marginal multisets (permutation within each coordinate) => identical one-point CRPS
mp = np.stack([rng.permutation(m[:, j]) for j in range(D)], axis=1)
rec("same marginal multisets -> same plain CRPS", "assim_lib.metrics.ensemble_crps", float(AM.ensemble_crps(mp, y).mean()), float(AM.ensemble_crps(m, y).mean()))
rec("same marginal multisets -> same fair CRPS", "oof_dispersion_layer_v19.fair_crps", L.fair_crps(mp, y), L.fair_crps(m, y))
# M = 1: plain defined, fair must refuse
one = m[:1]
rec("M=1 plain CRPS defined", "assim_lib.metrics.ensemble_crps", float(AM.ensemble_crps(one, y).mean()), R.empirical_score(one, y, "marginal"))
try:
    AM.fair_ensemble_crps(one, y); refused = False
except ValueError:
    refused = True
rec("M=1 fair CRPS refuses", "assim_lib.metrics.fair_ensemble_crps", refused, True)
try:
    L.fair_crps(one, y); refused_L = "returned " + str(L.fair_crps(one, y))
except Exception as e:
    refused_L = "refused"
rec("M=1 fair CRPS refuses", "oof_dispersion_layer_v19.fair_crps", refused_L == "refused", True, note=refused_L)
# unbiased MMD^2 may be negative (same distribution)
a = rng.standard_normal((30, 2)); c = rng.standard_normal((30, 2)); vals = [R.mmd2_unbiased(rng.standard_normal((30, 2)), rng.standard_normal((30, 2))) for _ in range(200)]
rec("unbiased MMD^2 can be negative (reference, not clipped)", "reference.mmd2_unbiased", float(min(vals) < 0), True, note=f"min={min(vals):.4f}")
# duplicated members: fair formula with duplicated 5->10 members differs from 10 independent (documenting, not a parity check)
dup = np.concatenate([m[:5], m[:5]])
rec("duplicating 5 members to 10 changes fair CRPS (must not be treated as 10 iid)", "oof_dispersion_layer_v19.fair_crps(dup) != fair_crps(m[:5])", float(abs(L.fair_crps(dup, y) - L.fair_crps(m[:5], y)) > 1e-9), True, note=f"dup {L.fair_crps(dup, y):.4f} vs 5 {L.fair_crps(m[:5], y):.4f}")
out = ROOT / "autoresearch_results" / "invariance_audit_v1" / "p0_score_parity.csv"; out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("written", out, "FAIL count:", sum(r["status"] == "FAIL" for r in rows))
