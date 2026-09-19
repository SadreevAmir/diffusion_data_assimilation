#!/usr/bin/env python3
"""Audit of the layer's sampling law (review §4): three independent checks on the v15 construction.

(1) Marginal preservation. The mixed score Z = sqrt(1-l)*a_std + sqrt(l)*g has unit variance but is not Gaussian
    (a_std is a sample-standardised anomaly over M members), so U = Phi(Z) need not be uniform and F^{-1}(U) need not
    have the cell marginal F. We measure, on held-out dates, the distribution of U across members x pixels per
    distance stratum (KS distance to uniform), for lambda in {1 (pure Gaussian field), 0.5 (v15), 0 (pure pattern)}.
(2) Two-bank cross-score (review §4.3). The raw ensemble is split into banks A (members 0-4) and B (5-9); the layer is
    run INDEPENDENTLY on each bank (its own standardisation, its own fields, same fitted parameters). Cross-CRPS
    S_cross = 1/(2M) sum_i (|x_i^A - y| + |x_i^B - y|) - 1/(2M^2) sum_ij |x_i^A - x_j^B| estimates the CRPS of the law
    of a random member without assuming independence within a bank. Compared with the plain and fair CRPS of the
    10-member layered ensemble and of each 5-member bank.
(3) Unit test: y ~ U[0,1], ensemble = (m/(M+1)) permuted; fair CRPS expectation M/[6(M+1)] < 1/6 — fair scores reward
    stratified, non-independent members.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from scipy.special import ndtr
import importlib, os
L = importlib.import_module("docs.research." + os.environ.get("LAYER_MODULE", "oof_dispersion_layer_v15"))


def crps_plain(m, y):
    M = m.shape[0]; t1 = np.abs(m - y[None]).mean(0); ms = np.sort(m, 0); idx = np.arange(M)
    t2 = (ms * (2 * idx - M + 1)[:, None]).sum(0) * 2 / (M * M)  # plain (1/M^2) pairwise
    return float(np.mean(t1 - 0.5 * t2))


def crps_fair(m, y):
    M = m.shape[0]; t1 = np.abs(m - y[None]).mean(0); ms = np.sort(m, 0); idx = np.arange(M)
    t2 = (ms * (2 * idx - M + 1)[:, None]).sum(0) * 2 / (M * (M - 1))
    return float(np.mean(t1 - 0.5 * t2))


def crps_cross(a, b, y):
    M = a.shape[0]
    t1 = 0.5 * (np.abs(a - y[None]).mean(0) + np.abs(b - y[None]).mean(0))
    t2 = np.abs(a[:, None] - b[None]).mean((0, 1))
    return float(np.mean(t1 - 0.5 * t2))


def unit_test(M=10, n=200000, seed=0):
    rng = np.random.default_rng(seed); y = rng.random(n)
    ens = np.tile((np.arange(1, M + 1) / (M + 1))[:, None], (1, n)); ens = np.take_along_axis(ens, rng.permuted(np.tile(np.arange(M)[:, None], (1, n)), axis=0), 0)
    return dict(M=M, fair=crps_fair(ens, y), plain=crps_plain(ens, y), expected_fair=M / (6 * (M + 1)), true_continuous=1 / 6)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples-dir", type=Path, required=True); p.add_argument("--experiment", type=Path, required=True)
    p.add_argument("--mode", default="validation_strict_full"); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--folds", type=int, default=5); p.add_argument("--seed", type=int, default=20260919)
    a = p.parse_args(); rng = np.random.default_rng(a.seed)
    print("unit test:", unit_test(10), unit_test(32), flush=True)
    cases = L.load_cases(a.samples_dir, a.experiment, a.mode)
    for c in cases:
        L.prepare(c)
    n = len(cases); fold_size = int(np.ceil(n / a.folds))
    pit = {lam: {s: [] for s, _, _ in [(name, lo, hi) for lo, hi, name in L.STRATA]} for lam in (1.0, 0.5, 0.0)}
    scores = []
    for k in range(a.folds):
        held = list(range(k * fold_size, min(n, (k + 1) * fold_size))); purge = set(held) | {h - 1 for h in held} | {h + 1 for h in held}
        train = [cases[i] for i in range(n) if i not in purge]; params = L.fit(train)
        if hasattr(L, "_mixed_score_cdf") and getattr(L, "F_MIX", None) is None:
            L.F_MIX = L._mixed_score_cdf(cases[0]["anom"].shape[0], L.LAMBDA)
        for i in held:
            c = cases[i]; w = c["water"]; y = c["truth"]
            # (1) PIT of the mixed score per lambda: reconstruct the score exactly as simulate() does
            mu2 = L.kriged_mean(c, params["ell1"], params["s_track"], None); cell = L.cell_index(c["dist_d"], mu2)
            for lam in (1.0, 0.5, 0.0):
                for m in range(c["anom"].shape[0]):
                    g = L.smooth_unit(rng.standard_normal(w.shape), {"a": 0.0, "l1": params["ell"]["l1"], "l2": params["ell"]["l2"]}, w, rng.standard_normal(w.shape))
                    score = np.sqrt(1 - lam) * c["anom"][m] + np.sqrt(lam) * g
                    if lam == 0.5 and "fmix_tab" in params:
                        key = cell[w] if params["fmix_tab"].shape[0] > 7 else L.dist_bin(c["dist_d"])[w]
                        u = L.F_cell(score[w], key, params["fmix_tab"]); uu = np.full(w.shape, np.nan); uu[w] = u; u = uu
                    elif lam == 0.5 and getattr(L, "F_MIX", None) is not None:
                        u = L.F_MIX(score)
                    else:
                        u = ndtr(np.clip(score, -6, 6))
                    for lo, hi, name in L.STRATA:
                        sel = w & (c["dist_d"] >= lo) & (c["dist_d"] <= hi)
                        if sel.sum() > 100:
                            pit[lam][name].append(np.nanquantile(u[sel], np.linspace(0.05, 0.95, 19)))
            # (2) two banks, independent layer runs
            full = L.simulate(c, params, 10, rng)
            cA = dict(c); cA["ens"] = c["ens"][:5]; L.prepare(cA)
            cB = dict(c); cB["ens"] = c["ens"][5:]; L.prepare(cB)
            A = L.simulate(cA, params, 5, rng); B = L.simulate(cB, params, 5, rng)
            scores.append(dict(date=c["date"].isoformat(), full_plain=crps_plain(full[:, w], y[w]), full_fair=crps_fair(full[:, w], y[w]),
                               bankA_fair=crps_fair(A[:, w], y[w]), bankB_fair=crps_fair(B[:, w], y[w]), bankA_plain=crps_plain(A[:, w], y[w]),
                               cross_AB=crps_cross(A[:, w], B[:, w], y[w]), raw_fair=crps_fair(c["ens"][:, w], y[w]), raw_cross=crps_cross(c["ens"][:5][:, w], c["ens"][5:][:, w], y[w])))
            print(f"{c['date']}: full fair {scores[-1]['full_fair']:.4f} plain {scores[-1]['full_plain']:.4f} | banks fair {scores[-1]['bankA_fair']:.4f}/{scores[-1]['bankB_fair']:.4f} cross {scores[-1]['cross_AB']:.4f} | raw fair {scores[-1]['raw_fair']:.4f} cross {scores[-1]['raw_cross']:.4f}", flush=True)
        print(f"fold {k} done", flush=True)
    pit_summary = {}
    for lam in pit:
        pit_summary[str(lam)] = {}
        for name, qs in pit[lam].items():
            if not qs:
                continue
            q = np.mean(qs, 0); ks = float(np.max(np.abs(q - np.linspace(0.05, 0.95, 19))))
            pit_summary[str(lam)][name] = dict(ks_to_uniform=ks, q05=float(q[0]), q50=float(q[9]), q95=float(q[-1]))
    means = {k: float(np.mean([s[k] for s in scores])) for k in scores[0] if k != "date"}
    print("PIT KS to uniform by lambda / stratum:", json.dumps(pit_summary, indent=1))
    print("mean scores:", json.dumps(means, indent=1))
    a.output.write_text(json.dumps(dict(unit_test=[unit_test(10), unit_test(32)], pit=pit_summary, scores=scores, means=means), indent=1))


if __name__ == "__main__":
    main()
