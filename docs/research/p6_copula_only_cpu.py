#!/usr/bin/env python3
"""P6.2 (Astra pack §P6.2, review-3 §9): isolated dependence test at LITERALLY identical per-pixel marginals.

For each held-out dev date (cross-fit as the v19 layer): run the frozen v19 layer once -> 10 layered members. Sort the
values per pixel: this sorted multiset S[p] is kept fixed. Each dependence variant only assigns, per pixel, which member
gets which sorted value, via the rank field of a continuous score:
  stationary  : the layer's own long-scale stationary Gaussian field g (fresh draws)
  gen_same    : standardised anomaly pattern of the generator for the SAME date (c['anom'])
  gen_other   : standardised anomaly pattern of the generator for a RANDOM other dev date (chosen by RNG, never by score)
  mixed       : sqrt(1-LAMBDA) anom_same + sqrt(LAMBDA) g  (the v19 coupling)
  layer_v19   : the layer output itself (must coincide with 'mixed' up to tie handling in atoms)
Asserts: identical sorted multisets across variants; identical plain and fair marginal CRPS (to 1e-12).
Joint scores fixed in advance (from audit_multivariate_scores_cpu.py): energy score on 16x16 patches (fair), variogram
score p=.5 at lags 1..32, fair CRPS of block means at 1/4/16/64 px, and CRPS of fixed projections (domain mean, four
quadrant means). Paired per-date comparison against 'stationary'.  Ordinary (plain) versions are also written.
Run in the server CPU container from the worktree root:
  python3 -m docs.research.p6_copula_only_cpu --samples-dir ... --experiment ... --mode validation_strict_full --output ...
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from docs.research import oof_dispersion_layer_v19 as L
from docs.research.audit_multivariate_scores_cpu import energy_patches, variogram_score, block_means, fair_crps_1d
from docs.research.audit_layer_sampling_law_cpu import crps_plain, crps_fair


def rank_assign(sorted_vals: np.ndarray, field: np.ndarray) -> np.ndarray:
    """sorted_vals (M,H,W) ascending per pixel; field (M,H,W) continuous -> member with rank r gets r-th sorted value."""
    ranks = np.argsort(np.argsort(field, axis=0), axis=0)
    return np.take_along_axis(sorted_vals, ranks, axis=0)


def projections_crps(m: np.ndarray, y: np.ndarray, w: np.ndarray) -> dict:
    H, W = y.shape; out = {}
    regions = {"domain": w, "q_nw": w & (np.arange(H)[:, None] < H // 2) & (np.arange(W)[None, :] < W // 2), "q_ne": w & (np.arange(H)[:, None] < H // 2) & (np.arange(W)[None, :] >= W // 2),
               "q_sw": w & (np.arange(H)[:, None] >= H // 2) & (np.arange(W)[None, :] < W // 2), "q_se": w & (np.arange(H)[:, None] >= H // 2) & (np.arange(W)[None, :] >= W // 2)}
    for name, sel in regions.items():
        if sel.sum() < 100:
            continue
        mm = m[:, sel].mean(1)[:, None]; yy = np.array([y[sel].mean()])
        out[f"proj_{name}_fair"] = fair_crps_1d(mm, yy); out[f"proj_{name}_plain"] = crps_plain(mm, yy)
    return out


def main():
    p = argparse.ArgumentParser(); p.add_argument("--samples-dir", type=Path, required=True); p.add_argument("--experiment", type=Path, required=True)
    p.add_argument("--mode", default="validation_strict_full"); p.add_argument("--output", type=Path, required=True); p.add_argument("--folds", type=int, default=5); p.add_argument("--seed", type=int, default=20260921)
    a = p.parse_args(); rng = np.random.default_rng(a.seed)
    cases = L.load_cases(a.samples_dir, a.experiment, a.mode)
    for c in cases:
        L.prepare(c)
    n = len(cases); fs = int(np.ceil(n / a.folds)); rows = []; variants = ("stationary", "gen_same", "gen_other", "mixed", "layer_v19")
    for k in range(a.folds):
        held = list(range(k * fs, min(n, (k + 1) * fs))); purge = set(held) | {h - 1 for h in held} | {h + 1 for h in held}
        train = [cases[i] for i in range(n) if i not in purge]; params = L.fit(train)
        for i in held:
            c = cases[i]; w = c["water"]; y = c["truth"]
            out = L.simulate(c, params, 10, rng); S = np.sort(out, axis=0)
            other = cases[int(rng.choice([j for j in range(n) if j != i]))]  # random other date, never by score
            g = np.stack([L.smooth_unit(rng.standard_normal(w.shape), {"a": 0.0, "l1": params["ell"]["l1"], "l2": params["ell"]["l2"]}, w, rng.standard_normal(w.shape)) for _ in range(10)])
            fields = {"stationary": g, "gen_same": c["anom"] + 1e-9 * rng.standard_normal(g.shape), "gen_other": other["anom"] + 1e-9 * rng.standard_normal(g.shape), "mixed": np.sqrt(1 - L.LAMBDA) * c["anom"] + np.sqrt(L.LAMBDA) * g}
            ens = {name: rank_assign(S, f) for name, f in fields.items()}; ens["layer_v19"] = out
            # asserts: identical sorted multisets and identical marginal CRPS
            for name, e in ens.items():
                assert np.array_equal(np.sort(e, axis=0), S), name
            ref_plain = crps_plain(ens["stationary"][:, w], y[w]); ref_fair = crps_fair(ens["stationary"][:, w], y[w])
            for name, e in ens.items():
                assert abs(crps_plain(e[:, w], y[w]) - ref_plain) < 1e-12 and abs(crps_fair(e[:, w], y[w]) - ref_fair) < 1e-12, (name, crps_plain(e[:, w], y[w]) - ref_plain)
            r = {"date": c["date"].isoformat(), "marginal_crps_plain": ref_plain, "marginal_crps_fair": ref_fair}
            for name, e in ens.items():
                rr = {"es16_fair": energy_patches(e, y, w, np.random.default_rng(1000 + i)), "vs05": variogram_score(e, y, w, np.random.default_rng(2000 + i))}
                for s_ in (1, 4, 16, 64):
                    bm = block_means(e, s_, w); by = block_means(y[None], s_, w)[0]; rr[f"block{s_}_fair"] = fair_crps_1d(bm, by); rr[f"block{s_}_plain"] = crps_plain(bm, by)
                rr.update(projections_crps(e, y, w)); r[name] = rr
            rows.append(r); print(f"{r['date']}: " + " | ".join(f"{v}: es16 {r[v]['es16_fair']:.4f} vs {r[v]['vs05']:.4g} b16 {r[v]['block16_fair']:.4f} dom {r[v].get('proj_domain_fair', float('nan')):.5f}" for v in variants), flush=True)
    keys = sorted(rows[0]["stationary"]); summary = {}
    for v in variants:
        summary[v] = {k: float(np.mean([r[v][k] for r in rows])) for k in keys}
        summary[v]["dates_better_than_stationary_es16"] = int(sum(r[v]["es16_fair"] < r["stationary"]["es16_fair"] for r in rows))
        summary[v]["dates_better_than_stationary_block16"] = int(sum(r[v]["block16_fair"] < r["stationary"]["block16_fair"] for r in rows))
    a.output.write_text(json.dumps({"summary": summary, "per_date": rows, "marginal_crps_plain_mean": float(np.mean([r["marginal_crps_plain"] for r in rows])), "marginal_crps_fair_mean": float(np.mean([r["marginal_crps_fair"] for r in rows]))}, indent=1))
    print("\nvariant      es16_fair  vs05     block1  block4  block16 block64  proj_dom  dates<stat(es16/b16)")
    for v in variants:
        s = summary[v]; print(f"{v:12s} {s['es16_fair']:.4f}  {s['vs05']:.4g}  {s['block1_fair']:.4f}  {s['block4_fair']:.4f}  {s['block16_fair']:.4f}  {s['block64_fair']:.4f}  {s.get('proj_domain_fair', float('nan')):.5f}  {s['dates_better_than_stationary_es16']}/{s['dates_better_than_stationary_block16']}")


if __name__ == "__main__":
    main()
