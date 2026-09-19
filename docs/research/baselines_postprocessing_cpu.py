#!/usr/bin/env python3
"""Standard post-processing baselines for the generator ensemble, cross-fitted on the same 40 dates as the layer
(review §5 / roadmap 2.2). All baselines keep the RAW ensemble mean (no assimilation update) so that they isolate the
value of classical spread recalibration:

  emos_scale   : members = mean + s_k (member - mean), s_k = RMSE/spread fitted per distance stratum on training folds
                 (EMOS-type variance inflation; Gneiting et al. 2005 with the ensemble mean as location)
  ecc_q        : out-of-fold cell marginals (same cells as the layer: distance x raw-mean bin) around the RAW mean in the
                 layer's latent, with the generator's member ranks (ECC-Q, Schefzik et al. 2013): tests marginals +
                 generator dependence without kriging
  conformal    : split-conformal symmetric band around the raw mean: members = mean +/- q_alpha per stratum where q is
                 the OOF quantile of |truth - mean|; members placed at quantile levels (k-0.5)/M of the band
Compared with raw and with the frozen layer (v19) on fair CRPS, rank TV, spread/RMSE, cov80, zero mass.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from docs.research import oof_dispersion_layer_v19 as L


def strata_of(c):
    d = c["dist_d"]; out = np.zeros(d.shape, int)
    for i, (lo, hi, _) in enumerate(L.STRATA):
        out[(d >= lo) & (d <= hi)] = i
    return out


def emos_fit(train):
    num = np.zeros(len(L.STRATA)); den = np.zeros(len(L.STRATA))
    for c in train:
        w = c["water"]; st = strata_of(c)[w]; m = c["ens"][:, w]; y = c["truth"][w]; mu = m.mean(0)
        for i in range(len(L.STRATA)):
            sel = st == i
            if sel.sum() > 50:
                num[i] += np.sum((mu[sel] - y[sel]) ** 2); den[i] += np.sum(m[:, sel].var(0, ddof=1))
    return np.sqrt(np.maximum(num, 1e-12) / np.maximum(den, 1e-12))


def emos_apply(c, s):
    w = c["water"]; st = strata_of(c); mu = c["ens"].mean(0); scale = s[st]
    out = np.clip(mu[None] + scale[None] * (c["ens"] - mu[None]), 0.0, L.CAP)
    return np.where(w[None], out, 0.0)


def ecc_fit(train):
    """OOF residual quantiles relative to the raw-mean latent, per (distance, raw-mean) cell."""
    for c in train:
        c["resid_raw"] = c["zt"] - c["mu"]; c["cell_raw"] = L.cell_index(c["dist_d"], c["mu"])
    return L._cell_quantiles_keys(train, "resid_raw", "cell_raw")


def ecc_apply(c, Q):
    w = c["water"]; cell = L.cell_index(c["dist_d"], c["mu"]); M = c["ens"].shape[0]
    zk = np.stack([L.enc(e, w) for e in c["ens"]])[:, w]
    ranks = np.argsort(np.argsort(zk, axis=0), axis=0)  # 0..M-1 per pixel
    u = (ranks + 0.5) / M
    out = np.zeros((M, *w.shape))
    for m in range(M):
        r = L.Finv_cell(u[m], cell[w], Q); zm = np.zeros(w.shape); zm[w] = c["mu"][w] + r; out[m] = np.where(w, L.dec(zm), 0.0)
    return out


def conformal_fit(train, M=10):
    levels = (np.arange(M) + 0.5) / M; qs = np.zeros((len(L.STRATA), M))
    pools = [[] for _ in L.STRATA]
    for c in train:
        w = c["water"]; st = strata_of(c)[w]; e = c["truth"][w] - c["ens"].mean(0)[w]
        for i in range(len(L.STRATA)):
            sel = st == i
            if sel.sum():
                pools[i].append(e[sel] if sel.sum() <= 20000 else np.random.default_rng(1).choice(e[sel], 20000, replace=False))
    for i in range(len(L.STRATA)):
        v = np.concatenate(pools[i]) if pools[i] else np.concatenate([np.concatenate(p) for p in pools if p]); qs[i] = np.quantile(v, levels)
    return qs


def conformal_apply(c, qs):
    w = c["water"]; st = strata_of(c); mu = c["ens"].mean(0); M = qs.shape[1]
    out = np.stack([np.clip(mu + qs[st, m], 0.0, L.CAP) for m in range(M)])
    return np.where(w[None], out, 0.0)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--samples-dir", type=Path, required=True); p.add_argument("--experiment", type=Path, required=True)
    p.add_argument("--mode", default="validation_strict_full"); p.add_argument("--output", type=Path, required=True); p.add_argument("--folds", type=int, default=5); p.add_argument("--seed", type=int, default=20260919)
    a = p.parse_args(); rng = np.random.default_rng(a.seed)
    cases = L.load_cases(a.samples_dir, a.experiment, a.mode)
    for c in cases:
        L.prepare(c)
    n = len(cases); fs = int(np.ceil(n / a.folds)); ens = {k: {} for k in ("raw", "emos_scale", "ecc_q", "conformal", "layer_v19")}
    for k in range(a.folds):
        held = list(range(k * fs, min(n, (k + 1) * fs))); purge = set(held) | {h - 1 for h in held} | {h + 1 for h in held}
        train = [cases[i] for i in range(n) if i not in purge]
        s = emos_fit(train); Q = ecc_fit(train); qs = conformal_fit(train); params = L.fit(train)
        for i in held:
            c = cases[i]; d = c["date"]
            ens["raw"][d] = c["ens"]; ens["emos_scale"][d] = emos_apply(c, s); ens["ecc_q"][d] = ecc_apply(c, Q); ens["conformal"][d] = conformal_apply(c, qs); ens["layer_v19"][d] = L.simulate(c, params, 10, rng)
        print(f"fold {k}: emos scales {np.round(s, 2)}", flush=True)
    res = {name: L.evaluate(cases, e, name, rng) for name, e in ens.items()}
    pd = {}
    for name, e in ens.items():
        pdn = {}; L.evaluate(cases, e, name, rng, pdn); pd[name] = pdn
    dates = sorted(pd["raw"])
    print(f"{'method':12s} {'fairCRPS':>9s} {'RMSE':>7s} {'s/r':>5s} {'rankTV':>7s} {'cov80':>6s} {'zero(t .54)':>11s}  dates<raw")
    for name in ens:
        r = res[name]["all_water"]; cov = np.mean([res[name][s].get("cov80", np.nan) for s in res[name] if s != "all_water"])
        better = int(sum(pd[name][d]["fair_crps"] < pd["raw"][d]["fair_crps"] for d in dates))
        print(f"{name:12s} {r['fair_crps']:9.4f} {r['rmse']:7.4f} {r['spread_over_rmse']:5.2f} {r['rank_tv']:7.3f} {cov:6.2f} {r['member_zero']:11.2f}  {better}/{len(dates)}")
    a.output.write_text(json.dumps({"results": res, "per_date": pd}, indent=1, default=float))


if __name__ == "__main__":
    main()
