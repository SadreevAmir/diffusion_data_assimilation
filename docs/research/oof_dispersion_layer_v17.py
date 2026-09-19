#!/usr/bin/env python3
"""OOF dispersion layer v17 = v16 with the CDF of the mixed score estimated EMPIRICALLY on the training folds from the
generator's actual standardised anomalies (per distance-to-track stratum), instead of the Gaussian Monte-Carlo table:
the audit showed that with real anomalies (many identical members over water, atoms) the mixed score is far from the
Gaussian-anomaly law (PIT KS .06 unchanged by the MC table), so the empirical table is the honest PIT.

v16 = v15 with an EXACT probability-integral transform of the mixed score. The mixed score
sqrt(1-l) a_std + sqrt(l) g has unit variance but is not Gaussian (a_std is a sample-standardised anomaly over M
members: bounded, platykurtic), so Phi(score) is not uniform (audit: KS .05-.06 at lambda=.5, tails compressed).
v16 maps the score through the tabulated CDF F_mix of the mixed score, obtained by Monte Carlo from the same
construction with Gaussian anomalies (M members standardised with ddof=1, mixed with an independent N(0,1)); the
residual non-uniformity then only reflects non-Gaussianity of the generator's anomalies (re-audited).

v15 = v13 with the independent component restricted to the LONG scale of the fitted kernel
(a = 0: no 1-2 px component). Motivation: variogram scores at lags 1-4 px and the high-wavenumber spectrum ratio
(1.46 vs raw 1.11) showed that v13 members carry excess fine-scale roughness coming from the synthetic field; the fine
scale is supplied by the generator pattern instead.

v13 = mixture copula: the Gaussian score of member k is sqrt(1-LAMBDA) * (standardised generator
anomaly of member k) + sqrt(LAMBDA) * (independent stationary two-scale field, as in v9). v12 members were exactly
symmetric around the kriged mean (no sampling variability across the 10 members: sample mean 0, variance 1 at every
pixel), which is not an exchangeable ensemble — its rank histogram was U-shaped at 11-40 px despite s/r 1.1, and fair
CRPS assumes exchangeable members. The synthetic component restores the missing sampling variability.

v12 = v11 with the generator anomalies STANDARDISED pixelwise (a / ensemble std, floor) and
Gaussianised (u = Phi(a_std)) before the cell-wise quantile mapping, so that the OOF marginal is reproduced at every
pixel (v11 reproduced it only cell-wide and inherited the generator's pixelwise over-confidence: truth above all members
12 % of the time, top-bin excess at 11-40 px). The generator contributes its spatial correlation pattern only.

v11 = "generator copula + OOF marginals": the spatial dependence of the members comes from the
generator's own anomalies (member latent minus the member-mean latent), rank-mapped cell by cell onto the OOF residual
law relative to the kriged mean (r = Finv_resid(F_anom(a))). No synthetic Gaussian field: the generator's texture
(leads, edge geometry) is kept, the mean is corrected by kriging and the marginals (dispersion, atoms) by the OOF law.
Motivation: on winter days the generator's members are already well calibrated and textured; the stationary copula of
v3-v10 replaced that texture and lost to raw on 14/40 dates of 2019 even when the mean was not degraded.
Members are one-to-one with the generator members (M = generator M).

v9 = v6 + (a) a PER-DATE microscale nugget a0_d in the kriging prior, chosen for each date by
leave-one-out cross-validation of the track innovations themselves (closed-form LOO of the kriging system; uses only the
day's observations, no truth) — on days whose innovations are spatially incoherent (winter leads) the nugget is large and
nothing is propagated, on days with coherent innovations (summer edge errors) the nugget is ~0, so that isolated track innovations (leads, polynyas)
are not propagated over the kernel range, and (b) day-d track pixels pinned to the observation (obs error 0 in OSSE;
a real system would add N(0, sigma_obs) there).
Motivation: on low-error (winter) days the pooled kriging spread a few large local innovations into the pack
(2019-01-13: 1.6 % of track pixels carry 55 % of the innovation energy; layered RMSE .112 -> .149).

v6 = v3 with the near-track shrinkage removed: the per-cell residual marginals already
encode the conditional shrinkage next to day-d tracks (cells are indexed by distance), so the copula score is only
zeroed ON day-d track pixels (hard taper) and the day-d nugget is reduced so that members honour exact OSSE observations.

v3 docstring:: kriged mean first; residual noise tapered to zero at day-d tracks (no kriging of the noise), then OOF residual law relative to the kriged mean (Gaussian copula, empirical per-cell marginals) for saved assimilation ensembles (dev-2022).

Point analysis = latent mean of the saved members.  Residual law in a latent
with exact atoms (water -> minus distance to ice edge, cap -> censoring):
    z* - mu = b(cell) + s(cell) * g,   g ~ smooth Gaussian field (Gaussian kernel, length ell)
cell = (distance-to-day-d-track bin) x (background class).  b, s, ell are fitted
on training folds only (5 contiguous folds of 8 dates, +-1 date purge), then
members are drawn on held-out dates by conditional simulation with exact
kriging on day-d track pixels and noisy kriging (nugget tau) on d-1/d-2 pixels.
Reports raw vs layered fair CRPS, RMSE, spread/RMSE by distance strata,
exact-zero shares, rank TV and 80% coverage.  Development evidence only;
for test-2023 the parameters would be fitted on all of 2022.
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
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.special import ndtr, ndtri

from assim_lib.config import load_json, merge_config_overrides, resolve_path
from assim_lib.data import build_dataset

CAP = 0.9970703125
DELTA_CAP = 0.01
GAMMA = 0.02  # latent slope per pixel in open water (declared)
R_MAX = 40.0
TAU_STALE = 0.02
TAPER_PX = 2.0
ANOM_SD_FLOOR = 1e-3
LAMBDA = 0.5  # weight of the synthetic stationary field in the score
TAU_DAY = 0.003  # declared observation error in latent units on day-d track pixels (OSSE: obs exact)
EDGE_BAND_PX = 5.0  # sub-threshold analysis-mean values are kept within this band of the mean ice edge
MEAN_ZERO_THRESHOLD = 0.15  # analysis-mean values below the IIEE edge threshold are encoded as open water
DIST_EDGES = [0, 2, 5, 10, 20, 40]  # bins: 0, 1-2, 3-5, 6-10, 11-20, 21-40, >40
LAGS = (1, 2, 4, 8, 16)
N_QUANT = 401
Q_GRID = np.linspace(0.0, 1.0, N_QUANT)
U_CLIP = 1e-4
STRATA = [(0, 0, "on_track_d"), (1, 2, "d1_2"), (3, 5, "d3_5"), (6, 10, "d6_10"), (11, 20, "d11_20"), (21, 40, "d21_40"), (41, 10_000, "d_gt_40")]


def enc(a: np.ndarray, water: np.ndarray) -> np.ndarray:
    z = a.astype(np.float64).copy()
    ice = (a > 0) & water
    if ice.any():
        d = distance_transform_edt(~ice)
    else:
        d = np.full(a.shape, R_MAX)
    z[~ice] = -GAMMA * np.minimum(d[~ice], R_MAX)
    z[a >= CAP] = CAP + DELTA_CAP
    return np.where(water, z, 0.0)


def dec(z: np.ndarray) -> np.ndarray:
    return np.clip(z, 0.0, CAP)


def bg_class(bg: np.ndarray) -> np.ndarray:
    c = np.ones(bg.shape, dtype=np.int64)
    c[bg <= 0] = 0
    c[bg >= 0.95] = 2
    return c


def dist_bin(d: np.ndarray) -> np.ndarray:
    return np.searchsorted(np.array(DIST_EDGES, dtype=float), d, side="left")  # 0..6


MU_EDGES = np.array([-0.5, 0.0, 0.15, 0.3, 0.5, 0.7, 0.85, 0.92, 0.96, 0.98, 0.99, CAP], dtype=float)  # 13 bins, finer near the cap


NMU = len(MU_EDGES) + 1


def mu_bin(mu: np.ndarray) -> np.ndarray:
    return np.searchsorted(MU_EDGES, mu, side="right")  # 0..10


def cell_index(d: np.ndarray, mu: np.ndarray) -> np.ndarray:
    return dist_bin(d) * NMU + mu_bin(mu)


def _unit_field(x: np.ndarray, ell: float, water: np.ndarray) -> np.ndarray:
    """Mask-aware Gaussian smoothing of white noise with exact pointwise unit variance."""
    sig = ell / np.sqrt(2.0)  # filtered white noise has correlation exp(-h^2/(4 sig^2)) = rho with length ell
    wf = water.astype(float)
    num = gaussian_filter(np.where(water, x, 0.0), sig, mode="constant")
    den = gaussian_filter(wf, sig, mode="constant")
    # var(num) = sum k^2 w = G_{sig/sqrt2}(w) / (4 pi sig^2)
    k2w = gaussian_filter(wf, sig / np.sqrt(2.0), mode="constant") / (4 * np.pi * sig**2)
    std_local = np.sqrt(np.maximum(k2w, 1e-12)) / np.maximum(den, 1e-6)
    y = np.where(den > 1e-6, num / np.maximum(den, 1e-6), 0.0)
    return np.where(water, y / std_local, 0.0)


def smooth_unit(x: np.ndarray, ell: dict, water: np.ndarray, x2: np.ndarray | None = None) -> np.ndarray:
    """Unit-variance field with two-scale correlation a*rho_l1 + (1-a)*rho_l2 (independent white inputs)."""
    a = ell["a"]
    if x2 is None:
        x2 = np.random.default_rng(0).standard_normal(x.shape)
    return np.sqrt(a) * _unit_field(x, ell["l1"], water) + np.sqrt(1 - a) * _unit_field(x2, ell["l2"], water)


def rho_apply(field_on_T: np.ndarray, ell: dict) -> np.ndarray:
    """Apply the two-scale covariance operator to a field supported on T."""
    a = ell["a"]
    out = a * gaussian_filter(field_on_T, ell["l1"], mode="constant") * (2 * np.pi * ell["l1"] ** 2)
    out += (1 - a) * gaussian_filter(field_on_T, ell["l2"], mode="constant") * (2 * np.pi * ell["l2"] ** 2)
    return out


def rho_matrix(d2: np.ndarray, ell: dict) -> np.ndarray:
    return ell["a"] * np.exp(-d2 / (2 * ell["l1"] ** 2)) + (1 - ell["a"]) * np.exp(-d2 / (2 * ell["l2"] ** 2))


def fair_crps(m: np.ndarray, y: np.ndarray) -> float:
    M = m.shape[0]
    t1 = np.abs(m - y[None]).mean(0)
    t2 = np.zeros_like(y)
    for i in range(M):
        t2 += np.abs(m[i][None] - m).sum(0)
    return float((t1 - t2 / (2 * M * (M - 1))).mean())


def rank_tv(m: np.ndarray, y: np.ndarray, rng) -> float:
    M = m.shape[0]
    below = (m < y[None]).sum(0)
    ties = (m == y[None]).sum(0)
    r = below + np.floor(rng.random(y.shape) * (ties + 1)).astype(int)
    h = np.bincount(r, minlength=M + 1).astype(float)
    h /= h.sum()
    return float(0.5 * np.abs(h - 1 / (M + 1)).sum())


def load_cases(samples_dir: Path, experiment: Path, mode: str, split: str = "valid") -> list[dict]:
    e = load_json(experiment)
    dc = merge_config_overrides(load_json(resolve_path(e["data_config"], experiment.parent)), e.get("data_overrides"))
    dc = merge_config_overrides(dc, e["modes"][mode].get("data_overrides", {}))
    fields = list(dc["fields"])
    dc["observed_channels"] = [fields.index("siconc")]
    ds = build_dataset(dc, split=split)
    cases = []
    for f in sorted(glob.glob(str(samples_dir / "*.npz"))):
        d = date.fromisoformat(Path(f).stem.split("_")[1])
        z = np.load(f)
        valid = z["valid_mask"]
        vm = torch.as_tensor(valid[None].astype(np.float32))
        saved = ds.assimilation_range
        ds.assimilation_range = 1
        _, day_mask, _ = ds._sral_track_observations(d, ds.hour_index, vm)
        ds.assimilation_range = saved
        day_d = (day_mask[0].numpy() > 0)[: valid.shape[0], : valid.shape[1]] & valid
        cases.append({
            "date": d, "ens": z["analysis_ensemble"].astype(np.float64), "truth": z["truth"].astype(np.float64),
            "bg": z["background"].astype(np.float64), "obs": z["obs_values"].astype(np.float64),
            "cond": z["conditioning_mask"] & valid, "day_d": day_d, "water": valid,
        })
    return cases


def prepare(c: dict) -> None:
    w = c["water"]
    c["zt"] = enc(c["truth"], w)
    mean = c["ens"].mean(0)
    core = (mean >= MEAN_ZERO_THRESHOLD) & w
    band = (distance_transform_edt(~core) <= EDGE_BAND_PX) if core.any() else np.zeros_like(core)
    keep = core | (band & (mean > 0))
    c["mu"] = enc(np.where(keep, mean, 0.0), w)
    c["resid"] = c["zt"] - c["mu"]
    c["dist_d"] = distance_transform_edt(~c["day_d"]) if c["day_d"].any() else np.full(w.shape, 1e4)
    c["cell"] = cell_index(c["dist_d"], c["mu"])
    # Observation latent: observed positive values as-is (cap -> CAP+delta); observed open water
    # is encoded consistently with the analysis-mean edge (at least one pixel into the water side).
    zobs = c["obs"].astype(np.float64).copy()
    zobs[c["obs"] >= CAP] = CAP + DELTA_CAP
    zobs = np.where(c["obs"] <= 0, np.minimum(c["mu"], -GAMMA), zobs)
    c["zobs"] = np.where(w, zobs, 0.0)
    zk = np.stack([enc(e, w) for e in c["ens"]])
    a = zk - zk.mean(0, keepdims=True)
    sd = a.std(0, ddof=1)
    c["anom"] = a / np.maximum(sd, ANOM_SD_FLOOR)[None]  # standardised anomalies (pattern only)


def _cell_quantiles(train: list[dict]) -> np.ndarray:
    n_cells = 7 * NMU
    pools = [[] for _ in range(n_cells)]
    rng = np.random.default_rng(1)
    for c in train:
        w = c["water"]; r = c["resid"][w]; k = c["cell"][w]
        for i in range(n_cells):
            v = r[k == i]
            if v.size > 20000:
                v = rng.choice(v, 20000, replace=False)
            pools[i].append(v)
    Q = np.zeros((n_cells, N_QUANT))
    for i in range(n_cells):
        v = np.concatenate(pools[i]) if pools[i] else np.zeros(0)
        if v.size < 500:
            mb = i % NMU
            v = np.concatenate([np.concatenate(pools[db * NMU + mb]) for db in range(7) if pools[db * NMU + mb]]) if any(pools[db * NMU + mb] for db in range(7)) else np.zeros(0)
            if v.size < 500:
                v = np.concatenate([np.concatenate(pl) for pl in pools if pl])
        Q[i] = np.quantile(v, Q_GRID)
    return Q


def F_cell(r: np.ndarray, cell: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Empirical CDF value of residual r in its cell (vectorised interpolation)."""
    u = np.empty_like(r, dtype=np.float64)
    for i in np.unique(cell):
        m = cell == i
        u[m] = np.interp(r[m], Q[i], Q_GRID)
    return np.clip(u, U_CLIP, 1 - U_CLIP)


def Finv_cell(u: np.ndarray, cell: np.ndarray, Q: np.ndarray) -> np.ndarray:
    r = np.empty_like(u, dtype=np.float64)
    for i in np.unique(cell):
        m = cell == i
        r[m] = np.interp(u[m], Q_GRID, Q[i])
    return r



A0_GRID = (0.0, 0.1, 0.3, 1.0, 3.0)


def _track_system(c: dict, ell: dict, var_scale: float = 1.0, a0: float = 0.0):
    T = c["cond"]
    idx = np.flatnonzero(T.ravel()); nT = idx.size
    ti = np.unravel_index(idx, T.shape)
    nug = np.where(c["day_d"][ti], TAU_DAY**2, TAU_STALE**2) / max(var_scale, 1e-6) + a0
    yy, xx = np.array(ti[0], float), np.array(ti[1], float)
    d2 = (yy[:, None] - yy[None]) ** 2 + (xx[:, None] - xx[None]) ** 2
    Sigma_TT = rho_matrix(d2, ell) + np.diag(nug) + 1e-6 * np.eye(nT)
    L = np.linalg.cholesky(Sigma_TT)

    def krig(v_T: np.ndarray) -> np.ndarray:
        wgt = np.linalg.solve(L.T, np.linalg.solve(L, v_T))
        g = np.zeros(T.shape); g[ti] = wgt
        return rho_apply(g, ell)

    def loo_rms(v_T: np.ndarray) -> float:
        """Leave-one-out prediction error of the innovations under this prior: e_i = (S^-1 v)_i / (S^-1)_ii (Dubrule 1983)."""
        Sinv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(nT)))
        e = (Sinv @ v_T) / np.diag(Sinv)
        return float(np.sqrt(np.mean(e**2)))

    return ti, nug, krig, loo_rms


def select_a0(c: dict, ell: dict, s_track: float) -> tuple[float, dict]:
    """Per-date relative nugget by LOO CV of the day's own track innovations (no truth involved)."""
    scores = {}
    for a0 in A0_GRID:
        ti, _, _, loo_rms = _track_system(c, ell, var_scale=s_track**2, a0=a0)
        scores[a0] = loo_rms((c["zobs"] - c["mu"])[ti] / s_track)
    return min(scores, key=scores.get), scores


def kriged_mean(c: dict, ell: dict, s_track: float, a0: float | None = None) -> np.ndarray:
    """Deterministic analysis: mu' = mu + K r_T with a stationary unit-correlation prior of scale s_track and relative microscale nugget a0 (per date by LOO if None)."""
    if a0 is None:
        a0, scores = select_a0(c, ell, s_track)
        c["a0"] = a0; c["a0_loo"] = scores
    ti, nug, krig, _ = _track_system(c, ell, var_scale=s_track**2, a0=a0)
    r_T = (c["zobs"] - c["mu"])[ti] / s_track
    upd = s_track * krig(r_T)
    return np.where(c["water"], c["mu"] + upd, 0.0)


def _kernel_from_scores(train: list[dict], key_resid: str, key_cell: str, Q: np.ndarray) -> dict:
    cors = {lag: [] for lag in LAGS}
    for c in train:
        w = c["water"]
        g = np.zeros(w.shape)
        g[w] = ndtri(F_cell(c[key_resid][w], c[key_cell][w], Q))
        for lag in LAGS:
            a1 = g[:, lag:]; a2 = g[:, :-lag]; m = w[:, lag:] & w[:, :-lag]
            b1 = g[lag:, :]; b2 = g[:-lag, :]; mv = w[lag:, :] & w[:-lag, :]
            num = (a1 * a2)[m].sum() + (b1 * b2)[mv].sum()
            den = (a1 * a1)[m].sum() + (b1 * b1)[mv].sum()
            cors[lag].append(num / max(den, 1e-9))
    lag_arr = np.array(LAGS, float); cor_arr = np.array([np.mean(cors[l]) for l in LAGS])
    best = None
    for l1 in (1.0, 1.5, 2.0, 3.0, 4.0):
        for l2 in (8.0, 12.0, 16.0, 24.0, 32.0, 48.0):
            g1 = np.exp(-lag_arr**2 / (2 * l1**2)); g2 = np.exp(-lag_arr**2 / (2 * l2**2))
            denom = np.sum((g1 - g2) ** 2)
            a_hat = float(np.clip(np.sum((cor_arr - g2) * (g1 - g2)) / max(denom, 1e-12), 0.0, 1.0))
            err = float(np.sum((a_hat * g1 + (1 - a_hat) * g2 - cor_arr) ** 2))
            if best is None or err < best[0]:
                best = (err, a_hat, l1, l2)
    _, a_mix, l1, l2 = best
    return {"a": a_mix, "l1": l1, "l2": l2, "empirical_corr": {int(l): float(v) for l, v in zip(LAGS, cor_arr)}}


def _cell_quantiles_keys(train: list[dict], key_resid: str, key_cell: str) -> np.ndarray:
    n_cells = 7 * NMU
    pools = [[] for _ in range(n_cells)]
    rng = np.random.default_rng(1)
    for c in train:
        w = c["water"]; r = c[key_resid][w]; k = c[key_cell][w]
        for i in range(n_cells):
            v = r[k == i]
            if v.size > 20000:
                v = rng.choice(v, 20000, replace=False)
            pools[i].append(v)
    Q = np.zeros((n_cells, N_QUANT))
    for i in range(n_cells):
        v = np.concatenate(pools[i]) if pools[i] else np.zeros(0)
        if v.size < 500:
            mb = i % NMU
            v = np.concatenate([np.concatenate(pools[db * NMU + mb]) for db in range(7) if pools[db * NMU + mb]]) if any(pools[db * NMU + mb] for db in range(7)) else np.zeros(0)
            if v.size < 500:
                v = np.concatenate([np.concatenate(pl) for pl in pools if pl])
        Q[i] = np.quantile(v, Q_GRID)
    return Q


def fit(train: list[dict]) -> dict:
    # pass 1: marginal residual law -> kernel and a track-scale for the deterministic update
    Q1 = _cell_quantiles_keys(train, "resid", "cell")
    k1 = _kernel_from_scores(train, "resid", "cell", Q1)
    s_track = float(np.sqrt(np.mean(np.concatenate([c["resid"][c["water"]] ** 2 for c in train]))))
    # pass 2: kriged mean on training cases, residual law relative to it, cells by the kriged mean
    ell1 = {"a": k1["a"], "l1": k1["l1"], "l2": k1["l2"]}
    for c in train:
        c["mu2"] = kriged_mean(c, ell1, s_track, None)
        c["resid2"] = c["zt"] - c["mu2"]
        c["cell2"] = cell_index(c["dist_d"], c["mu2"])
    print("    a0 per training date:", {c["date"].isoformat(): c["a0"] for c in train}, flush=True)
    Q2 = _cell_quantiles_keys(train, "resid2", "cell2")
    Qa = _anomaly_quantiles(train)
    k2 = _kernel_from_scores(train, "resid2", "cell2", Q2)
    # v17: empirical CDF of the mixed score per distance bin from the training cases' own anomalies
    rng_f = np.random.default_rng(5); ell_f = {"a": 0.0, "l1": k2["l1"], "l2": k2["l2"]}
    pools = [[] for _ in range(7)]
    for c in train:
        w = c["water"]; db = dist_bin(c["dist_d"])
        for m in range(c["anom"].shape[0]):
            g = smooth_unit(rng_f.standard_normal(w.shape), ell_f, w, rng_f.standard_normal(w.shape))
            sc = (np.sqrt(1 - LAMBDA) * c["anom"][m] + np.sqrt(LAMBDA) * g)[w]; dbw = db[w]
            for b in range(7):
                v = sc[dbw == b]
                if v.size:
                    pools[b].append(v if v.size <= 20000 else rng_f.choice(v, 20000, replace=False))
    fmix_tab = np.zeros((7, N_QUANT))
    for b in range(7):
        v = np.concatenate(pools[b]) if pools[b] else np.concatenate([np.concatenate(pl) for pl in pools if pl])
        fmix_tab[b] = np.quantile(v, Q_GRID)
    return {"Q": Q2, "Qa": Qa, "ell": {"a": k2["a"], "l1": k2["l1"], "l2": k2["l2"]}, "ell1": ell1, "s_track": s_track,
            "empirical_corr": k2["empirical_corr"], "empirical_corr_pass1": k1["empirical_corr"], "fmix_tab": fmix_tab}


def _anomaly_quantiles(train: list[dict]) -> np.ndarray:
    """Per-cell empirical quantiles of the generator's member anomalies (latent), all members pooled."""
    n_cells = 7 * NMU
    pools = [[] for _ in range(n_cells)]
    rng = np.random.default_rng(2)
    for c in train:
        w = c["water"]; k = c["cell2"][w]
        for i in np.unique(k):
            v = c["anom"][:, w][:, k == i].ravel()
            if v.size > 40000:
                v = rng.choice(v, 40000, replace=False)
            pools[i].append(v)
    cat = lambda idx: np.concatenate([np.concatenate(pools[j]) for j in idx if pools[j]]) if any(pools[j] for j in idx) else np.zeros(0)
    Q = np.zeros((n_cells, N_QUANT))
    for i in range(n_cells):
        mb = i % NMU
        v = cat([i])
        if v.size < 500:
            v = cat([d * NMU + mb for d in range(7)])
        if v.size < 500:
            v = cat(range(n_cells))
        Q[i] = np.quantile(v, Q_GRID)
    return Q


_FMIX_CACHE = {}


def _mixed_score_cdf(M: int, lam: float, n: int = 400_000, seed: int = 11):
    """Tabulated CDF of sqrt(1-lam)*a_std + sqrt(lam)*g where a_std is one of M standardised (ddof=1) Gaussian draws."""
    key = (M, lam)
    if key not in _FMIX_CACHE:
        r = np.random.default_rng(seed); a = r.standard_normal((n, M)); a = (a - a.mean(1, keepdims=True)) / np.maximum(a.std(1, ddof=1, keepdims=True), 1e-6)
        z = np.sqrt(1 - lam) * a[:, 0] + np.sqrt(lam) * r.standard_normal(n); z.sort()
        grid = np.linspace(-6, 6, 2401); cdf = np.searchsorted(z, grid, side="right") / n
        _FMIX_CACHE[key] = (grid, cdf)
    grid, cdf = _FMIX_CACHE[key]
    return lambda x: np.interp(np.clip(x, -6, 6), grid, cdf)


F_MIX = None


def simulate(c: dict, params: dict, M: int, rng) -> np.ndarray:
    global F_MIX
    if F_MIX is None:
        F_MIX = _mixed_score_cdf(c["anom"].shape[0], LAMBDA)
    w = c["water"]; Q = params["Q"]; ell = params["ell"]
    mu2 = kriged_mean(c, params["ell1"], params["s_track"], None)
    pin = c["day_d"] & w
    cell = cell_index(c["dist_d"], mu2)
    # residual noise: unit-variance two-scale field, tapered to zero on/near day-d track pixels;
    # marginal law per (distance, kriged-mean) cell comes from the OOF empirical quantiles
    Qa = params["Qa"]; Mg = c["anom"].shape[0]
    out = np.zeros((Mg, *w.shape))
    for m in range(Mg):
        # rank-map the generator's anomaly through the cell's anomaly CDF onto the cell's OOF residual law
        g = smooth_unit(rng.standard_normal(w.shape), {"a": 0.0, "l1": ell["l1"], "l2": ell["l2"]}, w, rng.standard_normal(w.shape))  # v15: long scale only
        score = np.sqrt(1 - LAMBDA) * c["anom"][m] + np.sqrt(LAMBDA) * g
        u = np.clip(F_cell(score[w], dist_bin(c["dist_d"])[w], params["fmix_tab"]), U_CLIP, 1 - U_CLIP)  # v17: empirical PIT per distance bin
        r = Finv_cell(u, cell[w], Q)
        zm = np.zeros(w.shape); zm[w] = mu2[w] + r
        zm[pin] = c["zobs"][pin]  # exact observations of day d (OSSE)
        out[m] = np.where(w, dec(zm), 0.0)
    return out


def evaluate(cases: list[dict], members_by_case: dict, label: str, rng, per_date: dict | None = None) -> dict:
    acc = defaultdict(lambda: defaultdict(list))
    for c in cases:
        m = members_by_case[c["date"]]; y = c["truth"]; w = c["water"]
        mean = m.mean(0); sd = m.std(0, ddof=1)
        if per_date is not None:
            per_date[c["date"].isoformat()] = {"fair_crps": fair_crps(m[:, w], y[w]), "rmse": float(np.sqrt(np.mean((mean[w] - y[w]) ** 2))),
                                                "member_cap_mass": float(np.mean(m[:, w] >= CAP)), "truth_cap_mass": float(np.mean(y[w] >= CAP))}
        for lo, hi, name in STRATA:
            sel = w & (c["dist_d"] >= lo) & (c["dist_d"] <= hi)
            if sel.sum() < 50:
                continue
            a = acc[name]; mm = m[:, sel]; yy = y[sel]
            a["rmse"].append(float(np.sqrt(np.mean((mean[sel] - yy) ** 2))))
            a["spread"].append(float(np.sqrt(np.mean(sd[sel] ** 2))))
            a["fair_crps"].append(fair_crps(mm, yy))
            a["rank_tv"].append(rank_tv(mm, yy, rng))
            lo_q, hi_q = np.quantile(mm, [0.1, 0.9], axis=0)
            a["cov80"].append(float(np.mean((yy >= lo_q) & (yy <= hi_q))))
            a["truth_zero"].append(float(np.mean(yy == 0)))
            a["member_zero"].append(float(np.mean(mm == 0)))
            a["member_cap"].append(float(np.mean(mm >= CAP)))
            a["truth_cap"].append(float(np.mean(yy >= CAP)))
        a = acc["all_water"]; mm = m[:, w]; yy = y[w]
        a["rmse"].append(float(np.sqrt(np.mean((mean[w] - yy) ** 2)))); a["spread"].append(float(np.sqrt(np.mean(sd[w] ** 2))))
        a["fair_crps"].append(fair_crps(mm, yy)); a["rank_tv"].append(rank_tv(mm, yy, rng))
        a["truth_zero"].append(float(np.mean(yy == 0))); a["member_zero"].append(float(np.mean(mm == 0)))
    out = {}
    for name, d in acc.items():
        s = {k: float(np.mean(v)) for k, v in d.items()}
        s["spread_over_rmse"] = s["spread"] / max(s["rmse"], 1e-9)
        out[name] = s
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--samples-dir", type=Path, required=True)
    p.add_argument("--experiment", type=Path, required=True)
    p.add_argument("--mode", default="validation_strict_full")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--members", type=int, default=10)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260918)
    p.add_argument("--save-members-dir", type=Path, default=None)
    p.add_argument("--apply-samples-dir", type=Path, default=None, help="frozen application: fit on ALL dev cases, apply to these cases (test split)")
    p.add_argument("--apply-mode", default="test_strict_full")
    p.add_argument("--apply-experiment", type=Path, default=None, help="experiment file for the apply cases (default: same as --experiment)")
    p.add_argument("--apply-split", default="test")
    p.add_argument("--save-members-dates", nargs="*", default=("2022-01-16", "2022-04-11", "2022-07-15"))
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    cases = load_cases(a.samples_dir, a.experiment, a.mode)
    for c in cases:
        prepare(c)
    if a.apply_samples_dir is not None:
        params = fit(cases)
        test_cases = load_cases(a.apply_samples_dir, a.apply_experiment or a.experiment, a.apply_mode, split=a.apply_split)
        for c in test_cases:
            prepare(c)
        layered = {c["date"]: simulate(c, params, a.members, rng) for c in test_cases}
        fold_params = [{"fold": "all_dev", "ell": params["ell"], "ell1": params["ell1"], "s_track": params["s_track"], "empirical_corr": params["empirical_corr"]}]
        cases = test_cases
        print(f"applied frozen dev-fitted layer to {len(test_cases)} test cases", flush=True)
    else:
        n = len(cases); fold_size = int(np.ceil(n / a.folds))
        layered = {}
        fold_params = []
        for k in range(a.folds):
            held = list(range(k * fold_size, min(n, (k + 1) * fold_size)))
            purge = set(held) | {h - 1 for h in held} | {h + 1 for h in held}
            train = [cases[i] for i in range(n) if i not in purge]
            params = fit(train)
            fold_params.append({"fold": k, "ell": params["ell"], "ell1": params["ell1"], "s_track": params["s_track"], "empirical_corr": params["empirical_corr"], "empirical_corr_pass1": params["empirical_corr_pass1"]})
            for i in held:
                layered[cases[i]["date"]] = simulate(cases[i], params, a.members, rng)
            print(f"fold {k}: kernel={params['ell']} corr={params['empirical_corr']}", flush=True)
    raw = {c["date"]: c["ens"] for c in cases}
    pd_raw, pd_lay = {}, {}
    res = {"raw": evaluate(cases, raw, "raw", rng, pd_raw), "layered": evaluate(cases, layered, "layered", rng, pd_lay), "fold_params": fold_params,
           "config": {"tau_day": TAU_DAY, "mean_zero_threshold": MEAN_ZERO_THRESHOLD, "gamma": GAMMA, "r_max": R_MAX, "tau_stale": TAU_STALE, "delta_cap": DELTA_CAP, "members": a.members, "folds": a.folds}}
    dates = sorted(pd_raw)
    boot = {}
    for metric in ("fair_crps", "rmse"):
        delta = np.array([pd_lay[d][metric] - pd_raw[d][metric] for d in dates])
        bs = [float(np.mean(delta[rng.integers(0, len(delta), len(delta))])) for _ in range(4000)]
        boot[metric] = {"mean_delta": float(delta.mean()), "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))], "dates_improved": int((delta < 0).sum()), "n_dates": len(delta)}
    res["paired_bootstrap_layered_minus_raw"] = boot
    res["cap_mass"] = {"truth": float(np.mean([pd_raw[d]["truth_cap_mass"] for d in dates])), "raw": float(np.mean([pd_raw[d]["member_cap_mass"] for d in dates])), "layered": float(np.mean([pd_lay[d]["member_cap_mass"] for d in dates]))}
    res["per_date"] = {"raw": pd_raw, "layered": pd_lay}
    res["a0_per_date"] = {c["date"].isoformat(): {"a0": c.get("a0"), "loo": {str(k): v for k, v in c.get("a0_loo", {}).items()}} for c in cases}
    if a.save_members_dir is not None:
        a.save_members_dir.mkdir(parents=True, exist_ok=True)
        for c in cases:
            if "all" in a.save_members_dates or c["date"].isoformat() in a.save_members_dates:
                np.savez_compressed(a.save_members_dir / f"layered_{c['date'].isoformat()}.npz", members=layered[c["date"]].astype(np.float32), raw=c["ens"].astype(np.float32), truth=c["truth"].astype(np.float32), background=c["bg"].astype(np.float32), day_d=c["day_d"], cond=c["cond"], water=c["water"])
    print("bootstrap:", json.dumps(boot), "| cap mass:", res["cap_mass"])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(res, indent=2, default=float))
    hdr = f"{'stratum':12s} {'RMSE raw/lay':>16s} {'s/r raw/lay':>14s} {'fCRPS raw/lay':>16s} {'rankTV raw/lay':>16s} {'cov80 raw/lay':>14s} {'zero t/raw/lay':>16s}"
    print(hdr)
    for name in res["raw"]:
        r, l = res["raw"][name], res["layered"][name]
        print(f"{name:12s} {r['rmse']:7.4f}/{l['rmse']:7.4f} {r['spread_over_rmse']:6.2f}/{l['spread_over_rmse']:6.2f} {r['fair_crps']:7.4f}/{l['fair_crps']:7.4f} {r['rank_tv']:7.3f}/{l['rank_tv']:7.3f} {r.get('cov80',float('nan')):6.2f}/{l.get('cov80',float('nan')):6.2f} {r['truth_zero']:.2f}/{r['member_zero']:.2f}/{l['member_zero']:.2f}")


if __name__ == "__main__":
    main()
