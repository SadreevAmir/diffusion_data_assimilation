#!/usr/bin/env python3
"""P2 (Astra pack v1, EXPERIMENT_SPECIFICATIONS §P2): quantitative prediction and same-budget correction, no new training.

For a trained network (final checkpoint) and each category c:
  * budget M = 64 members; L independent keys x m = 64/L members per key, L in {1,2,4,8,16,64};
    32 independent forecast runs per (c, L); ORDINARY empirical kernel score and marginal CRPS of the 64-member
    ensemble against the 32 shared fresh truths of the validation bank (per category: the 32 conditions' truth
    replicates of that category, first condition's bank is used for all runs -> identical truths across L).
  * label-free ingredients: V_c (kernel: 1/2 E MMD^2(Q_K, Q_K'); CRPS-features: 1/2 E[|X-Z| - |X-X'|] per coordinate),
    W_c = 1 - E k(X,X') at a common key (kernel) / (1/2) E|X-X'| per coordinate (CRPS),
    R(Qbar)_c from an iid new-key-per-member bank of 4096 members (minus W/4096).
  * prediction (T4.1): E S(Qhat_{L,m}) = R(Qbar) + V/L + W/M; T4.2: S(L=1) - S(L=64) = (1 - 1/64) V.
Negative controls: K0 (invariant by construction) and Kfresh; the informative-key Bernoulli example (T6) is enumerated
analytically in `informative_key_control()`.
Executed BEFORE the review that the protocol requires (user instruction: work autonomously); flagged in the report.
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import p1_pilot as P
R = P.R; SCALES = P.SCALES; NCAT, KDIM, YDIM = P.NCAT, P.KDIM, P.YDIM
OUT = P.OUT.parent / "p2_correction"
LS = (1, 2, 4, 8, 16, 64); M = 64; RUNS = 32


def crps_features_v(X, Xp, Z):
    """1/2 E[|X-Z| - |X-X'|] averaged over coordinates: V in the CRPS (L2 of CDFs) geometry, marginal-CRPS normalisation."""
    return 0.5 * (np.abs(X - Z).mean(1) - np.abs(X - Xp).mean(1))


def run(block: int, arm: str, resume: bool) -> None:
    torch = P._torch(); d = OUT / f"block{block}_{arm}"
    if d.exists() and not resume:
        raise FileExistsError(d)
    d.mkdir(parents=True, exist_ok=True)
    s = P.seeds(block); step = P.CFG["training"]["steps"]; model = P.load_model(torch, block, arm, step)
    bank = np.load(P.block_dir(block) / "validation_bank.npz"); cats_bank = bank["cats"]; truths_bank = bank["truths"]
    truths = {c: truths_bank[np.flatnonzero(cats_bank == c)[0]] for c in range(NCAT)}  # 32 fresh truths per category (shared)
    rng = np.random.default_rng(np.random.SeedSequence([s["validation"], step, 31, block]))
    rows = []
    # ---- ingredients (label-free): V, W, R(Qbar)
    ncal = 4096; cc = np.repeat(np.arange(NCAT), ncal)
    K = rng.standard_normal((len(cc), KDIM)); K2 = rng.standard_normal((len(cc), KDIM))
    X = P.sample_conditions(torch, model, arm, cc, K, 1, rng)[:, 0]; Xp = P.sample_conditions(torch, model, arm, cc, K, 1, rng)[:, 0]
    Z = P.sample_conditions(torch, model, arm, cc, K2, 1, rng)[:, 0]
    Hk = R.v_triplets(X, Xp, Z, SCALES); Hc = crps_features_v(X, Xp, Z)
    ing = {}
    for c in range(NCAT):
        m = cc == c
        kxx = np.mean([R.rbf(X[m][i:i + 1], Xp[m][i:i + 1], SCALES)[0, 0] for i in range(m.sum())])  # same-key pairs
        Wk = 1.0 - kxx; Wc = 0.5 * np.abs(X[m] - Xp[m]).mean()
        # iid mixture bank: new key per member (Qbar) -> ordinary score vs truths, minus W/n
        Kb = rng.standard_normal((ncal, KDIM)); B = P.sample_conditions(torch, model, arm, np.full(ncal, c), Kb, 1, rng)[:, 0]
        Rk = np.mean([R.kernel_empirical_score(B, y, SCALES) for y in truths[c]]) - Wk / ncal
        Rc = np.mean([R.empirical_score(B, y, "marginal") for y in truths[c]]) - Wc / ncal
        # raw underlying-law risk at fixed external keys (should be R(Qbar)+V): 8 keys x 512 members
        raw = []
        for _ in range(8):
            k0 = rng.standard_normal((1, KDIM)); Bk = P.sample_conditions(torch, model, arm, np.array([c]), k0, 512, rng)[0]
            raw.append(np.mean([R.kernel_empirical_score(Bk, y, SCALES) for y in truths[c]]) - Wk / 512)
        ing[c] = {"V_kernel": float(Hk[m].mean()), "V_kernel_se": float(Hk[m].std(ddof=1) / np.sqrt(m.sum())), "V_crps": float(Hc[m].mean()), "V_crps_se": float(Hc[m].std(ddof=1) / np.sqrt(m.sum())),
                  "W_kernel": float(Wk), "W_crps": float(Wc), "R_bar_kernel": float(Rk), "R_bar_crps": float(Rc), "R_raw_fixedkey_kernel": float(np.mean(raw)), "R_raw_fixedkey_kernel_se": float(np.std(raw, ddof=1) / np.sqrt(len(raw)))}
    (d / "ingredients.json").write_text(json.dumps(ing, indent=1))
    # ---- budget matrix: ordinary score of the declared 64-member ensemble, 32 runs per (c, L)
    for c in range(NCAT):
        for L in LS:
            mm = M // L
            for r in range(RUNS):
                keys = np.repeat(rng.standard_normal((L, KDIM)), mm, axis=0)  # key fixed within each trajectory, L groups
                z0 = rng.standard_normal((M, YDIM))
                ens = P.heun_sample(torch, model, np.full(M, c), keys if arm != "K0" else np.zeros_like(keys), z0, P.CFG["sampling"]["steps"], P.CFG["sampling"]["t_end"])
                sk = float(np.mean([R.kernel_empirical_score(ens, y, SCALES) for y in truths[c]])); sc = float(np.mean([R.empirical_score(ens, y, "marginal") for y in truths[c]]))
                rows.append({"block": block, "arm": arm, "cat": c, "L": L, "m": mm, "run": r, "kernel_score": sk, "crps": sc,
                             "pred_kernel": ing[c]["R_bar_kernel"] + ing[c]["V_kernel"] / L + ing[c]["W_kernel"] / M, "pred_crps": ing[c]["R_bar_crps"] + ing[c]["V_crps"] / L + ing[c]["W_crps"] / M})
        print(f"{arm} b{block} cat {c}: " + " ".join(f"L{L}={np.mean([x['kernel_score'] for x in rows if x['cat'] == c and x['L'] == L]):.4f}" for L in LS) + f" | V {ing[c]['V_kernel']:.4f} W {ing[c]['W_kernel']:.3f} Rbar {ing[c]['R_bar_kernel']:.4f}", flush=True)
    with (d / "budget_matrix.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summary(block, arm)


def summary(block: int, arm: str) -> None:
    d = OUT / f"block{block}_{arm}"; rows = [{k: float(v) if k != "arm" else v for k, v in r.items()} for r in csv.DictReader((d / "budget_matrix.csv").open())]
    ing = json.loads((d / "ingredients.json").read_text())
    out = []
    for L in LS:
        sel = [r for r in rows if r["L"] == L]
        # per category means, then average over categories (equal weights = uniform C)
        mk = np.mean([np.mean([r["kernel_score"] for r in sel if r["cat"] == c]) for c in range(NCAT)]); mc = np.mean([np.mean([r["crps"] for r in sel if r["cat"] == c]) for c in range(NCAT)])
        pk = np.mean([ing[str(c)]["R_bar_kernel"] + ing[str(c)]["V_kernel"] / L + ing[str(c)]["W_kernel"] / M for c in range(NCAT)]); pc = np.mean([ing[str(c)]["R_bar_crps"] + ing[str(c)]["V_crps"] / L + ing[str(c)]["W_crps"] / M for c in range(NCAT)])
        se = np.std([np.mean([r["kernel_score"] for r in sel if r["cat"] == c]) for c in range(NCAT)], ddof=1) / np.sqrt(NCAT)
        out.append({"block": block, "arm": arm, "L": L, "kernel_measured": float(mk), "kernel_predicted": float(pk), "kernel_se_over_cats": float(se), "crps_measured": float(mc), "crps_predicted": float(pc)})
    V = np.mean([ing[str(c)]["V_kernel"] for c in range(NCAT)]); Vc = np.mean([ing[str(c)]["V_crps"] for c in range(NCAT)])
    # paired T4.2 check: per (cat, run) difference S(L=1) - S(L=64), same run index (independent draws, paired only by index)
    d1 = np.array([[r["kernel_score"] for r in rows if r["cat"] == c and r["L"] == 1] for c in range(NCAT)]); d64 = np.array([[r["kernel_score"] for r in rows if r["cat"] == c and r["L"] == 64] for c in range(NCAT)])
    gain = (d1 - d64).mean(1); pred_gain = (1 - 1 / M) * np.array([ing[str(c)]["V_kernel"] for c in range(NCAT)])
    c1 = np.array([[r["crps"] for r in rows if r["cat"] == c and r["L"] == 1] for c in range(NCAT)]); c64 = np.array([[r["crps"] for r in rows if r["cat"] == c and r["L"] == 64] for c in range(NCAT)])
    res = {"table": out, "V_kernel_mean": float(V), "V_crps_mean": float(Vc), "gain_L1_minus_L64_kernel": float(gain.mean()), "gain_se_over_cats": float(gain.std(ddof=1) / np.sqrt(NCAT)), "predicted_gain_kernel": float(pred_gain.mean()),
           "gain_L1_minus_L64_crps": float((c1 - c64).mean()), "predicted_gain_crps": float((1 - 1 / M) * Vc), "per_cat_gain_kernel": gain.tolist(), "per_cat_pred_gain_kernel": pred_gain.tolist(),
           "raw_fixedkey_minus_Rbar_kernel_mean": float(np.mean([ing[str(c)]["R_raw_fixedkey_kernel"] - ing[str(c)]["R_bar_kernel"] for c in range(NCAT)]))}
    (d / "summary.json").write_text(json.dumps(res, indent=1))
    print(f"\n{arm} b{block}  V_k={V:.4f}  gain(L1-L64) measured {res['gain_L1_minus_L64_kernel']:.4f} ± {res['gain_se_over_cats']:.4f} vs predicted {res['predicted_gain_kernel']:.4f} | raw-fixed-key minus Rbar = {res['raw_fixedkey_minus_Rbar_kernel_mean']:.4f} (should be V)")
    for o in out:
        print(f"  L={o['L']:2d}: kernel measured {o['kernel_measured']:.4f} predicted {o['kernel_predicted']:.4f} | crps measured {o['crps_measured']:.4f} predicted {o['crps_predicted']:.4f}")


def informative_key_control() -> None:
    """T6 / P2.4 negative control by exact enumeration: K in {-1,+1}, Y|K ~ Bernoulli(.5 + beta K), model Q(Y=1|K) = .5 + gamma K.
    Binary CRPS = Brier. V = gamma^2, B = beta^2, A = beta*gamma; gain of full averaging = V - 2A; partial alpha: alpha(2-alpha)V - 2 alpha A."""
    rows = []
    for beta in (0.0, 0.1, 0.2):
        for gamma in (0.0, 0.1, 0.2):
            def risk(qfun):
                tot = 0.0
                for k in (-1, 1):
                    p = 0.5 + beta * k; q = qfun(k)
                    tot += 0.5 * (p * (1 - q) ** 2 + (1 - p) * q ** 2)
                return tot
            base = risk(lambda k: 0.5 + gamma * k)
            for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
                r = risk(lambda k: (1 - alpha) * (0.5 + gamma * k) + alpha * 0.5)
                rows.append({"beta": beta, "gamma": gamma, "alpha": alpha, "gain_enumerated": base - r, "gain_formula": alpha * (2 - alpha) * gamma ** 2 - 2 * alpha * beta * gamma})
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "informative_key_control.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    err = max(abs(r["gain_enumerated"] - r["gain_formula"]) for r in rows); print("informative-key control: max |enumerated - formula| =", err)
    for r in rows:
        if r["alpha"] == 1.0:
            print(f"  beta={r['beta']:.1f} gamma={r['gamma']:.1f}: full averaging gain {r['gain_enumerated']:+.4f} (V-2A = {r['gamma']**2 - 2*r['beta']*r['gamma']:+.4f})")


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("cmd", choices=["run", "summary", "control"]); p.add_argument("--block", type=int, default=0); p.add_argument("--arm", default="K1"); p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    {"run": lambda: run(a.block, a.arm, a.resume), "summary": lambda: summary(a.block, a.arm), "control": informative_key_control}[a.cmd]()
