#!/usr/bin/env python3
"""Review-2 §4: decomposition analysis on the ALREADY TRAINED confirm models (blocks 10-14, checkpoints 5k/10k/20k, arms
K0/K1/Kfresh/Krho05/Krho09). No new training.

Per (block, arm, step), on the frozen validation bank (16 categories x 32 external keys, 32 fresh truths each):
  raw      : MMD^2(Q_{c,K}, P*_c) per condition (U-statistic over the saved 64 members)  -> E_K MMD^2(Q_K, P*)
  mixture  : MMD^2(Qbar_c, P*_c) from a bank of 128 members with a NEW key per member (iid from Qbar)
  V        : label-free 1/2 E MMD^2(Q_K, Q_K') from 2048 triplets per category
  identity : E_K MMD^2(Q_K,P*) = MMD^2(Qbar,P*) + V   (same kernel and normalisation everywhere: mean RBF, scales .25/.5/1/2)
  centered : same three after mean matching at the SAME full external key (separate 256-member bank per condition for the mean),
             mixture centered with its own mean; the shift uses the known teacher mean (diagnostic, uses the oracle mean)
  structure: component probability (soft responsibilities of the known GMM) vs pi_c; per-component mean/cov error; full covariance
             error; variance along the two fixed teacher eigen-directions
  memory   : (K arms) 8 stored training examples per category: members at the STORED key vs the stored answer / vs fresh truths;
             at a NEW key; at a PERMUTED stored key of another example of the same category (distance to own vs other's answer);
             CFM loss with fresh Z0,t at the stored key on the stored answer vs on fresh answers
Outputs: autoresearch_results/invariance_audit_v1/confirm_decomposition/{block}_{arm}_{step}.json and summary CSV via `report`.
"""
from __future__ import annotations
import argparse, csv, json, os, sys
from pathlib import Path
import numpy as np
os.environ.setdefault("P1_STAGE", "confirm")
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import p1_pilot as P
R = P.R; SCALES = P.SCALES; NCAT, KDIM, YDIM = P.NCAT, P.KDIM, P.YDIM
OUT = P.OUT.parent / "confirm_decomposition"; ARMS = ("K0", "K1", "Kfresh", "Krho05", "Krho09"); STEPS = (5000, 10000, 20000); BLOCKS = (10, 11, 12, 13, 14)


def responsibilities(T, X):
    """Soft responsibilities of component 0 of the known 2-component teacher for points X (n,2)."""
    from scipy.stats import multivariate_normal as mvn
    p0 = T.weights[0] * mvn(T.means[0], T.covariances[0]).pdf(X); p1 = T.weights[1] * mvn(T.means[1], T.covariances[1]).pdf(X)
    return p0 / (p0 + p1)


def structure(T, X):
    r = responsibilities(T, X); w0 = float(r.mean()); out = {"p_comp0_model": w0, "p_comp0_true": float(T.weights[0])}
    for j, rj in ((0, r), (1, 1 - r)):
        if rj.sum() > 1e-6:
            mu = (rj[:, None] * X).sum(0) / rj.sum(); d = X - mu; cov = (rj[:, None, None] * d[:, :, None] * d[:, None, :]).sum(0) / rj.sum()
            out[f"comp{j}_mean_err2"] = float(((mu - T.means[j]) ** 2).sum()); out[f"comp{j}_cov_err_fro"] = float(np.linalg.norm(cov - T.covariances[j]))
    mu_t, cov_t = T.moments(); S = np.cov(X, rowvar=False, ddof=1); out["cov_err_fro"] = float(np.linalg.norm(S - cov_t))
    ev, evec = np.linalg.eigh(cov_t)
    for i in range(2):
        out[f"var_ratio_eig{i}"] = float((evec[:, i] @ S @ evec[:, i]) / ev[i])
    return out


def analyse(block: int, arm: str, step: int) -> None:
    torch = P._torch(); OUT.mkdir(parents=True, exist_ok=True); f = OUT / f"b{block}_{arm}_s{step}.json"
    if f.exists():
        return
    s = P.seeds(block); model = P.load_model(torch, block, arm, step)
    bank = np.load(P.block_dir(block) / "validation_bank.npz"); cats, ext_keys, truths = bank["cats"], bank["ext_keys"], bank["truths"]
    X = np.load(P.block_dir(block) / f"eval_{arm}" / f"members_step{step:05d}.npy").astype(np.float64)  # (512, 64, 2) at the external keys
    rng = np.random.default_rng(np.random.SeedSequence([s["validation"], step, 41]))
    n = len(cats)
    # label-free V per category (2048 triplets)
    cc = np.repeat(np.arange(NCAT), 2048); K = rng.standard_normal((len(cc), KDIM)); K2 = rng.standard_normal((len(cc), KDIM))
    A = P.sample_conditions(torch, model, arm, cc, K, 1, rng)[:, 0]; Ap = P.sample_conditions(torch, model, arm, cc, K, 1, rng)[:, 0]; Z = P.sample_conditions(torch, model, arm, cc, K2, 1, rng)[:, 0]
    H = R.v_triplets(A, Ap, Z, SCALES); V = {c: float(H[cc == c].mean()) for c in range(NCAT)}; Vse = {c: float(H[cc == c].std(ddof=1) / np.sqrt(2048)) for c in range(NCAT)}
    # mixture bank: new key per member, 128 members per condition (iid from Qbar_c)
    keys_mix = rng.standard_normal((n * 128, KDIM)); MIX = P.heun_sample(torch, model, np.repeat(cats, 128), keys_mix if arm != "K0" else np.zeros_like(keys_mix), rng.standard_normal((n * 128, YDIM)), 64, 0.999).reshape(n, 128, YDIM)
    # same-key pilot bank for mean matching (256 members per condition)
    PIL = P.sample_conditions(torch, model, arm, cats, ext_keys, 256, rng)
    rows = []
    for i in range(n):
        c = int(cats[i]); T = R.category_teacher(c); mu_t, _ = T.moments()
        raw = T.mmd2_from_samples(X[i], SCALES); mix = T.mmd2_from_samples(MIX[i], SCALES)
        Xc = X[i] - PIL[i].mean(0) + mu_t; MXc = MIX[i] - MIX[i].mean(0) + mu_t
        r = {"block": block, "arm": arm, "step": step, "cond": i, "cat": c, "mmd2_raw": raw, "mmd2_mixture": mix, "V": V[c], "V_se": Vse[c],
             "mmd2_raw_centered": T.mmd2_from_samples(Xc, SCALES), "mmd2_mixture_centered": T.mmd2_from_samples(MXc, SCALES),
             "mean_err2_raw": float(((X[i].mean(0) - mu_t) ** 2).sum() - np.trace(np.cov(X[i], rowvar=False, ddof=1)) / 64),
             "mean_err2_mixture": float(((MIX[i].mean(0) - mu_t) ** 2).sum() - np.trace(np.cov(MIX[i], rowvar=False, ddof=1)) / 128),
             "kernel_score_raw": float(np.mean([R.kernel_empirical_score(X[i], y, SCALES) for y in truths[i]])), "kernel_score_mixture64": float(np.mean([R.kernel_empirical_score(MIX[i][:64], y, SCALES) for y in truths[i]]))}
        r.update({f"raw_{k}": v for k, v in structure(T, X[i]).items()}); r.update({f"mix_{k}": v for k, v in structure(T, MIX[i]).items()})
        rows.append(r)
    res = {"per_condition": rows}
    # memorisation diagnostics for key arms (stored examples)
    if arm != "K0":
        data = np.load(P.block_dir(block) / "train_data.npz"); C, Y, Kt = data["C"], data["Y"], data["K"]
        sel = np.concatenate([np.flatnonzero(C == c)[:8] for c in range(NCAT)]); rk = np.random.default_rng(np.random.SeedSequence([s["validation"], step, 43]))
        Xs = P.sample_conditions(torch, model, arm, C[sel], Kt[sel], 64, rk); Xn = P.sample_conditions(torch, model, arm, C[sel], rk.standard_normal((len(sel), KDIM)), 64, rk)
        perm = np.concatenate([np.roll(np.arange(8), 1) + 8 * g for g in range(NCAT)]); Xperm = P.sample_conditions(torch, model, arm, C[sel], Kt[sel][perm], 64, rk)  # key of another example, same category
        mem = []
        for j, i in enumerate(sel):
            T = R.category_teacher(int(C[i])); fresh = T.sample(32, rk); other = Y[sel[perm[j]]]
            mem.append({"train_index": int(i), "cat": int(C[i]), "d_storedkey_to_stored_y": float(np.linalg.norm(Xs[j] - Y[i], axis=1).mean()), "d_newkey_to_stored_y": float(np.linalg.norm(Xn[j] - Y[i], axis=1).mean()),
                        "d_permkey_to_own_y": float(np.linalg.norm(Xperm[j] - Y[i], axis=1).mean()), "d_permkey_to_other_y": float(np.linalg.norm(Xperm[j] - other, axis=1).mean()),
                        "kernel_storedkey_vs_fresh": float(np.mean([R.kernel_empirical_score(Xs[j], y, SCALES) for y in fresh])), "kernel_newkey_vs_fresh": float(np.mean([R.kernel_empirical_score(Xn[j], y, SCALES) for y in fresh])),
                        "mmd2_storedkey": T.mmd2_from_samples(Xs[j], SCALES), "mmd2_newkey": T.mmd2_from_samples(Xn[j], SCALES), "d_stored_y_to_teacher_mean": float(np.linalg.norm(Y[i] - T.moments()[0]))})
        # CFM loss with fresh z0,t at the stored key: stored answer vs fresh answers
        with torch.no_grad():
            c1 = torch.as_tensor(np.eye(NCAT, dtype=np.float32)[C[sel]]); k1 = torch.as_tensor(Kt[sel].astype(np.float32)); losses = {"stored": [], "fresh": []}
            for rep in range(64):
                t = torch.as_tensor(rk.random(len(sel)).astype(np.float32)); z0 = torch.as_tensor(rk.standard_normal((len(sel), YDIM)).astype(np.float32))
                for name, y in (("stored", Y[sel]), ("fresh", np.stack([R.category_teacher(int(C[i])).sample(1, rk)[0] for i in sel]))):
                    yt = torch.as_tensor(y.astype(np.float32)); zt = (1 - t)[:, None] * z0 + t[:, None] * yt
                    v = model(P.net_input(torch, zt, c1, k1, t)); losses[name].append(float(((v - (yt - z0)) ** 2).mean()))
        res["memory"] = mem; res["cfm_loss_storedkey"] = {k: float(np.mean(v)) for k, v in losses.items()}
    f.write_text(json.dumps(res, indent=0, default=float))
    agg = lambda k: np.mean([r[k] for r in rows])
    print(f"b{block} {arm:7s} s{step:5d}: raw {agg('mmd2_raw'):.4f} = mix {agg('mmd2_mixture'):.4f} + V {agg('V'):.4f} (sum {agg('mmd2_mixture') + agg('V'):.4f}) | centered raw {agg('mmd2_raw_centered'):.4f} mix {agg('mmd2_raw_centered') - 0:.4f}/{agg('mmd2_mixture_centered'):.4f} | p0 err {np.mean([abs(r['raw_p_comp0_model'] - r['raw_p_comp0_true']) for r in rows]):.3f}", flush=True)


def report():
    rows = []
    for f in sorted(OUT.glob("b*_s*.json")):
        d = json.loads(f.read_text()); pc = d["per_condition"]
        b, arm, step = pc[0]["block"], pc[0]["arm"], pc[0]["step"]
        keys = ["mmd2_raw", "mmd2_mixture", "V", "mmd2_raw_centered", "mmd2_mixture_centered", "mean_err2_raw", "mean_err2_mixture", "kernel_score_raw", "kernel_score_mixture64", "raw_cov_err_fro", "mix_cov_err_fro", "raw_var_ratio_eig0", "raw_var_ratio_eig1", "mix_var_ratio_eig0", "mix_var_ratio_eig1"]
        r = {"block": b, "arm": arm, "step": step, **{k: float(np.mean([x[k] for x in pc])) for k in keys}}
        r["p0_abs_err_raw"] = float(np.mean([abs(x["raw_p_comp0_model"] - x["raw_p_comp0_true"]) for x in pc])); r["p0_abs_err_mix"] = float(np.mean([abs(x["mix_p_comp0_model"] - x["mix_p_comp0_true"]) for x in pc]))
        r["identity_residual"] = r["mmd2_raw"] - r["mmd2_mixture"] - r["V"]
        # joint SE of the residual over the 512 conditions (V enters per category; treat per-condition residual with category V)
        res_c = np.array([x["mmd2_raw"] - x["mmd2_mixture"] - x["V"] for x in pc]); r["identity_residual_se_cond"] = float(res_c.std(ddof=1) / np.sqrt(len(res_c)))
        if "memory" in d:
            m = d["memory"]; r.update({k: float(np.mean([x[k] for x in m])) for k in ("d_storedkey_to_stored_y", "d_newkey_to_stored_y", "d_permkey_to_own_y", "d_permkey_to_other_y", "kernel_storedkey_vs_fresh", "kernel_newkey_vs_fresh", "mmd2_storedkey", "mmd2_newkey", "d_stored_y_to_teacher_mean")}); r.update({f"cfm_loss_{k}": v for k, v in d["cfm_loss_storedkey"].items()})
        rows.append(r)
    with (OUT / "confirm_decomposition_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}, key=lambda k: (k not in ("block", "arm", "step"), k))); w.writeheader(); w.writerows(rows)
    print("written", OUT / "confirm_decomposition_summary.csv", len(rows), "rows")


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("cmd", choices=["all", "report"]); a = p.parse_args()
    if a.cmd == "all":
        for b in BLOCKS:
            for arm in ARMS:
                for step in STEPS:
                    analyse(b, arm, step)
    report()
