#!/usr/bin/env python3
"""P1 pilot of the invariance audit (Astra execution pack v1, EXPERIMENT_SPECIFICATIONS.md §P1).

Clean categorical key test: C uniform on 16 categories, Y in R^2 from the exact bimodal teacher
`reference.core.category_teacher`; 128 fixed training replies per category (2048 pairs); each pair
gets an 8-D standard-normal key K_i. Arms K0 (zero key), K1 (fixed key per example), Kfresh (new
standard-normal key at every presentation). MLP-CFM 3x256 SiLU, Adam 2e-4, batch 256, 10k steps,
checkpoints 0/500/2000/5000/10000, Heun-64 to T=1-1e-3, key fixed within a trajectory.

Commands: make-data | train | sample-evaluate | audit-v | solver-audit | report  (all take --block s)
Outputs under autoresearch_results/invariance_audit_v1/p1_pilot/block{s}/ ; nothing is overwritten
without --resume.  The teacher and all scores come from the reference module (not re-implemented).
"""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
REF = HERE / "astra_pack_v1" / "reference"
sys.path.insert(0, str(REF))
import core as R  # noqa: E402  (reference implementation, kept verbatim)

ROOT = HERE.parents[2]
import os
CFG = json.loads((HERE / "astra_pack_v1" / "P1_PILOT_CONFIG.json").read_text())
STAGE = os.environ.get("P1_STAGE", "pilot")  # "pilot" (10k, blocks 0/1) or "confirm" (20k, blocks 10-14; frozen before generation)
OUT = ROOT / "autoresearch_results" / "invariance_audit_v1" / ("p1_pilot" if STAGE == "pilot" else "p1_confirm")
if STAGE == "confirm":
    CFG["training"]["steps"] = 20000; CFG["training"]["checkpoints"] = [0, 500, 2000, 5000, 10000, 20000]
NCAT, NREP, KDIM, YDIM = 16, 128, 8, 2
SCALES = tuple(CFG["validation"]["kernel_bandwidths"])
FREQ = 10.0 ** np.linspace(-1, 1, 16)
ARMS = {"K0": ("zero", None), "K1": ("fixed_per_example", 1.0), "Kfresh": ("fresh_per_presentation", 0.0),
        "Krho05": ("rho_mixed", 0.5), "Krho09": ("rho_mixed", 0.9), "Kcached": ("cached_rho", 0.5), "Kclass": ("class_key", None)}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seeds(block: int) -> dict:
    return {k: v + block for k, v in CFG["seed_offsets"].items()}


def block_dir(block: int) -> Path:
    d = OUT / f"block{block}"; d.mkdir(parents=True, exist_ok=True); return d


# ----------------------------------------------------------------------------- data
def make_data(block: int, resume: bool) -> None:
    d = block_dir(block); f = d / "train_data.npz"
    if f.exists() and not resume:
        raise FileExistsError(f)
    s = seeds(block)
    rng_data = np.random.default_rng(s["data"]); rng_keys = np.random.default_rng(s["keys"])
    C = np.repeat(np.arange(NCAT), NREP)
    Y = np.concatenate([R.category_teacher(int(c)).sample(NREP, rng_data) for c in range(NCAT)])
    K = rng_keys.standard_normal((NCAT * NREP, KDIM))
    np.savez(f, C=C, Y=Y, K=K)
    man = {"block": block, "seeds": s, "n_train": int(len(C)), "distinct_conditions": NCAT, "replies_per_condition": NREP,
           "key_dim": KDIM, "sha256": sha(f), "teacher": "reference.core.category_teacher", "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (d / "train_data_manifest.json").write_text(json.dumps(man, indent=1))
    print("data", f, man["sha256"][:12])


# ----------------------------------------------------------------------------- model
def _torch():
    import torch
    torch.set_num_threads(8)
    return torch


def time_embed(torch, t):
    f = torch.as_tensor(FREQ, dtype=t.dtype)
    ang = 2 * np.pi * t[:, None] * f[None, :]
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)


def build_model(torch, init_seed: int):
    import torch.nn as nn
    g = torch.Generator().manual_seed(init_seed)
    layers = []; dims = [YDIM + NCAT + KDIM + 32] + CFG["model"]["hidden_widths"] + [YDIM]
    for i in range(len(dims) - 1):
        lin = nn.Linear(dims[i], dims[i + 1])
        with torch.no_grad():
            bound = float(np.sqrt(6.0 / (dims[i] + dims[i + 1])))
            lin.weight.uniform_(-bound, bound, generator=g); lin.bias.zero_()
        layers.append(lin)
        if i < len(dims) - 2:
            layers.append(nn.SiLU())
    return nn.Sequential(*layers)


def net_input(torch, z, c_onehot, key, t):
    return torch.cat([z, c_onehot, key, time_embed(torch, t)], dim=1)


def state_hash(torch, model) -> str:
    h = hashlib.sha256()
    for p in model.state_dict().values():
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


# ----------------------------------------------------------------------------- train
def train(block: int, arm: str, resume: bool) -> None:
    torch = _torch()
    d = block_dir(block); rd = d / f"train_{arm}"
    if rd.exists() and not resume:
        raise FileExistsError(rd)
    rd.mkdir(exist_ok=True)
    s = seeds(block); tr = CFG["training"]
    data = np.load(d / "train_data.npz"); C, Y, K = data["C"], data["Y"], data["K"]
    n = len(C); steps = tr["steps"]; bs = tr["batch_size"]
    # shared streams across arms of one block: batch indices, base noise z0, times t
    idx = np.random.default_rng(s["batches"]).integers(0, n, size=(steps, bs))
    rng_flow = np.random.default_rng(s["flow"])
    Z0 = rng_flow.standard_normal((steps, bs, YDIM)).astype(np.float32)
    T = rng_flow.random((steps, bs)).astype(np.float32)
    rng_aug = np.random.default_rng(s["augmentation"])  # consumed only by Kfresh
    mode, rho = ARMS[arm]
    model = build_model(torch, s["init"]); init_hash = state_hash(torch, model)
    opt = torch.optim.Adam(model.parameters(), lr=tr["lr"], betas=tuple(tr["betas"]), weight_decay=tr["weight_decay"])
    onehot = np.eye(NCAT, dtype=np.float32)[C]; Yt = torch.as_tensor(Y, dtype=torch.float32)
    Kt = torch.as_tensor(K, dtype=torch.float32); Ct = torch.as_tensor(onehot)
    _rc = np.random.default_rng(np.random.SeedSequence([s["augmentation"], 1]))
    Kcached_t = torch.as_tensor((0.5 * K + np.sqrt(0.75) * _rc.standard_normal(K.shape)).astype(np.float32))
    Kclass = np.random.default_rng(np.random.SeedSequence([s["keys"], 2])).standard_normal((NCAT, KDIM)).astype(np.float32); Kclass_t = torch.as_tensor(Kclass)
    np.save(rd / "class_keys.npy", Kclass) if mode == "class_key" else None
    log = []; t0 = time.time()

    def save(step):
        torch.save(model.state_dict(), rd / f"ckpt_{step:05d}.pt")

    for step in range(steps + 1):
        if step in tr["checkpoints"]:
            save(step)
        if step == steps:
            break
        b = torch.as_tensor(idx[step]); y = Yt[b]; c = Ct[b]
        if mode == "zero":
            key = torch.zeros(bs, KDIM)
        elif mode == "fixed_per_example":
            key = Kt[b]
        elif mode == "fresh_per_presentation":  # rho=0 -> pure standard normal, independent of the example
            key = torch.as_tensor(rng_aug.standard_normal((bs, KDIM)).astype(np.float32))
        elif mode == "rho_mixed":  # T1.4: rho K_i + sqrt(1-rho^2) E, fresh E at every presentation, stationary N(0,I) marginal
            key = rho * Kt[b] + float(np.sqrt(1 - rho * rho)) * torch.as_tensor(rng_aug.standard_normal((bs, KDIM)).astype(np.float32))
        elif mode == "cached_rho":  # control: rho K_i + sqrt(1-rho^2) E_i computed ONCE per example, then fixed (still an identifier)
            key = Kcached_t[b]
        else:  # class_key: one fixed 8-D key per category (redundant function of C, not an identifier)
            key = Kclass_t[C[idx[step]]]
        z0 = torch.as_tensor(Z0[step]); t = torch.as_tensor(T[step])
        zt = (1 - t)[:, None] * z0 + t[:, None] * y
        v = model(net_input(torch, zt, c, key, t))
        loss = ((v - (y - z0)) ** 2).mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tr["gradient_clip_norm"]); opt.step()
        if step % 500 == 0:
            log.append({"step": step, "loss": float(loss)}); print(f"{arm} b{block} step {step} loss {float(loss):.4f}", flush=True)
    man = {"block": block, "arm": arm, "key_mode": mode, "rho": rho, "seeds": s, "init_state_sha256": init_hash,
           "final_state_sha256": state_hash(torch, model), "steps": steps, "batch_size": bs, "runtime_s": time.time() - t0,
           "train_data_sha256": sha(d / "train_data.npz"), "torch": torch.__version__, "precision": "float32", "device": "cpu",
           "loss_log": log, "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (rd / "run_manifest.json").write_text(json.dumps(man, indent=1))


# ----------------------------------------------------------------------------- sampling
def heun_sample(torch, model, c_idx: np.ndarray, keys: np.ndarray, z0: np.ndarray, steps: int, t_end: float) -> np.ndarray:
    """Heun (explicit trapezoid), key fixed within each trajectory. Returns (n, 2) at t_end."""
    c = torch.as_tensor(np.eye(NCAT, dtype=np.float32)[c_idx]); k = torch.as_tensor(keys.astype(np.float32))
    z = torch.as_tensor(z0.astype(np.float32)); ts = np.linspace(0.0, t_end, steps + 1, dtype=np.float32)
    with torch.no_grad():
        for i in range(steps):
            t0 = torch.full((len(z),), float(ts[i])); t1 = torch.full((len(z),), float(ts[i + 1])); h = float(ts[i + 1] - ts[i])
            v0 = model(net_input(torch, z, c, k, t0)); zp = z + h * v0
            v1 = model(net_input(torch, zp, c, k, t1)); z = z + 0.5 * h * (v0 + v1)
    return z.numpy().astype(np.float64)


CURRENT_BLOCK = [0]


def load_model(torch, block: int, arm: str, step: int):
    CURRENT_BLOCK[0] = block
    m = build_model(torch, seeds(block)["init"]); m.load_state_dict(torch.load(block_dir(block) / f"train_{arm}" / f"ckpt_{step:05d}.pt")); m.eval(); return m


def sample_conditions(torch, model, arm: str, cats: np.ndarray, keys: np.ndarray, M: int, rng: np.random.Generator, steps=None, t_end=None):
    """cats (n,), keys (n,8): M members per condition with independent base noise; key fixed per condition (zero for K0)."""
    sm = CFG["sampling"]; steps = steps or sm["steps"]; t_end = t_end or sm["t_end"]
    n = len(cats); cc = np.repeat(cats, M)
    if arm == "K0":
        keys = np.zeros_like(keys)
    elif arm == "Kclass":  # the category's own fixed key (regenerated from the block's key seed, identical to training)
        keys = np.random.default_rng(np.random.SeedSequence([seeds(CURRENT_BLOCK[0])["keys"], 2])).standard_normal((NCAT, KDIM)).astype(np.float32)[cats]
    kk = np.repeat(keys, M, axis=0)
    z0 = rng.standard_normal((n * M, YDIM))
    return heun_sample(torch, model, cc, kk, z0, steps, t_end).reshape(n, M, YDIM)


# ----------------------------------------------------------------------------- evaluation
def teacher_stats(c: int):
    T = R.category_teacher(c); mu, cov = T.moments()
    angle = 2 * np.pi * c / 16; a = 0.7 * np.array([np.cos(2 * angle), np.sin(2 * angle)]); pi = 0.25 + 0.5 * (c % 4) / 3
    sd = float(np.sqrt(a @ T.covariances[0] @ a)); m = float(a @ a)
    from scipy.special import ndtr
    p_event = pi * ndtr(m / sd) + (1 - pi) * ndtr(-m / sd)  # P(a^T (Y - mu_c) > 0), mu_c = center
    return T, mu, cov, a, p_event


def evaluate_samples(c: int, X: np.ndarray, truths: np.ndarray, pilot: np.ndarray | None) -> dict:
    """X (M,2) members of one condition; truths (32,2) fresh teacher draws; pilot (>=1024,2) independent bank for mean matching."""
    T, mu, cov, a, p_true = teacher_stats(c); M = len(X)
    out = {"mmd2_joint": T.mmd2_from_samples(X, SCALES)}
    out["crps_marginal"] = float(np.mean([R.empirical_score(X, y, "marginal") for y in truths]))
    out["energy_score"] = float(np.mean([R.empirical_score(X, y, "energy") for y in truths]))
    out["kernel_score"] = float(np.mean([R.kernel_empirical_score(X, y, SCALES) for y in truths]))
    xbar = X.mean(0); S = np.cov(X, rowvar=False, ddof=1)
    out["mean_mse_raw"] = float(((xbar - mu) ** 2).sum()); out["mean_mse_corrected"] = float(((xbar - mu) ** 2).sum() - np.trace(S) / M)
    out["cov_err_fro"] = float(np.linalg.norm(S - cov)); ev_true = np.sort(np.linalg.eigvalsh(cov)); ev = np.sort(np.linalg.eigvalsh(S))
    out["eig_ratio_min"] = float(ev[0] / ev_true[0]); out["eig_ratio_max"] = float(ev[1] / ev_true[1])
    out["sd0_ratio"] = float(np.sqrt(S[0, 0] / cov[0, 0])); out["sd1_ratio"] = float(np.sqrt(S[1, 1] / cov[1, 1]))
    center = 2 * np.array([np.cos(2 * np.pi * c / 16), np.sin(2 * np.pi * c / 16)])
    out["p_event_model"] = float(np.mean((X - center) @ a > 0)); out["p_event_true"] = float(p_true)
    # coverage of central 80% marginal intervals (quantile levels .1/.9 of members) for the 32 truths
    lo, hi = np.quantile(X, 0.1, axis=0), np.quantile(X, 0.9, axis=0)
    out["cov80_marginal"] = float(np.mean((truths >= lo) & (truths <= hi)))
    if pilot is not None:
        Xm = X - pilot.mean(0) + mu
        out["mmd2_joint_mean_matched"] = T.mmd2_from_samples(Xm, SCALES)
        out["kernel_score_mean_matched"] = float(np.mean([R.kernel_empirical_score(Xm, y, SCALES) for y in truths]))
    return out


def sample_evaluate(block: int, arm: str, resume: bool) -> None:
    torch = _torch(); d = block_dir(block); ed = d / f"eval_{arm}"
    if ed.exists() and not resume:
        raise FileExistsError(ed)
    ed.mkdir(exist_ok=True); s = seeds(block); va = CFG["validation"]; M = CFG["sampling"]["members"]
    rng_val = np.random.default_rng(s["validation"])
    nk = va["external_keys_per_category"]; cats = np.repeat(np.arange(NCAT), nk)
    ext_keys = rng_val.standard_normal((NCAT * nk, KDIM))
    truths = np.stack([R.category_teacher(int(c)).sample(va["truth_replicates_per_condition"], rng_val) for c in cats])  # shared bank
    np.savez(d / "validation_bank.npz", cats=cats, ext_keys=ext_keys, truths=truths) if not (d / "validation_bank.npz").exists() else None
    rng_pilot = np.random.default_rng(np.random.SeedSequence([s["validation"], 77]))
    rows = []
    for step in CFG["training"]["checkpoints"]:
        model = load_model(torch, block, arm, step)
        rng_gen = np.random.default_rng(np.random.SeedSequence([s["validation"], step, 1]))
        X = sample_conditions(torch, model, arm, cats, ext_keys, M, rng_gen)
        # independent pilot bank (1024 members, one external key per category, separate rng) for diagnostic mean matching
        pk = rng_pilot.standard_normal((NCAT, KDIM))
        P = sample_conditions(torch, model, arm, np.arange(NCAT), pk, 1024, rng_pilot)
        for i, c in enumerate(cats):
            r = evaluate_samples(int(c), X[i], truths[i], P[c]); r.update({"block": block, "arm": arm, "step": step, "cat": int(c), "cond": i, "M": M})
            rows.append(r)
        np.save(ed / f"members_step{step:05d}.npy", X.astype(np.float32))
        agg = {k: float(np.mean([r[k] for r in rows if r["step"] == step])) for k in rows[-1] if isinstance(rows[-1][k], float)}
        print(f"{arm} b{block} step {step}: mmd2 {agg['mmd2_joint']:.4f} crps {agg['crps_marginal']:.4f} meanMSEc {agg['mean_mse_corrected']:.4f} sd0r {agg['sd0_ratio']:.2f} sd1r {agg['sd1_ratio']:.2f}", flush=True)
    import csv
    with (ed / "per_condition_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    # train-condition diagnostic for K1: stored key vs new key at final checkpoint (first 8 examples per category)
    if arm == "K1":
        model = load_model(torch, block, arm, CFG["training"]["steps"]); data = np.load(d / "train_data.npz")
        sel = np.concatenate([np.flatnonzero(data["C"] == c)[:8] for c in range(NCAT)])
        rng_tr = np.random.default_rng(np.random.SeedSequence([s["validation"], 5]))
        Xs = sample_conditions(torch, model, arm, data["C"][sel], data["K"][sel], M, rng_tr)
        Xn = sample_conditions(torch, model, arm, data["C"][sel], rng_tr.standard_normal((len(sel), KDIM)), M, rng_tr)
        diag = []
        for j, i in enumerate(sel):
            T = R.category_teacher(int(data["C"][i]))
            diag.append({"train_index": int(i), "cat": int(data["C"][i]),
                         "stored_key_dist_to_stored_y": float(np.linalg.norm(Xs[j] - data["Y"][i], axis=1).mean()),
                         "new_key_dist_to_stored_y": float(np.linalg.norm(Xn[j] - data["Y"][i], axis=1).mean()),
                         "stored_key_mmd2_teacher": T.mmd2_from_samples(Xs[j], SCALES), "new_key_mmd2_teacher": T.mmd2_from_samples(Xn[j], SCALES)})
        (ed / "train_condition_diagnostic.json").write_text(json.dumps(diag, indent=1))


def baselines(block: int) -> None:
    """Simple conditional baselines on the same validation bank: empirical category measure (128 replies), fitted Gaussian, oracle finite-M."""
    d = block_dir(block); s = seeds(block); data = np.load(d / "train_data.npz"); bank = np.load(d / "validation_bank.npz")
    M = CFG["sampling"]["members"]; rng = np.random.default_rng(np.random.SeedSequence([s["validation"], 9]))
    rows = []
    for i, c in enumerate(bank["cats"]):
        c = int(c); T = R.category_teacher(c); Yc = data["Y"][data["C"] == c]; truths = bank["truths"][i]
        emp = Yc[rng.choice(len(Yc), M)]  # iid draws from the empirical category measure (rho=0 target)
        mu, S = Yc.mean(0), np.cov(Yc, rowvar=False, ddof=1); gauss = rng.multivariate_normal(mu, S, M)
        orc = T.sample(M, rng)
        for name, X in (("empirical_128", emp), ("fitted_gaussian", gauss), ("oracle_finiteM", orc)):
            r = evaluate_samples(c, X, truths, None); r.update({"block": block, "arm": name, "step": -1, "cat": c, "cond": i, "M": M}); rows.append(r)
    import csv
    with (d / "baselines_per_condition.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for name in ("empirical_128", "fitted_gaussian", "oracle_finiteM"):
        print(name, {k: round(float(np.mean([r[k] for r in rows if r["arm"] == name])), 4) for k in ("mmd2_joint", "crps_marginal", "mean_mse_corrected")})


# ----------------------------------------------------------------------------- label-free V
def audit_v(block: int, arm: str) -> None:
    torch = _torch(); d = block_dir(block); s = seeds(block); nt = CFG["validation"]["label_free_triplets_per_category"]
    rows = []
    for step in CFG["training"]["checkpoints"]:
        model = load_model(torch, block, arm, step); rng = np.random.default_rng(np.random.SeedSequence([s["validation"], step, 3]))
        cats = np.repeat(np.arange(NCAT), nt); K = rng.standard_normal((len(cats), KDIM)); K2 = rng.standard_normal((len(cats), KDIM))
        X = sample_conditions(torch, model, arm, cats, K, 1, rng)[:, 0]; Xp = sample_conditions(torch, model, arm, cats, K, 1, rng)[:, 0]
        Z = sample_conditions(torch, model, arm, cats, K2, 1, rng)[:, 0]
        H = R.v_triplets(X, Xp, Z, SCALES)
        # energy-feature V (for CRPS-type scores): 1/2 energy distance between Q_K and Q_K' from independent pairs
        e = 0.5 * (np.linalg.norm(X - Z, axis=1) / np.sqrt(YDIM) - np.linalg.norm(X - Xp, axis=1) / np.sqrt(YDIM))
        for c in range(NCAT):
            m = cats == c
            rows.append({"block": block, "arm": arm, "step": step, "cat": c, "n_triplets": int(m.sum()), "V_kernel": float(H[m].mean()), "V_kernel_se": float(H[m].std(ddof=1) / np.sqrt(m.sum())),
                         "V_energy": float(e[m].mean())})
        V = float(H.mean()); print(f"{arm} b{block} step {step}: V_kernel {V:.4f} (Hoeffding r={R.hoeffding_radius(len(H)):.3f} for {len(H)} triplets, conditional on this model) V_energy {float(e.mean()):.4f}", flush=True)
        rows.append({"block": block, "arm": arm, "step": step, "cat": -1, "n_triplets": int(len(H)), "V_kernel": V, "V_kernel_se": float(H.std(ddof=1) / np.sqrt(len(H))), "V_energy": float(e.mean())})
    import csv
    with (d / f"key_sensitivity_{arm}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


# ----------------------------------------------------------------------------- solver audit
def solver_audit(block: int, arm: str) -> None:
    torch = _torch(); from scipy.integrate import solve_ivp
    d = block_dir(block); s = seeds(block); sa = CFG["solver_audit"]; step = CFG["training"]["steps"]
    model = load_model(torch, block, arm, step); rng = np.random.default_rng(np.random.SeedSequence([s["validation"], 11]))
    cats = np.repeat(np.array(sa["categories"]), sa["keys_per_category"]); keys = rng.standard_normal((len(cats), KDIM)); M = 64
    z0 = rng.standard_normal((len(cats) * M, YDIM)); cc = np.repeat(cats, M); kk = np.repeat(keys if arm != "K0" else np.zeros_like(keys), M, axis=0)
    ref = {}
    rows = []
    for t_end in sa["endpoints"]:
        for n in sa["heun_steps"]:
            ref[(t_end, n)] = heun_sample(torch, model, cc, kk, z0, n, t_end)
        # DOP853 per trajectory (32 trajectories: first 4 of each condition)
        pick = np.concatenate([np.arange(i * M, i * M + 4) for i in range(len(cats))])
        c1 = torch.as_tensor(np.eye(NCAT, dtype=np.float32)[cc[pick]]); k1 = torch.as_tensor(kk[pick].astype(np.float32))
        def f(t, y):
            z = torch.as_tensor(y.reshape(-1, YDIM).astype(np.float32)); tt = torch.full((len(z),), float(t))
            with torch.no_grad():
                return model(net_input(torch, z, c1, k1, tt)).numpy().astype(np.float64).ravel()
        sol = solve_ivp(f, (0.0, t_end), z0[pick].ravel(), method="DOP853", rtol=sa["rtol"], atol=sa["atol"])
        dop = sol.y[:, -1].reshape(-1, YDIM)
        for n in sa["heun_steps"]:
            disp = np.linalg.norm(ref[(t_end, n)][pick] - dop, axis=1)
            rows.append({"block": block, "arm": arm, "t_end": t_end, "solver": f"Heun{n}", "nfe": 2 * n, "vs": "DOP853", "mean_disp": float(disp.mean()), "max_disp": float(disp.max()), "nfe_dop853": int(sol.nfev)})
        # MMD difference between Heun64 and Heun128 across all members per condition
        for n in sa["heun_steps"]:
            mm = [R.category_teacher(int(cats[i])).mmd2_from_samples(ref[(t_end, n)][i * M:(i + 1) * M], SCALES) for i in range(len(cats))]
            rows.append({"block": block, "arm": arm, "t_end": t_end, "solver": f"Heun{n}", "nfe": 2 * n, "vs": "teacher", "mean_disp": float(np.mean(mm)), "max_disp": float(np.max(mm)), "nfe_dop853": -1})
    import csv
    with (d / f"solver_audit_{arm}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for r in rows:
        print(r)


# ----------------------------------------------------------------------------- report
def report(blocks: list[int]) -> None:
    import csv
    rows = []
    for b in blocks:
        for arm in ARMS:
            p = block_dir(b) / f"eval_{arm}" / "per_condition_metrics.csv"
            if p.exists():
                rows += [{k: (float(v) if k not in ("arm",) else v) for k, v in r.items()} for r in csv.DictReader(p.open())]
    keys = ["mmd2_joint", "mmd2_joint_mean_matched", "crps_marginal", "kernel_score", "energy_score", "mean_mse_corrected", "cov_err_fro", "sd0_ratio", "sd1_ratio", "p_event_model", "p_event_true", "cov80_marginal"]
    table = []
    for b in blocks:
        for arm in ARMS:
            for step in CFG["training"]["checkpoints"]:
                sel = [r for r in rows if r["block"] == b and r["arm"] == arm and r["step"] == step]
                if sel:
                    table.append({"block": b, "arm": arm, "step": step, "n_cond": len(sel), **{k: float(np.mean([r[k] for r in sel])) for k in keys if k in sel[0]},
                                  "mmd2_joint_se_cond": float(np.std([r["mmd2_joint"] for r in sel], ddof=1) / np.sqrt(len(sel)))})
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "p1_checkpoint_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0])); w.writeheader(); w.writerows(table)
    # primary contrast: K1 - K0 joint regret per block and step (paired over the same 512 conditions and truth bank)
    contrasts = []
    for b in blocks:
        for step in CFG["training"]["checkpoints"]:
            k0 = {r["cond"]: r for r in rows if r["block"] == b and r["arm"] == "K0" and r["step"] == step}
            for other in ("K1", "Kfresh"):
                k1 = {r["cond"]: r for r in rows if r["block"] == b and r["arm"] == other and r["step"] == step}
                common = sorted(set(k0) & set(k1))
                if common:
                    for metric in ("mmd2_joint", "crps_marginal", "mean_mse_corrected", "mmd2_joint_mean_matched"):
                        if metric in k1[common[0]]:
                            dlt = np.array([k1[c][metric] - k0[c][metric] for c in common])
                            contrasts.append({"block": b, "step": step, "contrast": f"{other}-K0", "metric": metric, "mean_delta": float(dlt.mean()), "se_over_conditions": float(dlt.std(ddof=1) / np.sqrt(len(dlt))), "n_cond": len(dlt)})
    with (OUT / "p1_primary_contrasts.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(contrasts[0])); w.writeheader(); w.writerows(contrasts)
    for t in table:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in t.items() if k in ("block", "arm", "step", "mmd2_joint", "crps_marginal", "mean_mse_corrected", "sd0_ratio", "sd1_ratio")})
    for c in contrasts:
        if c["metric"] == "mmd2_joint":
            print(c)


# ----------------------------------------------------------------------------- final-checkpoint diagnostics (registered extension)
def diag_final(block: int, arm: str, triplets_per_cat: int = 4096) -> None:
    """(a) mean matching with a pilot bank at the SAME external key as each evaluated condition (the per-checkpoint
    diagnostic used one pilot key per category, which for key-sensitive models mixes key-dependent mean shifts into the
    'mean-matched' score); (b) label-free V with enough triplets for a meaningful Hoeffding certificate."""
    torch = _torch(); d = block_dir(block); s = seeds(block); step = CFG["training"]["steps"]; M = CFG["sampling"]["members"]
    model = load_model(torch, block, arm, step); bank = np.load(d / "validation_bank.npz"); cats, keys, truths = bank["cats"], bank["ext_keys"], bank["truths"]
    X = np.load(d / f"eval_{arm}" / f"members_step{step:05d}.npy").astype(np.float64)
    rng = np.random.default_rng(np.random.SeedSequence([s["validation"], step, 21]))
    P = sample_conditions(torch, model, arm, cats, keys, 1024, rng)  # same key per condition, independent base noise
    rows = []
    for i, c in enumerate(cats):
        T, mu, cov, a, _ = teacher_stats(int(c)); Xm = X[i] - P[i].mean(0) + mu
        rows.append({"cond": i, "cat": int(c), "mmd2_joint": T.mmd2_from_samples(X[i], SCALES), "mmd2_joint_mean_matched_samekey": T.mmd2_from_samples(Xm, SCALES),
                     "kernel_score_mean_matched_samekey": float(np.mean([R.kernel_empirical_score(Xm, y, SCALES) for y in truths[i]])),
                     "pilot_mean_shift_norm": float(np.linalg.norm(P[i].mean(0) - mu))})
    import csv
    with (d / f"eval_{arm}" / "final_mean_matched_samekey.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"{arm} b{block} step {step}: mmd2 {np.mean([r['mmd2_joint'] for r in rows]):.4f} mean-matched(same key) {np.mean([r['mmd2_joint_mean_matched_samekey'] for r in rows]):.4f} |shift| {np.mean([r['pilot_mean_shift_norm'] for r in rows]):.3f}", flush=True)
    rngv = np.random.default_rng(np.random.SeedSequence([s["validation"], step, 23]))
    cc = np.repeat(np.arange(NCAT), triplets_per_cat); K = rngv.standard_normal((len(cc), KDIM)); K2 = rngv.standard_normal((len(cc), KDIM))
    Xa = sample_conditions(torch, model, arm, cc, K, 1, rngv)[:, 0]; Xb = sample_conditions(torch, model, arm, cc, K, 1, rngv)[:, 0]; Z = sample_conditions(torch, model, arm, cc, K2, 1, rngv)[:, 0]
    H = R.v_triplets(Xa, Xb, Z, SCALES); n = len(H); r = R.hoeffding_radius(n)
    out = {"block": block, "arm": arm, "step": step, "n_triplets": n, "V_kernel": float(H.mean()), "se": float(H.std(ddof=1) / np.sqrt(n)), "hoeffding_radius_95": r,
           "V_lower_bound_95": float(max(0.0, H.mean() - r)), "V_energy": float((0.5 * (np.linalg.norm(Xa - Z, axis=1) - np.linalg.norm(Xa - Xb, axis=1)) / np.sqrt(YDIM)).mean()),
           "per_category_V_kernel": [float(H[cc == c].mean()) for c in range(NCAT)]}
    (d / f"key_sensitivity_{arm}_final_large.json").write_text(json.dumps(out, indent=1))
    print(f"{arm} b{block}: V_kernel {out['V_kernel']:.4f} ± {out['se']:.4f} (n={n}; Hoeffding r={r:.4f}; lower bound {out['V_lower_bound_95']:.4f}) V_energy {out['V_energy']:.4f}", flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument("cmd", choices=["make-data", "train", "sample-evaluate", "baselines", "audit-v", "solver-audit", "diag-final", "report"])
    p.add_argument("--block", type=int, default=0); p.add_argument("--blocks", type=int, nargs="*", default=[0, 1]); p.add_argument("--arm", default="K1"); p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    {"make-data": lambda: make_data(a.block, a.resume), "train": lambda: train(a.block, a.arm, a.resume), "sample-evaluate": lambda: sample_evaluate(a.block, a.arm, a.resume),
     "baselines": lambda: baselines(a.block), "audit-v": lambda: audit_v(a.block, a.arm), "solver-audit": lambda: solver_audit(a.block, a.arm), "diag-final": lambda: diag_final(a.block, a.arm), "report": lambda: report(a.blocks)}[a.cmd]()


if __name__ == "__main__":
    main()
