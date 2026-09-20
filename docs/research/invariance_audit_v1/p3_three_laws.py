#!/usr/bin/env python3
"""P3 (Astra pack v1, EXPERIMENT_SPECIFICATIONS §P3): the 'three laws' on a linear-Gaussian teacher with an exact posterior.

Teacher: X ~ N(0, I_2), Y = A X + eta in R^8, A = QR-orthonormalised fixed Gaussian 8x2 (seed 1729, signs canonicalised),
eta ~ N(0, S0), S0[i,j] = .3^2 exp(-|i-j|/2) + .05^2 1{i=j}.  Training: N fixed pairs (X_i, Y_i) and keys K_i in R^8; at every
presentation Xt = X_i + tau E_x, Kt = rho K_i + sqrt(1-rho^2) E_k.  Evaluation conditions (Xt, K) and truths Y follow the same
population law.  True posterior: Y | Xt=x ~ N(A x/(1+tau^2), S0 + tau^2/(1+tau^2) A A^T)  (independent of rho and K).
Empirical target weights: w_i(x,k) ∝ exp(-|x-X_i|^2/(2 tau^2) - |k - rho K_i|^2/(2(1-rho^2))).

Four laws per validation condition: (1) exact Gaussian posterior; (2) iid atoms from the empirical conditional measure;
(3) exact CFM field of the empirical mixture integrated to T<1 (DOP853), compared with the analytic law sum_i w_i N(T y_i,(1-T)^2 I);
(4) the trained network with the same T and Heun-64.  Grid: N in {256, 2048}, tau in {.1,.5}, rho in {0,.9}, seed blocks 0,1.
Executed before review (user instruction: run everything); flagged in the report.
"""
from __future__ import annotations
import argparse, csv, json, os, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE / "astra_pack_v1" / "reference")); sys.path.insert(0, str(HERE))
import core as R  # noqa: E402
import p1_pilot as P  # noqa: E402  (time embedding, torch helpers)

ROOT = HERE.parents[2]; OUT = ROOT / "autoresearch_results" / "invariance_audit_v1" / "p3_three_laws"
XDIM, YDIM, KDIM = 2, 8, 8; SCALES = (0.25, 0.5, 1.0, 2.0); T_END = 0.999; STEPS = 20000; M = 128; NVAL = 128
GRID = [(N, tau, rho) for N in (256, 2048) for tau in (0.1, 0.5) for rho in (0.0, 0.9)]


def teacher_matrix():
    g = np.random.default_rng(1729).standard_normal((YDIM, XDIM)); q, r = np.linalg.qr(g)
    return q * np.sign(np.diag(r))[None, :]


A = teacher_matrix(); idx = np.arange(YDIM); S0 = 0.3 ** 2 * np.exp(-np.abs(idx[:, None] - idx[None, :]) / 2) + 0.05 ** 2 * np.eye(YDIM)


def posterior(x, tau):
    return A @ x / (1 + tau ** 2), S0 + tau ** 2 / (1 + tau ** 2) * A @ A.T


def seeds(block):  # same stream names as P1, offset by 500 to keep P3 blocks distinct
    return {k: v + 500 + block for k, v in P.CFG["seed_offsets"].items()}


def run_dir(N, tau, rho, block):
    d = OUT / f"N{N}_tau{tau}_rho{rho}_b{block}"; d.mkdir(parents=True, exist_ok=True); return d


def make_data(N, tau, rho, block):
    s = seeds(block); rd = np.random.default_rng(s["data"]); rk = np.random.default_rng(s["keys"])
    X = rd.standard_normal((N, XDIM)); Y = X @ A.T + R.gaussian_noise(S0, N, rd); K = rk.standard_normal((N, KDIM))
    return X, Y, K


def net_input(torch, z, x, k, t):
    return torch.cat([z, x, k, P.time_embed(torch, t)], dim=1)


def build(torch, init_seed):
    import torch.nn as nn
    g = torch.Generator().manual_seed(init_seed); dims = [YDIM + XDIM + KDIM + 32, 256, 256, 256, YDIM]; layers = []
    for i in range(len(dims) - 1):
        lin = nn.Linear(dims[i], dims[i + 1])
        with torch.no_grad():
            b = float(np.sqrt(6.0 / (dims[i] + dims[i + 1]))); lin.weight.uniform_(-b, b, generator=g); lin.bias.zero_()
        layers.append(lin)
        if i < len(dims) - 2:
            layers.append(nn.SiLU())
    return nn.Sequential(*layers)


def train(N, tau, rho, block):
    torch = P._torch(); d = run_dir(N, tau, rho, block); s = seeds(block)
    if (d / "ckpt_final.pt").exists():
        return
    X, Y, K = make_data(N, tau, rho, block); np.savez(d / "train_data.npz", X=X, Y=Y, K=K)
    bs = 256; ib = np.random.default_rng(s["batches"]).integers(0, N, size=(STEPS, bs)); rf = np.random.default_rng(s["flow"])
    Z0 = rf.standard_normal((STEPS, bs, YDIM)).astype(np.float32); Tt = rf.random((STEPS, bs)).astype(np.float32); ra = np.random.default_rng(s["augmentation"])
    model = build(torch, s["init"]); opt = torch.optim.Adam(model.parameters(), lr=2e-4); Xt_, Yt_, Kt_ = (torch.as_tensor(a, dtype=torch.float32) for a in (X, Y, K))
    t0 = time.time(); log = []
    for step in range(STEPS):
        b = torch.as_tensor(ib[step]); y = Yt_[b]
        x = Xt_[b] + tau * torch.as_tensor(ra.standard_normal((bs, XDIM)).astype(np.float32))
        k = rho * Kt_[b] + float(np.sqrt(1 - rho * rho)) * torch.as_tensor(ra.standard_normal((bs, KDIM)).astype(np.float32))
        z0 = torch.as_tensor(Z0[step]); t = torch.as_tensor(Tt[step]); zt = (1 - t)[:, None] * z0 + t[:, None] * y
        loss = ((model(net_input(torch, zt, x, k, t)) - (y - z0)) ** 2).mean(); opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 2000 == 0:
            log.append({"step": step, "loss": float(loss)})
    torch.save(model.state_dict(), d / "ckpt_final.pt")
    (d / "run_manifest.json").write_text(json.dumps({"N": N, "tau": tau, "rho": rho, "block": block, "seeds": s, "steps": STEPS, "runtime_s": time.time() - t0, "loss_log": log, "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=1))
    print(f"trained N{N} tau{tau} rho{rho} b{block} in {time.time() - t0:.0f}s loss {float(loss):.4f}", flush=True)


def heun(torch, model, x, k, z0, steps=64, t_end=T_END):
    x = torch.as_tensor(x.astype(np.float32)); k = torch.as_tensor(k.astype(np.float32)); z = torch.as_tensor(z0.astype(np.float32)); ts = np.linspace(0, t_end, steps + 1)
    with torch.no_grad():
        for i in range(steps):
            h = float(ts[i + 1] - ts[i]); t0 = torch.full((len(z),), float(ts[i])); t1 = torch.full((len(z),), float(ts[i + 1]))
            v0 = model(net_input(torch, z, x, k, t0)); v1 = model(net_input(torch, z + h * v0, x, k, t1)); z = z + 0.5 * h * (v0 + v1)
    return z.numpy().astype(np.float64)


def evaluate(N, tau, rho, block):
    torch = P._torch(); from scipy.integrate import solve_ivp
    d = run_dir(N, tau, rho, block); s = seeds(block); data = np.load(d / "train_data.npz"); X, Y, K = data["X"], data["Y"], data["K"]
    model = build(torch, s["init"]); model.load_state_dict(torch.load(d / "ckpt_final.pt")); model.eval()
    rv = np.random.default_rng(s["validation"])
    Xv = rv.standard_normal((NVAL, XDIM)); Xt = Xv + tau * rv.standard_normal((NVAL, XDIM)); Kv = rv.standard_normal((NVAL, KDIM))
    truths = np.stack([R.gaussian_noise(posterior(x, tau)[1], 32, rv) + posterior(x, tau)[0] for x in Xt])  # 32 fresh truths per condition
    rows = []
    for i in range(NVAL):
        mu, C = posterior(Xt[i], tau); G = R.GaussianMixture(np.array([1.0]), mu[None], C[None])
        logw = -((Xt[i] - X) ** 2).sum(1) / (2 * tau ** 2) - ((Kv[i] - rho * K) ** 2).sum(1) / (2 * (1 - rho * rho))
        w = np.exp(logw - logw.max()); w /= w.sum(); ess = float(1 / (w @ w)); srt = np.sort(((Xt[i] - X) ** 2).sum(1))
        laws = {}
        laws["1_true_posterior"] = G.sample(M, rv)
        laws["2_empirical_atoms"] = Y[rv.choice(N, M, p=w)]
        laws["2b_empirical_at_T"] = R.exact_path_samples(M, T_END, Y, w, rv)  # analytic law at T: sum w_i N(T y_i, (1-T)^2 I)
        z0 = rv.standard_normal((M, YDIM))
        laws["4_network"] = heun(torch, model, np.repeat(Xt[i:i + 1], M, 0), np.repeat(Kv[i:i + 1], M, 0), z0)
        if i < 16:  # exact field of the empirical mixture, DOP853, 64 starts (numerical audit)
            sol = solve_ivp(lambda t, y: R.exact_cfm_field(y.reshape(-1, YDIM), t, Y, w).ravel(), (0.0, T_END), z0[:64].ravel(), method="DOP853", rtol=1e-7, atol=1e-8)
            laws["3_exact_field_ode"] = sol.y[:, -1].reshape(-1, YDIM)
        for name, S in laws.items():
            xb = S.mean(0); cov = np.cov(S, rowvar=False, ddof=1)
            rows.append({"cond": i, "law": name, "n": len(S), "mmd2_to_posterior": G.mmd2_from_samples(S, SCALES), "mean_err2": float(((xb - mu) ** 2).sum()), "cov_err_fro": float(np.linalg.norm(cov - C)),
                         "tr_ratio": float(np.trace(cov) / np.trace(C)), "crps_marginal": float(np.mean([R.empirical_score(S, y, "marginal") for y in truths[i]])), "energy": float(np.mean([R.empirical_score(S, y, "energy") for y in truths[i]])),
                         "ess": ess, "max_w": float(w.max()), "gap_two_smallest_sqdist": float(srt[1] - srt[0]), "N": N, "tau": tau, "rho": rho, "block": block})
    with (d / "laws_per_condition.csv").open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    agg = {}
    for name in sorted({r["law"] for r in rows}):
        sel = [r for r in rows if r["law"] == name]; agg[name] = {k: float(np.mean([r[k] for r in sel])) for k in ("mmd2_to_posterior", "mean_err2", "cov_err_fro", "tr_ratio", "crps_marginal", "energy")}
    agg["ess_mean"] = float(np.mean([r["ess"] for r in rows if r["law"] == "4_network"])); agg["ess_median"] = float(np.median([r["ess"] for r in rows if r["law"] == "4_network"]))
    (d / "summary.json").write_text(json.dumps(agg, indent=1))
    print(f"N{N} tau{tau} rho{rho} b{block}: ESS med {agg['ess_median']:.1f} | " + " ".join(f"{k.split('_')[0]}:{v['mmd2_to_posterior']:.4f}" for k, v in agg.items() if isinstance(v, dict)), flush=True)


def report():
    rows = []
    for (N, tau, rho) in GRID:
        for b in (0, 1):
            p = run_dir(N, tau, rho, b) / "summary.json"
            if p.exists():
                a = json.loads(p.read_text())
                for law, v in a.items():
                    if isinstance(v, dict):
                        rows.append({"N": N, "tau": tau, "rho": rho, "block": b, "law": law, "ess_median": a["ess_median"], **v})
    with (OUT / "p3_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for r in rows:
        print(f"N{r['N']:5d} tau{r['tau']} rho{r['rho']} b{r['block']} {r['law']:22s} mmd2 {r['mmd2_to_posterior']:.4f} tr {r['tr_ratio']:.2f} meanerr {r['mean_err2']:.4f} crps {r['crps_marginal']:.4f} ESSmed {r['ess_median']:.1f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("cmd", choices=["all", "report"]); a = p.parse_args()
    if a.cmd == "all":
        OUT.mkdir(parents=True, exist_ok=True)
        for (N, tau, rho) in GRID:
            for b in (0, 1):
                train(N, tau, rho, b); evaluate(N, tau, rho, b)
        report()
    else:
        report()
