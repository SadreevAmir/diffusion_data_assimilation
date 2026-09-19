#!/usr/bin/env python3
"""Three laws (review §3): for the noisy-partial-observation synthetic (fixed observed coordinates, fresh noise per step)
compare, on the same held-out conditions,
  P*      : the true posterior p(y | noisy obs)  (grid marginalisation over the latent; grid convergence checked 81 -> 161)
  P^_{D,s}: the augmented empirical mixture sum_i w_i(c) delta_{y_i}, w_i ∝ phi_s(c - c_i)  (Theorem 2 target; no training)
  Q_theta : the trained network
  ODE(P^) : Euler-32 integration of the EXACT CFM velocity field of the mixture (isolates integrator error from learning)
Metrics against P*: spread ratio, energy score vs a fresh true draw (and the true-vs-true reference), rank TV of a fresh true
draw among members, mean-RMSE vs the true posterior mean. Reported for N in {512, 2048}, s_obs in {0.1, 0.3}.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np, torch
from docs.research.general_ml.synth_realizations_per_condition import Truth, M_DIM
from docs.research.general_ml import synth_noisy_partial_obs as S

torch.set_num_threads(4)
FM = S.FIXED_MASK


def true_posterior_grid(truth, cond, s_obs, rng, n_draw, ngrid):
    G = np.stack(np.meshgrid(np.linspace(-1, 1, ngrid), np.linspace(-1, 1, ngrid), indexing="ij"), -1).reshape(-1, 2)
    mask = cond[:M_DIM] > 0; obs = cond[M_DIM:][mask]
    F = truth.f(G); sig = truth.sigma(G); K = truth.L @ truth.L.T; Kss = K[np.ix_(mask, mask)]
    ll = np.empty(len(G))
    for g in range(len(G)):
        C = sig[g] ** 2 * Kss + (s_obs ** 2 + 1e-6) * np.eye(mask.sum()); r = obs - F[g][mask]
        ll[g] = -0.5 * r @ np.linalg.solve(C, r) - 0.5 * np.linalg.slogdet(C)[1]
    w = np.exp(ll - ll.max()); w /= w.sum()
    gi = rng.choice(len(G), n_draw, p=w); draws = np.empty((n_draw, M_DIM))
    for j, g in enumerate(gi):
        Sy = sig[g] ** 2 * K; C = Sy[np.ix_(mask, mask)] + (s_obs ** 2 + 1e-6) * np.eye(mask.sum()); Gm = Sy[:, mask] @ np.linalg.inv(C)
        mu = F[g] + Gm @ (obs - F[g][mask]); Sp = Sy - Gm @ Sy[mask, :]; Sp = 0.5 * (Sp + Sp.T) + 1e-9 * np.eye(M_DIM)
        draws[j] = rng.multivariate_normal(mu, Sp)
    return draws


def true_posterior_moments(truth, cond, s_obs, ngrid):
    """Exact posterior mean and pixelwise variance on a grid (for the grid-convergence check)."""
    G = np.stack(np.meshgrid(np.linspace(-1, 1, ngrid), np.linspace(-1, 1, ngrid), indexing="ij"), -1).reshape(-1, 2)
    mask = cond[:M_DIM] > 0; obs = cond[M_DIM:][mask]
    F = truth.f(G); sig = truth.sigma(G); K = truth.L @ truth.L.T; Kss = K[np.ix_(mask, mask)]
    ll = np.empty(len(G)); mus = np.empty((len(G), M_DIM)); vars_ = np.empty((len(G), M_DIM))
    for g in range(len(G)):
        Sy = sig[g] ** 2 * K; C = Sy[np.ix_(mask, mask)] + (s_obs ** 2 + 1e-6) * np.eye(mask.sum()); r = obs - F[g][mask]
        Ci = np.linalg.inv(C); ll[g] = -0.5 * r @ Ci @ r - 0.5 * np.linalg.slogdet(C)[1]
        Gm = Sy[:, mask] @ Ci; mus[g] = F[g] + Gm @ r; vars_[g] = np.diag(Sy - Gm @ Sy[mask, :])
    w = np.exp(ll - ll.max()); w /= w.sum()
    mean = w @ mus; var = w @ (vars_ + mus ** 2) - mean ** 2
    return mean, var


def mixture_weights(c_test, C_train, s_obs):
    """w_i(c) ∝ phi_s(c - c_i) on the observed coordinates (the fresh-noise kernel of Theorem 2)."""
    d2 = ((c_test[None, M_DIM + FM] - C_train[:, M_DIM + FM]) ** 2).sum(-1)
    logw = -0.5 * d2 / max(s_obs, 1e-3) ** 2; logw -= logw.max(); w = np.exp(logw); return w / w.sum()


def mixture_draws(w, Y_train, rng, M):
    return Y_train[rng.choice(len(w), M, p=w)]


def exact_cfm_ode(w, Y_train, rng, M, steps=32):
    """Integrate the exact CFM velocity of the mixture sum_i w_i delta_{y_i} with linear paths and Gaussian source:
    v*(z,t) = (sum_i r_i y_i - z)/(1-t), r_i ∝ w_i phi_{(1-t)^2 I}(z - t y_i)."""
    Y = Y_train; out = np.empty((M, M_DIM))
    for m in range(M):
        z = rng.standard_normal(M_DIM)
        for k in range(steps):
            t = k / steps; s2 = max((1 - t) ** 2, 1e-6)
            logr = np.log(w + 1e-300) - 0.5 * ((z[None] - t * Y) ** 2).sum(-1) / s2; logr -= logr.max(); r = np.exp(logr); r /= r.sum()
            v = (r @ Y - z) / max(1 - t, 1e-3); z = z + v / steps
        out[m] = z
    return out


def metrics(ens, true_draws, y_fresh, rng):
    M = ens.shape[1]
    spread = float(np.sqrt(ens.var(1, ddof=1).mean())); tspread = float(np.sqrt(true_draws.var(1, ddof=1).mean()))
    rmse = float(np.sqrt(((ens.mean(1) - true_draws.mean(1)) ** 2).mean()))
    d1 = np.linalg.norm(ens - y_fresh[:, None], axis=-1).mean(1)
    d2 = np.linalg.norm(ens[:, :, None] - ens[:, None], axis=-1).sum((1, 2)) / (M * (M - 1))
    es = float(np.mean(d1 - 0.5 * d2))
    below = (ens < y_fresh[:, None]).sum(1); ties = (ens == y_fresh[:, None]).sum(1)
    r = (below + np.floor(rng.random(below.shape) * (ties + 1))).astype(int).ravel(); h = np.bincount(r, minlength=M + 1) / r.size
    return dict(spread_over_true=spread / tspread, rmse_mean_vs_true_mean=rmse, energy_score=es, rank_tv=float(0.5 * np.abs(h - 1 / (M + 1)).sum()))


def main():
    p = argparse.ArgumentParser(); p.add_argument("--steps", type=int, default=20000); p.add_argument("--n-test", type=int, default=96); p.add_argument("--output", type=Path, default=Path("docs/research/general_ml/synth_three_laws.json"))
    a = p.parse_args(); truth = Truth(seed=123); out = {}; M = 32
    for N in (512, 2048):
        for s_obs in (0.1, 0.3):
            rng = np.random.default_rng(0); torch.manual_seed(0); gen = torch.Generator().manual_seed(1)
            x_tr = rng.uniform(-1, 1, (N, 2)); y_tr = truth.sample(x_tr, rng)[0]; c_tr = S.make_obs(y_tr, rng, 0.0)
            x_te = rng.uniform(-1, 1, (a.n_test, 2)); y_te = truth.sample(x_te, rng)[0]; c_te = S.make_obs(y_te, rng, s_obs)
            t0 = time.time()
            m81 = [true_posterior_moments(truth, c_te[i], s_obs, 81) for i in range(a.n_test)]; m161 = [true_posterior_moments(truth, c_te[i], s_obs, 161) for i in range(a.n_test)]
            grid_check = dict(spread81=float(np.sqrt(np.mean([v for _, v in m81]))), spread161=float(np.sqrt(np.mean([v for _, v in m161]))), mean_diff=float(np.sqrt(np.mean([(a1 - a2) ** 2 for (a1, _), (a2, _) in zip(m81, m161)]))))
            post161 = np.stack([true_posterior_grid(truth, c_te[i], s_obs, rng, M, 161) for i in range(a.n_test)])
            y_fresh = np.stack([true_posterior_grid(truth, c_te[i], s_obs, rng, 1, 161)[0] for i in range(a.n_test)])
            print(f"[N={N} s={s_obs}] grid check: spread 81 {grid_check['spread81']:.4f} vs 161 {grid_check['spread161']:.4f}, mean diff {grid_check['mean_diff']:.4f} ({time.time()-t0:.0f}s)", flush=True)
            # laws
            W = np.stack([mixture_weights(c_te[i], c_tr, s_obs) for i in range(a.n_test)])
            n_eff = float(np.mean(1.0 / (W ** 2).sum(1))); wmax = float(np.mean(W.max(1)))
            mix = np.stack([mixture_draws(W[i], y_tr, rng, M) for i in range(a.n_test)])
            ode = np.stack([exact_cfm_ode(W[i], y_tr, rng, M) for i in range(a.n_test)])
            # network (same recipe as v3, fresh noise per step)
            C = torch.as_tensor(c_tr, dtype=torch.float32); Y = torch.as_tensor(y_tr, dtype=torch.float32)
            net = S.VNet(); opt = torch.optim.Adam(net.parameters(), lr=1e-3)
            for step in range(1, a.steps + 1):
                idx = torch.randint(0, N, (256,)); y1 = Y[idx]; c = C[idx].clone(); c[:, M_DIM:] += s_obs * torch.randn_like(c[:, M_DIM:]) * c[:, :M_DIM]
                y0 = torch.randn_like(y1); t = torch.rand(256, 1); yt = (1 - t) * y0 + t * y1
                loss = ((net(yt, c, t) - (y1 - y0)) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
            net.eval(); Q = S.sample(net, torch.as_tensor(c_te, dtype=torch.float32), M, gen=gen).numpy()
            res = {"grid_check": grid_check, "n_eff_mean": n_eff, "w_max_mean": wmax}
            for name, ens in (("true_posterior", post161), ("empirical_mixture", mix), ("exact_cfm_ode_of_mixture", ode), ("network", Q)):
                res[name] = metrics(ens, post161, y_fresh, rng)
                print(f"[N={N} s={s_obs}] {name:26s} spread/true {res[name]['spread_over_true']:.3f} rmse(mean) {res[name]['rmse_mean_vs_true_mean']:.3f} ES {res[name]['energy_score']:.3f} rankTV {res[name]['rank_tv']:.3f}", flush=True)
            print(f"[N={N} s={s_obs}] mixture N_eff {n_eff:.1f}, max w {wmax:.2f}", flush=True)
            out[f"N{N}_s{s_obs}"] = res; a.output.write_text(json.dumps(out, indent=1))
    print("done")


if __name__ == "__main__":
    main()
