#!/usr/bin/env python3
"""Synthetic: condition = NOISY PARTIAL OBSERVATION of the target (no key / no background) — the 'tracks only' design.

Truth as in synth_realizations_per_condition (x ~ U[-1,1]^2 latent, y | x ~ N(f(x), sigma(x)^2 K), m = 16).
Condition: S = 6 random coordinates of y observed with noise N(0, s_obs^2); the latent x is NOT given.
The true posterior p(y | obs) = sum_x w(x) N(y; mu_post(x), Sigma_post(x)) is computed exactly on an x-grid
(Gaussian likelihood of the observations given x), so spread and ranks are checked against the true posterior.
Hypothesis: with s_obs = 0 the exact observed values identify the training target (fingerprint) and the model collapses
with training; with s_obs > 0 the condition has many consistent targets and the model stays calibrated.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from docs.research.general_ml.synth_realizations_per_condition import Truth, M_DIM

torch.set_num_threads(4)
S_OBS = 6
GRID = np.stack(np.meshgrid(np.linspace(-1, 1, 81), np.linspace(-1, 1, 81), indexing="ij"), -1).reshape(-1, 2)


class VNet(nn.Module):
    def __init__(self, h=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(M_DIM + 2 * M_DIM + 1, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, M_DIM))

    def forward(self, yt, c, t):
        return self.net(torch.cat([yt, c, t], -1))


def make_obs(y, rng, s_obs):
    N = y.shape[0]; mask = np.zeros((N, M_DIM)); vals = np.zeros((N, M_DIM))
    for i in range(N):
        idx = rng.choice(M_DIM, S_OBS, replace=False); mask[i, idx] = 1
        vals[i, idx] = y[i, idx] + s_obs * rng.normal(size=S_OBS)
    return np.concatenate([mask, vals], 1)


def true_posterior(truth, cond, s_obs, rng, n_draw=256):
    """Exact posterior over y given noisy partial obs, marginalising x on a grid. Returns draws and marginal sd."""
    mask = cond[:M_DIM] > 0; obs = cond[M_DIM:][mask]
    F = truth.f(GRID); sig = truth.sigma(GRID); K = truth.L @ truth.L.T
    Kss = K[np.ix_(mask, mask)]
    # log-likelihood of obs under each grid x: N(obs; f_S(x), sigma(x)^2 Kss + s_obs^2 I)
    ll = np.empty(len(GRID)); posts = []
    for g in range(len(GRID)):
        C = sig[g] ** 2 * Kss + (s_obs ** 2 + 1e-6) * np.eye(mask.sum())
        r = obs - F[g][mask]
        Ci = np.linalg.inv(C); ll[g] = -0.5 * r @ Ci @ r - 0.5 * np.linalg.slogdet(C)[1]
    w = np.exp(ll - ll.max()); w /= w.sum()
    # draw: pick x by weight, then y | x, obs (Gaussian conditioning)
    gi = rng.choice(len(GRID), n_draw, p=w); draws = np.empty((n_draw, M_DIM))
    for j, g in enumerate(gi):
        Sy = sig[g] ** 2 * K; C = Sy[np.ix_(mask, mask)] + (s_obs ** 2 + 1e-6) * np.eye(mask.sum())
        G = Sy[:, mask] @ np.linalg.inv(C)
        mu = F[g] + G @ (obs - F[g][mask]); Sp = Sy - G @ Sy[mask, :]
        Sp = 0.5 * (Sp + Sp.T) + 1e-9 * np.eye(M_DIM)
        draws[j] = rng.multivariate_normal(mu, Sp)
    return draws


@torch.no_grad()
def sample(net, c, M, steps=32, gen=None):
    N = c.shape[0]; cr = c.repeat_interleave(M, 0); y = torch.randn(N * M, M_DIM, generator=gen)
    for i in range(steps):
        y = y + net(y, cr, torch.full((N * M, 1), i / steps)) / steps
    return y.view(N, M, M_DIM)


def evaluate(net, truth, cond, y_true, post_draws, rng, M=32, gen=None):
    ens = sample(net, torch.as_tensor(cond, dtype=torch.float32), M, gen=gen).numpy()
    sd = ens.std(1, ddof=1); mean = ens.mean(1)
    true_sd = post_draws.std(1, ddof=1); true_mean = post_draws.mean(1)
    spread = float(np.sqrt(np.mean(sd ** 2))); true_spread = float(np.sqrt(np.mean(true_sd ** 2)))
    rmse = float(np.sqrt(np.mean((mean - y_true) ** 2))); rmse_opt = float(np.sqrt(np.mean((true_mean - y_true) ** 2)))
    below = (ens < y_true[:, None]).sum(1).ravel(); h = np.bincount(below, minlength=M + 1) / below.size
    tv = float(0.5 * np.abs(h - 1 / (M + 1)).sum())
    d1 = np.linalg.norm(ens - y_true[:, None], axis=-1).mean(1)
    d2 = np.linalg.norm(ens[:, :, None] - ens[:, None], axis=-1).sum((1, 2)) / (M * (M - 1))
    es = float(np.mean(d1 - 0.5 * d2))
    dt1 = np.linalg.norm(post_draws[:, :M] - y_true[:, None], axis=-1).mean(1)
    dt2 = np.linalg.norm(post_draws[:, :M, None] - post_draws[:, None, :M], axis=-1).sum((1, 2)) / (M * (M - 1))
    es_true = float(np.mean(dt1 - 0.5 * dt2))
    return dict(spread=spread, true_spread=true_spread, spread_over_true=spread / true_spread, rmse_mean=rmse, rmse_optimal=rmse_opt, rank_tv=tv, energy_score=es, energy_score_true_posterior=es_true)


def run(s_obs, N, steps, eval_every, seed, truth, out, tag):
    rng = np.random.default_rng(seed); torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed + 1)
    x_tr = rng.uniform(-1, 1, (N, 2)); y_tr = truth.sample(x_tr, rng)[0]; c_tr = make_obs(y_tr, rng, s_obs)
    x_te = rng.uniform(-1, 1, (128, 2)); y_te = truth.sample(x_te, rng)[0]; c_te = make_obs(y_te, rng, s_obs)
    post = np.stack([true_posterior(truth, c_te[i], s_obs, rng) for i in range(len(c_te))])
    C = torch.as_tensor(c_tr, dtype=torch.float32); Y = torch.as_tensor(y_tr, dtype=torch.float32)
    net = VNet(); opt = torch.optim.Adam(net.parameters(), lr=1e-3); curve = []; t0 = time.time()
    for step in range(1, steps + 1):
        idx = torch.randint(0, N, (256,)); y1 = Y[idx]; c = C[idx]
        y0 = torch.randn_like(y1); t = torch.rand(256, 1); yt = (1 - t) * y0 + t * y1
        loss = ((net(yt, c, t) - (y1 - y0)) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
        if step % eval_every == 0:
            net.eval(); e = evaluate(net, truth, c_te, y_te, post, rng, gen=gen); net.train()
            curve.append(dict(step=step, loss=float(loss), oos=e))
            print(f"[{tag} s_obs={s_obs} N={N}] step {step:6d} | spread/true {e['spread_over_true']:.3f} rmse {e['rmse_mean']:.3f} (opt {e['rmse_optimal']:.3f}) rankTV {e['rank_tv']:.3f} ES {e['energy_score']:.4f} (true {e['energy_score_true_posterior']:.4f}) ({time.time()-t0:.0f}s)", flush=True)
    out[f"{tag}_s{s_obs}_N{N}"] = dict(s_obs=s_obs, N=N, curve=curve)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--steps", type=int, default=20000); p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--output", type=Path, default=Path("docs/research/general_ml/synth_noisy_partial_obs.json")); a = p.parse_args()
    truth = Truth(seed=123); out = {}
    for N in (512, 2048):
        for s_obs in (0.0, 0.03, 0.1):
            run(s_obs, N, a.steps, a.eval_every, 0, truth, out, "partial"); a.output.write_text(json.dumps(out, indent=1))
    print("done")


if __name__ == "__main__":
    main()
