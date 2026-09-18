#!/usr/bin/env python3
"""Synthetic study: does conditional flow matching learn conditional uncertainty when each condition is observed k times?

Ground truth: x ~ U[-1,1]^2, y | x ~ N(f(x), sigma(x)^2 K) in R^m (m=16), f smooth (random Fourier features),
sigma(x) heteroscedastic, K an exponential correlation. Everything about p(y|x) is known, so calibration can be
measured against the TRUE conditional law (spread / true sigma, energy score vs fresh draws, rank of fresh draws).

Grid: realizations per condition k in K_GRID, with (a) fixed sample budget N*k = BUDGET and (b) fixed N = N_FIXED.
For every k a small MLP-CFM is trained; every EVAL_EVERY steps we sample M members on held-out and on training
conditions and record spread/true-sigma, RMSE of the mean vs f(x), rank-histogram TV of fresh draws, energy score.
Prediction under the 'one realisation per condition' hypothesis: for k=1 spread/true-sigma decays with training
(collapse) while OOS skill first improves then degrades; for large k it converges to 1.
"""
from __future__ import annotations

import argparse, json, math, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(4)
M_DIM = 16
K_GRID = (1, 2, 4, 16)
BUDGET = 2048
N_FIXED = 512


class Truth:
    def __init__(self, seed=0):
        g = np.random.default_rng(seed)
        self.W = g.normal(size=(2, 24)) * 2.0
        self.b = g.uniform(0, 2 * np.pi, 24)
        self.A = g.normal(size=(24, M_DIM)) / np.sqrt(24)
        idx = np.arange(M_DIM)
        K = np.exp(-np.abs(idx[:, None] - idx[None]) / 4.0)
        self.L = np.linalg.cholesky(K)

    def f(self, x):
        return np.cos(x @ self.W + self.b) @ self.A

    def sigma(self, x):
        return 0.25 + 0.15 * np.sin(3.0 * x[:, 0]) * np.cos(2.0 * x[:, 1])

    def sample(self, x, rng, n=1):
        eps = rng.normal(size=(n, x.shape[0], M_DIM)) @ self.L.T
        return self.f(x)[None] + self.sigma(x)[None, :, None] * eps  # (n, N, m)


class VNet(nn.Module):
    def __init__(self, h=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(M_DIM + 2 + 1, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, M_DIM))

    def forward(self, yt, x, t):
        return self.net(torch.cat([yt, x, t], -1))


@torch.no_grad()
def sample(net, x, M, steps=32, gen=None):
    N = x.shape[0]
    xr = x.repeat_interleave(M, 0)
    y = torch.randn(N * M, M_DIM, generator=gen)
    for i in range(steps):
        t = torch.full((N * M, 1), i / steps)
        y = y + net(y, xr, t) / steps
    return y.view(N, M, M_DIM)


def evaluate(net, truth, x, rng, M=32, gen=None):
    xt = torch.as_tensor(x, dtype=torch.float32)
    ens = sample(net, xt, M, gen=gen).numpy()  # (N, M, m)
    mean = ens.mean(1); sd = ens.std(1, ddof=1)
    f = truth.f(x); sig = truth.sigma(x)
    rmse = float(np.sqrt(np.mean((mean - f) ** 2)))
    spread = float(np.sqrt(np.mean(sd ** 2)))
    true_spread = float(np.sqrt(np.mean(sig ** 2)))  # marginal sd of each coordinate is sigma(x) (K has unit diagonal)
    # fresh draws from the truth for proper scores / ranks
    y = truth.sample(x, rng, n=1)[0]  # (N, m)
    below = (ens < y[:, None]).sum(1); ranks = below.ravel()
    h = np.bincount(ranks, minlength=M + 1) / ranks.size
    tv = float(0.5 * np.abs(h - 1 / (M + 1)).sum())
    # energy score (multivariate proper score) vs the fresh draw
    d1 = np.linalg.norm(ens - y[:, None], axis=-1).mean(1)
    d2 = np.linalg.norm(ens[:, :, None] - ens[:, None], axis=-1).sum((1, 2)) / (M * (M - 1))
    es = float(np.mean(d1 - 0.5 * d2))
    return dict(rmse_mean=rmse, spread=spread, true_spread=true_spread, spread_over_true=spread / true_spread,
                spread_over_rmse=spread / rmse, rank_tv=tv, energy_score=es)


def run(k: int, N: int, steps: int, eval_every: int, seed: int, truth: Truth, out: dict, tag: str):
    rng = np.random.default_rng(seed); torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed + 1)
    x_train = rng.uniform(-1, 1, (N, 2)); y_train = truth.sample(x_train, rng, n=k)  # (k, N, m)
    X = torch.as_tensor(np.repeat(x_train[None], k, 0).reshape(-1, 2), dtype=torch.float32)
    Y = torch.as_tensor(y_train.reshape(-1, M_DIM), dtype=torch.float32)
    x_test = rng.uniform(-1, 1, (256, 2)); x_in = x_train[:256]
    net = VNet(); opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    n = X.shape[0]; curve = []
    t0 = time.time()
    for step in range(1, steps + 1):
        idx = torch.randint(0, n, (256,))
        x, y1 = X[idx], Y[idx]
        y0 = torch.randn_like(y1); t = torch.rand(256, 1)
        yt = (1 - t) * y0 + t * y1
        loss = ((net(yt, x, t) - (y1 - y0)) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % eval_every == 0:
            net.eval()
            e_out = evaluate(net, truth, x_test, rng, gen=gen); e_in = evaluate(net, truth, x_in, rng, gen=gen)
            net.train()
            curve.append(dict(step=step, loss=float(loss), oos=e_out, ins=e_in))
            print(f"[{tag} k={k} N={N}] step {step:6d} loss {float(loss):.4f} | OOS rmse {e_out['rmse_mean']:.3f} spread/true {e_out['spread_over_true']:.3f} s/r {e_out['spread_over_rmse']:.2f} rankTV {e_out['rank_tv']:.3f} ES {e_out['energy_score']:.4f} | in-sample rmse {e_in['rmse_mean']:.3f} spread/true {e_in['spread_over_true']:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    out[f"{tag}_k{k}_N{N}"] = dict(k=k, N=N, curve=curve)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, default=Path("docs/research/general_ml/synth_realizations_per_condition.json"))
    a = p.parse_args()
    truth = Truth(seed=123)
    out = {}
    for k in K_GRID:  # fixed budget: N*k = BUDGET
        run(k, BUDGET // k, a.steps, a.eval_every, a.seed, truth, out, "budget")
        a.output.write_text(json.dumps(out, indent=1))
    for k in (1, 4, 16):  # fixed number of conditions
        if k * N_FIXED == BUDGET:
            continue
        run(k, N_FIXED, a.steps, a.eval_every, a.seed, truth, out, "fixedN")
        a.output.write_text(json.dumps(out, indent=1))
    print("done")


if __name__ == "__main__":
    main()
