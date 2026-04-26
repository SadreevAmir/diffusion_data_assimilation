from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_rank_histogram(counts: np.ndarray, path: str, title: str) -> None:
    counts = np.asarray(counts, dtype=np.float64)
    probs = counts / max(1.0, counts.sum())
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(np.arange(probs.size), probs, color="black", width=0.85)
    ax.set_xlabel("Rank of truth")
    ax.set_ylabel("Probability")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_coverage(levels: list[float], empirical: list[float], path: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(levels, empirical, marker="o", label="empirical")
    ax.plot([0, 1], [0, 1], color="black", linewidth=1, linestyle="--", label="ideal")
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_spread_skill(rows: list[dict[str, float]], path: str, title: str) -> None:
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(5, 5))
    spread = [r["mean_spread"] for r in rows]
    rmse = [r["rmse"] for r in rows]
    ax.plot(spread, rmse, marker="o")
    lim = max(max(spread), max(rmse), 1e-12)
    ax.plot([0, lim], [0, lim], color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("Predicted spread")
    ax.set_ylabel("Empirical RMSE")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_density_curve(rows: list[dict[str, object]], path: str, metric: str) -> None:
    grouped: dict[float, list[float]] = {}
    for row in rows:
        grouped.setdefault(float(row["density"]), []).append(float(row[metric]))
    xs = sorted(grouped)
    ys = [float(np.mean(grouped[x])) for x in xs]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, ys, marker="o")
    ax.set_xlabel("Observation density")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs observation density")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_example_panel(x_true: np.ndarray, mask: np.ndarray, ensemble: np.ndarray, path: str, max_members: int = 3) -> None:
    mean = ensemble.mean(axis=0)
    spread = ensemble.std(axis=0)
    err = mean - x_true
    n_members = min(max_members, ensemble.shape[0])
    cols = 5 + n_members
    fig, axes = plt.subplots(1, cols, figsize=(2.4 * cols, 2.7))
    panels = [
        (x_true[0], "x_true", "viridis"),
        (mask, "mask", "gray"),
        (mean[0], "ens mean", "viridis"),
        (spread[0], "ens spread", "magma"),
        (err[0], "error", "coolwarm"),
    ]
    panels.extend((ensemble[i, 0], f"member {i}", "viridis") for i in range(n_members))
    for ax, (arr, title, cmap) in zip(axes, panels):
        ax.imshow(arr, cmap=cmap)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
