#!/usr/bin/env python3
"""CPU toy admission for a face-aware conservative refinement kernel.

The experiment deliberately does not use a separate ice-mask generator.  Exact
zero/interior/cap faces are choices inside an autoregressive allocation kernel.
For a parent budget S and child capacities c_i, each child allocation is chosen
inside the exact feasible interval left by the remaining children.  The last
child is the remainder, so conservation is algebraic rather than penalized.

The script performs two checks:

1. FP64 round-trip and permutation tests on irregular/partial-water blocks.
2. A small learned conditional distribution on a four-child synthetic law with
   atoms on feasible faces and a continuous logistic-normal interior.

This is an algorithmic admission test only, not evidence about sea-ice skill.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


LOWER = 0
INTERIOR = 1
UPPER = 2
FORCED = 3


@dataclass
class EncodedStep:
    child: int
    branch: int
    logit_fraction: float | None


def feasible_interval(remaining: float, capacity: float, rest_capacity: float) -> tuple[float, float]:
    lower = max(0.0, remaining - rest_capacity)
    upper = min(capacity, remaining)
    return lower, upper


def encode_allocation(
    allocation: np.ndarray,
    capacities: np.ndarray,
    order: tuple[int, ...],
    atol: float = 1e-12,
) -> tuple[float, list[EncodedStep]]:
    allocation = np.asarray(allocation, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.float64)
    if allocation.shape != capacities.shape:
        raise ValueError("allocation and capacities must have identical shape")
    if sorted(order) != list(range(len(allocation))):
        raise ValueError("order must be a permutation of child indices")
    if np.any(allocation < -atol) or np.any(allocation > capacities + atol):
        raise ValueError("allocation violates child capacities")

    parent_sum = float(allocation.sum(dtype=np.float64))
    remaining = parent_sum
    steps: list[EncodedStep] = []
    for position, child in enumerate(order[:-1]):
        rest = order[position + 1 :]
        rest_capacity = float(capacities[list(rest)].sum(dtype=np.float64))
        lower, upper = feasible_interval(remaining, float(capacities[child]), rest_capacity)
        width = upper - lower
        value = float(allocation[child])
        if width <= atol:
            branch = FORCED
            logit_fraction = None
        else:
            fraction = (value - lower) / width
            if fraction <= atol:
                branch = LOWER
                logit_fraction = None
            elif fraction >= 1.0 - atol:
                branch = UPPER
                logit_fraction = None
            else:
                branch = INTERIOR
                logit_fraction = math.log(fraction) - math.log1p(-fraction)
        steps.append(EncodedStep(child, branch, logit_fraction))
        remaining -= value
    return parent_sum, steps


def decode_allocation(
    parent_sum: float,
    capacities: np.ndarray,
    order: tuple[int, ...],
    steps: list[EncodedStep],
) -> np.ndarray:
    capacities = np.asarray(capacities, dtype=np.float64)
    allocation = np.zeros_like(capacities)
    remaining = float(parent_sum)
    for position, step in enumerate(steps):
        child = order[position]
        if step.child != child:
            raise ValueError("encoded child order mismatch")
        rest = order[position + 1 :]
        rest_capacity = float(capacities[list(rest)].sum(dtype=np.float64))
        lower, upper = feasible_interval(remaining, float(capacities[child]), rest_capacity)
        if step.branch in (LOWER, FORCED):
            value = lower
        elif step.branch == UPPER:
            value = upper
        elif step.branch == INTERIOR:
            if step.logit_fraction is None:
                raise ValueError("interior branch requires a coordinate")
            fraction = 1.0 / (1.0 + math.exp(-step.logit_fraction))
            value = lower + (upper - lower) * fraction
        else:
            raise ValueError(f"unknown branch {step.branch}")
        allocation[child] = value
        remaining -= value
    allocation[order[-1]] = remaining
    return allocation


def random_observed_allocation(rng: np.random.Generator, capacities: np.ndarray) -> np.ndarray:
    modes = rng.choice(3, size=len(capacities), p=[0.30, 0.45, 0.25])
    fractions = rng.beta(1.6, 1.8, size=len(capacities))
    fractions[modes == 0] = 0.0
    fractions[modes == 2] = 1.0
    return capacities * fractions


def roundtrip_audit(seed: int, cases: int) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    max_roundtrip = 0.0
    max_budget_error = 0.0
    max_lower_violation = 0.0
    max_upper_violation = 0.0
    tested = 0
    patterns = [
        np.array([1.0, 1.0, 1.0, 1.0]),
        np.array([1.0, 0.0, 0.5, 1.0]),
        np.array([0.2, 0.8, 0.0, 1.0]),
        np.array([0.0, 0.0, 1.0, 0.3]),
    ]
    permutations = list(itertools.permutations(range(4)))
    for _ in range(cases):
        capacities = patterns[int(rng.integers(0, len(patterns)))].copy()
        capacities *= rng.uniform(0.1, 3.0)
        allocation = random_observed_allocation(rng, capacities)
        for order in permutations:
            parent_sum, encoded = encode_allocation(allocation, capacities, order)
            decoded = decode_allocation(parent_sum, capacities, order, encoded)
            max_roundtrip = max(max_roundtrip, float(np.max(np.abs(decoded - allocation))))
            max_budget_error = max(max_budget_error, abs(float(decoded.sum()) - parent_sum))
            max_lower_violation = max(max_lower_violation, float(np.max(np.maximum(-decoded, 0.0))))
            max_upper_violation = max(
                max_upper_violation, float(np.max(np.maximum(decoded - capacities, 0.0)))
            )
            tested += 1
    return {
        "cases": cases,
        "orders_per_case": len(permutations),
        "decoded_allocations": tested,
        "max_roundtrip_abs": max_roundtrip,
        "max_budget_abs": max_budget_error,
        "max_lower_violation": max_lower_violation,
        "max_upper_violation": max_upper_violation,
    }


def two_level_roundtrip_audit(seed: int, cases: int) -> dict[str, float | int]:
    """Round-trip a 1x1 -> 2x2 -> 4x4 irregular-capacity hierarchy."""
    rng = np.random.default_rng(seed + 7)
    groups = (
        np.array([0, 1, 4, 5]),
        np.array([2, 3, 6, 7]),
        np.array([8, 9, 12, 13]),
        np.array([10, 11, 14, 15]),
    )
    max_native_roundtrip = 0.0
    max_root_budget_error = 0.0
    max_group_budget_error = 0.0
    max_bound_violation = 0.0
    for _ in range(cases):
        water_weights = rng.uniform(0.05, 1.0, size=16)
        water_weights[rng.random(16) < 0.25] = 0.0
        if not np.any(water_weights > 0.0):
            water_weights[int(rng.integers(0, 16))] = 1.0
        native_capacities = water_weights * rng.uniform(0.4, 1.3)
        native = random_observed_allocation(rng, native_capacities)
        coarse_capacities = np.asarray([native_capacities[group].sum() for group in groups])
        coarse = np.asarray([native[group].sum() for group in groups])

        root_order = tuple(int(i) for i in rng.permutation(4))
        root_sum, root_steps = encode_allocation(coarse, coarse_capacities, root_order)
        decoded_coarse = decode_allocation(root_sum, coarse_capacities, root_order, root_steps)
        decoded_native = np.zeros(16, dtype=np.float64)
        for group_index, group in enumerate(groups):
            local_order = tuple(int(i) for i in rng.permutation(4))
            local_sum, local_steps = encode_allocation(
                native[group], native_capacities[group], local_order
            )
            local_decoded = decode_allocation(
                float(decoded_coarse[group_index]),
                native_capacities[group],
                local_order,
                local_steps,
            )
            decoded_native[group] = local_decoded
            max_group_budget_error = max(
                max_group_budget_error,
                abs(float(local_decoded.sum()) - float(decoded_coarse[group_index])),
                abs(local_sum - float(coarse[group_index])),
            )

        max_native_roundtrip = max(
            max_native_roundtrip, float(np.max(np.abs(decoded_native - native)))
        )
        max_root_budget_error = max(
            max_root_budget_error, abs(float(decoded_native.sum()) - float(native.sum()))
        )
        max_bound_violation = max(
            max_bound_violation,
            float(np.max(np.maximum(-decoded_native, 0.0))),
            float(np.max(np.maximum(decoded_native - native_capacities, 0.0))),
        )
    return {
        "cases": cases,
        "max_native_roundtrip_abs": max_native_roundtrip,
        "max_root_budget_abs": max_root_budget_error,
        "max_group_budget_abs": max_group_budget_error,
        "max_bound_violation": max_bound_violation,
    }


def feature_vector(context: float, parent_fraction: float, remaining: float, position: int) -> np.ndarray:
    remaining_capacity = 4.0 - position
    remaining_fraction = remaining / remaining_capacity
    return np.asarray(
        [
            context,
            context * context,
            parent_fraction,
            remaining_fraction,
            position / 2.0,
            math.sin(math.pi * context),
        ],
        dtype=np.float32,
    )


def true_step_parameters(features: np.ndarray) -> tuple[np.ndarray, float, float]:
    context = float(features[0])
    parent_fraction = float(features[2])
    remaining_fraction = float(features[3])
    position = float(features[4])
    logits = np.asarray(
        [
            -0.35 - 1.05 * context + 0.55 * (1.0 - remaining_fraction) - 0.20 * position,
            1.10 - 0.40 * abs(context) + 0.20 * math.cos(math.pi * parent_fraction),
            -0.35 + 1.05 * context + 0.55 * remaining_fraction + 0.20 * position,
        ],
        dtype=np.float64,
    )
    mean = 1.05 * context + 0.75 * (remaining_fraction - 0.5) + 0.25 * (position - 0.5)
    sigma = 0.48 + 0.08 * (1.0 - abs(context))
    return logits, mean, sigma


def sample_synthetic_kernel(
    rng: np.random.Generator, context: float, parent_fraction: float
) -> tuple[np.ndarray, list[tuple[np.ndarray, int, float | None]]]:
    capacities = np.ones(4, dtype=np.float64)
    parent_sum = 4.0 * parent_fraction
    remaining = parent_sum
    allocation = np.zeros(4, dtype=np.float64)
    records: list[tuple[np.ndarray, int, float | None]] = []
    for position in range(3):
        rest_capacity = 3.0 - position
        lower, upper = feasible_interval(remaining, 1.0, rest_capacity)
        width = upper - lower
        if width <= 1e-12:
            value = lower
        else:
            features = feature_vector(context, parent_fraction, remaining, position)
            logits, mean, sigma = true_step_parameters(features)
            probabilities = np.exp(logits - logits.max())
            probabilities /= probabilities.sum()
            branch = int(rng.choice(3, p=probabilities))
            z_value: float | None = None
            if branch == LOWER:
                value = lower
            elif branch == UPPER:
                value = upper
            else:
                z_value = float(rng.normal(mean, sigma))
                fraction = 1.0 / (1.0 + math.exp(-z_value))
                value = lower + width * fraction
            records.append((features, branch, z_value))
        allocation[position] = value
        remaining -= value
    allocation[3] = remaining
    return allocation, records


class TinyKernel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(6, 48), nn.SiLU(), nn.Linear(48, 48), nn.SiLU())
        self.branch = nn.Linear(48, 3)
        self.interior = nn.Linear(48, 2)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.net(features)
        logits = self.branch(hidden)
        mean, raw_log_sigma = self.interior(hidden).unbind(-1)
        log_sigma = raw_log_sigma.clamp(-2.5, 1.0)
        return logits, mean, log_sigma


def make_dataset(seed: int, samples: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    feature_rows: list[np.ndarray] = []
    branches: list[int] = []
    interiors: list[float] = []
    interior_mask: list[bool] = []
    for _ in range(samples):
        context = float(rng.uniform(-1.0, 1.0))
        if rng.random() < 0.08:
            parent_fraction = float(rng.choice([0.0, 1.0]))
        else:
            parent_fraction = float(rng.beta(1.4, 1.4))
        _, records = sample_synthetic_kernel(rng, context, parent_fraction)
        for features, branch, z_value in records:
            feature_rows.append(features)
            branches.append(branch)
            interior_mask.append(z_value is not None)
            interiors.append(0.0 if z_value is None else z_value)
    return (
        np.stack(feature_rows),
        np.asarray(branches, dtype=np.int64),
        np.asarray(interiors, dtype=np.float32),
        np.asarray(interior_mask, dtype=bool),
    )


def train_tiny_kernel(seed: int, train_samples: int, updates: int) -> tuple[TinyKernel, dict[str, float]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    features, branches, interiors, interior_mask = make_dataset(seed + 11, train_samples)
    test_features, test_branches, test_interiors, test_mask = make_dataset(seed + 29, 4000)
    x = torch.from_numpy(features)
    y_branch = torch.from_numpy(branches)
    y_z = torch.from_numpy(interiors)
    z_mask = torch.from_numpy(interior_mask)
    model = TinyKernel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed + 101)
    losses: list[float] = []
    for _ in range(updates):
        indices = torch.randint(0, len(x), (512,), generator=generator)
        logits, mean, log_sigma = model(x[indices])
        branch_loss = F.cross_entropy(logits, y_branch[indices])
        mask = z_mask[indices]
        if bool(mask.any()):
            residual = (y_z[indices][mask] - mean[mask]) / log_sigma[mask].exp()
            interior_loss = (0.5 * residual.square() + log_sigma[mask]).mean()
        else:
            interior_loss = mean.sum() * 0.0
        loss = branch_loss + 0.35 * interior_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    with torch.no_grad():
        tx = torch.from_numpy(test_features)
        logits, mean, log_sigma = model(tx)
        branch_nll = float(F.cross_entropy(logits, torch.from_numpy(test_branches)))
        mask = torch.from_numpy(test_mask)
        residual = (torch.from_numpy(test_interiors)[mask] - mean[mask]) / log_sigma[mask].exp()
        interior_nll = float((0.5 * residual.square() + log_sigma[mask]).mean())
        branch_accuracy = float((logits.argmax(-1) == torch.from_numpy(test_branches)).float().mean())
    return model, {
        "loss_first_50": float(np.mean(losses[:50])),
        "loss_last_50": float(np.mean(losses[-50:])),
        "heldout_branch_nll": branch_nll,
        "heldout_interior_nll_without_constant": interior_nll,
        "heldout_branch_accuracy": branch_accuracy,
        "training_records": int(len(features)),
    }


def sample_learned_kernel(
    model: TinyKernel,
    rng: np.random.Generator,
    context: float,
    parent_fraction: float,
) -> np.ndarray:
    remaining = 4.0 * parent_fraction
    allocation = np.zeros(4, dtype=np.float64)
    for position in range(3):
        lower, upper = feasible_interval(remaining, 1.0, 3.0 - position)
        width = upper - lower
        if width <= 1e-12:
            value = lower
        else:
            features = torch.from_numpy(
                feature_vector(context, parent_fraction, remaining, position)
            ).unsqueeze(0)
            with torch.no_grad():
                logits, mean, log_sigma = model(features)
            probabilities = torch.softmax(logits[0], dim=0).numpy()
            branch = int(rng.choice(3, p=probabilities))
            if branch == LOWER:
                value = lower
            elif branch == UPPER:
                value = upper
            else:
                z_value = float(rng.normal(float(mean[0]), math.exp(float(log_sigma[0]))))
                fraction = 1.0 / (1.0 + math.exp(-z_value))
                value = lower + width * fraction
        allocation[position] = value
        remaining -= value
    allocation[3] = remaining
    return allocation


def face_rates(samples: np.ndarray, atol: float = 1e-12) -> np.ndarray:
    zero = np.mean(np.isclose(samples, 0.0, atol=atol), axis=0)
    cap = np.mean(np.isclose(samples, 1.0, atol=atol), axis=0)
    return np.stack([zero, cap], axis=-1)


def energy_distance(x: np.ndarray, y: np.ndarray, limit: int = 600) -> float:
    x = x[:limit]
    y = y[:limit]
    xy = np.linalg.norm(x[:, None, :] - y[None, :, :], axis=-1).mean()
    xx = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1).mean()
    yy = np.linalg.norm(y[:, None, :] - y[None, :, :], axis=-1).mean()
    return float(2.0 * xy - xx - yy)


def distribution_audit(model: TinyKernel, seed: int, draws: int) -> tuple[dict, list[dict]]:
    scenarios = [
        (-0.8, 0.18),
        (-0.8, 0.52),
        (-0.8, 0.84),
        (0.0, 0.18),
        (0.0, 0.52),
        (0.0, 0.84),
        (0.8, 0.18),
        (0.8, 0.52),
        (0.8, 0.84),
    ]
    rows: list[dict] = []
    max_budget_error = 0.0
    max_bound_violation = 0.0
    for index, (context, parent_fraction) in enumerate(scenarios):
        truth_rng = np.random.default_rng(seed + 1000 + index)
        model_rng = np.random.default_rng(seed + 2000 + index)
        truth = np.stack(
            [sample_synthetic_kernel(truth_rng, context, parent_fraction)[0] for _ in range(draws)]
        )
        generated = np.stack(
            [sample_learned_kernel(model, model_rng, context, parent_fraction) for _ in range(draws)]
        )
        budget = 4.0 * parent_fraction
        max_budget_error = max(
            max_budget_error, float(np.max(np.abs(generated.sum(axis=1) - budget)))
        )
        max_bound_violation = max(
            max_bound_violation,
            float(np.max(np.maximum(-generated, 0.0))),
            float(np.max(np.maximum(generated - 1.0, 0.0))),
        )
        truth_cov = np.cov(truth, rowvar=False)
        generated_cov = np.cov(generated, rowvar=False)
        covariance_relative_error = float(
            np.linalg.norm(generated_cov - truth_cov) / max(np.linalg.norm(truth_cov), 1e-12)
        )
        rows.append(
            {
                "context": context,
                "parent_fraction": parent_fraction,
                "mean_mae": float(np.mean(np.abs(generated.mean(0) - truth.mean(0)))),
                "face_rate_mae": float(np.mean(np.abs(face_rates(generated) - face_rates(truth)))),
                "covariance_relative_error": covariance_relative_error,
                "energy_distance": energy_distance(truth, generated),
                "truth_mean": truth.mean(0).tolist(),
                "generated_mean": generated.mean(0).tolist(),
                "truth_face_rates": face_rates(truth).tolist(),
                "generated_face_rates": face_rates(generated).tolist(),
            }
        )
    summary = {
        "mean_mae": float(np.mean([row["mean_mae"] for row in rows])),
        "face_rate_mae": float(np.mean([row["face_rate_mae"] for row in rows])),
        "covariance_relative_error": float(
            np.mean([row["covariance_relative_error"] for row in rows])
        ),
        "energy_distance": float(np.mean([row["energy_distance"] for row in rows])),
        "max_budget_abs": max_budget_error,
        "max_bound_violation": max_bound_violation,
    }
    return summary, rows


def save_figure(rows: list[dict], destination: Path) -> None:
    labels = [f"c={r['context']:+.1f}\ns={r['parent_fraction']:.2f}" for r in rows]
    x = np.arange(len(rows))
    mean_error = [r["mean_mae"] for r in rows]
    face_error = [r["face_rate_mae"] for r in rows]
    covariance_error = [r["covariance_relative_error"] for r in rows]
    energy = [r["energy_distance"] for r in rows]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    for axis, values, title in zip(
        axes,
        [mean_error, face_error, covariance_error, energy],
        ["child mean MAE", "zero/cap rate MAE", "covariance relative error", "energy distance"],
    ):
        axis.bar(x, values, color="#2676b8")
        axis.set_title(title)
        axis.set_xticks(x, labels, rotation=50, ha="right")
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Face-aware conservative refinement: held-out synthetic conditionals")
    fig.tight_layout()
    fig.savefig(destination, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=73129)
    parser.add_argument("--roundtrip-cases", type=int, default=250)
    parser.add_argument("--train-samples", type=int, default=24000)
    parser.add_argument("--updates", type=int, default=900)
    parser.add_argument("--draws", type=int, default=3000)
    args = parser.parse_args()

    torch.set_num_threads(8)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    roundtrip = roundtrip_audit(args.seed, args.roundtrip_cases)
    two_level_roundtrip = two_level_roundtrip_audit(args.seed, args.roundtrip_cases)
    model, training = train_tiny_kernel(args.seed, args.train_samples, args.updates)
    distribution, scenarios = distribution_audit(model, args.seed, args.draws)

    gates = {
        "roundtrip_exact": roundtrip["max_roundtrip_abs"] < 1e-10,
        "two_level_roundtrip_exact": two_level_roundtrip["max_native_roundtrip_abs"] < 1e-10,
        "budget_exact": max(
            roundtrip["max_budget_abs"],
            two_level_roundtrip["max_root_budget_abs"],
            two_level_roundtrip["max_group_budget_abs"],
            distribution["max_budget_abs"],
        )
        < 1e-10,
        "bounds_exact": max(
            roundtrip["max_lower_violation"],
            roundtrip["max_upper_violation"],
            two_level_roundtrip["max_bound_violation"],
            distribution["max_bound_violation"],
        )
        < 1e-10,
        "learned_means": distribution["mean_mae"] < 0.025,
        "learned_faces": distribution["face_rate_mae"] < 0.035,
        "learned_covariance": distribution["covariance_relative_error"] < 0.20,
        "learned_energy": distribution["energy_distance"] < 0.004,
    }
    result = {
        "schema_version": 1,
        "status": "pass" if all(gates.values()) else "fail",
        "claim_boundary": (
            "Synthetic CPU algorithm admission only; parent budget is supplied and no sea-ice "
            "forecast, generated-parent robustness, or native-grid calibration claim is made."
        ),
        "separate_mask_generator": False,
        "kernel_law": (
            "autoregressive feasible-interval allocation with lower/interior/upper face atom "
            "inside each refinement step"
        ),
        "config": vars(args) | {"output_dir": str(args.output_dir)},
        "roundtrip": roundtrip,
        "two_level_roundtrip": two_level_roundtrip,
        "training": training,
        "distribution": distribution,
        "gates": gates,
        "scenarios": scenarios,
    }
    result_path = args.output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    torch.save({"model": model.state_dict(), "seed": args.seed}, args.output_dir / "tiny_kernel.pt")
    save_figure(scenarios, args.output_dir / "toy_comparison.png")
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "status",
                    "roundtrip",
                    "two_level_roundtrip",
                    "training",
                    "distribution",
                    "gates",
                )
            },
            indent=2,
        )
    )
    if result["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
