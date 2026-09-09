"""CPU mechanics pilot for a censored joint generator trained by energy score.

The toy has two conditioning states and four jointly generated coordinates:
``(SIC_a, SIT_a, SIC_b, SIT_b)``.  Its known target is a censored correlated
Gaussian.  This isolates whether an implicit joint generator can learn exact
boundary atoms without independent occurrence masks or per-member MSE.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn

from .clearml_tracking import ClearMLTracker
from .trainer import _atomic_json


CONDITION_NAMES = ("open_thin", "icy")
CHANNEL_NAMES = ("sic_a", "sit_a", "sic_b", "sit_b")
DIMENSION = len(CHANNEL_NAMES)
ENSEMBLE_MEMBERS = 4
DEFAULT_UPDATES = 2_000
DEFAULT_EVALUATION_SAMPLES = 100_000
DEFAULT_SEED = 1701
GATE_TOLERANCES = {
    "learned_normalized_mean_linf": 0.10,
    "learned_normalized_covariance_linf": 0.10,
    "learned_boundary_mass_linf": 0.05,
    "learned_joint_sit_zero_abs": 0.05,
    "learned_energy_score_gap_vs_target": 0.10,
    "independent_marginal_boundary_mass_linf": 0.02,
    "open_thin_independent_joint_sit_zero_abs_min": 0.03,
    "icy_independent_normalized_covariance_linf_min": 0.05,
    "icy_independent_energy_score_gap_min": 0.01,
}


def censor_physical(latent: torch.Tensor) -> torch.Tensor:
    """Apply the physical censoring map that defines the modeled law."""

    if latent.shape[-1] != DIMENSION or not torch.all(torch.isfinite(latent)):
        raise ValueError("latent values must be finite with four trailing channels")
    sic = latent[..., 0::2].clamp(0.0, 1.0)
    sit = latent[..., 1::2].clamp_min(0.0)
    return torch.stack((sic[..., 0], sit[..., 0], sic[..., 1], sit[..., 1]), dim=-1)


def target_parameters() -> tuple[torch.Tensor, torch.Tensor]:
    """Known conditional latent means and full Cholesky factors."""

    means = torch.tensor(
        [
            [-0.030, -0.020, 0.050, 0.015],
            [0.820, 0.650, 0.780, 0.550],
        ],
        dtype=torch.float64,
    )
    factors = torch.tensor(
        [
            [
                [0.120, 0.000, 0.000, 0.000],
                [0.060, 0.060, 0.000, 0.000],
                [0.080, 0.010, 0.080, 0.000],
                [0.040, 0.040, 0.030, 0.050],
            ],
            [
                [0.180, 0.000, 0.000, 0.000],
                [0.100, 0.120, 0.000, 0.000],
                [0.100, 0.030, 0.140, 0.000],
                [0.050, 0.080, 0.070, 0.100],
            ],
        ],
        dtype=torch.float64,
    )
    return means, factors


def sample_censored_gaussian(
    means: torch.Tensor,
    factors: torch.Tensor,
    condition: torch.Tensor,
    noise: torch.Tensor,
) -> torch.Tensor:
    if condition.ndim != 1 or noise.shape[:-1] != condition.shape:
        raise ValueError("condition and noise batch shapes differ")
    selected_mean = means[condition]
    selected_factor = factors[condition]
    latent = selected_mean + torch.einsum("bij,bj->bi", selected_factor, noise)
    return censor_physical(latent)


def unbiased_energy_score(
    members: torch.Tensor, truth: torch.Tensor, scales: torch.Tensor
) -> torch.Tensor:
    """Unbiased ensemble energy score using a fixed normalized L2 distance."""

    if members.ndim != 3 or truth.shape != members.shape[:1] + members.shape[2:]:
        raise ValueError("members/truth must have shapes [B,M,D] and [B,D]")
    if members.shape[1] < 2 or scales.shape != (members.shape[2],):
        raise ValueError("energy score requires M>=2 and one scale per channel")
    if not torch.all(torch.isfinite(members)) or not torch.all(torch.isfinite(truth)):
        raise FloatingPointError("energy score inputs contain NaN/Inf")
    if not torch.all(torch.isfinite(scales)) or not torch.all(scales > 0):
        raise ValueError("energy score scales must be finite and positive")
    normalized_members = members / scales
    normalized_truth = truth / scales
    first = torch.linalg.vector_norm(
        normalized_members - normalized_truth[:, None], dim=-1
    ).mean(dim=1)
    pairwise = torch.cdist(normalized_members, normalized_members)
    member_count = members.shape[1]
    second = pairwise.sum(dim=(1, 2)) / (2.0 * member_count * (member_count - 1))
    score = (first - second).mean()
    if not torch.isfinite(score):
        raise FloatingPointError("energy score is NaN/Inf")
    return score


class ConditionalAffineGenerator(nn.Module):
    """Two-condition affine joint latent generator with full covariance."""

    def __init__(self, *, seed: int) -> None:
        super().__init__()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        self.means = nn.Parameter(
            0.08 * torch.randn((2, DIMENSION), generator=generator, dtype=torch.float64)
        )
        raw = 0.03 * torch.randn(
            (2, DIMENSION, DIMENSION), generator=generator, dtype=torch.float64
        )
        diagonal = math.log(math.expm1(0.18))
        raw.diagonal(dim1=-2, dim2=-1).fill_(diagonal)
        self.raw_factors = nn.Parameter(raw)

    def factors(self) -> torch.Tensor:
        lower = torch.tril(self.raw_factors, diagonal=-1)
        diagonal = torch.nn.functional.softplus(
            self.raw_factors.diagonal(dim1=-2, dim2=-1)
        ) + 1e-4
        return lower + torch.diag_embed(diagonal)

    def forward(self, condition: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        if condition.ndim != 1 or noise.ndim != 3:
            raise ValueError("generator expects condition [B] and noise [B,M,D]")
        if noise.shape[0] != condition.shape[0] or noise.shape[2] != DIMENSION:
            raise ValueError("generator condition/noise shapes differ")
        means = self.means[condition]
        factors = self.factors()[condition]
        latent = means[:, None] + torch.einsum("bij,bmj->bmi", factors, noise)
        return censor_physical(latent)


def _balanced_conditions(batch_size: int) -> torch.Tensor:
    if batch_size <= 0 or batch_size % 2:
        raise ValueError("batch size must be a positive even number")
    return torch.arange(2, dtype=torch.long).repeat_interleave(batch_size // 2)


def _training_scales(*, seed: int, count_per_condition: int = 16_384) -> torch.Tensor:
    means, factors = target_parameters()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    condition = torch.arange(2, dtype=torch.long).repeat_interleave(count_per_condition)
    noise = torch.randn((condition.numel(), DIMENSION), generator=generator, dtype=torch.float64)
    samples = sample_censored_gaussian(means, factors, condition, noise)
    scales = samples.std(dim=0, unbiased=True)
    if not torch.all(torch.isfinite(scales)) or not torch.all(scales > 1e-3):
        raise ValueError("train-derived toy scales are degenerate")
    return scales


def _dead_censoring_probe(
    model: ConditionalAffineGenerator,
    scales: torch.Tensor,
    *, seed: int,
) -> dict[str, float]:
    target_means, target_factors = target_parameters()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    condition = _balanced_conditions(256)
    target_noise = torch.randn((256, DIMENSION), generator=generator, dtype=torch.float64)
    member_noise = torch.randn(
        (256, ENSEMBLE_MEMBERS, DIMENSION), generator=generator, dtype=torch.float64
    )
    truth = sample_censored_gaussian(
        target_means, target_factors, condition, target_noise
    )
    members = model(condition, member_noise)
    loss = unbiased_energy_score(members, truth, scales)
    gradients = torch.autograd.grad(
        loss, (model.means, model.raw_factors), retain_graph=False
    )
    result = {
        "initial_loss": float(loss),
        "mean_gradient_norm": float(torch.linalg.vector_norm(gradients[0])),
        "factor_gradient_norm": float(torch.linalg.vector_norm(gradients[1])),
        "strictly_positive_output_fraction": float((members > 0).double().mean()),
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise FloatingPointError("dead-censoring probe contains NaN/Inf")
    if result["mean_gradient_norm"] <= 0 or result["factor_gradient_norm"] <= 0:
        raise RuntimeError("censoring killed a required generator gradient")
    return result


def fit_censored_joint_toy(
    *,
    updates: int = DEFAULT_UPDATES,
    seed: int = DEFAULT_SEED,
    report_loss: Callable[[int, float], None] | None = None,
) -> tuple[ConditionalAffineGenerator, dict[str, Any]]:
    updates = int(updates)
    if not 1 <= updates <= DEFAULT_UPDATES:
        raise ValueError("toy update budget must be in [1,2000]")
    torch.manual_seed(int(seed))
    torch.set_num_threads(min(torch.get_num_threads(), 6))
    target_means, target_factors = target_parameters()
    scales = _training_scales(seed=seed + 11)
    model = ConditionalAffineGenerator(seed=seed + 17)
    dead_censoring = _dead_censoring_probe(model, scales, seed=seed + 23)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=updates, eta_min=0.001
    )
    target_generator = torch.Generator(device="cpu")
    target_generator.manual_seed(seed + 101)
    member_generator = torch.Generator(device="cpu")
    member_generator.manual_seed(seed + 202)
    condition = _balanced_conditions(256)
    losses = []
    for update in range(1, updates + 1):
        target_noise = torch.randn(
            (condition.numel(), DIMENSION),
            generator=target_generator,
            dtype=torch.float64,
        )
        member_noise = torch.randn(
            (condition.numel(), ENSEMBLE_MEMBERS, DIMENSION),
            generator=member_generator,
            dtype=torch.float64,
        )
        truth = sample_censored_gaussian(
            target_means, target_factors, condition, target_noise
        )
        members = model(condition, member_noise)
        loss = unbiased_energy_score(members, truth, scales)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradients = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        if not torch.isfinite(gradients):
            raise FloatingPointError("toy gradient norm is NaN/Inf")
        optimizer.step()
        scheduler.step()
        value = float(loss.detach())
        losses.append(value)
        if report_loss is not None and (update == 1 or update % 50 == 0):
            report_loss(update, value)
    if not all(math.isfinite(value) for value in losses):
        raise FloatingPointError("toy loss history contains NaN/Inf")
    training = {
        "updates": updates,
        "ensemble_members": ENSEMBLE_MEMBERS,
        "batch_size": int(condition.numel()),
        "learning_rate": 0.02,
        "minimum_learning_rate": 0.001,
        "objective": "unbiased_sample_energy_score",
        "per_member_mse": False,
        "straight_through_gradient": False,
        "train_derived_scales": scales.tolist(),
        "dead_censoring_probe": dead_censoring,
        "first_100_loss_mean": float(sum(losses[:100]) / min(100, len(losses))),
        "last_100_loss_mean": float(sum(losses[-100:]) / min(100, len(losses))),
    }
    return model, training


def _law_statistics(samples: torch.Tensor) -> dict[str, Any]:
    if samples.ndim != 2 or samples.shape[1] != DIMENSION:
        raise ValueError("law statistics expect [N,4]")
    centered = samples - samples.mean(dim=0)
    covariance = centered.T @ centered / (samples.shape[0] - 1)
    return {
        "mean": samples.mean(dim=0).tolist(),
        "covariance": covariance.tolist(),
        "boundary_mass": {
            "sic_a_zero": float((samples[:, 0] == 0).double().mean()),
            "sic_a_one": float((samples[:, 0] == 1).double().mean()),
            "sit_a_zero": float((samples[:, 1] == 0).double().mean()),
            "sic_b_zero": float((samples[:, 2] == 0).double().mean()),
            "sic_b_one": float((samples[:, 2] == 1).double().mean()),
            "sit_b_zero": float((samples[:, 3] == 0).double().mean()),
            "joint_sit_zero": float(
                ((samples[:, 1] == 0) & (samples[:, 3] == 0)).double().mean()
            ),
        },
    }


def _linf(left: Any, right: Any) -> float:
    left_tensor = torch.as_tensor(left, dtype=torch.float64)
    right_tensor = torch.as_tensor(right, dtype=torch.float64)
    return float((left_tensor - right_tensor).abs().max())


def _normalized_mean_linf(left: Any, right: Any, scales: torch.Tensor) -> float:
    return _linf(
        torch.as_tensor(left, dtype=torch.float64) / scales,
        torch.as_tensor(right, dtype=torch.float64) / scales,
    )


def _normalized_covariance_linf(
    left: Any, right: Any, scales: torch.Tensor
) -> float:
    normalization = scales[:, None] * scales[None, :]
    return _linf(
        torch.as_tensor(left, dtype=torch.float64) / normalization,
        torch.as_tensor(right, dtype=torch.float64) / normalization,
    )


@torch.no_grad()
def evaluate_censored_joint_toy(
    model: ConditionalAffineGenerator,
    training: dict[str, Any],
    *,
    sample_count: int = DEFAULT_EVALUATION_SAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    sample_count = int(sample_count)
    if sample_count < 1_000:
        raise ValueError("toy evaluation requires at least 1000 samples per condition")
    target_means, target_factors = target_parameters()
    independent_factors = torch.diag_embed(
        torch.linalg.vector_norm(target_factors, dim=-1)
    )
    scales = torch.tensor(training["train_derived_scales"], dtype=torch.float64)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 303)
    result: dict[str, Any] = {"sample_count_per_condition": sample_count, "conditions": {}}
    for condition_index, condition_name in enumerate(CONDITION_NAMES):
        condition = torch.full((sample_count,), condition_index, dtype=torch.long)
        target_noise = torch.randn(
            (sample_count, DIMENSION), generator=generator, dtype=torch.float64
        )
        learned_noise = torch.randn(
            (sample_count, DIMENSION), generator=generator, dtype=torch.float64
        )
        independent_noise = torch.randn(
            (sample_count, DIMENSION), generator=generator, dtype=torch.float64
        )
        target = sample_censored_gaussian(
            target_means, target_factors, condition, target_noise
        )
        learned = model(condition, learned_noise[:, None]).squeeze(1)
        independent = sample_censored_gaussian(
            target_means, independent_factors, condition, independent_noise
        )
        target_stats = _law_statistics(target)
        learned_stats = _law_statistics(learned)
        independent_stats = _law_statistics(independent)

        score_members = 8
        learned_score_noise = torch.randn(
            (sample_count, score_members, DIMENSION),
            generator=generator,
            dtype=torch.float64,
        )
        target_score_noise = torch.randn(
            (sample_count, score_members, DIMENSION),
            generator=generator,
            dtype=torch.float64,
        )
        independent_score_noise = torch.randn(
            (sample_count, score_members, DIMENSION),
            generator=generator,
            dtype=torch.float64,
        )
        learned_members = model(condition, learned_score_noise)
        target_members = censor_physical(
            target_means[condition_index][None, None]
            + torch.einsum(
                "ij,bmj->bmi", target_factors[condition_index], target_score_noise
            )
        )
        independent_members = censor_physical(
            target_means[condition_index][None, None]
            + torch.einsum(
                "ij,bmj->bmi",
                independent_factors[condition_index],
                independent_score_noise,
            )
        )
        target_boundary = target_stats["boundary_mass"]
        learned_boundary = learned_stats["boundary_mass"]
        independent_boundary = independent_stats["boundary_mass"]
        result["conditions"][condition_name] = {
            "target": target_stats,
            "learned": learned_stats,
            "independent_marginal_baseline": independent_stats,
            "errors": {
                "learned_normalized_mean_linf": _normalized_mean_linf(
                    learned_stats["mean"], target_stats["mean"], scales
                ),
                "learned_normalized_covariance_linf": _normalized_covariance_linf(
                    learned_stats["covariance"], target_stats["covariance"], scales
                ),
                "learned_boundary_mass_linf": max(
                    abs(learned_boundary[key] - target_boundary[key])
                    for key in target_boundary
                ),
                "learned_joint_sit_zero_abs": abs(
                    learned_boundary["joint_sit_zero"]
                    - target_boundary["joint_sit_zero"]
                ),
                "independent_marginal_boundary_mass_linf": max(
                    abs(independent_boundary[key] - target_boundary[key])
                    for key in target_boundary
                    if key != "joint_sit_zero"
                ),
                "independent_joint_sit_zero_abs": abs(
                    independent_boundary["joint_sit_zero"]
                    - target_boundary["joint_sit_zero"]
                ),
                "independent_normalized_covariance_linf": _normalized_covariance_linf(
                    independent_stats["covariance"], target_stats["covariance"], scales
                ),
            },
            "energy_score": {
                "target_oracle": float(unbiased_energy_score(target_members, target, scales)),
                "learned": float(unbiased_energy_score(learned_members, target, scales)),
                "independent_marginal_baseline": float(
                    unbiased_energy_score(independent_members, target, scales)
                ),
            },
        }
    condition_decisions = {}
    for condition_name, condition_result in result["conditions"].items():
        errors = condition_result["errors"]
        scores = condition_result["energy_score"]
        criteria = {
            "normalized_mean": errors["learned_normalized_mean_linf"]
            < GATE_TOLERANCES["learned_normalized_mean_linf"],
            "normalized_covariance": errors["learned_normalized_covariance_linf"]
            < GATE_TOLERANCES["learned_normalized_covariance_linf"],
            "boundary_mass": errors["learned_boundary_mass_linf"]
            < GATE_TOLERANCES["learned_boundary_mass_linf"],
            "joint_sit_zero": errors["learned_joint_sit_zero_abs"]
            < GATE_TOLERANCES["learned_joint_sit_zero_abs"],
            "energy_score_gap": scores["learned"] - scores["target_oracle"]
            < GATE_TOLERANCES["learned_energy_score_gap_vs_target"],
            "energy_score_beats_independent": scores["learned"]
            < scores["independent_marginal_baseline"],
        }
        condition_decisions[condition_name] = {
            "criteria": criteria,
            "passed": all(criteria.values()),
        }
    open_errors = result["conditions"]["open_thin"]["errors"]
    icy_errors = result["conditions"]["icy"]["errors"]
    negative_control = {
        "marginals_match": all(
            condition["errors"]["independent_marginal_boundary_mass_linf"]
            < GATE_TOLERANCES["independent_marginal_boundary_mass_linf"]
            for condition in result["conditions"].values()
        ),
        "open_thin_joint_zero_fails": open_errors[
            "independent_joint_sit_zero_abs"
        ]
        > GATE_TOLERANCES["open_thin_independent_joint_sit_zero_abs_min"],
        "icy_covariance_fails": icy_errors[
            "independent_normalized_covariance_linf"
        ]
        > GATE_TOLERANCES["icy_independent_normalized_covariance_linf_min"],
        "icy_energy_score_fails": (
            result["conditions"]["icy"]["energy_score"][
                "independent_marginal_baseline"
            ]
            - result["conditions"]["icy"]["energy_score"]["target_oracle"]
        )
        > GATE_TOLERANCES["icy_independent_energy_score_gap_min"],
    }
    passed = all(
        decision["passed"] for decision in condition_decisions.values()
    ) and all(negative_control.values())
    result["gate"] = {
        "tolerances": GATE_TOLERANCES,
        "conditions": condition_decisions,
        "negative_control": negative_control,
        "passed": passed,
        "permits_real_data_mechanics_review": passed,
        "permits_substantive_training": False,
    }
    return result


def _require_finite_scalars(value: Any, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _require_finite_scalars(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _require_finite_scalars(child, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise FloatingPointError(f"{path} is NaN/Inf")


@torch.no_grad()
def _make_figure(
    result: dict[str, Any], model: ConditionalAffineGenerator, *, seed: int
):
    import matplotlib.pyplot as plt

    target_means, target_factors = target_parameters()
    independent_factors = torch.diag_embed(
        torch.linalg.vector_norm(target_factors, dim=-1)
    )
    generator = torch.Generator(device="cpu").manual_seed(seed + 404)
    figure, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    for row, condition_name in enumerate(CONDITION_NAMES):
        condition = torch.full((3000,), row, dtype=torch.long)
        noises = [
            torch.randn((3000, DIMENSION), generator=generator, dtype=torch.float64)
            for _ in range(3)
        ]
        samples = (
            sample_censored_gaussian(target_means, target_factors, condition, noises[0]),
            model(condition, noises[1][:, None]).squeeze(1),
            sample_censored_gaussian(
                target_means, independent_factors, condition, noises[2]
            ),
        )
        for column, (title, values) in enumerate(
            zip(("target", "learned joint", "independent marginals"), samples, strict=True)
        ):
            axes[row, column].scatter(
                values[:, 1].detach().cpu().numpy(),
                values[:, 3].detach().cpu().numpy(),
                s=3,
                alpha=0.2,
                rasterized=True,
            )
            axes[row, column].set_title(f"{condition_name}: {title}")
            axes[row, column].set_xlabel("SIT a")
            axes[row, column].set_ylabel("SIT b")
            axes[row, column].grid(alpha=0.2)
    figure.suptitle("Censored joint energy-score toy: dependence and zero atoms")
    return figure


def run(output_dir: Path) -> dict[str, Any]:
    if torch.cuda.device_count() != 0:
        raise RuntimeError("censored joint energy toy must have zero visible GPUs")
    if os.environ.get("CLEARML_REQUIRE_ONLINE") != "1":
        raise RuntimeError("censored joint energy toy requires online ClearML")
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"refusing to reuse toy output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    tracker = ClearMLTracker(
        "sea_ice_two_stage",
        f"censored_joint_energy_toy_{output_dir.name}",
        tags=[
            "cpu-only",
            "censored-joint-law",
            "energy-score",
            "mechanics-toy",
            "zero-gpu",
            "no-per-member-mse",
        ],
    )
    contract = {
        "conditions": CONDITION_NAMES,
        "channels": CHANNEL_NAMES,
        "updates": DEFAULT_UPDATES,
        "evaluation_samples_per_condition": DEFAULT_EVALUATION_SAMPLES,
        "ensemble_members_training": ENSEMBLE_MEMBERS,
        "seed": DEFAULT_SEED,
        "target": "known_censored_correlated_gaussian",
        "objective": "unbiased_sample_energy_score",
        "independent_pixel_occurrence": False,
        "future_support_mask": False,
        "per_member_mse": False,
        "gpu_devices": 0,
        "gate_tolerances": GATE_TOLERANCES,
    }
    tracker.connect("censored_joint_energy_toy_contract", contract)
    try:
        model, training = fit_censored_joint_toy(
            updates=DEFAULT_UPDATES,
            seed=DEFAULT_SEED,
            report_loss=lambda update, value: tracker.report_scalar(
                "censored_joint_energy_toy/train", "energy_score", value, update
            ),
        )
        evaluation = evaluate_censored_joint_toy(
            model,
            training,
            sample_count=DEFAULT_EVALUATION_SAMPLES,
            seed=DEFAULT_SEED,
        )
        result = {
            "status": "complete",
            "clearml_task_id": str(tracker.task.id),
            "contract": contract,
            "training": training,
            "evaluation": evaluation,
        }
        _require_finite_scalars(result)
        _atomic_json(output_dir / "result.json", result)
        torch.save(model.state_dict(), output_dir / "model.pt")
        figure = _make_figure(result, model, seed=DEFAULT_SEED)
        image_path = output_dir / "sit_joint_scatter.png"
        figure.savefig(image_path, dpi=180)
        import matplotlib.pyplot as plt

        plt.close(figure)
        tracker.report_image(
            "censored_joint_energy_toy/samples", "sit_dependence", image_path, 0
        )
        tracker.connect("censored_joint_energy_toy_result", result)
        tracker.upload_artifact("censored_joint_energy_toy_result", output_dir / "result.json")
        return result
    finally:
        tracker.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.output.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
