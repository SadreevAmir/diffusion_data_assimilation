"""FP64 oracle distinguishing posterior means from atom-preserving joint draws.

This is a CPU-only mathematical gate.  It does not train a network and it does
not claim that the toy finite law is the final sea-ice model.  Its purpose is
to test the endpoint operation required by any learned law that must preserve
exact open-water atoms and dependence between fields/locations/horizons.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch


EPSILONS = (0.01, 0.02)
DEFAULT_SAMPLE_COUNT = 100_000
DEFAULT_SEED = 1701


def _toy_joint_law() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return physical atoms, weights, means and scales for two linked sites/leads.

    Channel order is ``(SIC_a, SIT_a, SIC_b, SIT_b)``.  The five atoms encode
    open/open, thin/thin, thick/thick, thin/thick and open/thin states.  Their
    non-product weights make the test genuinely joint rather than two
    independent pixelwise Bernoulli problems.
    """

    atoms = torch.tensor(
        [
            [0.000, 0.000, 0.000, 0.000],
            [0.020, 0.008, 0.025, 0.010],
            [0.950, 1.500, 0.900, 1.200],
            [0.030, 0.012, 0.850, 1.100],
            [0.000, 0.000, 0.018, 0.007],
        ],
        dtype=torch.float64,
    )
    weights = torch.tensor([0.30, 0.20, 0.25, 0.15, 0.10], dtype=torch.float64)
    means = torch.tensor(
        [0.19301218262209952, 0.18969311571248842] * 2, dtype=torch.float64
    )
    scales = torch.tensor(
        [0.36890927421384667, 0.4359687842884708] * 2, dtype=torch.float64
    )
    return atoms, weights, means, scales


def _posterior_probabilities(
    state: torch.Tensor,
    normalized_atoms: torch.Tensor,
    weights: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    residual = state[:, None] - (1.0 - epsilon) * normalized_atoms[None]
    logits = -0.5 * residual.square().sum(dim=-1) / (epsilon * epsilon)
    logits = logits + torch.log(weights)[None]
    probabilities = torch.softmax(logits, dim=1)
    if not torch.all(torch.isfinite(probabilities)):
        raise FloatingPointError("joint posterior probabilities contain NaN/Inf")
    return probabilities


def _covariance(value: torch.Tensor) -> torch.Tensor:
    centered = value - value.mean(dim=0)
    return centered.T @ centered / (value.shape[0] - 1)


def _energy_score(
    probabilities: torch.Tensor, atoms: torch.Tensor, truth: torch.Tensor
) -> torch.Tensor:
    observation_distance = torch.linalg.vector_norm(
        atoms[None] - truth[:, None], dim=-1
    )
    atom_distance = torch.cdist(atoms, atoms)
    first = (probabilities * observation_distance).sum(dim=1)
    second = 0.5 * torch.einsum(
        "bi,ij,bj->b", probabilities, atom_distance, probabilities
    )
    return first - second


def _exact_atom_mass(samples: torch.Tensor, atoms: torch.Tensor) -> float:
    exact = (samples[:, None] == atoms[None]).all(dim=-1).any(dim=1)
    return float(exact.double().mean())


def _state_frequencies(indices: torch.Tensor, count: int) -> torch.Tensor:
    return torch.bincount(indices, minlength=count).to(torch.float64) / indices.numel()


def _finite_scalars(value: Any, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _finite_scalars(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _finite_scalars(child, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise FloatingPointError(f"{path} is NaN/Inf")


@torch.no_grad()
def joint_atom_law_oracle_report(
    *, sample_count: int = DEFAULT_SAMPLE_COUNT, seed: int = DEFAULT_SEED
) -> dict[str, Any]:
    """Compare a posterior-mean endpoint with an exact joint posterior draw."""

    sample_count = int(sample_count)
    if sample_count < 1_000:
        raise ValueError("joint atom oracle requires at least 1000 samples")
    atoms, weights, means, scales = _toy_joint_law()
    normalized_atoms = (atoms - means) / scales
    target_mean = weights @ atoms
    target_centered = atoms - target_mean
    target_covariance = (target_centered.T * weights) @ target_centered
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    truth_indices = torch.multinomial(
        weights, sample_count, replacement=True, generator=generator
    )
    truth = atoms[truth_indices]
    normalized_truth = normalized_atoms[truth_indices]
    noise = torch.randn(
        normalized_truth.shape, dtype=torch.float64, generator=generator
    )
    result: dict[str, Any] = {
        "status": "complete",
        "purpose": "cpu_only_atom_bearing_endpoint_law_gate",
        "sample_count": sample_count,
        "seed": int(seed),
        "channel_order": ["sic_a", "sit_a", "sic_b", "sit_b"],
        "atoms": atoms.tolist(),
        "target_weights": weights.tolist(),
        "target_zero_mass": {
            "sit_a": float(weights[atoms[:, 1] == 0].sum()),
            "sit_b": float(weights[atoms[:, 3] == 0].sum()),
        },
        "target_mean": target_mean.tolist(),
        "target_covariance": target_covariance.tolist(),
        "epsilons": {},
        "pure_zero_delta": {},
        "fixed_ambiguous_state_probe": {},
    }
    target_frequencies = _state_frequencies(truth_indices, atoms.shape[0])
    pair_open = (atoms[:, 1] == 0) & (atoms[:, 3] == 0)
    one_hot_truth = torch.nn.functional.one_hot(
        truth_indices, num_classes=atoms.shape[0]
    ).to(torch.float64)

    for epsilon in EPSILONS:
        state = (1.0 - epsilon) * normalized_truth + epsilon * noise
        posterior = _posterior_probabilities(
            state, normalized_atoms, weights, epsilon
        )
        posterior_mean = posterior @ atoms
        posterior_draw_indices = torch.multinomial(
            posterior, 1, replacement=True, generator=generator
        ).squeeze(1)
        posterior_draw = atoms[posterior_draw_indices]
        draw_frequencies = _state_frequencies(
            posterior_draw_indices, atoms.shape[0]
        )
        mean_covariance = _covariance(posterior_mean)
        draw_covariance = _covariance(posterior_draw)
        brier = (posterior - one_hot_truth).square().sum(dim=1)
        log_score = -torch.log(
            posterior.gather(1, truth_indices[:, None]).squeeze(1).clamp_min(1e-300)
        )
        mixture_energy = _energy_score(posterior, atoms, truth)
        degenerate_mean_energy = torch.linalg.vector_norm(
            posterior_mean - truth, dim=1
        )
        open_probability = posterior[:, pair_open].sum(dim=1)
        open_truth = pair_open[truth_indices].to(torch.float64)
        posterior_open_brier = (open_probability - open_truth).square().mean()
        deterministic_open = (
            (posterior_mean[:, 1] == 0) & (posterior_mean[:, 3] == 0)
        ).to(torch.float64)
        deterministic_open_brier = (deterministic_open - open_truth).square().mean()
        epsilon_result = {
            "posterior_draw": {
                "exact_joint_atom_mass": _exact_atom_mass(posterior_draw, atoms),
                "state_frequencies": draw_frequencies.tolist(),
                "frequency_linf_error_vs_target_law": float(
                    (draw_frequencies - weights).abs().max()
                ),
                "frequency_linf_error_vs_realized_truth": float(
                    (draw_frequencies - target_frequencies).abs().max()
                ),
                "mean_linf_error": float(
                    (posterior_draw.mean(dim=0) - target_mean).abs().max()
                ),
                "covariance_linf_error": float(
                    (draw_covariance - target_covariance).abs().max()
                ),
                "zero_mass_sit_a": float((posterior_draw[:, 1] == 0).double().mean()),
                "zero_mass_sit_b": float((posterior_draw[:, 3] == 0).double().mean()),
            },
            "posterior_mean": {
                # This diagnostic is reported but deliberately not used as an
                # atom-preservation gate: FP64 posterior weights for remote
                # atoms can underflow and make an otherwise interior mean
                # exactly equal to a positive atom numerically.
                "exact_joint_atom_mass": _exact_atom_mass(posterior_mean, atoms),
                "mean_linf_error": float(
                    (posterior_mean.mean(dim=0) - target_mean).abs().max()
                ),
                "covariance_linf_error": float(
                    (mean_covariance - target_covariance).abs().max()
                ),
                "zero_mass_sit_a": float((posterior_mean[:, 1] == 0).double().mean()),
                "zero_mass_sit_b": float((posterior_mean[:, 3] == 0).double().mean()),
                "mean_nearest_atom_distance": float(
                    torch.cdist(posterior_mean, atoms).min(dim=1).values.mean()
                ),
            },
            "proper_scores": {
                "categorical_brier_posterior": float(brier.mean()),
                "categorical_log_score_posterior": float(log_score.mean()),
                "joint_energy_score_posterior_law": float(mixture_energy.mean()),
                "joint_energy_score_degenerate_posterior_mean": float(
                    degenerate_mean_energy.mean()
                ),
                "joint_open_brier_posterior_law": float(posterior_open_brier),
                "joint_open_brier_degenerate_posterior_mean": float(
                    deterministic_open_brier
                ),
            },
        }
        result["epsilons"][str(epsilon)] = epsilon_result

    zero_atoms = torch.zeros((1, atoms.shape[1]), dtype=torch.float64)
    zero_weights = torch.ones(1, dtype=torch.float64)
    normalized_zero_atoms = (zero_atoms - means) / scales
    zero_clean = normalized_zero_atoms.expand(sample_count, -1)
    for epsilon in EPSILONS:
        # A delta law has no ambiguity: both its posterior mean and posterior
        # draw must remain exactly zero.  This guards against manufacturing a
        # positive floor inside the endpoint implementation itself.
        zero_state = (1.0 - epsilon) * zero_clean + epsilon * noise
        zero_posterior = _posterior_probabilities(
            zero_state, normalized_zero_atoms, zero_weights, epsilon
        )
        zero_mean = zero_posterior @ zero_atoms
        zero_draw_indices = torch.multinomial(
            zero_posterior, 1, replacement=True, generator=generator
        ).squeeze(1)
        zero_draw = zero_atoms[zero_draw_indices]
        result["pure_zero_delta"][str(epsilon)] = {
            "posterior_mean_exact_zero_mass": float(
                (zero_mean == 0).all(dim=1).double().mean()
            ),
            "posterior_draw_exact_zero_mass": float(
                (zero_draw == 0).all(dim=1).double().mean()
            ),
            "posterior_probability": float(zero_posterior[:, 0].mean()),
        }

    probe_epsilon = 0.02
    probe_state_value = (
        (1.0 - probe_epsilon) * 0.5 * (normalized_atoms[0] + normalized_atoms[1])
    )
    probe_state = probe_state_value.expand(sample_count, -1)
    probe_posterior = _posterior_probabilities(
        probe_state, normalized_atoms, weights, probe_epsilon
    )
    probe_draw_indices = torch.multinomial(
        probe_posterior, 1, replacement=True, generator=generator
    ).squeeze(1)
    probe_draw_frequencies = _state_frequencies(
        probe_draw_indices, atoms.shape[0]
    )
    result["fixed_ambiguous_state_probe"] = {
        "epsilon": probe_epsilon,
        "normalized_state": probe_state_value.tolist(),
        "posterior_probabilities": probe_posterior[0].tolist(),
        "conditional_draw_frequencies": probe_draw_frequencies.tolist(),
        "draw_frequency_linf_error_vs_posterior": float(
            (probe_draw_frequencies - probe_posterior[0]).abs().max()
        ),
        "posterior_linf_difference_from_prior": float(
            (probe_posterior[0] - weights).abs().max()
        ),
    }

    result["gate"] = {
        "posterior_draw_preserves_atoms": all(
            result["epsilons"][str(epsilon)]["posterior_draw"][
                "exact_joint_atom_mass"
            ]
            == 1.0
            for epsilon in EPSILONS
        ),
        "posterior_mean_erases_zero_atoms": all(
            result["epsilons"][str(epsilon)]["posterior_mean"]["zero_mass_sit_a"]
            == 0.0
            and result["epsilons"][str(epsilon)]["posterior_mean"][
                "zero_mass_sit_b"
            ]
            == 0.0
            for epsilon in EPSILONS
        ),
        "joint_draw_matches_target_law": all(
            result["epsilons"][str(epsilon)]["posterior_draw"][
                "frequency_linf_error_vs_target_law"
            ]
            < 0.01
            and result["epsilons"][str(epsilon)]["posterior_draw"][
                "covariance_linf_error"
            ]
            < 0.03
            for epsilon in EPSILONS
        ),
        "proper_scores_favor_joint_posterior_law": all(
            result["epsilons"][str(epsilon)]["proper_scores"][
                "joint_energy_score_posterior_law"
            ]
            < result["epsilons"][str(epsilon)]["proper_scores"][
                "joint_energy_score_degenerate_posterior_mean"
            ]
            and result["epsilons"][str(epsilon)]["proper_scores"][
                "joint_open_brier_posterior_law"
            ]
            < result["epsilons"][str(epsilon)]["proper_scores"][
                "joint_open_brier_degenerate_posterior_mean"
            ]
            for epsilon in EPSILONS
        ),
        "pure_zero_delta_preserved": all(
            result["pure_zero_delta"][str(epsilon)][
                "posterior_draw_exact_zero_mass"
            ]
            == 1.0
            for epsilon in EPSILONS
        ),
        "fixed_state_conditional_draw_matches_posterior": (
            result["fixed_ambiguous_state_probe"][
                "draw_frequency_linf_error_vs_posterior"
            ]
            < 0.01
            and result["fixed_ambiguous_state_probe"][
                "posterior_linf_difference_from_prior"
            ]
            > 0.1
        ),
        "interpretation": (
            "For this finite-mixture comparator, a categorical joint posterior "
            "draw preserves zero atoms and the joint law, whereas its posterior "
            "mean erases mixed-law zeros. A deterministic non-injective censored "
            "transport remains a separate atom-bearing candidate."
        ),
        "permits_gpu_training": False,
    }
    _finite_scalars(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = joint_atom_law_oracle_report(
        sample_count=arguments.samples, seed=arguments.seed
    )
    payload = json.dumps(report, indent=2, sort_keys=True)
    if arguments.output is not None:
        if arguments.output.exists() or arguments.output.is_symlink():
            raise FileExistsError(f"refusing to overwrite oracle report: {arguments.output}")
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
