"""Bounded real-data CPU integration for the IDEA-F1 mask-only runner."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch

from .config import load_json
from .data import build_dataset
from .direct_dynamics_mixed_support_admission import MODE as ADMISSION_MODE
from .direct_dynamics_mixed_support_mask_runner import (
    CfmBatch,
    build_matched_models,
    coarse_mask_batch_from_item,
    d0_edge_neighbourhood,
    masked_cfm_loss,
    sample_masks,
)


MODE = "direct_dynamics_mixed_support_mask_cpu_integration_v1"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _gradient_norms(parameters) -> list[float]:
    values = []
    for parameter in parameters:
        if parameter.grad is None:
            values.append(0.0)
        else:
            values.append(float(parameter.grad.norm().item()))
    return values


def _synthetic_source_case(*, birth: bool, seed: int) -> dict[str, Any]:
    valid = torch.ones((1, 1, 16, 16))
    d0 = torch.zeros_like(valid) if birth else torch.ones_like(valid)
    target = torch.zeros((1, 3, 16, 16))
    if birth:
        target[..., 5:10, 6:12] = 1
    condition = torch.zeros((1, 15, 16, 16))
    condition[:, :1] = d0
    batch = CfmBatch(target=target, condition=condition, d0_occurrence=d0, valid=valid)
    model, _ = build_matched_models(seed=seed, hidden_channels=8)
    generator = torch.Generator().manual_seed(seed + 1)
    uniform = torch.rand(target.shape, generator=generator).clamp(1e-5, 1 - 1e-5)
    noise = torch.randn(target.shape, generator=generator)
    loss, per_case = masked_cfm_loss(
        model,
        batch,
        time=torch.tensor([0.37]),
        noise=noise,
        uniform=uniform,
    )
    loss.backward()
    source_norms = _gradient_norms(model.source_head.parameters())
    front_norms = _gradient_norms(model.front_head.parameters())
    return {
        "kind": "remote_birth_from_empty_d0" if birth else "death_from_full_d0",
        "initial_front_support_cells": int(d0_edge_neighbourhood(d0, valid).sum().item()),
        "target_changed_cells": int((target != d0).sum().item()),
        "loss": float(loss.detach().item()),
        "per_case_loss": [float(value) for value in per_case.detach()],
        "source_gradient_norms": source_norms,
        "source_gradients_finite_positive": all(value > 0 and torch.isfinite(torch.tensor(value)) for value in source_norms),
        "front_gradient_norms": front_norms,
        "front_gradients_exact_zero_without_front_support": all(value == 0 for value in front_norms),
    }


def _real_case_check(item: dict[str, Any], means: list[float], stds: list[float], *, seed: int) -> dict[str, Any]:
    batch = coarse_mask_batch_from_item(item, means, stds)
    candidate, control = build_matched_models(seed=seed, hidden_channels=8)
    parameter_count = sum(parameter.numel() for parameter in candidate.parameters())
    generator = torch.Generator().manual_seed(seed + 1)
    uniform = torch.rand(batch.target.shape, generator=generator).clamp(1e-5, 1 - 1e-5)
    noise = torch.randn(batch.target.shape, generator=generator)
    time = torch.tensor([0.43])
    candidate_loss, candidate_cases = masked_cfm_loss(
        candidate, batch, time=time, noise=noise, uniform=uniform
    )
    candidate_loss.backward()
    candidate_source_norms = _gradient_norms(candidate.source_head.parameters())
    control_loss, control_cases = masked_cfm_loss(
        control, batch, time=time, noise=noise, uniform=uniform
    )
    control_loss.backward()

    state = (1 - time[:, None, None, None]) * torch.where(
        batch.valid > 0,
        torch.logit(0.5 * (batch.target + uniform)),
        torch.zeros_like(batch.target),
    ) + time[:, None, None, None] * torch.where(batch.valid > 0, noise, torch.zeros_like(noise))
    reference = candidate(state, time, batch.condition, batch.d0_occurrence, batch.valid)
    land = batch.valid == 0
    altered_state = torch.where(land, torch.full_like(state, 1e6), state)
    altered_condition = torch.where(land, torch.full_like(batch.condition, -1e6), batch.condition)
    altered_d0 = torch.where(land, torch.ones_like(batch.d0_occurrence), batch.d0_occurrence)
    changed = candidate(altered_state, time, altered_condition, altered_d0, batch.valid)
    forward_land_invariant = torch.equal(reference, changed)
    sample_reference = sample_masks(
        candidate,
        condition=batch.condition,
        d0_occurrence=batch.d0_occurrence,
        valid=batch.valid,
        noise=noise,
    )
    sample_changed = sample_masks(
        candidate,
        condition=altered_condition,
        d0_occurrence=altered_d0,
        valid=batch.valid,
        noise=altered_state,
    )
    sampling_land_invariant = torch.equal(sample_reference, sample_changed)
    return {
        "case_id": item["meta"]["case_id"],
        "coarse_shape": list(batch.target.shape[-2:]),
        "parameter_count_each_arm": parameter_count,
        "initial_parameters_identical": all(
            torch.equal(left.detach(), right.detach())
            for left, right in zip(candidate.parameters(), control.parameters(), strict=True)
        ),
        "candidate_loss": float(candidate_loss.detach().item()),
        "control_loss": float(control_loss.detach().item()),
        "candidate_per_case_loss": [float(value) for value in candidate_cases.detach()],
        "control_per_case_loss": [float(value) for value in control_cases.detach()],
        "candidate_source_gradient_norms": candidate_source_norms,
        "candidate_source_gradients_finite_positive": all(
            value > 0 and torch.isfinite(torch.tensor(value)) for value in candidate_source_norms
        ),
        "forward_land_perturbation_invariant": bool(forward_land_invariant),
        "sampling_land_perturbation_invariant": bool(sampling_land_invariant),
        "decoded_land_exact_zero": bool(torch.all(sample_reference[land.expand_as(sample_reference)] == 0)),
    }


def run(config_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    config_path, output_path = Path(config_path), Path(output_path)
    config = load_json(config_path)
    if config.get("mode") != ADMISSION_MODE:
        raise ValueError("integration requires the admitted frozen IDEA-F1 config")
    if config.get("resource_kind") != "cpu" or config.get("dataset_split") != "train":
        raise ValueError("integration is CPU-only and train-only")
    if bool(config.get("test_2023_access_allowed", True)):
        raise ValueError("test-2023 must remain sealed")
    dataset_config = dict(load_json(config["dataset_config"]))
    dataset_config["dynamic_forcing_stats"] = config["dynamic_forcing_stats"]
    dataset = build_dataset(dataset_config, "train")
    indices = dataset.strided_case_indices(max_cases=1, stride_days=int(config["stride_days"]))
    if len(indices) != 1:
        raise RuntimeError("no train integration case")
    seed = int(config["seed"])
    real = _real_case_check(dataset[indices[0]], dataset_config["means"], dataset_config["stds"], seed=seed)
    birth = _synthetic_source_case(birth=True, seed=seed + 10)
    death = _synthetic_source_case(birth=False, seed=seed + 20)
    checks = (
        real["initial_parameters_identical"],
        real["candidate_source_gradients_finite_positive"],
        real["forward_land_perturbation_invariant"],
        real["sampling_land_perturbation_invariant"],
        real["decoded_land_exact_zero"],
        birth["initial_front_support_cells"] == 0,
        birth["source_gradients_finite_positive"],
        birth["front_gradients_exact_zero_without_front_support"],
        death["initial_front_support_cells"] == 0,
        death["source_gradients_finite_positive"],
        death["front_gradients_exact_zero_without_front_support"],
    )
    result = {
        "schema_version": MODE,
        "status": "complete",
        "resource_kind": "cpu",
        "dataset_split": "train",
        "test_2023_accessed": False,
        "seed_controls_model_initialization_noise_and_dequantization": True,
        "conditioning": "causal d0 plus forcing/calendar; no future fields",
        "loss": "continuous dequantized CFM, positive per-case denominator, no clamp and no STE",
        "land_contract": "zero before every forward and after every sampling step",
        "real_case": real,
        "synthetic_source_branch": {"birth": birth, "death": death},
        "gate": {
            "runner_cpu_integration_pass": bool(all(checks)),
            "gpu_training_authorized": False,
            "next_required_action": "Astra exact-code review, then zero-update GPU preflight only",
        },
        "claim_boundary": "runner integration only; untrained outputs make no skill or calibration claim",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("output")
    args = parser.parse_args()
    result = run(args.config, args.output)
    print(json.dumps({"status": result["status"], "gate": result["gate"]}, sort_keys=True))


if __name__ == "__main__":
    main()
