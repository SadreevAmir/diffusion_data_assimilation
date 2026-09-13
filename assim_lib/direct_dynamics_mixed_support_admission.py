"""Bounded CPU admission for a mixed-support sea-ice occurrence law.

This is a representation test, not a calibration experiment.  Binary ice
occurrence is continuously dequantized for CFM and recovered by a fixed zero
threshold.  No straight-through estimator or future mask conditioning is used.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .config import load_json
from .data import build_dataset
from .direct_dynamics_cascade import masked_block_average
from .structured_joint_state import (
    decode_structured_joint_trajectory,
    encode_structured_joint_trajectory,
    validate_structured_state_stats,
)
from .transforms import channel_denormalize


MODE = "direct_dynamics_mixed_support_cpu_admission_v1"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def binary_dequantized_logit(binary: torch.Tensor, uniform: torch.Tensor) -> torch.Tensor:
    """Encode a Bernoulli field into disjoint continuous half-lines."""
    if binary.shape != uniform.shape or not binary.is_floating_point():
        raise ValueError("binary and uniform must be same-shape floating tensors")
    if not torch.all((binary == 0) | (binary == 1)):
        raise ValueError("binary target must contain only zero and one")
    if not torch.all(torch.isfinite(uniform)) or not torch.all((uniform > 0) & (uniform < 1)):
        raise ValueError("uniform filler must lie strictly inside (0,1)")
    return torch.logit(0.5 * (binary + uniform))


def decode_binary_dequantized_logit(latent: torch.Tensor) -> torch.Tensor:
    if not torch.all(torch.isfinite(latent)):
        raise ValueError("binary latent must be finite")
    return (latent >= 0).to(latent.dtype)


def coarse_occurrence(sic: torch.Tensor, valid: torch.Tensor, factor: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return B^coarse = 1{D_omega SIC > 0} and its active support."""
    if torch.any(sic[valid.expand_as(sic) > 0] < 0):
        raise ValueError("SIC must be non-negative on valid ocean")
    coarse_sic, ocean_fraction = masked_block_average(sic, valid, factor)
    support = ocean_fraction > 0
    return ((coarse_sic > 0) & support).to(sic.dtype), support


def occurrence_change_counts(
    initial_sic: torch.Tensor,
    future_sic: torch.Tensor,
    valid: torch.Tensor,
    factor: int,
) -> dict[str, int]:
    initial_native = (initial_sic > 0) & (valid > 0)
    future_native = (future_sic > 0) & (valid > 0)
    native_change = initial_native ^ future_native
    initial_coarse, support = coarse_occurrence(initial_sic, valid, factor)
    future_coarse, _ = coarse_occurrence(future_sic, valid, factor)
    coarse_change = (initial_coarse != future_coarse) & support
    visible_native = coarse_change.repeat_interleave(factor, -2).repeat_interleave(factor, -1)
    visible_native &= valid > 0
    return {
        "native_changed_cells": int(native_change.sum().item()),
        "native_changed_cells_visible_at_coarse": int((native_change & visible_native).sum().item()),
        "coarse_changed_cells": int(coarse_change.sum().item()),
        "coarse_birth_cells": int(((initial_coarse == 0) & (future_coarse == 1) & support).sum().item()),
        "coarse_death_cells": int(((initial_coarse == 1) & (future_coarse == 0) & support).sum().item()),
    }


def _masked_structured_decode(latent: torch.Tensor, valid: torch.Tensor, stats: dict[str, Any]) -> torch.Tensor:
    latent_valid = valid.repeat(1, latent.shape[1], 1, 1) > 0
    safe = torch.where(latent_valid, latent, torch.zeros_like(latent))
    physical = decode_structured_joint_trajectory(safe, stats, physical_dtype=torch.float64)
    physical_valid = valid.repeat(1, physical.shape[1], 1, 1) > 0
    return torch.where(physical_valid, physical, torch.zeros_like(physical))


def structured_roundtrip(
    physical: torch.Tensor,
    valid: torch.Tensor,
    stats: dict[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    latent = encode_structured_joint_trajectory(physical, valid, stats, generator=generator)
    decoded = _masked_structured_decode(latent, valid, stats)
    target = physical.to(torch.float64)
    mask = valid.repeat(1, physical.shape[1], 1, 1) > 0
    sic_mask = mask[:, 0::2]
    sit_mask = mask[:, 1::2]
    cap = float(stats["sic_cap"])
    tolerance = 1e-7
    target_sic, decoded_sic = target[:, 0::2], decoded[:, 0::2]
    target_sit, decoded_sit = target[:, 1::2], decoded[:, 1::2]
    occurrence_exact = torch.equal((target_sic > 0)[sic_mask], (decoded_sic > 0)[sic_mask])
    cap_exact = torch.equal(
        (torch.abs(target_sic - cap) <= tolerance)[sic_mask],
        (torch.abs(decoded_sic - cap) <= tolerance)[sic_mask],
    )
    max_sic_error = float(torch.abs(decoded_sic[sic_mask] - target_sic[sic_mask]).max().item())
    max_sit_error = float(torch.abs(decoded_sit[sit_mask] - target_sit[sit_mask]).max().item())

    altered = latent.clone()
    land = ~(valid.repeat(1, latent.shape[1], 1, 1) > 0)
    altered[land] = 3.0
    altered_decoded = _masked_structured_decode(altered, valid, stats)
    land_invariant = torch.equal(decoded, altered_decoded)
    land_zero = bool(torch.all(decoded[~mask] == 0))
    return {
        "occurrence_exact": bool(occurrence_exact),
        "cap_exact": bool(cap_exact),
        "max_abs_sic_error": max_sic_error,
        "max_abs_sit_error": max_sit_error,
        "land_invariant": bool(land_invariant),
        "decoded_land_exact_zero": land_zero,
    }


def dequantized_cfm_gradient_check(binary: torch.Tensor, valid: torch.Tensor, *, seed: int) -> dict[str, Any]:
    """Verify a real pathwise CFM gradient before any hard decoding."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    epsilon = torch.finfo(binary.dtype).eps
    uniform = torch.rand(binary.shape, generator=generator, dtype=binary.dtype).clamp(epsilon, 1 - epsilon)
    clean = binary_dequantized_logit(binary, uniform)
    noise = torch.randn(binary.shape, generator=generator, dtype=binary.dtype)
    time = torch.linspace(0.15, 0.85, binary.shape[0], dtype=binary.dtype).view(-1, 1, 1, 1)
    state = (1 - time) * clean + time * noise
    target_velocity = noise - clean
    model = nn.Sequential(
        nn.Conv2d(binary.shape[1], 8, 3, padding=1),
        nn.SiLU(),
        nn.Conv2d(8, binary.shape[1], 1),
    )
    prediction = model(state)
    expanded = valid.expand_as(prediction)
    loss = ((prediction - target_velocity).square() * expanded).sum() / expanded.sum().clamp_min(1)
    loss.backward()
    gradient_norms = [float(parameter.grad.norm().item()) for parameter in model.parameters()]
    decoded_exact = torch.equal(decode_binary_dequantized_logit(clean), binary)
    return {
        "objective": "continuous_dequantized_binary_cfm_no_ste",
        "loss": float(loss.item()),
        "gradient_norms": gradient_norms,
        "all_parameter_gradients_finite_positive": all(math.isfinite(value) and value > 0 for value in gradient_norms),
        "hard_decode_roundtrip_exact": bool(decoded_exact),
    }


def synthetic_representation_birth_death_check(*, seed: int) -> dict[str, Any]:
    """Show only that the codec represents birth/death without an initial edge."""
    initial_empty = torch.zeros((1, 1, 8, 8), dtype=torch.float32)
    future_birth = initial_empty.clone()
    future_birth[..., 2:5, 3:7] = 1
    initial_ice = future_birth.clone()
    future_death = torch.zeros_like(initial_ice)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    epsilon = torch.finfo(torch.float32).eps

    def roundtrip(value: torch.Tensor) -> bool:
        uniform = torch.rand(value.shape, generator=generator).clamp(epsilon, 1 - epsilon)
        return torch.equal(decode_binary_dequantized_logit(binary_dequantized_logit(value, uniform)), value)

    return {
        "initial_empty_has_edge": False,
        "future_birth_count": int(future_birth.sum().item()),
        "empty_edge_birth_roundtrip_exact": roundtrip(future_birth),
        "future_death_count": int(initial_ice.sum().item()),
        "death_roundtrip_exact": roundtrip(future_death),
        "scope": "representation_codec_only_not_a_model_source_branch_test",
    }


# Compatibility alias for immutable admission callers.  New evidence must use
# the explicit representation-only name above.
synthetic_source_support_check = synthetic_representation_birth_death_check


def run_admission(config_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    config_path, output_path = Path(config_path), Path(output_path)
    config = load_json(config_path)
    if config.get("mode") != MODE or config.get("dataset_split") != "train":
        raise ValueError("mixed-support admission is train-only and requires the exact mode")
    if config.get("resource_kind") != "cpu" or bool(config.get("test_2023_access_allowed", True)):
        raise ValueError("admission must be CPU-only with test-2023 forbidden")
    factor = int(config["coarse_factor"])
    if factor != 4:
        raise ValueError("v1 admission is frozen at 320x256 -> 80x64")
    dataset_config = load_json(config["dataset_config"])
    dataset_config = dict(dataset_config)
    dataset_config["dynamic_forcing_stats"] = config["dynamic_forcing_stats"]
    stats = load_json(config["structured_state_stats"])
    validate_structured_state_stats(stats)
    dataset = build_dataset(dataset_config, "train")
    indices = dataset.strided_case_indices(
        max_cases=int(config["max_cases"]), stride_days=int(config["stride_days"])
    )
    if len(indices) != int(config["max_cases"]):
        raise ValueError("train split does not provide all predeclared strided anchors")

    totals = {key: 0 for key in (
        "native_changed_cells", "native_changed_cells_visible_at_coarse", "coarse_changed_cells",
        "coarse_birth_cells", "coarse_death_cells",
    )}
    cases = []
    roundtrips = []
    first_binary = None
    first_valid = None
    means, stds = dataset_config["means"], dataset_config["stds"]
    for order, index in enumerate(indices):
        item = dataset[index]
        valid = item["valid_mask"][:1].unsqueeze(0).to(torch.float32)
        initial = channel_denormalize(item["structured_conditioning"][:2], means, stds).unsqueeze(0)
        future = item["structured_physical_truth"].unsqueeze(0).to(torch.float32)
        per_lead = {}
        coarse_targets = []
        for lead_index, lead_day in enumerate(dataset.trajectory_lead_days):
            counts = occurrence_change_counts(
                initial[:, 0:1], future[:, 2 * lead_index : 2 * lead_index + 1], valid, factor
            )
            for key, value in counts.items():
                totals[key] += value
            per_lead[f"d{lead_day}"] = counts
            target, support = coarse_occurrence(
                future[:, 2 * lead_index : 2 * lead_index + 1], valid, factor
            )
            coarse_targets.append(target)
        if first_binary is None:
            first_binary = torch.cat(coarse_targets, dim=1)
            first_valid = support.to(torch.float32)
        roundtrip = structured_roundtrip(future, valid, stats, seed=int(config["seed"]) + order)
        roundtrips.append(roundtrip)
        cases.append({
            "dataset_index": int(index),
            "case_id": item["meta"]["case_id"],
            "target_paths": item["meta"]["target_trajectory_paths"],
            "per_lead": per_lead,
            "roundtrip": roundtrip,
        })

    if first_binary is None or first_valid is None:
        raise RuntimeError("no admission anchors were evaluated")
    gradient = dequantized_cfm_gradient_check(first_binary, first_valid, seed=int(config["seed"]) + 1000)
    source_support = synthetic_representation_birth_death_check(seed=int(config["seed"]) + 2000)
    changed = totals["native_changed_cells"]
    retention = totals["native_changed_cells_visible_at_coarse"] / max(changed, 1)
    roundtrip_pass = all(
        result["occurrence_exact"] and result["cap_exact"]
        and result["max_abs_sic_error"] <= 1e-6 and result["max_abs_sit_error"] <= 1e-6
        and result["land_invariant"] and result["decoded_land_exact_zero"]
        for result in roundtrips
    )
    gate = {
        "representation_pass": bool(
            roundtrip_pass
            and gradient["all_parameter_gradients_finite_positive"]
            and gradient["hard_decode_roundtrip_exact"]
            and source_support["empty_edge_birth_roundtrip_exact"]
            and source_support["death_roundtrip_exact"]
            and totals["coarse_birth_cells"] > 0
            and totals["coarse_death_cells"] > 0
        ),
        "occurrence_retention_is_descriptive_not_a_calibration_gate": True,
        "gpu_training_authorized": False,
    }
    result = {
        "schema_version": MODE,
        "status": "complete",
        "resource_kind": "cpu",
        "dataset_split": "train",
        "test_2023_accessed": False,
        "target_law": "B80=1{masked_mean_pool(SIC)>0}; joint leads d3,d6,d9",
        "decoder_contract": "K<=B; A=B*(K*a_cap+(1-K)*a_cap*sigmoid(Q)); H=B*exp(L)",
        "inactive_filler_law": stats["inactive_filler_law"],
        "discrete_training_path": "continuous dequantization plus CFM; fixed zero-threshold decoder; no STE",
        "coarse_shape": [80, 64],
        "case_selection": {"rule": "strided train anchors", "indices": indices},
        "occurrence_change": {**totals, "native_change_retention_fraction": retention},
        "structured_roundtrip": {
            "all_cases_pass": roundtrip_pass,
            "max_abs_sic_error": max(item["max_abs_sic_error"] for item in roundtrips),
            "max_abs_sit_error": max(item["max_abs_sit_error"] for item in roundtrips),
        },
        "gradient_check": gradient,
        "source_support_check": source_support,
        "gate": gate,
        "claim_boundary": "representation admission only; no learned sampler or calibration claim",
        "cases": cases,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("output")
    args = parser.parse_args()
    result = run_admission(args.config, args.output)
    print(json.dumps({"status": result["status"], "gate": result["gate"]}, sort_keys=True))


if __name__ == "__main__":
    main()
