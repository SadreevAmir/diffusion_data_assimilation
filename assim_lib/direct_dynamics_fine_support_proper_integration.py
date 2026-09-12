"""CPU integration gate for terminal support-aware fine refinement.

This module exercises the production fine-stage mechanics with a compact
network: colored initial law, variance-preconditioned velocity, projected
detail state, 31 frozen RK4 intervals, one trainable terminal interval, the
physical SIC decoder, and the joint proper objective.  It never loads the
publication checkpoint and is not a training or evaluation result.
"""

from __future__ import annotations

import copy
import math
from typing import Any

import torch
from torch import nn

from .direct_dynamics_cascade import masked_block_average, project_detail
from .direct_dynamics_cascade_fine import (
    DIRECT_OUTPUT_CHANNELS,
    FINE_INPUT_CHANNELS,
    teacher_coarse_condition,
)
from .direct_dynamics_cascade_fine_colored import (
    ColoredVariancePreconditionedFineCascadeSampler,
)
from .direct_dynamics_fine_support_proper_admission import (
    support_aware_physical_objective,
)
from .runtime import make_normalized_xy_grid


class CompactFineNetwork(nn.Module):
    """Small nontrivial network used only to prove the integrated graph."""

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Conv2d(FINE_INPUT_CHANNELS, DIRECT_OUTPUT_CHANNELS, 1)

    def forward(
        self, model_input: torch.Tensor, timestep: torch.Tensor, **_: Any
    ) -> tuple[torch.Tensor]:
        time = timestep.to(model_input).reshape(-1, 1, 1, 1) / 1000.0
        return (self.projection(model_input) + 0.01 * time,)


def _model_output(value: Any) -> torch.Tensor:
    if hasattr(value, "sample"):
        return value.sample
    if isinstance(value, (tuple, list)):
        return value[0]
    return value


def fine_velocity(
    model: nn.Module,
    state: torch.Tensor,
    condition: torch.Tensor,
    active: torch.Tensor,
    grid: torch.Tensor,
    time: float | torch.Tensor,
) -> torch.Tensor:
    batch = state.shape[0]
    scalar_time = torch.as_tensor(time, device=state.device, dtype=state.dtype)
    if scalar_time.ndim != 0:
        raise ValueError("fine RK4 time must be scalar")
    timestamp = scalar_time.expand(batch) * 1000.0
    model_input = torch.cat((state, grid.expand(batch, -1, -1, -1), condition), dim=1)
    with torch.autocast(
        device_type=state.device.type,
        dtype=torch.bfloat16,
        enabled=state.device.type == "cuda",
    ):
        velocity = _model_output(model(model_input, timestamp, return_dict=False))
    velocity = velocity.float()
    return torch.where(active.expand_as(velocity) > 0, velocity, torch.zeros_like(velocity))


def fine_rk4_interval(
    model: nn.Module,
    state: torch.Tensor,
    condition: torch.Tensor,
    active: torch.Tensor,
    grid: torch.Tensor,
    t0: float | torch.Tensor,
    t1: float | torch.Tensor,
) -> torch.Tensor:
    """Exactly reproduce torchdiffeq's reverse-time RK4 3/8 arithmetic."""
    physical_t0 = torch.as_tensor(t0, device=state.device, dtype=state.dtype)
    physical_t1 = torch.as_tensor(t1, device=state.device, dtype=state.dtype)
    solver_t0 = -physical_t0
    solver_t1 = -physical_t1
    step = solver_t1 - solver_t0

    def reverse_velocity(solver_time: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return -fine_velocity(
            model, value, condition, active, grid, -solver_time
        )

    one_third = 1.0 / 3.0
    two_thirds = 2.0 / 3.0
    k1 = reverse_velocity(solver_t0, state)
    k2 = reverse_velocity(
        solver_t0 + step * one_third,
        state + step * k1 * one_third,
    )
    k3 = reverse_velocity(
        solver_t0 + step * two_thirds,
        state + step * (k2 - k1 * one_third),
    )
    k4 = reverse_velocity(
        solver_t1,
        state + step * (k1 - k2 + k3),
    )
    result = state + (k1 + 3.0 * (k2 + k3) + k4) * step * 0.125
    return torch.where(active.expand_as(result) > 0, result, torch.zeros_like(result))


def fine_frozen_prefix(
    model: nn.Module,
    initial_state: torch.Tensor,
    condition: torch.Tensor,
    active: torch.Tensor,
    grid: torch.Tensor,
    *,
    intervals: int = 32,
) -> torch.Tensor:
    if intervals != 32:
        raise ValueError("fine refinement requires exactly 32 RK4 intervals")
    state = torch.where(
        active.expand_as(initial_state) > 0,
        initial_state.float(),
        torch.zeros_like(initial_state.float()),
    )
    with torch.no_grad():
        timepoints = torch.linspace(
            1.0,
            0.0,
            intervals + 1,
            device=state.device,
            dtype=state.dtype,
        )
        for interval in range(intervals - 1):
            state = fine_rk4_interval(
                model,
                state,
                condition,
                active,
                grid,
                timepoints[interval],
                timepoints[interval + 1],
            )
    return state.detach()


def terminal_fine_residual(
    model: nn.Module,
    prefix: torch.Tensor,
    condition: torch.Tensor,
    active: torch.Tensor,
    grid: torch.Tensor,
) -> torch.Tensor:
    return project_detail(
        fine_rk4_interval(model, prefix, condition, active, grid, 1.0 / 32.0, 0.0),
        active,
    )


def _repeat_members(value: torch.Tensor, members: int) -> torch.Tensor:
    return value[:, None].expand(-1, members, *value.shape[1:]).flatten(0, 1)


def compact_fine_integration_check() -> dict[str, Any]:
    """Prove the complete fine-refinement graph on CPU without training."""
    torch.manual_seed(20260912)
    members = 4
    height = width = 8
    mask = torch.ones(1, 1, height, width, dtype=torch.float32)
    mask[..., 0, 0] = 0.0
    structured = torch.randn(1, 15, height, width, dtype=torch.float32)
    structured[:, 2:3] = mask
    means = torch.tensor([0.20, 0.30] * 3, dtype=torch.float64)
    stds = torch.tensor([0.35, 0.45] * 3, dtype=torch.float64)
    truth_physical = torch.rand(1, 6, height, width, dtype=torch.float64)
    truth_physical[:, 1::2] *= 2.0
    truth_physical[:, 0::2, :2, :2] = 0.0
    truth_normalized = (
        truth_physical - means.reshape(1, 6, 1, 1)
    ) / stds.reshape(1, 6, 1, 1)
    truth_normalized = truth_normalized.float()
    condition, coarse_normalized, lift_residual = teacher_coarse_condition(
        truth_normalized, structured, mask
    )
    del lift_residual

    sampler = ColoredVariancePreconditionedFineCascadeSampler(
        CompactFineNetwork(),
        target_residual_rms=(0.4,) * 6,
        projected_base_rms=(0.7,) * 6,
        blend=0.75,
        channel_scales=(0.5,) * 6,
    )
    frozen_model = sampler.sampler.model.eval()
    for parameter in frozen_model.parameters():
        parameter.requires_grad_(False)
    terminal_model = copy.deepcopy(frozen_model).train()
    for parameter in terminal_model.parameters():
        parameter.requires_grad_(True)

    condition_m = _repeat_members(condition, members)
    mask_m = _repeat_members(mask, members)
    white = torch.randn(members, 6, height, width, dtype=torch.float32)
    colored = sampler.project_initial_noise(white, mask_m)
    grid = make_normalized_xy_grid(height, width, device=torch.device("cpu"), dtype=torch.float32)
    prefix = fine_frozen_prefix(frozen_model, colored, condition_m, mask_m, grid)
    candidate_residual = terminal_fine_residual(
        terminal_model, prefix, condition_m, mask_m, grid
    )
    with torch.no_grad():
        control_residual = terminal_fine_residual(
            frozen_model, prefix, condition_m, mask_m, grid
        )
        zeros = torch.zeros(members, 6, height, width, dtype=torch.float32)
        production = sampler.sample_conditioned(
            background=zeros,
            background_mask=torch.ones_like(zeros),
            obs_values=zeros[:, :2],
            obs_mask=zeros[:, :2],
            water_mask=mask_m,
            size=(height, width),
            num_timesteps=33,
            device=torch.device("cpu"),
            method="rk4",
            start_mode="noise",
            initial_noise=white,
            sample_target="state",
            model_conditioning=condition_m,
            state_channels=6,
            end_time=0.0,
        )
    lift = condition_m[:, -6:]
    candidate_normalized = (lift + candidate_residual).unflatten(0, (1, members))
    control_normalized = (lift + control_residual).unflatten(0, (1, members))
    candidate_control = float((candidate_normalized.detach() - control_normalized).abs().max())
    production_replay = float((control_normalized.flatten(0, 1) - production).abs().max())
    if candidate_control > 1e-6 or production_replay > 2e-5:
        raise RuntimeError("terminal fine path does not replay production at step zero")
    if prefix.requires_grad or prefix.grad_fn is not None:
        raise RuntimeError("fine frozen prefix retained an autograd graph")

    scale = stds.reshape(1, 1, 6, 1, 1)
    offset = means.reshape(1, 1, 6, 1, 1)
    candidate_physical = candidate_normalized.double() * scale + offset
    coarse_physical = (
        _repeat_members(coarse_normalized, members).unflatten(0, (1, members)).double()
        * scale
        + offset
    )
    objective, crps, energy, decoded = support_aware_physical_objective(
        candidate_physical,
        coarse_physical,
        truth_physical,
        mask.double(),
        stds,
    )
    objective.backward()
    terminal_gradients = [
        value.grad for value in terminal_model.parameters() if value.grad is not None
    ]
    frozen_gradients = [value.grad for value in frozen_model.parameters()]
    if not terminal_gradients or not all(torch.isfinite(value).all() for value in terminal_gradients):
        raise FloatingPointError("terminal fine parameters lack finite gradients")
    terminal_norm = float(
        torch.sqrt(sum(value.float().square().sum() for value in terminal_gradients))
    )
    if not math.isfinite(terminal_norm) or terminal_norm <= 0:
        raise FloatingPointError("terminal fine parameter gradient is dead")
    if any(value is not None for value in frozen_gradients):
        raise RuntimeError("gradient leaked through the frozen fine prefix")
    recovered, fraction = masked_block_average(
        decoded.flatten(0, 1)[:, 0::2], mask_m
    )
    expected = coarse_physical.flatten(0, 1)[:, 0::2].clamp(0, 1)
    coarse_error = float((recovered - expected)[fraction > 0].abs().max())
    if coarse_error > 3e-14:
        raise RuntimeError("integrated decoded sample lost SIC coarse consistency")
    residual_coarse, residual_fraction = masked_block_average(candidate_residual, mask_m)
    residual_error = float(residual_coarse[residual_fraction > 0].abs().max())
    if residual_error > 3e-6:
        raise RuntimeError("integrated fine residual left the detail nullspace")
    return {
        "status": "pass",
        "scientific_role": "compact_cpu_integration_not_training_or_evaluation",
        "training_performed": False,
        "synthetic_sampling_performed": True,
        "checkpoint_sampling_performed": False,
        "gpu_used": False,
        "rk4_intervals": 32,
        "frozen_prefix_intervals": 31,
        "trainable_terminal_intervals": 1,
        "members": members,
        "candidate_control_max_abs": candidate_control,
        "production_replay_max_abs": production_replay,
        "objective": float(objective.detach()),
        "fair_crps": float(crps.detach()),
        "joint_energy": float(energy.detach()),
        "terminal_parameter_gradient_norm": terminal_norm,
        "frozen_prefix_has_no_graph": True,
        "frozen_parameter_gradients_absent": True,
        "residual_nullspace_max_abs": residual_error,
        "decoded_sic_coarse_consistency_max_abs": coarse_error,
        "colored_initial_state_used": True,
        "variance_preconditioned_velocity_used": True,
    }
