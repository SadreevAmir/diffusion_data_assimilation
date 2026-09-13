"""Exact frozen-prefix/trainable-suffix RK4 utilities for direct dynamics.

The production direct model is sampled with sixteen reverse-time RK4 3/8
intervals.  These helpers freeze the first twelve intervals and differentiate
the last four.  ``exact_two_pass_score_vjp`` keeps all cross-member terms of a
joint proper score while replaying only one member's suffix graph at a time.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Callable

import torch


TOTAL_INTERVALS = 16
FROZEN_INTERVALS = 12
TRAINABLE_INTERVALS = TOTAL_INTERVALS - FROZEN_INTERVALS


def _model_output(value: Any) -> torch.Tensor:
    if hasattr(value, "sample"):
        return value.sample
    if isinstance(value, (tuple, list)):
        return value[0]
    return value


def _autocast_context(state: torch.Tensor, enabled: bool):
    if enabled and state.device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def direct_velocity(
    model: torch.nn.Module,
    state: torch.Tensor,
    condition: torch.Tensor,
    valid: torch.Tensor,
    grid: torch.Tensor,
    time: torch.Tensor,
    *,
    autocast_bf16: bool = True,
) -> torch.Tensor:
    if state.ndim != 4 or condition.ndim != 4 or valid.ndim != 4 or grid.ndim != 4:
        raise ValueError("state, condition, valid, and grid must be BCHW tensors")
    if state.shape[0] != condition.shape[0] or state.shape[0] != valid.shape[0]:
        raise ValueError("state, condition, and valid batch sizes differ")
    if state.shape[-2:] != condition.shape[-2:] or state.shape[-2:] != valid.shape[-2:]:
        raise ValueError("state, condition, and valid spatial shapes differ")
    if valid.shape[1] != 1:
        raise ValueError("valid must have one channel")
    if time.ndim != 0 or time.device != state.device or time.dtype != state.dtype:
        raise ValueError("time must be a scalar tensor matching state device and dtype")
    timestamp = time.expand(state.shape[0]) * 1000.0
    model_input = torch.cat(
        (state, grid.expand(state.shape[0], -1, -1, -1), condition), dim=1
    )
    with _autocast_context(state, autocast_bf16):
        velocity = _model_output(model(model_input, timestamp))
    velocity = velocity.to(dtype=state.dtype)
    support = valid.expand_as(velocity) > 0
    velocity = torch.where(support, velocity, torch.zeros_like(velocity))
    if not torch.all(torch.isfinite(velocity[support])):
        raise FloatingPointError("direct velocity is non-finite on valid ocean")
    return velocity


def rk4_interval(
    model: torch.nn.Module,
    state: torch.Tensor,
    condition: torch.Tensor,
    valid: torch.Tensor,
    grid: torch.Tensor,
    t0: torch.Tensor,
    t1: torch.Tensor,
    *,
    autocast_bf16: bool = True,
) -> torch.Tensor:
    """One torchdiffeq-compatible fixed RK4 3/8 interval."""

    if t0.ndim != 0 or t1.ndim != 0:
        raise ValueError("RK4 interval endpoints must be scalar tensors")
    step = t1 - t0
    one_third = 1.0 / 3.0
    two_thirds = 2.0 / 3.0

    def velocity(value: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        return direct_velocity(
            model,
            value,
            condition,
            valid,
            grid,
            time,
            autocast_bf16=autocast_bf16,
        )

    k1 = velocity(state, t0)
    k2 = velocity(state + step * k1 * one_third, t0 + step * one_third)
    k3 = velocity(state + step * (k2 - k1 * one_third), t0 + step * two_thirds)
    k4 = velocity(state + step * (k1 - k2 + k3), t1)
    result = state + (k1 + 3.0 * (k2 + k3) + k4) * step * 0.125
    support = valid.expand_as(result) > 0
    result = torch.where(support, result, torch.zeros_like(result))
    if not torch.all(torch.isfinite(result[support])):
        raise FloatingPointError("direct RK4 interval produced NaN/Inf")
    return result


def integrate_intervals(
    model: torch.nn.Module,
    state: torch.Tensor,
    condition: torch.Tensor,
    valid: torch.Tensor,
    grid: torch.Tensor,
    *,
    first_interval: int,
    final_interval: int,
    total_intervals: int = TOTAL_INTERVALS,
    autocast_bf16: bool = True,
) -> torch.Tensor:
    if total_intervals != TOTAL_INTERVALS:
        raise ValueError("reviewed direct suffix requires sixteen RK4 intervals")
    if not 0 <= first_interval <= final_interval <= total_intervals:
        raise ValueError("invalid RK4 interval range")
    result = state
    times = torch.linspace(
        1.0,
        0.0,
        total_intervals + 1,
        device=state.device,
        dtype=state.dtype,
    )
    for interval in range(first_interval, final_interval):
        result = rk4_interval(
            model,
            result,
            condition,
            valid,
            grid,
            times[interval],
            times[interval + 1],
            autocast_bf16=autocast_bf16,
        )
    return result


def frozen_prefix(
    model: torch.nn.Module,
    noise: torch.Tensor,
    condition: torch.Tensor,
    valid: torch.Tensor,
    grid: torch.Tensor,
    *,
    autocast_bf16: bool = True,
) -> torch.Tensor:
    support = valid.expand_as(noise) > 0
    state = torch.where(support, noise.float(), torch.zeros_like(noise.float()))
    with torch.no_grad():
        result = integrate_intervals(
            model,
            state,
            condition,
            valid,
            grid,
            first_interval=0,
            final_interval=FROZEN_INTERVALS,
            autocast_bf16=autocast_bf16,
        )
    return result.detach()


def trainable_suffix(
    model: torch.nn.Module,
    prefix_state: torch.Tensor,
    condition: torch.Tensor,
    valid: torch.Tensor,
    grid: torch.Tensor,
    *,
    autocast_bf16: bool = True,
) -> torch.Tensor:
    return integrate_intervals(
        model,
        prefix_state,
        condition,
        valid,
        grid,
        first_interval=FROZEN_INTERVALS,
        final_interval=TOTAL_INTERVALS,
        autocast_bf16=autocast_bf16,
    )


def exact_two_pass_score_vjp(
    model: torch.nn.Module,
    prefix_states: torch.Tensor,
    condition: torch.Tensor,
    valid: torch.Tensor,
    grid: torch.Tensor,
    score: Callable[[torch.Tensor], torch.Tensor],
    *,
    autocast_bf16: bool = True,
    replay_atol: float = 0.0,
    evidence_callback: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor], None
    ]
    | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Backpropagate one joint member score with a deterministic exact replay.

    ``score`` receives all members with shape ``[B,M,C,H,W]``.  Its output
    gradient therefore already contains every cross-member pair term.  The
    model suffix is then replayed member-by-member with those fixed output
    gradients, which is algebraically identical to one large autograd graph
    when the suffix is deterministic.
    """

    if prefix_states.ndim != 5 or prefix_states.shape[1] < 2:
        raise ValueError("prefix_states must have shape [B,M,C,H,W] with M>=2")
    batch, members = prefix_states.shape[:2]
    if condition.shape[0] != batch or valid.shape[0] != batch:
        raise ValueError("condition/valid batch differs from prefix states")
    if model.training:
        raise ValueError("exact replay requires eval mode so dropout is disabled")

    first_pass = []
    with torch.no_grad():
        for member in range(members):
            first_pass.append(
                trainable_suffix(
                    model,
                    prefix_states[:, member],
                    condition,
                    valid,
                    grid,
                    autocast_bf16=autocast_bf16,
                )
            )
    outputs = torch.stack(first_pass, dim=1)
    score_outputs = outputs.detach().requires_grad_(True)
    objective = score(score_outputs)
    if objective.ndim != 0 or not torch.isfinite(objective):
        raise FloatingPointError("joint score must be one finite scalar")
    output_gradient = torch.autograd.grad(objective, score_outputs)[0].detach()
    if not torch.all(torch.isfinite(output_gradient)) or output_gradient.abs().sum() <= 0:
        raise FloatingPointError("joint score output gradient is dead or non-finite")
    if evidence_callback is not None:
        evidence_callback(outputs.detach(), objective.detach(), output_gradient)

    replay_max_abs = 0.0
    for member in range(members):
        replay = trainable_suffix(
            model,
            prefix_states[:, member],
            condition,
            valid,
            grid,
            autocast_bf16=autocast_bf16,
        )
        difference = float((replay.detach() - outputs[:, member]).abs().max().cpu())
        replay_max_abs = max(replay_max_abs, difference)
        if difference > float(replay_atol):
            raise RuntimeError(
                f"member {member} suffix replay mismatch {difference} exceeds {replay_atol}"
            )
        torch.autograd.backward(replay, output_gradient[:, member])
    return outputs.detach(), objective.detach(), {
        "output_gradient_norm": float(torch.linalg.vector_norm(output_gradient.float()).cpu()),
        "suffix_replay_max_abs": replay_max_abs,
    }
