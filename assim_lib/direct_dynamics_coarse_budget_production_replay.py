"""Production-order RK4 path for the coarse-budget branch only.

The legacy terminal-refinement helper is intentionally left unchanged because
completed experiments bind its arithmetic.  This module mirrors the current
``torchdiffeq`` fixed-grid RK4 operation order and preserves production's
member-by-member batch size during actual-checkpoint replay.
"""

from __future__ import annotations

from typing import Any

import torch


def _model_output(value: Any) -> torch.Tensor:
    if hasattr(value, "sample"):
        return value.sample
    if isinstance(value, (tuple, list)):
        return value[0]
    return value


def production_velocity(
    model: torch.nn.Module,
    state: torch.Tensor,
    encoded_condition: torch.Tensor,
    active: torch.Tensor,
    grid: torch.Tensor,
    time: torch.Tensor,
) -> torch.Tensor:
    if time.ndim != 0:
        raise ValueError("production RK4 time must be scalar")
    batch = state.shape[0]
    timestamp = time.expand(batch) * 1000.0
    model_input = torch.cat(
        (state, grid.expand(batch, -1, -1, -1), encoded_condition), dim=1
    )
    with torch.autocast(
        device_type=state.device.type,
        dtype=torch.bfloat16,
        enabled=state.device.type == "cuda",
    ):
        velocity = _model_output(model(model_input, timestamp))
    velocity = velocity.to(dtype=state.dtype)
    support = active.expand_as(velocity) > 0
    return torch.where(support, velocity, torch.zeros_like(velocity))


def production_rk4_interval(
    model: torch.nn.Module,
    state: torch.Tensor,
    encoded_condition: torch.Tensor,
    active: torch.Tensor,
    grid: torch.Tensor,
    start: torch.Tensor,
    end: torch.Tensor,
) -> torch.Tensor:
    """Match ``torchdiffeq.rk4_alt_step_func`` expression ordering."""
    if start.ndim != 0 or end.ndim != 0:
        raise ValueError("production RK4 interval endpoints must be scalar")
    step = end - start
    one_third = 1.0 / 3.0
    two_thirds = 2.0 / 3.0
    k1 = production_velocity(model, state, encoded_condition, active, grid, start)
    k2 = production_velocity(
        model,
        state + step * k1 * one_third,
        encoded_condition,
        active,
        grid,
        start + step * one_third,
    )
    k3 = production_velocity(
        model,
        state + step * (k2 - k1 * one_third),
        encoded_condition,
        active,
        grid,
        start + step * two_thirds,
    )
    k4 = production_velocity(
        model,
        state + step * (k1 - k2 + k3),
        encoded_condition,
        active,
        grid,
        end,
    )
    result = state + (k1 + 3.0 * (k2 + k3) + k4) * step * 0.125
    support = active.expand_as(result) > 0
    result = torch.where(support, result, torch.zeros_like(result))
    if not torch.isfinite(result[support]).all():
        raise FloatingPointError("production-order RK4 produced NaN/Inf")
    return result


def production_frozen_prefix(
    model: torch.nn.Module,
    noise: torch.Tensor,
    encoded_condition: torch.Tensor,
    active: torch.Tensor,
    grid: torch.Tensor,
    *,
    intervals: int = 16,
    return_states: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    if intervals != 16:
        raise ValueError("coarse-budget production replay requires sixteen intervals")
    support = active.expand_as(noise) > 0
    state = torch.where(support, noise.float(), torch.zeros_like(noise.float()))
    times = torch.linspace(
        1.0, 0.0, intervals + 1, device=state.device, dtype=state.dtype
    )
    states = [state.detach().clone()]
    with torch.no_grad():
        for index in range(intervals - 1):
            state = production_rk4_interval(
                model,
                state,
                encoded_condition,
                active,
                grid,
                times[index],
                times[index + 1],
            )
            if return_states:
                states.append(state.detach().clone())
    state = state.detach()
    if return_states:
        return state, tuple(states)
    return state


def production_terminal_sample(
    model: torch.nn.Module,
    prefix_state: torch.Tensor,
    encoded_condition: torch.Tensor,
    active: torch.Tensor,
    grid: torch.Tensor,
) -> torch.Tensor:
    times = torch.linspace(
        1.0, 0.0, 17, device=prefix_state.device, dtype=prefix_state.dtype
    )
    return production_rk4_interval(
        model,
        prefix_state,
        encoded_condition,
        active,
        grid,
        times[-2],
        times[-1],
    )


def sequential_terminal_candidate_control(
    frozen_model: torch.nn.Module,
    terminal_model: torch.nn.Module,
    noise: torch.Tensor,
    encoded_condition: torch.Tensor,
    active: torch.Tensor,
    grid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Replay production batch=1 while retaining all candidate graphs."""
    if not (
        noise.shape[0] == encoded_condition.shape[0] == active.shape[0]
    ):
        raise ValueError("member batch sizes differ in production replay")
    prefixes = []
    candidates = []
    controls = []
    for index in range(noise.shape[0]):
        prefix = production_frozen_prefix(
            frozen_model,
            noise[index : index + 1],
            encoded_condition[index : index + 1],
            active[index : index + 1],
            grid,
        )
        candidate = production_terminal_sample(
            terminal_model,
            prefix,
            encoded_condition[index : index + 1],
            active[index : index + 1],
            grid,
        )
        with torch.no_grad():
            control = production_terminal_sample(
                frozen_model,
                prefix,
                encoded_condition[index : index + 1],
                active[index : index + 1],
                grid,
            )
        prefixes.append(prefix)
        candidates.append(candidate)
        controls.append(control)
    return torch.cat(candidates), torch.cat(controls), torch.cat(prefixes)
