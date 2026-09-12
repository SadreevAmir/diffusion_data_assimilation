"""Production-order RK4 path for the coarse-budget branch only.

The legacy terminal-refinement helper is intentionally left unchanged because
completed experiments bind its arithmetic.  This module mirrors the current
``torchdiffeq`` fixed-grid RK4 operation order and preserves production's
member-by-member batch size during actual-checkpoint replay.
"""

from __future__ import annotations

import copy
import math
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


class _DiagnosticCoarse(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input = torch.nn.Conv2d(14, 12, 3, padding=1)
        self.output = torch.nn.Conv2d(12, 6, 3, padding=1)

    def forward(
        self, value: torch.Tensor, timestamp: torch.Tensor, return_dict: bool = True
    ) -> tuple[torch.Tensor]:
        del return_dict
        time = timestamp.reshape(-1, 1, 1, 1) / 1000.0
        return (self.output(torch.tanh(self.input(value))) + 0.017 * time,)


class _RecordingModel(torch.nn.Module):
    """Record exact solver/model boundaries without changing model arithmetic."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model
        self.calls: list[dict[str, torch.Tensor]] = []

    def forward(
        self,
        value: torch.Tensor,
        timestamp: torch.Tensor,
        return_dict: bool = True,
    ) -> Any:
        output = self.model(value, timestamp, return_dict=return_dict)
        tensor = _model_output(output)
        self.calls.append(
            {
                "input": value.detach().clone(),
                "timestamp": timestamp.detach().clone(),
                "velocity": tensor.detach().clone(),
            }
        )
        return output


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left - right).abs().max())


def _first_record_difference(
    reference: list[dict[str, torch.Tensor]],
    candidate: list[dict[str, torch.Tensor]],
) -> dict[str, Any] | None:
    if len(reference) != len(candidate):
        return {
            "kind": "call_count",
            "reference_calls": len(reference),
            "candidate_calls": len(candidate),
        }
    for call_index, (expected, actual) in enumerate(zip(reference, candidate)):
        differences = {
            name: _max_abs(expected[name], actual[name])
            for name in ("input", "timestamp", "velocity")
        }
        if any(value != 0.0 for value in differences.values()):
            return {
                "kind": "rk4_stage",
                "call_index": call_index,
                "interval": call_index // 4,
                "stage": call_index % 4 + 1,
                **{f"{name}_max_abs": value for name, value in differences.items()},
            }
    return None


def _production_sampler_terminal(
    model: torch.nn.Module,
    noise: torch.Tensor,
    condition: torch.Tensor,
    active: torch.Tensor,
) -> torch.Tensor:
    """Call the actual public Sampler path used by coarse production."""
    from .sampler import Sampler

    zeros = torch.zeros_like(condition)
    return Sampler(model).sample_conditioned(
        background=zeros,
        background_mask=torch.ones_like(zeros),
        obs_values=zeros,
        obs_mask=zeros,
        water_mask=active,
        valid_mask=active,
        state_mask=active.expand_as(noise),
        size=tuple(noise.shape[-2:]),
        num_timesteps=17,
        device=torch.device("cpu"),
        method="rk4",
        start_mode="noise",
        initial_noise=noise,
        sample_target="state",
        model_conditioning=condition,
        state_channels=noise.shape[1],
        end_time=0.0,
    )


def compact_first_divergence_diagnostic() -> dict[str, Any]:
    """Prove exact replay against Sampler and localize the legacy divergence."""
    from .direct_dynamics_cascade_coarse_proper_refinement import (
        rk4_interval as legacy_rk4_interval,
    )
    from .runtime import make_normalized_xy_grid

    torch.manual_seed(20260913)
    base = _DiagnosticCoarse().eval()
    frozen = copy.deepcopy(base).eval()
    for parameter in frozen.parameters():
        parameter.requires_grad_(False)
    terminal = copy.deepcopy(frozen).train()
    for parameter in terminal.parameters():
        parameter.requires_grad_(True)
    batch = 4
    noise = torch.randn((batch, 6, 5, 4), dtype=torch.float32)
    condition = torch.randn((batch, 6, 5, 4), dtype=torch.float32)
    active = torch.ones((batch, 1, 5, 4), dtype=torch.float32)
    active[:, :, 0, 0] = 0
    noise = torch.where(active.expand_as(noise) > 0, noise, torch.zeros_like(noise))
    grid = make_normalized_xy_grid(
        5, 4, device=torch.device("cpu"), dtype=torch.float32
    )
    times = torch.linspace(1.0, 0.0, 17, dtype=torch.float32)

    production_model = _RecordingModel(copy.deepcopy(base).eval())
    reference_terminal = _production_sampler_terminal(
        production_model, noise[:1], condition[:1], active[:1]
    )
    replay_model = _RecordingModel(copy.deepcopy(base).eval())
    replay_states = [noise[:1]]
    replay = noise[:1]
    for index in range(16):
        replay = production_rk4_interval(
            replay_model,
            replay,
            condition[:1],
            active[:1],
            grid,
            times[index],
            times[index + 1],
        )
        replay_states.append(replay)
    production_states = [
        production_model.calls[4 * index]["input"][:, :6] for index in range(16)
    ] + [reference_terminal]
    state_deltas = [
        _max_abs(expected, actual)
        for expected, actual in zip(production_states, replay_states)
    ]
    replay_difference = _first_record_difference(
        production_model.calls, replay_model.calls
    )
    if replay_difference is not None or any(value != 0.0 for value in state_deltas):
        raise RuntimeError("versioned helper differs from the public production Sampler")

    legacy_model = _RecordingModel(copy.deepcopy(base).eval())
    legacy = noise[:1]
    for index in range(16):
        legacy = legacy_rk4_interval(
            legacy_model,
            legacy,
            condition[:1],
            active[:1],
            grid,
            float(times[index]),
            float(times[index + 1]),
        )
    first_divergent = _first_record_difference(
        production_model.calls, legacy_model.calls
    )
    if first_divergent is None and not torch.equal(reference_terminal, legacy):
        first_divergent = {
            "kind": "terminal_update",
            "terminal_max_abs": _max_abs(reference_terminal, legacy),
        }
    if first_divergent is None:
        raise RuntimeError("compact diagnostic did not expose legacy FP divergence")

    batched_terminal = _production_sampler_terminal(
        copy.deepcopy(base).eval(), noise, condition, active
    )
    sequential_terminal = torch.cat(
        [
            _production_sampler_terminal(
                copy.deepcopy(base).eval(),
                noise[index : index + 1],
                condition[index : index + 1],
                active[index : index + 1],
            )
            for index in range(batch)
        ]
    )
    batch_sequential_max_abs = _max_abs(batched_terminal, sequential_terminal)

    candidate, control, prefix = sequential_terminal_candidate_control(
        frozen, terminal, noise, condition, active, grid
    )
    if not torch.equal(candidate.detach(), control):
        raise RuntimeError("step-zero sequential candidate differs from control")
    candidate.square().mean().backward()
    gradients = [
        parameter.grad for parameter in terminal.parameters() if parameter.grad is not None
    ]
    if not gradients or not all(torch.isfinite(value).all() for value in gradients):
        raise FloatingPointError("versioned terminal path lacks finite gradients")
    gradient_norm = float(
        torch.sqrt(sum(value.double().square().sum() for value in gradients))
    )
    if not math.isfinite(gradient_norm) or gradient_norm <= 0:
        raise FloatingPointError("versioned terminal path has a dead gradient")
    if any(parameter.grad is not None for parameter in frozen.parameters()):
        raise RuntimeError("versioned replay leaked gradient into frozen model")
    return {
        "status": "pass",
        "first_legacy_divergence": first_divergent,
        "production_state_max_abs": state_deltas,
        "production_call_count": len(production_model.calls),
        "production_order_all_states_bitwise_equal": True,
        "batch4_vs_sequential_batch1_max_abs": batch_sequential_max_abs,
        "sequential_candidate_control_bitwise_equal": True,
        "prefix_has_no_graph": not prefix.requires_grad,
        "terminal_parameter_gradient_norm": gradient_norm,
        "frozen_parameter_gradients_absent": True,
    }
