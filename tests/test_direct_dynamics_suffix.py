from __future__ import annotations

import copy

import torch
from torchdiffeq import odeint

from assim_lib.direct_dynamics_suffix import (
    FROZEN_INTERVALS,
    TOTAL_INTERVALS,
    exact_two_pass_score_vjp,
    frozen_prefix,
    integrate_intervals,
)
from assim_lib.sampler import Sampler
from assim_lib.runtime import make_normalized_xy_grid


class TinyVelocity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.12))
        self.bias = torch.nn.Parameter(torch.tensor(-0.03))

    def forward(self, value, timestep, return_dict=False):
        state = value[:, :2]
        condition = value[:, 4:5]
        velocity = self.scale * state + self.bias * condition + timestep[:, None, None, None] / 10000
        return (velocity,)


class CoupledNonlinearVelocity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(5, 2, kernel_size=3, padding=1)
        with torch.no_grad():
            values = torch.arange(self.conv.weight.numel(), dtype=torch.float32)
            self.conv.weight.copy_(0.004 * torch.sin(values).reshape_as(self.conv.weight))
            self.conv.bias.copy_(torch.tensor((0.01, -0.02)))

    def forward(self, value, timestep, return_dict=False):
        velocity = torch.tanh(self.conv(value)) + timestep[:, None, None, None] / 20000
        return (velocity,)


def _case():
    generator = torch.Generator().manual_seed(29)
    batch, members, height, width = 2, 4, 4, 4
    noise = torch.randn(batch, members, 2, height, width, generator=generator)
    condition = torch.randn(batch, 1, height, width, generator=generator)
    valid = torch.ones(batch, 1, height, width)
    valid[..., 0, 0] = 0
    grid = torch.zeros(1, 2, height, width)
    return noise, condition, valid, grid


def test_frozen_prefix_plus_suffix_equals_full_rk4():
    model = TinyVelocity().eval()
    noise, condition, valid, grid = _case()
    flat_noise = noise.flatten(0, 1)
    flat_condition = condition[:, None].expand(-1, 4, -1, -1, -1).flatten(0, 1)
    flat_valid = valid[:, None].expand(-1, 4, -1, -1, -1).flatten(0, 1)
    full = integrate_intervals(
        model,
        flat_noise,
        flat_condition,
        flat_valid,
        grid,
        first_interval=0,
        final_interval=TOTAL_INTERVALS,
        autocast_bf16=False,
    )
    prefix = frozen_prefix(
        model,
        flat_noise,
        flat_condition,
        flat_valid,
        grid,
        autocast_bf16=False,
    )
    assert FROZEN_INTERVALS == 12
    suffix = integrate_intervals(
        model,
        prefix,
        flat_condition,
        flat_valid,
        grid,
        first_interval=FROZEN_INTERVALS,
        final_interval=TOTAL_INTERVALS,
        autocast_bf16=False,
    )
    assert torch.equal(full, suffix)


def test_custom_rk4_replays_public_sampler_with_masked_initial_noise():
    model = CoupledNonlinearVelocity().eval()
    sampler = Sampler(model, structured_velocity_parameterization="raw")
    generator = torch.Generator().manual_seed(71)
    flat_noise = torch.randn(1, 2, 8, 8, generator=generator)
    flat_condition = torch.randn(1, 1, 8, 8, generator=generator)
    flat_valid = torch.ones(1, 1, 8, 8)
    flat_valid[..., :2, :] = 0
    flat_valid[..., :, :2] = 0
    grid = make_normalized_xy_grid(8, 8)
    masked_noise = torch.where(
        flat_valid.expand_as(flat_noise) > 0,
        flat_noise,
        torch.zeros_like(flat_noise),
    )
    custom = integrate_intervals(
        model,
        masked_noise,
        flat_condition,
        flat_valid,
        grid,
        first_interval=0,
        final_interval=TOTAL_INTERVALS,
        autocast_bf16=False,
    )

    def production_velocity(time, state):
        timestamp = time.expand(state.shape[0]) * 1000.0
        model_input = torch.cat((state, grid, flat_condition), dim=1)
        velocity = model(model_input, timestamp)[0].to(state.dtype)
        return torch.where(
            flat_valid.expand_as(velocity) > 0,
            velocity,
            torch.zeros_like(velocity),
        )

    times = torch.linspace(1.0, 0.0, 17)
    production_trajectory = odeint(
        production_velocity,
        masked_noise,
        times,
        method="rk4",
        options={"step_size": 1.0 / 16.0},
    )
    custom_states = [masked_noise]
    state = masked_noise
    from assim_lib.direct_dynamics_suffix import rk4_interval

    for index in range(16):
        state = rk4_interval(
            model,
            state,
            flat_condition,
            flat_valid,
            grid,
            times[index],
            times[index + 1],
            autocast_bf16=False,
        )
        custom_states.append(state)
    assert torch.equal(torch.stack(custom_states), production_trajectory)
    zeros = torch.zeros_like(flat_noise)
    production = sampler.sample_conditioned(
        background=zeros,
        background_mask=torch.ones_like(zeros),
        obs_values=zeros,
        obs_mask=zeros,
        water_mask=flat_valid,
        size=tuple(flat_noise.shape[-2:]),
        num_timesteps=17,
        method="rk4",
        start_mode="noise",
        initial_noise=flat_noise,
        sample_target="state",
        model_conditioning=flat_condition,
        state_channels=2,
        end_time=0.0,
        state_mask=flat_valid.expand_as(flat_noise),
    )
    assert torch.equal(custom, production)
    assert torch.all(production[..., 0, 0] == 0)


def test_two_pass_vjp_matches_full_joint_graph_gradient():
    base = TinyVelocity().eval()
    replay_model = copy.deepcopy(base).eval()
    full_model = copy.deepcopy(base).eval()
    noise, condition, valid, grid = _case()
    batch, members = noise.shape[:2]
    flat_condition = condition[:, None].expand(-1, members, -1, -1, -1).flatten(0, 1)
    flat_valid = valid[:, None].expand(-1, members, -1, -1, -1).flatten(0, 1)
    prefix = frozen_prefix(
        base,
        noise.flatten(0, 1),
        flat_condition,
        flat_valid,
        grid,
        autocast_bf16=False,
    ).unflatten(0, (batch, members))

    def joint_score(outputs):
        target = torch.zeros_like(outputs[:, 0])
        observation = (outputs - target[:, None]).square().mean()
        pair = outputs.new_zeros(())
        for left in range(members):
            for right in range(left + 1, members):
                pair = pair + (outputs[:, left] - outputs[:, right]).abs().mean()
        return observation - 0.03 * pair

    replay_model.zero_grad(set_to_none=True)
    captured = {}

    def capture(outputs, objective, output_gradient):
        captured["outputs"] = outputs.clone()
        captured["objective"] = objective.clone()
        captured["output_gradient"] = output_gradient.clone()

    replay_outputs, replay_score, evidence = exact_two_pass_score_vjp(
        replay_model,
        prefix,
        condition,
        valid,
        grid,
        joint_score,
        autocast_bf16=False,
        replay_atol=0.0,
        evidence_callback=capture,
    )
    replay_gradient = {name: parameter.grad.clone() for name, parameter in replay_model.named_parameters()}

    full_model.zero_grad(set_to_none=True)
    outputs = []
    for member in range(members):
        outputs.append(
            integrate_intervals(
                full_model,
                prefix[:, member],
                condition,
                valid,
                grid,
                first_interval=FROZEN_INTERVALS,
                final_interval=TOTAL_INTERVALS,
                autocast_bf16=False,
            )
        )
    full_score = joint_score(torch.stack(outputs, dim=1))
    full_score.backward()
    assert torch.equal(replay_score, full_score.detach())
    assert evidence["suffix_replay_max_abs"] == 0
    assert torch.equal(captured["outputs"], replay_outputs)
    assert torch.equal(captured["objective"], replay_score)
    assert torch.isfinite(captured["output_gradient"]).all()
    for name, parameter in full_model.named_parameters():
        assert torch.allclose(replay_gradient[name], parameter.grad, atol=1e-6, rtol=1e-6)


def test_two_pass_rejects_training_mode_dropout_risk():
    model = TinyVelocity().train()
    noise, condition, valid, grid = _case()
    prefix = noise
    try:
        exact_two_pass_score_vjp(
            model,
            prefix,
            condition,
            valid,
            grid,
            lambda value: value.square().mean(),
            autocast_bf16=False,
        )
    except ValueError as error:
        assert "eval mode" in str(error)
    else:
        raise AssertionError("training-mode replay was not rejected")
