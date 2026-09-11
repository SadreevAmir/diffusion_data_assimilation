import contextlib

import torch
from torchdiffeq import odeint

from assim_lib.direct_dynamics_cascade_coarse_proper_refinement import (
    frozen_prefix,
    hybrid_terminal_sample,
    proper_objective,
    rk4_interval,
    standardized_fair_crps,
    standardized_joint_energy,
)


def test_fair_crps_two_member_closed_form():
    members = torch.zeros(1, 2, 6, 1, 1)
    members[:, 1] = 2.0
    truth = torch.ones(1, 6, 1, 1)
    fraction = torch.ones(1, 1, 1, 1)
    assert standardized_fair_crps(members, truth, fraction).item() == 0.0


def test_joint_energy_two_member_closed_form():
    members = torch.zeros(1, 2, 6, 1, 1)
    members[:, 1] = 2.0
    truth = torch.ones(1, 6, 1, 1)
    fraction = torch.ones(1, 1, 1, 1)
    assert abs(standardized_joint_energy(members, truth, fraction).item()) < 1e-5
    objective, crps, energy = proper_objective(members, truth, fraction)
    assert torch.isfinite(objective + crps + energy)


def test_scores_ignore_inactive_nan_and_preserve_backward():
    members = torch.zeros(1, 2, 6, 1, 2, requires_grad=True)
    truth = torch.zeros(1, 6, 1, 2)
    fraction = torch.tensor([[[[1.0, 0.0]]]])
    with torch.no_grad():
        members[..., 1] = torch.nan
        truth[..., 1] = torch.nan
        members[:, 1, :, :, 0] = 1.0
    objective, crps, energy = proper_objective(members, truth, fraction)
    assert torch.isfinite(objective + crps + energy)
    objective.backward()
    assert members.grad is not None
    assert torch.isfinite(members.grad[..., 0]).all()


def test_joint_energy_exact_zero_has_finite_zero_gradient():
    members = torch.zeros(1, 4, 6, 2, 2, requires_grad=True)
    truth = torch.zeros(1, 6, 2, 2)
    fraction = torch.ones(1, 1, 2, 2)
    score = standardized_joint_energy(members, truth, fraction)
    score.backward()
    assert score.item() == 0.0
    assert torch.isfinite(members.grad).all()
    assert torch.count_nonzero(members.grad) == 0


def test_fractional_ocean_weights_change_score_case_equally():
    members = torch.zeros(1, 2, 6, 1, 2)
    members[:, :, :, :, 1] = 2.0
    truth = torch.zeros(1, 6, 1, 2)
    equal = standardized_fair_crps(members, truth, torch.ones(1, 1, 1, 2))
    coastal = standardized_fair_crps(
        members, truth, torch.tensor([[[[1.0, 0.25]]]])
    )
    assert coastal < equal


class _SquareVelocity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.calls = 0

    def forward(self, model_input, timesteps, return_dict=False):
        self.calls += 1
        state = model_input[:, :6]
        time = timesteps.reshape(-1, 1, 1, 1) / 1000.0
        return (self.scale * (state.square() + 0.1 * time),)


def test_rk4_reverse_interval_matches_torchdiffeq_three_eighths_rule(monkeypatch):
    monkeypatch.setattr(torch, "autocast", lambda **_: contextlib.nullcontext())
    state = torch.ones(1, 6, 2, 2)
    condition = torch.zeros(1, 48, 2, 2)
    active = torch.ones(1, 1, 2, 2)
    grid = torch.zeros(1, 2, 2, 2)
    step = -1.0 / 16.0
    def velocity(value, time):
        return value**2 + 0.1 * time
    k1 = velocity(1.0, 1 / 16)
    k2 = velocity(1.0 + step * k1 / 3.0, 1 / 16 + step / 3.0)
    k3 = velocity(1.0 + step * (k2 - k1 / 3.0), 1 / 16 + 2 * step / 3.0)
    k4 = velocity(1.0 + step * (k1 - k2 + k3), 0.0)
    expected = 1.0 + step * (k1 + 3.0 * k2 + 3.0 * k3 + k4) / 8.0
    result = rk4_interval(_SquareVelocity(), state, condition, active, grid, 1 / 16, 0.0)
    assert torch.allclose(result, torch.full_like(result, expected), atol=1e-7, rtol=0)


def test_hybrid_replays_torchdiffeq_and_gradient_is_suffix_only(monkeypatch):
    monkeypatch.setattr(torch, "autocast", lambda **_: contextlib.nullcontext())
    frozen = _SquareVelocity()
    frozen.scale.requires_grad_(False)
    candidate = _SquareVelocity()
    noise = torch.tensor(
        [[[[1.0, 9.0], [0.5, 9.0]]]]
    ).expand(1, 6, 2, 2).clone()
    condition = torch.zeros(1, 48, 2, 2)
    active = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    grid = torch.zeros(1, 2, 2, 2)
    prefix = frozen_prefix(frozen, noise, condition, active, grid)
    assert not prefix.requires_grad
    assert frozen.calls == 60
    result = hybrid_terminal_sample(candidate, prefix, condition, active, grid)
    assert candidate.calls == 4
    result.sum().backward()
    assert candidate.scale.grad is not None
    assert torch.isfinite(candidate.scale.grad) and candidate.scale.grad != 0
    assert frozen.scale.grad is None

    masked_noise = torch.where(active.expand_as(noise) > 0, noise, torch.zeros_like(noise))
    reference_model = _SquareVelocity()
    reference_model.scale.requires_grad_(False)
    def f(time, state):
        velocity = reference_model.scale * (state.square() + 0.1 * time)
        return torch.where(active.expand_as(velocity) > 0, velocity, torch.zeros_like(velocity))
    timeline = torch.linspace(1.0, 0.0, 17)
    reference = odeint(
        f,
        masked_noise,
        timeline,
        method="rk4",
        options={"step_size": 1.0 / 16.0},
    )[-1]
    assert torch.allclose(result.detach(), reference, atol=2e-6, rtol=0)


def test_launcher_pins_gpu_and_closes_lifecycle():
    source = open(
        "scripts/run_direct_dynamics_cascade_coarse_proper_refinement.sh",
        encoding="utf-8",
    ).read()
    assert 'export CUDA_VISIBLE_DEVICES="$GPU_UUID"' in source
    assert "exit.json" in source
    assert "gpu became busy" in source.lower()
    assert "1740s" in source and "--kill-after=60s" in source
