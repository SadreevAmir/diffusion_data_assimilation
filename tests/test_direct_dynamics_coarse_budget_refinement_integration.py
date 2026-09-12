import json
from pathlib import Path

import torch
from torchdiffeq import odeint

from assim_lib.direct_dynamics_cascade_coarse_proper_refinement import (
    _velocity,
    frozen_prefix,
    hybrid_terminal_sample,
)
from assim_lib.direct_dynamics_coarse_budget_refinement_integration import (
    _CompactTerminalModel,
    _validate_config,
    anchored_canonical_physical_coarse,
    compact_coarse_budget_integration_check,
    differentiable_frozen_allocation_projection,
)
from assim_lib.runtime import make_normalized_xy_grid


ROOT = Path(__file__).resolve().parents[1]


def _normalization():
    means = torch.tensor([0.31, 0.42, 0.27, 0.63, 0.38, 0.81], dtype=torch.float32)
    stds = torch.tensor([0.22, 0.35, 0.19, 0.44, 0.24, 0.51], dtype=torch.float32)
    return means, stds


def test_anchored_physical_chart_restores_sic_atoms_and_replays_step_zero():
    means, stds = _normalization()
    zero = -means / stds
    one = (torch.ones_like(means) - means) / stds
    base = torch.stack((zero, one), dim=-1).reshape(1, 6, 1, 2)
    candidate = base.clone().requires_grad_(True)
    base_physical, candidate_physical = anchored_canonical_physical_coarse(
        base, candidate, means, stds
    )
    assert torch.equal(base_physical, candidate_physical.detach())
    for channel in (0, 2, 4):
        assert float(base_physical[0, channel, 0, 0]) == 0.0
        assert float(base_physical[0, channel, 0, 1]) == 1.0


def test_anchored_physical_chart_has_declared_affine_derivative_at_atoms():
    means, stds = _normalization()
    encoded_zero = -means / stds
    base = encoded_zero.reshape(1, 6, 1, 1)
    candidate = base.clone().requires_grad_(True)
    _, physical = anchored_canonical_physical_coarse(base, candidate, means, stds)
    weights = torch.arange(1, 7, dtype=torch.float64).reshape(1, 6, 1, 1)
    (physical * weights).sum().backward()
    expected = weights.float() * stds.reshape(1, 6, 1, 1)
    assert torch.equal(candidate.grad, expected)


def test_anchored_physical_chart_tracks_both_float32_nextafter_directions():
    means, stds = _normalization()
    encoded_one = (torch.ones_like(means) - means) / stds
    base = encoded_one.reshape(1, 6, 1, 1)
    for direction in (-torch.inf, torch.inf):
        candidate = torch.nextafter(base, torch.full_like(base, direction))
        base_physical, candidate_physical = anchored_canonical_physical_coarse(
            base, candidate, means, stds
        )
        expected = base_physical + (
            candidate.double() - base.double()
        ) * stds.double().reshape(1, 6, 1, 1)
        assert torch.equal(candidate_physical, expected)


def test_custom_rk4_path_bitwise_replays_torchdiffeq_production_order():
    torch.manual_seed(31)
    model = _CompactTerminalModel().eval()
    state = torch.randn((2, 6, 4, 4), dtype=torch.float32)
    condition = torch.randn((2, 6, 4, 4), dtype=torch.float32)
    active = torch.ones((2, 1, 4, 4), dtype=torch.float32)
    grid = make_normalized_xy_grid(
        4, 4, device=torch.device("cpu"), dtype=torch.float32
    )
    times = torch.linspace(1.0, 0.0, 17, dtype=torch.float32)

    def function(time, value):
        return _velocity(model, value, condition, active, grid, time)

    production = odeint(
        function,
        state,
        times,
        method="rk4",
        options={"step_size": 1.0 / 16.0},
    )[-1]
    prefix = frozen_prefix(model, state, condition, active, grid)
    custom = hybrid_terminal_sample(model, prefix, condition, active, grid)
    assert torch.equal(custom, production)


def test_frozen_allocation_projection_directional_difference_away_from_kinks():
    base_fine = torch.tensor(
        [[[[0.18, 0.32, 0.41, 0.59],
           [0.28, 0.42, 0.51, 0.69],
           [0.31, 0.49, 0.58, 0.72],
           [0.41, 0.59, 0.68, 0.82]]]],
        dtype=torch.float64,
    )
    base_coarse = torch.tensor([[[[0.30, 0.55], [0.45, 0.70]]]], dtype=torch.float64)
    candidate = base_coarse.clone().requires_grad_(True)
    mask = torch.ones_like(base_fine)
    direction = torch.tensor([[[[0.3, -0.4], [0.2, 0.1]]]], dtype=torch.float64)
    weight = torch.arange(1, 17, dtype=torch.float64).reshape_as(base_fine)
    decoded = differentiable_frozen_allocation_projection(
        base_fine, base_coarse, candidate, mask
    )
    (decoded * weight).sum().backward()
    predicted = float((candidate.grad * direction).sum())
    epsilon = 1e-6
    plus = differentiable_frozen_allocation_projection(
        base_fine, base_coarse, base_coarse + epsilon * direction, mask
    )
    minus = differentiable_frozen_allocation_projection(
        base_fine, base_coarse, base_coarse - epsilon * direction, mask
    )
    observed = float(((plus - minus) * weight).sum() / (2 * epsilon))
    assert abs(predicted - observed) <= 2e-6 + 2e-5 * abs(predicted)


def test_clip_outside_support_is_locally_flat():
    base_fine = torch.tensor([[[[0.2, 0.4], [0.6, 0.8]]]], dtype=torch.float64)
    base_coarse = torch.full((1, 1, 1, 1), 0.5, dtype=torch.float64)
    candidate = torch.tensor([[[[-0.2]]]], dtype=torch.float64, requires_grad=True)
    mask = torch.ones_like(base_fine)
    decoded = differentiable_frozen_allocation_projection(
        base_fine, base_coarse, candidate, mask
    )
    decoded.sum().backward()
    assert torch.equal(candidate.grad, torch.zeros_like(candidate.grad))


def test_endpoint_generalized_gradient_can_move_atom_inward():
    base_fine = torch.tensor([[[[0.1, 0.4], [0.7, 0.9]]]], dtype=torch.float64)
    base_coarse = torch.zeros((1, 1, 1, 1), dtype=torch.float64)
    candidate = base_coarse.clone().requires_grad_(True)
    mask = torch.ones_like(base_fine)
    decoded = differentiable_frozen_allocation_projection(
        base_fine, base_coarse, candidate, mask
    )
    decoded[0, 0, 1, 1].backward()
    assert float(candidate.grad) == 4.0


def test_interior_kkt_kink_fails_closed_instead_of_silent_surrogate():
    base_fine = torch.tensor([[[[0.0, 0.5], [0.5, 1.0]]]], dtype=torch.float64)
    base_coarse = torch.full((1, 1, 1, 1), 0.5, dtype=torch.float64)
    candidate = base_coarse.clone().requires_grad_(True)
    mask = torch.ones_like(base_fine)
    try:
        differentiable_frozen_allocation_projection(
            base_fine, base_coarse, candidate, mask
        )
    except RuntimeError as error:
        assert "undefined at an exact interior KKT kink" in str(error)
    else:
        raise AssertionError("exact interior KKT kink did not fail closed")


def test_endpoint_tie_and_coast_match_declared_one_sided_derivatives():
    base_fine = torch.tensor([[[[0.9, 0.9], [0.2, 0.0]]]], dtype=torch.float64)
    mask = torch.tensor([[[[1.0, 1.0], [1.0, 0.0]]]], dtype=torch.float64)
    weight = torch.tensor([[[[2.0, 5.0], [1.0, 0.0]]]], dtype=torch.float64)
    epsilon = 1e-7

    zero = torch.zeros((1, 1, 1, 1), dtype=torch.float64, requires_grad=True)
    at_zero = differentiable_frozen_allocation_projection(
        base_fine, zero.detach(), zero, mask
    )
    (at_zero * weight).sum().backward()
    plus = differentiable_frozen_allocation_projection(
        base_fine, zero.detach(), zero.detach() + epsilon, mask
    )
    right = float(((plus - at_zero.detach()) * weight).sum() / epsilon)
    assert abs(float(zero.grad) - right) < 1e-6

    one = torch.ones((1, 1, 1, 1), dtype=torch.float64, requires_grad=True)
    at_one = differentiable_frozen_allocation_projection(
        base_fine, one.detach(), one, mask
    )
    (at_one * weight).sum().backward()
    minus = differentiable_frozen_allocation_projection(
        base_fine, one.detach(), one.detach() - epsilon, mask
    )
    left = float(((at_one.detach() - minus) * weight).sum() / epsilon)
    assert abs(float(one.grad) - left) < 1e-6


def test_compact_terminal_path_replays_and_reaches_parameters():
    result = compact_coarse_budget_integration_check()
    assert result["status"] == "pass"
    assert result["candidate_control_max_abs"] <= 1e-7
    assert result["step0_replay_max_abs"] == 0.0
    assert result["terminal_parameter_gradient_norm"] > 0
    assert result["frozen_prefix_has_no_graph"]
    assert result["frozen_residual_has_no_graph"]
    assert result["frozen_parameter_gradients_absent"]
    assert result["loss_domain"].startswith("all valid ocean")


def test_config_freezes_allocation_and_test_2023():
    config = json.loads(
        (ROOT / "config/experiments/admit_direct_dynamics_coarse_budget_refinement_cpu_v1.json").read_text()
    )
    _validate_config(config)
    assert config["optimizer_steps"] == 0
    assert config["sampling_performed"] is False
    assert config["test_2023"] == "closed"
