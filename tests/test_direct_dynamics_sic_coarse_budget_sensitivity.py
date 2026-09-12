import json
from pathlib import Path

import torch

from assim_lib.direct_dynamics_sic_coarse_budget_sensitivity import (
    _coarse_budget_vjp_statistics,
    _decode_with_shifted_budget,
    _finite_difference_gate,
    _validate_config,
)
from assim_lib.direct_dynamics_cascade import smooth_right_inverse


ROOT = Path(__file__).resolve().parents[1]


def _statistics(raw, coarse, gradient):
    return _coarse_budget_vjp_statistics(
        raw,
        coarse,
        torch.ones_like(raw),
        gradient,
        kink_margin=1e-4,
        finite_difference_epsilon=1e-6,
        finite_difference_atol=1e-10,
        finite_difference_rtol=1e-5,
    )


def test_frozen_config_is_cpu_only_validation_evidence():
    config = json.loads(
        (ROOT / "config/experiments/audit_sic_coarse_budget_sensitivity_v1.json").read_text()
    )
    _validate_config(config)
    assert config["optimizer_steps"] == 0
    assert config["sampling_performed"] is False
    assert config["test_2023"] == "closed"


def test_all_free_block_has_expected_total_vjp_and_finite_difference():
    raw = torch.tensor([[[[0.2, 0.4], [0.6, 0.8]]]], dtype=torch.float64)
    coarse = torch.full((1, 1, 1, 1), 0.5, dtype=torch.float64)
    gradient = torch.tensor([[[[1.0, 2.0], [5.0, 9.0]]]], dtype=torch.float64)
    result = _statistics(raw, coarse, gradient)
    assert result["stable_differentiable_blocks"] == 1
    assert abs(result["coarse_budget_vjp_l2_norm"] - 17.0) < 1e-12
    assert result["finite_difference"]["relative_error"] < 1e-8


def test_mixed_saturation_uses_n_over_k_budget_derivative():
    raw = torch.tensor([[[[-0.3, 0.4], [0.6, 1.3]]]], dtype=torch.float64)
    coarse = torch.full((1, 1, 1, 1), 0.5, dtype=torch.float64)
    gradient = torch.tensor([[[[1.0, 2.0], [5.0, 9.0]]]], dtype=torch.float64)
    result = _statistics(raw, coarse, gradient)
    assert result["stable_differentiable_blocks"] == 1
    assert abs(result["coarse_budget_vjp_l2_norm"] - 14.0) < 1e-10
    assert result["finite_difference"]["relative_error"] < 1e-8


def test_endpoint_is_reported_undefined_not_given_surrogate_gradient():
    raw = torch.zeros((1, 1, 2, 2), dtype=torch.float64)
    coarse = torch.zeros((1, 1, 1, 1), dtype=torch.float64)
    gradient = torch.ones_like(raw)
    result = _statistics(raw, coarse, gradient)
    assert result["stable_differentiable_blocks"] == 0
    assert result["exact_clipping_kink_blocks"] == 1
    assert result["excluded_from_restricted_probe_blocks"] == 1
    assert result["coarse_budget_vjp_l2_norm"] is None
    assert result["finite_difference"] is None


def test_shifted_budget_exactly_replays_at_zero_delta():
    raw = torch.tensor([[[[-0.3, 0.4], [0.6, 1.3]]]], dtype=torch.float64)
    coarse = torch.full((1, 1, 1, 1), 0.5, dtype=torch.float64)
    mask = torch.ones_like(raw)
    first = _decode_with_shifted_budget(raw, coarse, coarse, mask)
    second = _decode_with_shifted_budget(raw, coarse, coarse.clone(), mask)
    assert torch.equal(first, second)


def test_production_lift_contributes_on_multiblock_coastal_grid():
    coarse = torch.tensor([[[[0.45, 0.50], [0.55, 0.60]]]], dtype=torch.float64)
    mask = torch.ones((1, 1, 4, 4), dtype=torch.float64)
    mask[:, :, 0, 0] = 0.0
    raw = smooth_right_inverse(coarse, mask)
    gradient = torch.randn(
        raw.shape, generator=torch.Generator().manual_seed(29), dtype=torch.float64
    )
    result = _coarse_budget_vjp_statistics(
        raw,
        coarse,
        mask,
        gradient,
        kink_margin=1e-4,
        finite_difference_epsilon=1e-6,
        finite_difference_atol=1e-10,
        finite_difference_rtol=1e-5,
    )
    assert result["stable_differentiable_blocks"] == 4
    assert result["production_lift_vjp_l2_norm"] > 1e-6
    assert result["finite_difference"]["relative_error"] < 1e-5


def test_finite_difference_gate_rejects_corrupted_vjp():
    try:
        _finite_difference_gate(1.0, 1.1, atol=1e-10, rtol=1e-5)
    except RuntimeError as error:
        assert "failed finite-difference admission" in str(error)
    else:
        raise AssertionError("corrupted VJP was not rejected")
