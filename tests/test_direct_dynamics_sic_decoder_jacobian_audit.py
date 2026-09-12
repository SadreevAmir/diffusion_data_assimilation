import json
from pathlib import Path

import torch

from assim_lib.direct_dynamics_sic_decoder_jacobian_audit import (
    _jacobian_statistics,
    _validate_config,
)
from assim_lib.direct_dynamics_fine_support_proper_admission import (
    differentiable_fixed_budget_projection,
)
from assim_lib.direct_dynamics_sic_support_decoder_scoring import (
    canonical_physical_decode,
)


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_config_is_cpu_only_validation_evidence():
    config = json.loads(
        (ROOT / "config/experiments/audit_sic_decoder_jacobian_v1.json").read_text()
    )
    _validate_config(config)
    assert config["optimizer_steps"] == 0
    assert config["sampling_performed"] is False
    assert config["test_2023"] == "closed"


def test_full_free_block_has_rank_three_and_removes_mean_gradient():
    raw = torch.tensor([[[[[0.2, 0.4], [0.6, 0.8]]]]], dtype=torch.float64)
    coarse = torch.full((1, 1, 1, 1, 1), 0.5, dtype=torch.float64)
    mask = torch.ones(1, 1, 2, 2)
    gradient = torch.ones_like(raw)
    result = _jacobian_statistics(raw, coarse, mask, gradient)
    assert result["jacobian_rank_sum"] == 3
    assert result["free_count_histogram"] == {"0": 0, "1": 0, "2": 0, "3": 0, "4": 1}
    assert result["proper_gradient_energy_fraction_retained"] == 0.0


def test_zero_budget_block_has_zero_rank_and_zero_retained_gradient():
    raw = torch.randn(1, 1, 1, 2, 2, generator=torch.Generator().manual_seed(7))
    coarse = torch.zeros(1, 1, 1, 1, 1)
    mask = torch.ones(1, 1, 2, 2)
    gradient = torch.randn(raw.shape, generator=torch.Generator().manual_seed(9))
    result = _jacobian_statistics(raw, coarse, mask, gradient)
    assert result["jacobian_rank_sum"] == 0
    assert result["proper_gradient_energy_fraction_retained"] == 0.0


def test_canonical_decode_preserves_atoms_but_not_nextafter_values():
    means = torch.tensor([0.2, 0.3] * 3)
    stds = torch.tensor([0.4, 0.5] * 3)
    normalized = torch.zeros(1, 1, 6, 1, 2)
    encoded_zero = -means[0] / stds[0]
    encoded_one = (torch.tensor(1.0) - means[0]) / stds[0]
    normalized[:, :, 0, 0, 0] = encoded_zero
    normalized[:, :, 0, 0, 1] = torch.nextafter(encoded_one, torch.tensor(float("inf")))
    physical = canonical_physical_decode(normalized, means, stds)
    assert physical[0, 0, 0, 0, 0] == 0.0
    assert physical[0, 0, 0, 0, 1] != 1.0


def test_mixed_block_manual_jg_matches_reviewed_backward():
    raw = torch.tensor([[[[[-0.3, 0.4], [0.6, 1.3]]]]], dtype=torch.float64)
    coarse = torch.full((1, 1, 1, 1, 1), 0.5, dtype=torch.float64)
    mask = torch.ones(1, 1, 2, 2)
    gradient = torch.tensor([[[[[1.0, 2.0], [5.0, 9.0]]]]], dtype=torch.float64)
    result = _jacobian_statistics(raw, coarse, mask, gradient)
    leaf = raw.flatten(0, 1).detach().requires_grad_(True)
    decoded = differentiable_fixed_budget_projection(
        leaf, coarse.flatten(0, 1), mask, 2
    )
    (decoded * gradient.flatten(0, 1)).sum().backward()
    expected_fraction = float(leaf.grad.square().sum() / gradient.square().sum())
    assert result["jacobian_rank_sum"] == 1
    assert result["proper_gradient_energy_fraction_retained"] == expected_fraction
