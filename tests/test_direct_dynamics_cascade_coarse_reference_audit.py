import torch

from assim_lib.direct_dynamics_cascade_coarse_reference_audit import (
    _case_channel_covariance,
    _date_blocks,
    _weighted_region_masks,
)


def test_covariance_removes_case_channel_dc():
    torch.manual_seed(8)
    value = torch.randn(2, 6, 8, 8)
    weight = torch.ones(2, 1, 8, 8)
    shifted = value + torch.arange(6)[None, :, None, None]
    assert torch.allclose(
        _case_channel_covariance(value, weight),
        _case_channel_covariance(shifted, weight),
        atol=1e-10,
    )


def test_weighted_regions_partition_active_support():
    active = torch.ones(1, 1, 8, 8)
    fraction = torch.full_like(active, 0.75)
    regions = _weighted_region_masks(active, fraction)
    assert torch.equal(regions["interior"] + regions["coast"], regions["ocean"])


def test_date_blocks_average_moments_after_slice_measurement():
    value = torch.arange(24 * 24 * 2, dtype=torch.float64).reshape(24 * 24, 2)
    blocked = _date_blocks(value)
    assert blocked.shape == (24, 2)
    assert torch.equal(blocked[0], value[:24].mean(dim=0))
