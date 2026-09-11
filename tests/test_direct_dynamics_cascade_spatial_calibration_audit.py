import torch

from assim_lib.direct_dynamics_cascade_spatial_calibration_audit import (
    _case_ocean_patch_cross_power,
    _case_ocean_patch_power,
    component_statistics,
    cross_statistics,
)


def test_component_statistics_are_date_blocked_and_finite():
    torch.manual_seed(4)
    members = torch.randn(2, 8, 2, 32, 32)
    truth = torch.randn(2, 2, 32, 32)
    valid = torch.ones(2, 1, 32, 32)
    result = component_statistics(members, truth, valid)
    assert len(result["regions"]["ocean"]["case_channel_ensemble_variance"]) == 2
    power = torch.tensor(result["fully_ocean_patch_power_low_mid_high"]["ensemble_anomaly"])
    assert power.shape == (2, 2, 3)
    assert torch.isfinite(power).all()


def test_zero_residual_has_zero_cross_statistics():
    torch.manual_seed(5)
    coarse = torch.randn(2, 8, 2, 32, 32)
    residual = torch.zeros_like(coarse)
    valid = torch.ones(2, 1, 32, 32)
    result = cross_statistics(coarse, residual, valid)
    covariance = torch.tensor(result["regions"]["ocean"]["case_channel_cross_covariance"])
    cross_power = torch.tensor(result["fully_ocean_patch_cross_power_low_mid_high"])
    assert torch.equal(covariance, torch.zeros_like(covariance))
    assert torch.equal(cross_power, torch.zeros_like(cross_power))


def test_patch_power_rejects_shape_mismatch():
    value = torch.zeros(1, 8, 2, 32, 32)
    valid = torch.ones(1, 1, 31, 32)
    try:
        _case_ocean_patch_power(value, valid)
    except ValueError:
        pass
    else:
        raise AssertionError("shape mismatch must fail")


def test_cross_power_rejects_shape_mismatch():
    left = torch.zeros(1, 8, 2, 32, 32)
    right = torch.zeros(1, 8, 3, 32, 32)
    valid = torch.ones(1, 1, 32, 32)
    try:
        _case_ocean_patch_cross_power(left, right, valid)
    except ValueError:
        pass
    else:
        raise AssertionError("shape mismatch must fail")
