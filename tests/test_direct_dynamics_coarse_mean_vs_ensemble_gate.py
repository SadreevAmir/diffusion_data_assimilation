import torch

from assim_lib.direct_dynamics_coarse_mean_vs_ensemble_gate import _comparison


def test_mc_adjustment_uses_unbiased_variance_over_member_count():
    truth = torch.zeros(12, 1, 32, 32)
    persistence = torch.zeros_like(truth)
    deterministic = torch.ones_like(truth)
    members = torch.arange(8, dtype=torch.float32)[None, :, None, None, None].expand(12, -1, 1, 32, 32)
    active = torch.ones(12, 1, 32, 32)
    result = _comparison(deterministic, members, truth, persistence, active, active)
    row = result["regions"]["ocean"]
    expected_mse = 3.5**2
    unbiased_variance = torch.arange(8, dtype=torch.float64).var(unbiased=True).item()
    assert abs(row["ensemble_mean_case_mse"][0][0] - expected_mse) < 1e-12
    assert abs(row["ensemble_predictive_mean_adjusted_case_mse"][0][0] - (expected_mse - unbiased_variance / 8)) < 1e-12
    assert abs(
        row["joint_channel_mean_deterministic_minus_adjusted_ensemble"]["estimate"]
        - (1.0 - (expected_mse - unbiased_variance / 8))
    ) < 1e-12
