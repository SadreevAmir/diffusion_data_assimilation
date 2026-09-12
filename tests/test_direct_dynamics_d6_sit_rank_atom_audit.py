import torch

from assim_lib.direct_dynamics_d6_sit_rank_atom_audit import _conditional_distribution


def test_zero_positive_rank_mixture_reconstructs_direct_case_histogram():
    members = torch.tensor(
        [[[[[-0.1, 0.0], [0.005, 0.02]]], [[[0.0, 0.01], [0.02, 0.03]]]]],
        dtype=torch.float64,
    )
    truth = torch.tensor([[[[0.0, 0.0], [0.01, 0.02]]]], dtype=torch.float64)
    result = _conditional_distribution(members, truth, torch.ones_like(truth), torch.tensor(0.0), torch.tensor(1.0))
    assert result["rank_mixture_reconstruction_max_abs"] < 2e-15
    assert result["groups"]["exact_zero_truth"]["case_count"] == 1
    assert result["groups"]["positive_truth"]["case_count"] == 1
    for group in result["groups"].values():
        probabilities = group["per_case"][0]["member_value_probabilities"]
        assert abs(sum(probabilities.values()) - 1.0) < 1e-14
        assert "rank_tv" not in group


def test_negative_truth_fails_closed():
    members = torch.zeros((1, 2, 1, 1, 1), dtype=torch.float64)
    truth = torch.full((1, 1, 1, 1), -0.1, dtype=torch.float64)
    try:
        _conditional_distribution(members, truth, torch.ones_like(truth), torch.tensor(0.0), torch.tensor(1.0))
    except ValueError as error:
        assert "negative" in str(error)
    else:
        raise AssertionError("negative d6 SIT truth was accepted")


def test_actual_encoded_zero_nextafter_categories_and_tie_are_exact():
    mean = torch.tensor(0.18969311571248842, dtype=torch.float32)
    std = torch.tensor(0.4359687842884708, dtype=torch.float32)
    zero = (torch.tensor(0.0, dtype=torch.float32) - mean) / std
    below = torch.nextafter(zero, torch.tensor(float("-inf"), dtype=torch.float32))
    above = torch.nextafter(zero, torch.tensor(float("inf"), dtype=torch.float32))
    members = torch.stack((below, zero, above)).reshape(1, 3, 1, 1, 1)
    truth = zero.reshape(1, 1, 1, 1)
    result = _conditional_distribution(members, truth, torch.ones_like(truth), mean, std)
    probabilities = result["groups"]["exact_zero_truth"]["per_case"][0]["member_value_probabilities"]
    assert probabilities == {"lt_0": 1 / 3, "eq_0": 1 / 3, "gt_0_le_0p01": 1 / 3, "gt_0p01": 0.0}
    ranks = result["per_case_direct_rank_frequencies"][0]
    assert ranks == [0.0, 0.5, 0.5, 0.0]
