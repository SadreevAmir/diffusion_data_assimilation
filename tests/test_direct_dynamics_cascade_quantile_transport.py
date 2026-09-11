import torch

from assim_lib.direct_dynamics_cascade_quantile_transport import empirical_midrank_transport, extreme_mean_mae


def test_midrank_tie_maps_deterministically():
    source = torch.tensor([0.0, 0.0, 1.0, 2.0])
    target = torch.tensor([10.0, 20.0, 30.0, 40.0])
    result = empirical_midrank_transport(torch.tensor([0.0, 1.0, 2.0]), source, target)
    torch.testing.assert_close(result, torch.tensor([10.0, 30.0, 40.0]))


def test_transport_is_memberwise_and_monotone():
    source = torch.tensor([0.0, 1.0, 2.0, 3.0])
    target = torch.tensor([0.0, 0.0, 5.0, 8.0])
    member = torch.tensor([[[1.5]]])
    alone = empirical_midrank_transport(member, source, target)
    together = empirical_midrank_transport(torch.stack((member, member + 1), dim=1), source, target)
    assert alone.item() == together[:, 0].item()
    ordered = empirical_midrank_transport(torch.tensor([-1.0, 0.0, 1.0, 2.0, 4.0]), source, target)
    assert torch.all(ordered[1:] >= ordered[:-1])


def test_empirical_extremes_are_bounded_by_fit_truth():
    source = torch.tensor([0.0, 1.0])
    target = torch.tensor([3.0, 7.0])
    result = empirical_midrank_transport(torch.tensor([-100.0, 100.0]), source, target)
    torch.testing.assert_close(result, torch.tensor([3.0, 7.0]))


def test_empty_case_tail_is_null_and_nonempty_cases_are_aggregated():
    ensemble = torch.zeros(2, 2, 6, 1, 2)
    truth = torch.zeros(2, 6, 1, 2)
    truth[0, :, 0, 1] = 2.0
    valid = torch.ones(2, 1, 1, 2)
    thresholds = {
        0: {"sic": (0.0, 1.0), "sit": (0.0, 1.0)},
        1: {"sic": (0.0, 1.0), "sit": (0.0, 1.0)},
    }
    result = extreme_mean_mae(ensemble, truth, valid, thresholds)["d3_sic"]
    assert result["high_nonempty_case_count"] == 1
    assert result["high_per_case"] == [2.0, None]
    assert result["high_quantile_threshold_mae"] == 2.0
    assert result["low_selected_fraction_per_case"] == [0.5, 1.0]
