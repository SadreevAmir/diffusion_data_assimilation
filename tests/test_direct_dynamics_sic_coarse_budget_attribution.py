import json

import pytest
import torch

from assim_lib import direct_dynamics_sic_coarse_budget_attribution as attribution
from assim_lib.direct_dynamics_sic_coarse_budget_attribution import (
    actual_zero_fraction,
    attribute_zero_deficit,
    fully_zero_truth_blocks,
    maximum_zero_fraction,
)


def test_maximum_zero_fraction_handles_zero_integer_and_fractional_budgets():
    coarse = torch.tensor([[[[0.0, 0.25, 0.26, 1.0]]]], dtype=torch.float64)
    count = torch.full_like(coarse, 4)
    expected = torch.tensor([[[[1.0, 0.75, 0.5, 0.0]]]], dtype=torch.float64)
    assert torch.equal(maximum_zero_fraction(coarse, count), expected)


def test_fully_zero_selector_uses_exact_atoms_and_ignores_land():
    truth = torch.zeros(1, 1, 2, 4, dtype=torch.float64)
    mask = torch.ones_like(truth)
    mask[0, 0, 0, 0] = 0
    truth[0, 0, 0, 0] = 3.0
    truth[0, 0, 0, 2] = torch.nextafter(
        torch.tensor(0.0, dtype=torch.float64), torch.tensor(1.0, dtype=torch.float64)
    )
    selector, count = fully_zero_truth_blocks(truth, mask)
    assert selector.tolist() == [[[[True, False]]]]
    assert count.tolist() == [[[[3, 4]]]]


def test_attribution_identity_separates_budget_and_allocation():
    truth = torch.zeros(1, 1, 2, 2, dtype=torch.float64)
    mask = torch.ones_like(truth)
    # Budget 1.0 across four cells permits at most 3/4 zeros.  This allocation
    # uses only 1/2 zeros, so actual nonzero=.5=.25 mandatory+.25 allocation.
    decoded = torch.tensor(
        [[[[[0.0, 0.0], [0.4, 0.6]]]]], dtype=torch.float64
    )
    coarse = torch.full((1, 1, 1, 1, 1), 0.25, dtype=torch.float64)
    result = attribute_zero_deficit(decoded, coarse, truth, mask)
    row = result["case_equal"]
    assert row["actual_zero_fraction"] == 0.5
    assert row["maximum_feasible_zero_fraction"] == 0.75
    assert row["mandatory_nonzero_from_coarse_budget"] == 0.25
    assert row["excess_nonzero_from_within_block_allocation"] == 0.25
    assert row["mandatory_share_of_actual_nonzero"] == 0.5
    assert row["allocation_share_of_actual_nonzero"] == 0.5
    assert result["maximum_identity_error"] == 0.0


def test_zero_deficit_returns_null_shares():
    truth = torch.zeros(1, 1, 2, 2, dtype=torch.float64)
    mask = torch.ones_like(truth)
    decoded = torch.zeros(1, 1, 1, 2, 2, dtype=torch.float64)
    coarse = torch.zeros(1, 1, 1, 1, 1, dtype=torch.float64)
    row = attribute_zero_deficit(decoded, coarse, truth, mask)["case_equal"]
    assert row["actual_nonzero_fraction"] == 0.0
    assert row["zero_deficit"] is True
    assert row["mandatory_share_of_actual_nonzero"] is None
    assert row["allocation_share_of_actual_nonzero"] is None


def test_case_equal_attribution_is_ocean_cell_weighted_with_coastal_blocks():
    truth = torch.zeros(2, 1, 2, 4, dtype=torch.float64)
    truth[1, 0, :, :2] = 0.5  # Exclude case 1's first block.
    mask = torch.ones_like(truth)
    mask[0, 0, :, :2] = 0
    mask[0, 0, 0, 0] = 1  # Case 0 selected blocks have 1 and 4 ocean cells.
    decoded = torch.zeros(2, 1, 1, 2, 4, dtype=torch.float64)
    decoded[0, 0, 0, 0, 0] = 1.0
    decoded[1, 0, 0, 0, 2:] = torch.tensor([0.4, 0.6])
    coarse = torch.tensor(
        [[[[[1.0, 0.0]]]], [[[[0.0, 0.25]]]]], dtype=torch.float64
    )
    result = attribute_zero_deficit(decoded, coarse, truth, mask)
    assert result["fully_zero_truth_block_count_per_case"] == [2, 1]
    assert result["fully_zero_truth_ocean_cell_count_per_case"] == [5, 4]
    # Case 0 zero fraction is 4/5; case 1 is 2/4. Equal case mean is .65.
    assert abs(result["case_equal"]["actual_zero_fraction"] - 0.65) < 1e-14


def test_actual_zero_fraction_is_memberwise_and_mask_aware():
    decoded = torch.tensor(
        [[[[[0.0, 1.0], [0.0, 1.0]]], [[[0.0, 0.0], [1.0, 0.0]]]]],
        dtype=torch.float64,
    )
    mask = torch.ones(1, 1, 2, 2, dtype=torch.float64)
    mask[0, 0, 1, 1] = 0
    result = actual_zero_fraction(decoded, mask)
    assert torch.equal(result[:, :, 0, 0, 0], torch.tensor([[2 / 3, 2 / 3]], dtype=torch.float64))


def test_source_failure_is_recorded_after_reservation(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"schema_version": "sic_coarse_budget_attribution_v1"}),
        encoding="utf-8",
    )
    output = tmp_path / "status.json"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(attribution.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(attribution.torch, "set_num_threads", lambda _value: None)
    monkeypatch.setattr(attribution.torch, "set_num_interop_threads", lambda _value: None)
    monkeypatch.setattr(attribution.torch, "get_num_threads", lambda: 6)
    monkeypatch.setattr(attribution.torch, "get_num_interop_threads", lambda: 1)

    def fail_source(_config, _repo):
        raise ValueError("injected source attribution failure")

    monkeypatch.setattr(attribution, "_load_bound_source", fail_source)
    with pytest.raises(ValueError, match="injected source attribution failure"):
        attribution.run(config_path, output)
    status = json.loads(output.read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    assert status["error_type"] == "ValueError"
    assert status["error"] == "injected source attribution failure"
