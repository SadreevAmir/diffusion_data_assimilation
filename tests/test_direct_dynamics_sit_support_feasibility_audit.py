import torch

from assim_lib.direct_dynamics_sit_support_feasibility_audit import (
    _case_equal_summary,
    support_decomposition,
)


def _probe(values: torch.Tensor, source: float):
    fine = values.reshape(1, 1, 2, 2)
    mask = torch.ones_like(fine)
    coarse = torch.tensor([[[[source]]]], dtype=torch.float32)
    return support_decomposition(fine, mask, coarse, fp32_ulps=128)


def test_negative_coarse_requires_unavoidable_coarse_change():
    result = _probe(torch.tensor([-2.0, -1.0, 0.0, 1.0]), -0.5)
    assert torch.allclose(result["coarse"], torch.tensor([[[[-0.5]]]], dtype=torch.float64))
    assert torch.allclose(result["q"], torch.tensor([[[[0.75]]]], dtype=torch.float64))
    assert torch.allclose(result["unavoidable"], torch.tensor([[[[0.5]]]], dtype=torch.float64))
    assert torch.allclose(result["mixed_sign"], torch.tensor([[[[0.25]]]], dtype=torch.float64))
    assert bool(result["negative_coarse"].item())


def test_nonnegative_coarse_can_still_require_mixed_sign_change():
    result = _probe(torch.tensor([-1.0, 0.0, 1.0, 2.0]), 0.5)
    assert torch.allclose(result["q"], torch.tensor([[[[0.25]]]], dtype=torch.float64))
    assert torch.equal(result["unavoidable"], torch.zeros_like(result["unavoidable"]))
    assert torch.allclose(result["mixed_sign"], result["q"])
    assert not bool(result["negative_coarse"].item())


def test_already_nonnegative_block_needs_no_change():
    result = _probe(torch.tensor([0.0, 1.0, 2.0, 3.0]), 1.5)
    assert torch.equal(result["q"], torch.zeros_like(result["q"]))
    assert torch.equal(result["unavoidable"], torch.zeros_like(result["unavoidable"]))
    assert torch.equal(result["mixed_sign"], torch.zeros_like(result["mixed_sign"]))


def test_source_coarse_mismatch_fails():
    try:
        _probe(torch.tensor([0.0, 1.0, 2.0, 3.0]), 1.0)
    except RuntimeError as error:
        assert "source_coarse" in str(error)
    else:
        raise AssertionError("source coarse mismatch was accepted")


def test_case_equal_summary_handles_multiple_cases_members_and_coastal_weights():
    # [case=2, member=2, channel=1, y=1, x=2], flattened over case/member.
    fraction = torch.tensor(
        [
            [[[1.0, 0.5]]],
            [[[1.0, 0.5]]],
            [[[0.25, 1.0]]],
            [[[0.25, 1.0]]],
        ],
        dtype=torch.float64,
    )
    negative = torch.tensor(
        [
            [[[True, False]]],
            [[[False, False]]],
            [[[True, False]]],
            [[[True, True]]],
        ]
    )
    q = torch.tensor(
        [
            [[[2.0, 4.0]]],
            [[[0.0, 2.0]]],
            [[[1.0, 3.0]]],
            [[[5.0, 1.0]]],
        ],
        dtype=torch.float64,
    )
    decomposition = {
        "fraction": fraction,
        "negative_coarse": negative,
        "q": q,
        "unavoidable": q * 0.25,
        "mixed_sign": q * 0.75,
        "tolerance_metres": 1e-5,
        "max_source_coarse_error_metres": 0.0,
        "max_projection_identity_error_metres": 0.0,
        "max_decomposition_identity_error_metres": 0.0,
        "minimum_mixed_sign_metres": 0.0,
    }
    result = _case_equal_summary(decomposition, cases=2, members=2)
    # Case 0: 1/4 active blocks negative; weighted mass 1 / 3.
    # Case 1: 3/4 active blocks negative; weighted mass 1.5 / 2.5.
    assert result["per_case_negative_coarse_block_fraction"] == [0.25, 0.75]
    assert torch.allclose(
        torch.tensor(result["per_case_negative_coarse_ocean_fraction_weighted"]),
        torch.tensor([1.0 / 3.0, 0.6]),
    )
    # q case means: (2+2+0+1)/3=5/3, (0.25+3+1.25+1)/2.5=2.2.
    assert torch.allclose(
        torch.tensor(result["per_case_q_metres"]), torch.tensor([5.0 / 3.0, 2.2])
    )
    assert abs(result["q_metres_case_equal"] - (5.0 / 3.0 + 2.2) / 2) < 1e-12
    assert abs(result["unavoidable_fraction_of_q"] - 0.25) < 1e-12
    assert abs(result["mixed_sign_fraction_of_q"] - 0.75) < 1e-12
