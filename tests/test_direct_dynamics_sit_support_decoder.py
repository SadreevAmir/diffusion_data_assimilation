import torch

from assim_lib.direct_dynamics_cascade import masked_block_average
from assim_lib.direct_dynamics_sit_support_decoder import (
    project_masked_blocks_to_nonnegative_mean,
    support_decoder_checks,
)


def _decode(fine, mask):
    coarse, _ = masked_block_average(fine, mask)
    return project_masked_blocks_to_nonnegative_mean(fine, coarse, mask), coarse


def test_analytic_positive_mixed_sign_projection():
    fine = torch.tensor([[[[-1.0, 1.0], [2.0, 0.0]]]])
    mask = torch.ones_like(fine)
    decoded, coarse = _decode(fine, mask)
    expected = torch.tensor([[[[0.0, 0.5], [1.5, 0.0]]]])
    assert torch.allclose(decoded, expected, rtol=0, atol=1e-7)
    checks = support_decoder_checks(fine, coarse, mask)
    assert checks["minimum_valid_value"] == 0.0
    assert checks["maximum_coarse_error"] <= checks["fp32_source_tolerance"]


def test_negative_coarse_block_becomes_zero():
    fine = torch.tensor([[[[-4.0, -3.0], [-2.0, 1.0]]]])
    mask = torch.ones_like(fine)
    decoded, coarse = _decode(fine, mask)
    assert float(coarse) < 0
    assert torch.equal(decoded, torch.zeros_like(decoded))


def test_identity_on_admissible_field():
    fine = torch.tensor([[[[0.0, 0.25], [1.25, 3.0]]]])
    mask = torch.ones_like(fine)
    decoded, _ = _decode(fine, mask)
    assert torch.allclose(decoded, fine, rtol=0, atol=2e-7)


def test_one_two_three_and_four_cell_coastal_blocks():
    for count in (1, 2, 3, 4):
        fine = torch.tensor([[[[-1.0, 0.5], [2.0, 3.0]]]])
        mask = torch.zeros_like(fine)
        mask.view(-1)[:count] = 1
        decoded, coarse = _decode(fine, mask)
        recovered, fraction = masked_block_average(decoded, mask)
        assert torch.all(decoded[mask > 0] >= 0)
        assert torch.equal(decoded[mask == 0], torch.zeros_like(decoded[mask == 0]))
        assert torch.allclose(recovered, coarse.clamp_min(0), rtol=0, atol=2e-7)
        assert float(fraction) == count / 4


def test_positive_active_example_has_finite_nonzero_gradients_without_ste():
    fine = torch.tensor(
        [[[[-1.25, 0.25], [1.0, 3.0]]]], dtype=torch.float64, requires_grad=True
    )
    mask = torch.ones_like(fine)
    coarse, _ = masked_block_average(fine, mask)
    decoded = project_masked_blocks_to_nonnegative_mean(fine, coarse, mask)
    weights = torch.tensor([[[[0.5, 1.0], [2.0, 4.0]]]], dtype=torch.float64)
    gradient = torch.autograd.grad((decoded * weights).sum(), fine)[0]
    assert torch.isfinite(gradient).all()
    assert float(gradient.abs().sum()) > 0


def test_fully_negative_block_has_zero_local_gradient():
    fine = torch.tensor(
        [[[[-4.0, -3.0], [-2.0, -1.0]]]], dtype=torch.float64, requires_grad=True
    )
    mask = torch.ones_like(fine)
    coarse, _ = masked_block_average(fine, mask)
    decoded = project_masked_blocks_to_nonnegative_mean(fine, coarse, mask)
    gradient = torch.autograd.grad(decoded.sum(), fine)[0]
    assert torch.isfinite(gradient).all()
    assert torch.equal(gradient, torch.zeros_like(gradient))


def test_half_input_keeps_promoted_output_when_solution_exceeds_half_range():
    fine = torch.tensor(
        [[[[65504.0, -65504.0], [-65504.0, -65504.0]]]], dtype=torch.float16
    )
    coarse = torch.tensor([[[[60000.0]]]], dtype=torch.float16)
    mask = torch.ones_like(fine)
    decoded = project_masked_blocks_to_nonnegative_mean(fine, coarse, mask)
    assert decoded.dtype == torch.float32
    assert torch.isfinite(decoded).all()
    assert float(decoded.max()) > torch.finfo(torch.float16).max
    recovered, _ = masked_block_average(decoded, mask)
    assert torch.allclose(recovered, coarse.float(), rtol=0, atol=0.02)


def test_large_common_offset_preserves_small_positive_budget():
    fine = torch.full((1, 1, 2, 2), 1e8, dtype=torch.float32)
    coarse = torch.ones((1, 1, 1, 1), dtype=torch.float32)
    mask = torch.ones_like(fine)
    decoded = project_masked_blocks_to_nonnegative_mean(fine, coarse, mask)
    assert torch.equal(decoded, torch.ones_like(decoded))
    recovered, _ = masked_block_average(decoded, mask)
    assert torch.equal(recovered, coarse)


def test_finite_input_target_sum_overflow_fails_closed():
    maximum = torch.finfo(torch.float32).max
    fine = torch.full((1, 1, 2, 2), maximum, dtype=torch.float32)
    coarse = torch.full((1, 1, 1, 1), maximum, dtype=torch.float32)
    mask = torch.ones_like(fine)
    try:
        project_masked_blocks_to_nonnegative_mean(fine, coarse, mask)
    except FloatingPointError as error:
        assert "target sum overflowed" in str(error)
    else:
        raise AssertionError("finite-input target overflow was accepted")
