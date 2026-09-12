import torch

from assim_lib.direct_dynamics_cascade import masked_block_average
from assim_lib.direct_dynamics_sic_support_decoder import (
    project_masked_blocks_to_unit_interval_mean,
    sic_support_decoder_checks,
)


def test_capped_simplex_projection_preserves_clipped_block_mean_and_support():
    fine = torch.tensor(
        [[[[ -0.4, 0.2, 0.7, 1.8], [0.3, 1.4, -0.5, 0.9]]]],
        dtype=torch.float32,
    )
    mask = torch.ones_like(fine)
    coarse = torch.tensor([[[[0.25, 0.75]]]], dtype=torch.float32)
    decoded = project_masked_blocks_to_unit_interval_mean(fine, coarse, mask)
    recovered, fraction = masked_block_average(decoded, mask)
    assert decoded.dtype == torch.float64
    assert float(decoded.min()) >= 0.0
    assert float(decoded.max()) <= 1.0
    assert torch.allclose(recovered, coarse.double(), atol=2e-14, rtol=0)
    assert torch.all(fraction > 0)


def test_decoder_clips_coarse_support_and_handles_coastal_occupancy():
    fine = torch.linspace(-2, 2, 32, dtype=torch.float32).reshape(1, 1, 4, 8)
    mask = torch.zeros_like(fine)
    coarse = torch.tensor(
        [[[[-0.2, 0.0, 0.4, 0.7], [1.2, 1.0, 0.3, 0.8]]]],
        dtype=torch.float32,
    )
    for block, count in enumerate((1, 2, 3, 4, 1, 2, 3, 4)):
        row, column = divmod(block, 4)
        for offset in range(count):
            mask[0, 0, 2 * row + offset // 2, 2 * column + offset % 2] = 1
    decoded = project_masked_blocks_to_unit_interval_mean(fine, coarse, mask)
    recovered, fraction = masked_block_average(decoded, mask)
    assert torch.allclose(recovered[fraction > 0], coarse.double().clamp(0, 1)[fraction > 0], atol=2e-14, rtol=0)
    assert torch.equal(decoded[mask == 0], torch.zeros_like(decoded[mask == 0]))
    checks = sic_support_decoder_checks(fine, coarse, mask)
    assert checks["minimum_valid_sic"] >= 0
    assert checks["maximum_valid_sic"] <= 1
    assert checks["maximum_clipped_coarse_error"] <= 2e-14


def test_decoder_is_idempotent_on_feasible_fields():
    fine = torch.tensor([[[[0.0, 0.2], [0.8, 1.0]]]], dtype=torch.float64)
    mask = torch.ones_like(fine)
    coarse, _ = masked_block_average(fine, mask)
    decoded = project_masked_blocks_to_unit_interval_mean(fine, coarse, mask)
    assert torch.allclose(decoded, fine, atol=2e-14, rtol=0)


def test_projection_is_stable_for_extreme_equal_inputs():
    fine = torch.full((1, 1, 2, 2), 1e20, dtype=torch.float64)
    mask = torch.ones_like(fine)
    coarse = torch.full((1, 1, 1, 1), 0.5, dtype=torch.float64)
    decoded = project_masked_blocks_to_unit_interval_mean(fine, coarse, mask)
    assert torch.equal(decoded, torch.full_like(decoded, 0.5))


def test_projection_matches_independent_known_kkt_solution():
    # Here theta=0: clipping alone gives [0, .2, .8, 1] and the requested
    # block sum is exactly two, so this is the unique Euclidean projection.
    fine = torch.tensor([[[[-1.0, 0.2], [0.8, 2.0]]]], dtype=torch.float64)
    mask = torch.ones_like(fine)
    coarse = torch.full((1, 1, 1, 1), 0.5, dtype=torch.float64)
    expected = torch.tensor([[[[0.0, 0.2], [0.8, 1.0]]]], dtype=torch.float64)
    decoded = project_masked_blocks_to_unit_interval_mean(fine, coarse, mask)
    assert torch.allclose(decoded, expected, atol=2e-14, rtol=0)
