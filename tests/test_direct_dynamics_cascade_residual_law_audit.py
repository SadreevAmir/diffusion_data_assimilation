import unittest

import torch

from assim_lib.direct_dynamics_cascade import masked_block_average, project_detail
from assim_lib.direct_dynamics_cascade_residual_law_audit import (
    CHANNELS,
    SEASONAL_MONTH_DAYS,
    TRAIN_YEARS,
    _directional_increment_square,
    _fully_ocean_patch_spectrum,
    _phase_second_moment,
    _region_masks,
)


class ResidualLawAuditTests(unittest.TestCase):
    def test_protocol_is_frozen_and_has_six_channels(self) -> None:
        self.assertEqual(len(CHANNELS), 6)
        self.assertEqual(TRAIN_YEARS, tuple(range(2016, 2022)))
        self.assertEqual(SEASONAL_MONTH_DAYS, ((1, 15), (4, 15), (7, 15), (10, 15)))

    def test_projected_base_has_zero_block_average(self) -> None:
        generator = torch.Generator().manual_seed(7)
        valid = torch.ones((3, 1, 16, 16))
        base = project_detail(torch.randn((3, 6, 16, 16), generator=generator), valid)
        coarse, fraction = masked_block_average(base, valid)
        self.assertLess(float(coarse[fraction.expand_as(coarse) > 0].abs().max()), 2e-6)

    def test_regions_partition_valid_ocean(self) -> None:
        valid = torch.ones((1, 1, 8, 8))
        valid[..., 3, 3] = 0
        interior, coast = _region_masks(valid)
        self.assertFalse(torch.any(interior.bool() & coast.bool()))
        self.assertTrue(torch.equal((interior + coast).bool(), valid.bool()))

    def test_directional_increment_detects_x_ramp(self) -> None:
        value = torch.arange(8.0)[None, None, None, :].expand(2, 6, 8, 8)
        valid = torch.ones((2, 1, 8, 8))
        dx, dy = _directional_increment_square(value, valid)
        self.assertTrue(torch.allclose(dx, torch.ones_like(dx)))
        self.assertTrue(torch.allclose(dy, torch.zeros_like(dy)))

    def test_phase_metric_keeps_phase_information(self) -> None:
        value = torch.zeros((1, 6, 8, 8))
        value[..., 0::2, 0::2] = 2
        result = _phase_second_moment(value, torch.ones((1, 1, 8, 8)), 2)
        self.assertTrue(torch.all(result[:, :, 0, 0] == 4))
        self.assertTrue(torch.all(result[:, :, 1, 1] == 0))

    def test_patch_spectrum_excludes_non_ocean_patches(self) -> None:
        generator = torch.Generator().manual_seed(9)
        value = torch.randn((1, 6, 64, 64), generator=generator)
        valid = torch.ones((1, 1, 64, 64))
        valid[..., :32, :32] = 0
        sums, counts = _fully_ocean_patch_spectrum(value, valid, patch_size=32)
        self.assertEqual(tuple(sums.shape), (6, 3))
        self.assertEqual(counts.tolist(), [3, 3, 3])


if __name__ == "__main__":
    unittest.main()
