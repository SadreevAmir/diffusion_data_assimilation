import unittest

import torch

from assim_lib.direct_dynamics_cascade import masked_block_average, project_detail
from assim_lib.direct_dynamics_cascade_fine_colored import (
    ColoredVariancePreconditionedFineCascadeSampler,
    binomial_blur,
    colored_projected_gaussian,
)


class ZeroModel(torch.nn.Module):
    def forward(self, model_input, timestep, return_dict=False):
        del timestep, return_dict
        return (torch.zeros_like(model_input[:, :6]),)


class ColoredProjectedGaussianTests(unittest.TestCase):
    def test_binomial_blur_preserves_constants(self):
        value = torch.full((2, 6, 12, 10), 3.25)
        self.assertTrue(torch.equal(binomial_blur(value), value))

    def test_blend_zero_matches_scaled_projected_white_noise(self):
        generator = torch.Generator().manual_seed(31)
        white = torch.randn((2, 6, 12, 10), generator=generator)
        mask = torch.ones((2, 1, 12, 10))
        scales = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
        actual = colored_projected_gaussian(
            white, mask, blend=0.0, channel_scales=scales
        )
        expected = project_detail(white, mask) * torch.tensor(scales).view(1, 6, 1, 1)
        self.assertTrue(torch.allclose(actual, expected, atol=2e-6, rtol=0))

    def test_colored_base_remains_in_detail_nullspace_with_coast(self):
        generator = torch.Generator().manual_seed(47)
        white = torch.randn((3, 6, 16, 14), generator=generator)
        mask = torch.ones((3, 1, 16, 14))
        mask[:, :, :3, :4] = 0
        result = colored_projected_gaussian(
            white,
            mask,
            blend=0.7,
            channel_scales=(0.07,) * 6,
        )
        coarse, fraction = masked_block_average(result, mask)
        active = fraction > 0
        self.assertLessEqual(float(coarse[active.expand_as(coarse)].abs().max()), 3e-6)
        self.assertTrue(torch.equal(result[mask.expand_as(result) == 0], torch.zeros_like(result)[mask.expand_as(result) == 0]))

    def test_invalid_blend_fails_closed(self):
        white = torch.zeros((1, 6, 8, 8))
        mask = torch.ones((1, 1, 8, 8))
        for blend in (-0.01, 1.0, float("nan")):
            with self.assertRaises(ValueError):
                colored_projected_gaussian(
                    white, mask, blend=blend, channel_scales=(1.0,) * 6
                )

    def test_sampler_exposes_exact_ode_initial_state(self):
        sampler = ColoredVariancePreconditionedFineCascadeSampler(
            ZeroModel(),
            target_residual_rms=(1.0,) * 6,
            projected_base_rms=(1.0,) * 6,
            blend=0.75,
            channel_scales=(0.2,) * 6,
        )
        generator = torch.Generator().manual_seed(91)
        white = torch.randn((2, 6, 12, 10), generator=generator)
        mask = torch.ones((2, 1, 12, 10))
        mask[:, :, :2, :3] = 0
        self.assertTrue(
            torch.equal(
                sampler.project_initial_noise(white, mask),
                colored_projected_gaussian(
                    white,
                    mask,
                    blend=0.75,
                    channel_scales=(0.2,) * 6,
                ),
            )
        )


if __name__ == "__main__":
    unittest.main()
