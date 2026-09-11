import unittest

import torch
from torch import nn

from assim_lib.direct_dynamics_cascade import masked_block_average, project_detail
from assim_lib.direct_dynamics_cascade_fine import generated_coarse_condition
from assim_lib.direct_dynamics_cascade_fine_matched_scale import (
    MatchedScaleFineCascadeSampler,
    matched_residual_flow_pair,
    scaled_projected_noise,
    validate_channel_scales,
)


SCALES = (0.06, 0.07, 0.08, 0.09, 0.10, 0.11)


class ZeroModel(nn.Module):
    def forward(self, model_input, timestep, return_dict=False):
        del timestep, return_dict
        return (torch.zeros_like(model_input[:, :6]),)


class MatchedScaleFineTests(unittest.TestCase):
    def test_scale_contract_rejects_unsafe_values(self) -> None:
        self.assertEqual(validate_channel_scales(list(SCALES)), SCALES)
        for bad in ([1.0] * 5, [0.0] * 6, [1.0] * 6, [float("nan")] * 6):
            with self.assertRaises(ValueError):
                validate_channel_scales(bad)

    def test_scaled_base_is_projected_and_channel_scaled(self) -> None:
        generator = torch.Generator().manual_seed(11)
        raw = torch.randn((2, 6, 12, 10), generator=generator)
        valid = torch.ones((2, 1, 12, 10))
        actual = scaled_projected_noise(raw, valid, SCALES)
        expected = project_detail(raw, valid) * torch.tensor(SCALES).reshape(1, 6, 1, 1)
        self.assertTrue(torch.allclose(actual, expected, atol=2e-6, rtol=0))
        coarse, fraction = masked_block_average(actual, valid)
        self.assertLess(float(coarse[fraction.expand_as(coarse) > 0].abs().max()), 2e-6)

    def test_flow_sign_and_endpoints_are_exact(self) -> None:
        generator = torch.Generator().manual_seed(13)
        clean = torch.randn((2, 6, 12, 10), generator=generator)
        raw = torch.randn((2, 6, 12, 10), generator=generator)
        valid = torch.ones((2, 1, 12, 10))
        for time_value in (0.0, 1.0):
            state, velocity, target, base = matched_residual_flow_pair(
                clean, raw, valid, torch.full((2,), time_value), SCALES
            )
            self.assertTrue(torch.allclose(velocity, base - target, atol=0, rtol=0))
            endpoint = target if time_value == 0 else base
            self.assertTrue(torch.allclose(state, endpoint, atol=2e-6, rtol=0))

    def test_sampler_scales_raw_noise_before_projection(self) -> None:
        sampler = MatchedScaleFineCascadeSampler(ZeroModel(), SCALES)
        generator = torch.Generator().manual_seed(17)
        raw = torch.randn((1, 6, 12, 10), generator=generator)
        valid = torch.ones((1, 1, 12, 10))
        self.assertTrue(
            torch.equal(
                sampler.project_initial_noise(raw, valid),
                scaled_projected_noise(raw, valid, SCALES),
            )
        )

    def test_zero_velocity_sampler_starts_from_the_matched_base(self) -> None:
        sampler = MatchedScaleFineCascadeSampler(ZeroModel(), SCALES)
        generator = torch.Generator().manual_seed(19)
        raw = torch.randn((1, 6, 12, 10), generator=generator)
        valid = torch.ones((1, 1, 12, 10))
        causal = torch.zeros((1, 15, 12, 10))
        causal[:, 2:3] = valid
        fine_condition = generated_coarse_condition(
            causal, torch.zeros((1, 6, 6, 5)), valid
        )
        zeros = torch.zeros((1, 6, 12, 10))
        result = sampler.sample_conditioned(
            background=zeros,
            background_mask=torch.ones_like(zeros),
            obs_values=zeros[:, :2],
            obs_mask=zeros[:, :2],
            water_mask=valid,
            valid_mask=valid,
            state_mask=valid.expand_as(zeros),
            size=(12, 10),
            num_timesteps=2,
            device=torch.device("cpu"),
            method="rk4",
            start_mode="noise",
            initial_noise=raw,
            sample_target="state",
            model_conditioning=fine_condition,
            state_channels=6,
            end_time=0.0,
        )
        expected = scaled_projected_noise(raw, valid, SCALES)
        self.assertTrue(torch.allclose(result, expected, atol=3e-6, rtol=0))


if __name__ == "__main__":
    unittest.main()
