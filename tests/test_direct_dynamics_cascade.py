import unittest

import torch

from assim_lib.direct_dynamics_cascade import (
    decompose,
    masked_block_average,
    project_detail,
    reconstruct,
    residual_flow_pair,
    smooth_right_inverse,
)


class DirectDynamicsCascadeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.generator = torch.Generator().manual_seed(101)
        self.mask = torch.ones(2, 1, 8, 10)
        self.mask[:, :, :2, :2] = 0
        self.mask[1, :, 4:6, 6:8] = 0

    def test_exact_roundtrip_on_valid_ocean(self) -> None:
        value = torch.randn(2, 6, 8, 10, generator=self.generator)
        state = decompose(value, self.mask)
        actual = reconstruct(state, self.mask)
        valid = self.mask.expand_as(value) > 0
        self.assertTrue(torch.allclose(actual[valid], value[valid], atol=3e-7, rtol=0))
        self.assertTrue(torch.equal(actual[~valid], torch.zeros_like(actual[~valid])))

    def test_smooth_lift_is_a_right_inverse(self) -> None:
        coarse = torch.randn(2, 6, 4, 5, generator=self.generator)
        coarse[:, :, 0, 0] = 0
        lift = smooth_right_inverse(coarse, self.mask)
        recovered, fraction = masked_block_average(lift, self.mask)
        active = fraction > 0
        self.assertTrue(torch.allclose(recovered[active], coarse[active], atol=2e-6, rtol=0))

    def test_inactive_coarse_values_cannot_leak_into_ocean(self) -> None:
        coarse = torch.randn(2, 6, 4, 5, generator=self.generator)
        first = coarse.clone()
        second = coarse.clone()
        first[:, :, 0, 0] = float("nan")
        second[:, :, 0, 0] = float("inf")
        self.assertTrue(
            torch.equal(smooth_right_inverse(first, self.mask), smooth_right_inverse(second, self.mask))
        )

    def test_constant_ocean_has_no_coastal_ramp(self) -> None:
        value = torch.full((2, 6, 8, 10), 3.25)
        coarse, _ = masked_block_average(value, self.mask)
        lift = smooth_right_inverse(coarse, self.mask)
        valid = self.mask.expand_as(lift) > 0
        self.assertTrue(torch.allclose(lift[valid], torch.full_like(lift[valid], 3.25), atol=1e-6))

    def test_fractional_fine_mask_is_rejected(self) -> None:
        mask = self.mask.clone()
        mask[0, 0, 3, 3] = 1e-8
        with self.assertRaises(ValueError):
            masked_block_average(torch.ones(2, 6, 8, 10), mask)

    def test_boolean_fine_mask_is_supported(self) -> None:
        value = torch.ones(2, 6, 8, 10)
        coarse, fraction = masked_block_average(value, self.mask.bool())
        self.assertTrue(torch.equal(coarse[fraction > 0], torch.ones_like(coarse[fraction > 0])))

    def test_detail_projection_has_zero_coarse_component_and_is_idempotent(self) -> None:
        value = torch.randn(2, 6, 8, 10, generator=self.generator)
        detail = project_detail(value, self.mask)
        coarse_detail, fraction = masked_block_average(detail, self.mask)
        self.assertLess(float(coarse_detail[fraction > 0].abs().max()), 2e-6)
        twice = project_detail(detail, self.mask)
        self.assertTrue(torch.allclose(twice, detail, atol=3e-6, rtol=0))

    def test_invalid_land_values_do_not_change_decomposition(self) -> None:
        first = torch.randn(2, 6, 8, 10, generator=self.generator)
        second = first.clone()
        invalid = self.mask.expand_as(first) == 0
        second[invalid] = float("nan")
        state_first = decompose(first, self.mask)
        state_second = decompose(second, self.mask)
        self.assertTrue(torch.equal(state_first.coarse, state_second.coarse))
        self.assertTrue(torch.equal(state_first.residual, state_second.residual))

    def test_nonfinite_valid_ocean_fails_closed(self) -> None:
        value = torch.zeros(2, 6, 8, 10)
        value[0, 0, 3, 3] = float("inf")
        with self.assertRaises(FloatingPointError):
            decompose(value, self.mask)

    def test_flow_sign_and_subspace_are_exact(self) -> None:
        clean = torch.randn(2, 6, 8, 10, generator=self.generator)
        noise = torch.randn(2, 6, 8, 10, generator=self.generator)
        state, velocity, clean_detail, noise_detail = residual_flow_pair(
            clean, noise, self.mask, torch.ones(2)
        )
        self.assertTrue(torch.allclose(state, noise_detail, atol=2e-6, rtol=0))
        recovered_at_zero = state - velocity
        self.assertTrue(torch.allclose(recovered_at_zero, clean_detail, atol=3e-6, rtol=0))
        coarse_velocity, fraction = masked_block_average(velocity, self.mask)
        self.assertLess(float(coarse_velocity[fraction > 0].abs().max()), 3e-6)

    def test_flow_rejects_invalid_time_and_overflow(self) -> None:
        clean = torch.ones(2, 6, 8, 10)
        noise = -clean
        for time in (torch.tensor([float("nan"), 0.5]), torch.tensor([-0.1, 0.5]), torch.tensor([1.1, 0.5])):
            with self.assertRaises(ValueError):
                residual_flow_pair(clean, noise, self.mask, time)
        checkerboard = torch.tensor([[1.0, -1.0], [-1.0, 1.0]]).repeat(4, 5)
        huge = checkerboard[None, None].expand(2, 6, -1, -1) * 2e38
        with self.assertRaises(FloatingPointError):
            residual_flow_pair(huge, -huge, self.mask, torch.full((2,), 0.5))

    def test_float64_identities_and_reconstruction_from_coarse(self) -> None:
        value = torch.randn(2, 6, 8, 10, generator=self.generator, dtype=torch.float64)
        state = decompose(value, self.mask.double())
        recovered = reconstruct(state, self.mask.double())
        valid = self.mask.expand_as(value) > 0
        self.assertTrue(torch.allclose(recovered[valid], value[valid], atol=1e-14, rtol=0))


if __name__ == "__main__":
    unittest.main()
