import unittest

import torch

from assim_lib.bounded_sic import (
    bounded_sic_velocity_loss,
    decode_bounded_sic,
    encode_bounded_sic,
    open_unit_uniform_like,
)


class BoundedSicTest(unittest.TestCase):
    def test_exact_zero_and_positive_values_round_trip(self):
        concentration = torch.tensor([[[[0.0, 1e-6, 0.15, 0.8, 0.95, 0.999]]]])
        occurrence_uniform = torch.full_like(concentration, 0.37)
        zero_uniform = torch.full_like(concentration, 0.61)
        latent = encode_bounded_sic(concentration, occurrence_uniform, zero_uniform)
        restored = decode_bounded_sic(latent)
        self.assertEqual(float(restored[0, 0, 0, 0]), 0.0)
        torch.testing.assert_close(restored[..., 1:], concentration[..., 1:], atol=1e-7, rtol=1e-6)

    def test_occurrence_is_decoded_from_latent_sign(self):
        latent = torch.tensor([[[[-2.0, 2.0]], [[8.0, -8.0]]]])
        restored = decode_bounded_sic(latent)
        self.assertEqual(float(restored[0, 0, 0, 0]), 0.0)
        self.assertGreater(float(restored[0, 0, 0, 1]), 0.0)
        self.assertLess(float(restored[0, 0, 0, 1]), 1.0)

    def test_exact_one_requires_a_separate_upper_atom_head(self):
        concentration = torch.ones((1, 1, 1, 1))
        auxiliary = torch.full_like(concentration, 0.5)
        with self.assertRaisesRegex(ValueError, r"\[0, 1\)"):
            encode_bounded_sic(concentration, auxiliary, auxiliary)

    def test_auxiliary_uniform_is_open_and_reproducible(self):
        reference = torch.zeros((2, 1, 3, 4))
        first = open_unit_uniform_like(reference, generator=torch.Generator().manual_seed(7))
        second = open_unit_uniform_like(reference, generator=torch.Generator().manual_seed(7))
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(bool(torch.all((first > 0.0) & (first < 1.0))))

    def test_channel_balanced_loss(self):
        prediction = torch.zeros((1, 2, 1, 2))
        target = torch.tensor([[[[1.0, 1.0]], [[2.0, 2.0]]]])
        valid = torch.ones((1, 1, 1, 2))
        loss = bounded_sic_velocity_loss(prediction, target, valid)
        self.assertAlmostEqual(float(loss), 2.5)


if __name__ == "__main__":
    unittest.main()
