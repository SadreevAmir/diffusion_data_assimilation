import unittest

import torch

from assim_lib.structured_sic import (
    decode_zero_inflated_sic,
    encode_zero_inflated_sic,
    make_lagged_observation_channels,
)


class LaggedConditioningTest(unittest.TestCase):
    def test_channels_keep_lags_age_and_provenance_separate(self):
        background = torch.full((1, 1, 2, 2), 0.25)
        values = torch.tensor([[[[0.5, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.8, 0.0]]]])
        masks = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 0.0]]]])
        result = make_lagged_observation_channels(
            background, values, masks, torch.tensor([[0.0, 2.0]]), torch.tensor([[0.0, 1.0]])
        )
        self.assertEqual(result.shape, (1, 12, 2, 2))
        first, second = result[:, :6], result[:, 6:]
        self.assertAlmostEqual(first[0, 0, 0, 0].item(), 0.25)
        self.assertEqual(first[0, 3, 0, 0].item(), 0.0)
        self.assertEqual(first[0, 4, 0, 0].item(), 1.0)
        self.assertEqual(first[0, 5, 0, 0].item(), 0.0)
        self.assertAlmostEqual(second[0, 0, 1, 0].item(), 0.55)
        self.assertEqual(second[0, 3, 1, 0].item(), 2.0)
        self.assertEqual(second[0, 4, 1, 0].item(), 0.0)
        self.assertEqual(second[0, 5, 1, 0].item(), 1.0)
        self.assertTrue(torch.all(first[0, :, 0, 1] == 0))

    def test_invalid_provenance_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "provenance"):
            make_lagged_observation_channels(
                torch.zeros(1, 1, 1, 1),
                torch.zeros(1, 1, 1, 1),
                torch.ones(1, 1, 1, 1),
                torch.zeros(1, 1),
                torch.full((1, 1), 2.0),
            )


class BoundedRepresentationTest(unittest.TestCase):
    def test_round_trip_preserves_zero_and_interior(self):
        concentration = torch.tensor([0.0, 0.1, 0.1501, 0.4, 0.999])
        occurrence, intensity = encode_zero_inflated_sic(concentration)
        decoded = decode_zero_inflated_sic(occurrence, intensity)
        self.assertTrue(torch.equal(decoded[:2], torch.zeros(2)))
        self.assertTrue(torch.allclose(decoded[2:], concentration[2:], atol=2e-6))

    def test_arbitrary_finite_logits_are_physical_without_clipping(self):
        occurrence = torch.tensor([-100.0, 0.1, 100.0])
        intensity = torch.tensor([-100.0, 0.0, 100.0])
        decoded = decode_zero_inflated_sic(occurrence, intensity)
        self.assertEqual(decoded[0].item(), 0.0)
        self.assertTrue(torch.all((decoded >= 0) & (decoded <= 1)))
        self.assertGreater(decoded[1].item(), 0.15)
        self.assertLessEqual(decoded[2].item(), 1.0)

    def test_out_of_range_training_target_fails_closed(self):
        with self.assertRaisesRegex(ValueError, r"\[0,1\]"):
            encode_zero_inflated_sic(torch.tensor([-0.01, 0.5]))


if __name__ == "__main__":
    unittest.main()
