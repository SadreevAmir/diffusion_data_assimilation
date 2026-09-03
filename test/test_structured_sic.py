import unittest

import torch

from assim_lib.structured_sic import (
    decode_zero_inflated_sic,
    encode_zero_inflated_sic,
    make_lagged_observation_channels,
)


class LaggedConditioningTest(unittest.TestCase):
    def test_channels_keep_lags_age_and_provenance_separate(self):
        background = torch.stack(
            (torch.full((1, 2, 2), 0.25), torch.full((1, 2, 2), 0.60)), dim=1
        )
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
        self.assertAlmostEqual(second[0, 0, 1, 0].item(), 0.20)
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

    def test_current_background_cannot_be_broadcast_over_lags(self):
        with self.assertRaisesRegex(ValueError, "background_trajectory"):
            make_lagged_observation_channels(
                torch.zeros(1, 1, 1, 1),
                torch.zeros(1, 2, 1, 1),
                torch.ones(1, 2, 1, 1),
                torch.tensor([[0.0, 1.0]]),
                torch.zeros(1, 2),
            )


class BoundedRepresentationTest(unittest.TestCase):
    def test_round_trip_preserves_zero_and_interior(self):
        concentration = torch.tensor([0.0, 1e-8, 0.001, 0.1, 0.15, 0.1501, 0.4, 0.999, 1.0])
        occurrence, intensity = encode_zero_inflated_sic(concentration)
        decoded = decode_zero_inflated_sic(occurrence, intensity)
        self.assertEqual(occurrence[0].item(), 0.0)
        self.assertTrue(torch.all(occurrence[1:] == 1))
        self.assertTrue(torch.equal(decoded, concentration))

    def test_decoder_fails_closed_instead_of_clipping_invalid_coordinates(self):
        with self.assertRaisesRegex(ValueError, "binary"):
            decode_zero_inflated_sic(torch.tensor([0.2]), torch.tensor([0.1]))
        with self.assertRaisesRegex(ValueError, r"\[0,1\]"):
            decode_zero_inflated_sic(torch.tensor([1.0]), torch.tensor([1.01]))
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            decode_zero_inflated_sic(torch.tensor([1.0]), torch.tensor([0.0]))

    def test_out_of_range_training_target_fails_closed(self):
        with self.assertRaisesRegex(ValueError, r"\[0,1\]"):
            encode_zero_inflated_sic(torch.tensor([-0.01, 0.5]))


if __name__ == "__main__":
    unittest.main()
