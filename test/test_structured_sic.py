import unittest

import torch

from assim_lib.structured_sic import (
    continuous_dequantize_occurrence,
    decode_cfm_targets,
    decode_zero_inflated_sic,
    encode_zero_inflated_sic,
    exact_one_policy,
    make_cfm_targets,
    make_lagged_observation_channels,
)


class LaggedConditioningTest(unittest.TestCase):
    def test_exact_lag_major_layout_for_every_batch_and_pixel(self):
        background = torch.tensor(
            [
                [[[0.1, 0.2]], [[0.3, 0.4]], [[0.5, 0.6]]],
                [[[0.6, 0.5]], [[0.4, 0.3]], [[0.2, 0.1]]],
            ]
        )
        values = torch.tensor(
            [
                [[[0.2, 0.8]], [[0.7, 0.1]], [[0.9, 0.4]]],
                [[[0.1, 0.7]], [[0.8, 0.2]], [[0.3, 0.9]]],
            ]
        )
        masks = torch.tensor(
            [
                [[[1.0, 0.0]], [[1.0, 1.0]], [[0.0, 1.0]]],
                [[[0.0, 1.0]], [[1.0, 0.0]], [[1.0, 1.0]]],
            ]
        )
        ages = torch.tensor([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])
        provenance = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]])

        result = make_lagged_observation_channels(
            background, values, masks, ages, provenance
        )

        expected_lags = []
        for lag in range(3):
            mask = masks[:, lag]
            synthetic = provenance[:, lag, None, None] * mask
            expected_lags.append(
                torch.stack(
                    (
                        (values[:, lag] - background[:, lag]) * mask,
                        values[:, lag] * mask,
                        mask,
                        ages[:, lag, None, None] * mask,
                        mask - synthetic, synthetic, mask - synthetic, synthetic,
                    ),
                    dim=1,
                )
            )
        expected = torch.cat(expected_lags, dim=1)

        self.assertTrue(torch.equal(result, expected))

    def test_channels_keep_lags_age_and_provenance_separate(self):
        background = torch.stack(
            (
                torch.full((1, 2, 2), 0.25),
                torch.full((1, 2, 2), 0.60),
                torch.full((1, 2, 2), 0.40),
            ),
            dim=1,
        )
        values = torch.tensor(
            [[[[0.5, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.8, 0.0]], [[0.0, 0.3], [0.0, 0.0]]]]
        )
        masks = torch.tensor(
            [[[[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [0.0, 0.0]]]]
        )
        result = make_lagged_observation_channels(
            background,
            values,
            masks,
            torch.tensor([[0.0, 1.0, 2.0]]),
            torch.tensor([[0.0, 1.0, 0.0]]),
        )
        self.assertEqual(result.shape, (1, 24, 2, 2))
        first, second, third = result[:, :8], result[:, 8:16], result[:, 16:]
        self.assertAlmostEqual(first[0, 0, 0, 0].item(), 0.25)
        self.assertEqual(first[0, 3, 0, 0].item(), 0.0)
        self.assertEqual(first[0, 4, 0, 0].item(), 1.0)
        self.assertEqual(first[0, 5, 0, 0].item(), 0.0)
        self.assertAlmostEqual(second[0, 0, 1, 0].item(), 0.20)
        self.assertEqual(second[0, 3, 1, 0].item(), 1.0)
        self.assertEqual(second[0, 4, 1, 0].item(), 0.0)
        self.assertEqual(second[0, 5, 1, 0].item(), 1.0)
        self.assertAlmostEqual(third[0, 0, 0, 1].item(), -0.10)
        self.assertEqual(third[0, 3, 0, 1].item(), 2.0)
        self.assertEqual(third[0, 4, 0, 1].item(), 1.0)
        self.assertEqual(third[0, 5, 0, 1].item(), 0.0)
        self.assertTrue(torch.all(first[0, :, 0, 1] == 0))

    def test_invalid_provenance_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "provenance"):
            make_lagged_observation_channels(
                torch.zeros(1, 3, 1, 1),
                torch.zeros(1, 3, 1, 1),
                torch.ones(1, 3, 1, 1),
                torch.tensor([[0.0, 1.0, 2.0]]),
                torch.full((1, 3), 2.0),
            )

    def test_non_binary_mask_and_negative_age_fail_closed(self):
        finite = torch.zeros(1, 3, 1, 1)
        ages = torch.tensor([[0.0, 1.0, 2.0]])
        provenance = torch.zeros(1, 3)
        with self.assertRaisesRegex(ValueError, "masks must be binary"):
            make_lagged_observation_channels(
                finite,
                finite,
                torch.full_like(finite, 0.5),
                ages,
                provenance,
            )
        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            make_lagged_observation_channels(
                finite,
                finite,
                torch.ones_like(finite),
                torch.tensor([[0.0, -1.0, 2.0]]),
                provenance,
            )

    def test_current_background_cannot_be_broadcast_over_lags(self):
        with self.assertRaisesRegex(ValueError, "background_trajectory"):
            make_lagged_observation_channels(
                torch.zeros(1, 1, 1, 1),
                torch.zeros(1, 3, 1, 1),
                torch.ones(1, 3, 1, 1),
                torch.tensor([[0.0, 1.0, 2.0]]),
                torch.zeros(1, 3),
            )

    def test_lag_count_and_age_order_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "exactly three lags"):
            make_lagged_observation_channels(
                torch.zeros(1, 2, 1, 1),
                torch.zeros(1, 2, 1, 1),
                torch.ones(1, 2, 1, 1),
                torch.tensor([[0.0, 1.0]]),
                torch.zeros(1, 2),
            )
        with self.assertRaisesRegex(ValueError, r"exactly \[0,1,2\]"):
            make_lagged_observation_channels(
                torch.zeros(1, 3, 1, 1),
                torch.zeros(1, 3, 1, 1),
                torch.ones(1, 3, 1, 1),
                torch.tensor([[0.0, 2.0, 1.0]]),
                torch.zeros(1, 3),
            )

    def test_non_finite_lagged_inputs_fail_closed(self):
        finite = torch.zeros(1, 3, 1, 1)
        ages = torch.tensor([[0.0, 1.0, 2.0]])
        provenance = torch.zeros(1, 3)
        observed = torch.ones_like(finite)
        with self.assertRaisesRegex(ValueError, "observed pixels"):
            make_lagged_observation_channels(torch.full_like(finite, float("nan")), finite, observed, ages, provenance)
        with self.assertRaisesRegex(ValueError, "observed pixels"):
            make_lagged_observation_channels(finite, torch.full_like(finite, float("inf")), observed, ages, provenance)
        with self.assertRaisesRegex(ValueError, "finite"):
            make_lagged_observation_channels(
                finite, finite, torch.full_like(finite, float("nan")), ages, provenance
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            make_lagged_observation_channels(
                finite, finite, finite, torch.full_like(ages, float("nan")), provenance
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            make_lagged_observation_channels(
                finite, finite, finite, ages, torch.full_like(provenance, float("inf"))
            )

    def test_nan_missing_values_are_zeroed_without_leakage(self):
        background = torch.tensor([[[[0.2, float("nan")]]] * 3])
        values = torch.tensor([[[[0.5, float("nan")]]] * 3])
        masks = torch.tensor([[[[1.0, 0.0]]] * 3])
        result = make_lagged_observation_channels(
            background, values, masks, torch.tensor([[0.0, 1.0, 2.0]]), torch.zeros(1, 3)
        )
        self.assertTrue(torch.all(torch.isfinite(result)))
        self.assertTrue(torch.all(result[..., 1] == 0))

    def test_geometry_and_value_provenance_are_distinct(self):
        base = torch.zeros(1, 3, 1, 1)
        mask = torch.ones_like(base)
        result = make_lagged_observation_channels(
            base, base, mask, torch.tensor([[0.0, 1.0, 2.0]]), torch.zeros(1, 3), torch.ones_like(base)
        )
        self.assertEqual(result[0, 4, 0, 0].item(), 1.0)
        self.assertEqual(result[0, 5, 0, 0].item(), 0.0)
        self.assertEqual(result[0, 6, 0, 0].item(), 0.0)
        self.assertEqual(result[0, 7, 0, 0].item(), 1.0)


class BoundedRepresentationTest(unittest.TestCase):
    def test_complete_dequantized_law_and_zero_auxiliary(self):
        sic = torch.tensor([0.0, 0.2, 1.0])
        occurrence, intensity = make_cfm_targets(
            sic, torch.tensor([0.2, 0.4, 0.8]), torch.tensor([0.7, 0.1, 0.3])
        )
        self.assertTrue(torch.equal(occurrence, torch.tensor([0.1, 0.7, 0.9])))
        self.assertTrue(torch.equal(intensity, torch.tensor([0.7, 0.2, 1.0])))
        self.assertTrue(torch.equal(decode_cfm_targets(occurrence, intensity), sic))
        self.assertTrue(torch.equal(continuous_dequantize_occurrence(torch.tensor([0., 1.]), torch.zeros(2)), torch.tensor([0., .5])))

    def test_exact_one_policy_uses_training_inventory(self):
        self.assertEqual(exact_one_policy(torch.tensor([0.0, 0.9])), "no_exact_one_atom")
        self.assertEqual(exact_one_policy(torch.tensor([0.0, 1.0])), "explicit_exact_one_atom")
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
        with self.assertRaisesRegex(ValueError, "absent atoms"):
            decode_zero_inflated_sic(torch.tensor([0.0]), torch.tensor([0.1]))

    def test_out_of_range_training_target_fails_closed(self):
        with self.assertRaisesRegex(ValueError, r"\[0,1\]"):
            encode_zero_inflated_sic(torch.tensor([-0.01, 0.5]))

    def test_non_finite_encoded_coordinates_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            encode_zero_inflated_sic(torch.tensor([float("nan")]))
        with self.assertRaisesRegex(ValueError, "finite"):
            decode_zero_inflated_sic(
                torch.tensor([1.0]), torch.tensor([float("inf")])
            )


if __name__ == "__main__":
    unittest.main()
