import unittest

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

from paper.score_aware_raw_reweighting_reference import (
    build_case_rows,
    fit_fold_ridge,
    predict_member_risks,
    select_raw_scenarios,
)


class ScoreAwareRawReweightingReferenceTest(unittest.TestCase):
    def setUp(self):
        if np is None:
            return
        rows = np.arange(1, 31, dtype=float)[:, None]
        columns = np.arange(1, 14, dtype=float)[None, :]
        self.design = np.sin(rows / columns) + rows * columns / 1000.0
        self.targets = 0.2 + self.design @ np.linspace(-0.1, 0.1, 13)

    @staticmethod
    def _manual_weighted_mean(field, weights):
        numerator = sum(
            float(field[row, column]) * float(weights[row, column])
            for row in range(field.shape[0])
            for column in range(field.shape[1])
        )
        return numerator / sum(float(value) for value in weights.flat)

    @classmethod
    def _manual_semivariogram(cls, field, weights, lag):
        numerator = 0.0
        denominator = 0.0
        rows, columns = field.shape
        for row in range(rows):
            for column in range(columns - lag):
                pair_weight = 0.5 * (
                    float(weights[row, column]) + float(weights[row, column + lag])
                )
                increment = float(field[row, column + lag] - field[row, column])
                numerator += pair_weight * 0.5 * increment**2
                denominator += pair_weight
        for row in range(rows - lag):
            for column in range(columns):
                pair_weight = 0.5 * (
                    float(weights[row, column]) + float(weights[row + lag, column])
                )
                increment = float(field[row + lag, column] - field[row, column])
                numerator += pair_weight * 0.5 * increment**2
                denominator += pair_weight
        return numerator / denominator

    @classmethod
    def _manual_descriptors(cls, field, weights):
        mean = cls._manual_weighted_mean(field, weights)
        variance = cls._manual_weighted_mean((field - mean) ** 2, weights)
        return (
            mean,
            cls._manual_weighted_mean(field >= 0.15, weights),
            mean,
            variance**0.5,
            cls._manual_semivariogram(field, weights, 1),
            cls._manual_semivariogram(field, weights, 4),
        )

    def _asymmetric_case(self):
        rows, columns = np.indices((5, 6), dtype=float)
        weights = 1.0 + 0.7 * rows + 0.13 * columns + 0.03 * rows * columns
        members = np.stack(
            [
                np.clip(
                    0.025 * (index + 1)
                    + 0.031 * rows**2
                    + 0.019 * columns
                    + 0.004 * (index + 1) * rows * columns,
                    0.0,
                    1.0,
                )
                for index in range(10)
            ]
        )
        truth = np.clip(0.09 + 0.021 * rows + 0.047 * columns**1.5, 0.0, 1.0)
        return members, truth, weights

    def test_all_thirteen_predictors_match_independent_asymmetric_grid_oracle(self):
        if np is None:
            self.skipTest("numpy is not installed in the minimal local environment")
        members, truth, weights = self._asymmetric_case()
        actual = build_case_rows(members, truth, weights)["predictors"]
        raw_mean = members.mean(axis=0)
        case_descriptors = self._manual_descriptors(raw_mean, weights)
        mean_area = self._manual_weighted_mean(raw_mean, weights)

        for member_index, member in enumerate(members):
            expected = self._manual_descriptors(member, weights) + (
                abs(self._manual_weighted_mean(member, weights) - mean_area),
            ) + case_descriptors
            for column, value in enumerate(expected):
                with self.subTest(member=member_index, predictor_column=column):
                    self.assertAlmostEqual(actual[member_index, column], value, places=13)

    def test_spatial_weight_target_matches_manual_oracle(self):
        if np is None:
            self.skipTest("numpy is not installed in the minimal local environment")
        members, truth, weights = self._asymmetric_case()
        actual = build_case_rows(members, truth, weights)["targets"]
        for member_index, member in enumerate(members):
            error = member - truth
            expected = self._manual_weighted_mean(abs(error), weights) + 0.25 * self._manual_weighted_mean(
                error**2, weights
            )
            with self.subTest(member=member_index):
                self.assertAlmostEqual(actual[member_index], expected, places=13)

    def test_forecast_predictors_are_invariant_to_heldout_truth(self):
        if np is None:
            self.skipTest("numpy is not installed in the minimal local environment")
        members, truth, weights = self._asymmetric_case()
        first = build_case_rows(members, truth, weights)
        altered_truth = np.flipud(1.0 - truth)
        second = build_case_rows(members, altered_truth, weights)
        np.testing.assert_array_equal(first["predictors"], second["predictors"])
        self.assertFalse(np.array_equal(first["targets"], second["targets"]))

    def test_fit_is_deterministic_and_intercept_is_unpenalized(self):
        if np is None:
            self.skipTest("numpy is not installed in the minimal local environment")
        first = fit_fold_ridge(self.design, self.targets)
        second = fit_fold_ridge(self.design.tolist(), self.targets.tolist())
        self.assertEqual(first["intercept"], float(self.targets.mean()))
        np.testing.assert_array_equal(first["coefficients"], second["coefficients"])
        np.testing.assert_allclose(first["training_predictor_mean"], self.design.mean(0))

    def test_prediction_requires_exactly_ten_members(self):
        if np is None:
            self.skipTest("numpy is not installed in the minimal local environment")
        model = fit_fold_ridge(self.design, self.targets)
        risks = predict_member_risks(model, self.design[:10])
        self.assertEqual(risks.shape, (10,))
        with self.assertRaises(ValueError):
            predict_member_risks(model, self.design[:9])

    def test_equal_risks_preserve_index_order_and_all_members(self):
        result = select_raw_scenarios([2.0] * 10)
        self.assertEqual(result["risk_ordered_raw_member_indices"], list(range(10)))
        self.assertEqual(result["source_raw_member_indices"], list(range(10)))
        self.assertEqual(result["source_multiplicities"], [1] * 10)
        self.assertAlmostEqual(result["effective_sample_size"], 10.0)

    def test_nonuniform_weights_can_still_quantize_to_null_action(self):
        result = select_raw_scenarios([0.02 * index for index in range(10)])
        self.assertGreater(max(result["normalized_weights"]), min(result["normalized_weights"]))
        self.assertEqual(result["source_multiplicities"], [1] * 10)
        self.assertEqual(sorted(result["source_raw_member_indices"]), list(range(10)))
        self.assertAlmostEqual(result["effective_sample_size"], 10.0)

    def test_lower_risk_receives_more_mass_with_deterministic_ties(self):
        result = select_raw_scenarios([0.0, 0.0] + [10.0] * 8)
        self.assertEqual(result["risk_ordered_raw_member_indices"], list(range(10)))
        self.assertEqual(result["source_multiplicities"][:2], [5, 5])
        self.assertEqual(result["source_multiplicities"][2:], [0] * 8)
        self.assertAlmostEqual(sum(result["normalized_weights"]), 1.0)

    def test_extreme_risk_range_is_numerically_stable(self):
        result = select_raw_scenarios([0.0] + [1e300] * 9)
        self.assertEqual(result["source_raw_member_indices"], [0] * 10)
        self.assertEqual(result["effective_sample_size"], 1.0)

    def test_invalid_inputs_fail_closed(self):
        if np is not None:
            bad_designs = [
                self.design[:, :12],
                np.column_stack([self.design[:, :12], np.ones(len(self.design))]),
                np.where(np.arange(13) == 0, np.nan, self.design[0])[None, :],
            ]
            for design in bad_designs:
                with self.subTest(shape=design.shape):
                    with self.assertRaises(ValueError):
                        fit_fold_ridge(design, np.ones(len(design)))
        for risks in ([0.0] * 9, [0.0] * 9 + [float("nan")], [False] * 10):
            with self.subTest(risks=risks):
                with self.assertRaises(ValueError):
                    select_raw_scenarios(risks)


if __name__ == "__main__":
    unittest.main()
