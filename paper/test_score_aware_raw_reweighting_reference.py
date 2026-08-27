import unittest

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

from paper.score_aware_raw_reweighting_reference import (
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
