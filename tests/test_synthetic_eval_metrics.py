import unittest

import numpy as np

from synthetic_eval.metrics import ensemble_crps, interval_coverage, rank_histogram, spread_skill


class SyntheticEvalMetricTests(unittest.TestCase):
    def test_crps_single_member_absolute_error(self):
        ensemble = np.array([[1.0, 3.0]])
        truth = np.array([2.5, 1.0])
        np.testing.assert_allclose(ensemble_crps(ensemble, truth), np.array([1.5, 2.0]))

    def test_crps_two_members_known_value(self):
        ensemble = np.array([[0.0], [2.0]])
        truth = np.array([1.0])
        np.testing.assert_allclose(ensemble_crps(ensemble, truth), np.array([0.5]))

    def test_rank_histogram_random_ties_are_in_range(self):
        ensemble = np.array([[1.0, 2.0], [1.0, 3.0]])
        truth = np.array([1.0, 2.5])
        counts = rank_histogram(ensemble, truth, seed=42)
        self.assertEqual(counts.shape[0], 3)
        self.assertEqual(int(counts.sum()), 2)

    def test_interval_coverage(self):
        ensemble = np.array([[0.0, 10.0], [1.0, 11.0], [2.0, 12.0], [3.0, 13.0]])
        truth = np.array([1.5, 20.0])
        coverage = interval_coverage(ensemble, truth, levels=(0.5,))
        np.testing.assert_array_equal(coverage["0.5"], np.array([True, False]))

    def test_spread_skill(self):
        ensemble = np.array([[0.0, 2.0], [2.0, 4.0]])
        truth = np.array([1.0, 5.0])
        out = spread_skill(ensemble, truth)
        self.assertAlmostEqual(out["skill_rmse"], np.sqrt(2.0))
        self.assertAlmostEqual(out["spread"], np.sqrt(2.0))
        self.assertAlmostEqual(out["spread_skill_ratio"], 1.0)


if __name__ == "__main__":
    unittest.main()
