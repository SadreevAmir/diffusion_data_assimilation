import unittest

import numpy as np

from assim_lib.clean_baseline_diagnostics import fair_crps_sum, tie_aware_rank_counts


class CleanBaselineDiagnosticsTest(unittest.TestCase):
    def test_ties_are_split_over_all_admissible_ranks(self):
        ensemble = np.asarray([[[0.0, 0.0]], [[0.0, 1.0]]], dtype=np.float64)
        truth = np.asarray([[0.0, 1.0]], dtype=np.float64)
        valid = np.ones_like(truth, dtype=bool)
        counts = tie_aware_rank_counts(ensemble, truth, valid)
        np.testing.assert_allclose(counts, [1.0 / 3.0, 1.0 / 3.0 + 0.5, 1.0 / 3.0 + 0.5])
        self.assertAlmostEqual(float(counts.sum()), 2.0)

    def test_tie_aware_ranks_are_deterministic(self):
        ensemble = np.zeros((10, 2, 2), dtype=np.float64)
        truth = np.zeros((2, 2), dtype=np.float64)
        valid = np.ones((2, 2), dtype=bool)
        first = tie_aware_rank_counts(ensemble, truth, valid)
        second = tie_aware_rank_counts(ensemble, truth, valid)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_allclose(first, np.full(11, 4.0 / 11.0))

    def test_fair_crps_uses_off_diagonal_pair_correction(self):
        ensemble = np.asarray([[[0.0]], [[1.0]]], dtype=np.float64)
        truth = np.asarray([[0.5]], dtype=np.float64)
        valid = np.ones_like(truth, dtype=bool)
        self.assertAlmostEqual(fair_crps_sum(ensemble, truth, valid), 0.0)

    def test_non_finite_valid_member_fails_closed(self):
        ensemble = np.asarray([[[0.0]], [[np.nan]]])
        truth = np.asarray([[0.0]])
        valid = np.ones_like(truth, dtype=bool)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            tie_aware_rank_counts(ensemble, truth, valid)


if __name__ == "__main__":
    unittest.main()
