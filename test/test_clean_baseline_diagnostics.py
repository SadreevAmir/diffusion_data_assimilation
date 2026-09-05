import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from assim_lib.clean_baseline_diagnostics import (
    fair_crps_sum,
    make_comparison_panels,
    reconcile_case_inputs,
    tie_aware_rank_counts,
)


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

    def test_fixed_comparison_rejects_different_observation_conditions(self):
        with TemporaryDirectory() as root:
            candidate = Path(root) / "candidate"
            reference = Path(root) / "reference"
            output = Path(root) / "panels"
            candidate.mkdir()
            reference.mkdir()
            base = {
                "analysis_ensemble": np.zeros((2, 1, 2, 2), dtype=np.float32),
                "truth": np.zeros((1, 2, 2), dtype=np.float32),
                "background": np.zeros((1, 2, 2), dtype=np.float32),
                "obs_values": np.zeros((1, 2, 2), dtype=np.float32),
                "obs_mask": np.zeros((1, 2, 2), dtype=bool),
                "valid_mask": np.ones((1, 2, 2), dtype=bool),
                "fields": np.asarray(["siconc"]),
            }
            np.savez_compressed(candidate / "0000_2022-01-02_h23.npz", **base)
            changed = dict(base)
            changed["obs_mask"] = base["obs_mask"].copy()
            changed["obs_mask"][0, 0, 0] = True
            np.savez_compressed(reference / "0000_2022-01-02_h23.npz", **changed)

            with self.assertRaisesRegex(ValueError, "observation masks differ"):
                make_comparison_panels(
                    candidate,
                    reference,
                    output,
                    case_orders=(0,),
                    member_ids=(0, 1),
                )

    def test_all_case_reconciliation_rejects_hidden_condition_mismatch(self):
        with TemporaryDirectory() as root:
            candidate = Path(root) / "candidate" / "samples"
            reference = Path(root) / "reference" / "samples"
            candidate.mkdir(parents=True)
            reference.mkdir(parents=True)
            cases = [{"case_order": 0, "dataset_index": 0, "target_date": "2022-01-02", "hour": 23}]
            (candidate.parent / "metadata.json").write_text(json.dumps({"cases": cases}))
            (reference.parent / "metadata.json").write_text(json.dumps({"cases": cases}))
            base = {
                "analysis_ensemble": np.zeros((2, 1, 2, 2), dtype=np.float32),
                "truth": np.zeros((1, 2, 2), dtype=np.float32),
                "background": np.zeros((1, 2, 2), dtype=np.float32),
                "obs_values": np.zeros((1, 2, 2), dtype=np.float32),
                "obs_mask": np.zeros((1, 2, 2), dtype=bool),
                "valid_mask": np.ones((1, 2, 2), dtype=bool),
                "fields": np.asarray(["siconc"]),
            }
            filename = "0000_2022-01-02_h23.npz"
            np.savez_compressed(candidate / filename, **base)
            changed = dict(base)
            changed["truth"] = base["truth"].copy()
            changed["truth"][0, 1, 1] = 0.5
            np.savez_compressed(reference / filename, **changed)

            with self.assertRaisesRegex(ValueError, "truth differs"):
                reconcile_case_inputs(candidate, reference)


if __name__ == "__main__":
    unittest.main()
