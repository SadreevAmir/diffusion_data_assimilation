#!/usr/bin/env python3
from __future__ import annotations

import unittest

from paper.raw_member_reweighting_runner import (
    ARTIFACT_POLICY,
    SOURCE_EXPERIMENT,
    construct_case,
    purged_folds,
    standardize_and_select_analogs,
    synthetic_result,
    validate_interface,
)


class RawMemberRunnerTests(unittest.TestCase):
    def test_frozen_folds_and_purge(self):
        folds = purged_folds()
        self.assertEqual(len(folds), 5)
        self.assertEqual(folds[0][0], tuple(range(8)))
        self.assertNotIn(8, folds[0][1])
        self.assertNotIn(9, folds[0][1])
        self.assertNotIn(10, folds[0][1])
        self.assertEqual(sorted(index for holdout, _ in folds for index in holdout), list(range(40)))

    def test_interface_is_exact(self):
        validate_interface({"source_experiment": SOURCE_EXPERIMENT}, ARTIFACT_POLICY)
        with self.assertRaises(ValueError):
            validate_interface({"source_experiment": SOURCE_EXPERIMENT, "neighbors": 11}, ARTIFACT_POLICY)

    def test_training_only_analog_distance_and_ties(self):
        training = {index: [index + offset / 10 for offset in range(6)] for index in range(12)}
        selected = standardize_and_select_analogs([5 + offset / 10 for offset in range(6)], training)
        self.assertEqual(selected[:3], [5, 4, 6])
        flat = {index: [1.0] * 6 for index in range(10)}
        with self.assertRaisesRegex(ValueError, "scale"):
            standardize_and_select_analogs([1.0] * 6, flat)

    def test_complete_fields_are_copied_with_all_masks(self):
        ranks = [0.0] * 10
        means = [0.5] * 10
        raw = [[[0.0, 0.1], [0.15, 1.0]] for _ in range(10)]
        candidate, diagnostics = construct_case(ranks, means, raw)
        self.assertEqual(len(candidate), 10)
        self.assertTrue(diagnostics["exact_source_copy"])
        self.assertTrue(all(diagnostics["mask_invariants"].values()))
        self.assertEqual(sum(diagnostics["source_multiplicities"]), 10)

    def test_rejects_incomplete_and_nonfinite_fields(self):
        ranks = [(index + 0.5) / 10 for index in range(10)]
        with self.assertRaises(ValueError):
            construct_case(ranks, list(range(10)), [[[0.0]]] * 9)
        raw = [[[0.0]] for _ in range(10)]
        raw[3] = [[float("nan")]]
        with self.assertRaisesRegex(ValueError, "finite"):
            construct_case(ranks, list(range(10)), raw)

    def test_synthetic_record_is_outcome_agnostic(self):
        result = synthetic_result()
        self.assertEqual(result["status"], "PASS")
        self.assertFalse(result["decision_bearing"])
        self.assertFalse(result["project_data_metrics_emitted"])
        self.assertEqual(len(result["folds"]), 5)


if __name__ == "__main__":
    unittest.main()
