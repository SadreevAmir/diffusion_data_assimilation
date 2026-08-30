import unittest

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

from paper.score_aware_raw_reweighting_runner import (
    ARTIFACT_POLICY,
    SOURCE_EXPERIMENT,
    construct_case,
    synthetic_result,
    validate_interface,
)


class ScoreAwareRawReweightingRunnerTest(unittest.TestCase):
    def setUp(self):
        if np is None:
            return
        rows = np.arange(1, 41, dtype=float)[:, None]
        columns = np.arange(1, 14, dtype=float)[None, :]
        self.training = np.sin(rows / columns) + rows * columns / 1000.0
        self.targets = 0.2 + self.training @ np.linspace(-0.1, 0.1, 13)
        self.heldout = self.training[:10] + np.linspace(0.0, 0.2, 10)[:, None]
        self.raw = np.arange(300, dtype=float).reshape(10, 5, 6) / 299.0

    def test_interface_has_no_tuning_fields(self):
        validate_interface({"source_experiment": SOURCE_EXPERIMENT}, ARTIFACT_POLICY)
        for parameters, policy in (
            ({}, ARTIFACT_POLICY),
            ({"source_experiment": SOURCE_EXPERIMENT, "ridge": 2.0}, ARTIFACT_POLICY),
            ({"source_experiment": SOURCE_EXPERIMENT}, "selected_artifacts"),
        ):
            with self.subTest(parameters=parameters, policy=policy):
                with self.assertRaises(ValueError):
                    validate_interface(parameters, policy)

    def test_candidate_is_exact_complete_field_copy(self):
        if np is None:
            self.skipTest("numpy is unavailable")
        candidate, diagnostics = construct_case(
            self.training, self.targets, self.heldout, self.raw
        )
        for output_index, source_index in enumerate(diagnostics["source_raw_member_indices"]):
            np.testing.assert_array_equal(candidate[output_index], self.raw[source_index])
        self.assertTrue(diagnostics["exact_source_copy"])
        self.assertTrue(all(all(record.values()) for record in diagnostics["member_mask_invariants"]))

    def test_heldout_inputs_do_not_fit_scaling(self):
        if np is None:
            self.skipTest("numpy is unavailable")
        _, first = construct_case(self.training, self.targets, self.heldout, self.raw)
        _, second = construct_case(self.training, self.targets, self.heldout + 1000.0, self.raw)
        np.testing.assert_array_equal(
            first["training_predictor_mean"], second["training_predictor_mean"]
        )
        np.testing.assert_array_equal(
            first["training_predictor_scale"], second["training_predictor_scale"]
        )

    def test_bad_member_shape_and_nonfinite_values_fail_closed(self):
        if np is None:
            self.skipTest("numpy is unavailable")
        for raw in (self.raw[:9], self.raw.reshape(10, -1), np.where(self.raw == 0, np.nan, self.raw)):
            with self.subTest(shape=raw.shape):
                with self.assertRaises(ValueError):
                    construct_case(self.training, self.targets, self.heldout, raw)

    def test_synthetic_result_is_non_decision_bearing(self):
        if np is None:
            self.skipTest("numpy is unavailable")
        result = synthetic_result()
        self.assertEqual(result["status"], "PASS")
        self.assertFalse(result["decision_bearing"])
        self.assertFalse(result["project_data_metrics_emitted"])
        self.assertEqual(result["fold_count"], 5)
        self.assertEqual(result["candidate_members"], 10)


if __name__ == "__main__":
    unittest.main()
