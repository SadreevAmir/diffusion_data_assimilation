from __future__ import annotations

import copy
import unittest

from paper.validate_occurrence_intensity_e2_compact_result import METRICS, validate_result


def valid_result() -> dict:
    case_ids = [f"case-{index:02d}" for index in range(8)]
    values = {
        "innovation_rmse": ([1.0] * 8, [0.89] * 8),
        "analysis_fair_crps": ([1.0] * 8, [1.01] * 8),
        "offtrack_anomaly_energy": ([1.0] * 8, [1.04] * 8),
    }
    return {
        "schema_version": "occurrence-intensity-e2-compact-v1",
        "e1_experiment_id": "accepted_e1", "e2_experiment_id": "frozen_e2",
        "inventory_sha256": "a" * 64, "e1_code_sha256": "b" * 64,
        "e2_code_sha256": "c" * 64, "case_ids": case_ids,
        "lag_order": ["current", "one_day_old", "two_days_old"], "ensemble_size": 10,
        "casewise": {metric: {"e1": list(pair[0]), "e2": list(pair[1])} for metric, pair in values.items()},
        "aggregates": {
            metric: {"e1_mean": pair[0][0], "e2_mean": pair[1][0],
                     "relative_change": pair[1][0] / pair[0][0] - 1.0}
            for metric, pair in values.items()
        },
        "invariants": {key: True for key in (
            "finite", "range", "exact_mask", "provenance", "orientation",
            "masked_leakage", "zero_hard_clips", "lag_operator_parity")},
        "operational_checks": {key: True for key in (
            "complete_case_vectors", "identity_match", "inventory_match",
            "code_digest_match", "ten_member_schedule_match")},
        "secondary_diagnostics": {key: {"reported": True} for key in (
            "analysis_crps", "absolute_rank_adequacy", "attainable_coverage",
            "boundary", "spatial_physical")},
        "decision": "TEMPORAL_MECHANISM_USEFUL",
    }


class E2CompactResultTest(unittest.TestCase):
    def test_useful_result_passes(self):
        self.assertEqual(validate_result(valid_result()), "TEMPORAL_MECHANISM_USEFUL")

    def test_finite_threshold_failure_is_scientific_negative(self):
        result = valid_result()
        result["casewise"]["innovation_rmse"]["e2"] = [0.91] * 8
        result["aggregates"]["innovation_rmse"] = {"e1_mean": 1.0, "e2_mean": 0.91, "relative_change": -0.09}
        result["decision"] = "TEMPORAL_MECHANISM_NEGATIVE"
        self.assertEqual(validate_result(result), "TEMPORAL_MECHANISM_NEGATIVE")

    def test_invariant_failure_requires_invalid_decision(self):
        result = valid_result(); result["invariants"]["lag_operator_parity"] = False
        with self.assertRaisesRegex(ValueError, "E2_INVALID"):
            validate_result(result)
        result["decision"] = "E2_INVALID"
        self.assertEqual(validate_result(result), "E2_INVALID")

    def test_nonfinite_and_incomplete_vectors_fail_closed(self):
        result = valid_result(); result["casewise"]["innovation_rmse"]["e2"][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite numeric"):
            validate_result(result)
        result = valid_result(); result["casewise"]["analysis_fair_crps"]["e1"].pop()
        with self.assertRaisesRegex(ValueError, "one value per case"):
            validate_result(result)

    def test_aggregate_identity_and_schema_drift_fail_closed(self):
        result = valid_result(); result["aggregates"]["offtrack_anomaly_energy"]["relative_change"] = 0.0
        with self.assertRaisesRegex(ValueError, "relative_change"):
            validate_result(result)
        result = valid_result(); result["e2_experiment_id"] = result["e1_experiment_id"]
        with self.assertRaisesRegex(ValueError, "identities must differ"):
            validate_result(result)
        result = valid_result(); result["extra"] = True
        with self.assertRaisesRegex(ValueError, "keys must be exact"):
            validate_result(result)

    def test_every_metric_is_required(self):
        for metric in METRICS:
            result = valid_result(); del result["casewise"][metric]
            with self.subTest(metric=metric):
                with self.assertRaisesRegex(ValueError, "metric keys must be exact"):
                    validate_result(result)


if __name__ == "__main__":
    unittest.main()
