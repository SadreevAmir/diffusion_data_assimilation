from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from paper.validate_casewise_safety_selector_compact_outputs import (
    FILES, EXPECTED_CASE_IDS, expected_training_case_ids, validate_directory,
)


def valid_payloads():
    cases = []
    for index, case_id in enumerate(EXPECTED_CASE_IDS):
        projected = index % 2 == 0
        raw_loss = 0.10
        projected_loss = 0.097 if projected else 0.099
        cases.append({
            "case_id": case_id, "fold": index // 8,
            "training_case_ids": expected_training_case_ids(index // 8),
            "descriptors": [float(index + item) for item in range(6)],
            "training_mean": [float(item) for item in range(6)],
            "training_scale": [1.0] * 6, "raw_coefficients": [0.0] * 6,
            "projected_spread_coefficients": [0.0] * 6,
            "raw_intercept": raw_loss, "projected_spread_intercept": projected_loss,
            "raw_predicted_loss": raw_loss, "projected_spread_predicted_loss": projected_loss,
            "selected_action": "projected_spread" if projected else "raw",
            "action_margin": raw_loss - projected_loss,
            "raw_source_sha256": "a" * 64, "projected_spread_source_sha256": "b" * 64,
            "analysis_fair_crps_delta": -0.002, "analysis_crps_delta": 0.0001,
            "bitwise_copy_pass": True,
        })
    return {
        "cases": {"schema_version": "casewise-safety-selection-v1", "cases": cases},
        "aggregate": {
            "schema_version": "casewise-safety-aggregate-v1", "num_cases": 40,
            "ensemble_size": 10, "action_counts": {"raw": 20, "projected_spread": 20},
            "raw_analysis_fair_crps": 0.058, "candidate_analysis_fair_crps": 0.056,
            "raw_analysis_crps": 0.062, "candidate_analysis_crps": 0.0621,
            "bitwise_copy_failures": 0, "non_degenerate_policy": True,
            "raw_source_sha256": "a" * 64, "projected_spread_source_sha256": "b" * 64,
        },
        "uncertainty": {
            "schema_version": "casewise-safety-paired-uncertainty-v1",
            "analysis_fair_crps": {"point_delta": -0.002, "date_interval": [-0.003, -0.001], "four_case_block_interval": [-0.004, 0.0]},
            "analysis_crps": {"point_delta": 0.0001, "date_interval": [-0.001, 0.001], "four_case_block_interval": [-0.002, 0.002]},
        },
        "gate": {
            "schema_version": "casewise-safety-no-compensation-gate-v1",
            "families": {name: True for name in ("proper_score", "reliability", "boundary", "spatial_physical", "operational")},
            "overall_eligible": True,
        },
    }


class CompactOutputTests(unittest.TestCase):
    def write(self, payloads):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        directory = Path(temporary.name)
        for key, filename in FILES.items():
            (directory / filename).write_text(json.dumps(payloads[key]), encoding="utf-8")
        return directory

    def test_exact_payload_passes(self):
        validate_directory(self.write(valid_payloads()))

    def test_margin_and_action_mutations_fail_closed(self):
        payloads = valid_payloads(); payloads["cases"]["cases"][0]["action_margin"] = 0.0
        with self.assertRaisesRegex(ValueError, "action_margin"):
            validate_directory(self.write(payloads))
        payloads = valid_payloads(); payloads["cases"]["cases"][0]["selected_action"] = "raw"
        with self.assertRaisesRegex(ValueError, "selected_action"):
            validate_directory(self.write(payloads))

    def test_fold_source_and_aggregate_mutations_fail_closed(self):
        payloads = valid_payloads(); payloads["cases"]["cases"][8]["training_case_ids"].pop()
        with self.assertRaisesRegex(ValueError, "non-circular purge"):
            validate_directory(self.write(payloads))
        payloads = valid_payloads(); payloads["cases"]["cases"][4]["raw_source_sha256"] = "c" * 64
        with self.assertRaisesRegex(ValueError, "source hashes"):
            validate_directory(self.write(payloads))
        payloads = valid_payloads(); payloads["aggregate"]["action_counts"]["raw"] += 1
        with self.assertRaisesRegex(ValueError, "action_counts"):
            validate_directory(self.write(payloads))

    def test_uncertainty_and_gate_mutations_fail_closed(self):
        payloads = valid_payloads(); payloads["uncertainty"]["analysis_fair_crps"]["point_delta"] = -0.001
        with self.assertRaisesRegex(ValueError, "point_delta"):
            validate_directory(self.write(payloads))
        payloads = valid_payloads(); payloads["gate"]["families"]["proper_score"] = False
        with self.assertRaisesRegex(ValueError, "proper_score"):
            validate_directory(self.write(payloads))
        payloads = valid_payloads(); payloads["gate"]["overall_eligible"] = False
        with self.assertRaisesRegex(ValueError, "conjunction"):
            validate_directory(self.write(payloads))

    def test_extra_key_and_file_fail_closed(self):
        payloads = valid_payloads(); payloads["aggregate"]["unreviewed"] = True
        with self.assertRaisesRegex(ValueError, "wrong schema"):
            validate_directory(self.write(payloads))
        directory = self.write(valid_payloads())
        (directory / "extra.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "exactly the four"):
            validate_directory(directory)


if __name__ == "__main__":
    unittest.main()
