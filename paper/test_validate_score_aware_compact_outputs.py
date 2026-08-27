"""Focused fixtures for score-aware compact directory parity."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from paper.score_aware_raw_reweighting_reference import select_raw_scenarios
from paper.validate_score_aware_compact_outputs import FILES, validate_directory


def valid_payloads() -> dict[str, object]:
    cases = []
    totals = [0] * 10
    for case_index in range(40):
        selection = select_raw_scenarios([float((member + case_index) % 4) for member in range(10)])
        for member, count in enumerate(selection["source_multiplicities"]):
            totals[member] += count
        cases.append({
            "case_id": f"case-{case_index:02d}", "fold": case_index // 8,
            "predicted_risks": selection["predicted_risks"],
            "normalized_weights": selection["normalized_weights"],
            "source_raw_member_indices": selection["source_raw_member_indices"],
            "source_multiplicities": selection["source_multiplicities"],
            "unique_selected_raw_members": selection["unique_selected_raw_members"],
            "effective_sample_size": selection["effective_sample_size"],
            "bitwise_copy_pass": True, "mask_invariants_pass": True,
            "analysis_fair_crps_delta": -0.01, "analysis_crps_delta": 0.001,
        })
    return {
        "cases": {"schema_version": "score-aware-case-selection-v1", "cases": cases},
        "aggregate": {
            "schema_version": "score-aware-selection-aggregate-v1", "num_cases": 40,
            "source_multiplicity_totals": totals,
            "mean_unique_selected_raw_members": sum(c["unique_selected_raw_members"] for c in cases) / 40,
            "mean_effective_sample_size": sum(c["effective_sample_size"] for c in cases) / 40,
            "bitwise_copy_failures": 0, "mask_invariant_failures": 0,
            "all_invariants_pass": True,
        },
        "uncertainty": {
            "schema_version": "score-aware-paired-uncertainty-v1",
            "analysis_fair_crps": {"point_delta": -0.01, "date_interval": [-0.02, -0.001], "four_case_block_interval": [-0.03, 0.0]},
            "analysis_crps": {"point_delta": 0.001, "date_interval": [-0.001, 0.003], "four_case_block_interval": [-0.002, 0.004]},
        },
        "gate": {
            "schema_version": "score-aware-no-compensation-gate-v1",
            "families": {name: True for name in ("proper_score", "reliability", "boundary", "spatial_physical", "operational")},
            "overall_eligible": True,
        },
    }


class CompactDirectoryTests(unittest.TestCase):
    def write(self, payloads: dict[str, object]) -> Path:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        directory = Path(temporary.name)
        for key, filename in FILES.items():
            (directory / filename).write_text(json.dumps(payloads[key]), encoding="utf-8")
        return directory

    def test_exact_directory_passes(self) -> None:
        validate_directory(self.write(valid_payloads()))

    def test_changed_weight_fails_closed(self) -> None:
        payloads = valid_payloads()
        payloads["cases"]["cases"][0]["normalized_weights"][0] += 0.01
        with self.assertRaisesRegex(ValueError, "normalized_weights"):
            validate_directory(self.write(payloads))

    def test_changed_ess_and_aggregate_fail_closed(self) -> None:
        payloads = valid_payloads()
        payloads["cases"]["cases"][3]["effective_sample_size"] += 0.1
        with self.assertRaisesRegex(ValueError, "effective_sample_size"):
            validate_directory(self.write(payloads))
        payloads = valid_payloads()
        payloads["aggregate"]["source_multiplicity_totals"][2] += 1
        with self.assertRaisesRegex(ValueError, "aggregate counts"):
            validate_directory(self.write(payloads))

    def test_invariant_and_gate_disagreement_fail_closed(self) -> None:
        payloads = valid_payloads()
        payloads["cases"]["cases"][0]["bitwise_copy_pass"] = False
        with self.assertRaisesRegex(ValueError, "copy/mask aggregate"):
            validate_directory(self.write(payloads))
        payloads = valid_payloads()
        payloads["uncertainty"]["analysis_fair_crps"]["point_delta"] = -0.02
        with self.assertRaisesRegex(ValueError, "point_delta"):
            validate_directory(self.write(payloads))
        payloads = valid_payloads()
        payloads["gate"]["overall_eligible"] = False
        with self.assertRaisesRegex(ValueError, "conjunction"):
            validate_directory(self.write(payloads))

    def test_extra_file_fails_closed(self) -> None:
        directory = self.write(valid_payloads())
        (directory / "unreviewed.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "exactly the four"):
            validate_directory(directory)


if __name__ == "__main__":
    unittest.main()
