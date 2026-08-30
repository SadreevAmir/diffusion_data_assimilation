#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from paper import score_aware_raw_reweighting_reference as reference
from paper.validate_score_aware_raw_reweighting_admission import (
    CONTRACT,
    REFERENCE,
    _assert_nested_close,
    load_and_validate,
    validate_record,
    validate_semantic_parity,
)
from paper.check_publication_artifacts import validate_score_aware_reconciliation_consistency
from paper.test_validate_score_aware_compact_outputs import CompactDirectoryTests, valid_payloads
from paper.validate_score_aware_compact_outputs import directory_sha256


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(runner: Path, compact_directory: Path) -> dict[str, object]:
    return {
        "schema_version": "score-aware-raw-reweighting-admission-v2",
        "reviewed_mode": "validation_reviewed_score_aware_raw_reweighting",
        "publication_commit": "a" * 40,
        "runner_sha256": digest(runner),
        "contract_sha256": digest(CONTRACT),
        "reference_sha256": digest(REFERENCE),
        "compact_directory_sha256": directory_sha256(compact_directory),
        "decision_bearing_validation": "PASS",
        "deviations": [],
    }


class AdmissionTests(unittest.TestCase):
    def test_numpy_scalar_is_compared_as_a_number(self):
        if reference.np is None:
            self.skipTest("numpy is not installed")
        _assert_nested_close(reference.np.float64(0.25), reference.np.float64(0.25), "scalar")
        with self.assertRaisesRegex(ValueError, "scalar diverges"):
            _assert_nested_close(reference.np.float64(0.3), reference.np.float64(0.25), "scalar")

    def compact_directory(self) -> Path:
        return CompactDirectoryTests.write(self, valid_payloads())

    def test_reference_has_semantic_parity_or_requires_declared_dependency(self):
        if reference.np is None:
            with self.assertRaisesRegex(RuntimeError, "numpy is required"):
                validate_semantic_parity(REFERENCE)
        else:
            validate_semantic_parity(REFERENCE)

    def test_selection_divergence_fails_before_ridge(self):
        source = REFERENCE.read_text(encoding="utf-8").replace(
            'ordered_indices = sorted(range(MEMBERS), key=lambda index: (risks[index], index))',
            'ordered_indices = list(reversed(sorted(range(MEMBERS), key=lambda index: (risks[index], index))))',
        )
        with tempfile.TemporaryDirectory() as temporary:
            runner = Path(temporary) / "runner.py"
            runner.write_text(source, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "selection"):
                validate_semantic_parity(runner)

    def test_case_identifier_reordering_divergence_fails_closed(self):
        source = REFERENCE.read_text(encoding="utf-8").replace(
            "identifiers = tuple(case_ids)",
            "identifiers = tuple(sorted(case_ids))",
        )
        with tempfile.TemporaryDirectory() as temporary:
            runner = Path(temporary) / "runner.py"
            runner.write_text(source, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r"folds\[1\]"):
                validate_semantic_parity(runner)

    def test_purged_neighbor_in_training_divergence_fails_closed(self):
        source = REFERENCE.read_text(encoding="utf-8").replace(
            "excluded_start = max(0, start - PURGE)",
            "excluded_start = max(0, start - PURGE + 1)",
        )
        with tempfile.TemporaryDirectory() as temporary:
            runner = Path(temporary) / "runner.py"
            runner.write_text(source, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r"folds\[0\]"):
                validate_semantic_parity(runner)

    def test_predictor_column_order_divergence_fails_closed(self):
        if reference.np is None:
            self.skipTest("numpy is not installed in the minimal local environment")
        source = REFERENCE.read_text(encoding="utf-8").replace(
            "predictors.append(member_features + case_features)",
            "predictors.append(member_features[::-1] + case_features)",
        )
        with tempfile.TemporaryDirectory() as temporary:
            runner = Path(temporary) / "runner.py"
            runner.write_text(source, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "training_rows"):
                validate_semantic_parity(runner)

    def test_target_truth_coefficient_divergence_fails_closed(self):
        if reference.np is None:
            self.skipTest("numpy is not installed in the minimal local environment")
        source = REFERENCE.read_text(encoding="utf-8").replace(
            "+ 0.25 * _weighted_mean(error**2, weights)",
            "+ 0.50 * _weighted_mean(error**2, weights)",
        )
        with tempfile.TemporaryDirectory() as temporary:
            runner = Path(temporary) / "runner.py"
            runner.write_text(source, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "training_rows"):
                validate_semantic_parity(runner)

    def test_nonretained_training_row_divergence_fails_closed(self):
        if reference.np is None:
            self.skipTest("numpy is not installed in the minimal local environment")
        source = REFERENCE.read_text(encoding="utf-8").replace(
            'retained = folds[fold_index]["training_case_ids"]',
            'retained = folds[fold_index]["training_case_ids"] + (folds[fold_index]["holdout_case_ids"][0],)',
        )
        with tempfile.TemporaryDirectory() as temporary:
            runner = Path(temporary) / "runner.py"
            runner.write_text(source, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "training_rows"):
                validate_semantic_parity(runner)

    def test_record_binds_all_three_artifacts_and_rejects_deviations(self):
        directory = self.compact_directory()
        good = record(REFERENCE, directory)
        self.assertEqual(validate_record(good, REFERENCE, directory), good["reviewed_mode"])
        for key in ("runner_sha256", "contract_sha256", "reference_sha256", "compact_directory_sha256"):
            bad = dict(good); bad[key] = "0" * 64
            with self.assertRaisesRegex(ValueError, key):
                validate_record(bad, REFERENCE, directory)
        bad = dict(good); bad["deviations"] = ["waiver"]
        with self.assertRaisesRegex(ValueError, "empty list"):
            validate_record(bad, REFERENCE, directory)

    def test_full_boundary_never_admits_without_semantic_parity(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = self.compact_directory()
            path = Path(temporary) / "admission.json"
            path.write_text(json.dumps(record(REFERENCE, directory)), encoding="utf-8")
            if reference.np is None:
                with self.assertRaisesRegex(RuntimeError, "numpy is required"):
                    load_and_validate(path, REFERENCE, directory)
            else:
                result = load_and_validate(path, REFERENCE, directory)
                self.assertEqual(result.reviewed_mode, record(REFERENCE, directory)["reviewed_mode"])

    def test_combined_admission_identity_drives_reconciled_marker(self):
        directory = self.compact_directory()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "admission.json"
            path.write_text(json.dumps(record(REFERENCE, directory), sort_keys=True), encoding="utf-8")
            with patch(
                "paper.validate_score_aware_raw_reweighting_admission.validate_semantic_parity"
            ):
                result = load_and_validate(path, REFERENCE, directory)
            self.assertEqual(result.admission_record_sha256, digest(path))
            marker = (
                "SCORE_AWARE_RESULT: status=RECONCILED_NEGATIVE; "
                "experiment_id=score_aware_valid; candidate=score_aware_raw; "
                f"admission_record_sha256={result.admission_record_sha256}; "
                f"compact_directory_sha256={result.compact_directory_sha256}; "
                "completed_cases=40; ensemble_size=10; proper_score=false; "
                "reliability=true; boundary=true; spatial_physical=true; "
                "operational=true; overall_eligible=false"
            )
            reconciliation = "Status: RECONCILED_NEGATIVE\n" + marker
            validate_score_aware_reconciliation_consistency(
                marker, marker, marker, marker, reconciliation
            )

    def test_admission_record_substitution_during_combined_admission_fails_closed(self):
        directory = self.compact_directory()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "admission.json"
            frozen = record(REFERENCE, directory)
            path.write_text(json.dumps(frozen), encoding="utf-8")

            def mutate_record(_directory: Path) -> None:
                path.write_text(json.dumps(frozen, indent=2), encoding="utf-8")

            with patch(
                "paper.validate_score_aware_raw_reweighting_admission.validate_directory",
                side_effect=mutate_record,
            ), patch(
                "paper.validate_score_aware_raw_reweighting_admission.validate_semantic_parity"
            ):
                with self.assertRaisesRegex(ValueError, "admission record changed"):
                    load_and_validate(path, REFERENCE, directory)

    def test_post_admission_file_substitution_fails_closed(self):
        directory = self.compact_directory()
        frozen = record(REFERENCE, directory)
        gate = directory / "gate_decision.json"
        document = json.loads(gate.read_text(encoding="utf-8"))
        gate.write_text(json.dumps(document, indent=2), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "compact_directory_sha256"):
            validate_record(frozen, REFERENCE, directory)

    def test_substitution_after_semantic_admission_fails_closed(self):
        directory = self.compact_directory()
        frozen = record(REFERENCE, directory)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "admission.json"
            path.write_text(json.dumps(frozen), encoding="utf-8")

            def mutate_after_semantic(_runner: Path) -> None:
                gate = directory / "gate_decision.json"
                document = json.loads(gate.read_text(encoding="utf-8"))
                gate.write_text(json.dumps(document, indent=2), encoding="utf-8")

            with patch(
                "paper.validate_score_aware_raw_reweighting_admission.validate_semantic_parity",
                side_effect=mutate_after_semantic,
            ):
                with self.assertRaisesRegex(ValueError, "changed during combined admission"):
                    load_and_validate(path, REFERENCE, directory)


if __name__ == "__main__":
    unittest.main()
