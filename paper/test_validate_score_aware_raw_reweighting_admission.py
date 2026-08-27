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
    load_and_validate,
    validate_record,
    validate_semantic_parity,
)
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
                self.assertEqual(load_and_validate(path, REFERENCE, directory), record(REFERENCE, directory)["reviewed_mode"])

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
