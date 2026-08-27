#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from paper import score_aware_raw_reweighting_reference as reference
from paper.validate_score_aware_raw_reweighting_admission import (
    CONTRACT,
    REFERENCE,
    load_and_validate,
    validate_record,
    validate_semantic_parity,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(runner: Path) -> dict[str, object]:
    return {
        "schema_version": "score-aware-raw-reweighting-admission-v1",
        "reviewed_mode": "validation_reviewed_score_aware_raw_reweighting",
        "publication_commit": "a" * 40,
        "runner_sha256": digest(runner),
        "contract_sha256": digest(CONTRACT),
        "reference_sha256": digest(REFERENCE),
        "decision_bearing_validation": "PASS",
        "deviations": [],
    }


class AdmissionTests(unittest.TestCase):
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
        good = record(REFERENCE)
        self.assertEqual(validate_record(good, REFERENCE), good["reviewed_mode"])
        for key in ("runner_sha256", "contract_sha256", "reference_sha256"):
            bad = dict(good); bad[key] = "0" * 64
            with self.assertRaisesRegex(ValueError, key):
                validate_record(bad, REFERENCE)
        bad = dict(good); bad["deviations"] = ["waiver"]
        with self.assertRaisesRegex(ValueError, "empty list"):
            validate_record(bad, REFERENCE)

    def test_full_boundary_never_admits_without_semantic_parity(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "admission.json"
            path.write_text(json.dumps(record(REFERENCE)), encoding="utf-8")
            if reference.np is None:
                with self.assertRaisesRegex(RuntimeError, "numpy is required"):
                    load_and_validate(path, REFERENCE)
            else:
                self.assertEqual(load_and_validate(path, REFERENCE), record(REFERENCE)["reviewed_mode"])


if __name__ == "__main__":
    unittest.main()
