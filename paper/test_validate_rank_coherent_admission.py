#!/usr/bin/env python3
"""Focused tests for the explicit admission-record CLI boundary."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from paper.validate_rank_coherent_admission import load_and_validate


PAPER_DIR = Path(__file__).resolve().parent
SCRIPT = PAPER_DIR / "validate_rank_coherent_admission.py"


def valid_record() -> dict[str, object]:
    return {
        "reviewed_mode": "validation_reviewed_rank_coherent",
        "publication_commit": "a" * 40,
        "runner_sha256": "b" * 64,
        "contract_sha256": "c" * 64,
        "synthetic_result_sha256": "d" * 64,
        "test_command": "python3 trusted_test.py",
        "test_sentinel": "trusted rank-coherent adapter: PASS",
        "decision_bearing_validation": "PASS",
        "deviations": [],
    }


class AdmissionCliTests(unittest.TestCase):
    def write_record(self, directory: Path, value: object) -> Path:
        path = directory / "admission.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_valid_record_returns_literal_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self.write_record(Path(temporary), valid_record())
            self.assertEqual(load_and_validate(path), "validation_reviewed_rank_coherent")
            result = subprocess.run(
                [sys.executable, str(SCRIPT), str(path)],
                check=True,
                capture_output=True,
                text=True,
            )
        self.assertEqual(result.stdout, "reviewed_mode=validation_reviewed_rank_coherent\n")
        self.assertEqual(result.stderr, "")

    def test_non_object_and_deviation_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with self.assertRaisesRegex(ValueError, "JSON object"):
                load_and_validate(self.write_record(directory, []))
            record = valid_record()
            record["deviations"] = ["waiver"]
            with self.assertRaisesRegex(ValueError, "empty list"):
                load_and_validate(self.write_record(directory, record))


if __name__ == "__main__":
    unittest.main()
