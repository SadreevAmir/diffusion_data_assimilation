#!/usr/bin/env python3
"""Focused tests for the explicit admission-record CLI boundary."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from paper.rank_coherent_reference import frozen_contract_sha256
from paper import test_rank_coherent_runner_prototype as prototype_fixture
from paper.validate_rank_coherent_admission import (
    directory_sha256,
    load_and_validate,
    load_and_validate_combined,
)


PAPER_DIR = Path(__file__).resolve().parent
SCRIPT = PAPER_DIR / "validate_rank_coherent_admission.py"


def valid_record() -> dict[str, object]:
    return {
        "reviewed_mode": "validation_reviewed_rank_coherent",
        "publication_commit": "a" * 40,
        "runner_sha256": "b" * 64,
        "contract_sha256": frozen_contract_sha256(),
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

    def decision_bearing_directory(self) -> tuple[tempfile.TemporaryDirectory, Path]:
        temporary, output, process = prototype_fixture.PrototypeTests.invoke(
            self, prototype_fixture.synthetic_source()
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        metadata_path = output / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["decision_bearing"] = True
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        return temporary, output

    def test_valid_record_returns_atomic_identities(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self.write_record(Path(temporary), valid_record())
            self.assertEqual(load_and_validate(path), "validation_reviewed_rank_coherent")
            result_directory, output = self.decision_bearing_directory()
            self.addCleanup(result_directory.cleanup)
            admitted = load_and_validate_combined(path, output)
            self.assertEqual(admitted.reviewed_mode, "validation_reviewed_rank_coherent")
            self.assertEqual(admitted.admission_record_sha256, __import__("hashlib").sha256(path.read_bytes()).hexdigest())
            self.assertEqual(admitted.compact_directory_sha256, directory_sha256(output))
            result = subprocess.run(
                [sys.executable, str(SCRIPT), str(path), str(output)],
                check=True,
                capture_output=True,
                text=True,
            )
        self.assertEqual(json.loads(result.stdout), admitted.__dict__)
        self.assertEqual(result.stderr, "")

    def test_input_substitutions_during_combined_admission_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self.write_record(Path(temporary), valid_record())
            result_directory, output = self.decision_bearing_directory()
            self.addCleanup(result_directory.cleanup)

            def mutate_directory(_path: Path, *, decision_bearing: bool) -> None:
                metadata_path = output / "metadata.json"
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

            with patch(
                "paper.validate_rank_coherent_admission.validate_result_directory",
                side_effect=mutate_directory,
            ):
                with self.assertRaisesRegex(ValueError, "compact directory changed"):
                    load_and_validate_combined(path, output)

            metadata_path = output / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

            def mutate_record(_path: Path, *, decision_bearing: bool) -> None:
                path.write_text(json.dumps(valid_record(), indent=2), encoding="utf-8")

            with patch(
                "paper.validate_rank_coherent_admission.validate_result_directory",
                side_effect=mutate_record,
            ):
                with self.assertRaisesRegex(ValueError, "admission record changed"):
                    load_and_validate_combined(path, output)

    def test_non_object_and_deviation_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with self.assertRaisesRegex(ValueError, "JSON object"):
                load_and_validate(self.write_record(directory, []))
            record = valid_record()
            record["deviations"] = ["waiver"]
            with self.assertRaisesRegex(ValueError, "empty list"):
                load_and_validate(self.write_record(directory, record))

    def test_well_formed_but_unbound_contract_digest_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            record = valid_record()
            record["contract_sha256"] = "c" * 64
            with self.assertRaisesRegex(ValueError, "frozen local contract"):
                load_and_validate(self.write_record(Path(temporary), record))


if __name__ == "__main__":
    unittest.main()
