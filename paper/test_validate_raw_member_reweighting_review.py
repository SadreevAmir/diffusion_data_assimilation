#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from paper.validate_raw_member_reweighting_review import (
    CONTRACT,
    REQUIRED_KEYS,
    build_admission_payload,
    load_and_validate,
    validate_record,
    validate_semantic_parity,
)


REFERENCE = Path(__file__).with_name("raw_member_reweighting_reference.py")
RUNNER = Path(__file__).with_name("raw_member_reweighting_runner.py")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def valid_record(runner: Path, synthetic: Path) -> dict[str, object]:
    return {
        "reviewed_mode": "validation_reviewed_raw_member_reweighting",
        "publication_commit": "a" * 40,
        "runner_sha256": digest(runner),
        "contract_sha256": digest(CONTRACT),
        "synthetic_result_sha256": digest(synthetic),
        "test_command": "python -m unittest paper.test_runner",
        "test_sentinel": "raw_member_reweighting_runner=PASS",
        "decision_bearing_validation": "PASS",
        "deviations": [],
    }


class ReviewAdmissionTests(unittest.TestCase):
    def test_reference_satisfies_semantic_surface(self):
        validate_semantic_parity(REFERENCE)

    def test_separate_runner_satisfies_semantic_surface(self):
        validate_semantic_parity(RUNNER)

    def test_oracle_drift_fails_closed(self):
        source = REFERENCE.read_text(encoding="utf-8").replace(
            "return positions\n\n\ndef selection_diagnostics",
            "return list(reversed(positions))\n\n\ndef selection_diagnostics",
        )
        with tempfile.TemporaryDirectory() as temporary:
            runner = Path(temporary) / "runner.py"
            runner.write_text(source, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "diverges"):
                validate_semantic_parity(runner)

    def test_exact_record_and_artifact_bindings(self):
        with tempfile.TemporaryDirectory() as temporary:
            synthetic = Path(temporary) / "synthetic.json"
            synthetic.write_text('{"status":"PASS"}\n', encoding="utf-8")
            record = valid_record(RUNNER, synthetic)
            self.assertEqual(
                validate_record(record, RUNNER, synthetic), record["reviewed_mode"]
            )
            for key in ("runner_sha256", "contract_sha256", "synthetic_result_sha256"):
                bad = dict(record)
                bad[key] = "0" * 64
                with self.assertRaisesRegex(ValueError, key):
                    validate_record(bad, RUNNER, synthetic)

    def test_every_placeholder_and_nonpass_state_is_no_go(self):
        with tempfile.TemporaryDirectory() as temporary:
            synthetic = Path(temporary) / "synthetic.json"
            synthetic.write_text("{}", encoding="utf-8")
            good = valid_record(RUNNER, synthetic)
            mutations = (
                ("reviewed_mode", "<literal implemented trusted mode>"),
                ("publication_commit", None),
                ("test_command", "<exact command>"),
                ("test_sentinel", ""),
                ("decision_bearing_validation", "FAIL"),
                ("deviations", ["waiver"]),
            )
            for key, value in mutations:
                bad = dict(good)
                bad[key] = value
                with self.assertRaises((ValueError, TypeError), msg=key):
                    validate_record(bad, RUNNER, synthetic)
            bad = dict(good)
            bad["extra"] = True
            self.assertNotEqual(set(bad), REQUIRED_KEYS)
            with self.assertRaisesRegex(ValueError, "exact-schema"):
                validate_record(bad, RUNNER, synthetic)

    def test_combined_admission_returns_only_bound_literal_mode(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            synthetic = root / "synthetic.json"
            synthetic.write_text('{"status":"PASS"}\n', encoding="utf-8")
            record_path = root / "review.json"
            record_path.write_text(
                json.dumps(valid_record(RUNNER, synthetic), sort_keys=True),
                encoding="utf-8",
            )
            self.assertEqual(
                load_and_validate(record_path, RUNNER, synthetic),
                "validation_reviewed_raw_member_reweighting",
            )

    def test_admission_payload_carries_every_verified_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            synthetic = root / "synthetic.json"
            synthetic.write_text('{"status":"PASS"}\n', encoding="utf-8")
            record_path = root / "review.json"
            record = valid_record(RUNNER, synthetic)
            record_path.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
            payload = build_admission_payload(record_path, RUNNER, synthetic)
            self.assertEqual(payload["admission"], "GO")
            self.assertEqual(payload["reviewed_mode"], record["reviewed_mode"])
            self.assertEqual(payload["review_record_sha256"], digest(record_path))
            for key in (
                "publication_commit",
                "runner_sha256",
                "contract_sha256",
                "synthetic_result_sha256",
            ):
                self.assertEqual(payload[key], record[key])


if __name__ == "__main__":
    unittest.main()
