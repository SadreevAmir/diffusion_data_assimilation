from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from paper.casewise_safety_selector_runner_prototype import __file__ as RUNNER_NAME
from paper.test_validate_casewise_safety_selector_compact_outputs import CompactOutputTests, valid_payloads
from paper.validate_casewise_safety_selector_admission import (
    FROZEN_ARTIFACTS, PAPER_DIR, directory_sha256, load_and_validate, sha256_file,
    validate_record, validate_semantic_parity,
)
from paper import casewise_safety_selector_reference as reference

RUNNER = Path(RUNNER_NAME)


class AdmissionTests(unittest.TestCase):
    def synthetic_directory(self):
        return CompactOutputTests.write(self, valid_payloads())

    def record(self, directory):
        return {
            "schema_version": "casewise-safety-selector-admission-v1",
            "reviewed_mode": "validation_reviewed_casewise_safety_selector",
            "publication_commit": "a" * 40,
            "runner_sha256": sha256_file(RUNNER),
            "frozen_artifact_sha256": {name: sha256_file(PAPER_DIR / name) for name in FROZEN_ARTIFACTS},
            "synthetic_result_sha256": directory_sha256(directory),
            "decision_bearing_validation": "PASS",
            "deviations": [],
        }

    def test_positive_atomic_fixture(self):
        directory = self.synthetic_directory(); record = self.record(directory)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "admission.json"; path.write_text(json.dumps(record), encoding="utf-8")
            if reference.np is None:
                with self.assertRaisesRegex(RuntimeError, "numpy is required"):
                    load_and_validate(path, RUNNER, directory)
                return
            result = load_and_validate(path, RUNNER, directory)
        self.assertEqual(result.reviewed_mode, record["reviewed_mode"])

    def test_record_rejects_identity_and_review_drift(self):
        directory = self.synthetic_directory(); good = self.record(directory)
        for key in ("runner_sha256", "synthetic_result_sha256"):
            bad = copy.deepcopy(good); bad[key] = "0" * 64
            with self.assertRaises(ValueError): validate_record(bad, RUNNER, directory)
        bad = copy.deepcopy(good); bad["frozen_artifact_sha256"][FROZEN_ARTIFACTS[0]] = "0" * 64
        with self.assertRaisesRegex(ValueError, "digest mismatch"): validate_record(bad, RUNNER, directory)
        for key, value in (("decision_bearing_validation", "FAIL"), ("deviations", ["drift"])):
            bad = copy.deepcopy(good); bad[key] = value
            with self.assertRaisesRegex(ValueError, "deviations"): validate_record(bad, RUNNER, directory)

    def test_semantic_parity_and_changed_runner_fail_closed(self):
        source = RUNNER.read_text(encoding="utf-8").replace("ACTION_MARGIN = 0.002", "ACTION_MARGIN = 0.003")
        with tempfile.TemporaryDirectory() as temporary:
            changed = Path(temporary) / "runner.py"; changed.write_text(source, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "ACTION_MARGIN"): validate_semantic_parity(changed)
        if reference.np is None:
            with self.assertRaisesRegex(RuntimeError, "numpy is required"):
                validate_semantic_parity(RUNNER)
        else:
            validate_semantic_parity(RUNNER)

    def test_synthetic_payload_drift_fails_closed(self):
        directory = self.synthetic_directory(); frozen = self.record(directory)
        gate = directory / "gate_decision.json"
        payload = json.loads(gate.read_text(encoding="utf-8")); payload["overall_eligible"] = False
        gate.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "synthetic_result_sha256"): validate_record(frozen, RUNNER, directory)


if __name__ == "__main__":
    unittest.main()
