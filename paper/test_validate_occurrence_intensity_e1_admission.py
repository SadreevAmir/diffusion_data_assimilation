import copy
import json
import os
import unittest
from pathlib import Path

from paper.validate_occurrence_intensity_e1_admission import validate_request, validate_response


REQUEST = json.loads(
    (Path(__file__).with_name("OCCURRENCE_INTENSITY_E1_ADMISSION_REQUEST.json")).read_text(encoding="utf-8")
)


class OccurrenceIntensityE1AdmissionTest(unittest.TestCase):
    def test_runner_is_executable_and_validates_emitted_compact_result(self):
        runner = Path(REQUEST["runner"])
        self.assertTrue(os.access(runner, os.X_OK))
        source = runner.read_text(encoding="utf-8")
        self.assertIn("paper/validate_occurrence_intensity_e1_compact_result.py", source)
        self.assertIn('"$OUTPUT_DIR/run_status.json"', source)
        self.assertIn('"$OUTPUT_DIR/artifact_manifest.json"', source)
        self.assertIn('command -v python3', source)
        self.assertIn('command -v python', source)
        self.assertIn('set PYTHON_BIN explicitly', source)

    def test_current_request_and_literal_pass(self):
        validate_request(REQUEST)
        self.assertEqual(
            validate_response(REQUEST, {"decision": "PASS", "failed_checks": [], "failed_tests": []}),
            "PASS",
        )

    def test_complete_reject_is_accepted(self):
        response = {
            "decision": "REJECT",
            "failed_checks": [REQUEST["required_checks"][0]],
            "failed_tests": ["suite.case: expected finite output, observed NaN"],
        }
        self.assertEqual(validate_response(REQUEST, response), "REJECT")

    def test_pass_with_failure_or_extra_field_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "PASS cannot contain failures"):
            validate_response(REQUEST, {"decision": "PASS", "failed_checks": [REQUEST["required_checks"][0]], "failed_tests": []})
        with self.assertRaisesRegex(ValueError, "keys must be exact"):
            validate_response(REQUEST, {"decision": "PASS", "failed_checks": [], "failed_tests": [], "launch_authorized": True})

    def test_incomplete_or_unknown_reject_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "at least one failure"):
            validate_response(REQUEST, {"decision": "REJECT", "failed_checks": [], "failed_tests": []})
        with self.assertRaisesRegex(ValueError, "unknown failed_checks"):
            validate_response(REQUEST, {"decision": "REJECT", "failed_checks": ["unreviewed_check"], "failed_tests": []})
        with self.assertRaisesRegex(ValueError, "identifier and message"):
            validate_response(REQUEST, {"decision": "REJECT", "failed_checks": [], "failed_tests": ["suite.case"]})

    def test_request_contract_drift_fails_closed(self):
        changed = copy.deepcopy(REQUEST)
        changed["launch_authorized"] = True
        with self.assertRaisesRegex(ValueError, "must not authorize launch"):
            validate_request(changed)

    def test_request_identities_and_checklist_cannot_drift(self):
        for key in ("config", "runner", "implementation", "tests", "compact_result_validator",
                    "required_checks", "required_test_command"):
            changed = copy.deepcopy(REQUEST)
            if isinstance(changed[key], list):
                changed[key] = list(reversed(changed[key]))
            else:
                changed[key] += ".changed"
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, f"unexpected {key}"):
                    validate_request(changed)


if __name__ == "__main__":
    unittest.main()
