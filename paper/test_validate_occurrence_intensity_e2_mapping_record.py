import copy
import hashlib
import json
import unittest
from pathlib import Path

from paper.test_validate_occurrence_intensity_e2_compact_result import valid_result
from paper.validate_occurrence_intensity_e2_mapping_record import validate_mapping_record

PAPER = Path(__file__).parent
REQUEST = json.loads((PAPER / "OCCURRENCE_INTENSITY_E1_ADMISSION_REQUEST.json").read_text())
PASS = {"decision": "PASS", "failed_checks": [], "failed_tests": []}
CONTRACT = (PAPER / "NEXT_OCCURRENCE_INTENSITY_E2_CONTRACT.md").read_bytes()


def _fixture():
    synthetic = json.dumps(valid_result(), sort_keys=True).encode()
    return {"e1_admission_decision": "PASS", "reviewed_mode": "reviewed_occurrence_intensity_e2", "publication_commit": "a" * 40, "runner_sha256": "b" * 64, "contract_sha256": hashlib.sha256(CONTRACT).hexdigest(), "synthetic_result_sha256": hashlib.sha256(synthetic).hexdigest(), "test_command": "python -m unittest paper.test_occurrence_intensity_e2", "test_sentinel": "E2_MAPPING_AND_RESULT_PASS", "decision_bearing_validation": "PASS", "deviations": []}, synthetic


class OccurrenceIntensityE2MappingRecordTest(unittest.TestCase):
    def test_complete_record_is_go(self):
        record, synthetic = _fixture()
        self.assertEqual(validate_mapping_record(record, REQUEST, PASS, CONTRACT, synthetic), "GO")

    def test_e1_reject_or_record_decision_drift_fails_closed(self):
        record, synthetic = _fixture()
        reject = {"decision": "REJECT", "failed_checks": [REQUEST["required_checks"][0]], "failed_tests": []}
        with self.assertRaisesRegex(ValueError, "literal PASS"):
            validate_mapping_record(record, REQUEST, reject, CONTRACT, synthetic)
        record["e1_admission_decision"] = "REJECT"
        with self.assertRaisesRegex(ValueError, "literal E1 PASS"):
            validate_mapping_record(record, REQUEST, PASS, CONTRACT, synthetic)

    def test_contract_and_synthetic_digests_are_bound(self):
        record, synthetic = _fixture()
        for key in ("contract_sha256", "synthetic_result_sha256"):
            changed = copy.deepcopy(record)
            changed[key] = "0" * 64
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "does not bind"):
                validate_mapping_record(changed, REQUEST, PASS, CONTRACT, synthetic)

    def test_synthetic_result_must_pass_semantic_oracle(self):
        record, _ = _fixture()
        invalid = valid_result()
        invalid["decision"] = "E2_INVALID"
        synthetic = json.dumps(invalid, sort_keys=True).encode()
        record["synthetic_result_sha256"] = hashlib.sha256(synthetic).hexdigest()
        with self.assertRaisesRegex(ValueError, "decision must equal recomputed"):
            validate_mapping_record(record, REQUEST, PASS, CONTRACT, synthetic)

    def test_formats_deviations_and_schema_drift_fail_closed(self):
        record, synthetic = _fixture()
        mutations = {"reviewed_mode": "<placeholder>", "publication_commit": "A" * 40, "runner_sha256": "g" * 64, "deviations": ["waiver"], "decision_bearing_validation": "FAIL"}
        for key, value in mutations.items():
            changed = copy.deepcopy(record)
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_mapping_record(changed, REQUEST, PASS, CONTRACT, synthetic)
        changed = copy.deepcopy(record)
        changed["launch_authorized"] = True
        with self.assertRaisesRegex(ValueError, "keys must be exact"):
            validate_mapping_record(changed, REQUEST, PASS, CONTRACT, synthetic)


if __name__ == "__main__":
    unittest.main()
