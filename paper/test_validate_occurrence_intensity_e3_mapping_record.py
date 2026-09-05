import hashlib
import json
import unittest

from paper.test_validate_occurrence_intensity_e3_compact_result import CONFIG, valid_result
from paper.validate_occurrence_intensity_e3_mapping_record import validate_mapping_record

CONTRACT = b"frozen E3 contract\n"


def _fixture():
    status, manifest = valid_result()
    status_bytes = json.dumps(status, sort_keys=True).encode()
    manifest_bytes = json.dumps(manifest, sort_keys=True).encode()
    record = {
        "upstream_base": "accepted_e1", "upstream_decision": "E1_ACCEPTED",
        "reviewed_mode": "occurrence_intensity_e3_trusted_review",
        "publication_commit": "a" * 40, "runner_sha256": "b" * 64,
        "contract_sha256": hashlib.sha256(CONTRACT).hexdigest(),
        "config_sha256": hashlib.sha256(CONFIG).hexdigest(),
        "run_status_sha256": hashlib.sha256(status_bytes).hexdigest(),
        "artifact_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "test_command": "python3 test/test_occurrence_intensity_e3.py",
        "test_sentinel": "E3_SENTINEL_PASS", "decision_bearing_validation": "PASS",
        "deviations": [],
    }
    return record, status_bytes, manifest_bytes


class E3MappingRecordTest(unittest.TestCase):
    def test_complete_record_reaches_review_only(self):
        record, status, manifest = _fixture()
        self.assertEqual(validate_mapping_record(record, CONTRACT, CONFIG, status, manifest),
                         "READY_FOR_CONTROLLER_ADMISSION_REVIEW")

    def test_upstream_must_be_an_accepted_named_base(self):
        for key, value in (("upstream_base", "<placeholder>"),
                           ("upstream_decision", "TEMPORAL_MECHANISM_NEGATIVE")):
            record, status, manifest = _fixture(); record[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_mapping_record(record, CONTRACT, CONFIG, status, manifest)

    def test_every_artifact_binding_is_recomputed(self):
        for key in ("contract_sha256", "config_sha256", "run_status_sha256",
                    "artifact_manifest_sha256"):
            record, status, manifest = _fixture(); record[key] = "0" * 64
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "does not bind"):
                validate_mapping_record(record, CONTRACT, CONFIG, status, manifest)

    def test_semantic_failure_deviation_and_schema_drift_fail_closed(self):
        record, status, manifest = _fixture()
        changed_manifest = json.loads(manifest); changed_manifest["clipping_calls"] = 1
        manifest = json.dumps(changed_manifest, sort_keys=True).encode()
        record["artifact_manifest_sha256"] = hashlib.sha256(manifest).hexdigest()
        with self.assertRaisesRegex(ValueError, "clipping_calls"):
            validate_mapping_record(record, CONTRACT, CONFIG, status, manifest)
        record, status, manifest = _fixture(); record["deviations"] = ["waiver"]
        with self.assertRaisesRegex(ValueError, "empty list"):
            validate_mapping_record(record, CONTRACT, CONFIG, status, manifest)
        record, status, manifest = _fixture(); record["launch_authorized"] = True
        with self.assertRaisesRegex(ValueError, "keys must be exact"):
            validate_mapping_record(record, CONTRACT, CONFIG, status, manifest)

    def test_test_identity_and_validation_must_be_literal(self):
        for key, value in (("test_command", "python3 some_other_test.py"),
                           ("test_sentinel", "PASS"),
                           ("decision_bearing_validation", "FAIL")):
            record, status, manifest = _fixture(); record[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_mapping_record(record, CONTRACT, CONFIG, status, manifest)


if __name__ == "__main__":
    unittest.main()
