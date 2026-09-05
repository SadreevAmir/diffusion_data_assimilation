import hashlib
import json
import unittest

from paper.test_validate_occurrence_intensity_e4_compact_result import CONFIG, valid_result
from paper.validate_occurrence_intensity_e4_mapping_record import validate_mapping_record

CONTRACT = b"frozen E4 contract\n"


def fixture():
    status, manifest = valid_result()
    status_bytes = json.dumps(status, sort_keys=True).encode()
    manifest_bytes = json.dumps(manifest, sort_keys=True).encode()
    record = {
        "upstream_base": "accepted_e3", "upstream_decision": "ATOM_AWARE_TRANSFORM_USEFUL",
        "reviewed_mode": "occurrence_intensity_e4_trusted_review",
        "publication_commit": "a" * 40, "runner_sha256": "b" * 64,
        "contract_sha256": hashlib.sha256(CONTRACT).hexdigest(),
        "config_sha256": hashlib.sha256(CONFIG).hexdigest(),
        "run_status_sha256": hashlib.sha256(status_bytes).hexdigest(),
        "artifact_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "test_command": "python3 -m unittest test.test_occurrence_intensity_e4",
        "test_sentinel": "E4_SENTINEL_PASS", "decision_bearing_validation": "PASS",
        "deviations": [],
    }
    return record, status_bytes, manifest_bytes


class E4MappingRecordTest(unittest.TestCase):
    def test_complete_record_reaches_review_only(self):
        record, status, manifest = fixture()
        self.assertEqual(validate_mapping_record(record, CONTRACT, CONFIG, status, manifest),
                         "READY_FOR_CONTROLLER_ADMISSION_REVIEW")

    def test_provenance_bindings_and_semantics_fail_closed(self):
        for key, value in (("upstream_decision", "STRUCTURED_ANALOG_NEGATIVE"),
                           ("contract_sha256", "0" * 64), ("deviations", ["waiver"]),
                           ("decision_bearing_validation", "FAIL")):
            record, status, manifest = fixture(); record[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_mapping_record(record, CONTRACT, CONFIG, status, manifest)

    def test_embedded_compact_result_is_revalidated(self):
        record, status, manifest = fixture()
        changed = json.loads(manifest); changed["clipping_calls"] = 1
        manifest = json.dumps(changed, sort_keys=True).encode()
        record["artifact_manifest_sha256"] = hashlib.sha256(manifest).hexdigest()
        with self.assertRaisesRegex(ValueError, "clipping_calls"):
            validate_mapping_record(record, CONTRACT, CONFIG, status, manifest)


if __name__ == "__main__":
    unittest.main()
