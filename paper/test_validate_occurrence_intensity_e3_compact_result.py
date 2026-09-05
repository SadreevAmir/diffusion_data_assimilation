import copy
import hashlib
import unittest

from paper.validate_occurrence_intensity_e3_compact_result import validate_result


CONFIG = b'{"frozen":true}\n'


def valid_result():
    status = {"status": "completed", "selection_role": "engineering_only", "cases": 8,
              "scientific_gate": False}
    manifest = {
        "config_sha256": hashlib.sha256(CONFIG).hexdigest(),
        "training_interior_count": 7, "unique_knot_count": 6,
        "roundtrip_max_abs_error": 1e-8, "zero_atoms_preserved": True,
        "one_atoms_preserved": True, "clipping_calls": 0,
        "raw_arrays": "not_persisted",
    }
    return status, manifest


class E3CompactResultTest(unittest.TestCase):
    def test_valid_sentinel_passes(self):
        status, manifest = valid_result()
        self.assertEqual(validate_result(status, manifest, CONFIG), "E3_SENTINEL_PASS")

    def test_status_and_schema_drift_fail_closed(self):
        status, manifest = valid_result()
        status["scientific_gate"] = True
        with self.assertRaisesRegex(ValueError, "non-scientific"):
            validate_result(status, manifest, CONFIG)
        status, manifest = valid_result(); manifest["extra"] = True
        with self.assertRaisesRegex(ValueError, "keys must be exact"):
            validate_result(status, manifest, CONFIG)

    def test_digest_support_and_roundtrip_are_recomputed(self):
        for key, value, message in (
            ("config_sha256", "0" * 64, "does not bind"),
            ("unique_knot_count", 7, "support counts"),
            ("roundtrip_max_abs_error", 2e-7, "exceeds"),
        ):
            status, manifest = valid_result(); manifest[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, message):
                validate_result(status, manifest, CONFIG)

    def test_atoms_clipping_and_inventory_fail_closed(self):
        for key, value in (("zero_atoms_preserved", False), ("one_atoms_preserved", False),
                           ("clipping_calls", 1), ("raw_arrays", "persisted")):
            status, manifest = valid_result(); manifest[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_result(status, manifest, CONFIG)


if __name__ == "__main__":
    unittest.main()
