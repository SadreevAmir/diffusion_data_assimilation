import copy
import hashlib
import unittest

from paper.validate_occurrence_intensity_e4_compact_result import FEATURE_NAMES, validate_result

CONFIG = b'{"frozen":true}\n'


def valid_result():
    status = {"status": "completed", "selection_role": "engineering_only", "cases": 8,
              "scientific_gate": False}
    indices = [list(range(10)) for _ in range(8)]
    distances = [[float(i) for i in range(10)] for _ in range(8)]
    manifest = {
        "config_sha256": hashlib.sha256(CONFIG).hexdigest(), "feature_names": FEATURE_NAMES,
        "training_records": 12, "analog_indices": indices, "analog_distances": distances,
        "training_only_standardization": True, "chronological_ties": True,
        "complete_field_shape": [5, 4], "result_shape": [8, 10, 5, 4],
        "clipping_calls": 0, "raw_arrays": "not_persisted",
    }
    return status, manifest


class E4CompactResultTest(unittest.TestCase):
    def test_valid_sentinel_passes(self):
        self.assertEqual(validate_result(*valid_result(), CONFIG), "E4_SENTINEL_PASS")

    def test_schema_digest_and_literals_fail_closed(self):
        for key, value in (("extra", True), ("config_sha256", "0" * 64),
                           ("training_only_standardization", False), ("clipping_calls", 1)):
            status, manifest = valid_result(); manifest[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_result(status, manifest, CONFIG)

    def test_analog_inventory_order_and_ties_fail_closed(self):
        mutations = []
        _, base = valid_result()
        duplicate = copy.deepcopy(base); duplicate["analog_indices"][0][1] = 0; mutations.append(duplicate)
        unordered = copy.deepcopy(base); unordered["analog_distances"][0][:2] = [2.0, 1.0]; mutations.append(unordered)
        bad_tie = copy.deepcopy(base); bad_tie["analog_distances"][0][:2] = [0.0, 0.0]; bad_tie["analog_indices"][0][:2] = [1, 0]; mutations.append(bad_tie)
        nonfinite = copy.deepcopy(base); nonfinite["analog_distances"][0][0] = float("nan"); mutations.append(nonfinite)
        for manifest in mutations:
            with self.subTest(), self.assertRaises(ValueError):
                validate_result(valid_result()[0], manifest, CONFIG)


if __name__ == "__main__":
    unittest.main()
