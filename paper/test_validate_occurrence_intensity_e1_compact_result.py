import copy
import hashlib
import unittest

from paper.validate_occurrence_intensity_e1_compact_result import (
    PANEL_ARTIFACTS,
    validate_result,
)


CONFIG = b'{"frozen":true}\n'


def valid_result():
    status = {
        "status": "completed", "selection_role": "engineering_only", "cases": 8,
        "clearml_enabled": True, "scientific_gate": False,
    }
    manifest = {
        "config_sha256": hashlib.sha256(CONFIG).hexdigest(),
        "conditioning_shape": [8, 24, 5, 4], "target_shape": [8, 2, 5, 4],
        "sample_shape": [8, 10, 1, 5, 4],
        "exact_one_policy": "explicit_exact_one_atom", "sample_finite": True,
        "sample_in_unit_interval": True, "hard_clip_count": 0,
        "masked_channel_leakage": [0.0] * 8, "provenance_consistent": [True] * 8,
        "orientation_landmarks": [[True, True] for _ in range(8)],
        "candidate_track_imprint_ratio": [1.0] * 8,
        "background_track_imprint_ratio": [1.0] * 8,
        "panel_artifacts": PANEL_ARTIFACTS, "raw_arrays": "not_persisted",
    }
    return status, manifest


class E1CompactResultTest(unittest.TestCase):
    def test_complete_engineering_result_passes(self):
        self.assertEqual(validate_result(*valid_result(), CONFIG), "E1_SENTINEL_PASS")

    def test_schema_digest_and_non_scientific_status_fail_closed(self):
        mutations = []
        status, manifest = valid_result(); manifest["extra"] = True; mutations.append((status, manifest))
        status, manifest = valid_result(); manifest["config_sha256"] = "0" * 64; mutations.append((status, manifest))
        status, manifest = valid_result(); status["scientific_gate"] = True; mutations.append((status, manifest))
        for status, manifest in mutations:
            with self.subTest(), self.assertRaises(ValueError):
                validate_result(status, manifest, CONFIG)

    def test_each_engineering_invariant_fails_closed(self):
        changes = {
            "sample_finite": False,
            "sample_in_unit_interval": False,
            "hard_clip_count": 1,
            "masked_channel_leakage": [0.0] * 7 + [1.1e-7],
            "provenance_consistent": [True] * 7 + [False],
            "orientation_landmarks": [[True, True] for _ in range(7)] + [[True, False]],
            "candidate_track_imprint_ratio": [1.0] * 7 + [1.051],
            "panel_artifacts": PANEL_ARTIFACTS[:-1],
        }
        for key, value in changes.items():
            status, manifest = valid_result(); manifest[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_result(status, manifest, CONFIG)

    def test_ratios_and_leakage_require_complete_finite_vectors(self):
        for key, value in (
            ("candidate_track_imprint_ratio", [1.0] * 7),
            ("background_track_imprint_ratio", [1.0] * 7 + [float("nan")]),
            ("masked_channel_leakage", [0.0] * 7 + [-1.0]),
        ):
            status, manifest = valid_result(); manifest[key] = copy.deepcopy(value)
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_result(status, manifest, CONFIG)


if __name__ == "__main__":
    unittest.main()
