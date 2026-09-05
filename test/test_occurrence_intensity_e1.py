import json
import tempfile
import unittest
from pathlib import Path

from assim_lib.occurrence_intensity_e1 import (
    controller_request, run_engineering_sentinel, validate_config,
)
from paper.validate_occurrence_intensity_e1_compact_result import validate_result


CONFIG = Path("config/experiments/occurrence_intensity_e1_sentinel.json")


class E1HandoffTest(unittest.TestCase):
    def test_literal_handoff_is_fail_closed_pending_review(self):
        request = controller_request(CONFIG)
        self.assertFalse(request["launch_authorized"])
        self.assertFalse(request["scientific_gate"])
        self.assertTrue(request["clearml_enabled"])
        self.assertEqual(request["cases"], 8)

    def test_clearml_and_rank_semantics_cannot_drift(self):
        config = json.loads(CONFIG.read_text())
        config["clearml"]["enabled"] = False
        with self.assertRaisesRegex(ValueError, "clearml"):
            validate_config(config)
        config = json.loads(CONFIG.read_text())
        config["rank_gate"] = True
        with self.assertRaisesRegex(ValueError, "rank gate"):
            validate_config(config)

    def test_channel_layout_and_schema_cannot_drift(self):
        config = json.loads(CONFIG.read_text())
        config["dataset"]["channel_layout_per_lag"][0] = "value"
        with self.assertRaisesRegex(ValueError, "channel_layout_per_lag"):
            validate_config(config)
        config = json.loads(CONFIG.read_text())
        config["dataset"]["unreviewed"] = True
        with self.assertRaisesRegex(ValueError, "dataset keys must be exact"):
            validate_config(config)
        config = json.loads(CONFIG.read_text())
        config["unreviewed"] = True
        with self.assertRaisesRegex(ValueError, "config keys must be exact"):
            validate_config(config)

    def test_complete_eight_case_runtime_emits_compact_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run_engineering_sentinel(CONFIG, directory)
            self.assertEqual(result["run_status"]["status"], "completed")
            self.assertEqual(result["run_status"]["cases"], 8)
            self.assertFalse(result["run_status"]["scientific_gate"])
            manifest = result["artifact_manifest"]
            self.assertEqual(manifest["conditioning_shape"], [8, 24, 5, 4])
            self.assertEqual(manifest["target_shape"], [8, 2, 5, 4])
            self.assertEqual(manifest["sample_shape"], [8, 10, 1, 5, 4])
            self.assertEqual(manifest["exact_one_policy"], "explicit_exact_one_atom")
            self.assertTrue(manifest["sample_finite"])
            self.assertEqual(manifest["raw_arrays"], "not_persisted")
            self.assertEqual(
                validate_result(result["run_status"], manifest, CONFIG.read_bytes()),
                "E1_SENTINEL_PASS",
            )
            self.assertTrue((Path(directory) / "run_status.json").is_file())
            self.assertTrue((Path(directory) / "artifact_manifest.json").is_file())
            for panel in manifest["panel_artifacts"]:
                panel_path = Path(directory) / panel
                self.assertTrue(panel_path.is_file())
                self.assertTrue(panel_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))


if __name__ == "__main__":
    unittest.main()
