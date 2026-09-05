import json
import unittest
from pathlib import Path

from assim_lib.occurrence_intensity_e1 import controller_request, validate_config


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


if __name__ == "__main__":
    unittest.main()
