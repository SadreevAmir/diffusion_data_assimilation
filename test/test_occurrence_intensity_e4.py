import json
import tempfile
import unittest
from pathlib import Path

import torch

from assim_lib.occurrence_intensity_e4 import (
    add_complete_residual_fields, controller_request, forecast_state_features,
    run_engineering_sentinel, select_complete_analogs,
)

CONFIG = Path("config/experiments/occurrence_intensity_e4_sentinel.json")


class E4StructuredAnalogTest(unittest.TestCase):
    def test_features_are_six_forecast_only_finite_values(self):
        fields = torch.stack((torch.zeros((3, 4)), torch.ones((3, 4)) * 0.2))
        features = forecast_state_features(fields)
        self.assertEqual(features.shape, (2, 6))
        self.assertTrue(torch.all(torch.isfinite(features)))
        self.assertEqual(features[0, 4:].tolist(), [0.5, 0.5])

    def test_selection_uses_chronological_ties(self):
        training = torch.zeros((12, 6))
        query = torch.zeros((1, 6))
        indices, distances = select_complete_analogs(training, query)
        self.assertEqual(indices.tolist(), [list(range(10))])
        self.assertEqual(distances.tolist(), [[0.0] * 10])

    def test_complete_fields_are_added_without_pixelwise_resampling(self):
        base = torch.full((1, 2, 2, 3), 0.5)
        residuals = torch.stack((torch.full((2, 3), 0.1), torch.full((2, 3), -0.2)))
        result = add_complete_residual_fields(base, residuals, torch.tensor([[0, 1]]))
        self.assertTrue(torch.allclose(result[0, 0] - base[0, 0], residuals[0]))
        self.assertTrue(torch.allclose(result[0, 1] - base[0, 1], residuals[1]))

    def test_out_of_support_fails_instead_of_clipping(self):
        with self.assertRaisesRegex(ValueError, "clipping is prohibited"):
            add_complete_residual_fields(torch.full((1, 1, 1, 1), 0.99),
                                         torch.full((1, 1, 1), 0.02),
                                         torch.zeros((1, 1), dtype=torch.long))

    def test_config_mutation_is_rejected_and_request_never_authorizes(self):
        self.assertFalse(controller_request(CONFIG)["launch_authorized"])
        config = json.loads(CONFIG.read_text())
        config["neighbors"] = 9
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mutated.json"
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "differs"):
                controller_request(path)

    def test_sentinel_emits_only_compact_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run_engineering_sentinel(CONFIG, directory)
            manifest = result["artifact_manifest"]
            self.assertEqual(manifest["clipping_calls"], 0)
            self.assertEqual(manifest["complete_field_shape"], [5, 4])
            self.assertEqual({p.name for p in Path(directory).iterdir()},
                             {"run_status.json", "artifact_manifest.json"})


if __name__ == "__main__":
    unittest.main()
