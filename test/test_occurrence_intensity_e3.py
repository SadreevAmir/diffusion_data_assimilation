import json
import tempfile
import unittest
from pathlib import Path

import torch

from assim_lib.occurrence_intensity_e3 import (
    controller_request, fit_interior_law, inverse_interior,
    run_engineering_sentinel, transform_interior,
)

CONFIG = Path("config/experiments/occurrence_intensity_e3_sentinel.json")


class E3AnamorphosisTest(unittest.TestCase):
    def test_atoms_duplicates_and_small_positive_roundtrip(self):
        training = torch.tensor([0.0, 1e-8, 0.1, 0.1, 0.4, 0.9, 1.0], dtype=torch.float64)
        values = torch.tensor([0.0, 1e-8, 0.1, 0.4, 0.9, 1.0], dtype=torch.float64)
        law = fit_interior_law(training)
        coordinates = transform_interior(values, law)
        restored = inverse_interior(coordinates, values, law)
        self.assertTrue(torch.equal(restored[[0, -1]], values[[0, -1]]))
        self.assertLessEqual(float(torch.max(torch.abs(restored - values))), 1e-7)
        self.assertTrue(torch.all(torch.diff(coordinates[1:-1]) > 0))

    def test_inverse_rejects_extrapolation_instead_of_clipping(self):
        template = torch.tensor([0.5])
        law = fit_interior_law(torch.tensor([0.0, 0.2, 0.8, 1.0]))
        with self.assertRaisesRegex(ValueError, "outside"):
            inverse_interior(torch.tensor([0.0]), template, law)

    def test_nonfinite_and_out_of_range_inputs_fail_closed(self):
        law = fit_interior_law(torch.tensor([0.2, 0.8]))
        for invalid in (torch.tensor([float("nan")]), torch.tensor([1.1])):
            with self.assertRaises(ValueError):
                transform_interior(invalid, law)

    def test_config_mutation_is_rejected_and_request_never_authorizes(self):
        request = controller_request(CONFIG)
        self.assertFalse(request["launch_authorized"])
        config = json.loads(CONFIG.read_text())
        config["hard_clipping"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mutated.json"
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "differs"):
                controller_request(path)

    def test_sentinel_emits_compact_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run_engineering_sentinel(CONFIG, directory)
            self.assertEqual(result["artifact_manifest"]["clipping_calls"], 0)
            self.assertLessEqual(result["artifact_manifest"]["roundtrip_max_abs_error"], 1e-7)
            self.assertEqual({p.name for p in Path(directory).iterdir()},
                             {"run_status.json", "artifact_manifest.json"})


if __name__ == "__main__":
    unittest.main()
