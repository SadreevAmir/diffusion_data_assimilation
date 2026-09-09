import unittest

import torch

from assim_lib.censored_joint_spatial_diagnostics import (
    SPATIAL_LAGS,
    mean_absolute_increment,
    snapshot_metrics,
)


class CensoredJointSpatialDiagnosticsTests(unittest.TestCase):
    def test_linear_field_has_exact_increment(self) -> None:
        x = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
        valid = torch.ones_like(x, dtype=torch.bool)
        self.assertAlmostEqual(
            mean_absolute_increment(x, valid, 1), 41.0 / 17.0, places=6
        )

    def test_snapshot_reports_zero_and_one_atoms(self) -> None:
        ensemble = torch.zeros(1, 2, 6, 12, 12)
        ensemble[:, 1, 0::2] = 1
        payload = {
            "physical_ensemble": ensemble,
            "truth": torch.zeros(1, 6, 12, 12),
            "valid_mask": torch.ones(1, 1, 12, 12),
        }
        metrics = snapshot_metrics(payload)
        sic = metrics["outputs"]["d3_sic"]
        sit = metrics["outputs"]["d3_sit"]
        self.assertEqual(tuple(SPATIAL_LAGS), (1, 2, 4, 8))
        self.assertAlmostEqual(sic["atoms"]["zero_frequency"], 0.5)
        self.assertAlmostEqual(sic["atoms"]["one_frequency"], 0.5)
        self.assertAlmostEqual(sit["atoms"]["zero_frequency"], 1.0)
        self.assertNotIn("one_frequency", sit["atoms"])


if __name__ == "__main__":
    unittest.main()
