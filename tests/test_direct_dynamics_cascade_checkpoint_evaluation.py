import unittest

import torch

from assim_lib.direct_dynamics_cascade_checkpoint_evaluation import (
    EXPECTED_LABELS,
    _even_positions,
    _predicted_open_water_sit,
    _validate_experiment,
)


class CascadeCheckpointEvaluationTest(unittest.TestCase):
    def test_even_positions_are_endpoint_inclusive_and_unique(self):
        positions = _even_positions(48, 12)
        self.assertEqual((positions[0], positions[-1]), (0, 47))
        self.assertEqual(len(set(positions)), 12)

    def test_contract_requires_ordered_raw_and_ema_pairs(self):
        experiment = {
            "cases": 12,
            "members": 8,
            "rk4_timepoints": 17,
            "checkpoints": [
                {
                    "label": label,
                    "checkpoint": ("ema_" if label.startswith("ema_") else "") + label.replace("ema_", "") + ".pth",
                    "sha256": "a" * 64,
                }
                for label in EXPECTED_LABELS
            ],
        }
        self.assertEqual(len(_validate_experiment(experiment)), 4)
        experiment["checkpoints"].reverse()
        with self.assertRaisesRegex(ValueError, "ordered"):
            _validate_experiment(experiment)

    def test_open_water_sit_uses_each_members_predicted_sic(self):
        sic = torch.tensor([[[[[0.0, 0.5]]], [[[0.0, 0.0]]]]])
        sit = torch.tensor([[[[[0.02, 2.0]]], [[[0.0, 0.03]]]]])
        fraction = torch.ones((1, 1, 1, 2))
        result = _predicted_open_water_sit(sic, sit, fraction)
        self.assertAlmostEqual(result["case_equal_mean_positive_sit"], (0.02 + 0.0 + 0.03) / 3)
        self.assertAlmostEqual(result["case_equal_fraction_above_0p01m"], 2 / 3)
        self.assertAlmostEqual(result["global_positive_sit_max"], 0.03)


if __name__ == "__main__":
    unittest.main()
