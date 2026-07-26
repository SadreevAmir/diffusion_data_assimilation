import unittest

from assim_lib.config import TrainingConfig, merge_config_overrides


class ConfigTests(unittest.TestCase):
    def test_nested_overrides_are_recursive_and_non_mutating(self):
        base = {
            "observation_mask": {
                "kind": "sral_tracks",
                "synthetic": {"kind": "generated_track", "n_tracks_range": [5, 9]},
            }
        }
        overrides = {"observation_mask": {"synthetic": {"n_tracks_range": [1, 2]}}}

        merged = merge_config_overrides(base, overrides)

        self.assertEqual(merged["observation_mask"]["kind"], "sral_tracks")
        self.assertEqual(merged["observation_mask"]["synthetic"]["kind"], "generated_track")
        self.assertEqual(merged["observation_mask"]["synthetic"]["n_tracks_range"], [1, 2])
        self.assertEqual(
            base["observation_mask"]["synthetic"]["n_tracks_range"],
            [5, 9],
        )

    def test_training_config_rejects_incompatible_residual_bridge(self):
        with self.assertRaisesRegex(ValueError, "incompatible"):
            TrainingConfig.from_dict(
                {
                    "training_objective": "residual_flow",
                    "sample_start_mode": "bridge",
                }
            )

    def test_legacy_objective_name_is_normalized(self):
        config = TrainingConfig.from_dict({"training_objective": "diffusion_residual"})
        self.assertEqual(config.training_objective, "residual_flow")


if __name__ == "__main__":
    unittest.main()
