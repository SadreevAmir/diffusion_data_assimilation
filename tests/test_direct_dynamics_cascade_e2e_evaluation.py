import unittest

import torch

from assim_lib.direct_dynamics_cascade_e2e_evaluation import (
    _coarse_lift_ensemble,
    _summary,
    _validate_experiment,
)


class CascadeE2EEvaluationTests(unittest.TestCase):
    def test_contract_requires_frozen_pair_and_cases(self) -> None:
        experiment = {
            "cases": 12,
            "members": 8,
            "coarse_rk4_timepoints": 17,
            "fine_rk4_timepoints": 17,
            "case_indices": list(range(12)),
            "case_ids": [f"case-{index}" for index in range(12)],
            "coarse": {"checkpoint": "ema_coarse_update_9711.pth"},
            "fine_checkpoints": [
                {"label": "raw_fine_512", "checkpoint": "raw.pth", "sha256": "a" * 64},
                {"label": "ema_fine_512", "checkpoint": "ema.pth", "sha256": "b" * 64},
            ],
        }
        coarse, fine = _validate_experiment(experiment)
        self.assertEqual(coarse["checkpoint"], "ema_coarse_update_9711.pth")
        self.assertEqual([row["label"] for row in fine], ["raw_fine_512", "ema_fine_512"])
        experiment["fine_checkpoints"] = [
            {"label": "raw_fine_2048", "checkpoint": "raw.pth", "sha256": "a" * 64},
            {"label": "ema_fine_2048", "checkpoint": "ema.pth", "sha256": "b" * 64},
        ]
        _, fine = _validate_experiment(experiment)
        self.assertEqual([row["label"] for row in fine], ["raw_fine_2048", "ema_fine_2048"])
        experiment["coarse_rk4_timepoints"] = 33
        experiment["fine_rk4_timepoints"] = 33
        _validate_experiment(experiment)
        experiment["fine_rk4_timepoints"] = 65
        with self.assertRaisesRegex(ValueError, "matched coarse/fine"):
            _validate_experiment(experiment)
        experiment["coarse_rk4_timepoints"] = 17
        experiment["fine_rk4_timepoints"] = 17
        experiment["fine_checkpoints"][1]["label"] = "ema_fine_512"
        with self.assertRaisesRegex(ValueError, "same update"):
            _validate_experiment(experiment)
        experiment["members"] = 7
        with self.assertRaisesRegex(ValueError, "12 cases and 8 members"):
            _validate_experiment(experiment)

    def test_coarse_lift_is_exact_for_partial_cells(self) -> None:
        generator = torch.Generator().manual_seed(41)
        coarse = torch.randn((2, 3, 6, 3, 4), generator=generator)
        valid = torch.ones((2, 1, 6, 8))
        valid[:, :, :1, :3] = 0
        lifted, error = _coarse_lift_ensemble(coarse, valid)
        self.assertEqual(tuple(lifted.shape), (2, 3, 6, 6, 8))
        self.assertLess(error, 3e-6)

    def test_summary_averages_all_six_outputs(self) -> None:
        row = {
            "case_equal_ensemble_mean_rmse": 0.8,
            "case_equal_persistence_rmse": 1.0,
            "skill_over_persistence": 0.2,
            "case_equal_fair_crps": 0.6,
            "case_equal_persistence_point_mass_crps": 1.0,
            "spread_skill_ratio": 0.5,
            "rank_tv_to_uniform": 0.1,
            "normalized_mean_rank": 0.5,
            "member_roughness": 2.0,
            "truth_roughness": 1.0,
        }
        metrics = {
            "outputs": {f"output-{index}": dict(row) for index in range(6)},
            "joint_energy_score_common_normalized_coordinate": 0.7,
        }
        summary = _summary(metrics)
        self.assertAlmostEqual(summary["mean_rmse_skill_over_persistence"], 0.2)
        self.assertAlmostEqual(summary["mean_fair_crps_ratio_to_persistence"], 0.6)
        self.assertEqual(summary["max_member_to_truth_roughness_ratio"], 2.0)


if __name__ == "__main__":
    unittest.main()
