import unittest

from assim_lib.direct_dynamics_cascade_colored_gate import (
    _candidate_summary,
    _compare,
    _validate_experiment,
)


class ColoredGateTests(unittest.TestCase):
    def _experiment(self):
        return {
            "split": "valid",
            "year": 2022,
            "cases": 12,
            "members": 8,
            "rk4_timepoints": [33, 65],
            "case_indices": list(range(12)),
            "case_ids": [f"case-{index}" for index in range(12)],
            "coarse": {"checkpoint": "ema_coarse_update_9711.pth"},
            "fine_candidates": [
                {
                    "label": label,
                    "checkpoint": "mechanics_update_0512.pth",
                    "checkpoint_sha256": "a" * 64,
                    "run_sha256": {},
                }
                for label in ("white_raw512", "colored_raw512")
            ],
            "gate": {
                "primary_standardized_crps_ratio_max": 1.01,
                "per_output_crps_ratio_max": 1.01,
                "per_output_rmse_ratio_max": 1.01,
                "per_output_rank_tv_increase_max": 0.01,
                "per_output_adjusted_ssr_error_increase_max": 0.02,
            },
        }

    def test_contract_is_validation_raw512_and_two_solvers(self):
        coarse, candidates = _validate_experiment(self._experiment())
        self.assertEqual(coarse["checkpoint"], "ema_coarse_update_9711.pth")
        self.assertEqual([row["label"] for row in candidates], ["white_raw512", "colored_raw512"])
        experiment = self._experiment()
        experiment["split"] = "test"
        with self.assertRaisesRegex(ValueError, "validation-2022"):
            _validate_experiment(experiment)

    def test_gate_rejects_one_bad_output_without_compensation(self):
        control = {
            "primary_mean_standardized_fair_crps": 1.0,
            "per_output": {
                "d3_sic": {
                    "fair_crps": 1.0,
                    "rmse": 1.0,
                    "rank_tv": 0.10,
                    "adjusted_spread_skill_error": 0.20,
                }
            },
        }
        candidate = {
            "primary_mean_standardized_fair_crps": 0.9,
            "per_output": {
                "d3_sic": {
                    "fair_crps": 1.02,
                    "rmse": 0.9,
                    "rank_tv": 0.10,
                    "adjusted_spread_skill_error": 0.20,
                }
            },
        }
        result = _compare(control, candidate, self._experiment()["gate"])
        self.assertFalse(result["pass"])
        self.assertFalse(result["outputs"]["d3_sic"]["checks"]["crps"])

    def test_summary_applies_small_ensemble_spread_correction(self):
        row = {
            "case_equal_fair_crps": 2.0,
            "case_equal_ensemble_mean_rmse": 3.0,
            "spread_skill_ratio": 0.5,
            "rank_tv_to_uniform": 0.1,
            "skill_over_persistence": 0.0,
            "case_equal_persistence_point_mass_crps": 4.0,
            "normalized_mean_rank": 0.5,
            "member_roughness": 1.0,
            "truth_roughness": 1.0,
        }
        metrics = {
            "outputs": {
                f"d{lead}_{field}": dict(row)
                for lead in (3, 6, 9)
                for field in ("sic", "sit")
            },
            "joint_energy_score_common_normalized_coordinate": 1.0,
        }
        summary = _candidate_summary(metrics, (2.0,) * 6)
        self.assertAlmostEqual(summary["primary_mean_standardized_fair_crps"], 1.0)
        self.assertAlmostEqual(
            summary["per_output"]["d3_sic"]["adjusted_spread_skill_ratio"],
            (9.0 / 8.0) ** 0.5 * 0.5,
        )


if __name__ == "__main__":
    unittest.main()
