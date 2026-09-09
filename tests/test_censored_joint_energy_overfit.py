import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from assim_lib.censored_joint_energy_overfit import (
    CONDITION_BATCH_SIZE,
    DIAGNOSTIC_UPDATES,
    ENSEMBLE_MEMBERS,
    FIXED_TIMESTEP,
    MAX_UPDATES,
    _final_gate,
    _ipc_preflight,
    censor_sic_sit,
    joint_field_energy_score,
)


class CensoredJointEnergyOverfitTests(unittest.TestCase):
    def test_protocol_budget_and_batch_semantics_are_frozen(self) -> None:
        self.assertEqual(DIAGNOSTIC_UPDATES, (0, 64, 256, 512))
        self.assertEqual(MAX_UPDATES, 512)
        self.assertEqual(CONDITION_BATCH_SIZE, 8)
        self.assertEqual(ENSEMBLE_MEMBERS, 2)
        self.assertEqual(CONDITION_BATCH_SIZE * ENSEMBLE_MEMBERS, 16)
        self.assertEqual(FIXED_TIMESTEP, 500.0)

    def test_ipc_preflight_rejects_insufficient_shared_memory(self) -> None:
        dataset = [{"value": torch.ones((6, 8, 8))} for _ in range(8)]
        with patch(
            "assim_lib.censored_joint_energy_overfit.shutil.disk_usage",
            return_value=SimpleNamespace(free=1),
        ):
            with self.assertRaises(RuntimeError):
                _ipc_preflight(dataset, tuple(range(8)))

    def test_censoring_is_exact_and_masks_inactive_values(self) -> None:
        latent = torch.tensor(
            [[[[ -2.0, float("inf")]], [[-3.0, float("nan")]],
              [[2.0, float("inf")]], [[-1.0, float("nan")]],
              [[0.2, float("inf")]], [[0.4, float("nan")]]]],
            dtype=torch.float32,
        )
        valid = torch.tensor([[[[1.0, 0.0]]]])
        physical = censor_sic_sit(
            latent,
            valid,
            means=(0.2, 0.3) * 3,
            stds=(0.4, 0.5) * 3,
        )
        self.assertEqual(float(physical[0, 0, 0, 0]), 0.0)
        self.assertEqual(float(physical[0, 1, 0, 0]), 0.0)
        self.assertEqual(float(physical[0, 2, 0, 0]), 1.0)
        self.assertTrue(torch.all(physical[..., 1] == 0))

    def test_energy_score_zero_distance_has_finite_zero_gradient(self) -> None:
        members = torch.zeros((1, 2, 6, 1, 1), requires_grad=True)
        truth = torch.zeros((1, 6, 1, 1))
        valid = torch.ones((1, 1, 1, 1))
        score = joint_field_energy_score(
            members, truth, valid, stds=(1.0,) * 6
        )
        score.backward()
        self.assertEqual(float(score), 0.0)
        self.assertTrue(torch.all(torch.isfinite(members.grad)))
        self.assertTrue(torch.all(members.grad == 0))

    def test_energy_score_rewards_joint_dependence_not_only_member_mean(self) -> None:
        truth = torch.zeros((2, 6, 1, 1))
        truth[0, :2] = -1.0
        truth[1, :2] = 1.0
        valid = torch.ones((2, 1, 1, 1))
        correct_joint = torch.zeros((2, 2, 6, 1, 1))
        correct_joint[:, 0, :2] = -1.0
        correct_joint[:, 1, :2] = 1.0
        wrong_joint = torch.zeros_like(correct_joint)
        wrong_joint[:, 0, 0] = -1.0
        wrong_joint[:, 0, 1] = 1.0
        wrong_joint[:, 1, 0] = 1.0
        wrong_joint[:, 1, 1] = -1.0
        correct_score = joint_field_energy_score(
            correct_joint, truth, valid, stds=(1.0,) * 6
        )
        wrong_score = joint_field_energy_score(
            wrong_joint, truth, valid, stds=(1.0,) * 6
        )
        self.assertLess(float(correct_score), float(wrong_score))

    def test_zero_initial_rmse_uses_strict_absolute_tolerance(self) -> None:
        def diagnostic(final: bool) -> dict:
            outputs = {}
            for lead in (3, 6, 9):
                for field in ("sic", "sit"):
                    rmse = 0.0
                    if final and lead == 3 and field == "sic":
                        rmse = 0.1
                    outputs[f"d{lead}_{field}"] = {
                        "rmse": rmse,
                        "positive_region_rmse": rmse,
                        "positive_prediction_mean": 1.0,
                        "positive_truth_mean": 1.0,
                    }
            return {
                "outputs": outputs,
                "zero_case_leads": [
                    {
                        "case": 0,
                        "lead": "d3",
                        "sic_mean": 0.0,
                        "sic_p95": 0.0,
                        "sit_mean_m": 0.0,
                        "sit_p95_m": 0.0,
                        "sit_gt_0p01_fraction": 0.0,
                    }
                ],
                "support": {
                    "sic_below_zero": 0,
                    "sic_above_one": 0,
                    "sit_below_zero": 0,
                },
            }

        diagnostics = {
            0: diagnostic(False),
            64: diagnostic(False),
            256: diagnostic(False),
            512: diagnostic(True),
        }
        gate = _final_gate([0.1] * 512, diagnostics)
        self.assertFalse(gate["criteria"]["rmse_each_output"])
        record = next(
            item
            for item in gate["rmse_comparisons"]
            if item["output"] == "d3_sic"
        )
        self.assertEqual(record["mode"], "absolute_final_rmse")
        self.assertEqual(record["threshold"], 0.001)
        self.assertEqual(gate["status"], "failed")


if __name__ == "__main__":
    unittest.main()
