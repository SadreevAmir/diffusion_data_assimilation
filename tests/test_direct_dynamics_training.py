import unittest
from datetime import date
from unittest.mock import patch

import torch
from torch import nn

from assim_lib.data import M2MForecastDataset
from assim_lib.direct_dynamics_training import (
    DIRECT_CONDITION_CHANNELS,
    DIRECT_INPUT_CHANNELS,
    DIRECT_OUTPUT_CHANNELS,
    DirectDynamicsTrainer,
)
from assim_lib.direct_dynamics_evaluation import _score, _selected_indices
from assim_lib.direct_dynamics_tail_diagnostic import _top_support_violations
from assim_lib.direct_dynamics_temperature_calibration import (
    CALIBRATION_INDICES,
    CONFIRMATION_INDICES,
    confirmation_gate,
    scaled_initial_noise,
    tail_gate,
    validate_panel_indices,
)
from assim_lib.sampler import Sampler


class _ZeroVelocity(nn.Module):
    def forward(self, value, timestep, return_dict=False):
        del timestep, return_dict
        return (torch.zeros_like(value[:, :DIRECT_OUTPUT_CHANNELS]),)


class DirectDynamicsContractTests(unittest.TestCase):
    def test_channel_contract(self):
        self.assertEqual(DIRECT_OUTPUT_CHANNELS, 6)
        self.assertEqual(DIRECT_CONDITION_CHANNELS, 15)
        self.assertEqual(DIRECT_INPUT_CHANNELS, 23)

    def test_dynamics_calendar_uses_resolved_slice(self):
        dataset = object.__new__(M2MForecastDataset)
        dataset.image_size = (2, 2)
        dataset.calendar_features = ("day_of_year", "time_index")
        dataset.dynamic_forcing_indices = (6, 7, 13, 14)
        dataset.trajectory_lead_days = (3, 6, 9)
        initial = torch.zeros((2, 2, 2))
        forcing = torch.zeros((4, 2, 2))
        masks = torch.ones((4, 2, 2))
        valid = torch.ones((2, 2, 2))

        with patch("assim_lib.data.calendar_feature_values", return_value=(1.0, 2.0, 3.0, 4.0)) as fn:
            condition, *_ = dataset._structured_dynamics_conditioning(
                date(2020, 1, 2), 12, initial, forcing, masks, valid
            )
        fn.assert_called_once_with(date(2020, 1, 2), 12, dataset.calendar_features)
        self.assertEqual(tuple(condition.shape), (15, 2, 2))

    def test_training_pair_input_loss_and_backward(self):
        trainer = object.__new__(DirectDynamicsTrainer)
        trainer._grid = torch.zeros((1, 2, 4, 5))
        truth = torch.randn((2, DIRECT_OUTPUT_CHANNELS, 4, 5))
        valid = torch.ones((2, 1, 4, 5))
        valid[:, :, 0] = 0
        batch = {
            "valid_mask": valid,
            "structured_conditioning": torch.randn((2, DIRECT_CONDITION_CHANNELS, 4, 5)),
        }
        time = torch.tensor([0.25, 0.75])
        state, target = trainer._make_training_pair(truth, batch, time)
        model_input = trainer._make_model_input(state, batch)
        model = nn.Conv2d(DIRECT_INPUT_CHANNELS, DIRECT_OUTPUT_CHANNELS, 1)
        prediction = model(model_input)
        loss = trainer._flow_matching_loss(prediction, target, batch)
        loss.backward()
        self.assertEqual(tuple(model_input.shape), (2, 23, 4, 5))
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.weight.grad)
        self.assertTrue(torch.all(state[:, :, 0] == 0))
        self.assertTrue(torch.all(target[:, :, 0] == 0))

    def test_real_sampler_six_channels_and_land_mask(self):
        sampler = Sampler(_ZeroVelocity())
        shape = (2, DIRECT_OUTPUT_CHANNELS, 4, 5)
        noise = torch.randn(shape)
        valid = torch.ones((2, 1, 4, 5))
        valid[:, :, 0] = 0
        sampled = sampler.sample_conditioned(
            background=torch.zeros(shape),
            obs_values=torch.zeros(shape),
            obs_mask=torch.zeros(shape),
            water_mask=valid,
            size=(4, 5),
            num_timesteps=3,
            device=torch.device("cpu"),
            method="euler",
            start_mode="noise",
            initial_noise=noise,
            sample_target="state",
            model_conditioning=torch.zeros((2, DIRECT_CONDITION_CHANNELS, 4, 5)),
            state_channels=DIRECT_OUTPUT_CHANNELS,
            end_time=0.0,
            state_mask=valid.expand_as(noise),
        )
        self.assertEqual(tuple(sampled.shape), shape)
        self.assertTrue(torch.all(sampled[:, :, 0] == 0))
        self.assertTrue(torch.equal(sampled[:, :, 1:], noise[:, :, 1:]))

    def test_raw_metrics_reject_nonfinite_and_differ_from_projection(self):
        valid = torch.ones((1, 1, 2, 2))
        raw = torch.zeros((1, 2, DIRECT_OUTPUT_CHANNELS, 2, 2))
        raw[:, :, 0] = 1.5
        raw[:, :, 1] = -0.5
        metrics = DirectDynamicsTrainer._raw_support_metrics(raw, valid)
        self.assertEqual(metrics["sic_raw_support_violation_fraction"], 1.0 / 3.0)
        self.assertEqual(metrics["sit_raw_support_violation_fraction"], 1.0 / 3.0)
        projected = DirectDynamicsTrainer._project_for_display(raw.flatten(0, 1))
        self.assertFalse(torch.equal(raw.flatten(0, 1), projected))
        raw[0, 0, 0, 0, 0] = float("nan")
        with self.assertRaises(FloatingPointError):
            DirectDynamicsTrainer._raw_support_metrics(raw, valid)

    def test_paired_evaluation_selection_and_raw_metrics(self):
        self.assertEqual(_selected_indices(25, 3), [0, 12, 24])
        truth = torch.zeros((2, DIRECT_OUTPUT_CHANNELS, 3, 4))
        truth[:, 0::2] = 0.4
        truth[:, 1::2] = 0.8
        ensemble = truth[:, None].repeat(1, 3, 1, 1, 1)
        ensemble[:, 0] -= 0.1
        ensemble[:, 2] += 0.1
        persistence = truth + 0.2
        valid = torch.ones((2, 1, 3, 4))
        metrics = _score(ensemble, truth, persistence, valid)
        self.assertEqual(set(metrics["leads"]), {"d3", "d6", "d9"})
        self.assertAlmostEqual(metrics["leads"]["d3"]["sic"]["ensemble_mean_rmse"], 0.0)
        self.assertAlmostEqual(metrics["leads"]["d3"]["sic"]["persistence_rmse"], 0.2)
        self.assertAlmostEqual(sum(metrics["leads"]["d3"]["sic"]["fractional_rank_counts"]), 24.0)
        self.assertEqual(metrics["support"]["sic"]["frequency"], 0.0)
        self.assertEqual(metrics["support"]["sit"]["frequency"], 0.0)

    def test_tail_localization_preserves_case_member_lead_and_pixel(self):
        ensemble = torch.zeros((2, 3, DIRECT_OUTPUT_CHANNELS, 4, 5))
        ensemble[1, 2, 4, 3, 1] = 1.7
        mask = torch.ones((2, 1, 4, 5))
        identities = [
            {"case_id": "a", "dataset_index": 10},
            {"case_id": "b", "dataset_index": 20},
        ]
        record = _top_support_violations(
            ensemble, mask, identities, field="sic", count=1
        )[0]
        self.assertEqual(record["case_order"], 1)
        self.assertEqual(record["member"], 2)
        self.assertEqual(record["lead_day"], 9)
        self.assertEqual((record["row"], record["column"]), (3, 1))
        self.assertAlmostEqual(record["violation_magnitude"], 0.7)

    def test_temperature_panels_are_date_disjoint_and_include_stress_cases(self):
        contract = validate_panel_indices(8544)
        self.assertEqual(len(CALIBRATION_INDICES), 12)
        self.assertEqual(len(CONFIRMATION_INDICES), 12)
        self.assertGreaterEqual(contract["minimum_anchor_separation_days"], 10)
        self.assertEqual(set(CALIBRATION_INDICES).intersection(CONFIRMATION_INDICES), set())
        self.assertEqual(contract["stress_dataset_indices"], [3883, 7766])

    def test_temperature_scales_only_paired_base_noise(self):
        base = scaled_initial_noise(4, 3, (2, 2), torch.device("cpu"), 1.0)
        warm = scaled_initial_noise(4, 3, (2, 2), torch.device("cpu"), 1.1)
        self.assertTrue(torch.equal(warm, base * 1.1))

    def test_temperature_tail_and_confirmation_gates_fail_closed(self):
        def tails(value):
            return {
                "leads": {
                    lead: {
                        field: {
                            "mean_delta": value,
                            "frequency_gt_0p01": value,
                            "frequency_gt_0p10": value,
                            "max_delta": value,
                        }
                        for field in ("sic", "sit")
                    }
                    for lead in ("d3", "d6", "d9")
                }
            }

        self.assertTrue(tail_gate(tails(0.01), tails(0.01))["passed"])
        self.assertFalse(tail_gate(tails(0.02), tails(0.01))["passed"])
        comparison = {
            "J_mean_fair_crps_ratio": 0.98,
            "fair_crps_ratios": {"x": 1.0},
            "rmse_ratios": {"x": 1.0},
            "mean_abs_ssr_error_reduction": 0.02,
            "mean_rank_tv_difference": -0.01,
            "rank_tv_differences": {"x": 0.0},
        }
        self.assertTrue(confirmation_gate(comparison, 0.99, {"passed": True})["passed"])
        self.assertFalse(confirmation_gate(comparison, 1.0, {"passed": True})["passed"])


if __name__ == "__main__":
    unittest.main()
