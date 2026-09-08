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


if __name__ == "__main__":
    unittest.main()
