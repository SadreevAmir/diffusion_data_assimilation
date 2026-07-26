import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from assim_lib.config import TrainingConfig
from assim_lib.evaluate import (
    PhysicalMetricAccumulator,
    _apply_evaluation_config,
    _publish_clearml_results,
    apply_sampler_normalization,
    denormalize_and_clip,
    generate_ensemble,
    validation_case_indices,
)
from assim_lib.sampler import Sampler


class _AllHourDataset:
    hours_per_day = 24
    hour_index = 23

    def __len__(self):
        return 24 * 365

    def strided_case_indices(self, max_cases, stride_days):
        return [0, stride_days * 24, 2 * stride_days * 24][:max_cases]


class _DatasetWithoutStridedCases:
    def __len__(self):
        return 4


class _NoiseReturningSampler:
    def __init__(self):
        self.batch_sizes = []
        self.kwargs = []

    def sample_conditioned(self, **kwargs):
        self.batch_sizes.append(kwargs["background"].shape[0])
        self.kwargs.append(kwargs)
        return kwargs["initial_noise"]


class _ConstantVelocityModel:
    def __call__(self, model_input, timestep):
        return SimpleNamespace(sample=torch.ones_like(model_input[:, :1]))


class _RecordingTracker:
    def __init__(self):
        self.scalars = []
        self.single_values = []
        self.artifacts = []

    def report_single_value(self, name, value):
        self.single_values.append((name, value))

    def report_scalar(self, title, series, value, iteration):
        self.scalars.append((title, series, value, iteration))

    def upload_artifact(self, name, path):
        self.artifacts.append((name, path))


class AssimLibEvaluationTests(unittest.TestCase):
    def test_denormalize_and_clip_only_concentration(self):
        normalized = torch.tensor([[[[-2.0]], [[2.0]]], [[[2.0]], [[-2.0]]]])
        physical = denormalize_and_clip(normalized, [0.5, 10.0], [0.5, 2.0], concentration_channel=0)

        np.testing.assert_allclose(physical[:, 0, 0, 0], np.array([0.0, 1.0]))
        np.testing.assert_allclose(physical[:, 1, 0, 0], np.array([14.0, 6.0]))

    def test_validation_case_indices_shift_to_validation_hour(self):
        indices = validation_case_indices(_AllHourDataset(), stride_days=15)

        self.assertEqual(indices, [23, 15 * 24 + 23, 30 * 24 + 23])

    def test_validation_case_indices_falls_back_for_smoke_data(self):
        indices = validation_case_indices(_DatasetWithoutStridedCases(), stride_days=15, max_cases=2)

        self.assertEqual(indices, [0, 1])

    def test_physical_metrics_include_background_and_skill(self):
        metrics = PhysicalMetricAccumulator(interval_levels=(0.5,))
        row = metrics.add(
            field="siconc",
            region="full",
            ensemble=np.array([[0.2, 1.0], [0.4, 0.8]]),
            truth=np.array([0.2, 0.8]),
            background=np.array([0.6, 0.4]),
            mask=np.array([True, True]),
        )

        self.assertIsNotNone(row)
        self.assertAlmostEqual(row["analysis_mean_rmse"], 0.1)
        self.assertAlmostEqual(row["background_rmse"], 0.4)
        self.assertAlmostEqual(row["analysis_rmse_skill"], 0.75)
        aggregate = metrics.finalize()[0]
        self.assertAlmostEqual(aggregate["analysis_mean_rmse"], 0.1)
        self.assertAlmostEqual(aggregate["background_rmse"], 0.4)

    def test_generate_ensemble_batches_members_and_preserves_seeded_noise(self):
        sampler = _NoiseReturningSampler()
        condition = torch.zeros((1, 1, 2, 2))
        config = TrainingConfig(image_size=(2, 2), sample_start_mode="noise")

        ensemble = generate_ensemble(
            sampler=sampler,
            background=condition,
            obs_values=condition,
            obs_mask=condition,
            water_mask=condition,
            valid_mask=condition,
            config=config,
            ensemble_size=17,
            sample_batch_size=16,
            num_timesteps=2,
            method="euler",
            device=torch.device("cpu"),
            seed=25,
            case_order=0,
            sample_target="state",
        )
        expected = []
        for member in range(17):
            torch.manual_seed(25 + member)
            expected.append(torch.randn_like(condition)[0])

        self.assertEqual(sampler.batch_sizes, [16, 1])
        self.assertEqual(sampler.kwargs[0]["cfg_mode"], "none")
        self.assertEqual(sampler.kwargs[0]["cfg_background_scale"], 1.0)
        self.assertEqual(sampler.kwargs[0]["cfg_observation_scale"], 1.0)
        torch.testing.assert_close(ensemble, torch.stack(expected))

    def test_sampler_metadata_overrides_evaluation_normalization(self):
        dataset = SimpleNamespace(
            means=[0.5, 0.5],
            stds=[0.5, 0.5],
            config={"means": [0.5, 0.5], "stds": [0.5, 0.5]},
        )
        sampler = SimpleNamespace(
            metadata={"normalization_means": [0.11, 1.2], "normalization_stds": [0.21, 0.7]}
        )
        data_config = {"means": [0.5, 0.5], "stds": [0.5, 0.5]}

        with self.assertWarns(UserWarning):
            means, stds = apply_sampler_normalization(dataset, sampler, data_config)

        self.assertEqual(means, [0.11, 1.2])
        self.assertEqual(stds, [0.21, 0.7])
        self.assertEqual(dataset.means, means)
        self.assertEqual(data_config["stds"], stds)

    def test_memory_efficient_euler_returns_final_state_without_trajectory_storage(self):
        condition = torch.zeros((1, 1, 2, 2))
        sample = Sampler(_ConstantVelocityModel()).sample_conditioned(
            background=condition,
            obs_values=condition,
            obs_mask=condition,
            water_mask=condition,
            size=(2, 2),
            num_timesteps=3,
            device=torch.device("cpu"),
            start_mode="noise",
            initial_noise=condition,
            memory_efficient_euler=True,
        )

        torch.testing.assert_close(sample, torch.full_like(condition, -0.999))

    def test_evaluation_config_populates_defaults_but_cli_wins(self):
        args = SimpleNamespace(run_dir="/run", ensemble_size=3, checkpoint_name=None)
        experiment = {"evaluation": {"checkpoint_name": "ema_best_model.pth", "ensemble_size": 15}}

        resolved = _apply_evaluation_config(args, experiment)

        self.assertEqual(resolved.checkpoint_name, "ema_best_model.pth")
        self.assertEqual(resolved.ensemble_size, 3)
        self.assertEqual(resolved.sample_batch_size, 16)
        self.assertEqual(resolved.split, "valid")

    def test_clearml_publisher_reports_physical_metrics_and_artifacts(self):
        tracker = _RecordingTracker()
        rows = [
            {
                "field": "siconc",
                "region": "full",
                "analysis_mean_rmse": 0.25,
                "analysis_rmse_skill": float("nan"),
            }
        ]

        _publish_clearml_results(
            tracker,
            output_dir=Path("/tmp/evaluation"),
            aggregate_rows=rows,
            metadata={"num_cases": 2},
            upload_samples=True,
        )

        self.assertEqual(tracker.single_values, [("num_cases", 2.0)])
        self.assertIn(("physical/analysis_mean_rmse", "siconc/full", 0.25, 0), tracker.scalars)
        self.assertNotIn("physical/analysis_rmse_skill", [row[0] for row in tracker.scalars])
        self.assertEqual(len(tracker.artifacts), 5)


if __name__ == "__main__":
    unittest.main()
