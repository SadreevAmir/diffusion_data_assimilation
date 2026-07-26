import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from assim_lib.data import M2MForecastDataset
from assim_lib.sampler import Sampler
from assim_lib.trainer import UNetTrainer
from assim_lib.transforms import make_conditioned_model_input


class AssimLibConditioningTests(unittest.TestCase):
    @staticmethod
    def _write_m2m_pair(root: Path, channels: int = 2, size: tuple[int, int] = (4, 4)) -> Path:
        preds = root / "dataset" / "preds"
        preds.mkdir(parents=True)
        height, width = size
        background = np.ones((channels, height, width), dtype=np.float32)
        truth = np.full((channels, height, width), 2.0, dtype=np.float32)
        np.save(preds / "ocean+atmosphere_24_20200101.npy", background)
        np.save(preds / "ocean+atmosphere_24_20200102.npy", truth)
        return root / "dataset"

    @staticmethod
    def _m2m_config(dataset_dir: Path, observation_mask: dict) -> dict:
        return {
            "dataset_name": "M2MForecastDataset",
            "dataset_dir": str(dataset_dir),
            "lead_time_hours": 24,
            "fields": ["var0", "var1"],
            "indices": [0, 1],
            "means": [0.0, 0.0],
            "stds": [1.0, 1.0],
            "padding_values": [0.0, 0.0],
            "image_size": [4, 4],
            "observed_channels": [0],
            "observation_mask": observation_mask,
            "train": {
                "back_start_day": "2020-01-01",
                "back_end_day": "2020-01-01",
                "obs_start_day": "2020-01-02",
                "obs_end_day": "2020-01-02",
            },
        }

    def test_water_mask_is_single_conditioning_channel(self):
        state = torch.zeros((1, 4, 3, 2))
        grid = torch.zeros((1, 2, 3, 2))
        background = torch.ones_like(state)
        obs_values = torch.full_like(state, 2.0)
        obs_mask = torch.full_like(state, 3.0)
        water_mask = torch.stack(
            [
                torch.ones((3, 2)),
                torch.zeros((3, 2)),
                torch.full((3, 2), 2.0),
                torch.full((3, 2), 3.0),
            ],
            dim=0,
        ).unsqueeze(0)

        model_input = make_conditioned_model_input(
            state,
            grid,
            background,
            torch.ones_like(background),
            obs_values,
            obs_mask,
            water_mask,
        )

        self.assertEqual(tuple(model_input.shape), (1, 23, 3, 2))
        torch.testing.assert_close(model_input[:, -1:], water_mask[:, :1])

    def test_sampler_uses_provided_initial_noise(self):
        initial_noise = torch.full((2, 1, 2, 2), 3.0)
        zeros = torch.zeros_like(initial_noise)
        sampler = Sampler(model=object())

        with patch("assim_lib.sampler.odeint") as odeint:
            odeint.side_effect = lambda f, x0, timesteps, **kwargs: torch.stack([x0, x0], dim=0)
            sample = sampler.sample_conditioned(
                background=zeros,
                obs_values=zeros,
                obs_mask=zeros,
                water_mask=torch.ones_like(zeros),
                size=(2, 2),
                num_timesteps=2,
                device=torch.device("cpu"),
                start_mode="noise",
                initial_noise=initial_noise,
            )

        torch.testing.assert_close(sample, initial_noise)

    def test_residual_sampler_returns_background_plus_generated_correction(self):
        background = torch.full((1, 1, 2, 2), 5.0)
        correction = torch.full_like(background, 2.0)
        sampler = Sampler(model=object())

        with patch("assim_lib.sampler.odeint") as odeint:
            odeint.side_effect = lambda f, x0, timesteps, **kwargs: torch.stack([x0, correction], dim=0)
            analysis = sampler.sample_conditioned(
                background=background,
                obs_values=torch.zeros_like(background),
                obs_mask=torch.zeros_like(background),
                water_mask=torch.ones_like(background),
                size=(2, 2),
                num_timesteps=2,
                device=torch.device("cpu"),
                start_mode="noise",
                sample_target="residual",
            )

        torch.testing.assert_close(analysis, torch.full_like(background, 7.0))

    def test_residual_sampler_with_zero_background_returns_absolute_prediction(self):
        background = torch.zeros((1, 1, 2, 2))
        correction = torch.full_like(background, 2.0)
        sampler = Sampler(model=object())

        with patch("assim_lib.sampler.odeint") as odeint:
            odeint.side_effect = lambda f, x0, timesteps, **kwargs: torch.stack([x0, correction], dim=0)
            analysis = sampler.sample_conditioned(
                background=background,
                obs_values=torch.zeros_like(background),
                obs_mask=torch.zeros_like(background),
                water_mask=torch.ones_like(background),
                size=(2, 2),
                num_timesteps=2,
                device=torch.device("cpu"),
                start_mode="noise",
                sample_target="residual",
            )

        torch.testing.assert_close(analysis, correction)

    def test_sampler_background_delta_cfg_scales_conditioning_predictions(self):
        class ConditioningVelocityModel:
            def __call__(self, model_input, timestep):
                background_mask = model_input[:, 4:5]
                obs_mask = model_input[:, 6:7]
                has_background = background_mask.reshape(background_mask.shape[0], -1).sum(dim=1) > 0
                has_obs = obs_mask.reshape(obs_mask.shape[0], -1).sum(dim=1) > 0
                values = torch.ones((model_input.shape[0], 1, 1, 1), dtype=model_input.dtype)
                values = torch.where(has_background.view(-1, 1, 1, 1), torch.full_like(values, 3.0), values)
                values = torch.where(has_obs.view(-1, 1, 1, 1), torch.full_like(values, 7.0), values)
                values = torch.where(
                    (has_background & has_obs).view(-1, 1, 1, 1),
                    torch.full_like(values, 10.0),
                    values,
                )
                return values.expand(-1, 1, model_input.shape[-2], model_input.shape[-1])

        condition = torch.zeros((1, 1, 2, 2))
        sample = Sampler(ConditioningVelocityModel()).sample_conditioned(
            background=torch.ones_like(condition),
            background_mask=torch.ones_like(condition),
            obs_values=torch.ones_like(condition),
            obs_mask=torch.ones_like(condition),
            water_mask=torch.ones_like(condition),
            size=(2, 2),
            num_timesteps=2,
            device=torch.device("cpu"),
            start_mode="noise",
            initial_noise=condition,
            memory_efficient_euler=True,
            cfg_mode="background_delta",
            cfg_background_scale=0.5,
            cfg_observation_scale=2.0,
        )

        # v_none + 0.5 * (v_bg - v_none) + 2.0 * (v_both - v_bg)
        expected_velocity = 1.0 + 0.5 * (3.0 - 1.0) + 2.0 * (10.0 - 3.0)
        torch.testing.assert_close(sample, torch.full_like(condition, -0.999 * expected_velocity))

    def test_sampler_independent_cfg_uses_observation_only_branch(self):
        class ConditioningVelocityModel:
            def __call__(self, model_input, timestep):
                background_mask = model_input[:, 4:5]
                obs_mask = model_input[:, 6:7]
                has_background = background_mask.reshape(background_mask.shape[0], -1).sum(dim=1) > 0
                has_obs = obs_mask.reshape(obs_mask.shape[0], -1).sum(dim=1) > 0
                values = torch.ones((model_input.shape[0], 1, 1, 1), dtype=model_input.dtype)
                values = torch.where(has_background.view(-1, 1, 1, 1), torch.full_like(values, 3.0), values)
                values = torch.where(has_obs.view(-1, 1, 1, 1), torch.full_like(values, 7.0), values)
                values = torch.where(
                    (has_background & has_obs).view(-1, 1, 1, 1),
                    torch.full_like(values, 10.0),
                    values,
                )
                return values.expand(-1, 1, model_input.shape[-2], model_input.shape[-1])

        condition = torch.zeros((1, 1, 2, 2))
        sample = Sampler(ConditioningVelocityModel()).sample_conditioned(
            background=torch.ones_like(condition),
            background_mask=torch.ones_like(condition),
            obs_values=torch.ones_like(condition),
            obs_mask=torch.ones_like(condition),
            water_mask=torch.ones_like(condition),
            size=(2, 2),
            num_timesteps=2,
            device=torch.device("cpu"),
            start_mode="noise",
            initial_noise=condition,
            memory_efficient_euler=True,
            cfg_mode="independent",
            cfg_background_scale=0.5,
            cfg_observation_scale=2.0,
        )

        # v_none + 0.5 * (v_bg - v_none) + 2.0 * (v_obs - v_none)
        expected_velocity = 1.0 + 0.5 * (3.0 - 1.0) + 2.0 * (7.0 - 1.0)
        torch.testing.assert_close(sample, torch.full_like(condition, -0.999 * expected_velocity))

    def test_background_dropout_only_zeroes_train_conditioning_input(self):
        trainer = object.__new__(UNetTrainer)
        trainer.config = SimpleNamespace(background_dropout_probability=1.0)
        trainer.model = SimpleNamespace(training=True)
        trainer._grid = torch.zeros((1, 2, 2, 2))
        state = torch.zeros((2, 2, 2, 2))
        batch = {
            "background": torch.ones((2, 2, 2, 2)),
            "obs_values": torch.zeros((2, 2, 2, 2)),
            "obs_mask": torch.zeros((2, 2, 2, 2)),
            "water_mask": torch.ones((2, 1, 2, 2)),
        }

        model_input = trainer._make_model_input(state, batch)
        torch.testing.assert_close(model_input[:, 4:6], torch.zeros_like(batch["background"]))
        torch.testing.assert_close(model_input[:, 6:8], torch.zeros_like(batch["background"]))
        torch.testing.assert_close(batch["background"], torch.ones_like(batch["background"]))

        trainer.model.training = False
        model_input = trainer._make_model_input(state, batch)
        torch.testing.assert_close(model_input[:, 4:6], batch["background"])
        torch.testing.assert_close(model_input[:, 6:8], torch.ones_like(batch["background"]))

    def test_conditioning_modes_drop_background_and_track_per_sample(self):
        trainer = object.__new__(UNetTrainer)
        trainer.config = SimpleNamespace(
            conditioning_mode_probabilities={
                "no_background": 0.25,
                "no_track": 0.25,
                "no_conditioning": 0.25,
                "both": 0.25,
            }
        )
        trainer.model = SimpleNamespace(training=True)
        batch = {
            "background": torch.ones((4, 1, 2, 2)),
            "obs_values": torch.ones((4, 1, 2, 2)),
            "obs_mask": torch.ones((4, 1, 2, 2)),
        }
        draws = torch.tensor([0.1, 0.3, 0.6, 0.9]).view(4, 1, 1, 1)

        with patch("assim_lib.trainer.torch.rand", return_value=draws):
            background, background_mask, obs_values, obs_mask = trainer._conditioned_inputs(batch)

        self.assertEqual(background[:, 0, 0, 0].tolist(), [0.0, 1.0, 0.0, 1.0])
        self.assertEqual(background_mask[:, 0, 0, 0].tolist(), [0.0, 1.0, 0.0, 1.0])
        self.assertEqual(obs_values[:, 0, 0, 0].tolist(), [1.0, 0.0, 0.0, 1.0])
        self.assertEqual(obs_mask[:, 0, 0, 0].tolist(), [1.0, 0.0, 0.0, 1.0])

    def test_residual_flow_trains_on_truth_minus_background(self):
        trainer = object.__new__(UNetTrainer)
        trainer.config = SimpleNamespace(training_objective="residual_flow")
        trainer.add_noise = lambda target, timesteps: (target, target)
        truth = torch.full((1, 1, 2, 2), 5.0)
        background = torch.full_like(truth, 2.0)

        model_state, target = trainer._make_training_pair(
            truth,
            {"background": background},
            torch.tensor([0.5]),
        )

        torch.testing.assert_close(model_state, torch.full_like(truth, 3.0))
        torch.testing.assert_close(target, torch.full_like(truth, 3.0))

    def test_residual_flow_uses_dropped_background_for_target(self):
        trainer = object.__new__(UNetTrainer)
        trainer.config = SimpleNamespace(training_objective="residual_flow")
        trainer.add_noise = lambda target, timesteps: (target, target)
        truth = torch.full((1, 1, 2, 2), 5.0)
        dropped_background = torch.zeros_like(truth)

        _, target = trainer._make_training_pair(
            truth,
            {"background": torch.full_like(truth, 2.0)},
            torch.tensor([0.5]),
            residual_background=dropped_background,
        )

        torch.testing.assert_close(target, truth)

    def test_m2m_masks_are_zero_in_land_invalid_and_padding_regions(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            preds = root / "dataset" / "preds"
            preds.mkdir(parents=True)

            background = np.array(
                [
                    [[1.0, 2.0], [3.0, 4.0]],
                    [[5.0, 6.0], [7.0, 8.0]],
                ],
                dtype=np.float32,
            )
            truth = np.array(
                [
                    [[10.0, 11.0], [12.0, np.nan]],
                    [[20.0, 21.0], [22.0, 23.0]],
                ],
                dtype=np.float32,
            )
            np.save(preds / "ocean+atmosphere_24_20200101.npy", background)
            np.save(preds / "ocean+atmosphere_24_20200102.npy", truth)

            land_mask = np.array([[False, True], [False, False]], dtype=np.float32)
            land_mask_path = root / "land_mask.npy"
            np.save(land_mask_path, land_mask)

            dataset = M2MForecastDataset(
                {
                    "dataset_name": "M2MForecastDataset",
                    "dataset_dir": str(root / "dataset"),
                    "mask_path": str(land_mask_path),
                    "mask_true_is_invalid": True,
                    "lead_time_hours": 24,
                    "fields": ["var0", "var1"],
                    "indices": [0, 1],
                    "means": [0.0, 0.0],
                    "stds": [1.0, 1.0],
                    "padding_values": [-1.0, -2.0],
                    "image_size": [3, 4],
                    "observed_channels": [0, 1],
                    "observation_mask": {"kind": "random", "density": 1.0},
                    "train": {
                        "back_start_day": "2020-01-01",
                        "back_end_day": "2020-01-01",
                        "obs_start_day": "2020-01-02",
                        "obs_end_day": "2020-01-02",
                    },
                },
                split="train",
            )
            sample = dataset[0]

        self.assertEqual(tuple(sample["water_mask"].shape), (1, 3, 4))
        self.assertEqual(tuple(sample["valid_mask"].shape), (2, 3, 4))
        self.assertEqual(tuple(sample["obs_mask"].shape), (2, 3, 4))

        # Land mask and padded bottom/right extent are invalid for every mask.
        self.assertEqual(float(sample["water_mask"][:, 0, 1].sum()), 0.0)
        self.assertEqual(float(sample["water_mask"][:, 2, :].sum()), 0.0)
        self.assertEqual(float(sample["water_mask"][:, :, 2:].sum()), 0.0)
        self.assertEqual(float(sample["valid_mask"][:, 0, 1].sum()), 0.0)
        self.assertEqual(float(sample["valid_mask"][:, 2, :].sum()), 0.0)
        self.assertEqual(float(sample["valid_mask"][:, :, 2:].sum()), 0.0)
        self.assertEqual(float(sample["obs_mask"][:, 0, 1].sum()), 0.0)
        self.assertEqual(float(sample["obs_mask"][:, 2, :].sum()), 0.0)
        self.assertEqual(float(sample["obs_mask"][:, :, 2:].sum()), 0.0)

        # valid_mask and the observation footprint carry only the global land/padding mask:
        # per-case truth/background finite info is intentionally excluded to avoid leaking
        # truth's coverage through the sampler's finalize overlay. A NaN in truth represents
        # "no ice" (water) in this dataset and is treated as a valid observable value.
        self.assertEqual(float(sample["valid_mask"][0, 1, 1]), 1.0)
        self.assertEqual(float(sample["valid_mask"][1, 1, 1]), 1.0)
        self.assertEqual(float(sample["obs_mask"][0, 1, 1]), 1.0)
        self.assertEqual(float(sample["obs_mask"][1, 1, 1]), 1.0)

    def test_sral_mix_can_force_synthetic_generated_tracks(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset_dir = self._write_m2m_pair(root)
            dataset = M2MForecastDataset(
                self._m2m_config(
                    dataset_dir,
                    {
                        "kind": "sral_tracks",
                        "synthetic_probability": 1.0,
                        "empty_probability": 0.0,
                        "synthetic": {"kind": "generated_track", "n_tracks_range": [1, 1]},
                    },
                ),
                split="train",
            )
            sample = dataset[0]

        self.assertEqual(sample["meta"]["mask_kind"], "synthetic_generated_track")
        self.assertEqual(sample["meta"]["sral_files_used"], 0)
        self.assertGreater(float(sample["obs_mask"].sum()), 0.0)

    def test_sral_mix_can_force_empty_conditioning(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset_dir = self._write_m2m_pair(root)
            dataset = M2MForecastDataset(
                self._m2m_config(
                    dataset_dir,
                    {
                        "kind": "sral_tracks",
                        "synthetic_probability": 0.0,
                        "empty_probability": 1.0,
                        "synthetic": {"kind": "generated_track", "n_tracks_range": [1, 1]},
                    },
                ),
                split="train",
            )
            sample = dataset[0]

        self.assertEqual(sample["meta"]["mask_kind"], "empty")
        self.assertEqual(float(sample["obs_mask"].sum()), 0.0)
        self.assertEqual(float(sample["obs_values"].sum()), 0.0)

    def test_real_sral_all_hours_uses_sample_hour_for_observation_values(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            preds = root / "dataset" / "preds"
            preds.mkdir(parents=True)
            sral_dir = root / "sral"
            sral_dir.mkdir()

            background = np.zeros((2, 24, 4, 4), dtype=np.float32)
            truth = np.zeros((2, 24, 4, 4), dtype=np.float32)
            for hour in range(24):
                truth[0, hour] = float(hour)
                truth[1, hour] = float(hour + 100)
            np.save(preds / "ocean+atmosphere_24_20200101.npy", background)
            np.save(preds / "ocean+atmosphere_24_20200102.npy", truth)

            sral = np.zeros((8, 4, 4), dtype=np.float32)
            sral[7] = 0.5
            np.save(sral_dir / "sral_20200102.npy", sral)

            config = self._m2m_config(
                root / "dataset",
                {
                    "kind": "sral_tracks",
                    "sral_transform_index": 11,
                    "synthetic_probability": 0.0,
                    "empty_probability": 0.0,
                },
            )
            config["sral_dir"] = str(sral_dir)
            config["hour_mode"] = "all"
            config["target_hour_index"] = 23
            dataset = M2MForecastDataset(config, split="train")
            sample = dataset[5]

        self.assertEqual(sample["meta"]["hour"], 5)
        self.assertEqual(sample["meta"]["mask_kind"], "sral_tracks")
        self.assertGreater(float(sample["obs_mask"][0].sum()), 0.0)
        self.assertEqual(float(sample["obs_values"][0, 0, 0]), 5.0)

    def test_strided_case_indices_follow_target_dates(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            preds = root / "dataset" / "preds"
            preds.mkdir(parents=True)
            field = np.ones((2, 4, 4), dtype=np.float32)
            for stamp in ("20200101", "20200130", "20200131", "20200301"):
                np.save(preds / f"ocean+atmosphere_24_{stamp}.npy", field)

            dataset = M2MForecastDataset(
                {
                    "dataset_name": "M2MForecastDataset",
                    "dataset_dir": str(root / "dataset"),
                    "lead_time_hours": 24,
                    "fields": ["var0", "var1"],
                    "indices": [0, 1],
                    "means": [0.0, 0.0],
                    "stds": [1.0, 1.0],
                    "padding_values": [0.0, 0.0],
                    "image_size": [4, 4],
                    "observed_channels": [0],
                    "observation_mask": {"kind": "generated_track", "n_tracks_range": [1, 1]},
                    "train": {
                        "back_start_day": "2020-01-01",
                        "back_end_day": "2020-03-01",
                        "obs_start_day": "2020-01-01",
                        "obs_end_day": "2020-03-01",
                    },
                },
                split="train",
            )

        self.assertEqual(dataset.strided_case_indices(stride_days=30), [0, 2, 3])

    def test_sample_metrics_select_strided_days_at_configured_hour_for_all_hours_data(self):
        class DailyDataset:
            hours_per_day = 24
            hour_index = 23

            @staticmethod
            def strided_case_indices(max_cases, stride_days):
                assert (max_cases, stride_days) == (3, 15)
                return [0, 30 * 24, 60 * 24]

        trainer = object.__new__(UNetTrainer)
        trainer.val_dataloader = SimpleNamespace(dataset=DailyDataset())

        self.assertEqual(
            trainer._metric_case_indices(3, stride_days=15), [23, (30 * 24) + 23, (60 * 24) + 23]
        )

    def test_metric_aggregation_is_pixel_weighted(self):
        metrics = UNetTrainer._finalize_metric_totals(
            "obs",
            {
                "analysis_abs": 4.0,
                "analysis_sq": 8.0,
                "background_abs": 6.0,
                "background_sq": 18.0,
                "count": 2.0,
            },
        )

        self.assertEqual(metrics["analysis_mae_obs"], 2.0)
        self.assertEqual(metrics["background_mae_obs"], 3.0)
        self.assertEqual(metrics["analysis_rmse_obs"], 2.0)
        self.assertEqual(metrics["background_rmse_obs"], 3.0)
        self.assertAlmostEqual(metrics["analysis_rmse_skill_obs"], 1.0 - (2.0 / 3.0))

    def test_sample_metrics_can_save_generated_ensemble(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = object.__new__(UNetTrainer)
            trainer.config = SimpleNamespace(
                metric_num_cases=2,
                metric_num_ensemble=2,
                metric_stride_days=15,
                metric_num_timesteps=3,
                num_sample_timesteps=3,
                eval_batch_size=2,
                image_size=(2, 2),
                sample_start_mode="noise",
                sample_start_noise_level=0.5,
                sample_enforce_observations=False,
                sample_obs_guidance_scale=0.0,
                sample_obs_guidance_eps=1e-8,
                metric_save_ensemble_samples=True,
                metric_ensemble_save_dtype="float16",
                training_objective="flow",
                sample_method="dopri5",
                sample_rtol=1e-5,
                sample_atol=1e-6,
            )
            trainer.output_dir = tmp_dir
            trainer.accelerator = SimpleNamespace(device=torch.device("cpu"))
            trainer.model = SimpleNamespace(eval=lambda: None)
            trainer.fields = ["siconc"]
            trainer.channel_means = [0.5]
            trainer.channel_stds = [0.5]
            trainer._sampling_model = lambda: nullcontext(object())
            trainer._metric_case_indices = lambda max_cases, stride_days: [7, 11]
            batch = {
                "truth": torch.zeros((2, 1, 2, 2)),
                "background": torch.ones((2, 1, 2, 2)),
                "obs_values": torch.zeros((2, 1, 2, 2)),
                "obs_mask": torch.ones((2, 1, 2, 2)),
                "valid_mask": torch.ones((2, 1, 2, 2)),
                "water_mask": torch.ones((2, 1, 2, 2)),
            }
            trainer._iter_indexed_batches = lambda indices, batch_size: [batch]

            with patch("assim_lib.trainer.Sampler") as sampler_cls:
                sampler_cls.return_value.sample_conditioned.side_effect = [
                    torch.full((2, 1, 2, 2), 2.0),
                    torch.full((2, 1, 2, 2), 4.0),
                ]
                metrics = trainer.compute_sample_validation_metrics(epoch=3)

            kwargs = sampler_cls.return_value.sample_conditioned.call_args_list[0].kwargs
            self.assertEqual(kwargs["method"], "dopri5")
            self.assertEqual(kwargs["rtol"], 1e-5)
            self.assertEqual(kwargs["atol"], 1e-6)
            self.assertFalse(kwargs["memory_efficient_euler"])
            self.assertEqual(metrics["background_rmse_siconc_full"], 0.5)
            self.assertEqual(metrics["analysis_rmse_mean_siconc_full"], 0.5)
            artifact = torch.load(
                Path(tmp_dir) / "samples" / "epoch_0003_metric_ensemble.pt",
                map_location="cpu",
                weights_only=True,
            )
            self.assertEqual(tuple(artifact["samples"].shape), (2, 2, 1, 2, 2))
            self.assertEqual(artifact["samples"].dtype, torch.float16)
            self.assertTrue(torch.equal(artifact["case_indices"], torch.tensor([7, 11])))
            self.assertEqual(float(artifact["samples"][0, 0, 0, 0, 0]), 2.0)
            self.assertEqual(float(artifact["samples"][0, 1, 0, 0, 0]), 4.0)
            self.assertEqual(artifact["values_space"], "normalized")
            self.assertEqual(artifact["metrics_values_space"], "physical")


if __name__ == "__main__":
    unittest.main()
