from __future__ import annotations

import hashlib
import json
import math
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from assim_lib import structured_joint_stats as structured_stats_module
from assim_lib.config import TrainingConfig
from assim_lib.data import M2MForecastDataset
from assim_lib.forecast import ForecastRecord, open_npy_mmap
from assim_lib.model_io import build_unet, load_sampler
from assim_lib.runtime import dataloader_batch_count, optimizer_step_budget
from assim_lib.sampler import Sampler
from assim_lib.structured_archive_audit import (
    _sral_provenance,
    _static_mask_provenance,
    build_archive_semantics_audit,
    validate_bound_archive_audit,
)
from assim_lib.structured_joint_state import (
    StructuredDecodeSaturationError,
    decode_structured_joint_state,
    decode_structured_joint_trajectory,
    dequantized_logit_moments,
    encode_structured_joint_state,
    encode_structured_joint_trajectory,
    validate_structured_state_stats,
)
from assim_lib.structured_trajectory_evaluation import (
    make_structured_trajectory_figure,
    structured_trajectory_metrics,
)
from assim_lib.trainer import (
    UNetTrainer,
    _latest_structured_sampling_valid,
    _require_finite_gradients,
    _require_finite_loss,
    _validate_existing_structured_run_files,
)


def _stats() -> dict:
    binary_mean, binary_std = dequantized_logit_moments(0.5)
    return {
        "schema_version": "structured_joint_state_stats_v3",
        "source_split": "train",
        "inactive_filler_law": "independent_standard_normal",
        "interior_value_law": "continuous_resolution_approximation",
        "data_config_sha256": "a" * 64,
        "pair_manifest_sha256": "b" * 64,
        "sic_cap": 0.75,
        "occurrence_probability": 0.5,
        "cap_probability_given_ice": 0.5,
        "occurrence_logit_mean": binary_mean,
        "occurrence_logit_std": binary_std,
        "cap_logit_mean": binary_mean,
        "cap_logit_std": binary_std,
        "sic_interior_logit_mean": 0.0,
        "sic_interior_logit_std": 1.0,
        "sit_positive_log_mean": 0.0,
        "sit_positive_log_std": 1.0,
        "conditioning_normalization": {
            "source_split": "train",
            "fields": ["siconc", "sithic"],
            "open_water_nan_as_physical_zero": True,
            "means": [0.5, 1.0],
            "stds": [0.5, 1.0],
        },
    }


class StructuredCodecTests(unittest.TestCase):
    def test_round_trip_preserves_joint_atom_cap_and_interior(self) -> None:
        physical = torch.tensor(
            [[[[0.0, 0.25, 0.75]], [[0.0, 2.0, 1.5]]]],
            dtype=torch.float32,
        )
        valid = torch.ones((1, 1, 1, 3), dtype=torch.float32)
        latent = encode_structured_joint_state(
            physical,
            valid,
            _stats(),
            generator=torch.Generator().manual_seed(7),
        )
        decoded = decode_structured_joint_state(latent, _stats())

        self.assertTrue(torch.allclose(decoded, physical, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.all(torch.isfinite(latent)))

    def test_joint_support_violation_fails_closed(self) -> None:
        physical = torch.tensor([[[[0.0]], [[1.0]]]], dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "joint support"):
            encode_structured_joint_state(physical, torch.ones((1, 1, 1, 1)), _stats())

    def test_stats_contract_rejects_validation_leakage(self) -> None:
        stats = _stats()
        stats["source_split"] = "valid"
        with self.assertRaisesRegex(ValueError, "train split"):
            validate_structured_state_stats(stats)

    def test_dequantized_binary_moments_match_known_symmetric_case(self) -> None:
        mean, std = dequantized_logit_moments(0.5)
        self.assertEqual(mean, 0.0)
        self.assertAlmostEqual(std, math.pi / math.sqrt(3.0))

    def test_inactive_fillers_are_continuous_normal_draws_not_flat_quantile_atoms(self) -> None:
        physical = torch.zeros((1, 2, 1, 4096), dtype=torch.float32)
        latent = encode_structured_joint_state(
            physical,
            torch.ones((1, 1, 1, 4096)),
            _stats(),
            generator=torch.Generator().manual_seed(91),
        )
        self.assertGreater(torch.unique(latent[:, 2]).numel(), 4000)
        self.assertGreater(torch.unique(latent[:, 3]).numel(), 4000)

    def test_trajectory_codec_round_trip_preserves_all_leads(self) -> None:
        one = torch.tensor([[[[0.0, 0.25]], [[0.0, 1.0]]]], dtype=torch.float32)
        physical = torch.cat((one, one, one, one), dim=1)
        latent = encode_structured_joint_trajectory(
            physical,
            torch.ones((1, 1, 1, 2)),
            _stats(),
            generator=torch.Generator().manual_seed(17),
        )
        self.assertEqual(tuple(latent.shape), (1, 16, 1, 2))
        self.assertTrue(torch.allclose(decode_structured_joint_trajectory(latent, _stats()), physical))

    def test_decoder_rejects_float32_active_branch_saturation(self) -> None:
        interior_overflow = torch.tensor([[[[1.0]], [[-1.0]], [[1.0e38]], [[0.0]]]])
        with self.assertRaisesRegex(StructuredDecodeSaturationError, "interior SIC saturated") as caught:
            decode_structured_joint_state(interior_overflow, _stats())
        self.assertEqual(caught.exception.diagnostics["sic_saturation_count"], 1)
        self.assertEqual(caught.exception.diagnostics["sit_saturation_count"], 0)
        sit_underflow = torch.tensor([[[[1.0]], [[1.0]], [[0.0]], [[-1.0e38]]]])
        with self.assertRaisesRegex(StructuredDecodeSaturationError, "active SIT saturated"):
            decode_structured_joint_state(sit_underflow, _stats())

        trajectory = torch.zeros((1, 8, 1, 1), dtype=torch.float32)
        trajectory[:, 4:8] = interior_overflow
        with self.assertRaises(StructuredDecodeSaturationError) as trajectory_error:
            decode_structured_joint_trajectory(trajectory, _stats())
        self.assertEqual(trajectory_error.exception.diagnostics["lead_index"], 1)

        nonfinite_later = torch.zeros((1, 8, 1, 1), dtype=torch.float32)
        nonfinite_later[:, :4] = interior_overflow
        nonfinite_later[:, 7] = torch.nan
        with self.assertRaisesRegex(ValueError, "finite before per-lead") as nonfinite_error:
            decode_structured_joint_trajectory(nonfinite_later, _stats())
        self.assertNotIsInstance(nonfinite_error.exception, StructuredDecodeSaturationError)

    def test_physical_sampling_completion_uses_latest_record_only(self) -> None:
        self.assertFalse(_latest_structured_sampling_valid([]))
        self.assertTrue(_latest_structured_sampling_valid([{"status": "passed"}]))
        self.assertFalse(
            _latest_structured_sampling_valid([{"status": "passed"}, {"status": "failed_decode_saturation"}])
        )


class StructuredConditioningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.preds = self.root / "preds"
        self.sral = self.root / "sral"
        self.preds.mkdir()
        self.sral.mkdir()
        self.mask_path = self.root / "mask.npy"
        np.save(self.mask_path, np.zeros((4, 4), dtype=np.float32))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _forecast(self, day: str, sic: float, sit: float, *, nan_at=None) -> None:
        field = np.zeros((15, 24, 4, 4), dtype=np.float16)
        field[0] = sic
        field[1] = sit
        field[6] = 1.0
        field[7] = 2.0
        field[13] = 3.0
        field[14] = 4.0
        if nan_at is not None:
            field[:, :, nan_at[0], nan_at[1]] = np.nan
        np.save(self.preds / f"ocean+atmosphere_24_{day}.npy", field)

    def _track(self, day: str, point: tuple[int, int]) -> None:
        track = np.full((2, 4, 4), np.nan, dtype=np.float32)
        track[1][point] = 0.0
        np.save(self.sral / f"sral_{day}.npy", track)

    def _config(self) -> dict:
        return {
            "dataset_name": "M2MForecastDataset",
            "dataset_dir": str(self.root),
            "sral_dir": str(self.sral),
            "mask_path": str(self.mask_path),
            "mask_true_is_invalid": True,
            "lead_time_hours": 24,
            "indices": [0, 1],
            "means": [0.5, 1.0],
            "stds": [0.5, 1.0],
            "padding_values": [0.0, 0.0],
            "image_size": [4, 4],
            "target_hour_index": 23,
            "hour_mode": "fixed",
            "background_strategy": "calendar_year_ago",
            "calendar_features": ["day_of_year", "hour"],
            "conditioning_layout": "structured_sic_lagged_v1",
            "observed_channels": [0],
            "assimilation_range": 3,
            "observation_mask": {
                "kind": "sral_tracks",
                "sral_transform_index": 1,
                "synthetic_probability": 0.0,
                "empty_probability": 0.0,
                "require_finite_model_values": False,
            },
            "train": {
                "back_start_day": "2019-01-01",
                "back_end_day": "2019-12-31",
                "obs_start_day": "2020-03-01",
                "obs_end_day": "2020-03-01",
            },
        }

    def _trajectory_config(self) -> dict:
        config = self._config()
        config.update(
            {
                "conditioning_layout": "structured_sic_sit_trajectory_v1",
                "future_horizon_days": 3,
                "observed_channels": [0, 1],
                "source_array_dtype": "float16",
                "model_nan_semantics": "joint_sic_sit_nan_means_open_water_zero",
                "trajectory_semantics": "consecutive_daily_archive_snapshots",
                "target_slice_index": 23,
                "utc_time_coordinate_verified": False,
                "time_claim_policy": "archive_date_only_no_utc_or_operational_lead_claim",
                "forbidden_time_labels": [
                    "operational_lead",
                    "23:00_UTC",
                    "24h_issue_time",
                ],
                "dynamic_forcing_indices": [],
                "calendar_features": ["day_of_year", "time_index"],
                "train": {
                    "back_start_day": "2019-01-01",
                    "back_end_day": "2019-12-31",
                    "obs_start_day": "2020-03-01",
                    "obs_end_day": "2020-03-04",
                },
            }
        )
        return config

    def _write_trajectory_fixture(self) -> None:
        for offset, day in enumerate(("2019-02-28", "2019-03-01", "2019-03-02", "2019-03-03", "2019-03-04")):
            self._forecast(day, 0.25 + 0.01 * offset, 0.75 + 0.02 * offset)
        for offset, day in enumerate(
            ("2020-02-28", "2020-02-29", "2020-03-01", "2020-03-02", "2020-03-03", "2020-03-04")
        ):
            self._forecast(day, 0.55 + 0.01 * offset, 1.25 + 0.02 * offset)
        self._track("2020-02-28", (2, 2))
        self._track("2020-02-29", (1, 1))
        self._track("2020-03-01", (0, 0))
        self._track("2020-03-03", (3, 3))

    def _assimilation_config(self) -> dict:
        config = self._trajectory_config()
        config.update(
            {
                "conditioning_layout": "structured_sic_sit_assimilation_v1",
                "future_horizon_days": 0,
                "trajectory_lead_days": [0],
                "trajectory_semantics": "analysis_snapshot_only",
            }
        )
        config["train"] = {
            **config["train"],
            "obs_end_day": "2020-03-01",
        }
        return config

    def _dynamics_config(self) -> dict:
        config = self._trajectory_config()
        config.update(
            {
                "conditioning_layout": "structured_sic_sit_dynamics_v1",
                "background_strategy": "none",
                "assimilation_range": 1,
                "future_horizon_days": 9,
                "trajectory_lead_days": [3, 6, 9],
                "trajectory_semantics": "state_only_forecast_snapshots_d_plus_3_6_9",
                "observation_mask": {"kind": "none"},
                "dynamic_forcing_indices": [6, 7, 13, 14],
                "dynamic_forcing_stats": {
                    "indices": [6, 7, 13, 14],
                    "means": [0.0, 0.0, 0.0, 0.0],
                    "stds": [1.0, 1.0, 1.0, 1.0],
                },
            }
        )
        config["train"] = {
            **config["train"],
            "obs_end_day": "2020-03-10",
        }
        return config

    def test_lags_are_separate_feb29_is_missing_and_nan_open_water_is_observed(self) -> None:
        self._forecast("2019-02-28", 0.3, 0.8)
        self._forecast("2019-03-01", 0.4, 0.9)
        self._forecast("2020-02-28", 0.7, 1.3)
        self._forecast("2020-02-29", 0.6, 1.2)
        self._forecast("2020-03-01", 0.8, 1.4, nan_at=(0, 0))
        self._track("2020-02-28", (2, 2))
        self._track("2020-02-29", (1, 1))
        self._track("2020-03-01", (0, 0))

        dataset = M2MForecastDataset(self._config(), split="train")
        item = dataset[0]
        condition = item["structured_conditioning"]

        self.assertEqual(dataset.conditioned_input_channels, 23)
        self.assertEqual(tuple(condition.shape), (17, 4, 4))
        self.assertEqual(float(condition[5, 0, 0]), 1.0)
        self.assertEqual(float(condition[3, 0, 0]), -1.0)
        self.assertAlmostEqual(float(condition[4, 0, 0]), -0.8, places=3)
        self.assertEqual(float(item["structured_physical_truth"][0, 0, 0]), 0.0)
        self.assertEqual(float(item["structured_physical_truth"][1, 0, 0]), 0.0)
        self.assertTrue(torch.equal(condition[6:9], torch.zeros_like(condition[6:9])))
        self.assertEqual(float(condition[11, 2, 2]), 1.0)
        self.assertEqual(item["meta"]["lag_available"], [True, False, True])

    def test_trajectory_conditioning_is_causal_and_backgrounds_are_per_lead(self) -> None:
        self._write_trajectory_fixture()
        config = self._trajectory_config()
        first = M2MForecastDataset(config, split="train")[0]
        self.assertEqual(first["structured_conditioning"].shape, (32, 4, 4))
        self.assertEqual(first["structured_flow_mask"].shape, (16, 4, 4))
        self.assertTrue(torch.all(first["structured_flow_mask"][:4, 0, 0] == 0))
        self.assertTrue(torch.all(first["structured_flow_mask"][4:, 0, 0] == 1))
        for lead in range(4):
            self.assertTrue(torch.equal(first["structured_conditioning"][3 * lead + 2], torch.ones(4, 4)))

        baseline_condition = first["structured_conditioning"].clone()
        self._forecast("2020-03-03", 0.20, 0.40)
        self._track("2020-03-03", (0, 3))
        open_npy_mmap.cache_clear()
        future_changed = M2MForecastDataset(config, split="train")[0]
        self.assertTrue(torch.equal(future_changed["structured_conditioning"], baseline_condition))
        self.assertFalse(
            torch.equal(
                future_changed["structured_physical_truth"][4:6],
                first["structured_physical_truth"][4:6],
            )
        )

        self._forecast("2019-03-03", 0.45, 1.45)
        open_npy_mmap.cache_clear()
        background_changed = M2MForecastDataset(config, split="train")[0]
        self.assertFalse(
            torch.equal(background_changed["structured_conditioning"][6:8], baseline_condition[6:8])
        )
        self.assertEqual(
            [Path(value).name for value in first["meta"]["background_trajectory_paths"]],
            [f"ocean+atmosphere_24_2019-03-{day:02d}.npy" for day in range(1, 5)],
        )

    def test_assimilation_has_one_target_and_no_future_background(self) -> None:
        self._write_trajectory_fixture()
        dataset = M2MForecastDataset(self._assimilation_config(), split="train")
        item = dataset[0]
        self.assertEqual(dataset.conditioned_input_channels, 23)
        self.assertEqual(tuple(item["truth"].shape), (2, 4, 4))
        self.assertEqual(tuple(item["structured_physical_truth"].shape), (2, 4, 4))
        self.assertEqual(tuple(item["structured_conditioning"].shape), (17, 4, 4))
        self.assertEqual(tuple(item["structured_flow_mask"].shape), (4, 4, 4))
        self.assertEqual(len(item["meta"]["background_trajectory_paths"]), 1)
        self.assertEqual(item["meta"]["trajectory_lead_days"], [0])
        self.assertFalse(item["meta"]["lag_background_innovations_used"])

    def test_dynamics_uses_exact_initial_state_and_only_d3_d6_d9_targets(self) -> None:
        for offset in range(-2, 10):
            day = date(2020, 3, 1) + timedelta(days=offset)
            self._forecast(
                day.isoformat(),
                0.50 + 0.01 * (offset + 2),
                1.00 + 0.02 * (offset + 2),
            )
        dataset = M2MForecastDataset(self._dynamics_config(), split="train")
        item = dataset[0]
        self.assertEqual(dataset.conditioned_input_channels, 29)
        self.assertEqual(tuple(item["truth"].shape), (6, 4, 4))
        self.assertEqual(tuple(item["structured_physical_truth"].shape), (6, 4, 4))
        self.assertEqual(tuple(item["structured_conditioning"].shape), (15, 4, 4))
        self.assertEqual(tuple(item["structured_flow_mask"].shape), (12, 4, 4))
        self.assertTrue(torch.all(item["structured_lag0_mask"] == 0))
        self.assertTrue(torch.all(item["structured_lag0_physical_values"] == 0))
        self.assertTrue(torch.all(item["obs_mask"] == 0))
        self.assertEqual(item["meta"]["trajectory_lead_days"], [3, 6, 9])
        self.assertEqual(item["meta"]["background_role"], "persistence_baseline_not_model_condition")
        self.assertTrue(torch.equal(item["structured_conditioning"][3:7, 0, 0], torch.arange(1.0, 5.0)))
        self.assertTrue(torch.all(item["structured_conditioning"][7:11] == 1))
        self.assertEqual(
            [Path(value).name for value in item["meta"]["target_trajectory_paths"]],
            [
                "ocean+atmosphere_24_2020-03-04.npy",
                "ocean+atmosphere_24_2020-03-07.npy",
                "ocean+atmosphere_24_2020-03-10.npy",
            ],
        )

    def test_truth_free_forecast_builder_needs_no_future_target_files(self) -> None:
        self._write_trajectory_fixture()
        for day in ("2020-03-02", "2020-03-03", "2020-03-04"):
            (self.preds / f"ocean+atmosphere_24_{day}.npy").unlink()
        open_npy_mmap.cache_clear()
        item = M2MForecastDataset.build_structured_forecast_item(self._trajectory_config(), "2020-03-01")
        self.assertNotIn("truth", item)
        self.assertNotIn("structured_physical_truth", item)
        self.assertEqual(tuple(item["background"].shape), (8, 4, 4))
        self.assertEqual(tuple(item["structured_conditioning"].shape), (32, 4, 4))
        self.assertEqual(item["meta"]["target_trajectory_paths"], [])
        self.assertTrue(item["meta"]["truth_free_forecast_builder"])
        self.assertFalse(any("2020-03-0" in value for value in item["meta"]["background_trajectory_paths"]))

    def test_exact_lag0_uses_raw_float16_pair_without_denormalization(self) -> None:
        self._write_trajectory_fixture()
        tiny_sic = np.float16(2.0**-24)
        current_path = self.preds / "ocean+atmosphere_24_2020-03-01.npy"
        current = np.load(current_path)
        current[0, 23, 0, 0] = tiny_sic
        current[1, 23, 0, 0] = np.float16(0.25)
        current[:, 23, 0, 1] = np.nan
        np.save(current_path, current)
        track_path = self.sral / "sral_2020-03-01.npy"
        track = np.load(track_path)
        track[1, 0, 1] = 0.0
        np.save(track_path, track)
        open_npy_mmap.cache_clear()

        config = self._trajectory_config()
        training = M2MForecastDataset(config, split="train")[0]
        forecast = M2MForecastDataset.build_structured_forecast_item(config, "2020-03-01")
        expected = torch.zeros((2, 4, 4), dtype=torch.float32)
        expected[0, 0, 0] = float(tiny_sic)
        expected[1, 0, 0] = 0.25
        for item in (training, forecast):
            self.assertTrue(torch.equal(item["structured_lag0_physical_values"], expected))
            self.assertGreater(float(item["structured_lag0_physical_values"][0, 0, 0]), 0.0)
            self.assertEqual(float(item["structured_lag0_physical_values"][:, 0, 1].sum()), 0.0)
        self.assertTrue(
            torch.equal(
                training["structured_physical_truth"][:2, :, :],
                torch.where(
                    torch.isfinite(torch.as_tensor(current[:2, 23], dtype=torch.float32)),
                    torch.as_tensor(current[:2, 23], dtype=torch.float32),
                    torch.zeros((2, 4, 4), dtype=torch.float32),
                ),
            )
        )

    def test_every_structured_background_and_lag_fails_on_raw_semantic_drift(self) -> None:
        self._write_trajectory_fixture()
        config = self._trajectory_config()
        background_path = self.preds / "ocean+atmosphere_24_2019-03-03.npy"
        background = np.load(background_path)
        background[0, 23, 0, 0] = np.nan
        np.save(background_path, background)
        open_npy_mmap.cache_clear()
        with self.assertRaisesRegex(ValueError, "mismatched SIC/SIT missingness"):
            M2MForecastDataset(config, split="train")[0]

        self._forecast("2019-03-03", 0.27, 0.79)
        lag_path = self.preds / "ocean+atmosphere_24_2020-02-28.npy"
        lag = np.load(lag_path)
        lag[:, 23, 0, 0] = np.inf
        np.save(lag_path, lag)
        open_npy_mmap.cache_clear()
        with self.assertRaisesRegex(ValueError, "Inf"):
            M2MForecastDataset(config, split="train")[0]

    def test_sparse_missing_sral_date_is_explicit_unavailable_not_dataset_failure(self) -> None:
        self._write_trajectory_fixture()
        self._forecast("2019-03-05", 0.31, 0.87)
        self._forecast("2020-03-05", 0.61, 1.37)
        config = self._trajectory_config()
        config["train"]["obs_end_day"] = "2020-03-05"
        dataset = M2MForecastDataset(config, split="train")
        self.assertEqual(len(dataset), 2)
        second = dataset[1]
        self.assertFalse(second["meta"]["lag_available"][0])
        self.assertEqual(int(second["structured_lag0_mask"].sum().item()), 0)

        provenance = dataset.provenance()
        admitted = {
            "sral_provenance": {
                **provenance["sral_source"],
                "availability_by_split_and_lag": {
                    "train": json.loads(json.dumps(provenance["sral_availability_by_lag"]))
                },
            }
        }
        dataset.validate_structured_sral_audit_contract(admitted)
        admitted["sral_provenance"]["availability_by_split_and_lag"]["train"]["lag0"][
            "missing_date_count"
        ] += 1
        with self.assertRaisesRegex(ValueError, "runtime SRAL availability"):
            dataset.validate_structured_sral_audit_contract(admitted)

    def test_entirely_observation_free_split_lag_is_rejected(self) -> None:
        self._write_trajectory_fixture()
        for path in self.sral.glob("*.npy"):
            path.unlink()
        self._track("2020-04-01", (0, 0))
        with self.assertRaisesRegex(ValueError, "observation law is empty"):
            M2MForecastDataset(self._trajectory_config(), split="train")


class StructuredTrainingAndSamplingTests(unittest.TestCase):
    def test_preconditioned_train_helper_and_sampler_share_exact_velocity(self) -> None:
        config = TrainingConfig.from_dict(
            {
                "image_size": [2, 2],
                "in_channels": 50,
                "out_channels": 16,
                "trajectory_horizon_days": 3,
                "training_objective": "structured_joint_state_flow",
                "structured_velocity_parameterization": ("gaussian_path_preconditioned"),
                "timestep_sampler": "stratified_uniform",
                "loss_domain": "valid",
                "obs_loss_weight": 0.0,
                "smoothness_loss_weight": 0.0,
                "sample_start_mode": "noise",
                "sample_end_time": 0.0,
                "sample_enforce_observations": False,
                "sample_obs_guidance_scale": 0.0,
                "sample_cfg_mode": "none",
                "conditioning_mode_probabilities": {"both": 1.0},
                "structured_state_stats": _stats(),
                "validation_weight_source": "ema",
                "minimum_optimizer_steps": 1,
                "run_name": "preconditioned_parity_test",
                "block_out_channels": [32, 32],
                "layers_per_block": 1,
                "down_block_types": ["DownBlock2D", "DownBlock2D"],
                "up_block_types": ["UpBlock2D", "UpBlock2D"],
                "norm_num_groups": 8,
            }
        )

        class ConstantResidual(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.last_state = None

            def forward(self, model_input, timestep, return_dict=False):
                self.last_state = model_input[:, :16].detach().clone()
                return (torch.full_like(model_input[:, :16], 0.25),)

        state = torch.ones((1, 16, 2, 2))
        timesteps = torch.ones(1)
        helper = SimpleNamespace(config=config)
        model_state = UNetTrainer._structured_model_state(helper, state, timesteps)
        self.assertTrue(torch.equal(model_state, state))
        full_velocity = UNetTrainer._reconstruct_model_velocity(
            helper, torch.full_like(state, 0.25), state, timesteps
        )
        self.assertTrue(torch.equal(full_velocity, torch.full_like(state, 1.25)))

        model = ConstantResidual()
        sampled = Sampler(
            model,
            structured_velocity_parameterization="gaussian_path_preconditioned",
        ).sample_conditioned(
            background=torch.zeros((1, 8, 2, 2)),
            obs_values=torch.zeros((1, 8, 2, 2)),
            obs_mask=torch.zeros((1, 8, 2, 2)),
            water_mask=torch.ones((1, 2, 2, 2)),
            size=(2, 2),
            num_timesteps=2,
            device="cpu",
            method="euler",
            memory_efficient_euler=True,
            model_conditioning=torch.zeros((1, 32, 2, 2)),
            state_channels=16,
            end_time=0.0,
            valid_mask=torch.ones((1, 2, 2, 2)),
            state_mask=torch.ones((1, 16, 2, 2)),
            initial_noise=state,
        )
        self.assertTrue(torch.equal(model.last_state, state))
        self.assertTrue(torch.equal(sampled, torch.full_like(state, -0.25)))

    def test_structured_diagnostics_do_not_advance_training_rng(self) -> None:
        trainer = SimpleNamespace(
            config=SimpleNamespace(training_objective="structured_joint_state_flow", validation_seed=2701),
            accelerator=SimpleNamespace(device=torch.device("cpu")),
        )
        trainer._structured_diagnostic_seed = lambda epoch, **kwargs: UNetTrainer._structured_diagnostic_seed(
            trainer, epoch, **kwargs
        )
        torch.manual_seed(12345)
        expected = torch.rand(7)
        torch.manual_seed(12345)
        with UNetTrainer._structured_diagnostic_rng(trainer, 4, stream_index=37):
            first = torch.rand(1024)
        actual = torch.rand(7)
        self.assertTrue(torch.equal(actual, expected))
        with UNetTrainer._structured_diagnostic_rng(trainer, 4, stream_index=37):
            second = torch.rand(1024)
        self.assertTrue(torch.equal(first, second))

    def test_resume_identity_mismatch_does_not_overwrite_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original_config = b'{"run":"original"}\n'
            original_metadata = b'{"resume_contract_sha256":"old"}\n'
            (root / "config.json").write_bytes(original_config)
            (root / "metadata.json").write_bytes(original_metadata)
            with self.assertRaisesRegex(ValueError, "config identity differs"):
                _validate_existing_structured_run_files(root, {"run": "different"}, "new")
            self.assertEqual((root / "config.json").read_bytes(), original_config)
            self.assertEqual((root / "metadata.json").read_bytes(), original_metadata)

    def test_drop_last_step_budget_matches_runtime_and_clears_minimum(self) -> None:
        self.assertEqual(dataloader_batch_count(2143, 16, drop_last=True), 133)
        self.assertEqual(optimizer_step_budget(133, 30, 1), (133, 3990))
        self.assertEqual(optimizer_step_budget(133, 31, 1), (133, 4123))

    def test_nonfinite_loss_and_gradient_fail_before_optimizer(self) -> None:
        with self.assertRaisesRegex(FloatingPointError, "case_id=bad"):
            _require_finite_loss(torch.tensor(float("nan")), "case_id=bad")
        model = torch.nn.Linear(2, 1)
        model.weight.grad = torch.full_like(model.weight, float("inf"))
        with self.assertRaisesRegex(FloatingPointError, "weight"):
            _require_finite_gradients(model, "case_id=bad")

    def test_atomic_structured_recovery_restores_accelerator_ema_and_history(self) -> None:
        class FakeAccelerator:
            is_main_process = True
            num_processes = 1
            device = torch.device("cpu")

            def __init__(self):
                self.state = {"model": 3.0, "optimizer": 4.0, "scheduler": 5.0, "rng": 6.0}

            def save_state(self, path, safe_serialization=False):
                Path(path).mkdir(parents=True)
                torch.save(self.state, Path(path) / "state.pt")

            def load_state(self, path):
                self.state = torch.load(Path(path) / "state.pt", weights_only=True)

        with tempfile.TemporaryDirectory() as directory:
            accelerator = FakeAccelerator()
            ema = torch.nn.Linear(1, 1, bias=False)
            ema.weight.data.fill_(7.0)
            trainer = SimpleNamespace(
                config=SimpleNamespace(
                    training_objective="structured_joint_state_flow",
                    resume_from_checkpoint="auto",
                    recovery_checkpoint_name="structured_recovery",
                    num_epochs=3,
                ),
                accelerator=accelerator,
                output_dir=directory,
                train_dataloader=[0, 1],
                ema_model=ema,
                _resume_contract_sha256="c" * 64,
                val_history=[{"epoch": 0, "val_loss": 1.25}],
                best_val_loss=1.25,
            )
            UNetTrainer._save_structured_recovery(trainer, 1, 2)
            self.assertTrue((Path(directory) / "structured_recovery/latest.json").is_file())
            accelerator.state = {"model": -1.0}
            ema.weight.data.zero_()
            trainer.val_history = []
            trainer.best_val_loss = float("inf")
            self.assertEqual(UNetTrainer._training_loop_start(trainer), (1, 2))
            self.assertEqual(accelerator.state["optimizer"], 4.0)
            self.assertEqual(float(ema.weight.item()), 7.0)
            self.assertEqual(trainer.val_history[0]["epoch"], 0)
            self.assertEqual(trainer.best_val_loss, 1.25)

    def test_stratified_uniform_has_exactly_one_draw_per_stratum(self) -> None:
        trainer = SimpleNamespace(
            accelerator=SimpleNamespace(device=torch.device("cpu")),
            config=SimpleNamespace(timestep_sampler="stratified_uniform"),
        )
        values = UNetTrainer._sample_timesteps(
            trainer,
            8,
            generator=torch.Generator().manual_seed(11),
        )
        ordered = values.sort().values
        lower = torch.arange(8) / 8.0
        upper = torch.arange(1, 9) / 8.0
        self.assertTrue(torch.all(ordered >= lower))
        self.assertTrue(torch.all(ordered < upper))

    def test_four_channel_sampler_integrates_to_exact_zero_endpoint(self) -> None:
        class UnitVelocity(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.input_channels = []

            def forward(self, value, timestep):
                self.input_channels.append(int(value.shape[1]))
                return (torch.ones((value.shape[0], 4, *value.shape[-2:])),)

        model = UnitVelocity()
        sampler = Sampler(model)
        result = sampler.sample_conditioned(
            background=torch.zeros((1, 2, 3, 3)),
            obs_values=torch.zeros((1, 2, 3, 3)),
            obs_mask=torch.zeros((1, 2, 3, 3)),
            water_mask=torch.ones((1, 1, 3, 3)),
            size=(3, 3),
            num_timesteps=2,
            method="euler",
            start_mode="noise",
            initial_noise=torch.ones((1, 4, 3, 3)),
            model_conditioning=torch.zeros((1, 17, 3, 3)),
            state_channels=4,
            end_time=0.0,
            device="cpu",
        )
        self.assertEqual(tuple(result.shape), (1, 4, 3, 3))
        self.assertTrue(torch.allclose(result, torch.zeros_like(result)))
        self.assertEqual(model.input_channels, [23])

    def test_rk4_casts_mixed_precision_velocity_to_float32_state(self) -> None:
        class BF16Velocity(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def forward(self, value, timestep):
                self.calls += 1
                return (
                    torch.ones(
                        (value.shape[0], 4, *value.shape[-2:]),
                        dtype=torch.bfloat16,
                    ),
                )

        model = BF16Velocity()
        result = Sampler(model).sample_conditioned(
            background=torch.zeros((1, 2, 2, 2), dtype=torch.float32),
            obs_values=torch.zeros((1, 2, 2, 2), dtype=torch.float32),
            obs_mask=torch.zeros((1, 2, 2, 2), dtype=torch.float32),
            water_mask=torch.ones((1, 1, 2, 2), dtype=torch.float32),
            size=(2, 2),
            num_timesteps=3,
            method="rk4",
            start_mode="noise",
            initial_noise=torch.ones((1, 4, 2, 2), dtype=torch.float32),
            model_conditioning=torch.zeros((1, 17, 2, 2), dtype=torch.float32),
            state_channels=4,
            end_time=0.0,
            device="cpu",
        )
        self.assertEqual(model.calls, 8)
        self.assertEqual(result.dtype, torch.float32)
        self.assertTrue(torch.allclose(result, torch.zeros_like(result)))

    def test_training_pair_and_input_use_exact_physical_target_and_four_latents(self) -> None:
        trainer = SimpleNamespace(
            config=SimpleNamespace(
                training_objective="structured_joint_state_flow",
                structured_state_stats=_stats(),
                in_channels=23,
            ),
            _grid=torch.zeros((1, 2, 2, 2)),
        )
        physical = torch.tensor(
            [[[[0.0, 0.25], [0.75, 0.5]], [[0.0, 1.0], [2.0, 1.5]]]],
            dtype=torch.float32,
        )
        batch = {
            "background": torch.zeros((1, 2, 2, 2)),
            "valid_mask": torch.ones((1, 2, 2, 2)),
            "structured_physical_truth": physical,
            "structured_conditioning": torch.zeros((1, 17, 2, 2)),
        }
        state, velocity = UNetTrainer._make_training_pair(
            trainer,
            truth=torch.full((1, 2, 2, 2), 123.0),
            batch=batch,
            timesteps=torch.tensor([0.25]),
            generator=torch.Generator().manual_seed(19),
        )
        model_input = UNetTrainer._make_model_input(trainer, state, batch)

        self.assertEqual(tuple(state.shape), (1, 4, 2, 2))
        self.assertEqual(tuple(velocity.shape), (1, 4, 2, 2))
        self.assertEqual(tuple(model_input.shape), (1, 23, 2, 2))
        self.assertTrue(torch.all(torch.isfinite(state)))

    def test_valid_loss_expands_one_physical_mask_channel_to_four_latents(self) -> None:
        trainer = SimpleNamespace(
            config=SimpleNamespace(loss_domain="valid"),
            _masked_mse=UNetTrainer._masked_mse,
        )
        prediction = torch.ones((1, 4, 2, 2))
        target = torch.zeros_like(prediction)
        batch = {"valid_mask": torch.tensor([[[[1.0, 0.0], [0.0, 0.0]], [[1.0, 0.0], [0.0, 0.0]]]])}
        loss = UNetTrainer._flow_matching_loss(trainer, prediction, target, batch)
        self.assertEqual(float(loss), 1.0)

    def test_clean_structured_config_is_fail_closed(self) -> None:
        config = TrainingConfig.from_dict(
            {
                "in_channels": 23,
                "out_channels": 4,
                "training_objective": "structured_joint_state_flow",
                "timestep_sampler": "stratified_uniform",
                "loss_domain": "valid",
                "obs_loss_weight": 0.0,
                "smoothness_loss_weight": 0.0,
                "sample_start_mode": "noise",
                "sample_end_time": 0.0,
                "sample_enforce_observations": False,
                "sample_obs_guidance_scale": 0.0,
                "sample_cfg_mode": "none",
                "conditioning_mode_probabilities": {"both": 1.0},
                "structured_state_stats": _stats(),
                "validation_weight_source": "ema",
                "minimum_optimizer_steps": 1,
                "run_name": "test",
            }
        )
        self.assertEqual(config.out_channels, 4)
        with self.assertRaisesRegex(ValueError, "diagnostic_min_optimizer_steps"):
            TrainingConfig.from_dict(
                {
                    **config.__dict__,
                    "diagnostic_min_optimizer_steps": 2,
                }
            )
        with self.assertRaisesRegex(ValueError, "out_channels=4"):
            TrainingConfig.from_dict(
                {
                    **config.__dict__,
                    "out_channels": 2,
                }
            )

    def test_trajectory_sampler_enforces_exact_day0_pair(self) -> None:
        class ZeroVelocity(torch.nn.Module):
            def forward(self, value, timestep):
                return (torch.zeros((value.shape[0], 16, *value.shape[-2:])),)

        lag0_mask = torch.zeros((1, 1, 2, 2))
        lag0_mask[..., 0, 0] = 1
        flow_mask = torch.ones((1, 16, 2, 2))
        flow_mask[:, :4] *= 1 - lag0_mask
        exact = torch.zeros((1, 2, 2, 2))
        exact[:, 0, 0, 0] = 0.25
        exact[:, 1, 0, 0] = 1.5
        sample = Sampler(ZeroVelocity()).sample_structured_trajectory(
            background_trajectory=torch.zeros((1, 8, 2, 2)),
            model_conditioning=torch.zeros((1, 32, 2, 2)),
            valid_mask=torch.ones((1, 2, 2, 2)),
            flow_mask=flow_mask,
            lag0_physical_values=exact,
            lag0_mask=lag0_mask,
            stats=_stats(),
            size=(2, 2),
            num_timesteps=2,
            method="euler",
            initial_noise=torch.zeros((1, 16, 2, 2)),
            device="cpu",
        )
        self.assertEqual(tuple(sample.shape), (1, 8, 2, 2))
        self.assertEqual(float(sample[0, 0, 0, 0]), 0.25)
        self.assertEqual(float(sample[0, 1, 0, 0]), 1.5)

    def test_cpu_ministep_checkpoint_reload_and_trajectory_sample(self) -> None:
        config = TrainingConfig.from_dict(
            {
                "image_size": [8, 8],
                "in_channels": 50,
                "out_channels": 16,
                "trajectory_horizon_days": 3,
                "training_objective": "structured_joint_state_flow",
                "timestep_sampler": "stratified_uniform",
                "loss_domain": "valid",
                "obs_loss_weight": 0.0,
                "smoothness_loss_weight": 0.0,
                "sample_start_mode": "noise",
                "sample_end_time": 0.0,
                "sample_enforce_observations": False,
                "sample_obs_guidance_scale": 0.0,
                "sample_cfg_mode": "none",
                "conditioning_mode_probabilities": {"both": 1.0},
                "structured_state_stats": _stats(),
                "validation_weight_source": "ema",
                "minimum_optimizer_steps": 1,
                "run_name": "test",
                "block_out_channels": [32, 32],
                "layers_per_block": 1,
                "down_block_types": ["DownBlock2D", "DownBlock2D"],
                "up_block_types": ["UpBlock2D", "UpBlock2D"],
                "norm_num_groups": 8,
            }
        )
        model = build_unet(config).cpu()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        one = torch.tensor([[[[0.25]], [[1.0]]]], dtype=torch.float32).expand(1, 2, 8, 8)
        physical = torch.cat((one, one, one, one), dim=1)
        flow_mask = torch.ones((1, 16, 8, 8))
        flow_mask[:, :4, 0, 0] = 0
        batch = {
            "background": torch.zeros((1, 8, 8, 8)),
            "valid_mask": torch.ones((1, 2, 8, 8)),
            "structured_physical_truth": physical,
            "structured_conditioning": torch.zeros((1, 32, 8, 8)),
            "structured_flow_mask": flow_mask,
        }
        trainer = SimpleNamespace(config=config, _grid=torch.zeros((1, 2, 8, 8)))
        state, target = UNetTrainer._make_training_pair(
            trainer,
            truth=torch.zeros((1, 8, 8, 8)),
            batch=batch,
            timesteps=torch.tensor([0.4]),
            generator=torch.Generator().manual_seed(4),
        )
        prediction = model(
            UNetTrainer._make_model_input(trainer, state, batch), torch.tensor([400.0]), return_dict=False
        )[0]
        loss = UNetTrainer._masked_mse(prediction, target, flow_mask)
        loss.backward()
        optimizer.step()
        self.assertTrue(math.isfinite(float(loss.detach().item())))

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            torch.save(model.state_dict(), run_dir / "last_model.pth")
            (run_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "data_config": {},
                        "training_config": {
                            "training_objective": "structured_joint_state_flow",
                            "structured_velocity_parameterization": "raw",
                        },
                    }
                )
            )
            reloaded = load_sampler(run_dir, "last_model.pth", config.__dict__, device="cpu")
            self.assertEqual(reloaded.structured_velocity_parameterization, "raw")
            mismatched = dict(config.__dict__)
            mismatched["structured_velocity_parameterization"] = "gaussian_path_preconditioned"
            with self.assertRaisesRegex(ValueError, "parameterization differs"):
                load_sampler(run_dir, "last_model.pth", mismatched, device="cpu")
            lag0_mask = torch.zeros((1, 1, 8, 8))
            lag0_mask[..., 0, 0] = 1
            exact = torch.zeros((1, 2, 8, 8))
            exact[:, 0, 0, 0] = 0.25
            exact[:, 1, 0, 0] = 1.0
            sampled = reloaded.sample_structured_trajectory(
                background_trajectory=batch["background"],
                model_conditioning=batch["structured_conditioning"],
                valid_mask=batch["valid_mask"],
                flow_mask=flow_mask,
                lag0_physical_values=exact,
                lag0_mask=lag0_mask,
                stats=_stats(),
                size=(8, 8),
                num_timesteps=2,
                method="euler",
                initial_noise=torch.zeros((1, 16, 8, 8)),
                device="cpu",
            )
            self.assertEqual(tuple(sampled.shape), (1, 8, 8, 8))
            self.assertEqual(float(sampled[0, 0, 0, 0]), 0.25)
            figure = make_structured_trajectory_figure(
                physical[0], physical[0], sampled[0], batch["valid_mask"][0], title="CPU reload smoke"
            )
            figure_path = run_dir / "dashboard.png"
            figure.savefig(figure_path)
            import matplotlib.pyplot as plt

            plt.close(figure)
            self.assertGreater(figure_path.stat().st_size, 1000)

            ensemble = torch.stack((sampled, sampled), dim=1)
            metrics = structured_trajectory_metrics(
                ensemble,
                physical,
                physical,
                batch["valid_mask"],
                lag0_mask=lag0_mask,
                sic_cap=0.75,
            )
            self.assertEqual(metrics["lag0_observation_max_abs_error"], 0.0)
            self.assertAlmostEqual(sum(metrics["lead0_sic_rank_counts"]), 64.0)

    def test_large_all_tie_rank_mass_and_strict_support(self) -> None:
        width = 60000
        truth = torch.zeros((1, 2, 1, width), dtype=torch.float32)
        ensemble = torch.zeros((1, 5, 2, 1, width), dtype=torch.float32)
        valid = torch.ones((1, 1, 1, width), dtype=torch.float32)
        metrics = structured_trajectory_metrics(ensemble, truth, truth, valid, sic_cap=0.9970703125)
        counts = metrics["lead0_sic_rank_counts"]
        self.assertAlmostEqual(sum(counts), float(width), places=8)
        self.assertTrue(all(abs(value - width / 6) < 1e-8 for value in counts))
        bad = ensemble.clone()
        bad[:, :, 0, 0, 0] = 0.5
        with self.assertRaisesRegex(ValueError, "joint SIC/SIT support"):
            structured_trajectory_metrics(bad, truth, truth, valid, sic_cap=0.9970703125)
        bad = ensemble.clone()
        bad[..., -1] = float("nan")
        valid[..., -1] = 0.0
        structured_trajectory_metrics(bad, truth, truth, valid, sic_cap=0.9970703125)

    def test_structured_rmse_accumulates_large_finite_sit_in_float64(self) -> None:
        truth = torch.tensor([[[[0.5]], [[1.0e30]]]], dtype=torch.float32)
        background = truth.clone()
        ensemble = torch.tensor(
            [[[[[0.5]], [[1.0e30]]], [[[0.5]], [[2.0e30]]]]],
            dtype=torch.float32,
        )
        valid = torch.ones((1, 1, 1, 1), dtype=torch.float32)
        metrics = structured_trajectory_metrics(
            ensemble,
            truth,
            background,
            valid,
            sic_cap=0.9970703125,
        )
        self.assertTrue(math.isfinite(metrics["lead0_sit_mean_rmse"]))
        self.assertGreater(metrics["lead0_sit_mean_rmse"], 0.0)


class StructuredArchiveAuditTests(unittest.TestCase):
    def test_static_mask_must_be_finite_binary_and_have_ocean(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mask.npy"
            config = {"mask_path": str(path), "mask_true_is_invalid": True}
            np.save(path, np.array([[0.0, 0.5]], dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "exactly binary"):
                _static_mask_provenance(config)
            np.save(path, np.array([[0.0, np.nan]], dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "finite everywhere"):
                _static_mask_provenance(config)
            np.save(path, np.ones((2, 2), dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "no valid ocean"):
                _static_mask_provenance(config)

    def test_sral_audit_uses_runtime_first_finite_combination_before_transform(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = np.full((2, 2, 2), np.nan, dtype=np.float32)
            second = np.full((2, 2, 2), np.nan, dtype=np.float32)
            first[1, 0, 0] = 3.0
            second[1, 0, 0] = 0.0
            np.save(root / "a_2020-01-01.npy", first)
            np.save(root / "b_2020-01-01.npy", second)
            empty_split = {
                "back_start_day": "2019-01-01",
                "back_end_day": "2019-01-01",
                "obs_start_day": "2020-01-01",
                "obs_end_day": "2020-01-01",
            }
            config = {
                "sral_dir": str(root),
                "future_horizon_days": 3,
                "observation_mask": {"sral_transform_index": 1},
                "train": empty_split,
                "valid": empty_split,
                "test": empty_split,
            }
            with self.assertRaisesRegex(ValueError, "no usable geometry"):
                _sral_provenance(config, [], np.ones((2, 2), dtype=bool))

    def test_sral_audit_rejects_zero_usable_observations_for_candidate_split_lag(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unrelated = np.full((2, 2, 2), np.nan, dtype=np.float32)
            unrelated[1, 0, 0] = 0.0
            np.save(root / "sral_2021-01-01.npy", unrelated)
            records = [
                ForecastRecord(date(year, 1, day), root / f"unused_{year}_{day}.npy")
                for year in (2019, 2020)
                for day in range(1, 5)
            ]
            split = {
                "back_start_day": "2019-01-01",
                "back_end_day": "2019-01-04",
                "obs_start_day": "2020-01-01",
                "obs_end_day": "2020-01-04",
            }
            config = {
                "sral_dir": str(root),
                "future_horizon_days": 3,
                "observation_mask": {"sral_transform_index": 1},
                "train": split,
                "valid": split,
                "test": split,
            }
            with self.assertRaisesRegex(ValueError, "split=train lag=0"):
                _sral_provenance(config, records, np.ones((2, 2), dtype=bool))

    def test_stats_cli_resolves_and_passes_data_config_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "data.json"
            output_path = root / "stats.json"
            data_path.write_text("{}", encoding="utf-8")
            config = {"sentinel": True}
            stats = {"audit": {"sample_count": 1}}
            with (
                patch.object(
                    structured_stats_module,
                    "parse_args",
                    return_value=SimpleNamespace(data_config=str(data_path), output=str(output_path)),
                ),
                patch.object(structured_stats_module, "load_json", return_value=config) as loader,
                patch.object(
                    structured_stats_module,
                    "build_structured_state_stats",
                    return_value=stats,
                ) as builder,
            ):
                structured_stats_module.main()
            loader.assert_called_once_with(data_path.resolve())
            builder.assert_called_once_with(config, data_path.resolve())
            self.assertEqual(json.loads(output_path.read_text()), stats)

    def test_full_archive_audit_binds_missingness_rule_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            preds = root / "preds"
            preds.mkdir()
            sral = root / "sral"
            sral.mkdir()
            mask_path = root / "mask.npy"
            np.save(mask_path, np.zeros((4, 4), dtype=np.float32))
            track = np.full((2, 4, 4), np.nan, dtype=np.float32)
            track[1, 0, 0] = 0.0
            np.save(sral / "sral_2020-01-01.npy", track)
            field = np.zeros((15, 24, 4, 4), dtype=np.float16)
            field[0] = 0.75
            field[1] = 1.0
            field[0, :, 0, 0] = np.nan
            field[1, :, 0, 0] = np.nan
            field[13, :, 1, 1] = np.nan
            record_path = None
            for year in (2019, 2020):
                for day in range(1, 6):
                    candidate = preds / f"ocean+atmosphere_24_{year}-01-{day:02d}.npy"
                    one = field.copy()
                    if not (year == 2020 and day == 1):
                        one[0, :, 0, 0] = 0.75
                        one[1, :, 0, 0] = 1.0
                    np.save(candidate, one)
                    if year == 2020 and day == 1:
                        record_path = candidate
            split = {
                "back_start_day": "2019-01-01",
                "back_end_day": "2019-01-05",
                "obs_start_day": "2020-01-01",
                "obs_end_day": "2020-01-05",
            }
            config = {
                "dataset_dir": str(root),
                "sral_dir": str(sral),
                "mask_path": str(mask_path),
                "mask_true_is_invalid": True,
                "lead_time_hours": 24,
                "target_slice_index": 23,
                "future_horizon_days": 3,
                "structured_sic_cap": 0.75,
                "source_array_dtype": "float16",
                "model_nan_semantics": "joint_sic_sit_nan_means_open_water_zero",
                "trajectory_semantics": "consecutive_daily_archive_snapshots",
                "utc_time_coordinate_verified": False,
                "time_claim_policy": "archive_date_only_no_utc_or_operational_lead_claim",
                "forbidden_time_labels": [
                    "operational_lead",
                    "23:00_UTC",
                    "24h_issue_time",
                ],
                "candidate_dynamic_indices": [6, 7, 13, 14],
                "dynamic_forcing_indices": [],
                "observation_mask": {
                    "kind": "sral_tracks",
                    "sral_transform_index": 1,
                },
                "train": split,
                "valid": split,
                "test": split,
                "archive_semantics_audit_path": "audit.json",
                "archive_semantics_audit_sha256": "pending",
            }
            audit = build_archive_semantics_audit(config)
            self.assertEqual(audit["status"], "data_semantics_verified")
            self.assertEqual(audit["sic_sit_pair_checks"]["ocean_joint_nan"], 1)
            self.assertEqual(audit["per_variable"]["wind_grid_x_archive_units"]["ocean_nan"], 10)
            self.assertEqual(audit["trajectory_date_audit"]["status"], "verified")
            self.assertEqual(
                audit["trajectory_date_audit"]["splits"]["train"]["eligible_anchor_count"],
                2,
            )
            train_lag0 = audit["sral_provenance"]["availability_by_split_and_lag"]["train"]["lag0"]
            self.assertEqual(train_lag0["candidate_date_count"], 2)
            self.assertEqual(train_lag0["usable_nonempty_geometry_date_count"], 1)
            self.assertEqual(train_lag0["missing_dates"], ["2020-01-02"])
            self.assertEqual(
                audit["dynamic_feature_policy"]["nan_rule"],
                "missing_with_explicit_mask; no imputation",
            )
            audit_path = root / "audit.json"
            audit_path.write_text(json.dumps(audit, indent=2) + "\n")
            config["archive_semantics_audit_sha256"] = hashlib.sha256(audit_path.read_bytes()).hexdigest()
            config_path = root / "data.json"
            config_path.write_text(json.dumps(config))
            self.assertEqual(
                validate_bound_archive_audit(config, config_path)["status"],
                "data_semantics_verified",
            )
            original_forecast_stat = record_path.stat()
            drifted = np.load(record_path)
            drifted[0, :, 0, 1] = 0.5
            np.save(record_path, drifted)
            import os

            os.utime(
                record_path,
                ns=(original_forecast_stat.st_atime_ns, original_forecast_stat.st_mtime_ns),
            )
            with self.assertRaisesRegex(ValueError, "forecast audited content"):
                validate_bound_archive_audit(config, config_path)

            # Rebuild the binding, then prove that same-path auxiliary contents
            # are also cryptographically admitted rather than trusted by name.
            audit = build_archive_semantics_audit(config)
            audit_path.write_text(json.dumps(audit, indent=2) + "\n")
            config["archive_semantics_audit_sha256"] = hashlib.sha256(audit_path.read_bytes()).hexdigest()
            config_path.write_text(json.dumps(config))
            original_stat = mask_path.stat()
            mask = np.load(mask_path)
            mask[0, 1] = 1.0
            np.save(mask_path, mask)
            os.utime(mask_path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
            with self.assertRaisesRegex(ValueError, "static land mask"):
                validate_bound_archive_audit(config, config_path)

            audit = build_archive_semantics_audit(config)
            audit_path.write_text(json.dumps(audit, indent=2) + "\n")
            config["archive_semantics_audit_sha256"] = hashlib.sha256(audit_path.read_bytes()).hexdigest()
            config_path.write_text(json.dumps(config))
            track_path = sral / "sral_2020-01-01.npy"
            track_stat = track_path.stat()
            changed_track = np.load(track_path)
            changed_track[1, 0, 1] = 0.0
            np.save(track_path, changed_track)
            os.utime(track_path, ns=(track_stat.st_atime_ns, track_stat.st_mtime_ns))
            with self.assertRaisesRegex(ValueError, "SRAL geometry"):
                validate_bound_archive_audit(config, config_path)

    def test_missing_or_empty_sral_archive_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = {
                "sral_dir": str(Path(directory) / "missing"),
                "observation_mask": {"kind": "sral_tracks", "sral_transform_index": 1},
            }
            with self.assertRaisesRegex(ValueError, "directory is missing"):
                M2MForecastDataset._load_sral_records(SimpleNamespace(config=config))
            Path(config["sral_dir"]).mkdir()
            with self.assertRaisesRegex(ValueError, "contains no"):
                M2MForecastDataset._load_sral_records(SimpleNamespace(config=config))


if __name__ == "__main__":
    unittest.main()
