from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from assim_lib.config import TrainingConfig
from assim_lib.data import M2MForecastDataset, calendar_feature_values, previous_calendar_date
from assim_lib.trainer import UNetTrainer
from assim_lib.transforms import make_conditioned_model_input


class CalendarPairingTests(unittest.TestCase):
    def test_frozen_evaluation_uses_validation_and_unchanged_sampling(self):
        root = Path(__file__).resolve().parents[1]
        payload = json.loads(
            (root / "config/experiments/evaluate_siconc_calendar_residual_cfm.json").read_text()
        )
        evaluation = payload["evaluation"]
        self.assertEqual(evaluation["split"], "valid")
        self.assertEqual(evaluation["stride_days"], 5)
        self.assertEqual(evaluation["max_cases"], 40)
        self.assertEqual(evaluation["ensemble_size"], 10)
        self.assertEqual(evaluation["seed"], 1234)
        self.assertEqual(evaluation["method"], "dopri5")
        self.assertEqual(evaluation["num_timesteps"], 25)
        self.assertEqual(evaluation["cfg_mode"], "none")
        self.assertTrue(evaluation["save_ensembles"])

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.preds = self.root / "preds"
        self.preds.mkdir()
        self.mask_path = self.root / "mask.npy"
        np.save(self.mask_path, np.zeros((8, 8), dtype=np.float32))

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _write_forecast(self, day: str, value: float) -> None:
        field = np.full((1, 24, 8, 8), value, dtype=np.float32)
        np.save(self.preds / f"ocean+atmosphere_24_{day}.npy", field)

    def _config(self) -> dict:
        return {
            "dataset_name": "M2MForecastDataset",
            "dataset_dir": str(self.root),
            "mask_path": str(self.mask_path),
            "mask_true_is_invalid": True,
            "lead_time_hours": 24,
            "indices": [0],
            "means": [0.5],
            "stds": [0.5],
            "padding_values": [0.0],
            "image_size": [8, 8],
            "target_hour_index": 23,
            "hour_mode": "fixed",
            "background_strategy": "calendar_year_ago",
            "calendar_features": ["day_of_year"],
            "resample_observation_masks_each_epoch": True,
            "observed_channels": [0],
            "assimilation_range": 1,
            "observation_mask": {"kind": "random", "density": 0.5},
            "train": {
                "back_start_day": "2020-01-01",
                "back_end_day": "2020-12-31",
                "obs_start_day": "2021-01-01",
                "obs_end_day": "2021-12-31",
            },
        }

    def test_exact_pairing_drops_unmatched_dates_instead_of_shifting(self) -> None:
        self._write_forecast("2020-03-01", 0.2)
        self._write_forecast("2020-03-02", 0.3)
        self._write_forecast("2021-03-01", 0.7)
        self._write_forecast("2021-03-03", 0.8)

        dataset = M2MForecastDataset(self._config(), split="train")

        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset.conditioned_input_channels, 10)
        provenance = dataset.provenance()
        self.assertEqual(provenance["num_pairs"], 1)
        self.assertEqual(provenance["target_candidates"], 2)
        self.assertEqual(provenance["dropped_target_dates"], 1)
        self.assertRegex(provenance["pair_manifest_sha256"], r"^[0-9a-f]{64}$")
        item = dataset[0]
        self.assertEqual(item["meta"]["background_date"], "2020-03-01")
        self.assertEqual(item["meta"]["target_date"], "2021-03-01")
        self.assertEqual(item["meta"]["background_offset_days"], 365)
        self.assertEqual(item["meta"]["background_strategy"], "calendar_year_ago")

    def test_epoch_conditioned_mask_rng_is_reproducible_and_changes(self) -> None:
        self._write_forecast("2020-03-01", 0.2)
        self._write_forecast("2021-03-01", 0.7)
        dataset = M2MForecastDataset(self._config(), split="train")

        dataset.set_epoch(0)
        mask_epoch_zero = dataset[0]["obs_mask"].clone()
        self.assertTrue(torch.equal(mask_epoch_zero, dataset[0]["obs_mask"]))
        dataset.set_epoch(1)
        mask_epoch_one = dataset[0]["obs_mask"]

        self.assertFalse(torch.equal(mask_epoch_zero, mask_epoch_one))

    def test_epoch_is_visible_to_persistent_workers(self) -> None:
        self._write_forecast("2020-03-01", 0.2)
        self._write_forecast("2021-03-01", 0.7)
        dataset = M2MForecastDataset(self._config(), split="train")
        loader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=2,
            persistent_workers=True,
            shuffle=False,
        )
        try:
            dataset.set_epoch(0)
            mask_epoch_zero = next(iter(loader))["obs_mask"].clone()
            dataset.set_epoch(1)
            mask_epoch_one = next(iter(loader))["obs_mask"].clone()
            self.assertFalse(torch.equal(mask_epoch_zero, mask_epoch_one))
        finally:
            if loader._iterator is not None:
                loader._iterator._shutdown_workers()

    def test_epoch_resampling_is_opt_in_for_legacy_configs(self) -> None:
        self._write_forecast("2020-03-01", 0.2)
        self._write_forecast("2021-03-01", 0.7)
        config = self._config()
        config["resample_observation_masks_each_epoch"] = False
        dataset = M2MForecastDataset(config, split="train")

        dataset.set_epoch(0)
        first = dataset[0]["obs_mask"]
        dataset.set_epoch(4)
        second = dataset[0]["obs_mask"]

        self.assertTrue(torch.equal(first, second))

    def test_open_water_nan_becomes_zero_target_but_remains_in_valid_domain(self) -> None:
        self._write_forecast("2020-03-01", 0.2)
        target = np.full((1, 24, 8, 8), np.nan, dtype=np.float32)
        np.save(self.preds / "ocean+atmosphere_24_2021-03-01.npy", target)
        dataset = M2MForecastDataset(self._config(), split="train")

        item = dataset[0]

        self.assertTrue(torch.all(item["truth"] == -1.0))
        self.assertTrue(torch.all(item["valid_mask"] == 1.0))

    def test_calendar_features_are_cyclic_and_leap_safe(self) -> None:
        first = calendar_feature_values(date(2019, 3, 1), 23, ("day_of_year", "hour"))
        second = calendar_feature_values(date(2020, 3, 1), 23, ("day_of_year", "hour"))
        self.assertEqual(first, second)
        self.assertIsNone(previous_calendar_date(date(2020, 2, 29)))


class ConditioningAndLossTests(unittest.TestCase):
    def test_optional_calendar_maps_preserve_legacy_channel_contract(self) -> None:
        state = torch.zeros((2, 1, 4, 4))
        grid = torch.zeros((2, 2, 4, 4))
        field = torch.zeros_like(state)
        mask = torch.ones_like(state)

        legacy = make_conditioned_model_input(state, grid, field, mask, field, mask, mask)
        with_calendar = make_conditioned_model_input(
            state,
            grid,
            field,
            mask,
            field,
            mask,
            torch.ones((2, 3, 4, 4)),
        )

        self.assertEqual(legacy.shape[1], 8)
        self.assertEqual(with_calendar.shape[1], 10)

    def test_valid_domain_loss_excludes_only_masked_cells(self) -> None:
        prediction = torch.tensor([[[[10.0, 2.0], [3.0, 4.0]]]])
        target = torch.zeros_like(prediction)
        valid_mask = torch.tensor([[[[0.0, 1.0], [0.0, 0.0]]]])

        loss = UNetTrainer._masked_mse(prediction, target, valid_mask)

        self.assertEqual(float(loss), 4.0)

    def test_loss_domain_is_explicit_and_validated(self) -> None:
        self.assertEqual(TrainingConfig().loss_domain, "full")
        self.assertEqual(TrainingConfig.from_dict({"loss_domain": "valid"}).loss_domain, "valid")
        with self.assertRaisesRegex(ValueError, "loss_domain"):
            TrainingConfig.from_dict({"loss_domain": "oceanish"})


if __name__ == "__main__":
    unittest.main()
