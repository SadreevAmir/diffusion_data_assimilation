import hashlib
import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from assim_lib.occurrence_intensity_e1_data_audit import (
    run_real_data_audit, validate_audit_config, validate_case_manifest,
)
from paper.validate_occurrence_intensity_e1_data_audit import validate_compact_audit

CONFIG = Path("config/experiments/occurrence_intensity_e1_data_audit.json")
MANIFEST = Path("paper/OCCURRENCE_INTENSITY_E1_REAL_CASES.json")


class FakeDataset:
    image_size = (320, 256)
    indices = [0, 1]
    means = [0.5, 0.5]
    stds = [0.5, 0.5]
    padding_values = [0.0, 0.0]
    hour_mode = "all"
    hours_per_day = 24
    background_strategy = "indexed"
    obs_shift = 0

    def __init__(self, root: Path):
        self.base_valid_mask = torch.zeros((2, *self.image_size))
        self.base_valid_mask[:, :4, :4] = 1
        cases = json.loads(MANIFEST.read_text())["cases"]
        self.obs_data = []
        self.records_by_date = {}
        self.sral_records = {}
        for case in cases:
            target = date.fromisoformat(case["target_date"])
            self.obs_data.append(SimpleNamespace(date=target))
            for lag in (0, 1, 2):
                current = target.fromordinal(target.toordinal() - lag)
                if current not in self.records_by_date:
                    path = root / f"source_{current.isoformat()}.npy"
                    field = np.zeros((2, 24, 4, 4), dtype=np.float32)
                    field[0] = (current.toordinal() % 101) / 100
                    field[1] = 0.25
                    np.save(path, field)
                    self.records_by_date[current] = SimpleNamespace(date=current, path=path)
                if current not in self.sral_records:
                    sral_path = root / f"sral_{current.isoformat()}.npy"
                    np.save(sral_path, np.ones((4, 4), dtype=np.float32))
                    self.sral_records[current] = [sral_path]

    def _num_days(self):
        return len(self.obs_data)

    def __getitem__(self, index):
        day, hour = divmod(index, 24)
        target = self.obs_data[day].date
        truth = torch.full((2, *self.image_size), (target.toordinal() % 97) / 96)
        return {
            "truth": truth, "valid_mask": self.base_valid_mask.clone(),
            "meta": {"case_id": f"{target.isoformat()}_h{hour:02d}"},
        }

    def sral_spatial_mask(self, target, *, day_offsets, transform_index):
        lag = list(day_offsets)[0]
        mask = torch.zeros(self.image_size)
        mask[lag:lag + 1, :4] = 1
        return mask

    def _sral_track_observations(self, target, hour, valid_mask):
        values = torch.zeros((2, *self.image_size))
        values[:, :4, :4] = 0.4
        return values, (values != 0).float(), {}

    def provenance(self):
        return {"split": "valid", "num_pairs": 8}


class E1RealDataAuditTest(unittest.TestCase):
    def test_frozen_config_and_manifest(self):
        validate_audit_config(json.loads(CONFIG.read_text()))
        validate_case_manifest(json.loads(MANIFEST.read_text()))

    def test_runner_uses_dataset_and_emits_valid_compact_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = run_real_data_audit(
                CONFIG, root / "output", dataset_builder=lambda config, split: FakeDataset(root)
            )
            self.assertEqual(len(result["per_case_audit"]), 8)
            self.assertTrue(all(len(case["lags"]) == 3 for case in result["per_case_audit"]))
            self.assertEqual(
                result["metadata"]["panel_layout"],
                [
                    "truth", "background_lag0", "mask_lag0", "background_lag1", "mask_lag1",
                    "background_lag2", "mask_lag2",
                ],
            )
            self.assertEqual(validate_compact_audit(CONFIG, root / "output"), "E1_REAL_DATA_AUDIT_PASS")

    def test_validator_rejects_future_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_real_data_audit(CONFIG, root / "output", dataset_builder=lambda config, split: FakeDataset(root))
            path = root / "output" / "per_case_audit.json"
            payload = json.loads(path.read_text())
            payload[0]["lags"][0]["future_date_leakage"] = True
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "future-date leakage"):
                validate_compact_audit(CONFIG, root / "output")

    def test_runner_rejects_missing_real_lag_footprint(self):
        class MissingLagDataset(FakeDataset):
            def sral_spatial_mask(self, target, *, day_offsets, transform_index):
                if list(day_offsets) == [2]:
                    return None
                return super().sral_spatial_mask(
                    target, day_offsets=day_offsets, transform_index=transform_index
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(
                ValueError, r"missing real SRAL footprint for .* lag=2"
            ):
                run_real_data_audit(
                    CONFIG,
                    root / "output",
                    dataset_builder=lambda config, split: MissingLagDataset(root),
                )

    def test_validator_rejects_zero_footprint_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_real_data_audit(
                CONFIG, root / "output", dataset_builder=lambda config, split: FakeDataset(root)
            )
            path = root / "output" / "per_case_audit.json"
            payload = json.loads(path.read_text())
            payload[0]["lags"][0]["footprint_pixels"] = 0
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "real lag footprint must be non-empty"):
                validate_compact_audit(CONFIG, root / "output")

    def test_validator_rejects_incomplete_sral_source_inventory_with_resealed_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_real_data_audit(
                CONFIG, root / "output", dataset_builder=lambda config, split: FakeDataset(root)
            )
            path = root / "output" / "metadata.json"
            payload = json.loads(path.read_text())
            payload["source_inventory"][0]["sources"][1]["sral_sha256"] = []
            inventory_bytes = json.dumps(
                payload["source_inventory"], sort_keys=True, separators=(",", ":")
            ).encode()
            payload["source_inventory_sha256"] = hashlib.sha256(inventory_bytes).hexdigest()
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "at least one real SRAL source hash"):
                validate_compact_audit(CONFIG, root / "output")

    def test_validator_rejects_case_metadata_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_real_data_audit(
                CONFIG, root / "output", dataset_builder=lambda config, split: FakeDataset(root)
            )
            path = root / "output" / "per_case_audit.json"
            payload = json.loads(path.read_text())
            payload[0]["coverage_slot"] = "tampered"
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "differs from the frozen manifest"):
                validate_compact_audit(CONFIG, root / "output")


if __name__ == "__main__":
    unittest.main()
