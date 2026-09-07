import hashlib
import json
import struct
import tempfile
import unittest
import zlib
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

import assim_lib.occurrence_intensity_e1_data_audit as audit_module
from assim_lib.occurrence_intensity_e1_data_audit import (
    run_real_data_audit,
    validate_audit_config,
    validate_case_manifest,
)
from paper.validate_occurrence_intensity_e1_data_audit import (
    _validate_manifest_contract,
    validate_compact_audit,
)

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
    observed_channels = [0]

    def __init__(self, root: Path):
        self.base_valid_mask = torch.zeros((2, *self.image_size))
        self.base_valid_mask[:, :4, :4] = 1
        cases = json.loads(MANIFEST.read_text())["cases"]
        self.obs_data = []
        self.records_by_date = {}
        self.sral_records = {}

        def ensure_forecast(current):
            if current in self.records_by_date:
                return
            path = root / f"source_{current.isoformat()}.npy"
            field = np.zeros((2, 24, 4, 4), dtype=np.float32)
            field[0] = (
                0.2 + 0.1 * (current.year - 2020)
                + current.timetuple().tm_yday / 1000
            )
            field[1] = 0.25
            np.save(path, field)
            self.records_by_date[current] = SimpleNamespace(date=current, path=path)

        for case in cases:
            target = date.fromisoformat(case["target_date"])
            self.obs_data.append(SimpleNamespace(date=target))
            for lag in (0, 1, 2):
                current = target.fromordinal(target.toordinal() - lag)
                background = current.replace(year=current.year - 1)
                ensure_forecast(current)
                ensure_forecast(background)
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
            "meta": {
                "case_id": f"{target.isoformat()}_h{hour:02d}",
                "target_path": str(self.records_by_date[target].path),
            },
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

    def test_runner_rejects_structurally_valid_case_identity_drift(self):
        manifest = json.loads(MANIFEST.read_text())
        manifest["cases"][0] = {
            "case_id": "2022-01-06_h23", "target_date": "2022-01-06", "hour": 23,
            "coverage_slot": "winter_early",
        }
        with self.assertRaisesRegex(ValueError, "frozen selection"):
            validate_case_manifest(manifest)

    def test_runner_rejects_structurally_valid_landmark_drift(self):
        manifest = json.loads(MANIFEST.read_text())
        manifest["orientation_landmarks"][0]["row"] += 1
        with self.assertRaisesRegex(ValueError, "frozen geographic anchors"):
            validate_case_manifest(manifest)

    def test_compact_validator_rejects_resealed_truth_dependent_manifest(self):
        manifest = json.loads(MANIFEST.read_text())
        manifest["truth_values_consulted"] = True
        with self.assertRaisesRegex(ValueError, "pre-truth selection"):
            _validate_manifest_contract(manifest)

    def test_compact_validator_rejects_resealed_selection_rule_drift(self):
        manifest = json.loads(MANIFEST.read_text())
        manifest["selection_rule"] = "eight_cases_selected_after_truth_review"
        with self.assertRaisesRegex(ValueError, "unreviewed selection rule"):
            _validate_manifest_contract(manifest)

    def test_compact_validator_rejects_selection_version_drift(self):
        manifest = json.loads(MANIFEST.read_text())
        manifest["selection_version"] = "e1_real_cases_v1"
        with self.assertRaisesRegex(ValueError, "selection version"):
            _validate_manifest_contract(manifest)

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
            self.assertEqual(
                result["metadata"]["panel_landmark_overlay"],
                {
                    "applied_to_each_tile": True,
                    "cross_radius_pixels": 3,
                    "alternating_values": [1.0, 0.0],
                    "landmarks": json.loads(MANIFEST.read_text())["orientation_landmarks"],
                },
            )
            self.assertEqual(validate_compact_audit(CONFIG, root / "output"), "E1_REAL_DATA_AUDIT_PASS")

    def test_runner_uses_target_year_values_and_previous_calendar_year_backgrounds(self):
        captured = []
        panel_fields = []
        original = audit_module.make_lagged_observation_channels
        original_panel = audit_module._panel

        def recording_channels(backgrounds, values, masks, ages, geometry, provenance):
            captured.append((backgrounds.clone(), values.clone(), masks.clone()))
            return original(backgrounds, values, masks, ages, geometry, provenance)

        def recording_panel(fields, landmarks):
            panel_fields.append([field.clone() for field in fields])
            return original_panel(fields, landmarks)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            seen_config = {}

            def builder(config, split):
                seen_config.update(config)
                return FakeDataset(root)

            with mock.patch.object(
                audit_module, "make_lagged_observation_channels", side_effect=recording_channels
            ), mock.patch.object(audit_module, "_panel", side_effect=recording_panel):
                result = run_real_data_audit(CONFIG, root / "output", dataset_builder=builder)

            self.assertEqual(seen_config["observed_channels"], [0])
            self.assertEqual(result["metadata"]["observed_channels"], [0])
            self.assertEqual(result["metadata"]["occurrence_encoder_value_space"], "physical_0_1")
            self.assertEqual(result["metadata"]["case_selection_version"], "e1_real_cases_v2")
            first = result["per_case_audit"][0]
            self.assertEqual(
                [lag["value_source_date"] for lag in first["lags"]],
                ["2022-01-02", "2022-01-01", "2021-12-31"],
            )
            self.assertEqual(
                [lag["background_source_date"] for lag in first["lags"]],
                ["2021-01-02", "2021-01-01", "2020-12-31"],
            )
            backgrounds, values, masks = captured[0]
            observed = masks.bool()
            self.assertTrue(torch.all((backgrounds[observed] >= 0) & (backgrounds[observed] <= 1)))
            self.assertTrue(torch.all((values[observed] >= 0) & (values[observed] <= 1)))
            self.assertTrue(torch.any((values - backgrounds)[observed] != 0))
            self.assertTrue(torch.all(torch.isnan(values[~observed])))
            self.assertTrue(all(lag["innovation_nonzero_pixels"] > 0 for lag in first["lags"]))
            expected_truth = (
                FakeDataset(root)[0]["truth"][0] * FakeDataset.stds[0]
                + FakeDataset.means[0]
            )
            torch.testing.assert_close(panel_fields[0][0], expected_truth)

    def test_runner_filters_target_raw_nan_and_counts_background_padding(self):
        class RawMissingSicDataset(FakeDataset):
            def __init__(self, root):
                super().__init__(root)
                first_case = json.loads(MANIFEST.read_text())["cases"][0]
                target = date.fromisoformat(first_case["target_date"])
                target_field = np.load(self.records_by_date[target].path)
                target_field[0, 23, 0, 0] = np.nan
                np.save(self.records_by_date[target].path, target_field)
                background = target.replace(year=target.year - 1)
                background_field = np.load(self.records_by_date[background].path)
                background_field[0, 23, 0, 1] = np.nan
                np.save(self.records_by_date[background].path, background_field)

        captured = []
        original = audit_module.make_lagged_observation_channels

        def recording_channels(backgrounds, values, masks, ages, geometry, provenance):
            captured.append((backgrounds.clone(), values.clone(), masks.clone()))
            return original(backgrounds, values, masks, ages, geometry, provenance)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch.object(
                audit_module, "make_lagged_observation_channels", side_effect=recording_channels
            ):
                result = run_real_data_audit(
                    CONFIG,
                    root / "output",
                    dataset_builder=lambda config, split: RawMissingSicDataset(root),
                )

            lag0 = result["per_case_audit"][0]["lags"][0]
            self.assertEqual(lag0["sral_footprint_pixels"], 4)
            self.assertEqual(lag0["footprint_pixels"], 3)
            self.assertEqual(lag0["target_raw_nonfinite_sral_pixels"], 1)
            self.assertEqual(lag0["background_raw_finite_observed_pixels"], 2)
            self.assertEqual(lag0["background_padding_observed_pixels"], 1)
            self.assertFalse(lag0["background_raw_finite_required_for_innovation"])
            backgrounds, values, masks = captured[0]
            self.assertEqual(float(masks[0, 0, 0, 0]), 0.0)
            self.assertEqual(float(backgrounds[0, 0, 0, 1]), 0.0)
            self.assertTrue(torch.all(torch.isfinite(backgrounds)))
            self.assertTrue(torch.all(torch.isfinite(values[masks.bool()])))

    def test_runner_executes_publication_channel_constructor_fail_closed(self):
        def unsafe_channels(backgrounds, values, masks, ages, geometry, provenance):
            result = torch.zeros((1, 24, *values.shape[-2:]), dtype=values.dtype)
            result[:, 1] = values[:, 0]
            return result

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch.object(
                audit_module, "make_lagged_observation_channels", side_effect=unsafe_channels
            ), self.assertRaisesRegex(ValueError, "leak non-finite values"):
                run_real_data_audit(
                    CONFIG, root / "output", dataset_builder=lambda config, split: FakeDataset(root)
                )

    def test_runner_rejects_constructor_that_zeroes_nonzero_innovations(self):
        original = audit_module.make_lagged_observation_channels

        def zeroed_innovations(backgrounds, values, masks, ages, geometry, provenance):
            result = original(backgrounds, values, masks, ages, geometry, provenance)
            lagged = result.reshape(1, 3, 8, *values.shape[-2:]).clone()
            lagged[:, :, 0] = 0
            return lagged.reshape_as(result)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch.object(
                audit_module,
                "make_lagged_observation_channels",
                side_effect=zeroed_innovations,
            ), self.assertRaisesRegex(ValueError, "lagged forecast protocol"):
                run_real_data_audit(
                    CONFIG, root / "output", dataset_builder=lambda config, split: FakeDataset(root)
                )

    def test_runner_rejects_standardized_sic_as_physical_encoder_input(self):
        class StandardizedSicDataset(FakeDataset):
            def __init__(self, root):
                super().__init__(root)
                first_case = json.loads(MANIFEST.read_text())["cases"][0]
                target = date.fromisoformat(first_case["target_date"])
                path = self.records_by_date[target].path
                field = np.load(path)
                field[0] = -1.0
                np.save(path, field)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, r"outside physical \[0,1\]"):
                run_real_data_audit(
                    CONFIG,
                    root / "output",
                    dataset_builder=lambda config, split: StandardizedSicDataset(root),
                )

    def test_validator_rejects_landmark_overlay_contract_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_real_data_audit(
                CONFIG, root / "output", dataset_builder=lambda config, split: FakeDataset(root)
            )
            path = root / "output" / "metadata.json"
            payload = json.loads(path.read_text())
            payload["panel_landmark_overlay"]["cross_radius_pixels"] = 2
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "landmark overlay"):
                validate_compact_audit(CONFIG, root / "output")

    def test_validator_rejects_resealed_missing_landmark_pixel(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            result = run_real_data_audit(
                CONFIG, output, dataset_builder=lambda config, split: FakeDataset(root)
            )
            panel = output / result["per_case_audit"][0]["panel"]
            payload = panel.read_bytes()
            ihdr_end = 8 + 12 + 13
            width, height = struct.unpack(">II", payload[16:24])
            compressed = payload[ihdr_end + 8:-16]
            self.assertEqual(payload[ihdr_end + 4:ihdr_end + 8], b"IDAT")
            scanlines = bytearray(zlib.decompress(compressed))
            row, column = 32, 32
            scanlines[row * (width + 1) + 1 + column] = 127

            def chunk(kind, body):
                content = kind + body
                return struct.pack(">I", len(body)) + content + struct.pack(">I", zlib.crc32(content))

            corrupted = (
                payload[:8]
                + chunk(b"IHDR", payload[16:29])
                + chunk(b"IDAT", zlib.compress(bytes(scanlines)))
                + chunk(b"IEND", b"")
            )
            panel.write_bytes(corrupted)
            audit_path = output / "per_case_audit.json"
            audit = json.loads(audit_path.read_text())
            audit[0]["panel_sha256"] = hashlib.sha256(corrupted).hexdigest()
            audit_path.write_text(json.dumps(audit))
            with self.assertRaisesRegex(ValueError, "overlay pixels"):
                validate_compact_audit(CONFIG, output)

    def test_runner_publishes_completion_marker_last_and_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            published = []

            def recording_write(path, payload):
                published.append(path.name)
                return original_write(path, payload)

            original_write = audit_module._write_json_atomic
            with mock.patch.object(audit_module, "_write_json_atomic", side_effect=recording_write):
                run_real_data_audit(
                    CONFIG, root / "output", dataset_builder=lambda config, split: FakeDataset(root)
                )
            self.assertEqual(
                published,
                ["run_status.json", "metadata.json", "per_case_audit.json", "run_status.json"],
            )
            self.assertEqual(list((root / "output").glob(".*.tmp")), [])

    def test_runner_invalidates_stale_completion_before_dataset_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            output.mkdir()
            (output / "run_status.json").write_text(json.dumps({"status": "completed"}))

            def failing_builder(config, split):
                raise RuntimeError("dataset unavailable")

            with self.assertRaisesRegex(RuntimeError, "dataset unavailable"):
                run_real_data_audit(CONFIG, output, dataset_builder=failing_builder)

            self.assertEqual(
                json.loads((output / "run_status.json").read_text()),
                {
                    "status": "running", "mode": audit_module.MODE,
                    "resource_kind": "server_cpu", "cases": 8,
                    "engineering_only": True, "calibration_claim": False,
                },
            )

    def test_runner_rejects_symlinked_output_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            redirected = root / "redirected"
            redirected.mkdir()
            output = root / "output"
            output.symlink_to(redirected, target_is_directory=True)

            with self.assertRaisesRegex(ValueError, "output directory must not be a symbolic link"):
                run_real_data_audit(
                    CONFIG, output, dataset_builder=lambda config, split: FakeDataset(root)
                )

            self.assertEqual(list(redirected.iterdir()), [])

    def test_runner_rejects_symlinked_panels_before_panel_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            output.mkdir()
            redirected = root / "redirected_panels"
            redirected.mkdir()
            (output / "panels").symlink_to(redirected, target_is_directory=True)

            with self.assertRaisesRegex(ValueError, "symbolic links"):
                run_real_data_audit(
                    CONFIG, output, dataset_builder=lambda config, split: FakeDataset(root)
                )

            self.assertEqual(list(redirected.iterdir()), [])

    def test_runner_rejects_symlinked_panel_file_before_any_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            panels = output / "panels"
            panels.mkdir(parents=True)
            redirected = root / "redirected_panel.png"
            redirected.write_bytes(b"unchanged")
            first_case = json.loads(MANIFEST.read_text())["cases"][0]["case_id"]
            panel = panels / f"{first_case}_truth_background_lag_masks.png"
            panel.symlink_to(redirected)

            with self.assertRaisesRegex(ValueError, "output tree must not contain symbolic links"):
                run_real_data_audit(
                    CONFIG, output, dataset_builder=lambda config, split: FakeDataset(root)
                )

            self.assertEqual(redirected.read_bytes(), b"unchanged")
            self.assertFalse((output / "run_status.json").exists())

    def test_validator_rejects_future_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_real_data_audit(
                CONFIG,
                root / "output",
                dataset_builder=lambda config, split: FakeDataset(root),
            )
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

    def test_runner_rejects_empty_sral_inventory_before_truth_read(self):
        class EmptyInventoryDataset(FakeDataset):
            def __init__(self, root):
                super().__init__(root)
                first_case = json.loads(MANIFEST.read_text())["cases"][0]
                self.sral_records[date.fromisoformat(first_case["target_date"])] = []

            def __getitem__(self, index):
                raise AssertionError("truth must not be read before source inventory is complete")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "missing pre-truth SRAL source metadata"):
                run_real_data_audit(
                    CONFIG,
                    root / "output",
                    dataset_builder=lambda config, split: EmptyInventoryDataset(root),
                )

    def test_runner_rejects_forecast_record_date_drift_before_truth_read(self):
        class DriftedRecordDataset(FakeDataset):
            def __init__(self, root):
                super().__init__(root)
                first_case = json.loads(MANIFEST.read_text())["cases"][0]
                target = date.fromisoformat(first_case["target_date"])
                self.records_by_date[target].date = target.fromordinal(target.toordinal() - 1)

            def __getitem__(self, index):
                raise AssertionError("truth must not be read before source dates are verified")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "forecast record date differs"):
                run_real_data_audit(
                    CONFIG,
                    root / "output",
                    dataset_builder=lambda config, split: DriftedRecordDataset(root),
                )

    def test_runner_rejects_truth_source_drift_after_pretruth_seal(self):
        class DriftedTruthSourceDataset(FakeDataset):
            def __init__(self, root):
                super().__init__(root)
                self.drifted_truth_path = root / "drifted_truth.npy"
                np.save(self.drifted_truth_path, np.zeros((2, 24, 4, 4), dtype=np.float32))

            def __getitem__(self, index):
                item = super().__getitem__(index)
                item["meta"]["target_path"] = str(self.drifted_truth_path)
                return item

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "truth source differs"):
                run_real_data_audit(
                    CONFIG,
                    root / "output",
                    dataset_builder=lambda config, split: DriftedTruthSourceDataset(root),
                )

    def test_runner_rejects_valid_domain_drift_after_pretruth_seal(self):
        class DriftedValidDomainDataset(FakeDataset):
            def __getitem__(self, index):
                item = super().__getitem__(index)
                item["valid_mask"][0, 10, 10] = 1
                return item

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "valid domain differs"):
                run_real_data_audit(
                    CONFIG,
                    root / "output",
                    dataset_builder=lambda config, split: DriftedValidDomainDataset(root),
                )

    def test_runner_rejects_runtime_forecast_source_drift_after_pretruth_seal(self):
        class DriftedRuntimeForecastDataset(FakeDataset):
            def __getitem__(self, index):
                item = super().__getitem__(index)
                day, _ = divmod(index, 24)
                target = self.obs_data[day].date
                lagged = target.fromordinal(target.toordinal() - 1)
                replacement = self.records_by_date[lagged].path.with_name("replacement.npy")
                np.save(replacement, np.zeros((2, 24, 4, 4), dtype=np.float32))
                self.records_by_date[lagged].path = replacement
                return item

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "runtime forecast source differs"):
                run_real_data_audit(
                    CONFIG,
                    root / "output",
                    dataset_builder=lambda config, split: DriftedRuntimeForecastDataset(root),
                )

    def test_runner_rejects_runtime_sral_source_drift_after_pretruth_seal(self):
        class DriftedRuntimeSralDataset(FakeDataset):
            def __getitem__(self, index):
                item = super().__getitem__(index)
                day, _ = divmod(index, 24)
                target = self.obs_data[day].date
                replacement = self.sral_records[target][0].with_name("replacement_sral.npy")
                np.save(replacement, np.ones((4, 4), dtype=np.float32))
                self.sral_records[target] = [replacement]
                return item

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "runtime SRAL sources differ"):
                run_real_data_audit(
                    CONFIG,
                    root / "output",
                    dataset_builder=lambda config, split: DriftedRuntimeSralDataset(root),
                )

    def test_validator_rejects_zero_footprint_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_real_data_audit(
                CONFIG, root / "output", dataset_builder=lambda config, split: FakeDataset(root)
            )
            path = root / "output" / "per_case_audit.json"
            payload = json.loads(path.read_text())
            lag = payload[0]["lags"][0]
            lag["footprint_pixels"] = 0
            lag["target_raw_nonfinite_sral_pixels"] = lag["sral_footprint_pixels"]
            lag["background_raw_finite_observed_pixels"] = 0
            lag["background_padding_observed_pixels"] = 0
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "real lag footprint must be non-empty"):
                validate_compact_audit(CONFIG, root / "output")

    def test_validator_rejects_resealed_footprint_not_matching_mask_panel(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_real_data_audit(
                CONFIG, root / "output", dataset_builder=lambda config, split: FakeDataset(root)
            )
            path = root / "output" / "per_case_audit.json"
            payload = json.loads(path.read_text())
            lag = payload[0]["lags"][0]
            lag["footprint_pixels"] += 100
            lag["sral_footprint_pixels"] += 100
            lag["background_raw_finite_observed_pixels"] += 100
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "differs from rendered mask pixels"):
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
            with self.assertRaisesRegex(ValueError, "one-to-one with hashes"):
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

    def test_validator_rejects_dataset_config_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_real_data_audit(
                CONFIG, root / "output", dataset_builder=lambda config, split: FakeDataset(root)
            )
            path = root / "output" / "metadata.json"
            payload = json.loads(path.read_text())
            payload["dataset_config_sha256"] = "0" * 64
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "frozen dataset config"):
                validate_compact_audit(CONFIG, root / "output")

    def test_validator_rejects_rehashed_complete_config_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_real_data_audit(
                CONFIG, root / "output", dataset_builder=lambda config, split: FakeDataset(root)
            )
            drifted_config = root / "drifted_config.json"
            payload = json.loads(CONFIG.read_text())
            payload["sral_transform_index"] = 10
            drifted_config.write_text(json.dumps(payload))
            metadata_path = root / "output" / "metadata.json"
            metadata = json.loads(metadata_path.read_text())
            metadata["config_sha256"] = hashlib.sha256(drifted_config.read_bytes()).hexdigest()
            metadata_path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "complete frozen audit contract"):
                validate_compact_audit(drifted_config, root / "output")

    def test_validator_rejects_source_path_hash_cardinality_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_real_data_audit(
                CONFIG, root / "output", dataset_builder=lambda config, split: FakeDataset(root)
            )
            path = root / "output" / "metadata.json"
            payload = json.loads(path.read_text())
            payload["source_inventory"][0]["sources"][0]["sral_paths"].append("unbound.npy")
            inventory_bytes = json.dumps(
                payload["source_inventory"], sort_keys=True, separators=(",", ":")
            ).encode()
            payload["source_inventory_sha256"] = hashlib.sha256(inventory_bytes).hexdigest()
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "one-to-one with hashes"):
                validate_compact_audit(CONFIG, root / "output")

    def test_validator_rejects_resealed_source_hash_not_matching_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_real_data_audit(
                CONFIG, root / "output", dataset_builder=lambda config, split: FakeDataset(root)
            )
            path = root / "output" / "metadata.json"
            payload = json.loads(path.read_text())
            payload["source_inventory"][0]["sources"][0]["value_forecast_sha256"] = "0" * 64
            inventory_bytes = json.dumps(
                payload["source_inventory"], sort_keys=True, separators=(",", ":")
            ).encode()
            payload["source_inventory_sha256"] = hashlib.sha256(inventory_bytes).hexdigest()
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "does not match its sealed hash"):
                validate_compact_audit(CONFIG, root / "output")

    def test_validator_rejects_panel_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_real_data_audit(
                CONFIG, root / "output", dataset_builder=lambda config, split: FakeDataset(root)
            )
            panel = next((root / "output" / "panels").glob("*.png"))
            payload = bytearray(panel.read_bytes())
            payload[-1] ^= 1
            panel.write_bytes(payload)
            with self.assertRaisesRegex(ValueError, "differs from its sealed hash"):
                validate_compact_audit(CONFIG, root / "output")

    def test_validator_rejects_resealed_extra_png_chunk(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            result = run_real_data_audit(
                CONFIG, output, dataset_builder=lambda config, split: FakeDataset(root)
            )
            panel = output / result["per_case_audit"][0]["panel"]
            payload = panel.read_bytes()
            ihdr_end = 8 + 12 + 13
            body = b"audit"
            kind = b"tEXt"
            extra = (
                struct.pack(">I", len(body)) + kind + body
                + struct.pack(">I", zlib.crc32(kind + body))
            )
            resealed = payload[:ihdr_end] + extra + payload[ihdr_end:]
            panel.write_bytes(resealed)
            audit_path = output / "per_case_audit.json"
            audit = json.loads(audit_path.read_text())
            audit[0]["panel_sha256"] = hashlib.sha256(resealed).hexdigest()
            audit_path.write_text(json.dumps(audit))
            with self.assertRaisesRegex(ValueError, "frozen PNG chunk sequence"):
                validate_compact_audit(CONFIG, output)

    def test_validator_rejects_resealed_concatenated_zlib_stream(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            result = run_real_data_audit(
                CONFIG, output, dataset_builder=lambda config, split: FakeDataset(root)
            )
            panel = output / result["per_case_audit"][0]["panel"]
            payload = panel.read_bytes()
            ihdr_end = 8 + 12 + 13
            compressed_length = struct.unpack(">I", payload[ihdr_end:ihdr_end + 4])[0]
            compressed_start = ihdr_end + 8
            compressed_end = compressed_start + compressed_length
            concatenated = payload[compressed_start:compressed_end] + zlib.compress(b"extra")
            kind = b"IDAT"
            idat = (
                struct.pack(">I", len(concatenated)) + kind + concatenated
                + struct.pack(">I", zlib.crc32(kind + concatenated))
            )
            resealed = payload[:ihdr_end] + idat + payload[compressed_end + 4:]
            panel.write_bytes(resealed)
            audit_path = output / "per_case_audit.json"
            audit = json.loads(audit_path.read_text())
            audit[0]["panel_sha256"] = hashlib.sha256(resealed).hexdigest()
            audit_path.write_text(json.dumps(audit))
            with self.assertRaisesRegex(ValueError, "concatenated zlib stream"):
                validate_compact_audit(CONFIG, output)

    def test_validator_rejects_extra_panel_from_reused_output_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_real_data_audit(
                CONFIG, root / "output", dataset_builder=lambda config, split: FakeDataset(root)
            )
            extra = root / "output" / "panels" / "stale_previous_attempt.png"
            extra.write_bytes(next((root / "output" / "panels").glob("*.png")).read_bytes())
            with self.assertRaisesRegex(ValueError, "inventory must match"):
                validate_compact_audit(CONFIG, root / "output")

    def test_validator_rejects_extra_nonselected_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_real_data_audit(
                CONFIG, root / "output", dataset_builder=lambda config, split: FakeDataset(root)
            )
            (root / "output" / "debug_dump.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "artifact inventory must match"):
                validate_compact_audit(CONFIG, root / "output")

    def test_validator_rejects_panel_reassigned_to_another_case(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_real_data_audit(
                CONFIG, root / "output", dataset_builder=lambda config, split: FakeDataset(root)
            )
            path = root / "output" / "per_case_audit.json"
            payload = json.loads(path.read_text())
            payload[0]["panel"] = payload[1]["panel"]
            payload[0]["panel_sha256"] = payload[1]["panel_sha256"]
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "not bound to its frozen case"):
                validate_compact_audit(CONFIG, root / "output")

    def test_validator_rejects_geographic_landmark_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_real_data_audit(
                CONFIG, root / "output", dataset_builder=lambda config, split: FakeDataset(root)
            )
            path = root / "output" / "per_case_audit.json"
            payload = json.loads(path.read_text())
            payload[0]["orientation_landmarks"][0]["row"] += 1
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "frozen geographic anchors"):
                validate_compact_audit(CONFIG, root / "output")


if __name__ == "__main__":
    unittest.main()
