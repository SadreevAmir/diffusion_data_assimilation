from __future__ import annotations

import json
import subprocess
import sys
import unittest
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from assim_lib import external_2024_calendar_raw as raw


class External2024CalendarRawTests(unittest.TestCase):
    def test_signed_contract_and_static_config_are_exact(self) -> None:
        self.assertEqual(raw._canonical_hash(raw.frozen_primary_contract()), raw.PRIMARY_CONTRACT_SHA256)
        self.assertEqual(raw.EXPECTED_DATES[0].isoformat(), "2024-01-01")
        self.assertEqual(raw.EXPECTED_DATES[-1].isoformat(), "2024-02-17")
        self.assertEqual(len(raw.EXPECTED_DATES), 48)
        self.assertEqual(
            raw._canonical_hash([value.isoformat() for value in raw.EXPECTED_DATES]),
            raw.TARGET_DATE_MANIFEST_SHA256,
        )
        repo = Path(raw.__file__).resolve().parents[1]
        config = json.loads((repo / raw.CONFIG_PATH).read_text(encoding="utf-8"))
        self.assertIsNone(config["data_overrides"]["split_protocol"])
        self.assertEqual(config["data_overrides"]["test"]["obs_start_day"], "2024-01-01")
        self.assertEqual(config["comparison"]["expected_num_cases"], 48)
        self.assertEqual(config["comparison"]["ensemble_size"], 10)
        self.assertTrue(config["comparison"]["save_ensembles"])

    def test_noise_schedule_is_exact_and_unique(self) -> None:
        values = [
            raw.noise_seed(case_index, member_index)
            for case_index in range(raw.CASE_COUNT)
            for member_index in range(raw.MEMBER_COUNT)
        ]
        self.assertEqual(values, list(range(1234, 1714)))
        with self.assertRaises(ValueError):
            raw.noise_seed(48, 0)

    def test_input_seal_hashes_only_exact_inventory(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / "repo"
            dataset = root / "dataset"
            preds = dataset / "preds"
            sral = root / "sral"
            preds.mkdir(parents=True)
            sral.mkdir()
            for relative in (
                raw.CONFIG_PATH,
                raw.DATA_CONFIG_PATH,
                raw.MODEL_CONFIG_PATH,
                "assim_lib/external_2024_calendar_raw.py",
                "assim_lib/compare_3dvar.py",
                "assim_lib/data.py",
                "assim_lib/evaluate.py",
            ):
                path = repo / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(relative, encoding="utf-8")
            for target in sorted({*raw.BACKGROUND_DATES, *raw.TRACK_DATES}):
                (preds / f"ocean+atmosphere_24_{target.isoformat()}.npy").write_bytes(b"p")
            for target in raw.TRACK_DATES:
                (sral / f"{target.isoformat()}.npy").write_bytes(b"s")
            mask = root / "land_mask.npy"
            mask.write_bytes(b"m")
            seal = raw.build_input_seal(repo, dataset, sral, mask)
            self.assertTrue(seal["dense_target_may_be_loaded_only_to_construct_masked_model_observations"])
            self.assertFalse(seal["holdout_outcomes_used_for_metrics_or_method_selection"])
            self.assertEqual(len(seal["forecast_records"]), 99)
            self.assertEqual(len(seal["sral_records"]), 51)
            payload = {key: value for key, value in seal.items() if key != "seal_sha256"}
            self.assertEqual(seal["seal_sha256"], raw._canonical_hash(payload))

    def test_sample_seal_is_complete_before_candidate_access(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            sampling = root / "sampling"
            samples = sampling / "samples"
            samples.mkdir(parents=True)
            cases = []
            for case_index, (target, background) in enumerate(
                zip(raw.EXPECTED_DATES, raw.BACKGROUND_DATES, strict=True)
            ):
                noise = [f"{raw.noise_seed(case_index, member):064x}" for member in range(10)]
                cases.append(
                    {
                        "case_order": case_index,
                        "target_date": target.isoformat(),
                        "background_date": background.isoformat(),
                        "conditioning_track_days": [
                            {
                                "offset": offset,
                                "date": (target - timedelta(days=offset)).isoformat(),
                                "num_sral_files": 1,
                                "sral_files": [
                                    f"/sealed/sral/{(target - timedelta(days=offset)).isoformat()}.npy"
                                ],
                            }
                            for offset in range(3)
                        ],
                        "track_imitation_date": (target + timedelta(days=1)).isoformat(),
                        "track_imitation_sral_files": [
                            f"/sealed/sral/{(target + timedelta(days=1)).isoformat()}.npy"
                        ],
                        "conditioning_obs_count": 3,
                        "track_imitation_count": 1,
                        "conditioning_mask_sha256": f"{case_index + 1:064x}",
                        "track_imitation_mask_sha256": f"{case_index + 101:064x}",
                        "mask_kind": "sral_tracks",
                        "sral_files_used": 3,
                        "empty_obs_days": 0,
                        "initial_noise_sha256": noise,
                        "scaled_initial_noise_sha256": noise,
                    }
                )
                np.savez_compressed(
                    samples / f"{case_index:03d}_{target.isoformat()}_h23.npz",
                    analysis_ensemble=np.zeros((10, 2, 3), dtype=np.float32),
                    valid_mask=np.ones((2, 3), dtype=bool),
                )
            (sampling / "cases.json").write_text(json.dumps(cases), encoding="utf-8")
            metadata = {
                "dataset_split": "test",
                "start_date": "2024-01-01",
                "end_date": "2024-02-17",
                "num_cases": 48,
                "case_stride_days": 1,
                "target_hour": 23,
                "ensemble_size": 10,
                "sample_batch_size": 10,
                "num_timesteps": 25,
                "method": "dopri5",
                "rtol": 1e-5,
                "atol": 1e-6,
                "inference_precision": "float32",
                "seed": 1234,
                "initial_noise_scale": 1.0,
                "field_protocol": "siconc-only",
                "conditioning_mode": "full",
                "cfg_mode": "none",
                "cfg_background_scale": 1.0,
                "cfg_observation_scale": 1.0,
                "save_ensembles": True,
                "sealed_raw_only": True,
                "outcome_metrics_computed": False,
                "truth_saved_with_raw_samples": False,
                "normalization_means": [0.0, 0.0],
                "normalization_stds": [1.0, 1.0],
                "checkpoint": f"/frozen/{raw.CHECKPOINT_NAME}",
                "checkpoint_training_split_kind": "legacy_overlapping_2023_validation",
                "checkpoint_selection": (
                    "fixed final epoch; legacy 2023 validation was not used for checkpoint selection"
                ),
                "temporal_split_protocol": "external_chronological_holdout",
                "cases_file": "cases.json",
            }
            (sampling / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            (sampling / "run_status.json").write_text(
                json.dumps({"status": "completed", "completed_cases": 48}), encoding="utf-8"
            )
            input_seal = {
                "sral_records": [{"path": f"{target.isoformat()}.npy"} for target in raw.TRACK_DATES]
            }
            sealed = raw.seal_samples(sampling, root / "sealed", input_seal)
            self.assertEqual(sealed["case_count"], 48)
            self.assertEqual(sum(len(row["members"]) for row in sealed["cases"]), 480)
            self.assertFalse(sealed["candidate_transform_started"])
            self.assertFalse(sealed["candidate_truth_read"])
            self.assertTrue((root / "sealed" / raw.SAMPLE_SEAL_NAME).is_file())
            self.assertEqual(
                raw.validate_existing_sample_seal(root / "sealed")["sealed_manifest_sha256"],
                sealed["sealed_manifest_sha256"],
            )
            # A crash after the atomic directory move must not be mistaken for a
            # resumable completed sampling stage; run() will resample safely.
            self.assertFalse(raw._sampling_stage_complete(sampling))
            with np.load(root / "sealed/samples/000_2024-01-01_h23.npz", allow_pickle=False) as payload:
                self.assertEqual(set(payload.files), {"analysis_ensemble", "valid_mask"})

    def test_duplicate_input_or_noise_inventory_fails_closed(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / "repo"
            dataset = root / "dataset"
            preds = dataset / "preds"
            sral = root / "sral"
            preds.mkdir(parents=True)
            sral.mkdir()
            for relative in (
                raw.CONFIG_PATH,
                raw.DATA_CONFIG_PATH,
                raw.MODEL_CONFIG_PATH,
                "assim_lib/external_2024_calendar_raw.py",
                "assim_lib/compare_3dvar.py",
                "assim_lib/data.py",
                "assim_lib/evaluate.py",
            ):
                path = repo / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(relative, encoding="utf-8")
            dates = sorted({*raw.BACKGROUND_DATES, *raw.TRACK_DATES})
            for target in dates:
                (preds / f"ocean+atmosphere_24_{target.isoformat()}.npy").write_bytes(b"p")
            (preds / "ocean+atmosphere_24_duplicate_2024-01-01.npy").write_bytes(b"p")
            for target in raw.TRACK_DATES:
                (sral / f"{target.isoformat()}.npy").write_bytes(b"s")
            mask = root / "mask.npy"
            mask.write_bytes(b"m")
            with self.assertRaisesRegex(ValueError, "exactly one file per date"):
                raw.build_input_seal(repo, dataset, sral, mask)

    def test_runtime_requires_offline_single_gpu(self) -> None:
        with (
            patch.dict(
                "os.environ",
                {"CUDA_VISIBLE_DEVICES": "0,1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
                clear=True,
            ),
            self.assertRaisesRegex(RuntimeError, "exactly one"),
        ):
            raw._validate_runtime()

    def test_initial_sample_seal_rejects_invalid_dtypes_and_range(self) -> None:
        valid = np.ones((2, 3), dtype=bool)
        with self.assertRaisesRegex(ValueError, "float32"):
            raw._validate_raw_sample_arrays(np.zeros((10, 2, 3), dtype=np.float64), valid, context="initial")
        with self.assertRaisesRegex(ValueError, "boolean"):
            raw._validate_raw_sample_arrays(
                np.zeros((10, 2, 3), dtype=np.float32),
                valid.astype(np.float32),
                context="initial",
            )
        out_of_range = np.zeros((10, 2, 3), dtype=np.float32)
        out_of_range[0, 0, 0] = 2.0
        with self.assertRaisesRegex(ValueError, "within"):
            raw._validate_raw_sample_arrays(out_of_range, valid, context="initial")

    def test_symlinked_input_root_and_checkpoint_mutation_are_rejected(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            real = root / "real"
            real.mkdir()
            alias = root / "alias"
            alias.symlink_to(real, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "is a symlink"):
                raw._resolved_without_symlinks(alias, "dataset root")
            expected = {"checkpoint_sha256": "a" * 64}
            with patch.object(raw, "_validate_source", return_value={"checkpoint_sha256": "b" * 64}):
                with self.assertRaisesRegex(RuntimeError, "checkpoint source changed"):
                    raw._checkpoint_source_unchanged(real, expected)

    def test_help_and_shell_are_fail_closed(self) -> None:
        repo = Path(raw.__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "-m", "assim_lib.external_2024_calendar_raw", "--help"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        script = (repo / "scripts/run_external_2024_calendar_raw48.sh").read_text(encoding="utf-8")
        self.assertIn('MODE" != "external_2024_calendar_global_bias_raw48_primary', script)
        self.assertNotIn("curl", script)
        self.assertNotIn("wget", script)


if __name__ == "__main__":
    unittest.main()
