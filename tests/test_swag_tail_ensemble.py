from __future__ import annotations

import json
import math
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from assim_lib import swag_tail_ensemble as ensemble
from assim_lib import swag_tail_training as training
from assim_lib.deep_ensemble import COMPACT_OUTPUTS


class SwagTailTests(unittest.TestCase):
    def test_frozen_config_and_seed_contract(self) -> None:
        repo = Path(ensemble.__file__).resolve().parents[1]
        config = json.loads((repo / ensemble.CONFIG_NAME).read_text(encoding="utf-8"))
        training._validate_config(config)
        self.assertEqual(training.SNAPSHOT_EPOCHS, tuple(range(30, 40)))
        self.assertEqual(config["swag"]["matrix_chunk_parameters"], ensemble.MATRIX_CHUNK)
        self.assertEqual(config["swag"]["snapshot_state"], "raw non-EMA end-of-epoch weights")
        self.assertTrue(config["swag"]["dependent_gate"]["no_compensation"])
        self.assertEqual(ensemble.WEIGHT_SAMPLE_SEEDS, tuple(range(170100, 170110)))
        seeds = [
            ensemble.noise_seed(case_index, member_index)
            for case_index in range(ensemble.CASE_COUNT)
            for member_index in range(ensemble.MEMBER_COUNT)
        ]
        self.assertEqual(seeds, list(range(1234, 1634)))
        self.assertEqual(
            COMPACT_OUTPUTS,
            {
                "run_status.json",
                "metadata.json",
                "aggregate_case_mean_metrics.json",
                "per_case_metrics.csv",
            },
        )

    def test_scheduler_freezes_exactly_at_epoch_thirty_boundary(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor(0.0))
        optimizer = torch.optim.AdamW([parameter], lr=training.BASE_LEARNING_RATE)
        scheduler = training.build_frozen_tail_scheduler(
            optimizer=optimizer,
            num_warmup_steps=500,
            num_training_steps=4000,
        )
        self.assertEqual(training._SCHEDULER_RECORD["tail_start_step"], 3000)
        at_tail = training.frozen_tail_factor(3000, total_steps=4000, warmup_steps=500, tail_start_step=3000)
        after_tail = training.frozen_tail_factor(
            3999, total_steps=4000, warmup_steps=500, tail_start_step=3000
        )
        before_tail = training.frozen_tail_factor(
            2500, total_steps=4000, warmup_steps=500, tail_start_step=3000
        )
        self.assertEqual(at_tail, after_tail)
        self.assertGreater(before_tail, at_tail)
        self.assertTrue(
            math.isclose(
                training._SCHEDULER_RECORD["tail_learning_rate"],
                training.BASE_LEARNING_RATE * at_tail,
            )
        )
        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.LambdaLR)

    def test_chunked_standard_swag_draws_are_exact_finite_and_iid_seeded(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshots = []
            matrix_values = []
            for index in range(10):
                values = torch.tensor([float(index), float(2 * index + 1)])
                path = root / f"snapshot_{index}.pth"
                torch.save({"weight": values}, path)
                snapshots.append(path)
                matrix_values.append(values.numpy())
            matrix_path = root / "matrix.f32"
            inventory, count, matrix_hash = ensemble.build_snapshot_matrix(snapshots, matrix_path)
            records, statistics = ensemble.sample_swag_checkpoints(
                matrix_path, inventory, count, root / "members", chunk_size=1
            )
            self.assertEqual(count, 2)
            self.assertEqual(len(matrix_hash), 64)
            self.assertEqual(len(records), 10)
            self.assertEqual(len({row["sampled_checkpoint_sha256"] for row in records}), 10)
            self.assertEqual(statistics["snapshot_count"], 10)
            self.assertGreater(statistics["positive_diagonal_variance_count"], 0)

            statistics.update(
                {
                    "snapshot_matrix_sha256": matrix_hash,
                    "parameter_inventory_sha256": ensemble._canonical_hash(inventory),
                }
            )
            marker = root / "weight_manifest.json"
            ensemble._atomic_json(
                marker,
                {
                    "schema_version": 1,
                    "parameter_count": count,
                    "parameter_inventory": inventory,
                    "snapshot_matrix_sha256": matrix_hash,
                    "statistics": statistics,
                    "weight_records": records,
                },
            )
            recovered = ensemble._validate_weight_stage(marker, matrix_path, root / "members")
            self.assertEqual(recovered[0], records)
            self.assertEqual(recovered[3], count)

            matrix = np.stack(matrix_values).astype(np.float64)
            mean = matrix.mean(axis=0)
            variance = np.maximum(np.mean(matrix * matrix, axis=0) - mean * mean, 0.0)
            generator = torch.Generator(device="cpu").manual_seed(ensemble.WEIGHT_SAMPLE_SEEDS[0])
            coefficient = torch.randn(10, dtype=torch.float32, generator=generator).numpy()
            epsilon = np.asarray(
                [
                    torch.randn(1, dtype=torch.float32, generator=generator).item(),
                    torch.randn(1, dtype=torch.float32, generator=generator).item(),
                ]
            )
            expected = (
                mean
                + math.sqrt(0.5) * np.sqrt(variance) * epsilon
                + (matrix - mean[None, :]).T @ coefficient / math.sqrt(18.0)
            ).astype(np.float32)
            checkpoint = torch.load(
                root / "members" / records[0]["sampled_checkpoint_name"],
                map_location="cpu",
                weights_only=True,
            )
            np.testing.assert_allclose(checkpoint["weight"].numpy(), expected, rtol=0, atol=1e-6)

    def test_swag_fails_closed_for_collapsed_tail(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshots = []
            for index in range(10):
                path = root / f"snapshot_{index}.pth"
                torch.save({"weight": torch.ones(2)}, path)
                snapshots.append(path)
            inventory, count, _ = ensemble.build_snapshot_matrix(snapshots, root / "matrix.f32")
            with self.assertRaisesRegex(ValueError, "collapsed"):
                ensemble.sample_swag_checkpoints(
                    root / "matrix.f32", inventory, count, root / "members", chunk_size=1
                )

    def test_worker_maps_every_case_to_frozen_latent_seed(self) -> None:
        hashes = []

        def fake_generate(**kwargs):
            effective_seed = kwargs["seed"] + kwargs["case_order"]
            digest = f"{effective_seed:064x}"
            kwargs["initial_noise_hashes"].append(digest)
            kwargs["scaled_initial_noise_hashes"].append(digest)
            hashes.append(digest)
            return torch.zeros(1)

        ensemble._active_member_index = 3
        ensemble._worker_cases.clear()
        arguments = {
            "ensemble_size": 1,
            "sample_batch_size": 1,
            "num_timesteps": 25,
            "method": "dopri5",
            "rtol": 1e-5,
            "atol": 1e-6,
            "initial_noise_scale": 1.0,
            "autocast_dtype": None,
            "case_order": 7,
            "seed": 1234,
            "initial_noise_hashes": [],
            "scaled_initial_noise_hashes": [],
        }
        try:
            with mock.patch.object(ensemble, "_base_generate_ensemble", side_effect=fake_generate):
                ensemble._worker_generate_ensemble(**arguments)
        finally:
            ensemble._active_member_index = None
        expected_seed = ensemble.noise_seed(7, 3)
        self.assertEqual(hashes, [f"{expected_seed:064x}"])
        self.assertEqual(ensemble._worker_cases[0]["latent_seed"], expected_seed)

    def test_training_entrypoint_supports_narrow_injection(self) -> None:
        source = (Path(ensemble.__file__).resolve().parent / "main.py").read_text(encoding="utf-8")
        self.assertIn("trainer_class=None", source)
        self.assertIn("scheduler_factory=None", source)
        self.assertIn("trainer_type = trainer_class or UNetTrainer", source)

    def test_resume_rejects_path_traversal_and_cleans_only_known_orphans(self) -> None:
        with TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            resume_root = output_dir / "swag_resume"
            resume_root.mkdir()
            orphan = resume_root / ".epoch_01.123.incomplete"
            orphan.mkdir()
            trainer = training.SwagTailTrainer.__new__(training.SwagTailTrainer)
            trainer.output_dir = str(output_dir)
            trainer.accelerator = SimpleNamespace(num_processes=1)
            self.assertEqual(trainer._training_loop_start(), (0, 0))
            self.assertFalse(orphan.exists())

            training._atomic_json(
                resume_root / "latest.json",
                {
                    "schema_version": training.RESUME_SCHEMA_VERSION,
                    "state_dir": "epoch_../../escape",
                    "resume_sha256": "0" * 64,
                },
            )
            with self.assertRaisesRegex(ValueError, "identity"):
                trainer._training_loop_start()

    def test_completed_compact_contract_is_idempotent_and_hash_checked(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            samples = root / "samples"
            samples.mkdir()
            paths = []
            rows = []
            for case_index, expected_date in enumerate(ensemble.EXPECTED_DATES):
                path = samples / f"{case_index:03d}_{expected_date.isoformat()}_h23.npz"
                np.savez_compressed(path, analysis_ensemble=np.zeros((10, 1, 1)))
                paths.append(path)
                rows.append({"case_index": case_index})
            admission = {"schema_version": 1}
            admission_hash = ensemble._canonical_hash(admission)
            ensemble._atomic_csv(root / "per_case_metrics.csv", rows)
            ensemble._atomic_json(
                root / "aggregate_case_mean_metrics.json",
                {
                    "status": "sampling_completed_gate_pending",
                    "admission_manifest_sha256": admission_hash,
                },
            )
            metadata = {
                "status": "completed",
                "mode": ensemble.MODE,
                "candidate_method": ensemble.CANDIDATE_METHOD,
                "source_experiment": ensemble.SOURCE_EXPERIMENT,
                "num_cases": ensemble.CASE_COUNT,
                "ensemble_size": ensemble.MEMBER_COUNT,
                "gate_pending": True,
                "test_data_used_for_this_selection": False,
                "admission_manifest": admission,
                "admission_manifest_sha256": admission_hash,
                "sample_manifest_sha256": ensemble._sample_manifest(paths),
            }
            ensemble._validate_completed_output(root, metadata)
            paths[0].write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "sample manifest"):
                ensemble._validate_completed_output(root, metadata)

    def test_standalone_help_and_shell_are_fail_closed(self) -> None:
        repo = Path(ensemble.__file__).resolve().parents[1]
        for module in ("assim_lib.swag_tail_training", "assim_lib.swag_tail_ensemble"):
            result = subprocess.run(
                [sys.executable, "-m", module, "--help"],
                cwd=repo,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
        script = (repo / "scripts/run_swag_tail_weight_posterior.sh").read_text(encoding="utf-8")
        self.assertIn('MODE" != "validation_swag_tail_weight_posterior_sampling', script)
        self.assertIn("CUDA_VISIBLE_DEVICES", script)
        self.assertIn("must name exactly one numeric GPU", script)
        self.assertNotIn("curl", script)
        self.assertNotIn("wget", script)


if __name__ == "__main__":
    unittest.main()
