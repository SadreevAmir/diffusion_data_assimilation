import json
import os
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from assim_lib.config import TrainingConfig
from assim_lib.direct_dynamics_cascade_coarse import COARSE_INPUT_CHANNELS
from assim_lib.direct_dynamics_cascade_coarse_training import (
    _Lifecycle,
    _record_failure,
    _seasonal_diagnostic_batch,
    run,
)
from assim_lib.direct_dynamics_cascade_fine_training import FINE_BATCH_KEYS


def _sample(index: int) -> dict:
    month = min(index // 4 + 1, 12)
    channels = {
        "truth": 6,
        "background": 6,
        "obs_values": 2,
        "obs_mask": 2,
        "valid_mask": 1,
        "water_mask": 5,
        "structured_conditioning": 15,
        "structured_physical_truth": 6,
        "structured_physical_background": 6,
    }
    sample = {key: torch.zeros(count, 8, 8) for key, count in channels.items()}
    sample["meta"] = {"case_id": f"2022-{month:02d}-03_slice12"}
    return sample


class CoarseCascadeRunnerTests(unittest.TestCase):
    def test_seasonal_diagnostic_batch_has_four_stable_months(self):
        dataset = [_sample(index) for index in range(48)]
        batch, positions = _seasonal_diagnostic_batch(dataset)
        self.assertEqual(positions, [4, 16, 28, 40])
        self.assertEqual(len(batch["meta"]["case_id"]), 4)
        self.assertEqual(set(batch), {*FINE_BATCH_KEYS, "meta"})
        self.assertEqual(len({case_id[5:7] for case_id in batch["meta"]["case_id"]}), 4)

    def test_committed_method_is_exact_bounded_lossless_contract(self):
        path = Path("config/methods/direct_dynamics_cascade_coarse_mechanics_2f.json")
        config = TrainingConfig.from_dict(json.loads(path.read_text(encoding="utf-8")))
        self.assertEqual(tuple(config.image_size), (160, 128))
        self.assertEqual((config.in_channels, config.out_channels), (COARSE_INPUT_CHANNELS, 6))
        self.assertEqual(config.train_batch_size * 512, 4096)
        self.assertEqual(config.num_sample_timesteps, 17)
        self.assertEqual(config.timestep_sampler, "stratified_uniform")
        self.assertFalse(config.activation_checkpointing)
        self.assertEqual(config.gradient_accumulation_steps, 1)
        self.assertEqual(config.metric_every_n_epochs, 0)
        self.assertEqual(config.sample_every_n_epochs, 0)

    def test_learning_curve_changes_only_budget_schedule_and_recovery(self):
        short = json.loads(
            Path("config/methods/direct_dynamics_cascade_coarse_mechanics_2f.json").read_text()
        )
        long = json.loads(
            Path("config/methods/direct_dynamics_cascade_coarse_learning_curve_2f.json").read_text()
        )
        differences = {key for key in set(short) | set(long) if short.get(key) != long.get(key)}
        self.assertEqual(
            differences,
            {
                "num_epochs",
                "base_output_dir",
                "run_name",
                "recovery_checkpoint_name",
            },
        )
        self.assertEqual(long["num_epochs"], 4)
        self.assertEqual(long.get("resume_from_checkpoint", ""), "")
        experiment = json.loads(
            Path("config/experiments/train_direct_dynamics_cascade_coarse_learning_curve_v1.json").read_text()
        )
        self.assertEqual(experiment["pilot"]["kind"], "learning_curve_2048")
        self.assertEqual(experiment["pilot"]["optimizer_updates"], 2048)

    def test_launcher_is_executable_bounded_and_uses_shared_gpu_lock(self):
        path = Path("scripts/run_direct_dynamics_cascade_coarse_mechanics.sh")
        launcher = path.read_text(encoding="utf-8")
        self.assertTrue(os.access(path, os.X_OK))
        self.assertIn(".gpu_job.lock", launcher)
        self.assertIn("exactly one", launcher)
        self.assertIn("{1..11}", launcher)
        self.assertIn("sleep 30", launcher)
        self.assertIn('"$observed_utilization" -ge 5', launcher)
        self.assertIn("timeout --foreground --signal=TERM --kill-after=2m 7080s", launcher)
        self.assertIn("COARSE_CASCADE_LAUNCH_ID", launcher)
        self.assertIn("COARSE_CASCADE_CONFIG", launcher)
        self.assertIn('PYTHON_MODE=("--preflight-only")', launcher)
        self.assertIn('"${PYTHON_MODE[@]}"', launcher)
        self.assertIn("usage: $0 [--preflight-only]", launcher)
        self.assertIn("train_direct_dynamics_cascade_coarse_learning_curve_v1.json", launcher)
        self.assertIn(
            "train_direct_dynamics_cascade_coarse_standardized_residual_v1.json",
            launcher,
        )
        self.assertIn(
            "train_direct_dynamics_cascade_coarse_mean_v1.json",
            launcher,
        )
        self.assertIn(
            "train_direct_dynamics_cascade_coarse_compact_v2.json",
            launcher,
        )
        self.assertNotIn("FINE_CASCADE_LAUNCH_ID", launcher)

    def test_preflight_result_is_explicitly_zero_optimizer_and_terminal(self):
        source = Path(
            "assim_lib/direct_dynamics_cascade_coarse_training.py"
        ).read_text(encoding="utf-8")
        self.assertIn('result["executed_optimizer_updates"] = 0', source)
        self.assertIn(
            '"executed_optimizer_updates": int(gate["optimizer_updates"])', source
        )
        self.assertIn('"preflight_passed",', source)
        self.assertIn('lifecycle.phase = "preflight_complete"', source)

    def test_standardized_residual_experiment_changes_only_coordinate_and_output(self):
        base = json.loads(
            Path(
                "config/experiments/train_direct_dynamics_cascade_coarse_residual_v1.json"
            ).read_text()
        )
        standardized = json.loads(
            Path(
                "config/experiments/"
                "train_direct_dynamics_cascade_coarse_standardized_residual_v1.json"
            ).read_text()
        )
        self.assertEqual(
            standardized["pilot"],
            {**base["pilot"], "kind": "standardized_persistence_residual_2048"},
        )
        self.assertEqual(standardized["data_config"], base["data_config"])
        self.assertEqual(standardized["model_config"], base["model_config"])
        self.assertEqual(standardized["data_overrides"], base["data_overrides"])
        self.assertEqual(standardized["residual_baseline"], base["residual_baseline"])
        self.assertEqual(
            set(standardized["training"]), {"base_output_dir", "run_name"}
        )
        self.assertEqual(len(standardized["residual_statistics"]["sha256"]), 64)

    def test_free_to_occupied_then_malformed_final_never_reaches_python(self):
        source = Path("scripts/run_direct_dynamics_cascade_coarse_mechanics.sh").read_text(encoding="utf-8")
        with TemporaryDirectory() as directory:
            root = Path(directory)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            result_root = root / "results"
            shared_launches = root / "shared_launches"
            shared_launches.mkdir()
            script = source.replace(
                'RESULT_ROOT="/home/autoresearch_results/direct_dynamics_cascade_v1"',
                f'RESULT_ROOT="{result_root}"',
            ).replace(
                'GPU_LOCK_ROOT="/home/autoresearch_results/direct_dynamics_all_hours_v1/launches"',
                f'GPU_LOCK_ROOT="{shared_launches}"',
            )
            launcher = root / "launcher.sh"
            launcher.write_text(script, encoding="utf-8")
            launcher.chmod(0o700)
            state_path = root / "gpu_state"
            python_called = root / "python_called"
            commands = {
                "git": "#!/usr/bin/env bash\nprintf '%040d\\n' 0\n",
                "flock": "#!/usr/bin/env bash\nexit 0\n",
                "sleep": "#!/usr/bin/env bash\nexit 0\n",
                "timeout": f"#!/usr/bin/env bash\ntouch '{python_called}'\nexit 0\n",
                "nvidia-smi": """#!/usr/bin/env bash
if [[ "$*" == *"query-gpu=count"* ]]; then
  echo 1
  exit 0
fi
count=0
if [[ -f "$GPU_SEQUENCE_STATE" ]]; then count="$(<"$GPU_SEQUENCE_STATE")"; fi
count=$((count + 1))
printf '%s\n' "$count" > "$GPU_SEQUENCE_STATE"
if [[ "$count" -eq 1 ]]; then echo '0, 0'
elif [[ "$count" -le 13 ]]; then echo '2048, 0'
else echo 'malformed'
fi
""",
            }
            for name, body in commands.items():
                path = fake_bin / name
                path.write_text(body, encoding="utf-8")
                path.chmod(0o700)
            environment = {
                **os.environ,
                "PATH": f"{fake_bin}:{os.environ['PATH']}",
                "COARSE_CASCADE_LAUNCH_ID": "admission-regression",
                "GPU_SEQUENCE_STATE": str(state_path),
            }
            result = subprocess.run(
                ["bash", str(launcher)],
                cwd=root,
                env=environment,
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("malformed observation", result.stderr)
            self.assertEqual(state_path.read_text(encoding="utf-8").strip(), "14")
            self.assertFalse(python_called.exists())

    def test_failure_cleanup_is_independent_and_preserves_primary_error(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            status_path = root / "status.json"

            def failing_mark_failed(**_kwargs):
                durable = json.loads(
                    (root / "output" / "coarse_cascade_failure.json").read_text(encoding="utf-8")
                )
                status = json.loads(status_path.read_text(encoding="utf-8"))
                self.assertEqual(durable["error"], "primary failure")
                self.assertEqual(status["error"], "primary failure")
                raise RuntimeError("mark cleanup")

            task = SimpleNamespace(mark_failed=Mock(side_effect=failing_mark_failed))
            tracker = SimpleNamespace(task=task, close=Mock(side_effect=RuntimeError("close cleanup")))
            accelerator = SimpleNamespace(end_training=Mock(side_effect=RuntimeError("accelerator cleanup")))
            lifecycle = _Lifecycle(
                run_id="failure-test",
                code_commit="b" * 40,
                phase="clearml_initialization",
                output_root=root / "output",
                output_writable=True,
                trainer=SimpleNamespace(clearml=tracker, accelerator=accelerator),
            )
            with patch.dict(os.environ, {"COARSE_CASCADE_STATUS_PATH": str(status_path)}):
                _record_failure(lifecycle, ValueError("primary failure"))
            tracker.close.assert_called_once()
            accelerator.end_training.assert_called_once()
            payload = json.loads(
                (root / "output" / "coarse_cascade_failure.json").read_text(encoding="utf-8")
            )
            self.assertEqual((payload["error_type"], payload["error"]), ("ValueError", "primary failure"))
            self.assertEqual(len(payload["cleanup_errors"]), 3)
            self.assertEqual(json.loads(status_path.read_text(encoding="utf-8"))["status"], "failed")

    def test_public_boundary_records_post_train_finalization_failure(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"

            def injected(*_args, lifecycle, **_kwargs):
                lifecycle.run_id = "post-train"
                lifecycle.code_commit = "c" * 40
                lifecycle.phase = "terminal_finalization"
                lifecycle.output_root = output
                lifecycle.output_writable = True
                lifecycle.trainer = SimpleNamespace(clearml=None, accelerator=None)
                raise FileNotFoundError("missing terminal gate")

            with (
                patch(
                    "assim_lib.direct_dynamics_cascade_coarse_training._run_impl",
                    side_effect=injected,
                ),
                self.assertRaisesRegex(FileNotFoundError, "missing terminal gate"),
            ):
                run(Path("unused.json"))
            payload = json.loads((output / "coarse_cascade_failure.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["phase"], "terminal_finalization")
            self.assertEqual(payload["error_type"], "FileNotFoundError")


if __name__ == "__main__":
    unittest.main()
