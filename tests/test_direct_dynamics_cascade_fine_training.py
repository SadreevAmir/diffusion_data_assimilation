import os
import json
import subprocess
import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch

from assim_lib.config import TrainingConfig
from assim_lib.direct_dynamics_cascade_fine_training import (
    FINE_BATCH_KEYS,
    _Lifecycle,
    _clean_code_identity,
    _evenly_spaced_indices,
    _fine_collate,
    _ipc_preflight,
    _initialize_trainer,
    _record_failure,
    run,
)


def _sample(case_id: str) -> dict:
    height, width = 4, 4
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
    sample = {key: torch.zeros(count, height, width, dtype=torch.float32) for key, count in channels.items()}
    sample["meta"] = {"case_id": case_id}
    sample["unneeded_large_tensor"] = torch.zeros(100, height, width)
    return sample


class FineCascadeRunnerTests(unittest.TestCase):
    def test_launcher_admits_only_frozen_fine_configs_and_preflight(self) -> None:
        path = Path("scripts/run_direct_dynamics_cascade_fine_mechanics.sh")
        launcher = path.read_text(encoding="utf-8")
        self.assertTrue(os.access(path, os.X_OK))
        self.assertIn("FINE_CASCADE_CONFIG", launcher)
        self.assertIn("train_direct_dynamics_cascade_fine_compact_v2.json", launcher)
        self.assertIn("train_direct_dynamics_cascade_fine_compact_2048_v3.json", launcher)
        self.assertIn("train_direct_dynamics_cascade_fine_full_epoch_v4.json", launcher)
        self.assertIn(
            "train_direct_dynamics_cascade_fine_variance_preconditioned_512_v6.json",
            launcher,
        )
        self.assertIn(
            "train_direct_dynamics_cascade_fine_colored_preconditioned_512_v7.json",
            launcher,
        )
        self.assertIn("scripts/require_single_gpu_uuid.sh", launcher)
        self.assertIn(
            "nvidia-smi --id=0 --query-gpu=memory.used,utilization.gpu", launcher
        )
        self.assertNotIn("head -n 1", launcher)
        self.assertNotIn("--query-gpu=count", launcher)
        self.assertIn('export CUDA_VISIBLE_DEVICES="$GPU_UUID"', launcher)
        self.assertIn("timeout --foreground --signal=TERM --kill-after=2m", launcher)
        self.assertIn('PYTHON_MODE=("--preflight-only")', launcher)
        self.assertIn('"${PYTHON_MODE[@]}"', launcher)
        self.assertIn("usage: $0 [--preflight-only]", launcher)
        self.assertIn('FINE_CASCADE_STATUS_PATH="$STATUS_DIR/status.json"', launcher)
        self.assertIn('> "$STATUS_DIR/exit.json"', launcher)

    def test_preflight_and_failure_lifecycle_are_explicit(self) -> None:
        source = Path("assim_lib/direct_dynamics_cascade_fine_training.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('result["executed_optimizer_updates"] = 0', source)
        self.assertIn('lifecycle.phase = "preflight_complete"', source)
        self.assertIn('"complete_pending_generated_coarse_e2e_and_independent_review"', source)
        self.assertIn("signal.signal(signal.SIGTERM, _terminate)", source)

    def test_failure_cleanup_preserves_primary_error_without_trainer(self) -> None:
        lifecycle = _Lifecycle(run_id="fine-test", code_commit="a" * 40)
        with patch("assim_lib.direct_dynamics_cascade_fine_training._launch_status") as status:
            _record_failure(lifecycle, RuntimeError("primary"))
        payload = status.call_args.kwargs
        self.assertEqual(payload["status"], "failed")
        self.assertEqual(payload["error"], "primary")
        self.assertEqual(payload["cleanup_errors"], [])

    def test_partial_trainer_is_visible_when_constructor_fails(self) -> None:
        class PartialTrainer:
            def __init__(self, **_kwargs):
                self.clearml = SimpleNamespace(name="already-created")
                raise RuntimeError("constructor failed after ClearML")

        lifecycle = _Lifecycle()
        with (
            patch(
                "assim_lib.direct_dynamics_cascade_fine_training.FineCascadeDynamicsTrainer",
                PartialTrainer,
            ),
            self.assertRaisesRegex(RuntimeError, "constructor failed after ClearML"),
        ):
            _initialize_trainer(lifecycle)
        self.assertEqual(lifecycle.trainer.clearml.name, "already-created")

    def test_preflight_never_constructs_optimizer_or_trainer(self) -> None:
        class Dataset:
            calendar_pairs = [(None, SimpleNamespace(date=date(2020, 1, 1)))]

            def __init__(self, length):
                self.length = length

            def __len__(self):
                return self.length

            def __getitem__(self, _index):
                return _sample("case")

        loader = [None] * 512
        sentinel = {"status": "valid"}
        smoke = {"status": "passed", "loss": 1.0}
        datasets = {
            "train": Dataset(51_792),
            "valid": Dataset(8_544),
        }
        with (
            patch.dict(os.environ, {"FINE_CASCADE_LAUNCH_ID": "preflight-unit"}, clear=False),
            patch("assim_lib.direct_dynamics_cascade_fine_training._clean_code_identity", return_value={"git_commit": "a" * 40, "git_worktree": "clean"}),
            patch(
                "assim_lib.direct_dynamics_cascade_fine_training.build_dataset",
                side_effect=lambda _config, split: datasets[split],
            ),
            patch("assim_lib.direct_dynamics_cascade_fine_training.validate_direct_dataset", return_value=sentinel),
            patch("assim_lib.direct_dynamics_cascade_fine_training._calendar_inventory", return_value={"ordered_inventory_sha256": "b" * 64}),
            patch("assim_lib.direct_dynamics_cascade_fine_training._case_id", return_value="case"),
            patch("assim_lib.direct_dynamics_cascade_fine_training._ipc_preflight", return_value={"status": "passed"}),
            patch("assim_lib.direct_dynamics_cascade_fine_training.build_dataloader", return_value=loader),
            patch("assim_lib.direct_dynamics_cascade_fine_training.build_unet", return_value=torch.nn.Identity()),
            patch("assim_lib.direct_dynamics_cascade_fine_training._gpu_admission_smoke", return_value=smoke),
            patch("assim_lib.direct_dynamics_cascade_fine_training._write_preflight_evidence"),
            patch("assim_lib.direct_dynamics_cascade_fine_training._launch_status"),
            patch("torch.cuda.device_count", return_value=1),
            patch("torch.optim.AdamW") as optimizer,
            patch("assim_lib.direct_dynamics_cascade_fine_training.FineCascadeDynamicsTrainer") as trainer,
        ):
            result = run(
                Path("config/experiments/train_direct_dynamics_cascade_fine_compact_v2.json"),
                preflight_only=True,
            )
        self.assertEqual(result["status"], "preflight_passed")
        self.assertEqual(result["executed_optimizer_updates"], 0)
        optimizer.assert_not_called()
        trainer.assert_not_called()

    def test_compact_preflight_launcher_passes_exact_typed_arguments(self) -> None:
        source = Path("scripts/run_direct_dynamics_cascade_fine_mechanics.sh").read_text()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            result_root = root / "results"
            shared_launches = root / "shared_launches"
            shared_launches.mkdir()
            captured = root / "captured_args"
            script = source.replace(
                'RESULT_ROOT="/home/autoresearch_results/direct_dynamics_cascade_v2"',
                f'RESULT_ROOT="{result_root}"',
            ).replace(
                'GPU_LOCK_ROOT="/home/autoresearch_results/direct_dynamics_all_hours_v1/launches"',
                f'GPU_LOCK_ROOT="{shared_launches}"',
            )
            scripts = root / "scripts"
            scripts.mkdir()
            helper = scripts / "require_single_gpu_uuid.sh"
            helper.write_bytes(Path("scripts/require_single_gpu_uuid.sh").read_bytes())
            helper.chmod(0o700)
            launcher = scripts / "launcher.sh"
            launcher.write_text(script)
            launcher.chmod(0o700)
            commands = {
                "git": "#!/usr/bin/env bash\nprintf '%040d\\n' 0\n",
                "flock": "#!/usr/bin/env bash\nexit 0\n",
                "nvidia-smi": """#!/usr/bin/env bash
if [[ "$*" == *"query-gpu=uuid"* ]]; then
  echo GPU-01234567-89ab-cdef-0123-456789abcdef
else
  echo 0,0
fi
""",
                "timeout": "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" > \"$ARGS_CAPTURE\"\nexit 0\n",
            }
            for name, body in commands.items():
                path = fake_bin / name
                path.write_text(body)
                path.chmod(0o700)
            environment = {
                **os.environ,
                "PATH": f"{fake_bin}:{os.environ['PATH']}",
                "ARGS_CAPTURE": str(captured),
                "FINE_CASCADE_LAUNCH_ID": "compact-preflight",
                "FINE_CASCADE_CONFIG": "config/experiments/train_direct_dynamics_cascade_fine_compact_v2.json",
            }
            completed = subprocess.run(
                ["bash", str(launcher), "--preflight-only"],
                cwd=root,
                env=environment,
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            arguments = captured.read_text().splitlines()
            self.assertEqual(
                arguments[-5:],
                [
                    "-m",
                    "assim_lib.direct_dynamics_cascade_fine_training",
                    "--config",
                    "config/experiments/train_direct_dynamics_cascade_fine_compact_v2.json",
                    "--preflight-only",
                ],
            )
            status = json.loads(
                (result_root / "launches" / "compact-preflight" / "status.json").read_text()
            )
            self.assertEqual(status["status"], "running")
            self.assertTrue(
                (result_root / "launches" / "compact-preflight" / "exit.json").is_file()
            )

    def test_malformed_gpu_observation_never_reaches_python(self) -> None:
        source = Path("scripts/run_direct_dynamics_cascade_fine_mechanics.sh").read_text()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            result_root = root / "results"
            lock_root = root / "locks"
            marker = root / "python_called"
            script = source.replace(
                'RESULT_ROOT="/home/autoresearch_results/direct_dynamics_cascade_v2"',
                f'RESULT_ROOT="{result_root}"',
            ).replace(
                'GPU_LOCK_ROOT="/home/autoresearch_results/direct_dynamics_all_hours_v1/launches"',
                f'GPU_LOCK_ROOT="{lock_root}"',
            )
            scripts = root / "scripts"
            scripts.mkdir()
            helper = scripts / "require_single_gpu_uuid.sh"
            helper.write_bytes(Path("scripts/require_single_gpu_uuid.sh").read_bytes())
            helper.chmod(0o700)
            launcher = scripts / "launcher.sh"
            launcher.write_text(script)
            launcher.chmod(0o700)
            commands = {
                "git": "#!/usr/bin/env bash\nprintf '%040d\\n' 0\n",
                "flock": "#!/usr/bin/env bash\nexit 0\n",
                "nvidia-smi": """#!/usr/bin/env bash
if [[ "$*" == *"query-gpu=uuid"* ]]; then
  echo GPU-01234567-89ab-cdef-0123-456789abcdef
else
  echo malformed
fi
""",
                "timeout": f"#!/usr/bin/env bash\ntouch '{marker}'\n",
            }
            for name, body in commands.items():
                path = fake_bin / name
                path.write_text(body)
                path.chmod(0o700)
            completed = subprocess.run(
                ["bash", str(launcher), "--preflight-only"],
                cwd=root,
                env={
                    **os.environ,
                    "PATH": f"{fake_bin}:{os.environ['PATH']}",
                    "FINE_CASCADE_LAUNCH_ID": "malformed-gpu",
                    "FINE_CASCADE_CONFIG": "config/experiments/train_direct_dynamics_cascade_fine_variance_preconditioned_512_v6.json",
                },
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("malformed observation", completed.stderr)
            self.assertFalse(marker.exists())

    def test_forbidden_fine_config_never_reaches_python(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "python_called"
            fake_bin = root / "bin"
            fake_bin.mkdir()
            timeout = fake_bin / "timeout"
            timeout.write_text(f"#!/usr/bin/env bash\ntouch '{marker}'\n")
            timeout.chmod(0o700)
            completed = subprocess.run(
                ["bash", str(Path.cwd() / "scripts/run_direct_dynamics_cascade_fine_mechanics.sh")],
                env={
                    **os.environ,
                    "PATH": f"{fake_bin}:{os.environ['PATH']}",
                    "FINE_CASCADE_LAUNCH_ID": "forbidden",
                    "FINE_CASCADE_CONFIG": "config/experiments/not-reviewed.json",
                },
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("unsafe FINE_CASCADE_CONFIG", completed.stderr)
            self.assertFalse(marker.exists())

    def test_clean_identity_strips_git_environment_poisoning(self) -> None:
        results = [
            SimpleNamespace(stdout="a" * 40 + "\n"),
            SimpleNamespace(stdout=""),
        ]
        with (
            patch.dict(
                os.environ,
                {
                    "GIT_DIR": "/tmp/attacker",
                    "GIT_WORK_TREE": "/tmp/attacker-tree",
                    "GIT_INDEX_FILE": "/tmp/attacker-index",
                    "GIT_COMMON_DIR": "/tmp/attacker-common",
                },
            ),
            patch("subprocess.run", side_effect=results) as run,
        ):
            identity = _clean_code_identity(Path("/trusted/repo"))
        self.assertEqual(identity["git_commit"], "a" * 40)
        for call in run.call_args_list:
            self.assertFalse(any(key.startswith("GIT_") for key in call.kwargs["env"]))
            self.assertEqual(call.kwargs["cwd"], Path("/trusted/repo"))
        self.assertIn("--untracked-files=all", run.call_args_list[1].args[0])

    def test_even_subset_is_unique_and_spans_exact_archive(self) -> None:
        indices = _evenly_spaced_indices(51_792, 4_096)
        self.assertEqual(len(indices), len(set(indices)))
        self.assertEqual((indices[0], indices[-1]), (0, 51_791))

    def test_compact_parent_collate_keeps_only_contract_and_case_ids(self) -> None:
        batch = _fine_collate([_sample("a"), _sample("b")])
        self.assertEqual(set(batch), {*FINE_BATCH_KEYS, "meta"})
        self.assertEqual(batch["meta"]["case_id"], ["a", "b"])
        self.assertEqual(batch["truth"].shape, (2, 6, 4, 4))

    def test_ipc_gate_accounts_for_simultaneous_persistent_queues(self) -> None:
        config = TrainingConfig(
            train_batch_size=8,
            eval_batch_size=4,
            num_workers_train=4,
            num_workers_val=2,
        )
        generous = SimpleNamespace(f_bavail=10_000_000, f_frsize=4096)
        with patch("os.statvfs", return_value=generous):
            result = _ipc_preflight([_sample("train")], [_sample("valid")], config)
        self.assertEqual(result["train_retained_batches"], 17)
        self.assertEqual(result["validation_retained_batches"], 9)
        self.assertEqual(result["prefetch_factor"], 4)

        tiny = SimpleNamespace(f_bavail=1, f_frsize=4096)
        with (
            patch("os.statvfs", return_value=tiny),
            self.assertRaisesRegex(RuntimeError, "shared-memory safety budget"),
        ):
            _ipc_preflight([_sample("train")], [_sample("valid")], config)


if __name__ == "__main__":
    unittest.main()
