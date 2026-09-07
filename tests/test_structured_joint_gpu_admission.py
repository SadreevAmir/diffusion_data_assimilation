from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from assim_lib import structured_joint_gpu_admission as admission
from assim_lib.config import load_json


class StructuredJointGPUAdmissionTests(unittest.TestCase):
    def _run_host_launcher_simulation(
        self,
        root: Path,
        *,
        container_exit: int,
        watchdog_timeout: bool,
        docker_run_exit: int = 0,
        cleanup_fails: bool = False,
    ) -> tuple[subprocess.CompletedProcess[str], str, Path]:
        repo = Path(admission.__file__).resolve().parents[1]
        worktree = root / "server-worktree"
        output = root / "admission-output"
        binary_dir = root / "bin"
        worktree.mkdir()
        binary_dir.mkdir()

        launcher_text = (
            repo / "scripts/run_structured_joint_gpu_admission_container.sh"
        ).read_text(encoding="utf-8")
        launcher_text = launcher_text.replace(
            'SERVER_WORKTREE="/home/a.madreev/diffusion_data_assimilation/'
            'autoresearch_worktrees/structured_joint_forecast_v2_20260907_0255"',
            f'SERVER_WORKTREE="{worktree}"',
        ).replace(
            'HOST_OUTPUT_ROOT="/home/a.madreev/diffusion_data_assimilation/'
            'autoresearch_results/structured_joint_gpu_admission_smoke"',
            f'HOST_OUTPUT_ROOT="{output}"',
        )
        launcher = root / "host-launcher.sh"
        launcher.write_text(launcher_text, encoding="utf-8")
        launcher.chmod(0o755)

        fake_docker = binary_dir / "docker"
        fake_docker.write_text(
            "#!/usr/bin/env python3\n"
            "import json, os, sys\n"
            "from pathlib import Path\n"
            "args = sys.argv[1:]\n"
            "state = Path(os.environ['FAKE_DOCKER_STATE'])\n"
            "log = Path(os.environ['FAKE_DOCKER_LOG'])\n"
            "with log.open('a', encoding='utf-8') as stream:\n"
            "    stream.write(' '.join(args) + '\\n')\n"
            "cid = 'a' * 64\n"
            "if args[:2] == ['image', 'inspect']:\n"
            f"    print('{admission.EXPECTED_IMAGE_ID}')\n"
            "elif args and args[0] == 'run':\n"
            "    name = args[args.index('--name') + 1]\n"
            "    cidfile = Path(args[args.index('--cidfile') + 1])\n"
            "    state.write_text(json.dumps({'id': cid, 'name': name}), encoding='utf-8')\n"
            "    cidfile.write_text(cid + '\\n', encoding='utf-8')\n"
            "    print(cid)\n"
            "    raise SystemExit(int(os.environ['FAKE_DOCKER_RUN_EXIT']))\n"
            "elif args[:2] == ['container', 'inspect']:\n"
            "    if not state.exists():\n"
            "        raise SystemExit(1)\n"
            "    item = json.loads(state.read_text(encoding='utf-8'))\n"
            "    if '--format' in args:\n"
            "        print(item['id'] + ' /' + item['name'])\n"
            "elif args[:2] == ['container', 'ls']:\n"
            "    if state.exists():\n"
            "        print(json.loads(state.read_text(encoding='utf-8'))['id'])\n"
            "elif args and args[0] == 'wait':\n"
            "    print(os.environ['FAKE_CONTAINER_EXIT'])\n"
            "elif args and args[0] == 'stop':\n"
            "    if os.environ['FAKE_CLEANUP_FAIL'] == '1':\n"
            "        raise SystemExit(41)\n"
            "elif args and args[0] == 'rm':\n"
            "    if os.environ['FAKE_CLEANUP_FAIL'] == '1':\n"
            "        raise SystemExit(42)\n"
            "    if state.exists():\n"
            "        state.unlink()\n"
            "    print(cid)\n"
            "else:\n"
            "    raise SystemExit('unexpected fake docker invocation: ' + repr(args))\n",
            encoding="utf-8",
        )
        fake_docker.chmod(0o755)
        fake_timeout = binary_dir / "timeout"
        fake_timeout.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "while [[ $# -gt 0 ]]; do\n"
            "  case \"$1\" in\n"
            "    --signal=*|--kill-after=*) shift ;;\n"
            "    *s) shift; break ;;\n"
            "    *) exit 97 ;;\n"
            "  esac\n"
            "done\n"
            "if [[ \"${FAKE_WATCHDOG_TIMEOUT:-0}\" == 1 "
            "&& \"${1:-}\" == docker && \"${2:-}\" == wait ]]; then\n"
            "  exit 124\n"
            "fi\n"
            "exec \"$@\"\n",
            encoding="utf-8",
        )
        fake_timeout.chmod(0o755)
        fake_flock = binary_dir / "flock"
        fake_flock.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        fake_flock.chmod(0o755)

        state = root / "docker-state.json"
        log = root / "docker.log"
        completed = subprocess.run(
            [str(launcher), "mutable:test", "attempt_001"],
            check=False,
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "PATH": f"{binary_dir}:{os.environ['PATH']}",
                "FAKE_DOCKER_STATE": str(state),
                "FAKE_DOCKER_LOG": str(log),
                "FAKE_CONTAINER_EXIT": str(container_exit),
                "FAKE_WATCHDOG_TIMEOUT": "1" if watchdog_timeout else "0",
                "FAKE_DOCKER_RUN_EXIT": str(docker_run_exit),
                "FAKE_CLEANUP_FAIL": "1" if cleanup_fails else "0",
            },
        )
        return completed, log.read_text(encoding="utf-8"), state

    def test_historical_protocol_resolves_contract_and_rejects_source_drift(self) -> None:
        repo = Path(admission.__file__).resolve().parents[1]
        protocol_path = repo / admission.PROTOCOL_RELATIVE_PATH
        protocol = admission._load_protocol(protocol_path, repo)
        experiment = repo / protocol["paths"]["experiment"]
        paths = admission._resolve_identity_paths(protocol, repo, experiment)
        self.assertEqual(set(paths), set(admission.IDENTITY_NAMES))
        config = admission._validate_frozen_contract(
            protocol,
            load_json(paths["data_config"]),
            load_json(paths["method_config"]),
            load_json(paths["stats"]),
        )
        self.assertEqual(config.train_batch_size, 16)
        self.assertEqual(config.mixed_precision, "bf16")
        self.assertEqual(config.sample_method, "rk4")
        self.assertEqual(config.num_sample_timesteps, 33)
        with self.assertRaisesRegex(ValueError, "server-validated SHA differs"):
            admission._validate_server_snapshot_manifest(protocol, repo)

    def test_extension_manifest_has_no_protocol_hash_cycle(self) -> None:
        repo = Path(admission.__file__).resolve().parents[1]
        protocol = load_json(repo / admission.PROTOCOL_RELATIVE_PATH)
        required = set(protocol["smoke_extension_required_files"])
        self.assertNotIn(protocol["paths"]["protocol"], required)
        self.assertEqual(
            required,
            {
                "assim_lib/structured_joint_gpu_admission.py",
                "scripts/run_structured_joint_gpu_admission_container.sh",
                "scripts/run_structured_joint_gpu_admission_smoke.sh",
                "tests/test_structured_joint_gpu_admission.py",
            },
        )

    def test_server_snapshot_test_evidence_requires_equal_passing_counts(self) -> None:
        pattern = re.compile(r"(\d+)/\1 PASS(?: .*)?")
        self.assertIsNotNone(pattern.fullmatch("42/42 PASS (34 structured + 8 publication)"))
        self.assertIsNone(pattern.fullmatch("41/42 PASS"))

    def test_verified_extension_manifest_binds_exact_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            required = ["code.py", "runner.sh", "tests.py"]
            hashes = {}
            for relative in required:
                path = repo / relative
                path.write_text(relative, encoding="utf-8")
                hashes[relative] = admission._file_sha256(path)
            manifest = {
                "schema_version": admission.EXTENSION_SNAPSHOT_SCHEMA,
                "server_cpu_tests_status": "passed",
                "server_cpu_tests": {"command": "python -m unittest", "passed": 3},
                "files": hashes,
            }
            manifest_path = repo / "extension.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            protocol = {
                "smoke_extension_validation": {
                    "status": "server_cpu_verified",
                    "manifest_path": "extension.json",
                    "manifest_sha256": admission._file_sha256(manifest_path),
                },
                "smoke_extension_required_files": required,
            }
            result = admission._validate_smoke_extension_manifest(protocol, repo)
            self.assertEqual(result["verified_file_count"], 3)
            (repo / "code.py").write_text("drift", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "server-validated SHA differs"):
                admission._validate_smoke_extension_manifest(protocol, repo)

    def test_host_launcher_binds_network_gpu_image_and_writable_caches(self) -> None:
        repo = Path(admission.__file__).resolve().parents[1]
        launcher = (repo / "scripts/run_structured_joint_gpu_admission_container.sh").read_text()
        self.assertIn("--network none", launcher)
        self.assertIn('--user "${HOST_UID}:${HOST_GID}"', launcher)
        self.assertIn("--cpus 4", launcher)
        self.assertIn(
            'HOST_OUTPUT_ROOT="/home/a.madreev/diffusion_data_assimilation/'
            'autoresearch_results/structured_joint_gpu_admission_smoke"',
            launcher,
        )
        self.assertIn(
            'CONTAINER_OUTPUT_ROOT="/home/autoresearch_results/'
            'structured_joint_gpu_admission_smoke"',
            launcher,
        )
        self.assertIn(
            '--volume "${HOST_OUTPUT_ROOT}:${CONTAINER_OUTPUT_ROOT}:rw"', launcher
        )
        self.assertIn('"${CONTAINER_OUTPUT_ROOT}/${ATTEMPT_NAME}"', launcher)
        self.assertIn(admission.EXPECTED_GPU_UUID, launcher)
        self.assertIn(
            "sha256:6461c778a994df485f968da62fb9992a800f91ad58d455e0db20bd31d8ecc9e5",
            launcher,
        )
        self.assertIn("docker image inspect --format '{{.Id}}'", launcher)
        self.assertIn("DOCKER_CONTROL_SECONDS=30", launcher)
        self.assertIn("DOCKER_START_SECONDS=60", launcher)
        self.assertIn('if ! ACTUAL_IMAGE_ID="$(timeout --signal=TERM --kill-after=5s', launcher)
        self.assertIn('identity="$(timeout --signal=TERM --kill-after=5s', launcher)
        self.assertIn('remaining="$(timeout --signal=TERM --kill-after=5s', launcher)
        self.assertIn("--pull=never", launcher)
        self.assertIn("--entrypoint /bin/bash", launcher)
        self.assertIn('"${EXPECTED_IMAGE_ID}" \\', launcher)
        self.assertIn("scripts/run_structured_joint_gpu_admission_smoke.sh \\", launcher)
        self.assertNotIn("bash scripts/run_structured_joint_gpu_admission_smoke.sh", launcher)
        self.assertIn("docker wait", launcher)
        self.assertIn("--name \"${OWN_CONTAINER_NAME}\"", launcher)
        self.assertIn("docker stop --time \"${STOP_GRACE_SECONDS}\" \"${OWN_CONTAINER_ID}\"", launcher)
        self.assertIn("docker rm -f \"${OWN_CONTAINER_ID}\"", launcher)
        self.assertNotIn("docker kill", launcher)
        for variable in ("HOME", "XDG_CACHE_HOME", "MPLCONFIGDIR", "TORCH_HOME"):
            self.assertIn(f"--env {variable}=/tmp/", launcher)
        for variable in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            self.assertIn(f"--env {variable}=1", launcher)

    def test_heavy_gpu_imports_occur_only_after_cuda_admission(self) -> None:
        repo = Path(admission.__file__).resolve().parents[1]
        source = (repo / "assim_lib/structured_joint_gpu_admission.py").read_text()
        gate = source.index("device_identity = _validate_cuda_admission(protocol)")
        for import_line in (
            "from diffusers.training_utils import EMAModel",
            "from .model_io import build_unet",
            "from .trainer import UNetTrainer",
        ):
            self.assertGreater(source.index(import_line), gate)

    def test_host_launcher_preserves_normal_exit_and_removes_only_owned_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            completed, log, state = self._run_host_launcher_simulation(
                Path(directory), container_exit=17, watchdog_timeout=False
            )
            owned_id = "a" * 64
            self.assertEqual(completed.returncode, 17, completed.stderr)
            run_line = next(line for line in log.splitlines() if line.startswith("run "))
            self.assertIn("--pull=never", run_line)
            self.assertIn(admission.EXPECTED_IMAGE_ID, run_line)
            self.assertNotIn("mutable:test", run_line)
            self.assertIn(f"rm {owned_id}", log)
            self.assertNotIn("stop --time", log)
            self.assertFalse(state.exists())

    def test_host_watchdog_stops_and_removes_only_exact_owned_container(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            completed, log, state = self._run_host_launcher_simulation(
                Path(directory), container_exit=0, watchdog_timeout=True
            )
            owned_id = "a" * 64
            self.assertEqual(completed.returncode, 124, completed.stderr)
            self.assertIn(f"stop --time 20 {owned_id}", log)
            self.assertIn(f"rm -f {owned_id}", log)
            self.assertNotIn("kill", log)
            self.assertFalse(state.exists())

    def test_failed_conditional_cleanup_retains_owned_identity_and_returns_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            completed, log, state = self._run_host_launcher_simulation(
                root,
                container_exit=0,
                watchdog_timeout=False,
                docker_run_exit=42,
                cleanup_fails=True,
            )
            owned_id = "a" * 64
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("cleanup could not be confirmed", completed.stderr)
            self.assertTrue(state.exists())
            cidfiles = list((root / "admission-output").glob(".*.cid"))
            self.assertEqual(len(cidfiles), 1)
            self.assertEqual(cidfiles[0].read_text(encoding="utf-8").strip(), owned_id)
            self.assertIn(f"stop --time 20 {owned_id}", log)
            self.assertIn(f"rm -f {owned_id}", log)

    def test_host_launcher_rejects_wrong_resolved_image_id_before_run(self) -> None:
        repo = Path(admission.__file__).resolve().parents[1]
        launcher = repo / "scripts/run_structured_joint_gpu_admission_container.sh"
        with tempfile.TemporaryDirectory() as directory:
            binary_dir = Path(directory) / "bin"
            binary_dir.mkdir()
            fake_docker = binary_dir / "docker"
            fake_docker.write_text(
                "#!/usr/bin/env bash\nprintf '%s\\n' 'sha256:wrong'\n",
                encoding="utf-8",
            )
            fake_docker.chmod(0o755)
            fake_timeout = binary_dir / "timeout"
            fake_timeout.write_text(
                "#!/usr/bin/env bash\n"
                "while [[ $# -gt 0 ]]; do\n"
                "  case \"$1\" in\n"
                "    --signal=*|--kill-after=*) shift ;;\n"
                "    *s) shift; break ;;\n"
                "    *) exit 97 ;;\n"
                "  esac\n"
                "done\n"
                "exec \"$@\"\n",
                encoding="utf-8",
            )
            fake_timeout.chmod(0o755)
            completed = subprocess.run(
                [str(launcher), "local:test", "attempt_001"],
                check=False,
                capture_output=True,
                text=True,
                env={**os.environ, "PATH": f"{binary_dir}:{os.environ['PATH']}"},
            )
        self.assertEqual(completed.returncode, 65)
        self.assertIn("container image ID differs", completed.stderr)

    def test_inner_runner_exports_offline_environment_and_exact_cli(self) -> None:
        repo = Path(admission.__file__).resolve().parents[1]
        launcher = repo / "scripts/run_structured_joint_gpu_admission_smoke.sh"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary_dir = root / "bin"
            binary_dir.mkdir()
            capture = root / "capture.txt"
            fake_python = binary_dir / "python3"
            fake_python.write_text(
                "#!/usr/bin/env bash\n"
                "printf '%s|%s|%s|%s|%s\\n' \"$CLEARML_OFFLINE_MODE\" "
                "\"$HF_HUB_OFFLINE\" \"$TRANSFORMERS_OFFLINE\" "
                "\"$WANDB_MODE\" \"$WANDB_DISABLED\" > \"$CAPTURE\"\n"
                "printf '%s\\n' \"$*\" >> \"$CAPTURE\"\n",
                encoding="utf-8",
            )
            fake_python.chmod(0o755)
            completed = subprocess.run(
                [str(launcher), "/tmp/new_admission_output"],
                check=False,
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "PATH": f"{binary_dir}:{os.environ['PATH']}",
                    "CAPTURE": str(capture),
                },
            )
            captured = capture.read_text(encoding="utf-8")
        self.assertEqual(completed.returncode, 0)
        self.assertIn("1|1|1|offline|true", captured)
        self.assertIn("-m assim_lib.structured_joint_gpu_admission", captured)
        self.assertIn("--output-dir /tmp/new_admission_output", captured)

    def test_contract_drift_fails_before_model_construction(self) -> None:
        repo = Path(admission.__file__).resolve().parents[1]
        protocol = load_json(repo / admission.PROTOCOL_RELATIVE_PATH)
        data = load_json(repo / protocol["paths"]["data_config"])
        method = load_json(repo / protocol["paths"]["method_config"])
        stats = load_json(repo / protocol["paths"]["stats"])
        method["sample_method"] = "euler"
        with self.assertRaisesRegex(ValueError, "contract differs"):
            admission._validate_frozen_contract(protocol, data, method, stats)

    def test_identity_snapshot_detects_content_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text('{"value": 1}\n', encoding="utf-8")
            snapshot = admission._take_identity_snapshot({"config": path})
            admission._assert_identity_snapshot(snapshot)
            path.write_text('{"value": 2}\n', encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "changed during admission"):
                admission._assert_identity_snapshot(snapshot)

    def test_output_and_lock_are_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "admission"
            output = root / "attempt_001"
            created = admission._prepare_output_directory(output, root)
            self.assertEqual(created, output)
            with self.assertRaises(FileExistsError):
                admission._prepare_output_directory(output, root)
            with self.assertRaisesRegex(ValueError, "direct child"):
                admission._prepare_output_directory(Path(directory) / "other", root)

            lock = Path(directory) / "admission.lock"
            with admission._exclusive_lock(lock):
                with self.assertRaisesRegex(RuntimeError, "already held"):
                    with admission._exclusive_lock(lock):
                        self.fail("a second admission acquired the same lock")

    def test_atomic_json_never_leaves_temporary_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / admission.RESULT_FILENAME
            admission._atomic_json(path, {"status": "passed"})
            self.assertEqual(json.loads(path.read_text()), {"status": "passed"})
            self.assertEqual(list(Path(directory).glob(".*.tmp")), [])

    def test_offline_environment_is_mandatory(self) -> None:
        required = {"CLEARML_OFFLINE_MODE": "1", "HF_HUB_OFFLINE": "1"}
        admission._validate_offline_environment(required, dict(required))
        with self.assertRaisesRegex(RuntimeError, "offline environment"):
            admission._validate_offline_environment(
                required, {"CLEARML_OFFLINE_MODE": "1"}
            )

    def test_nvidia_parser_and_cuda_gate_require_exact_single_idle_uuid(self) -> None:
        expected_uuid = "GPU-40ff1cbb-07cc-992e-23cb-8fe181972b76"
        protocol = {
            "expected_gpu_uuid": expected_uuid,
            "resource_gates": {
                "visible_cuda_device_count": 1,
                "max_preexisting_memory_mib": 512,
                "max_preexisting_utilization_percent": 5,
            },
        }
        properties = SimpleNamespace(
            uuid=expected_uuid,
            name="test GPU",
            major=8,
            minor=0,
            total_memory=32 * 1024**3,
        )
        with (
            patch.object(
                admission,
                "_run_nvidia_smi",
                return_value=f"{expected_uuid}, 7, 12, 0\n",
            ),
            patch.object(admission, "_run_nvidia_compute_apps", return_value=""),
            patch.object(admission.torch.cuda, "is_available", return_value=True),
            patch.object(admission.torch.cuda, "device_count", return_value=1),
            patch.object(admission.torch.cuda, "set_device"),
            patch.object(admission.torch.cuda, "current_device", return_value=0),
            patch.object(admission.torch.cuda, "is_bf16_supported", return_value=True),
            patch.object(
                admission.torch.cuda, "get_device_properties", return_value=properties
            ),
        ):
            result = admission._validate_cuda_admission(protocol)
        self.assertEqual(result["logical_device"], "cuda:0")
        self.assertEqual(result["uuid"], expected_uuid)

        class TorchCUuuidLike:
            def __str__(self) -> str:
                return expected_uuid.removeprefix("GPU-")

        properties.uuid = TorchCUuuidLike()
        with (
            patch.object(
                admission,
                "_run_nvidia_smi",
                return_value=f"{expected_uuid}, 7, 12, 0\n",
            ),
            patch.object(admission, "_run_nvidia_compute_apps", return_value=""),
            patch.object(admission.torch.cuda, "is_available", return_value=True),
            patch.object(admission.torch.cuda, "device_count", return_value=1),
            patch.object(admission.torch.cuda, "set_device"),
            patch.object(admission.torch.cuda, "current_device", return_value=0),
            patch.object(admission.torch.cuda, "is_bf16_supported", return_value=True),
            patch.object(
                admission.torch.cuda, "get_device_properties", return_value=properties
            ),
        ):
            result = admission._validate_cuda_admission(protocol)
        self.assertEqual(result["torch_uuid"], expected_uuid)

        with patch.object(
            admission,
            "_run_nvidia_smi",
            return_value=(
                f"{expected_uuid}, 0, 0, 0\n"
                "GPU-00000000-0000-0000-0000-000000000000, 1, 0, 0\n"
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "exactly one"):
                admission._validate_cuda_admission(protocol)

        with (
            patch.object(
                admission,
                "_run_nvidia_smi",
                return_value=f"{expected_uuid}, 0, 600, 0\n",
            ),
            self.assertRaisesRegex(RuntimeError, "already uses"),
        ):
            admission._validate_cuda_admission(protocol)

    def test_nfe_counter_has_a_hard_upper_bound(self) -> None:
        class Identity(torch.nn.Module):
            def forward(self, value):
                return value

        model = admission._NFECountingModel(Identity(), max_nfe=2)
        value = torch.ones(1)
        self.assertTrue(torch.equal(model(value), value))
        self.assertTrue(torch.equal(model(value), value))
        with self.assertRaisesRegex(RuntimeError, "max_nfe=2"):
            model(value)
        self.assertEqual(model.nfe, 3)

    def test_fixed_rk4_nfe_contract_is_exact(self) -> None:
        self.assertEqual(admission._expected_fixed_nfe("rk4", 33), 128)
        with self.assertRaisesRegex(ValueError, "fixed-step"):
            admission._expected_fixed_nfe("dopri5", 33)

    def test_actual_model_capacity_is_separated_from_controlled_decode(self) -> None:
        repo = Path(admission.__file__).resolve().parents[1]
        source = (repo / "assim_lib/structured_joint_gpu_admission.py").read_text()
        latent_call = source.index("latent_sample = sampler.sample_conditioned")
        nfe_gate = source.index("if counting_model.nfe != expected_nfe")
        decode_attempt = source.index("actual_physical = decode_structured_joint_trajectory")
        controlled = source.index("controlled_sample = Sampler(")
        self.assertLess(latent_call, nfe_gate)
        self.assertLess(nfe_gate, decode_attempt)
        self.assertLess(decode_attempt, controlled)
        self.assertIn("not_applicable_near_untrained_decode_saturation", source)

    def test_truth_free_exact_d0_and_joint_support_gate(self) -> None:
        sample = torch.zeros((1, 8, 2, 3), dtype=torch.float32)
        sample[:, 0, 0, 0] = 0.5
        sample[:, 1, 0, 0] = 1.0
        valid = torch.ones((1, 1, 2, 3), dtype=torch.float32)
        valid[..., 1, 2] = 0.0
        lag0_mask = torch.zeros_like(valid)
        lag0_mask[..., 0, 0] = 1.0
        exact = torch.zeros((1, 2, 2, 3), dtype=torch.float32)
        exact[:, 0, 0, 0] = 0.5
        exact[:, 1, 0, 0] = 1.0
        item = {
            "valid_mask": valid,
            "structured_lag0_mask": lag0_mask,
            "structured_lag0_physical_values": exact,
            "meta": {"target_trajectory_paths": []},
        }
        result = admission._validate_truth_free_sample(item, sample, sic_cap=0.75)
        self.assertTrue(result["truth_free"])
        self.assertTrue(result["lag0_exact_bitwise"])

        bad_support = sample.clone()
        bad_support[:, 0, 0, 1] = 0.25
        with self.assertRaisesRegex(ValueError, "joint SIC/SIT support"):
            admission._validate_truth_free_sample(item, bad_support, sic_cap=0.75)

        with self.assertRaisesRegex(ValueError, "not truth-free"):
            admission._validate_truth_free_sample(
                {**item, "truth": torch.zeros_like(sample)}, sample, sic_cap=0.75
            )


if __name__ == "__main__":
    unittest.main()
