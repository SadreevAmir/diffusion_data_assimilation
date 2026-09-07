from __future__ import annotations

import subprocess
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from assim_lib import structured_joint_full_launch_guard as guard


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/run_structured_joint_full_training_container.sh"
INNER = ROOT / "scripts/run_structured_joint_full_training_guarded.sh"
SMOKE_LAUNCHER = ROOT / "scripts/run_structured_joint_gpu_admission_container.sh"


class StructuredJointFullLauncherTests(unittest.TestCase):
    def _run_lifecycle_fixture(
        self,
        *,
        guard_failure: bool = False,
        create_failure: bool = False,
        cleanup_failure: bool = False,
    ) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
        bash_major = int(
            subprocess.run(
                ["bash", "-c", "printf %s \"${BASH_VERSINFO[0]}\""],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )
        if bash_major < 4:
            self.skipTest("behavioral launcher fixture requires server Bash >= 4")

        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        worktree = root / "worktree"
        results = root / "results"
        binary = root / "bin"
        (worktree / "assim_lib").mkdir(parents=True)
        results.mkdir()
        binary.mkdir()
        (worktree / "assim_lib" / "__init__.py").write_text("", encoding="utf-8")
        (worktree / "assim_lib" / "structured_joint_full_launch_guard.py").write_text(
            "import os\n"
            "if __name__ == '__main__' and os.environ.get('FAKE_GUARD_FAILURE') == '1':\n"
            "    raise SystemExit(81)\n",
            encoding="utf-8",
        )
        env_file = root / "clearml.env"
        env_file.write_text("placeholder=true\n", encoding="utf-8")
        smoke = root / "smoke.json"
        smoke.write_text("{}\n", encoding="utf-8")

        launcher_text = LAUNCHER.read_text(encoding="utf-8")
        launcher_text = launcher_text.replace(
            'WORKTREE="/home/a.madreev/diffusion_data_assimilation/autoresearch_worktrees/structured_joint_forecast_v2_20260907_0255"',
            f'WORKTREE="{worktree}"',
        ).replace(
            'HOST_RESULTS="/home/a.madreev/diffusion_data_assimilation/autoresearch_results"',
            f'HOST_RESULTS="{results}"',
        ).replace(
            'ENV_FILE="/home/a.madreev/diffusion_data_assimilation/.env"',
            f'ENV_FILE="{env_file}"',
        ).replace(
            'SMOKE_RESULT="${HOST_RESULTS}/structured_joint_gpu_admission_smoke/astra_ready_20260907_0710_retry6/result.json"',
            f'SMOKE_RESULT="{smoke}"',
        ).replace(
            'LOCK_PATH="${HOST_RESULTS}/structured_joint_gpu_admission_smoke/.${GPU_UUID}.host.lock"',
            f'LOCK_PATH="{root / "gpu.lock"}"',
        )
        launcher = root / "launcher.sh"
        launcher.write_text(launcher_text, encoding="utf-8")

        fake_docker = binary / "docker"
        fake_docker.write_text(
            "#!/usr/bin/env python3\n"
            "import json, os, sys\n"
            "from pathlib import Path\n"
            "args=sys.argv[1:]\n"
            "state=Path(os.environ['FAKE_DOCKER_STATE'])\n"
            "cid='a'*64\n"
            "def load(): return json.loads(state.read_text()) if state.exists() else None\n"
            "def save(v): state.write_text(json.dumps(v))\n"
            "if args[:2] == ['image','inspect']:\n"
            " print('sha256:6461c778a994df485f968da62fb9992a800f91ad58d455e0db20bd31d8ecc9e5')\n"
            "elif args[:2] == ['container','ls']:\n"
            " v=load(); print(v['id'] if v else '')\n"
            "elif args and args[0] == 'create':\n"
            " name=args[args.index('--name')+1]; user=args[args.index('--user')+1]\n"
            " cidfile=Path(args[args.index('--cidfile')+1]); cidfile.write_text(cid)\n"
            " save({'id':cid,'name':name,'user':user,'running':False}); print(cid)\n"
            " if os.environ.get('FAKE_CREATE_FAILURE') == '1': raise SystemExit(42)\n"
            "elif args and args[0] == 'inspect':\n"
            " v=load()\n"
            " if not v or args[-1] != v['id']: raise SystemExit(1)\n"
            " if '--format' not in args: print(json.dumps([v])); raise SystemExit(0)\n"
            " fmt=args[args.index('--format')+1]\n"
            " if '.Id}}|{{.Name}}' in fmt: print(v['id']+'|/'+v['name']+'|structured-joint-research|structured-joint-d0-d3-seed1701')\n"
            " elif '.State.Running' in fmt: print('true' if v['running'] else 'false')\n"
            " elif 'DeviceRequests' in fmt: print('GPU-40ff1cbb-07cc-992e-23cb-8fe181972b76')\n"
            " elif 'RestartPolicy' in fmt: print('no:0')\n"
            " elif 'ReadonlyRootfs' in fmt: print('true')\n"
            " elif '.Config.User' in fmt: print(v['user'])\n"
            " elif '.Mounts' in fmt: print('/workspace=false;/mnt=false;/home/.env=false;/run/structured_joint_admission_result.json=false;/run/structured_joint_gpu.lock=true;/home/autoresearch_results/structured_joint_d0_d3_real_lagged_31e=true;')\n"
            " else: raise SystemExit(98)\n"
            "elif args and args[0] == 'start':\n"
            " v=load(); v['running']=True; save(v)\n"
            "elif args and args[0] == 'logs':\n"
            " print('STRUCTURED_JOINT_FULL_GUARD_PASSED')\n"
            "elif args and args[0] == 'stop':\n"
            " v=load(); v['running']=False; save(v)\n"
            "elif args and args[0] == 'rm':\n"
            " if os.environ.get('FAKE_CLEANUP_FAILURE') == '1': raise SystemExit(43)\n"
            " state.unlink()\n"
            "else: raise SystemExit('unexpected docker invocation: '+repr(args))\n",
            encoding="utf-8",
        )
        fake_docker.chmod(0o755)
        state = root / "docker-state.json"
        completed = subprocess.run(
            ["bash", str(launcher)],
            check=False,
            capture_output=True,
            text=True,
            cwd=root,
            env={
                **os.environ,
                "PATH": f"{binary}:{os.environ['PATH']}",
                "FAKE_DOCKER_STATE": str(state),
                "FAKE_GUARD_FAILURE": "1" if guard_failure else "0",
                "FAKE_CREATE_FAILURE": "1" if create_failure else "0",
                "FAKE_CLEANUP_FAILURE": "1" if cleanup_failure else "0",
            },
        )
        return completed, state, results

    def test_shell_syntax_and_exact_single_gpu_contract(self) -> None:
        subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
        subprocess.run(["bash", "-n", str(INNER)], check=True)
        source = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn(guard.EXPECTED_GPU_UUID, source)
        self.assertIn(
            "sha256:6461c778a994df485f968da62fb9992a800f91ad58d455e0db20bd31d8ecc9e5",
            source,
        )
        self.assertIn('--gpus "device=${GPU_UUID}"', source)
        self.assertIn("--restart no", source)
        self.assertNotIn("on-failure", source)
        self.assertIn(
            'LOCK_PATH="${HOST_RESULTS}/structured_joint_gpu_admission_smoke/.${GPU_UUID}.host.lock"',
            source,
        )
        smoke_source = SMOKE_LAUNCHER.read_text(encoding="utf-8")
        self.assertIn('HOST_LOCK="${HOST_OUTPUT_ROOT}/.${GPU_UUID}.host.lock"', smoke_source)
        self.assertIn(
            "structured_joint_gpu_admission_smoke/.${GPU_UUID}.host.lock", source
        )
        self.assertIn('--volume "${LOCK_PATH}:/run/structured_joint_gpu.lock:rw"', source)
        self.assertIn("flock -u 9", source)
        self.assertIn("flock -w 120 9", INNER.read_text(encoding="utf-8"))

    def test_read_only_isolation_and_minimal_writable_mounts(self) -> None:
        source = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("--read-only", source)
        self.assertIn("--init", source)
        self.assertIn("--shm-size 32g", source)
        self.assertIn('--volume "${WORKTREE}:/workspace:ro"', source)
        self.assertIn("--volume /mnt:/mnt:ro", source)
        self.assertIn(
            '--volume "${OUTPUT_DIR}:${CONTAINER_RESULTS}/structured_joint_d0_d3_real_lagged_31e:rw"',
            source,
        )
        self.assertIn('--volume "${ENV_FILE}:/home/.env:ro"', source)
        self.assertIn(
            '--volume "${SMOKE_RESULT}:/run/structured_joint_admission_result.json:ro"',
            source,
        )
        self.assertIn("--tmpfs /tmp:rw,exec,nosuid", source)
        self.assertIn("--security-opt no-new-privileges", source)
        self.assertIn('--user "${host_uid}:${host_gid}"', source)
        self.assertNotIn('--volume "${HOST_RESULTS}:${CONTAINER_RESULTS}:rw"', source)

    def test_lifecycle_is_bounded_durable_and_fail_closed(self) -> None:
        source = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("timeout --signal=TERM --kill-after=5s", source)
        self.assertIn('--cidfile "${cidfile}"', source)
        self.assertIn("structured_joint_full_launch_record_v1", source)
        self.assertIn(
            'python3 "${WORKTREE}/assim_lib/structured_joint_full_launch_guard.py"', source
        )
        self.assertNotIn("PYTHONPATH=", source)
        self.assertIn("inspect_owned", source)
        self.assertIn("stop_and_remove_uncommitted", source)
        self.assertIn("cleanup_failed", source)
        self.assertIn("STRUCTURED_JOINT_FULL_GUARD_PASSED", source)
        self.assertNotIn("docker kill", source)
        self.assertNotIn("docker rm -f", source)

    def test_behavioral_success_leaves_exact_owned_container_running(self) -> None:
        completed, state, results = self._run_lifecycle_fixture()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertTrue(json.loads(state.read_text(encoding="utf-8"))["running"])
        records = list(results.glob("structured_joint_d0_d3_real_lagged_31e/launches/*/launch_record.json"))
        self.assertEqual(len(records), 1)
        self.assertEqual(json.loads(records[0].read_text(encoding="utf-8"))["status"], "running")

    def test_behavioral_create_failure_cleans_only_exact_owned_container(self) -> None:
        completed, state, results = self._run_lifecycle_fixture(create_failure=True)
        self.assertEqual(completed.returncode, 6, completed.stderr)
        self.assertFalse(state.exists())
        record = next(results.glob("structured_joint_d0_d3_real_lagged_31e/launches/*/launch_record.json"))
        self.assertEqual(
            json.loads(record.read_text(encoding="utf-8"))["status"],
            "cleaned_after_failed_launch",
        )

    def test_behavioral_cleanup_failure_is_visible_and_nonzero(self) -> None:
        completed, state, results = self._run_lifecycle_fixture(
            create_failure=True, cleanup_failure=True
        )
        self.assertEqual(completed.returncode, 91, completed.stderr)
        self.assertTrue(state.exists())
        record = next(results.glob("structured_joint_d0_d3_real_lagged_31e/launches/*/launch_record.json"))
        self.assertEqual(
            json.loads(record.read_text(encoding="utf-8"))["status"], "cleanup_failed"
        )

    def test_behavioral_identity_guard_failure_precedes_docker_create(self) -> None:
        completed, state, _ = self._run_lifecycle_fixture(guard_failure=True)
        self.assertEqual(completed.returncode, 81, completed.stderr)
        self.assertFalse(state.exists())

    def test_identity_guard_binds_complete_admission_chain(self) -> None:
        self.assertEqual(
            guard.SNAPSHOT_SHA256,
            "9eff2a34ad52bff934f0979fb03e3794480d9754eada91d9cc7c645a3641dd73",
        )
        self.assertEqual(
            guard.EXTENSION_SHA256,
            "e409bd39b0fb059aa2795fcff2c2bb5c17c149f191ec59ee5be853db2f2a1605",
        )
        self.assertEqual(
            guard.PROTOCOL_SHA256,
            "e5e7b8c0e372631bf47a7a2113823ebee0537fab27406ff577ce57ab6f501179",
        )
        self.assertEqual(
            guard.SMOKE_RESULT_SHA256,
            "3ee4d71fdf9ab04a3f04cd03bb304a59bdd9f6a3a09363bb78f38b8b8a2bab11",
        )
        self.assertTrue(
            {"assim_lib/model_io.py", "assim_lib/forecast.py", "assim_lib/transforms.py"}
            <= guard.REQUIRED_ADMITTED_CODE
        )
        inner = INNER.read_text(encoding="utf-8")
        self.assertIn("structured_joint_full_launch_guard", inner)
        self.assertIn("--check-gpu", inner)

    @patch("assim_lib.structured_joint_full_launch_guard.subprocess.run")
    def test_gpu_guard_rejects_any_preexisting_compute_process(self, run: Mock) -> None:
        run.side_effect = [
            subprocess.CompletedProcess(
                [], 0, f"{guard.EXPECTED_GPU_UUID}, 0, 0\n", ""
            ),
            subprocess.CompletedProcess(
                [], 0, f"{guard.EXPECTED_GPU_UUID}, 1234, 8\n", ""
            ),
        ]
        with self.assertRaisesRegex(RuntimeError, "compute process"):
            guard.validate_idle_gpu()

    @patch("assim_lib.structured_joint_full_launch_guard.subprocess.run")
    def test_gpu_guard_accepts_only_exact_idle_gpu(self, run: Mock) -> None:
        run.side_effect = [
            subprocess.CompletedProcess(
                [], 0, f"{guard.EXPECTED_GPU_UUID}, 0, 0\n", ""
            ),
            subprocess.CompletedProcess([], 0, "", ""),
        ]
        result = guard.validate_idle_gpu()
        self.assertEqual(result["uuid"], guard.EXPECTED_GPU_UUID)
        self.assertEqual(result["preexisting_compute_processes"], 0)


if __name__ == "__main__":
    unittest.main()
