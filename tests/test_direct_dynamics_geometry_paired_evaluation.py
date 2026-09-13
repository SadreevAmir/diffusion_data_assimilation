import contextlib
import json
import os
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np
import torch

from assim_lib.direct_dynamics_geometry_paired_evaluation import (
    SCHEMA_VERSION,
    _check_contract,
    _close_and_mark_complete,
    _coverage,
    _field_stats_tensor,
    _merge_status,
    _paired_bootstrap,
    _paired_bootstrap_rmse,
    _public_sample,
    _record_evidence_binding,
    _record_terminal_failure,
    _verify_optimizer_against_model,
)


class GeometryPairedEvaluationTest(unittest.TestCase):
    def test_contract_is_exact(self):
        experiment = json.loads(Path("config/experiments/evaluate_direct_dynamics_geometry_objective_ab_v1.json").read_text())
        protocol = experiment["protocol"]
        self.assertEqual(_check_contract(experiment), protocol)
        broken = dict(protocol, members=7)
        with self.assertRaisesRegex(ValueError, "members"):
            _check_contract({**experiment, "protocol": broken})
        with self.assertRaisesRegex(ValueError, "gate"):
            _check_contract({**experiment, "gate": {**experiment["gate"], "rank_tv_increase_max": 0.5}})

    def test_coverage_uses_finite_member_references(self):
        members = torch.arange(8, dtype=torch.float64).view(1, 8, 1, 1, 1)
        truth = torch.tensor([[[[3.5]]]], dtype=torch.float64)
        result = _coverage(members, truth, torch.ones_like(truth))
        self.assertEqual(result["inner_2_7"]["finite_m_reference"], 5 / 9)
        self.assertEqual(result["outer_1_8"]["finite_m_reference"], 7 / 9)
        self.assertEqual(result["inner_2_7"]["coverage"], 1.0)

        tied = torch.zeros((1, 8, 1, 1, 1), dtype=torch.float64)
        tied_result = _coverage(tied, torch.zeros((1, 1, 1, 1), dtype=torch.float64), torch.ones((1, 1, 1, 1)))
        self.assertAlmostEqual(tied_result["inner_2_7"]["coverage"], 5 / 9)
        self.assertAlmostEqual(tied_result["outer_1_8"]["coverage"], 7 / 9)

    def test_bootstrap_preserves_pairing_and_direction(self):
        indices = np.tile(np.arange(12), (100, 1))
        result = _paired_bootstrap([1.0] * 12, [2.0] * 12, indices)
        self.assertEqual(result["difference_ci95"], [-1.0, -1.0])
        self.assertEqual(result["ratio_ci95"], [0.5, 0.5])

    def test_rmse_bootstrap_reaggregates_case_mse(self):
        indices = np.tile(np.array([0, 1]), (100, 1))
        result = _paired_bootstrap_rmse([0.0, 2.0], [1.0, 1.0], indices)
        self.assertAlmostEqual(result["ratio"], 2**0.5)
        self.assertAlmostEqual(result["difference"], 2**0.5 - 1.0)

    def test_tuple_field_statistics_follow_production_path(self):
        result = _field_stats_tensor((0.2, 0.4))
        self.assertEqual(tuple(result.shape), (6,))
        self.assertTrue(torch.equal(result, torch.tensor([0.2, 0.4] * 3)))

    def test_public_sampler_enters_bf16_autocast(self):
        class Sampler:
            def sample_conditioned(self, **kwargs):
                return kwargs["initial_noise"]

        item = {key: torch.zeros((1, 2, 2)) for key in ("background", "obs_values", "obs_mask", "water_mask")}
        item["structured_conditioning"] = torch.zeros((17, 2, 2))
        item["valid_mask"] = torch.ones((1, 2, 2))
        with mock.patch("torch.autocast", return_value=contextlib.nullcontext()) as autocast:
            output = _public_sample(Sampler(), item, torch.ones((1, 6, 2, 2)), (2, 2), torch.device("cpu"))
        self.assertEqual(tuple(output.shape), (1, 6, 2, 2))
        autocast.assert_called_once_with(device_type="cuda", dtype=torch.bfloat16)

    def test_failure_is_durable_when_tracker_cleanup_also_fails(self):
        class Task:
            def mark_failed(self, **_kwargs):
                raise RuntimeError("tracking failed")

        class Tracker:
            task = Task()

            def close(self):
                raise RuntimeError("close failed")

        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "status.json"
            error = ValueError("primary")
            _record_terminal_failure(path, Tracker(), error)
            status = json.loads(path.read_text())
        self.assertEqual(status["status"], "failed_terminal_non_resumable")
        self.assertEqual(status["error"], "primary")

    def test_parity_evidence_survives_terminal_failure(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            status_path = root / "status.json"
            bindings_path = root / "bindings.json"
            step0 = root / "step0.pt"
            step0.write_bytes(b"before-parity")
            _merge_status(status_path, status="sampling", fixed_inputs_sha256="fixed")
            binding = _record_evidence_binding(bindings_path, "step0", step0)
            _merge_status(status_path, step0_parity_sha256=binding["sha256"])
            _record_terminal_failure(status_path, None, RuntimeError("parity mismatch"))
            status = json.loads(status_path.read_text())
        self.assertEqual(status["fixed_inputs_sha256"], "fixed")
        self.assertEqual(status["step0_parity_sha256"], binding["sha256"])
        self.assertEqual(status["status"], "failed_terminal_non_resumable")

    def test_adamw_state_requires_full_exact_step256_coverage(self):
        model = torch.nn.Linear(2, 1)
        states = {}
        identifiers = list(range(len(list(model.parameters()))))
        for identifier, parameter in zip(identifiers, model.parameters()):
            states[identifier] = {
                "step": torch.tensor(256.0),
                "exp_avg": torch.zeros_like(parameter),
                "exp_avg_sq": torch.ones_like(parameter),
            }
        payload = {"optimizer": {"param_groups": [{"params": identifiers, "lr": 1e-5, "weight_decay": 0.0}], "state": states}}
        _verify_optimizer_against_model(payload, model, "control")
        missing = {"optimizer": {"param_groups": payload["optimizer"]["param_groups"], "state": {0: states[0]}}}
        with self.assertRaisesRegex(ValueError, "coverage"):
            _verify_optimizer_against_model(missing, model, "control")
        states[0]["step"] = torch.tensor(256.5)
        with self.assertRaisesRegex(ValueError, "step"):
            _verify_optimizer_against_model(payload, model, "control")

    def test_close_failure_never_marks_result_complete(self):
        class Tracker:
            class Task:
                def mark_failed(self, **_kwargs):
                    pass
            task = Task()

            def close(self):
                raise RuntimeError("close failed")

        with TemporaryDirectory() as directory:
            root = Path(directory)
            result_path = root / "paired_evaluation.json"
            status_path = root / "status.json"
            result_path.write_text(json.dumps({"status": "scored_pending_visual_review"}))
            _merge_status(status_path, status="plotting")
            tracker = Tracker()
            try:
                _close_and_mark_complete(status_path, tracker, "task")
            except RuntimeError as error:
                _record_terminal_failure(status_path, tracker, error)
            result = json.loads(result_path.read_text())
            status = json.loads(status_path.read_text())
        self.assertEqual(result["status"], "scored_pending_visual_review")
        self.assertEqual(status["status"], "failed_terminal_non_resumable")

    def test_launcher_reuses_reviewed_admission_without_legacy_queries(self):
        source = Path("scripts/run_direct_dynamics_geometry_paired_evaluation.sh").read_text()
        self.assertIn("scripts/require_single_gpu_uuid.sh", source)
        self.assertIn("memory.used,utilization.gpu", source)
        self.assertIn("for sample in {1..11}", source)
        self.assertIn("export CUDA_VISIBLE_DEVICES=\"$GPU_UUID\"", source)
        self.assertIn("timeout --foreground", source)
        self.assertNotIn("query-gpu=count", source)
        self.assertNotIn("head -n", source)

    def test_malformed_gpu_observation_never_reaches_python(self):
        source = Path("scripts/run_direct_dynamics_geometry_paired_evaluation.sh").read_text()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            result_root = root / "results"
            launch_root = root / "launches"
            lock_root = root / "locks"
            lock_root.mkdir()
            marker = root / "python_called"
            script = source.replace(
                'RESULT_ROOT="/home/autoresearch_results/direct_dynamics_geometry_objective_v1/evaluations"',
                f'RESULT_ROOT="{result_root}"',
            ).replace(
                'LAUNCH_ROOT="/home/autoresearch_results/direct_dynamics_geometry_objective_v1/evaluation_launches"',
                f'LAUNCH_ROOT="{launch_root}"',
            ).replace(
                '/home/autoresearch_results/direct_dynamics_all_hours_v1/launches/.gpu_job.lock',
                str(lock_root / ".gpu_job.lock"),
            )
            scripts = root / "scripts"
            scripts.mkdir()
            helper = scripts / "require_single_gpu_uuid.sh"
            helper.write_bytes(Path("scripts/require_single_gpu_uuid.sh").read_bytes())
            helper.chmod(0o700)
            launcher = scripts / "launcher.sh"
            launcher.write_text(script)
            launcher.chmod(0o700)
            config = root / "config/experiments/evaluate_direct_dynamics_geometry_objective_ab_v1.json"
            config.parent.mkdir(parents=True)
            config.write_text("{}")
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
                ["bash", str(launcher)],
                cwd=root,
                env={
                    **os.environ,
                    "PATH": f"{fake_bin}:{os.environ['PATH']}",
                    "DIRECT_DYNAMICS_GEOMETRY_EVAL_ID": "malformed-gpu",
                },
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()
