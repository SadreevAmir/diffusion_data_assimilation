import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from assim_lib.direct_dynamics_cascade_mean_frozen_audit import (
    _selector_cases,
    _support_cases,
    _weighted_cases,
    run,
)


class FrozenMeanAuditTests(unittest.TestCase):
    def test_weighted_metrics_are_case_equal_and_keep_signed_bias(self) -> None:
        truth = torch.zeros((2, 1, 1, 2))
        estimate = torch.tensor([[[[1.0, -1.0]]], [[[3.0, 99.0]]]])
        fraction = torch.tensor([[[[1.0, 1.0]]], [[[1.0, 0.0]]]])
        scored = _weighted_cases(estimate, truth, fraction)
        self.assertAlmostEqual(scored["aggregate_rmse"], math.sqrt(5.0))
        self.assertAlmostEqual(scored["aggregate_mae"], 2.0)
        self.assertAlmostEqual(scored["aggregate_signed_bias"], 1.5)
        self.assertEqual(scored["case_signed_bias"], [0.0, 3.0])

    def test_support_keeps_true_global_maximum(self) -> None:
        values = torch.tensor([[[[-0.2, 1.1]]], [[[0.5, 3.0]]]])
        active = torch.ones((2, 1, 1, 2))
        scored = _support_cases(values, active, "sic")
        self.assertAlmostEqual(scored["global_max_excess"], 2.0)
        self.assertAlmostEqual(scored["pooled_violation_frequency"], 0.75)

    def test_empty_selector_is_null_not_zero(self) -> None:
        values = torch.ones((1, 1, 1, 2))
        selected = _selector_cases(
            values, values, torch.zeros_like(values, dtype=torch.bool), torch.ones_like(values)
        )
        self.assertEqual(selected["total_count"], 0)
        self.assertIsNone(selected["case_values"][0]["metrics"])

    def test_failure_is_gate_closed_even_if_tracker_cleanup_fails(self) -> None:
        class BrokenTask:
            def mark_failed(self, **kwargs):
                raise RuntimeError("mark cleanup")

        class BrokenTracker:
            task = BrokenTask()
            def close(self):
                raise RuntimeError("close cleanup")

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "audit"

            def fail(_config, target, lifecycle):
                target.mkdir()
                lifecycle["output"] = target
                lifecycle["tracker"] = BrokenTracker()
                raise ValueError("primary failure")

            with mock.patch(
                "assim_lib.direct_dynamics_cascade_mean_frozen_audit._run_impl",
                side_effect=fail,
            ):
                with self.assertRaisesRegex(ValueError, "primary failure"):
                    run(Path("unused"), output)
            failure = json.loads((output / "failure.json").read_text())
            self.assertFalse(failure["audit_integrity_pass"])
            self.assertFalse(failure["scientific_decision_permitted"])
            self.assertEqual(len(failure["cleanup_errors"]), 2)

    def test_contract_and_launcher_freeze_cpu_only_envelope(self) -> None:
        config = json.loads(Path(
            "config/experiments/audit_direct_dynamics_cascade_mean_frozen_v1.json"
        ).read_text())
        self.assertEqual(config["protocol"]["case_count"], 48)
        self.assertEqual(config["protocol"]["matched_updates"], [512, 1024, 1536, 2048])
        self.assertEqual(config["protocol"]["optimizer_steps"], 0)
        self.assertFalse(config["protocol"]["new_dates"])
        self.assertEqual(len(config["source_artifacts_sha256"]), 16)
        for digest in (*config["source_artifacts_sha256"].values(),
                       *config["source_files_sha256"].values()):
            self.assertEqual(len(digest), 64)
        path = Path("scripts/run_direct_dynamics_cascade_mean_frozen_audit.sh")
        source = path.read_text()
        self.assertTrue(os.access(path, os.X_OK))
        self.assertIn('export CUDA_VISIBLE_DEVICES=""', source)
        self.assertIn("--kill-after=30s 3570s", source)
        self.assertNotIn("nvidia-smi", source)


if __name__ == "__main__":
    unittest.main()
