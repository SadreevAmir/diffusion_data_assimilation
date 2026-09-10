import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from assim_lib.direct_dynamics_cascade_paired_evaluation import (
    SOURCE_LABELS,
    _joint_energy_score,
    _support_metrics,
    _weighted_case_fair_crps,
    _weighted_case_rmse,
    _weighted_case_rms,
    _weighted_fractional_rank,
    run,
)


class CoarsePairedEvaluationTests(unittest.TestCase):
    def test_rmse_is_case_equal_not_cell_pooled(self) -> None:
        truth = torch.zeros((2, 1, 1, 2))
        estimate = torch.tensor([[[[1.0, 1.0]]], [[[3.0, 99.0]]]])
        fraction = torch.tensor([[[[1.0, 1.0]]], [[[1.0, 0.0]]]])
        aggregate, cases = _weighted_case_rmse(estimate, truth, fraction)
        self.assertEqual(cases, [1.0, 3.0])
        self.assertAlmostEqual(aggregate, math.sqrt(5.0))

    def test_spread_aggregate_is_root_of_case_equal_variance(self) -> None:
        value = torch.tensor([[[[1.0, 1.0]]], [[[3.0, 99.0]]]])
        fraction = torch.tensor([[[[1.0, 1.0]]], [[[1.0, 0.0]]]])
        aggregate, cases = _weighted_case_rms(value, fraction)
        self.assertEqual(cases, [1.0, 3.0])
        self.assertAlmostEqual(aggregate, math.sqrt(5.0))

    def test_support_keeps_case_table_and_true_global_maximum(self) -> None:
        values = torch.tensor(
            [
                [[[[1.5]]], [[[1.0]]]],
                [[[[3.0]]], [[[1.0]]]],
            ]
        )
        result = _support_metrics(values, torch.ones((2, 1, 1, 1)), "sic")
        self.assertEqual(result["global_max_excess"], 2.0)
        self.assertEqual(result["case_values"]["max"], [0.5, 2.0])

    def test_failure_artifact_survives_broken_tracker_cleanup(self) -> None:
        class BrokenTask:
            def mark_failed(self, **kwargs) -> None:
                raise RuntimeError("mark cleanup")

        class BrokenTracker:
            task = BrokenTask()

            def close(self) -> None:
                raise RuntimeError("close cleanup")

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "failure-output"

            def fail(_config_path, target, lifecycle):
                target.mkdir()
                lifecycle["output"] = target
                lifecycle["tracker"] = BrokenTracker()
                raise ValueError("primary failure")

            with mock.patch(
                "assim_lib.direct_dynamics_cascade_paired_evaluation._run_impl",
                side_effect=fail,
            ):
                with self.assertRaisesRegex(ValueError, "primary failure"):
                    run(Path("unused.json"), output)
            failure = json.loads((output / "failure.json").read_text())
            self.assertEqual(failure["error"], "primary failure")
            self.assertEqual(
                failure["cleanup_errors"],
                [
                    "clearml_mark_failed: mark cleanup",
                    "clearml_close: close cleanup",
                ],
            )

    def test_fair_crps_uses_unbiased_pair_correction(self) -> None:
        members = torch.tensor([-1.0, 1.0]).reshape(1, 2, 1, 1, 1)
        truth = torch.zeros((1, 1, 1, 1))
        aggregate, cases = _weighted_case_fair_crps(
            members, truth, torch.ones((1, 1, 1, 1))
        )
        self.assertAlmostEqual(aggregate, 0.0)
        self.assertAlmostEqual(cases[0], 0.0)

    def test_fractional_rank_splits_exact_ties(self) -> None:
        members = torch.zeros((1, 2, 1, 1, 1))
        result = _weighted_fractional_rank(
            members, torch.zeros((1, 1, 1, 1)), torch.ones((1, 1, 1, 1))
        )
        self.assertTrue(
            torch.allclose(
                torch.tensor(result["case_equal_fractional_rank_frequencies"]),
                torch.full((3,), 1 / 3),
            )
        )
        self.assertAlmostEqual(result["rank_tv_to_uniform"], 0.0)
        self.assertAlmostEqual(result["normalized_mean_rank"], 0.5)

    def test_joint_energy_is_zero_for_exact_identical_members(self) -> None:
        truth = torch.arange(12, dtype=torch.float32).reshape(1, 6, 1, 2)
        ensemble = truth[:, None].repeat(1, 3, 1, 1, 1)
        aggregate, cases = _joint_energy_score(
            ensemble, truth, torch.ones((1, 1, 1, 2))
        )
        self.assertAlmostEqual(aggregate, 0.0)
        self.assertAlmostEqual(cases[0], 0.0)

    def test_config_and_launcher_freeze_the_review_contract(self) -> None:
        config = json.loads(
            Path(
                "config/experiments/evaluate_direct_dynamics_cascade_paired_v1.json"
            ).read_text()
        )
        self.assertEqual(config["members"], 8)
        self.assertEqual(config["rk4_steps"], 33)
        self.assertEqual(tuple(source["label"] for source in config["sources"]), SOURCE_LABELS)
        for source in config["sources"]:
            self.assertEqual(len(source["sha256"][source["checkpoint"]]), 64)
        path = Path("scripts/run_direct_dynamics_cascade_paired_evaluation.sh")
        source = path.read_text()
        self.assertTrue(os.access(path, os.X_OK))
        self.assertIn(".gpu_job.lock", source)
        self.assertIn("{1..11}", source)
        self.assertIn("sleep 30", source)
        self.assertIn('"$GPU_UTILIZATION" -ge 5', source)
        self.assertIn("--kill-after=30s 3570s", source)


if __name__ == "__main__":
    unittest.main()
