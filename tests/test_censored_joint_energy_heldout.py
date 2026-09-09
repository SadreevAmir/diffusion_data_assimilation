import json
import tempfile
import unittest
from pathlib import Path

import torch

from assim_lib.censored_joint_energy_heldout import (
    DIAGNOSTIC_UPDATES,
    MAX_UPDATES,
    VALIDATION_INDICES,
    VALIDATION_MANIFEST,
    _RunBoundary,
    _case_fair_crps,
    _correlation_matrix,
    _enforce_train_envelope,
    _fractional_rank_histogram,
    _save_then_evaluate_diagnostic,
    evaluate_group,
)


class _FakeTracker:
    def __init__(self, *, fail_close: bool = False) -> None:
        self.closed = False
        self.connected = []
        self.fail_close = fail_close

    def connect(self, name, value) -> None:
        self.connected.append((name, value))

    def close(self) -> None:
        self.closed = True
        if self.fail_close:
            raise RuntimeError("ClearML final flush failed")


class CensoredJointEnergyHeldoutTests(unittest.TestCase):
    def test_frozen_budget_and_manifest(self) -> None:
        self.assertEqual(MAX_UPDATES, 1024)
        self.assertEqual(DIAGNOSTIC_UPDATES, (0, 256, 512, 1024))
        self.assertEqual(tuple(row[0] for row in VALIDATION_MANIFEST), VALIDATION_INDICES)
        self.assertEqual(len(VALIDATION_MANIFEST), 12)

    def test_wrong_train_count_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "full train envelope differs"):
            _enforce_train_envelope(51_791, 6_474)
        with self.assertRaisesRegex(ValueError, "full train envelope differs"):
            _enforce_train_envelope(51_792, 6_473)

    def test_post_tracker_setup_failure_is_recorded_and_tracker_closed(self) -> None:
        tracker = _FakeTracker()
        progress = {
            "attempted_update": 0,
            "completed_optimizer_steps": 0,
            "losses": [],
            "diagnostics": {},
            "checkpoints": {},
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, "iterator setup failed"):
                with _RunBoundary(tracker, output, progress):
                    raise RuntimeError("iterator setup failed")
            failure = json.loads((output / "failure.json").read_text())
        self.assertTrue(tracker.closed)
        self.assertEqual(failure["attempted_update"], 0)
        self.assertEqual(failure["completed_optimizer_steps"], 0)
        self.assertEqual(failure["losses"], [])

    def test_successful_body_with_close_failure_is_not_reported_successful(self) -> None:
        tracker = _FakeTracker(fail_close=True)
        progress = {
            "attempted_update": 1024,
            "completed_optimizer_steps": 1024,
            "losses": [0.25],
            "diagnostics": {1024: {}},
            "checkpoints": {1024: {}},
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, "ClearML final flush failed"):
                with _RunBoundary(tracker, output, progress):
                    pass
            failure = json.loads((output / "failure.json").read_text())
        self.assertTrue(tracker.closed)
        self.assertEqual(failure["status"], "failed_finalization")
        self.assertEqual(failure["attempted_update"], 1024)
        self.assertEqual(failure["completed_optimizer_steps"], 1024)

    def test_evaluator_failure_keeps_raw_samples_and_identity(self) -> None:
        payload = {"physical_ensemble": torch.zeros(1, 8, 6, 2, 2)}
        identity = {
            "checkpoint": {"sha256": "checkpoint-sha"},
            "noise_sha256": "noise-sha",
            "group": "validation",
            "update": 0,
        }

        def fail(_payload):
            raise RuntimeError("metric evaluation failed")

        with tempfile.TemporaryDirectory() as temporary:
            stage = Path(temporary) / "validation"
            with self.assertRaisesRegex(RuntimeError, "metric evaluation failed"):
                _save_then_evaluate_diagnostic(
                    stage=stage,
                    payload=payload,
                    identity=identity,
                    evaluator=fail,
                )
            saved = torch.load(stage / "samples.pt", weights_only=False)
            diagnostic_failure = json.loads(
                (stage / "diagnostic_failure.json").read_text()
            )
        self.assertEqual(saved["checkpoint"]["sha256"], "checkpoint-sha")
        self.assertEqual(saved["noise_sha256"], "noise-sha")
        self.assertEqual(diagnostic_failure["error_type"], "RuntimeError")

    def test_fair_crps_two_member_example(self) -> None:
        members = torch.tensor([[0.0], [2.0]])
        truth = torch.tensor([1.0])
        self.assertAlmostEqual(_case_fair_crps(members, truth), 0.0)

    def test_fractional_rank_ties_are_exact(self) -> None:
        members = torch.tensor([[0.0], [0.0], [1.0]])
        truth = torch.tensor([0.0])
        histogram = _fractional_rank_histogram(members, truth)
        self.assertEqual(histogram, [1 / 3, 1 / 3, 1 / 3, 0.0])
        self.assertAlmostEqual(sum(histogram), 1.0)

    def test_zero_variance_correlation_is_null(self) -> None:
        result = _correlation_matrix(torch.zeros(8, 3))
        self.assertIsNone(result["correlation"][0][1])
        self.assertEqual(result["null_reasons"][0][1], "zero_member_variance")

    def test_group_metrics_use_case_equal_rmse(self) -> None:
        ensemble = torch.zeros(2, 8, 6, 12, 12)
        truth = torch.zeros(2, 6, 12, 12)
        truth[1, 1::2] = 2.0
        payload = {
            "physical_ensemble": ensemble,
            "truth": truth,
            "persistence": torch.zeros_like(truth),
            "valid_mask": torch.ones(2, 1, 12, 12),
        }
        result = evaluate_group(payload)
        self.assertAlmostEqual(result["outputs"]["d3_sit"]["rmse"], 2 ** 0.5)
        self.assertIsNone(result["outputs"]["d3_sic"]["fair_crps_ratio"])
        self.assertEqual(
            result["outputs"]["d3_sic"]["fair_crps_ratio_null_reason"],
            "zero_baseline_score",
        )


if __name__ == "__main__":
    unittest.main()
