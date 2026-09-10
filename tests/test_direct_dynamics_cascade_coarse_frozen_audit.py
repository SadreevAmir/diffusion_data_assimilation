import unittest

import torch
from torch import nn

from assim_lib.direct_dynamics_cascade_coarse_frozen_audit import (
    COARSE_SHIFTS,
    SELECTED_SOURCE_CASES,
    SOLVER_TIMEPOINTS,
    _channel_differences,
    _close_tracker_for_success,
    _decision,
    _phase_contrast,
    _phase_rms,
    _region_masks,
    _require_finite_scalars,
    _shift_audit,
    _Lifecycle,
)


class CoarseFrozenAuditTests(unittest.TestCase):
    def test_protocol_is_small_frozen_and_zero_optimizer_by_construction(self) -> None:
        self.assertEqual(SOLVER_TIMEPOINTS, (17, 33, 65))
        self.assertEqual(COARSE_SHIFTS, (1, 2, 4, 8))
        self.assertEqual(SELECTED_SOURCE_CASES, (0, 2))

    def test_channel_differences_do_not_cancel_across_cases(self) -> None:
        reference = torch.zeros((2, 6, 3, 3))
        candidate = reference.clone()
        candidate[0, 0] = 1
        candidate[1, 0] = -1
        metrics = _channel_differences(candidate, reference, torch.ones((2, 1, 3, 3)))
        self.assertAlmostEqual(metrics["aggregate"]["d3_sic"]["normalized_rmse"], 1.0)
        self.assertAlmostEqual(metrics["cases"]["case00"]["d3_sic"]["normalized_rmse"], 1.0)

    def test_region_masks_separate_interior_from_padding_control(self) -> None:
        active = torch.ones((1, 1, 40, 40))
        interior, padding = _region_masks(active, margin=4)
        self.assertFalse(torch.any(interior & padding))
        self.assertTrue(torch.equal(interior | padding, active.bool()))

    def test_phase_metric_detects_a_single_phase(self) -> None:
        error = torch.zeros((1, 6, 8, 8))
        error[..., 0::2, 0::2] = 2
        table = _phase_rms(error, torch.ones((1, 1, 8, 8)), 2)
        self.assertGreater(_phase_contrast(table), 1.0)

    def test_shift_audit_keeps_fine_packing_mask_separate_from_coarse_scoring(self) -> None:
        class PointwiseVelocity(nn.Module):
            def forward(self, value, timestep, return_dict=False):
                del timestep, return_dict
                return (value[:, :6],)

        fine_valid = torch.ones((1, 1, 48, 48))
        condition = torch.zeros((1, 15, 48, 48))
        condition[:, 2:3] = fine_valid
        state = torch.randn((1, 6, 24, 24))
        coarse_active = torch.ones((1, 1, 24, 24))
        audit, errors, maps = _shift_audit(
            PointwiseVelocity(),
            state=state,
            condition=condition,
            fine_valid=fine_valid,
            coarse_active=coarse_active,
            shifts_to_test=(1,),
            interior_margins=(4, 8),
        )
        self.assertTrue(all(error.shape == state.shape for error in errors.values()))
        self.assertEqual(maps, {})
        self.assertEqual(len(audit["shifts"]), 2)
        self.assertTrue(
            all(
                value["margin4"]["interior_absolute_rms"] == 0
                and value["margin8"]["interior_absolute_rms"] == 0
                for value in audit["shifts"].values()
            )
        )

    def test_decision_never_authorizes_training(self) -> None:
        channel_metrics = {
            f"d{lead}_{field}": {
                "normalized_rmse": 1e-5,
                "normalized_abs_p95": 2e-5,
                "normalized_abs_max": 3e-5,
            }
            for lead in (3, 6, 9)
            for field in ("sic", "sit")
        }
        comparison = {"aggregate": channel_metrics, "cases": {"case00": channel_metrics}}
        solver = {"rk4_17_vs_65": comparison, "rk4_33_vs_65": comparison}
        shift = {
            "shifts": {
                "dy0_dx1": {
                    "margin16": {"interior_relative_rms": 0.01},
                    "margin32": {"interior_relative_rms": 0.01},
                    "phase2_contrast": 0.1,
                    "phase4_contrast": 0.1,
                    "phase8_contrast": 0.1,
                }
            }
        }
        result = _decision(solver, shift)
        self.assertFalse(result["permits_training"])
        self.assertFalse(result["causal_diagnosis_permitted"])
        self.assertIn("inconclusive", result["status"])

    def test_recursive_finite_guard_rejects_nan(self) -> None:
        with self.assertRaises(FloatingPointError):
            _require_finite_scalars({"nested": [0.0, float("nan")]})

    def test_probe_masking_contract_keeps_land_at_zero(self) -> None:
        active = torch.tensor([[[[1.0, 0.0]]]])
        raw_noise = torch.tensor([[[[2.0, 9.0]]]]).expand(-1, 6, -1, -1)
        masked = torch.where(active.expand_as(raw_noise) > 0, raw_noise, torch.zeros_like(raw_noise))
        truth = torch.zeros_like(masked)
        probe = 0.5 * truth + 0.5 * masked
        self.assertTrue(torch.all(probe[..., 1] == 0))
        self.assertTrue(torch.all(probe[..., 0] == 1))

    def test_tracker_close_errors_cannot_be_published_as_success(self) -> None:
        class FailingTracker:
            def __init__(self, error):
                self.error = error

            def close(self):
                raise self.error

        for error in (RuntimeError("network close failed"), TimeoutError("SIGTERM")):
            with self.subTest(error=type(error).__name__):
                lifecycle = _Lifecycle(tracker=FailingTracker(error))
                with self.assertRaises(type(error)):
                    _close_tracker_for_success(lifecycle)
                self.assertIsNotNone(lifecycle.tracker)


if __name__ == "__main__":
    unittest.main()
