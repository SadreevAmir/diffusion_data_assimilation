import unittest

from assim_lib.bounded_clean_state_overfit import (
    ANCHOR_DAY_OFFSETS,
    DIAGNOSTIC_UPDATES,
    MAX_UPDATES,
    _final_gate,
    anchor_indices,
    diagnostic_indices,
)


def _diagnostic(rmse: float, spread_ratio: float, spread: float) -> dict:
    return {
        "support": {"sic_below_zero": 0, "sic_above_one": 0, "sit_below_zero": 0},
        "leads": {
            lead: {
                field: {
                    "ensemble_mean_rmse": rmse,
                    "member0_rmse": rmse,
                    "endpoint_to_terminal_spread_ratio": spread_ratio,
                    "endpoint_spread": spread,
                    "terminal_ode_spread": spread,
                }
                for field in ("sic", "sit")
            }
            for lead in ("d3", "d6", "d9")
        },
    }


class BoundedCleanStateOverfitTests(unittest.TestCase):
    def test_anchor_schedule_is_four_complete_days(self) -> None:
        indices = anchor_indices(10_000)
        self.assertEqual(ANCHOR_DAY_OFFSETS, (14, 105, 196, 287))
        self.assertEqual(len(indices), 96)
        for day in ANCHOR_DAY_OFFSETS:
            self.assertEqual(
                tuple(index for index in indices if index // 24 == day),
                tuple(day * 24 + hour for hour in range(24)),
            )
        self.assertEqual(
            diagnostic_indices(10_000),
            tuple(day * 24 + 23 for day in ANCHOR_DAY_OFFSETS),
        )

    def test_anchor_schedule_fails_outside_dataset(self) -> None:
        with self.assertRaises(ValueError):
            anchor_indices(ANCHOR_DAY_OFFSETS[-1] * 24)

    def test_numeric_gate_passes_only_complete_noncollapsed_result(self) -> None:
        losses = [1.0] * 32 + [0.5] * (MAX_UPDATES - 64) + [0.4] * 32
        diagnostics = {
            DIAGNOSTIC_UPDATES[0]: _diagnostic(0.2, 0.95, 0.02),
            DIAGNOSTIC_UPDATES[1]: _diagnostic(0.15, 0.94, 0.018),
            DIAGNOSTIC_UPDATES[2]: _diagnostic(0.1, 0.90, 0.015),
        }
        gate = _final_gate(losses, diagnostics)
        self.assertEqual(
            gate["status"], "passed_numeric_pending_solver_and_visual_review"
        )
        self.assertTrue(gate["visual_review_required"])
        self.assertFalse(gate["permits_long_training"])

    def test_numeric_gate_reports_but_does_not_reject_expected_overfit_collapse(self) -> None:
        losses = [1.0] * 32 + [0.4] * (MAX_UPDATES - 32)
        diagnostics = {
            DIAGNOSTIC_UPDATES[0]: _diagnostic(0.2, 0.95, 0.02),
            DIAGNOSTIC_UPDATES[1]: _diagnostic(0.15, 0.94, 0.018),
            DIAGNOSTIC_UPDATES[2]: _diagnostic(0.1, 0.50, 0.015),
        }
        gate = _final_gate(losses, diagnostics)
        self.assertEqual(
            gate["status"], "passed_numeric_pending_solver_and_visual_review"
        )
        self.assertIn("delta-law", gate["spread_interpretation"])

    def test_numeric_gate_rejects_nonfinite_diagnostics(self) -> None:
        losses = [1.0] * 32 + [0.4] * (MAX_UPDATES - 32)
        diagnostics = {
            DIAGNOSTIC_UPDATES[0]: _diagnostic(0.2, 0.95, 0.02),
            DIAGNOSTIC_UPDATES[1]: _diagnostic(0.15, 0.94, 0.018),
            DIAGNOSTIC_UPDATES[2]: _diagnostic(float("nan"), 0.90, 0.015),
        }
        with self.assertRaises(FloatingPointError):
            _final_gate(losses, diagnostics)

    def test_numeric_gate_rejects_nonfinite_intermediate_diagnostics_and_losses(self) -> None:
        diagnostics = {
            DIAGNOSTIC_UPDATES[0]: _diagnostic(0.2, 0.95, 0.02),
            DIAGNOSTIC_UPDATES[1]: _diagnostic(float("inf"), 0.94, 0.018),
            DIAGNOSTIC_UPDATES[2]: _diagnostic(0.1, 0.90, 0.015),
        }
        with self.assertRaises(FloatingPointError):
            _final_gate([1.0] * 32 + [0.4] * (MAX_UPDATES - 32), diagnostics)
        diagnostics[DIAGNOSTIC_UPDATES[1]] = _diagnostic(0.15, 0.94, 0.018)
        losses = [1.0] * MAX_UPDATES
        losses[100] = float("nan")
        with self.assertRaises(FloatingPointError):
            _final_gate(losses, diagnostics)


if __name__ == "__main__":
    unittest.main()
