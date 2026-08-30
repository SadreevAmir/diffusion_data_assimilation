from __future__ import annotations

import unittest
from pathlib import Path

from paper import casewise_safety_selector_runner_prototype as runner
from paper.validate_casewise_safety_selector_semantics import validate_semantic_parity


class CasewiseSelectorRunnerPrototypeTests(unittest.TestCase):
    def test_full_semantic_parity(self):
        path = Path(runner.__file__)
        if runner.np is None:
            with self.assertRaisesRegex(RuntimeError, "numpy is required"):
                validate_semantic_parity(path)
        else:
            validate_semantic_parity(path)

    def test_tie_and_submargin_select_raw(self):
        self.assertEqual(runner.select_action(0.5, 0.5)["selected_action"], "raw")
        self.assertEqual(runner.select_action(0.5019, 0.5)["selected_action"], "raw")
        self.assertEqual(
            runner.select_action(0.502, 0.5)["selected_action"],
            "projected_spread",
        )

    def test_nonfinite_inputs_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "finite numeric"):
            runner.select_action(float("nan"), 0.1)
        diagnostics = {
            "fair_crps": 0.1, "rank_abs_error": 0.1,
            "inner_coverage_error": 0.1, "boundary_error": 0.1,
            "iiee_ratio": 1.0, "edge_ratio": 1.0,
            "variogram_ratio": float("inf"),
        }
        limits = {"rank_limit": 0.1, "inner_limit": 0.1, "boundary_limit": 0.1}
        with self.assertRaisesRegex(ValueError, "finite numeric"):
            runner.scalar_loss(diagnostics, limits)

    def test_zero_variance_training_descriptor_fails_closed(self):
        if runner.np is None:
            self.skipTest("numpy is not installed")
        design = runner.np.ones((24, 6))
        losses = runner.np.linspace(0.0, 1.0, 24)
        with self.assertRaisesRegex(ValueError, "non-zero scale"):
            runner.fit_ridge(design, losses)

    def test_purge_is_non_circular_at_both_edges(self):
        identifiers = tuple(f"case-{index:02d}" for index in range(40))
        folds = runner.build_purged_folds(identifiers)
        self.assertEqual(folds[0]["training_case_ids"][0], "case-11")
        self.assertEqual(folds[-1]["training_case_ids"][-1], "case-28")


if __name__ == "__main__":
    unittest.main()
