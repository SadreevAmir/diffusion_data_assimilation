from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from paper import casewise_safety_selector_reference as reference
from paper.validate_casewise_safety_selector_semantics import validate_semantic_parity


REFERENCE = Path(reference.__file__)


class CasewiseSelectorSemanticTests(unittest.TestCase):
    def divergent(self, old, new):
        source = REFERENCE.read_text(encoding="utf-8").replace(old, new)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        runner = Path(temporary.name) / "runner.py"
        runner.write_text(source, encoding="utf-8")
        return runner

    def test_reference_passes_or_declares_numpy_dependency(self):
        if reference.np is None:
            with self.assertRaisesRegex(RuntimeError, "numpy is required"):
                validate_semantic_parity(REFERENCE)
        else:
            validate_semantic_parity(REFERENCE)

    def test_purge_divergence_fails_closed(self):
        runner = self.divergent("excluded_start = max(0, start - PURGE)",
                                "excluded_start = max(0, start - PURGE + 1)")
        with self.assertRaisesRegex(ValueError, "folds"):
            validate_semantic_parity(runner)

    def test_loss_weight_divergence_fails_closed(self):
        runner = self.divergent("4.0 * max(0.0, diagnostics[\"rank_abs_error\"]",
                                "3.0 * max(0.0, diagnostics[\"rank_abs_error\"]")
        with self.assertRaisesRegex(ValueError, "loss"):
            validate_semantic_parity(runner)

    def test_margin_rule_divergence_fails_closed(self):
        runner = self.divergent("margin >= ACTION_MARGIN", "margin > ACTION_MARGIN")
        with self.assertRaisesRegex(ValueError, "action"):
            validate_semantic_parity(runner)

    def test_ridge_divergence_fails_closed(self):
        if reference.np is None:
            self.skipTest("numpy is not installed")
        runner = self.divergent("singular**2 + RIDGE", "singular**2 + 2.0 * RIDGE")
        with self.assertRaisesRegex(ValueError, "ridge"):
            validate_semantic_parity(runner)


if __name__ == "__main__":
    unittest.main()
