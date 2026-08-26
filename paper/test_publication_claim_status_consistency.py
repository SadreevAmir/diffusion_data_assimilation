#!/usr/bin/env python3
"""Negative fixtures for Unknown/Rejected manuscript consistency."""

from __future__ import annotations

import unittest

from paper.check_publication_artifacts import PAPER_DIR, validate_claim_status_consistency


class ClaimStatusConsistencyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manuscript = (PAPER_DIR / "PAPER_DRAFT.md").read_text(encoding="utf-8")
        self.ledger = (PAPER_DIR / "CLAIM_LEDGER.md").read_text(encoding="utf-8")

    def test_current_status_map_passes(self) -> None:
        validate_claim_status_consistency(self.manuscript, self.ledger)

    def test_unknown_promotion_fails_closed(self) -> None:
        mutated = self.manuscript.replace("| C39 | Independent primary result section", "| C9, C39 | Independent primary result section", 1)
        with self.assertRaisesRegex(ValueError, "Unknown claim was promoted"):
            validate_claim_status_consistency(mutated, self.ledger)

    def test_missing_rejected_claim_fails_closed(self) -> None:
        mutated = self.manuscript.replace("| C38 | Section 8 later-mechanism family matrix", "| C1 | Section 8 later-mechanism family matrix", 1)
        with self.assertRaisesRegex(ValueError, "Rejected claim is missing"):
            validate_claim_status_consistency(mutated, self.ledger)

    def test_neutralized_decision_fails_closed(self) -> None:
        mutated = self.manuscript.replace("Overall rejection only; no unsupported family effect sizes", "Overall result only; no unsupported family effect sizes", 1)
        with self.assertRaisesRegex(ValueError, "lacks an explicit negative decision"):
            validate_claim_status_consistency(mutated, self.ledger)


if __name__ == "__main__":
    unittest.main()
