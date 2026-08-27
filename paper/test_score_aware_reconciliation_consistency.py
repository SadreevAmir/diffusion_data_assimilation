#!/usr/bin/env python3
"""Negative fixtures for the five-file score-aware result transition."""

from __future__ import annotations

import unittest

from paper.check_publication_artifacts import (
    PAPER_DIR,
    validate_score_aware_reconciliation_consistency,
)


POSITIVE = (
    "SCORE_AWARE_RESULT: status=RECONCILED_POSITIVE; "
    "experiment_id=score_aware_valid; candidate=score_aware_candidate; "
    "overall_eligible=true"
)
NEGATIVE = (
    "SCORE_AWARE_RESULT: status=RECONCILED_NEGATIVE; "
    "experiment_id=score_aware_valid; candidate=score_aware_candidate; "
    "overall_eligible=false"
)


class ScoreAwareReconciliationConsistencyTests(unittest.TestCase):
    def test_current_pre_result_state_passes(self) -> None:
        names = (
            "PAPER_DRAFT.md",
            "CLAIM_LEDGER.md",
            "PUBLICATION_READINESS.md",
            "REPRODUCIBILITY.md",
            "SCORE_AWARE_RESULT_RECONCILIATION.md",
        )
        documents = [(PAPER_DIR / name).read_text(encoding="utf-8") for name in names]
        validate_score_aware_reconciliation_consistency(*documents)

    def test_positive_transition_passes(self) -> None:
        validate_score_aware_reconciliation_consistency(*([POSITIVE] * 5))

    def test_negative_transition_passes(self) -> None:
        validate_score_aware_reconciliation_consistency(*([NEGATIVE] * 5))

    def test_partial_transition_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "must contain exactly one"):
            validate_score_aware_reconciliation_consistency(
                POSITIVE, POSITIVE, POSITIVE, "unchanged", POSITIVE
            )

    def test_cross_file_decision_mismatch_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "markers disagree"):
            validate_score_aware_reconciliation_consistency(
                POSITIVE, POSITIVE, NEGATIVE, POSITIVE, POSITIVE
            )

    def test_branch_boolean_contradiction_fails_closed(self) -> None:
        contradictory = POSITIVE.replace("overall_eligible=true", "overall_eligible=false")
        with self.assertRaisesRegex(ValueError, "contradicts overall_eligible"):
            validate_score_aware_reconciliation_consistency(*([contradictory] * 5))

    def test_pre_result_with_marker_fails_closed(self) -> None:
        reconciliation = "Status: PRE_RESULT_NO_TRUSTED_MODE\n" + NEGATIVE
        with self.assertRaisesRegex(ValueError, "pre-result"):
            validate_score_aware_reconciliation_consistency(
                "unchanged", "unchanged", "unchanged", "unchanged", reconciliation
            )


if __name__ == "__main__":
    unittest.main()
