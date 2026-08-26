#!/usr/bin/env python3
"""Negative fixtures for empirical claim-to-evidence traceability."""

from __future__ import annotations

import unittest

from paper.check_publication_artifacts import (
    PAPER_DIR,
    validate_empirical_traceability,
    validate_external_primary_consistency,
)


class EmpiricalTraceabilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manuscript = (PAPER_DIR / "PAPER_DRAFT.md").read_text(encoding="utf-8")

    def test_current_traceability_passes(self) -> None:
        validate_empirical_traceability(self.manuscript, PAPER_DIR)

    def test_missing_claim_coverage_fails_closed(self) -> None:
        mutated = self.manuscript.replace(
            "| C3, C4, C5, C6, C7 | Section 6 preliminary-guidance",
            "| C4, C5, C6, C7 | Section 6 preliminary-guidance",
            1,
        )
        with self.assertRaisesRegex(ValueError, "empirical evidence traceability mismatch"):
            validate_empirical_traceability(mutated, PAPER_DIR)

    def test_nonconcrete_presentation_fails_closed(self) -> None:
        mutated = self.manuscript.replace(
            "Section 6 preliminary-guidance and affine-logit paragraphs",
            "Preliminary-guidance and affine-logit discussion",
            1,
        )
        with self.assertRaisesRegex(ValueError, "no concrete manuscript presentation"):
            validate_empirical_traceability(mutated, PAPER_DIR)

    def test_source_path_escape_fails_closed(self) -> None:
        mutated = self.manuscript.replace(
            "| C39 | Independent primary result section and Section 8 independent-primary table | `EXTERNAL_PRIMARY_RESULT_RECONCILIATION.md` |",
            "| C39 | Independent primary result section and Section 8 independent-primary table | `../EXTERNAL_PRIMARY_RESULT_RECONCILIATION.md` |",
            1,
        )
        with self.assertRaisesRegex(ValueError, "source is missing or escapes paper"):
            validate_empirical_traceability(mutated, PAPER_DIR)


class ExternalPrimaryConsistencyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.documents = {
            name: (PAPER_DIR / name).read_text(encoding="utf-8")
            for name in (
                "PAPER_DRAFT.md",
                "CLAIM_LEDGER.md",
                "REPRODUCIBILITY.md",
            )
        }

    def test_current_external_primary_claims_pass(self) -> None:
        validate_external_primary_consistency(self.documents)

    def test_missing_absolute_rank_anchor_fails_closed(self) -> None:
        self.documents["CLAIM_LEDGER.md"] = self.documents["CLAIM_LEDGER.md"].replace(
            "absolute rank reliability", "relative reliability"
        )
        with self.assertRaisesRegex(ValueError, "decision anchors"):
            validate_external_primary_consistency(self.documents)

    def test_invalid_legacy_rank_claim_fails_closed(self) -> None:
        self.documents["PAPER_DRAFT.md"] += "\n88 percent upper-rank-bin\n"
        with self.assertRaisesRegex(ValueError, "invalid external-primary claims"):
            validate_external_primary_consistency(self.documents)


if __name__ == "__main__":
    unittest.main()
