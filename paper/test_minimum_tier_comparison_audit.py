#!/usr/bin/env python3
"""Negative fixtures for the fail-closed minimum-tier comparison audit."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from paper.check_publication_artifacts import (
    MINIMUM_TIER_CLAIM_CONSISTENCY_ANCHORS,
    MINIMUM_TIER_ROW,
    PAPER_DIR,
    READINESS_BLOCKER_ROW,
    REQUIRED_REGRESSION_SUITES,
    validate_documented_regression_suites,
    validate_minimum_tier_key_claims,
    validate_minimum_tier_comparisons,
    validate_readiness_blockers,
)


class MinimumTierComparisonAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.audit = (PAPER_DIR / "MINIMUM_TIER_COMPARISON_AUDIT.md").read_text(
            encoding="utf-8"
        )
        self.reproducibility = (PAPER_DIR / "REPRODUCIBILITY.md").read_text(
            encoding="utf-8"
        )
        self.manuscript = (PAPER_DIR / "PAPER_DRAFT.md").read_text(encoding="utf-8")
        self.readiness = (PAPER_DIR / "PUBLICATION_READINESS.md").read_text(
            encoding="utf-8"
        )

    def make_fixture(self, audit: str) -> tempfile.TemporaryDirectory[str]:
        temporary = tempfile.TemporaryDirectory()
        fixture_dir = Path(temporary.name)
        (fixture_dir / "MINIMUM_TIER_COMPARISON_AUDIT.md").write_text(
            audit, encoding="utf-8"
        )
        for row in MINIMUM_TIER_ROW.finditer(self.audit):
            source = row["source"]
            target = fixture_dir / source
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((PAPER_DIR / source).read_bytes())
        conformal_contract = "NEXT_CONFORMAL_BASELINE_CONTRACT.md"
        (fixture_dir / conformal_contract).write_bytes(
            (PAPER_DIR / conformal_contract).read_bytes()
        )
        probabilistic_contract = "NEXT_PROBABILISTIC_DA_COMPARISON_CONTRACT.md"
        (fixture_dir / probabilistic_contract).write_bytes(
            (PAPER_DIR / probabilistic_contract).read_bytes()
        )
        return temporary

    def assert_fixture_fails(self, audit: str, message: str) -> None:
        with self.make_fixture(audit) as temporary:
            with self.assertRaisesRegex(ValueError, message):
                validate_minimum_tier_comparisons(Path(temporary))

    def test_current_comparison_map_passes(self) -> None:
        validate_minimum_tier_comparisons(PAPER_DIR)

    def test_weakened_conformal_stop_go_fails_closed(self) -> None:
        with self.make_fixture(self.audit) as temporary:
            fixture_dir = Path(temporary)
            source = PAPER_DIR / "NEXT_CONFORMAL_BASELINE_CONTRACT.md"
            target = fixture_dir / source.name
            contract = source.read_text(encoding="utf-8")
            target.write_text(
                contract.replace("coverage of at least `0.85`", "improved coverage"),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "contract is incomplete or weakened"):
                validate_minimum_tier_comparisons(fixture_dir)

    def test_missing_required_row_fails_closed(self) -> None:
        first_row = MINIMUM_TIER_ROW.search(self.audit)
        self.assertIsNotNone(first_row)
        assert first_row is not None
        mutated = self.audit[: first_row.start()] + self.audit[first_row.end() :]
        self.assert_fixture_fails(mutated, "incomplete or duplicate comparison map")

    def test_weakened_probabilistic_da_stop_go_fails_closed(self) -> None:
        with self.make_fixture(self.audit) as temporary:
            fixture_dir = Path(temporary)
            source = PAPER_DIR / "NEXT_PROBABILISTIC_DA_COMPARISON_CONTRACT.md"
            target = fixture_dir / source.name
            contract = source.read_text(encoding="utf-8")
            target.write_text(
                contract.replace(
                    "no more than `1.10` times raw", "competitive with raw"
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError, "probabilistic DA comparison contract is incomplete or weakened"
            ):
                validate_minimum_tier_comparisons(fixture_dir)

    def test_changed_evidence_status_fails_closed(self) -> None:
        mutated = self.audit.replace("| PRESENT |", "| PRESENT_NEGATIVE |", 1)
        self.assertNotEqual(mutated, self.audit)
        self.assert_fixture_fails(mutated, "evidence-status map differs")

    def test_source_path_escape_fails_closed(self) -> None:
        first_row = MINIMUM_TIER_ROW.search(self.audit)
        self.assertIsNotNone(first_row)
        assert first_row is not None
        source = first_row["source"]
        mutated = self.audit.replace(f"`{source}`", f"`../{source}`", 1)
        self.assert_fixture_fails(mutated, "source is missing or escapes paper")

    def test_changed_compact_source_fails_closed(self) -> None:
        mutated = self.audit.replace(
            "| `CLAIM_LEDGER.md` | PRESENT_NEGATIVE |",
            "| `REPRODUCIBILITY.md` | PRESENT_NEGATIVE |",
            1,
        )
        self.assertNotEqual(mutated, self.audit)
        self.assert_fixture_fails(mutated, "source differs from evidence contract")

    def test_missing_source_evidence_anchor_fails_closed(self) -> None:
        with self.make_fixture(self.audit) as temporary:
            fixture_dir = Path(temporary)
            ledger = fixture_dir / "CLAIM_LEDGER.md"
            mutated = ledger.read_text(encoding="utf-8").replace("| C7 |", "| X7 |", 1)
            self.assertNotEqual(mutated, ledger.read_text(encoding="utf-8"))
            ledger.write_text(mutated, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "lacks evidence anchor"):
                validate_minimum_tier_comparisons(fixture_dir)

    def test_documented_regression_suites_match_executable_contract(self) -> None:
        validate_documented_regression_suites(self.reproducibility)

    def test_documented_regression_suite_omission_fails_closed(self) -> None:
        suite = REQUIRED_REGRESSION_SUITES[-1]
        mutated = self.reproducibility.replace(f"  {suite}\n", "", 1)
        with self.assertRaisesRegex(ValueError, "differ from the required"):
            validate_documented_regression_suites(mutated)

    def test_documented_regression_suite_addition_fails_closed(self) -> None:
        suite = REQUIRED_REGRESSION_SUITES[-1]
        mutated = self.reproducibility.replace(
            f"  {suite}\n", f"  paper.test_unreviewed_suite \\\n  {suite}\n", 1
        )
        with self.assertRaisesRegex(ValueError, "differ from the required"):
            validate_documented_regression_suites(mutated)

    def test_documented_regression_suite_reordering_fails_closed(self) -> None:
        first, second = REQUIRED_REGRESSION_SUITES[:2]
        mutated = self.reproducibility.replace(
            f"  {first} \\\n  {second} \\\n",
            f"  {second} \\\n  {first} \\\n",
            1,
        )
        with self.assertRaisesRegex(ValueError, "differ from the required"):
            validate_documented_regression_suites(mutated)

    def test_claim_consistency_contract_covers_all_publication_surfaces(self) -> None:
        self.assertEqual(
            set(MINIMUM_TIER_CLAIM_CONSISTENCY_ANCHORS),
            {
                "PAPER_DRAFT.md",
                "CLAIM_LEDGER.md",
                "PUBLICATION_READINESS.md",
                "REPRODUCIBILITY.md",
            },
        )
        for filename, anchors in MINIMUM_TIER_CLAIM_CONSISTENCY_ANCHORS.items():
            text = (PAPER_DIR / filename).read_text(encoding="utf-8")
            self.assertTrue(anchors)
            for anchor in anchors:
                self.assertIn(anchor, text)

    def test_key_claims_name_both_missing_families(self) -> None:
        validate_minimum_tier_key_claims(self.manuscript, PAPER_DIR)

    def test_readiness_blockers_match_normative_evidence(self) -> None:
        validate_readiness_blockers(self.readiness, PAPER_DIR)

    def test_readiness_cannot_drop_blocker(self) -> None:
        row = READINESS_BLOCKER_ROW.search(self.readiness)
        self.assertIsNotNone(row)
        assert row is not None
        mutated = self.readiness[: row.start()] + self.readiness[row.end() :]
        with self.assertRaisesRegex(ValueError, "matrix is incomplete"):
            validate_readiness_blockers(mutated, PAPER_DIR)

    def test_readiness_cannot_promote_missing_evidence(self) -> None:
        mutated = self.readiness.replace(
            "| Conformal intervals | `MINIMUM_TIER_COMPARISON_AUDIT.md` | MISSING |",
            "| Conformal intervals | `MINIMUM_TIER_COMPARISON_AUDIT.md` | PRESENT |",
            1,
        )
        self.assertNotEqual(mutated, self.readiness)
        with self.assertRaisesRegex(ValueError, "normative blocker set"):
            validate_readiness_blockers(mutated, PAPER_DIR)

    def test_readiness_blocker_requires_closure_condition(self) -> None:
        mutated = self.readiness.replace(
            "| Eligible spatially preserving calibration | `RESEARCH_PLAN.md` | MISSING_ELIGIBLE_RESULT | One frozen candidate passes every mandatory no-compensation gate family |",
            "| Eligible spatially preserving calibration | `RESEARCH_PLAN.md` | MISSING_ELIGIBLE_RESULT |  |",
            1,
        )
        self.assertNotEqual(mutated, self.readiness)
        with self.assertRaisesRegex(ValueError, "exact closure condition"):
            validate_readiness_blockers(mutated, PAPER_DIR)

    def test_abstract_cannot_omit_missing_family(self) -> None:
        start = self.manuscript.index("## Abstract")
        end = self.manuscript.index("## 1.", start)
        section = self.manuscript[start:end]
        mutated_section = section.replace("Conformal intervals", "Planned intervals", 1)
        self.assertNotEqual(mutated_section, section)
        mutated = self.manuscript[:start] + mutated_section + self.manuscript[end:]
        with self.assertRaisesRegex(ValueError, "Abstract omits missing family"):
            validate_minimum_tier_key_claims(mutated, PAPER_DIR)

    def test_contributions_cannot_promote_minimum_tier(self) -> None:
        start = self.manuscript.index("### Contributions supported by the present evidence")
        end = self.manuscript.index("## 2.", start)
        section = self.manuscript[start:end]
        mutated_section = section.replace(
            "Conformal intervals", "The baseline inventory", 1
        ).replace("remain\nmissing", "is complete", 1)
        self.assertNotEqual(mutated_section, section)
        mutated = self.manuscript[:start] + mutated_section + self.manuscript[end:]
        with self.assertRaisesRegex(ValueError, "Contributions omits missing family"):
            validate_minimum_tier_key_claims(mutated, PAPER_DIR)

    def test_conclusion_is_required(self) -> None:
        mutated = self.manuscript.replace("## 9. Conclusion", "## Closing remarks", 1)
        with self.assertRaisesRegex(ValueError, "Conclusion section is missing"):
            validate_minimum_tier_key_claims(mutated, PAPER_DIR)


if __name__ == "__main__":
    unittest.main()
