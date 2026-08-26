#!/usr/bin/env python3
"""Negative fixtures for the fail-closed minimum-tier comparison audit."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from paper.check_publication_artifacts import (
    MINIMUM_TIER_ROW,
    PAPER_DIR,
    REQUIRED_REGRESSION_SUITES,
    validate_documented_regression_suites,
    validate_minimum_tier_comparisons,
)


class MinimumTierComparisonAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.audit = (PAPER_DIR / "MINIMUM_TIER_COMPARISON_AUDIT.md").read_text(
            encoding="utf-8"
        )
        self.reproducibility = (PAPER_DIR / "REPRODUCIBILITY.md").read_text(
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


if __name__ == "__main__":
    unittest.main()
