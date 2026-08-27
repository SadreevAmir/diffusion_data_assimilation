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
    READINESS_CLOSURE_ROUTE_ROW,
    REQUIRED_REGRESSION_SUITES,
    SERVER_ONLY_COMMAND_INPUTS,
    validate_documented_regression_suites,
    validate_eligible_calibration_transition,
    validate_minimum_tier_key_claims,
    validate_minimum_tier_evidence_guards,
    validate_minimum_tier_comparisons,
    validate_publication_status,
    validate_readiness_blockers,
    validate_server_only_command_inputs,
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
        self.ledger = (PAPER_DIR / "CLAIM_LEDGER.md").read_text(encoding="utf-8")

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

    def test_server_only_command_inputs_match_exact_contract(self) -> None:
        validate_server_only_command_inputs(self.reproducibility)

    def test_server_only_command_omission_fails_closed(self) -> None:
        command = next(iter(SERVER_ONLY_COMMAND_INPUTS))
        line = next(
            line
            for line in self.reproducibility.splitlines(keepends=True)
            if line.startswith(f"| `{command}` |")
        )
        mutated = self.reproducibility.replace(line, "", 1)
        with self.assertRaisesRegex(ValueError, "matrix is incomplete"):
            validate_server_only_command_inputs(mutated)

    def test_external_input_command_cannot_be_local_oracle(self) -> None:
        mutated = self.reproducibility.replace(
            "| `calibration_summary_figure` | `SERVER_ONLY` |",
            "| `calibration_summary_figure` | `LOCAL_ORACLE` |",
            1,
        )
        self.assertNotEqual(mutated, self.reproducibility)
        with self.assertRaisesRegex(ValueError, "provenance or compact-input contract"):
            validate_server_only_command_inputs(mutated)

    def test_server_only_input_schema_cannot_be_weakened(self) -> None:
        mutated = self.reproducibility.replace(
            "exactly 160 rows", "a finite number of rows", 1
        )
        self.assertNotEqual(mutated, self.reproducibility)
        with self.assertRaisesRegex(ValueError, "provenance or compact-input contract"):
            validate_server_only_command_inputs(mutated)

    def test_server_only_producer_cannot_be_substituted(self) -> None:
        mutated = self.reproducibility.replace(
            "| `joint_crossfit_spread_calibration_valid` |",
            "| `joint_existing_ensemble_calibration_audit_valid` |",
            1,
        )
        self.assertNotEqual(mutated, self.reproducibility)
        with self.assertRaisesRegex(ValueError, "provenance or compact-input contract"):
            validate_server_only_command_inputs(mutated)

    def test_server_only_artifact_cannot_be_substituted(self) -> None:
        mutated = self.reproducibility.replace(
            "| `per_case_metrics.csv` | `per_case_metrics.csv`:",
            "| `metadata.json` | `per_case_metrics.csv`:",
            1,
        )
        self.assertNotEqual(mutated, self.reproducibility)
        with self.assertRaisesRegex(ValueError, "provenance or compact-input contract"):
            validate_server_only_command_inputs(mutated)

    def test_server_only_manifest_identity_cannot_be_weakened(self) -> None:
        mutated = self.reproducibility.replace(
            "artifact manifest binds `per_case_metrics.csv` by SHA-256",
            "artifact manifest may list `per_case_metrics.csv`",
            1,
        )
        self.assertNotEqual(mutated, self.reproducibility)
        with self.assertRaisesRegex(ValueError, "provenance or compact-input contract"):
            validate_server_only_command_inputs(mutated)

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

    def test_minimum_tier_evidence_guards_pass(self) -> None:
        validate_minimum_tier_evidence_guards(
            self.manuscript, self.ledger, self.readiness, self.audit
        )

    def test_eligible_calibration_guard_passes_current_blocked_state(self) -> None:
        validate_eligible_calibration_transition(
            self.manuscript, self.ledger, self.readiness, self.reproducibility
        )

    def test_eligible_calibration_positive_marker_updates_all_surfaces(self) -> None:
        marker = "a" * 64
        old = "| `eligible_calibration` | `MISSING_ELIGIBLE_RESULT` | `NONE` | `BLOCKED` |"
        new = f"| `eligible_calibration` | `ELIGIBLE` | `{marker}` | `DECISION_BEARING` |"
        originals = (self.manuscript, self.ledger, self.readiness, self.reproducibility)
        updated = [text.replace(old, new, 1) for text in originals]
        self.assertTrue(all(text != original for text, original in zip(updated, originals)))
        updated[0] += (
            f"\nEligible calibration decision: `ELIGIBLE`; compact record: `{marker}`; "
            "claim role: `DECISION_BEARING`.\n"
        )
        updated[2] = updated[2].replace(
            "| Eligible spatially preserving calibration | `RESEARCH_PLAN.md` | MISSING_ELIGIBLE_RESULT |",
            "| Eligible spatially preserving calibration | `RESEARCH_PLAN.md` | ELIGIBLE |",
            1,
        )
        updated[3] += (
            f"\nEligible calibration reproducibility identity: `{marker}`; "
            "verification: `HASH_VERIFIED`.\n"
        )
        validate_eligible_calibration_transition(*updated)

    def test_eligible_calibration_manuscript_record_must_match_guard(self) -> None:
        marker = "a" * 64
        mismatched = "b" * 64
        old = "| `eligible_calibration` | `MISSING_ELIGIBLE_RESULT` | `NONE` | `BLOCKED` |"
        new = f"| `eligible_calibration` | `ELIGIBLE` | `{marker}` | `DECISION_BEARING` |"
        updated = [
            text.replace(old, new, 1)
            for text in (self.manuscript, self.ledger, self.readiness, self.reproducibility)
        ]
        updated[0] += (
            f"\nEligible calibration decision: `ELIGIBLE`; compact record: `{mismatched}`; "
            "claim role: `DECISION_BEARING`.\n"
        )
        updated[2] = updated[2].replace(
            "| Eligible spatially preserving calibration | `RESEARCH_PLAN.md` | MISSING_ELIGIBLE_RESULT |",
            "| Eligible spatially preserving calibration | `RESEARCH_PLAN.md` | ELIGIBLE |",
            1,
        )
        updated[3] += (
            f"\nEligible calibration reproducibility identity: `{marker}`; "
            "verification: `HASH_VERIFIED`.\n"
        )
        with self.assertRaisesRegex(ValueError, "manuscript claim"):
            validate_eligible_calibration_transition(*updated)

    def test_eligible_calibration_reproducibility_record_must_match_guard(self) -> None:
        marker = "a" * 64
        mismatched = "b" * 64
        old = "| `eligible_calibration` | `MISSING_ELIGIBLE_RESULT` | `NONE` | `BLOCKED` |"
        new = f"| `eligible_calibration` | `ELIGIBLE` | `{marker}` | `DECISION_BEARING` |"
        updated = [
            text.replace(old, new, 1)
            for text in (self.manuscript, self.ledger, self.readiness, self.reproducibility)
        ]
        updated[0] += (
            f"\nEligible calibration decision: `ELIGIBLE`; compact record: `{marker}`; "
            "claim role: `DECISION_BEARING`.\n"
        )
        updated[2] = updated[2].replace(
            "| Eligible spatially preserving calibration | `RESEARCH_PLAN.md` | MISSING_ELIGIBLE_RESULT |",
            "| Eligible spatially preserving calibration | `RESEARCH_PLAN.md` | ELIGIBLE |",
            1,
        )
        updated[3] += (
            f"\nEligible calibration reproducibility identity: `{mismatched}`; "
            "verification: `HASH_VERIFIED`.\n"
        )
        with self.assertRaisesRegex(ValueError, "reproducibility identity"):
            validate_eligible_calibration_transition(*updated)

    def test_eligible_calibration_positive_guard_without_full_transition_fails(self) -> None:
        marker = "e" * 64
        old = "| `eligible_calibration` | `MISSING_ELIGIBLE_RESULT` | `NONE` | `BLOCKED` |"
        new = f"| `eligible_calibration` | `ELIGIBLE` | `{marker}` | `DECISION_BEARING` |"
        updated = [
            text.replace(old, new, 1)
            for text in (self.manuscript, self.ledger, self.readiness, self.reproducibility)
        ]
        with self.assertRaisesRegex(ValueError, "manuscript claim"):
            validate_eligible_calibration_transition(*updated)

    def test_eligible_calibration_positive_status_must_follow_blocker_matrix(self) -> None:
        marker = "f" * 64
        old = "| `eligible_calibration` | `MISSING_ELIGIBLE_RESULT` | `NONE` | `BLOCKED` |"
        new = f"| `eligible_calibration` | `ELIGIBLE` | `{marker}` | `DECISION_BEARING` |"
        updated = [
            text.replace(old, new, 1)
            for text in (self.manuscript, self.ledger, self.readiness, self.reproducibility)
        ]
        updated[0] += (
            f"\nEligible calibration decision: `ELIGIBLE`; compact record: `{marker}`; "
            "claim role: `DECISION_BEARING`.\n"
        )
        updated[2] = updated[2].replace(
            "| Eligible spatially preserving calibration | `RESEARCH_PLAN.md` | MISSING_ELIGIBLE_RESULT |",
            "| Eligible spatially preserving calibration | `RESEARCH_PLAN.md` | ELIGIBLE |",
            1,
        ).replace("Publication status: NOT_READY", "Publication status: READY_FOR_HUMAN_REVIEW", 1)
        updated[3] += (
            f"\nEligible calibration reproducibility identity: `{marker}`; "
            "verification: `HASH_VERIFIED`.\n"
        )
        with self.assertRaisesRegex(ValueError, "status is inconsistent"):
            validate_eligible_calibration_transition(*updated)

    def test_eligible_calibration_partial_positive_update_fails_closed(self) -> None:
        marker = "b" * 64
        mutated = self.manuscript.replace(
            "| `eligible_calibration` | `MISSING_ELIGIBLE_RESULT` | `NONE` | `BLOCKED` |",
            f"| `eligible_calibration` | `ELIGIBLE` | `{marker}` | `DECISION_BEARING` |",
            1,
        )
        with self.assertRaisesRegex(ValueError, "not atomic"):
            validate_eligible_calibration_transition(
                mutated, self.ledger, self.readiness, self.reproducibility
            )

    def test_eligible_calibration_positive_records_must_match(self) -> None:
        old = "| `eligible_calibration` | `MISSING_ELIGIBLE_RESULT` | `NONE` | `BLOCKED` |"
        texts = []
        for index, source in enumerate(
            (self.manuscript, self.ledger, self.readiness, self.reproducibility)
        ):
            record = ("c" if index < 3 else "d") * 64
            texts.append(source.replace(
                old,
                f"| `eligible_calibration` | `ELIGIBLE` | `{record}` | `DECISION_BEARING` |",
                1,
            ))
        with self.assertRaisesRegex(ValueError, "not atomic"):
            validate_eligible_calibration_transition(*texts)

    def test_conformal_decision_without_compact_record_fails_closed(self) -> None:
        mutated = self.manuscript + "\nThe result is `CONFORMAL_USEFUL`.\n"
        with self.assertRaisesRegex(ValueError, "outside the pre-result matrix"):
            validate_minimum_tier_evidence_guards(
                mutated, self.ledger, self.readiness, self.audit
            )

    def test_probabilistic_da_decision_without_compact_record_fails_closed(self) -> None:
        mutated = self.ledger.replace(
            "| `probabilistic_da` | `MISSING` | `NONE` | `PRE_RESULT_ONLY` |",
            "| `probabilistic_da` | `PROBABILISTIC_DA_USEFUL` | `NONE` | `RESULT` |",
            1,
        )
        with self.assertRaisesRegex(ValueError, "without compact evidence"):
            validate_minimum_tier_evidence_guards(
                self.manuscript, mutated, self.readiness, self.audit
            )

    def test_deterministic_promotion_without_compact_record_fails_closed(self) -> None:
        mutated = self.readiness.replace(
            "| `independent_deterministic` | `PRESENT_DEVELOPMENT_ONLY` | `NONE` | `DEVELOPMENT_ONLY` |",
            "| `independent_deterministic` | `PRESENT` | `NONE` | `RESULT` |",
            1,
        )
        with self.assertRaisesRegex(ValueError, "without compact evidence"):
            validate_minimum_tier_evidence_guards(
                self.manuscript, self.ledger, mutated, self.audit
            )

    def test_readiness_status_matches_open_blockers(self) -> None:
        frozen_handoff = (PAPER_DIR / "FROZEN_EVALUATION_HANDOFF.md").read_text(
            encoding="utf-8"
        )
        self.assertEqual(
            validate_publication_status(self.readiness, frozen_handoff), "NOT_READY"
        )

    def test_readiness_cannot_claim_ready_with_open_blocker_matrix(self) -> None:
        frozen_handoff = (PAPER_DIR / "FROZEN_EVALUATION_HANDOFF.md").read_text(
            encoding="utf-8"
        )
        mutated = self.readiness.replace(
            "Publication status: NOT_READY", "Publication status: READY_FOR_HUMAN_REVIEW", 1
        ).replace(
            "Required scientific blockers: an eligible spatially preserving calibration and\n"
            "the remaining minimum-tier comparisons",
            "Required scientific blockers: none",
            1,
        )
        self.assertNotEqual(mutated, self.readiness)
        with self.assertRaisesRegex(
            ValueError, "ready status contradicts unresolved normative blocker states"
        ):
            validate_publication_status(mutated, frozen_handoff)

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

    def test_readiness_cannot_drop_closure_route(self) -> None:
        row = READINESS_CLOSURE_ROUTE_ROW.search(self.readiness)
        self.assertIsNotNone(row)
        assert row is not None
        mutated = self.readiness[: row.start()] + self.readiness[row.end() :]
        with self.assertRaisesRegex(ValueError, "closure-route matrix is incomplete"):
            validate_readiness_blockers(mutated, PAPER_DIR)

    def test_readiness_cannot_duplicate_closure_contract(self) -> None:
        mutated = self.readiness.replace(
            "`NEXT_CONFORMAL_BASELINE_CONTRACT.md` | PAPER_DRAFT.md:Section 3 frozen outcome matrix",
            "`NEXT_RANK_COHERENT_CONTRACT.md` | PAPER_DRAFT.md:Section 3 frozen outcome matrix",
            1,
        )
        self.assertNotEqual(mutated, self.readiness)
        with self.assertRaisesRegex(ValueError, "closure routes differ"):
            validate_readiness_blockers(mutated, PAPER_DIR)

    def test_readiness_cannot_redirect_claim_ledger_route(self) -> None:
        mutated = self.readiness.replace(
            "CLAIM_LEDGER.md:C17 plus a decision-bearing conformal claim",
            "CLAIM_LEDGER.md:C17 only",
            1,
        )
        self.assertNotEqual(mutated, self.readiness)
        with self.assertRaisesRegex(ValueError, "closure routes differ"):
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
