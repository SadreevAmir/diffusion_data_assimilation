#!/usr/bin/env python3
"""Negative fixtures for the fail-closed minimum-tier comparison audit."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from paper.check_publication_artifacts import (
    FIGURE_TEXT_ANCHORS,
    MINIMUM_TIER_ADMISSION_TRANSITION_ANCHORS,
    MINIMUM_TIER_EVIDENCE_GUARD_ROWS,
    MINIMUM_TIER_HANDOFF_INVENTORY,
    MINIMUM_TIER_CLAIM_CONSISTENCY_ANCHORS,
    MINIMUM_TIER_ROW,
    PAPER_DIR,
    READINESS_BLOCKER_ROW,
    READINESS_CLOSURE_ROUTE_ROW,
    READINESS_DECISION_SURFACE_ANCHORS,
    READINESS_STOP_GO_CROSS_ARTIFACT_ANCHORS,
    REQUIRED_FILES,
    REQUIRED_REGRESSION_SUITES,
    SERVER_ONLY_COMMAND_INPUTS,
    validate_documented_regression_suites,
    validate_eligible_calibration_transition,
    validate_joint_readiness_transition,
    validate_minimum_tier_key_claims,
    validate_minimum_tier_publication_transition,
    validate_minimum_tier_evidence_guards,
    validate_minimum_tier_reproducibility_guards,
    validate_minimum_tier_handoff_inventory,
    validate_minimum_tier_comparisons,
    validate_publication_status,
    validate_required_publication_files,
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

    def test_reproducibility_cannot_omit_minimum_tier_identity(self) -> None:
        row = (
            "| `probabilistic_da` | `MISSING` | `NONE` | `PRE_RESULT_ONLY` |"
        )
        mutated = self.reproducibility.replace(row, "", 1)
        self.assertNotEqual(mutated, self.reproducibility)
        with self.assertRaisesRegex(ValueError, "reproducibility guard is incomplete"):
            validate_minimum_tier_reproducibility_guards(
                self.manuscript, mutated
            )

    def test_reproducibility_minimum_tier_identities_match_manuscript(self) -> None:
        validate_minimum_tier_reproducibility_guards(
            self.manuscript, self.reproducibility
        )

    def fully_closed_surfaces(self) -> tuple[str, str, str, str, str]:
        manuscript, ledger, readiness = self.manuscript, self.ledger, self.readiness
        reproducibility = self.reproducibility
        for index, (route, status) in enumerate((
            ("conformal", "CONFORMAL_USEFUL"),
            ("probabilistic_da", "PROBABILISTIC_DA_NEGATIVE"),
            ("independent_deterministic", "PRESENT_INDEPENDENT"),
        )):
            marker = f"{index + 1:064x}"
            manuscript, ledger, readiness = self.transition_surfaces_from(
                manuscript, ledger, readiness, route, status, marker
            )
            old_status, _, old_presentation = MINIMUM_TIER_EVIDENCE_GUARD_ROWS[route]
            old = f"| `{route}` | `{old_status}` | `NONE` | `{old_presentation}` |"
            new = f"| `{route}` | `{status}` | `{marker}` | `DECISION_BEARING` |"
            reproducibility = reproducibility.replace(old, new, 1)
        marker = "a" * 64
        old = "| `eligible_calibration` | `MISSING_ELIGIBLE_RESULT` | `NONE` | `BLOCKED` |"
        new = f"| `eligible_calibration` | `ELIGIBLE` | `{marker}` | `DECISION_BEARING` |"
        manuscript, ledger, readiness, reproducibility = (
            text.replace(old, new, 1)
            for text in (manuscript, ledger, readiness, reproducibility)
        )
        manuscript += (
            f"\nEligible calibration decision: `ELIGIBLE`; compact record: `{marker}`; "
            "claim role: `DECISION_BEARING`.\n"
        )
        reproducibility += (
            f"\nEligible calibration reproducibility identity: `{marker}`; "
            "verification: `HASH_VERIFIED`.\n"
        )
        readiness = readiness.replace(
            "| Eligible spatially preserving calibration | `RESEARCH_PLAN.md` | MISSING_ELIGIBLE_RESULT |",
            "| Eligible spatially preserving calibration | `RESEARCH_PLAN.md` | ELIGIBLE |",
            1,
        ).replace(
            "Publication status: NOT_READY", "Publication status: READY_FOR_HUMAN_REVIEW", 1
        ).replace(
            "Required scientific blockers: an eligible spatially preserving calibration and\n"
            "the remaining minimum-tier comparisons",
            "Required scientific blockers: none",
            1,
        )
        handoff = (PAPER_DIR / "FROZEN_EVALUATION_HANDOFF.md").read_text(encoding="utf-8")
        handoff = handoff.replace(
            "Status: AUTHORIZED_ACTIVE_PENDING_COMPACT_RESULT",
            "Status: COMPLETED_COMPACT_RESULT_RECONCILED",
            1,
        )
        return manuscript, ledger, readiness, reproducibility, handoff

    def transition_surfaces_from(
        self, manuscript: str, ledger: str, readiness: str,
        route: str, status: str, marker: str
    ) -> tuple[str, str, str]:
        originals = (self.manuscript, self.ledger, self.readiness)
        self.manuscript, self.ledger, self.readiness = manuscript, ledger, readiness
        try:
            return self.transition_surfaces(route, status, marker)
        finally:
            self.manuscript, self.ledger, self.readiness = originals

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
        for filename in ("RESEARCH_PLAN.md", "FROZEN_EVALUATION_HANDOFF.md"):
            (fixture_dir / filename).write_bytes((PAPER_DIR / filename).read_bytes())
        for filename in (
            "PAPER_DRAFT.md", "CLAIM_LEDGER.md", "PUBLICATION_READINESS.md"
        ):
            target = fixture_dir / filename
            if not target.exists():
                target.write_bytes((PAPER_DIR / filename).read_bytes())
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

    def test_minimum_tier_handoff_inventory_is_complete(self) -> None:
        validate_minimum_tier_handoff_inventory()

    def test_each_missing_handoff_file_fails_closed(self) -> None:
        for route, inventory in MINIMUM_TIER_HANDOFF_INVENTORY.items():
            for filename in inventory["files"]:
                with self.subTest(route=route, filename=filename):
                    required = tuple(name for name in REQUIRED_FILES if name != filename)
                    with self.assertRaisesRegex(
                        ValueError, f"inventory is incomplete for {route}"
                    ):
                        validate_minimum_tier_handoff_inventory(required_files=required)

    def test_each_missing_handoff_suite_fails_closed(self) -> None:
        for route, inventory in MINIMUM_TIER_HANDOFF_INVENTORY.items():
            for suite in inventory["suites"]:
                with self.subTest(route=route, suite=suite):
                    required = tuple(
                        name for name in REQUIRED_REGRESSION_SUITES if name != suite
                    )
                    with self.assertRaisesRegex(
                        ValueError, f"inventory is incomplete for {route}"
                    ):
                        validate_minimum_tier_handoff_inventory(required_suites=required)

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

    def test_stop_go_closure_bindings_cover_all_normative_artifacts(self) -> None:
        self.assertEqual(
            set(READINESS_STOP_GO_CROSS_ARTIFACT_ANCHORS),
            {
                "RESEARCH_PLAN.md",
                "NEXT_CONFORMAL_BASELINE_CONTRACT.md",
                "NEXT_PROBABILISTIC_DA_COMPARISON_CONTRACT.md",
                "FROZEN_EVALUATION_HANDOFF.md",
            },
        )

    def test_each_weakened_stop_go_closure_binding_fails_closed(self) -> None:
        for filename, anchors in READINESS_STOP_GO_CROSS_ARTIFACT_ANCHORS.items():
            with self.subTest(filename=filename), self.make_fixture(self.audit) as temporary:
                fixture_dir = Path(temporary)
                target = fixture_dir / filename
                text = target.read_text(encoding="utf-8")
                anchor = anchors[0]
                self.assertIn(anchor, text)
                target.write_text(
                    text.replace(anchor, "Weakened closure statement.", 1),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(
                    ValueError,
                    "normative stop/go closure binding is incomplete or weakened",
                ):
                    validate_readiness_blockers(self.readiness, fixture_dir)

    def test_decision_transition_guards_cover_all_three_publication_surfaces(self) -> None:
        self.assertEqual(
            set(READINESS_DECISION_SURFACE_ANCHORS),
            {"PAPER_DRAFT.md", "CLAIM_LEDGER.md", "PUBLICATION_READINESS.md"},
        )

    def test_admission_transition_guards_cover_all_three_routes(self) -> None:
        self.assertEqual(
            set(MINIMUM_TIER_ADMISSION_TRANSITION_ANCHORS),
            {
                "NEXT_CONFORMAL_BASELINE_CONTRACT.md",
                "NEXT_PROBABILISTIC_DA_COMPARISON_CONTRACT.md",
                "FROZEN_EVALUATION_HANDOFF.md",
            },
        )

    def test_each_weakened_admission_transition_boundary_fails_closed(self) -> None:
        for filename, anchors in MINIMUM_TIER_ADMISSION_TRANSITION_ANCHORS.items():
            with self.subTest(filename=filename), self.make_fixture(self.audit) as temporary:
                fixture_dir = Path(temporary)
                target = fixture_dir / filename
                text = target.read_text(encoding="utf-8")
                anchor = anchors[0]
                self.assertIn(anchor, text)
                target.write_text(
                    text.replace(anchor, "Weakened admission transition boundary.", 1),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(
                    ValueError,
                    "admission-to-transition boundary is incomplete or weakened",
                ):
                    validate_readiness_blockers(self.readiness, fixture_dir)

    def test_each_partial_publication_surface_update_fails_closed(self) -> None:
        for filename, anchors in READINESS_DECISION_SURFACE_ANCHORS.items():
            for index, anchor in enumerate(anchors):
                with (
                    self.subTest(filename=filename, anchor=index),
                    self.make_fixture(self.audit) as temporary,
                ):
                    fixture_dir = Path(temporary)
                    target = fixture_dir / filename
                    text = target.read_text(encoding="utf-8")
                    self.assertIn(anchor, text)
                    target.write_text(
                        text.replace(anchor, "Weakened publication transition.", 1),
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(
                        ValueError,
                        "publication decision transition is incomplete or weakened",
                    ):
                        validate_readiness_blockers(self.readiness, fixture_dir)

    def test_minimum_tier_evidence_guards_pass(self) -> None:
        validate_minimum_tier_evidence_guards(
            self.manuscript, self.ledger, self.readiness, self.audit
        )

    def transition_surfaces(self, route: str, status: str, marker: str) -> tuple[str, str, str]:
        old_status, old_record, old_presentation = {
            "conformal": ("MISSING", "NONE", "PRE_RESULT_ONLY"),
            "probabilistic_da": ("MISSING", "NONE", "PRE_RESULT_ONLY"),
            "independent_deterministic": (
                "PRESENT_DEVELOPMENT_ONLY", "NONE", "DEVELOPMENT_ONLY"
            ),
        }[route]
        old = f"| `{route}` | `{old_status}` | `{old_record}` | `{old_presentation}` |"
        new = f"| `{route}` | `{status}` | `{marker}` | `DECISION_BEARING` |"
        surfaces = tuple(
            text.replace(old, new, 1)
            for text in (self.manuscript, self.ledger, self.readiness)
        )
        blocker = {
            "conformal": "Conformal intervals",
            "probabilistic_da": "Probabilistic DA baseline such as EnKF/LETKF",
            "independent_deterministic": "Independent-strength deterministic background and 3D-Var",
        }[route]
        readiness = surfaces[2].replace(
            f"| {blocker} | `MINIMUM_TIER_COMPARISON_AUDIT.md` | {old_status} |",
            f"| {blocker} | `MINIMUM_TIER_COMPARISON_AUDIT.md` | {status} |",
            1,
        )
        return surfaces[0], surfaces[1], readiness

    def test_minimum_tier_positive_and_negative_transitions_are_executable(self) -> None:
        cases = (
            ("conformal", "CONFORMAL_USEFUL"),
            ("conformal", "CONFORMAL_NEGATIVE"),
            ("probabilistic_da", "PROBABILISTIC_DA_USEFUL"),
            ("probabilistic_da", "PROBABILISTIC_DA_NEGATIVE"),
            ("independent_deterministic", "PRESENT_INDEPENDENT"),
            ("independent_deterministic", "INDEPENDENT_NEGATIVE"),
        )
        for index, (route, status) in enumerate(cases):
            with self.subTest(route=route, status=status):
                surfaces = self.transition_surfaces(route, status, f"{index + 1:064x}")
                old_status, _, old_presentation = MINIMUM_TIER_EVIDENCE_GUARD_ROWS[route]
                old = f"| `{route}` | `{old_status}` | `NONE` | `{old_presentation}` |"
                new = (
                    f"| `{route}` | `{status}` | `{index + 1:064x}` | "
                    "`DECISION_BEARING` |"
                )
                reproducibility = self.reproducibility.replace(old, new, 1)
                self.assertNotEqual(reproducibility, self.reproducibility)
                validate_minimum_tier_publication_transition(
                    *surfaces, self.audit, reproducibility
                )

    def test_minimum_tier_outcome_without_reproducibility_update_fails_closed(self) -> None:
        surfaces = self.transition_surfaces(
            "probabilistic_da", "PROBABILISTIC_DA_NEGATIVE", "b" * 64
        )
        with self.assertRaisesRegex(ValueError, "reproducibility transition is not atomic"):
            validate_minimum_tier_publication_transition(
                *surfaces, self.audit, self.reproducibility
            )

    def test_minimum_tier_partial_transition_fails_closed(self) -> None:
        manuscript, _, _ = self.transition_surfaces(
            "conformal", "CONFORMAL_USEFUL", "a" * 64
        )
        with self.assertRaisesRegex(ValueError, "not atomic"):
            validate_minimum_tier_evidence_guards(
                manuscript, self.ledger, self.readiness, self.audit
            )

    def test_minimum_tier_compact_identities_must_match(self) -> None:
        manuscript, ledger, readiness = self.transition_surfaces(
            "probabilistic_da", "PROBABILISTIC_DA_NEGATIVE", "b" * 64
        )
        ledger = ledger.replace("b" * 64, "c" * 64, 1)
        with self.assertRaisesRegex(ValueError, "not atomic"):
            validate_minimum_tier_evidence_guards(
                manuscript, ledger, readiness, self.audit
            )

    def test_minimum_tier_label_only_transition_fails_closed(self) -> None:
        old = "| `conformal` | `MISSING` | `NONE` | `PRE_RESULT_ONLY` |"
        new = "| `conformal` | `CONFORMAL_NEGATIVE` | `NONE` | `DECISION_BEARING` |"
        surfaces = tuple(
            text.replace(old, new, 1)
            for text in (self.manuscript, self.ledger, self.readiness)
        )
        with self.assertRaisesRegex(ValueError, "without compact evidence"):
            validate_minimum_tier_evidence_guards(*surfaces, self.audit)

    def test_minimum_tier_transition_requires_blocker_row_update(self) -> None:
        manuscript, ledger, readiness = self.transition_surfaces(
            "conformal", "CONFORMAL_USEFUL", "d" * 64
        )
        readiness = readiness.replace(
            "| Conformal intervals | `MINIMUM_TIER_COMPARISON_AUDIT.md` | CONFORMAL_USEFUL |",
            "| Conformal intervals | `MINIMUM_TIER_COMPARISON_AUDIT.md` | MISSING |",
            1,
        )
        with self.assertRaisesRegex(ValueError, "readiness blocker row"):
            validate_minimum_tier_evidence_guards(
                manuscript, ledger, readiness, self.audit
            )

    def test_minimum_tier_transition_rejects_wrong_blocker_closure(self) -> None:
        manuscript, ledger, readiness = self.transition_surfaces(
            "probabilistic_da", "PROBABILISTIC_DA_NEGATIVE", "e" * 64
        )
        readiness = readiness.replace(
            "| Probabilistic DA baseline such as EnKF/LETKF | `MINIMUM_TIER_COMPARISON_AUDIT.md` | PROBABILISTIC_DA_NEGATIVE |",
            "| Probabilistic DA baseline such as EnKF/LETKF | `MINIMUM_TIER_COMPARISON_AUDIT.md` | PROBABILISTIC_DA_USEFUL |",
            1,
        )
        with self.assertRaisesRegex(ValueError, "readiness blocker row"):
            validate_minimum_tier_evidence_guards(
                manuscript, ledger, readiness, self.audit
            )

    def test_minimum_tier_transition_status_is_derived_from_matrix(self) -> None:
        manuscript, ledger, readiness = self.transition_surfaces(
            "independent_deterministic", "PRESENT_INDEPENDENT", "f" * 64
        )
        readiness = readiness.replace(
            "Publication status: NOT_READY", "Publication status: READY_FOR_HUMAN_REVIEW", 1
        )
        with self.assertRaisesRegex(ValueError, "publication status"):
            validate_minimum_tier_evidence_guards(
                manuscript, ledger, readiness, self.audit
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

    def test_joint_transition_accepts_only_complete_ready_matrix(self) -> None:
        manuscript, ledger, readiness, reproducibility, handoff = (
            self.fully_closed_surfaces()
        )

    def test_joint_transition_rejects_incomplete_required_file_inventory(self) -> None:
        manuscript, ledger, readiness, reproducibility, handoff = (
            self.fully_closed_surfaces()
        )
        with tempfile.TemporaryDirectory() as directory:
            artifact_dir = Path(directory)
            omitted = "LIMITATION_TRACEABILITY.md"
            for name in REQUIRED_FILES:
                if name != omitted:
                    (artifact_dir / name).write_text("fixture", encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError, "missing required publication files: LIMITATION_TRACEABILITY.md"
            ):
                validate_joint_readiness_transition(
                    manuscript,
                    ledger,
                    readiness,
                    self.audit,
                    reproducibility,
                    handoff,
                    artifact_dir=artifact_dir,
                )

    def test_joint_transition_rejects_missing_closure_contract_file(self) -> None:
        manuscript, ledger, readiness, reproducibility, handoff = (
            self.fully_closed_surfaces()
        )
        with tempfile.TemporaryDirectory() as directory:
            artifact_dir = Path(directory)
            omitted = "NEXT_CONFORMAL_BASELINE_CONTRACT.md"
            for name in REQUIRED_FILES:
                if name != omitted:
                    (artifact_dir / name).write_text("fixture", encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError, "closure contract is missing: NEXT_CONFORMAL_BASELINE_CONTRACT.md"
            ):
                validate_joint_readiness_transition(
                    manuscript,
                    ledger,
                    readiness,
                    self.audit,
                    reproducibility,
                    handoff,
                    artifact_dir=artifact_dir,
                )
        self.assertEqual(
            validate_joint_readiness_transition(
                manuscript, ledger, readiness, self.audit, reproducibility, handoff
            ),
            "READY_FOR_HUMAN_REVIEW",
        )

    def test_required_python_artifacts_reject_invalid_syntax(self) -> None:
        manuscript, ledger, readiness, reproducibility, handoff = (
            self.fully_closed_surfaces()
        )
        with tempfile.TemporaryDirectory() as directory:
            artifact_dir = Path(directory)
            for name in REQUIRED_FILES:
                source = (PAPER_DIR / name).read_bytes()
                (artifact_dir / name).write_bytes(source)
            (artifact_dir / "rank_coherent_reference.py").write_text(
                "def broken(:\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(
                ValueError,
                "required Python artifact has invalid syntax: rank_coherent_reference.py",
            ):
                validate_joint_readiness_transition(
                    manuscript,
                    ledger,
                    readiness,
                    self.audit,
                    reproducibility,
                    handoff,
                    artifact_dir=artifact_dir,
                )

    def test_required_python_artifacts_reject_false_executable_oracle(self) -> None:
        manuscript, ledger, readiness, reproducibility, handoff = (
            self.fully_closed_surfaces()
        )
        with tempfile.TemporaryDirectory() as directory:
            artifact_dir = Path(directory)
            for name in REQUIRED_FILES:
                source = (PAPER_DIR / name).read_bytes()
                (artifact_dir / name).write_bytes(source)
            (artifact_dir / "rank_coherent_reference.py").write_text(
                'print("rank-coherent reference checks: FALSE")\n', encoding="utf-8"
            )
            with self.assertRaisesRegex(
                ValueError, "rank-coherent executable oracle emitted unexpected stdout"
            ):
                validate_joint_readiness_transition(
                    manuscript,
                    ledger,
                    readiness,
                    self.audit,
                    reproducibility,
                    handoff,
                    artifact_dir=artifact_dir,
                )

    def test_required_python_artifacts_reject_nonzero_executable_oracle(self) -> None:
        manuscript, ledger, readiness, reproducibility, handoff = (
            self.fully_closed_surfaces()
        )
        with tempfile.TemporaryDirectory() as directory:
            artifact_dir = Path(directory)
            for name in REQUIRED_FILES:
                (artifact_dir / name).write_bytes((PAPER_DIR / name).read_bytes())
            (artifact_dir / "rank_coherent_reference.py").write_text(
                'raise SystemExit(7)\n', encoding="utf-8"
            )
            with self.assertRaisesRegex(
                ValueError, "rank-coherent executable oracle exited nonzero"
            ):
                validate_joint_readiness_transition(
                    manuscript, ledger, readiness, self.audit, reproducibility,
                    handoff, artifact_dir=artifact_dir,
                )

    def test_required_python_artifacts_reject_extra_oracle_stdout(self) -> None:
        manuscript, ledger, readiness, reproducibility, handoff = (
            self.fully_closed_surfaces()
        )
        with tempfile.TemporaryDirectory() as directory:
            artifact_dir = Path(directory)
            for name in REQUIRED_FILES:
                (artifact_dir / name).write_bytes((PAPER_DIR / name).read_bytes())
            (artifact_dir / "rank_coherent_reference.py").write_text(
                'print("debug")\nprint("rank-coherent reference checks: PASS")\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError, "rank-coherent executable oracle emitted unexpected stdout"
            ):
                validate_joint_readiness_transition(
                    manuscript, ledger, readiness, self.audit, reproducibility,
                    handoff, artifact_dir=artifact_dir,
                )

    def test_required_python_artifacts_reject_oracle_timeout(self) -> None:
        manuscript, ledger, readiness, reproducibility, handoff = (
            self.fully_closed_surfaces()
        )
        with tempfile.TemporaryDirectory() as directory:
            artifact_dir = Path(directory)
            for name in REQUIRED_FILES:
                (artifact_dir / name).write_bytes((PAPER_DIR / name).read_bytes())
            (artifact_dir / "rank_coherent_reference.py").write_text(
                'import time\ntime.sleep(10)\n', encoding="utf-8"
            )
            with self.assertRaisesRegex(
                ValueError, "rank-coherent executable oracle timed out"
            ):
                validate_joint_readiness_transition(
                    manuscript, ledger, readiness, self.audit, reproducibility,
                    handoff, artifact_dir=artifact_dir,
                )

    def test_joint_transition_rejects_ready_with_any_open_blocker(self) -> None:
        manuscript, ledger, readiness, reproducibility, handoff = (
            self.fully_closed_surfaces()
        )
        readiness = readiness.replace(
            "| Conformal intervals | `MINIMUM_TIER_COMPARISON_AUDIT.md` | CONFORMAL_USEFUL |",
            "| Conformal intervals | `MINIMUM_TIER_COMPARISON_AUDIT.md` | MISSING |",
            1,
        )
        with self.assertRaisesRegex(ValueError, "readiness blocker row"):
            validate_joint_readiness_transition(
                manuscript, ledger, readiness, self.audit, reproducibility, handoff
            )

    def test_joint_transition_requires_none_blocker_declaration(self) -> None:
        manuscript, ledger, readiness, reproducibility, handoff = (
            self.fully_closed_surfaces()
        )
        readiness = readiness.replace(
            "Required scientific blockers: none",
            "Required scientific blockers: stale blocker",
            1,
        )
        with self.assertRaisesRegex(ValueError, "requires no scientific blockers"):
            validate_joint_readiness_transition(
                manuscript, ledger, readiness, self.audit, reproducibility, handoff
            )

    def test_joint_transition_rejects_active_evaluation_handoff(self) -> None:
        manuscript, ledger, readiness, reproducibility, _ = self.fully_closed_surfaces()
        active_handoff = (PAPER_DIR / "FROZEN_EVALUATION_HANDOFF.md").read_text(
            encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "active evaluation"):
            validate_joint_readiness_transition(
                manuscript, ledger, readiness, self.audit, reproducibility, active_handoff
            )

    def test_joint_transition_rejects_missing_limitation_after_closure(self) -> None:
        manuscript, ledger, readiness, reproducibility, handoff = (
            self.fully_closed_surfaces()
        )
        manuscript = manuscript.replace(
            "residual upper-tail undercoverage remains",
            "residual upper-tail behavior is documented",
            1,
        )
        with self.assertRaisesRegex(ValueError, "L6 is absent from Section 7"):
            validate_joint_readiness_transition(
                manuscript, ledger, readiness, self.audit, reproducibility, handoff
            )

    def test_joint_transition_rejects_presentation_unit_mismatch_after_closure(self) -> None:
        manuscript, ledger, readiness, reproducibility, handoff = (
            self.fully_closed_surfaces()
        )
        manuscript = manuscript.replace(
            "Frozen-mechanism table; paired-diagnostic table; Figure 1",
            "Frozen-mechanism table; Figure 1",
            1,
        )
        with self.assertRaisesRegex(ValueError, "decision-bearing presentation audit mismatch"):
            validate_joint_readiness_transition(
                manuscript, ledger, readiness, self.audit, reproducibility, handoff
            )

    def test_joint_transition_rejects_compact_identity_mismatch_after_closure(self) -> None:
        manuscript, ledger, readiness, reproducibility, handoff = (
            self.fully_closed_surfaces()
        )
        marker = f"{1:064x}"
        manuscript = manuscript.replace(marker, "f" * 64, 1)
        with self.assertRaisesRegex(
            ValueError, "transition is not atomic across publication surfaces"
        ):
            validate_joint_readiness_transition(
                manuscript, ledger, readiness, self.audit, reproducibility, handoff
            )

    def test_joint_transition_rejects_claim_status_mismatch_after_closure(self) -> None:
        manuscript, ledger, readiness, reproducibility, handoff = (
            self.fully_closed_surfaces()
        )
        manuscript = manuscript.replace(
            "| C38 | Section 8 later-mechanism family matrix",
            "| C1 | Section 8 later-mechanism family matrix",
            1,
        )
        with self.assertRaisesRegex(ValueError, "Rejected claim is missing"):
            validate_joint_readiness_transition(
                manuscript, ledger, readiness, self.audit, reproducibility, handoff
            )

    def test_joint_transition_rejects_citation_mismatch_after_closure(self) -> None:
        manuscript, ledger, readiness, reproducibility, handoff = (
            self.fully_closed_surfaces()
        )
        manuscript = manuscript.replace("[8]", "[7]", 1)
        with self.assertRaisesRegex(ValueError, "reference/citation mismatch"):
            validate_joint_readiness_transition(
                manuscript, ledger, readiness, self.audit, reproducibility, handoff
            )

    def test_joint_transition_rejects_reference_audit_mismatch_after_closure(self) -> None:
        manuscript, ledger, readiness, reproducibility, handoff = (
            self.fully_closed_surfaces()
        )
        traceability = (PAPER_DIR / "REFERENCE_TRACEABILITY.md").read_text(
            encoding="utf-8"
        ).replace("10.1002/qj.2270", "10.1002/qj.invalid", 1)
        with self.assertRaisesRegex(ValueError, "reference 3 identity mismatch"):
            validate_joint_readiness_transition(
                manuscript,
                ledger,
                readiness,
                self.audit,
                reproducibility,
                handoff,
                reference_traceability=traceability,
            )

    def test_joint_transition_rejects_regression_command_omission_after_closure(self) -> None:
        manuscript, ledger, readiness, reproducibility, handoff = (
            self.fully_closed_surfaces()
        )
        suite = REQUIRED_REGRESSION_SUITES[-1]
        reproducibility = reproducibility.replace(f"  {suite}\n", "", 1)
        with self.assertRaisesRegex(ValueError, "differ from the required"):
            validate_joint_readiness_transition(
                manuscript, ledger, readiness, self.audit, reproducibility, handoff
            )

    def test_joint_transition_rejects_missing_figure_after_closure(self) -> None:
        manuscript, ledger, readiness, reproducibility, handoff = (
            self.fully_closed_surfaces()
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "linked figure does not exist"):
                validate_joint_readiness_transition(
                    manuscript,
                    ledger,
                    readiness,
                    self.audit,
                    reproducibility,
                    handoff,
                    figure_dir=Path(directory),
                )

    def test_joint_transition_rejects_figure_semantic_mutation_after_closure(self) -> None:
        manuscript, ledger, readiness, reproducibility, handoff = (
            self.fully_closed_surfaces()
        )
        with tempfile.TemporaryDirectory() as directory:
            figure_dir = Path(directory)
            for relative_name, anchors in FIGURE_TEXT_ANCHORS.items():
                target = figure_dir / relative_name
                target.parent.mkdir(parents=True, exist_ok=True)
                text = (PAPER_DIR / relative_name).read_text(encoding="utf-8")
                if relative_name == "figures/calibration_summary.svg":
                    text = text.replace(anchors[0], "semantic-anchor-removed", 1)
                target.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing semantic anchors"):
                validate_joint_readiness_transition(
                    manuscript,
                    ledger,
                    readiness,
                    self.audit,
                    reproducibility,
                    handoff,
                    figure_dir=figure_dir,
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
        with self.assertRaisesRegex(ValueError, "closure conditions differ"):
            validate_readiness_blockers(mutated, PAPER_DIR)

    def test_readiness_cannot_weaken_nonempty_closure_condition(self) -> None:
        mutated = self.readiness.replace(
            "One frozen candidate passes every mandatory no-compensation gate family",
            "One frozen candidate improves at least one mandatory gate family",
            1,
        )
        self.assertNotEqual(mutated, self.readiness)
        with self.assertRaisesRegex(ValueError, "closure conditions differ"):
            validate_readiness_blockers(mutated, PAPER_DIR)

    def test_readiness_cannot_weaken_conformal_closure_condition(self) -> None:
        mutated = self.readiness.replace(
            "The corresponding normative comparison row becomes decision-bearing "
            "after valid trusted execution",
            "The corresponding normative comparison contract is available",
            1,
        )
        self.assertNotEqual(mutated, self.readiness)
        with self.assertRaisesRegex(ValueError, "closure conditions differ"):
            validate_readiness_blockers(mutated, PAPER_DIR)

    def test_readiness_cannot_weaken_probabilistic_da_closure_condition(self) -> None:
        closure = (
            "The corresponding normative comparison row becomes decision-bearing "
            "after valid trusted execution"
        )
        first = self.readiness.index(closure)
        second = self.readiness.index(closure, first + len(closure))
        mutated = (
            self.readiness[:second]
            + "The corresponding normative comparison contract is available"
            + self.readiness[second + len(closure):]
        )
        self.assertNotEqual(mutated, self.readiness)
        with self.assertRaisesRegex(ValueError, "closure conditions differ"):
            validate_readiness_blockers(mutated, PAPER_DIR)

    def test_readiness_cannot_weaken_independent_strength_closure_condition(self) -> None:
        mutated = self.readiness.replace(
            "A frozen common-information comparison supplies evidence at the manuscript's required independent strength",
            "A development-only deterministic comparison is documented",
            1,
        )
        self.assertNotEqual(mutated, self.readiness)
        with self.assertRaisesRegex(ValueError, "closure conditions differ"):
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
