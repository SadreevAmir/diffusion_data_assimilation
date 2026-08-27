#!/usr/bin/env python3
"""Fail closed on broken links and contradictory publication metadata."""

from __future__ import annotations

import ast
import hashlib
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


PAPER_DIR = Path(__file__).resolve().parent

REQUIRED_REGRESSION_SUITES = (
    "paper.test_rank_coherent_runner_prototype",
    "paper.test_rank_coherent_adapter_parity",
    "paper.test_validate_rank_coherent_admission",
    "paper.test_validate_rank_coherent_manifest",
    "paper.test_publication_immutable_identities",
    "paper.test_publication_empirical_traceability",
    "paper.test_minimum_tier_comparison_audit",
    "paper.test_publication_figure_generators",
    "paper.test_publication_compact_payload_schemas",
    "paper.test_publication_reference_traceability",
    "paper.test_publication_limitation_traceability",
    "paper.test_publication_claim_status_consistency",
    "paper.test_score_aware_reconciliation_consistency",
    "paper.test_validate_server_only_manifest",
    "paper.test_server_only_consumer_cli",
    "paper.test_atomic_publish",
    "paper.test_raw_member_reweighting_reference",
    "paper.test_score_aware_raw_reweighting_reference",
    "paper.test_validate_score_aware_raw_reweighting_admission",
    "paper.test_validate_score_aware_compact_outputs",
)
REGRESSION_COMMAND = re.compile(
    r"python3 -m unittest -v \\\n(?P<body>(?:  paper\.[a-z0-9_]+(?: \\\n|\n))+)",
)
SERVER_ONLY_COMMAND_INPUTS = {
    "case_level_artifacts_long_form": (
        "joint_existing_ensemble_calibration_audit_valid",
        "per_case_metrics.csv",
        "per_case_metrics.csv`: exactly 160 rows; columns `target_date`, `fold`, "
        "`method` and the full proper-score, rank, boundary and spatial diagnostic "
        "family; unique ISO `target_date`/`method` pairs; exactly 40 dates and both "
        "exact raw/global-spread method labels on every date",
        "metadata.json`: `experiment_id == joint_existing_ensemble_calibration_audit_valid`; "
        "artifact manifest binds `per_case_metrics.csv` by SHA-256"
    ),
    "calibration_summary_figure": (
        "joint_crossfit_spread_calibration_valid",
        "aggregate_case_mean_metrics.json",
        "aggregate_case_mean_metrics.json`: one-row JSON list; `num_cases == 40`; "
        "finite raw and corrected values for ordinary CRPS, fair CRPS, spread-skill "
        "ratio and all four interval diagnostics",
        "metadata.json`: `experiment_id == joint_crossfit_spread_calibration_valid`; "
        "artifact manifest binds `aggregate_case_mean_metrics.json` by SHA-256"
    ),
    "joint_gate_figure": (
        "joint_existing_ensemble_mean_preserving_projected_spread_valid",
        "aggregate_case_mean_metrics.json",
        "aggregate_case_mean_metrics.json`: exact reviewed candidate identifier; "
        "exactly two full-region method rows; 40 finite cases per method; finite "
        "gate metrics and Boolean `overall_eligible == false`",
        "metadata.json`: `experiment_id == joint_existing_ensemble_mean_preserving_projected_spread_valid`; "
        "artifact manifest binds `aggregate_case_mean_metrics.json` by SHA-256"
    ),
}
SERVER_ONLY_COMMAND_ROW = re.compile(
    r"^\| `(?P<command>[a-z0-9_]+)` \| `(?P<execution>[A-Z_]+)` \| "
    r"`(?P<producer>[a-z0-9_]+)` \| `(?P<artifact>[a-z0-9_.]+)` \| "
    r"`(?P<schema>[^\n]+?) \| `(?P<manifest>[^\n]+?) \|$",
    re.MULTILINE,
)


def validate_documented_regression_suites(text: str) -> None:
    """Require the handoff command to match the executable suite contract exactly."""
    matches = list(REGRESSION_COMMAND.finditer(text))
    require(len(matches) == 1, "reproducibility must document one regression command")
    observed = tuple(
        re.findall(r"^  (paper\.[a-z0-9_]+)", matches[0]["body"], re.MULTILINE)
    )
    require(
        observed == REQUIRED_REGRESSION_SUITES,
        "documented regression suites differ from the required executable set",
    )


def validate_server_only_command_inputs(text: str) -> None:
    """Keep external compact-input generators separate from local oracles."""
    rows = list(SERVER_ONLY_COMMAND_ROW.finditer(text))
    observed = {
        row["command"]: (
            row["execution"], row["producer"], row["artifact"],
            row["schema"], row["manifest"],
        )
        for row in rows
    }
    expected = {
        command: ("SERVER_ONLY", producer, artifact, schema, manifest)
        for command, (producer, artifact, schema, manifest)
        in SERVER_ONLY_COMMAND_INPUTS.items()
    }
    require(
        len(rows) == len(observed) == len(expected),
        "server-only command matrix is incomplete or contains duplicates",
    )
    require(
        observed == expected,
        "server-only command provenance or compact-input contract differs from contract",
    )
    require(
        "rank_coherent_reference.py` command and the\npublication regression/integrity commands below are `LOCAL_ORACLE`" in text,
        "local-oracle execution boundary is missing",
    )


def validate_markdown_tables(text: str, filename: str) -> int:
    """Fail closed on malformed pipe tables that can silently misrender."""
    lines = text.splitlines()
    table_count = 0
    index = 0
    while index < len(lines) - 1:
        header = lines[index]
        separator = lines[index + 1]
        if not (header.startswith("|") and separator.startswith("|")):
            index += 1
            continue

        header_cells = [cell.strip() for cell in header.strip("|").split("|")]
        separator_cells = [cell.strip() for cell in separator.strip("|").split("|")]
        if not separator_cells or not all(
            re.fullmatch(r":?-{3,}:?", cell) for cell in separator_cells
        ):
            index += 1
            continue

        require(
            len(header_cells) == len(separator_cells),
            f"{filename} table header/separator width mismatch at line {index + 1}",
        )
        require(
            all(header_cells),
            f"{filename} table has an empty header cell at line {index + 1}",
        )
        table_count += 1
        index += 2
        row_count = 0
        while index < len(lines) and lines[index].startswith("|"):
            row_cells = [cell.strip() for cell in lines[index].strip("|").split("|")]
            require(
                len(row_cells) == len(header_cells),
                f"{filename} table row width mismatch at line {index + 1}",
            )
            require(
                all(row_cells),
                f"{filename} table has an empty body cell at line {index + 1}",
            )
            row_count += 1
            index += 1
        require(
            row_count > 0,
            f"{filename} table has no body rows at line {index + 1}",
        )
    return table_count
REQUIRED_FILES = (
    "PAPER_DRAFT.md",
    "CLAIM_LEDGER.md",
    "REPRODUCIBILITY.md",
    "RESEARCH_PLAN.md",
    "PUBLICATION_READINESS.md",
    "MINIMUM_TIER_COMPARISON_AUDIT.md",
    "NEXT_CONFORMAL_BASELINE_CONTRACT.md",
    "NEXT_PROBABILISTIC_DA_COMPARISON_CONTRACT.md",
    "NEXT_BASELINE_CONTRACT.md",
    "NEXT_METHOD_CONTRACT.md",
    "NEXT_RANK_COHERENT_CONTRACT.md",
    "NEXT_RAW_MEMBER_REWEIGHTING_CONTRACT.md",
    "raw_member_reweighting_reference.py",
    "test_raw_member_reweighting_reference.py",
    "NEXT_SCORE_AWARE_RAW_REWEIGHTING_CONTRACT.md",
    "SCORE_AWARE_RESULT_RECONCILIATION.md",
    "score_aware_raw_reweighting_reference.py",
    "test_score_aware_raw_reweighting_reference.py",
    "validate_score_aware_raw_reweighting_admission.py",
    "test_validate_score_aware_raw_reweighting_admission.py",
    "validate_score_aware_compact_outputs.py",
    "test_validate_score_aware_compact_outputs.py",
    "RANK_COHERENT_RUNNER_REVIEW_CHECKLIST.md",
    "RANK_COHERENT_CONTROLLER_HANDOFF.md",
    "FROZEN_EVALUATION_HANDOFF.md",
    "AMENDED_PRIMARY_EVALUATION_CONTRACT.md",
    "check_publication_artifacts.py",
    "make_calibration_summary_figure.py",
    "make_case_level_artifacts.py",
    "make_joint_gate_figure.py",
    "analog_residual_reference.py",
    "rank_coherent_reference.py",
    "rank_coherent_runner_prototype.py",
    "test_rank_coherent_runner_prototype.py",
    "rank_coherent_adapter_parity.py",
    "test_rank_coherent_adapter_parity.py",
    "validate_rank_coherent_admission.py",
    "test_validate_rank_coherent_admission.py",
    "rank_coherent_admission_manifest.json",
    "validate_rank_coherent_manifest.py",
    "test_validate_rank_coherent_manifest.py",
    "test_publication_immutable_identities.py",
    "test_publication_empirical_traceability.py",
    "test_minimum_tier_comparison_audit.py",
    "test_publication_figure_generators.py",
    "test_publication_compact_payload_schemas.py",
    "test_publication_reference_traceability.py",
    "REFERENCE_TRACEABILITY.md",
    "test_publication_limitation_traceability.py",
    "LIMITATION_TRACEABILITY.md",
    "test_publication_claim_status_consistency.py",
    "test_score_aware_reconciliation_consistency.py",
    "RANK_COHERENT_ADAPTER_SPEC.md",
    "NEXT_GENERATIVE_METHOD_CONTRACT.md",
    "LATENT_TEMPERATURE_RESULT_RECONCILIATION.md",
    "LOCKED_MC_DROPOUT_RESULT_RECONCILIATION.md",
    "EXTERNAL_PRIMARY_RESULT_RECONCILIATION.md",
    "guidance_mixture_reference.py",
    "deep_ensemble_reference.py",
    "validate_server_only_manifest.py",
    "test_validate_server_only_manifest.py",
)

REFERENCE_TRACEABILITY_ROWS = {
    1: ("10.48550/arXiv.2006.11239", "diffusion probabilistic models"),
    2: ("10.48550/arXiv.2210.02747", "Flow Matching"),
    3: ("10.1002/qj.2270", "fair ensemble scores"),
    4: ("10.1198/016214506000001437", "proper scoring rule"),
    5: ("10.1214/13-STS443", "ECC-Q"),
    6: ("10.1002/2015GL067232", "IIEE"),
    7: ("10.1080/01621459.2017.1307116", "split conformal inference"),
    8: ("10.1016/j.physd.2006.11.008", "LETKF"),
}

LIMITATION_TRACEABILITY_ROWS = {
    "L1": ("one legacy checkpoint, one sampling seed, ten members and 40 development dates", ("C11", "C35"), None),
    "L2": ("not an independent temporal generalization estimate", ("C3", "C15"), None),
    "L3": ("not direct satellite SIC retrievals", ("C2",), None),
    "L4": ("does not establish casewise or fieldwise coverage", ("C13",), None),
    "L5": ("Clipping complicates bounded mean comparisons", ("C12", "C19"), None),
    "L6": ("residual upper-tail undercoverage remains", ("C13", "C27"), None),
    "L7": ("No independent comparison with 3D-Var is claimed", ("C9", "C17"), "C9"),
    "L8": ("Generalization across checkpoints, seeds, ensemble sizes, regions or observation systems remains unverified", ("C10",), "C10"),
}


def validate_limitation_traceability(
    manuscript: str, claim_ledger: str, traceability: str
) -> None:
    """Require complete Section 7 coverage and exact claim-ledger support."""
    match = re.search(
        r"^## 7\. Limitations\n(?P<body>.*?)^## 8\.", manuscript, re.MULTILINE | re.DOTALL
    )
    require(match is not None, "manuscript limitation section is missing")
    limitation_section = match.group("body")
    normalized_limitation_section = re.sub(r"\s+", " ", limitation_section)
    rows: dict[str, tuple[str, tuple[str, ...], str]] = {}
    for line in traceability.splitlines():
        row = re.fullmatch(
            r"\| (L[1-9][0-9]*) \| `([^`]+)` \| ([^|]+) \| ([^|]+) \|", line
        )
        if not row:
            continue
        limitation_id = row.group(1)
        require(limitation_id not in rows, f"duplicate limitation row: {limitation_id}")
        claim_ids = tuple(re.findall(r"`(C[1-9][0-9]*)`", row.group(3)))
        rows[limitation_id] = (row.group(2), claim_ids, row.group(4))
    require(set(rows) == set(LIMITATION_TRACEABILITY_ROWS), "limitation traceability must cover exactly L1--L8")
    for limitation_id, (anchor, claim_ids, absence_id) in LIMITATION_TRACEABILITY_ROWS.items():
        observed_anchor, observed_claim_ids, support_kind = rows[limitation_id]
        require(observed_anchor == anchor, f"{limitation_id} anchor mismatch")
        require(
            anchor in normalized_limitation_section,
            f"{limitation_id} is absent from Section 7",
        )
        require(observed_claim_ids == claim_ids, f"{limitation_id} claim mapping mismatch")
        for claim_id in claim_ids:
            require(f"| {claim_id} |" in claim_ledger, f"{limitation_id} references missing {claim_id}")
        if absence_id:
            require(f"`{absence_id}` is `Unknown`" in support_kind, f"{limitation_id} lacks an explicit evidence-absence record")
            ledger_row = next((line for line in claim_ledger.splitlines() if line.startswith(f"| {absence_id} |")), "")
            require(
                "| Unknown" in ledger_row,
                f"{absence_id} is not an Unknown evidence record",
            )
    require("cover the complete limitation inventory in Section 7" in traceability, "limitation audit lacks a completeness boundary")


def validate_claim_status_consistency(manuscript: str, claim_ledger: str) -> None:
    """Keep every unknown or rejected ledger row aligned with manuscript maps."""
    statuses: dict[str, str] = {}
    for line in claim_ledger.splitlines():
        match = re.match(r"\| (C[1-9][0-9]*) \| [^|]+ \| ([^|]+) \|", line)
        if match:
            statuses[match.group(1)] = match.group(2).strip()

    unknown = {claim_id for claim_id, status in statuses.items() if status.startswith("Unknown")}
    rejected = {claim_id for claim_id, status in statuses.items() if status.startswith("Rejected")}
    require(unknown, "claim ledger has no explicit Unknown records")
    require(rejected, "claim ledger has no explicit Rejected records")

    scope_match = re.search(
        r"\| Publication scope and unsupported generalization claims \| (?P<ids>[^|]+) \|",
        manuscript,
    )
    require(scope_match is not None, "manuscript lacks the unsupported-claim scope row")
    scope_ids = set(re.findall(r"C[1-9][0-9]*", scope_match.group("ids")))
    require(unknown <= scope_ids, "Unknown claim is missing from the manuscript scope row")

    empirical_match = re.search(
        r"^### Empirical evidence traceability\n(?P<body>.*?)^### Final frozen development fallback",
        manuscript,
        re.MULTILINE | re.DOTALL,
    )
    require(empirical_match is not None, "manuscript empirical traceability section is missing")
    empirical_rows: dict[str, str] = {}
    for line in empirical_match.group("body").splitlines():
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != 4 or not re.fullmatch(r"C[1-9][0-9]*(?:, C[1-9][0-9]*)*", cells[0]):
            continue
        for claim_id in re.findall(r"C[1-9][0-9]*", cells[0]):
            require(claim_id not in empirical_rows, f"duplicate empirical status mapping: {claim_id}")
            empirical_rows[claim_id] = " ".join(cells[1:]).lower()

    require(not unknown & set(empirical_rows), "Unknown claim was promoted into the empirical evidence table")
    require(not rejected - set(empirical_rows), "Rejected claim is missing from the empirical evidence table")
    decision_terms = ("negative", "reject", "fail", "insufficient", "decision rule")
    weak_rows = {
        claim_id for claim_id in rejected
        if not any(term in empirical_rows[claim_id] for term in decision_terms)
    }
    require(not weak_rows, "Rejected claim lacks an explicit negative decision interpretation")


def validate_reference_traceability(text: str) -> None:
    """Require one exact, support-limited traceability row per bibliography item."""
    rows: dict[int, tuple[str, str]] = {}
    for line in text.splitlines():
        match = re.fullmatch(
            r"\| ([1-9][0-9]*) \| `([^`]+)` \| ([^|]+) \| ([^|]+) \|", line
        )
        if match:
            number = int(match.group(1))
            require(number not in rows, f"duplicate reference traceability row: {number}")
            rows[number] = (match.group(2), match.group(3) + match.group(4))
    require(set(rows) == set(REFERENCE_TRACEABILITY_ROWS),
            "reference traceability must cover exactly references 1--8")
    for number, (identity, statement_anchor) in REFERENCE_TRACEABILITY_ROWS.items():
        observed_identity, prose = rows[number]
        require(observed_identity == identity, f"reference {number} identity mismatch")
        require(statement_anchor in prose, f"reference {number} support mapping mismatch")
    require("does not support project-specific empirical values" in text,
            "reference audit lacks an explicit empirical-claim boundary")
LOCKED_DROPOUT_RECONCILIATION_ANCHORS = (
    "Status: NEGATIVE_DECISION_RECORDED_QUANTITATIVE_RECONCILIATION_PENDING",
    "gate.overall_eligible=false",
    "not a\ncomplete publication evidence unit",
    "must not infer a failed family or\na quantitative effect size",
    "locked_mc_dropout_p010_gate_valid",
    "locked_mc_dropout_p010_sampling_valid",
    "locked_mc_dropout_p010_final_ema_ensemble",
    "exactly 40 completed cases and ten finite members per case",
    "no_compensation_across_families=true",
    "analysis_fair_crps",
    "analysis_crps",
    "exactly fourteen admitted training-time\n  dropout layers",
    "271828000 + 140*case + 14*member + layer",
    "all 5,600 frozen-mask checks passing",
    "update all five files in one change",
    "python3 paper/check_publication_artifacts.py",
    "at least 3% lower than raw",
    "does not establish independent\ngeneralization",
    "cannot authorize\npost-hoc tuning",
    "Wrapper-only recovery boundary",
    "cases_file=cases.json",
    "does not authorize a GPU retry",
    "trusted server-CPU finalizer",
    "all sample hashes and finiteness",
)

LATENT_RECONCILIATION_ANCHORS = (
    "Status: RECONCILED_NEGATIVE",
    "gate.overall_eligible=false",
    "`0.0584905850`/`0.0631177443`",
    "`[0.0014355657,0.0075956683]`",
    "`[-0.0005197500,0.0082640843]`",
    "`0.0621082810`/`0.0684128432`",
    "Reliability passes; proper-\nscore, boundary and spatial/physical families fail",
    "latent_temperature_1p30_gate_retry1",
    "latent_temperature_1p30_sampling_retry1",
    "integrity-checked, server-side recovery",
    "exactly 40 completed cases and ten finite members per case",
    "gate.overall_eligible",
    "analysis_fair_crps",
    "analysis_crps",
    "base and\n  scaled latent-hash checks",
    "update all four files in one change",
    "python3 paper/check_publication_artifacts.py",
    "at least 3% lower than raw",
    "predeclared locked-MC-dropout pair",
    "do not change\nthe temperature",
)
CURRENT_DECISION_ANCHORS = (
    "Required scientific blockers: an eligible spatially preserving calibration and\nthe remaining minimum-tier comparisons",
    "Controller readiness: NOT_READY",
    "Scientific primary reconciliation: COMPLETE_NEGATIVE",
    "Normalized mean\nrank also worsens from `0.23610946912844127` raw to approximately `0.225`",
    "The package is not waiting on clean-checkpoint training.",
    "single frozen latent-temperature construction at scale `1.30`",
    "complete compact payload from `latent_temperature_1p30_gate_retry1` has\nbeen reconciled",
    "The result is negative: `overall_eligible=false`",
    "reliability\npasses, while proper-score, boundary and spatial/physical families fail",
    "exact frozen candidate\n`locked_mc_dropout_p010_final_ema_ensemble` with `overall_eligible=false`",
    "controller-recorded Boolean decision from the still\nmissing quantitative publication payload",
    "no family-specific or effect-size claim is admitted",
    "completed negative dropout result closes only this frozen candidate",
    "Claim C39 now records the reconciled negative decision from\nthe sole admissible compact payload",
    "cannot be reverted to a pending outcome\nor strengthened into a positive generalization claim",
    "already predeclared locked-MC-dropout\nfallback, unchanged",
    "latent_temperature_1p30_gate_retry1",
    "dependent unchanged CPU gate is\ncomplete and reconciled as the negative result above",
    "locked-MC-dropout chain is complete and negative",
    "activates the already frozen iid calendar\nglobal-bias fallback",
)
MINIMUM_TIER_OUTCOME_MATRIX_ANCHORS = (
    "The interpretation of the two frozen missing-family contracts is fixed before",
    "`CONFORMAL_USEFUL` | `PROBABILISTIC_DA_USEFUL`",
    "`CONFORMAL_USEFUL` | `PROBABILISTIC_DA_NEGATIVE`",
    "`CONFORMAL_NEGATIVE` | `PROBABILISTIC_DA_USEFUL`",
    "`CONFORMAL_NEGATIVE` | `PROBABILISTIC_DA_NEGATIVE`",
    "minimum-tier row closure and learned-joint\ncalibration eligibility remain logically independent",
    "`COMPARATOR_INVALID` or an invalid conformal execution leaves the\ncorresponding evidence row `MISSING`",
)
MINIMUM_TIER_CLAIM_CONSISTENCY_ANCHORS = {
    "PAPER_DRAFT.md": (
        "The deterministic comparison is not a third absent family",
        "without upgrading the deterministic evidence or selecting a calibrated\nensemble",
    ),
    "CLAIM_LEDGER.md": (
        "conformal and probabilistic-DA families remain absent; the deterministic comparison is present only as development evidence",
    ),
    "PUBLICATION_READINESS.md": (
        "Exactly two baseline-family result rows are `MISSING`",
        "The deterministic row is `PRESENT_DEVELOPMENT_ONLY`",
    ),
    "REPRODUCIBILITY.md": (
        "conformal and probabilistic DA are the only `MISSING` result rows",
        "deterministic comparison is `PRESENT_DEVELOPMENT_ONLY`",
    ),
}
READINESS_BLOCKERS = {
    "Eligible spatially preserving calibration": (
        "RESEARCH_PLAN.md",
        "MISSING_ELIGIBLE_RESULT",
    ),
    "Conformal intervals": ("MINIMUM_TIER_COMPARISON_AUDIT.md", "MISSING"),
    "Probabilistic DA baseline such as EnKF/LETKF": (
        "MINIMUM_TIER_COMPARISON_AUDIT.md",
        "MISSING",
    ),
    "Independent-strength deterministic background and 3D-Var": (
        "MINIMUM_TIER_COMPARISON_AUDIT.md",
        "PRESENT_DEVELOPMENT_ONLY",
    ),
}
READINESS_BLOCKER_ROW = re.compile(
    r"^\| (?P<blocker>[^|]+?) \| `(?P<source>RESEARCH_PLAN\.md|MINIMUM_TIER_COMPARISON_AUDIT\.md)` \| "
    r"(?P<status>[A-Z_]+) \| (?P<closure>[^|]*?) \|$",
    re.MULTILINE,
)
READINESS_CLOSURE_ROUTES = {
    "Eligible spatially preserving calibration": (
        "NEXT_RANK_COHERENT_CONTRACT.md",
        "PAPER_DRAFT.md:Section 6 mechanism result and no-compensation decision",
        "CLAIM_LEDGER.md:new decision-bearing calibration claim",
    ),
    "Conformal intervals": (
        "NEXT_CONFORMAL_BASELINE_CONTRACT.md",
        "PAPER_DRAFT.md:Section 3 frozen outcome matrix",
        "CLAIM_LEDGER.md:C17 plus a decision-bearing conformal claim",
    ),
    "Probabilistic DA baseline such as EnKF/LETKF": (
        "NEXT_PROBABILISTIC_DA_COMPARISON_CONTRACT.md",
        "PAPER_DRAFT.md:Section 3 frozen outcome matrix",
        "CLAIM_LEDGER.md:C17 plus a decision-bearing probabilistic-DA claim",
    ),
    "Independent-strength deterministic background and 3D-Var": (
        "FROZEN_EVALUATION_HANDOFF.md",
        "PAPER_DRAFT.md:Section 3 common-information comparison",
        "CLAIM_LEDGER.md:C9 and C17 evidence-strength transition",
    ),
}
READINESS_CLOSURE_FILE_ANCHORS = {
    "Eligible spatially preserving calibration": ("## 6.", "| ID | Claim | Status |"),
    "Conformal intervals": ("## 3.", "| C17 |"),
    "Probabilistic DA baseline such as EnKF/LETKF": ("## 3.", "| C17 |"),
    "Independent-strength deterministic background and 3D-Var": (
        "## 3.",
        "| C9 |",
    ),
}
READINESS_CLOSURE_ROUTE_ROW = re.compile(
    r"^\| (?P<blocker>[^|]+?) \| `(?P<contract>(?:NEXT_[A-Z_]+_CONTRACT|FROZEN_EVALUATION_HANDOFF)\.md)` \| "
    r"(?P<manuscript>[^|]+?) \| (?P<ledger>[^|]+?) \|$",
    re.MULTILINE,
)
MINIMUM_TIER_EVIDENCE_GUARD_ROWS = {
    "conformal": ("MISSING", "NONE", "PRE_RESULT_ONLY"),
    "probabilistic_da": ("MISSING", "NONE", "PRE_RESULT_ONLY"),
    "independent_deterministic": (
        "PRESENT_DEVELOPMENT_ONLY", "NONE", "DEVELOPMENT_ONLY"
    ),
}
MINIMUM_TIER_EVIDENCE_GUARD_ROW = re.compile(
    r"^\| `(?P<route>conformal|probabilistic_da|independent_deterministic)` \| "
    r"`(?P<status>[A-Z_]+)` \| `(?P<record>NONE|[0-9a-f]{64})` \| "
    r"`(?P<presentation>[A-Z_]+)` \|$",
    re.MULTILINE,
)

ELIGIBLE_CALIBRATION_GUARD_STATES = {
    ("MISSING_ELIGIBLE_RESULT", "NONE", "BLOCKED"),
    ("ELIGIBLE", "COMPACT_RECORD", "DECISION_BEARING"),
}
ELIGIBLE_CALIBRATION_GUARD_ROW = re.compile(
    r"^\| `eligible_calibration` \| `(?P<status>MISSING_ELIGIBLE_RESULT|ELIGIBLE)` \| "
    r"`(?P<record>NONE|[0-9a-f]{64})` \| `(?P<presentation>BLOCKED|DECISION_BEARING)` \|$",
    re.MULTILINE,
)


def validate_eligible_calibration_transition(
    manuscript: str, claim_ledger: str, readiness: str, reproducibility: str
) -> None:
    """Bind a positive calibration transition to one compact record everywhere."""
    documents = {
        "PAPER_DRAFT.md": manuscript,
        "CLAIM_LEDGER.md": claim_ledger,
        "PUBLICATION_READINESS.md": readiness,
        "REPRODUCIBILITY.md": reproducibility,
    }
    observed: dict[str, tuple[str, str, str]] = {}
    for filename, text in documents.items():
        rows = list(ELIGIBLE_CALIBRATION_GUARD_ROW.finditer(text))
        require(len(rows) == 1, f"{filename} eligible-calibration guard is missing or duplicated")
        row = rows[0]
        record = row["record"]
        state = (
            row["status"],
            "COMPACT_RECORD" if record != "NONE" else "NONE",
            row["presentation"],
        )
        require(
            state in ELIGIBLE_CALIBRATION_GUARD_STATES,
            f"{filename} contains an invalid eligible-calibration transition",
        )
        observed[filename] = (row["status"], record, row["presentation"])
    require(
        len(set(observed.values())) == 1,
        "eligible-calibration transition is not atomic across publication surfaces",
    )


def validate_minimum_tier_evidence_guards(
    manuscript: str, claim_ledger: str, readiness: str, audit: str
) -> None:
    """Forbid result language until every closure route names compact evidence."""
    documents = {
        "PAPER_DRAFT.md": manuscript,
        "CLAIM_LEDGER.md": claim_ledger,
        "PUBLICATION_READINESS.md": readiness,
        "MINIMUM_TIER_COMPARISON_AUDIT.md": audit,
    }
    for filename, text in documents.items():
        rows = list(MINIMUM_TIER_EVIDENCE_GUARD_ROW.finditer(text))
        observed = {
            row["route"]: (row["status"], row["record"], row["presentation"])
            for row in rows
        }
        require(
            len(rows) == len(observed) == len(MINIMUM_TIER_EVIDENCE_GUARD_ROWS),
            f"{filename} minimum-tier evidence guard is incomplete or duplicated",
        )
        require(
            observed == MINIMUM_TIER_EVIDENCE_GUARD_ROWS,
            f"{filename} contains a decision-bearing minimum-tier claim without compact evidence",
        )

    matrix = re.search(
        r"^\| Conformal decision \| Probabilistic-DA decision \|.*?"
        r"(?=^Across all four valid outcomes)",
        manuscript,
        re.MULTILINE | re.DOTALL,
    )
    require(matrix is not None, "manuscript pre-result outcome matrix is missing")
    outside_matrix = manuscript[: matrix.start()] + manuscript[matrix.end() :]
    for label in (
        "CONFORMAL_USEFUL", "CONFORMAL_NEGATIVE",
        "PROBABILISTIC_DA_USEFUL", "PROBABILISTIC_DA_NEGATIVE",
    ):
        require(
            label not in outside_matrix,
            f"decision-bearing label {label} appears outside the pre-result matrix",
        )
EXTERNAL_PRIMARY_HANDOFF_ANCHORS = {
    "PUBLICATION_READINESS.md": (
        "External primary evidence state: RECONCILED_NEGATIVE",
        "external_2024_calendar_global_bias_raw48_primary_v1",
        "All 48 cases, 480 member records",
        "`overall_eligible=false`",
        "Proper scores, absolute rank reliability,\ntruth-relative boundary calibration and spatial/physical preservation fail",
        "No resubmission, retuning or second external\nevaluation is admissible",
    ),
    "CLAIM_LEDGER.md": (
        "| C39 | The frozen calendar global-bias primary generalizes",
        "Rejected by the reconciled independent primary",
        "`overall_eligible=false`",
        "do not resubmit, retune or open another external evaluation",
    ),
}
EXTERNAL_PRIMARY_RECONCILIATION_ANCHORS = (
    "Status: reconciled negative from the exact permitted recovery.",
    "external_2024_calendar_global_bias_confirm48_primary_retry2",
    "superseded `retry1` metrics are excluded",
    "external_2024_calendar_global_bias_raw48_primary_v1",
    "four compact artifacts",
    "480 member records",
    "candidate-before-truth ordering",
    "`overall_eligible=false`",
    "Fair CRPS worsens from\n`0.05541808434196047` raw to `0.061833300537408264` candidate",
    "normalized\nmean rank worsens from `0.23610946912844127` raw to approximately `0.225`",
    "without retuning, resubmission or\nanother external evaluation",
)
EXTERNAL_PRIMARY_STATE_MARKER = (
    "External primary evidence state: RECONCILED_NEGATIVE"
)
EXTERNAL_PRIMARY_STATE_FILES = (
    "PAPER_DRAFT.md",
    "CLAIM_LEDGER.md",
    "REPRODUCIBILITY.md",
    "RESEARCH_PLAN.md",
    "PUBLICATION_READINESS.md",
)
EXTERNAL_PRIMARY_DECISION_FILES = (
    "PAPER_DRAFT.md",
    "CLAIM_LEDGER.md",
    "REPRODUCIBILITY.md",
)
EXTERNAL_PRIMARY_DECISION_ANCHORS = (
    "external_2024_calendar_global_bias_confirm48_primary_retry2",
    "overall_eligible=false",
    "0.05541808434196047",
    "0.061833300537408264",
    "0.23610946912844127",
    "0.225",
    "absolute rank reliability",
)
INVALID_EXTERNAL_PRIMARY_CLAIMS = (
    "88 percent upper-rank-bin",
    "88% upper-rank-bin",
)
RESEARCH_PLAN_CLOSED_PRIMARY_ANCHORS = (
    "frozen after the single completed independent primary",
    "cannot be used to select, retune or resubmit a replacement candidate",
    "archived months-long route, not\n  an active publication dependency or a fast fallback",
    "No second independent evaluation is authorized in the current campaign",
    "does not reuse the opened primary for selection",
)
REPRODUCIBILITY_SECTION_ORDER = (
    "## Frozen latent-temperature recovery handoff",
    "## Locked-MC-dropout wrapper recovery handoff",
    "## Completed purged analog-residual handoff",
    "## Completed coherent-member-offset handoff",
    "## Archived clean-checkpoint deep-ensemble executable handoff",
    "## Compact-artifact contract",
    "## Required reconciliation checks",
)
SCORE_AWARE_RECONCILIATION_ANCHORS = (
    "Status: PRE_RESULT_NO_TRUSTED_MODE",
    "## Mutually exclusive scientific branches",
    "### Positive branch",
    "### Negative branch",
    "## Atomic publication update map",
    "## Fail-closed consistency rules",
    "compact_directory_sha256",
    "decision_bearing=True",
)
SCORE_AWARE_RESULT_MARKER = re.compile(
    r"^SCORE_AWARE_RESULT: status=(?P<status>RECONCILED_(?:POSITIVE|NEGATIVE)); "
    r"experiment_id=(?P<experiment_id>[a-z0-9_]{1,64}); "
    r"candidate=(?P<candidate>[a-z0-9_]{1,64}); "
    r"admission_record_sha256=(?P<admission_digest>[0-9a-f]{64}); "
    r"compact_directory_sha256=(?P<digest>[0-9a-f]{64}); "
    r"completed_cases=(?P<completed_cases>[1-9][0-9]*); "
    r"ensemble_size=(?P<ensemble_size>[1-9][0-9]*); "
    r"proper_score=(?P<proper_score>true|false); "
    r"reliability=(?P<reliability>true|false); "
    r"boundary=(?P<boundary>true|false); "
    r"spatial_physical=(?P<spatial_physical>true|false); "
    r"operational=(?P<operational>true|false); "
    r"overall_eligible=(?P<eligible>true|false)$",
    re.MULTILINE,
)


def validate_score_aware_reconciliation_consistency(
    manuscript: str,
    claim_ledger: str,
    readiness: str,
    reproducibility: str,
    reconciliation: str,
) -> None:
    """Require one identical decision marker across all five publication files."""
    documents = {
        "PAPER_DRAFT.md": manuscript,
        "CLAIM_LEDGER.md": claim_ledger,
        "PUBLICATION_READINESS.md": readiness,
        "REPRODUCIBILITY.md": reproducibility,
        "SCORE_AWARE_RESULT_RECONCILIATION.md": reconciliation,
    }
    pre_result = "Status: PRE_RESULT_NO_TRUSTED_MODE" in reconciliation
    matches = {
        name: SCORE_AWARE_RESULT_MARKER.findall(text)
        for name, text in documents.items()
    }
    if pre_result:
        require(
            not any(matches.values()),
            "pre-result score-aware state contains a reconciled decision marker",
        )
        return

    for name, found in matches.items():
        require(
            len(found) == 1,
            f"{name} must contain exactly one score-aware result marker",
        )
    canonical = next(iter(matches.values()))[0]
    require(
        all(found[0] == canonical for found in matches.values()),
        "score-aware result markers disagree across publication files",
    )
    (
        status, _, _, _, _, completed_cases, ensemble_size,
        *family_values, eligible,
    ) = canonical
    require(
        completed_cases == "40" and ensemble_size == "10",
        "score-aware reconciled marker contradicts frozen completed counts",
    )
    family_eligible = all(value == "true" for value in family_values)
    require(
        (eligible == "true") == family_eligible,
        "score-aware overall_eligible contradicts mandatory families",
    )
    require(
        (status == "RECONCILED_POSITIVE" and family_eligible)
        or (status == "RECONCILED_NEGATIVE" and not family_eligible),
        "score-aware reconciliation branch contradicts mandatory families",
    )
AMENDED_PRIMARY_ANCHORS = {
    "AMENDED_PRIMARY_EVALUATION_CONTRACT.md": (
        "DEPLOYED_AUDITED_PRIMARY_IN_FLIGHT",
        "f2225da7a05cab53b14604e45bed840a0ec559aed20856ae8ef2dd72d915b8f8",
        "57e8dd1859c4ac9a144be68904450926a6098b08a3b58a35e3d7dcc6a5bb9185",
        "absolute uniformity criterion",
        "q={0,.15,.90,.95,.99}",
        "encoding diagnostics only",
    ),
    "RESEARCH_PLAN.md": (
        "AMENDED_PRIMARY_EVALUATION_CONTRACT.md",
        "near-one-versus-raw is not a valid gate",
        "development evidence only and cannot establish final success",
    ),
    "PUBLICATION_READINESS.md": (
        "no historical\n`overall_eligible` from the superseded gate is final success",
        "controller deployed and audited the exact amended primary and gate digests",
        "four compact artifacts establish both admissible provenance and the reconciled\nnegative scientific outcome",
        "no\nexternal primary remains active",
    ),
    "FROZEN_EVALUATION_HANDOFF.md": (
        "both exact amended-contract digests and controller\ndeploy/audit attestation are present",
        "truth-referenced `q={0,.15,.90,.95,.99}` high-SIC decisions",
    ),
}
PYTHON_FILES = tuple(name for name in REQUIRED_FILES if name.endswith(".py"))
RANK_COHERENT_CONTRACT_ANCHORS = (
    "Status: DESIGN_FROZEN_NO_RUNNER",
    "purged rank-targeted coherent anomaly transport",
    "five\n  contiguous eight-case holdouts",
    "non-circular three-case purge",
    "{0.0,0.5,0.75,1.0,1.25}",
    "no_positive_feasible_alpha=true",
    "overall_eligible=true",
    "source_experiment=joint_full_condition_validation_2022",
    "summary_only",
    "no currently\nimplemented trusted mode implements it",
    "rank_coherent_reference.py",
    "exact `valid`, 40-case, ten-member, stride-five\ndevelopment envelope",
    "all five mandatory\nfamily flags are present as JSON booleans",
    "ten rank-target counts must each equal 40",
    "maximum mean error\nmust not exceed `1e-10`",
)
RAW_MEMBER_REWEIGHTING_CONTRACT_ANCHORS = (
    "Status: DESIGN_FROZEN_CONTINGENT_NO_RUNNER",
    "activated only if the frozen\nwhole-field rank-coherent anomaly-transport candidate is completed and rejected",
    "five contiguous\n  eight-case holdouts",
    "non-circular three-case purge",
    "p[r] = (c[r] + 0.5) / 15",
    "u_j=(j+0.5)/10",
    "bitwise equality with that source field",
    "analysis_fair_crps",
    "overall_eligible=true",
    "probability sum error above `1e-12`",
    "source_experiment=joint_full_condition_validation_2022",
    "summary_only",
    "no experiment proposal or invented mode identifier",
)
RANK_COHERENT_REVIEW_CHECKLIST_ANCHORS = (
    "Status: REVIEW_CONTRACT_READY_NO_IMPLEMENTED_MODE",
    "source_experiment=joint_full_condition_validation_2022",
    "exact `valid`, 40-case, ten-member,\n  stride-five development envelope",
    "non-circular three-case\n  purge",
    "{0.0,0.5,0.75,1.0,1.25}",
    "no_positive_feasible_alpha",
    "maximum mean error",
    "analysis_fair_crps",
    "overall_eligible",
    "server dry run on synthetic fixtures",
    "independent reviewer records no deviations",
    "## Controller-visible admission record",
    '"reviewed_mode": "<implemented trusted mode>"',
    '"decision_bearing_validation": "PASS"',
    '"deviations": []',
    "must copy `reviewed_mode` literally",
    "hard `NO_GO`",
)
RANK_COHERENT_HANDOFF_ANCHORS = {
    "RANK_COHERENT_CONTROLLER_HANDOFF.md": (
        "b6b9b709c587d79625d0237b15c3e36f4e3404eef37ba0bbbe850875be95d86c",
        "fbda1c1dee61f80ebb4b37562364c34be3268aa6c0c2e87b2d222d9e95b888e6",
        "--source-experiment joint_full_condition_validation_2022",
        "exactly `run_status.json`,\n`aggregate_case_mean_metrics.json`, `per_case_metrics.csv`, and `metadata.json`",
        "`metadata.decision_bearing=false`",
        "replace only the prototype gate\nadapter and interval sensitivity",
        "python3 -m unittest -v paper/test_rank_coherent_runner_prototype.py",
    ),
    "PUBLICATION_READINESS.md": (
        "rank_coherent_reference.py",
        "exact `valid`, 40-case, ten-member, stride-five development\nenvelope",
        "implementation evidence only",
        "publication status remains `NOT_READY`",
        "missing or non-boolean mandatory gate families",
        "rank-target counts\nother than ten uses of 40",
        "maximum projection mean error\nabove `1e-10`",
        "aggregate deltas that do not equal candidate minus\nraw",
        "paired summaries that do not reconcile with the same aggregate metric set",
        "member-spatial decisions that do\nnot follow their frozen absolute tolerances",
        "family flags that differ from\ntheir complete criterion conjunctions",
    ),
    "REPRODUCIBILITY.md": (
        "## Rank-coherent runner review handoff",
        "python3 paper/rank_coherent_reference.py",
        "bounded\nprojection mean error above `1e-10`",
        "exact `valid`, 40-case, ten-member,\nstride-five development envelope",
        "not scientific evidence or an implemented mode",
        "overall_eligible` value different from their conjunction",
        "exact alpha/boolean consistency\nfor `no_positive_feasible_alpha`",
        "rank-target counts each equal to 40",
        "Negative fixtures exercise incomplete folds",
        "projection-invariant failure, an inconsistent aggregate\ndelta",
        "member-spatial pass flag that violates its frozen\ntolerance",
        "exact member-spatial-to-gate linkage",
    ),
}
RANK_COHERENT_IMMUTABLE_DIGESTS = {
    "rank_coherent_runner_prototype.py":
        "b6b9b709c587d79625d0237b15c3e36f4e3404eef37ba0bbbe850875be95d86c",
    "test_rank_coherent_runner_prototype.py":
        "fbda1c1dee61f80ebb4b37562364c34be3268aa6c0c2e87b2d222d9e95b888e6",
    "rank_coherent_reference.py":
        "d25095af94eb9e93c20e8495f9584c6f1d954c24dfa0a8b72ef65c51aeb1a170",
    "NEXT_RANK_COHERENT_CONTRACT.md":
        "1489a68f914131c6b6ad545a413903f366d28f8570e8e40cc674db77282197d1",
}
FIGURE_PATTERN = re.compile(r"!\[[^]]*\]\(([^)]+)\)")
REFERENCE_PATTERN = re.compile(r"^(\d+)\. ", re.MULTILINE)
CITATION_PATTERN = re.compile(r"\[([1-9]\d*(?:\s*,\s*[1-9]\d*)*)\]")
CLAIM_PATTERN = re.compile(r"^\| C(\d+) \|", re.MULTILINE)
TRACEABILITY_HEADING = "### Claim-ledger traceability\n"
TRACEABILITY_PATTERN = re.compile(r"\bC(\d+)\b")
EMPIRICAL_TRACEABILITY_HEADING = "### Empirical evidence traceability\n"
EMPIRICAL_CLAIM_IDS = set(range(3, 9)) | set(range(11, 40))
EMPIRICAL_TRACEABILITY_ROW = re.compile(
    r"^\| (?P<claims>C\d+(?:, C\d+)*) \| "
    r"(?P<presentation>[^|]+) \| `(?P<source>[^`]+)` \| (?P<role>[^|]+) \|$",
    re.MULTILINE,
)
DECISION_PRESENTATION_HEADING = "### Decision-bearing presentation audit\n"
DECISION_PRESENTATION_ROWS = {
    "global-spread": ("Frozen-mechanism table; paired-diagnostic table; Figure 1", "Negative", "Date and four-date-block intervals"),
    "hurdle-IDR/ECC-Q": ("Purged hurdle-IDR/ECC-Q table", "Negative", "Date interval reported in source"),
    "projected-spread": ("Projected-spread table; Figure 2", "Negative", "Date and four-date-block intervals"),
    "open-logit": ("Open-logit table", "Negative", "Not decision-critical"),
    "ZOIB-EMOS/ECC-Q": ("ZOIB-EMOS/ECC-Q table", "Negative", "Frozen paired intervals in source"),
    "later-mechanism-family": ("Later-mechanism family matrix", "Negative", "Claim-specific intervals where decision-bearing"),
    "amended-primary-policy": ("Amended primary decision policy", "Normative rejection rule", "Not applicable"),
    "locked-MC-dropout": ("Later-mechanism family matrix", "Negative overall only", "Unavailable; no effect-size claim"),
    "independent-primary": ("Independent-primary table", "Negative", "Not required for observed 48-case census"),
}
DECISION_PRESENTATION_ROW = re.compile(
    r"^\| `(?P<unit>[^`]+)` \| (?P<presentation>[^|]+) \| "
    r"(?P<decision>[^|]+) \| (?P<uncertainty>[^|]+) \|$",
    re.MULTILINE,
)
MINIMUM_TIER_COMPARISONS = {
    "Raw ensemble": "PRESENT",
    "Physical-space bias/spread scaling": "PRESENT_NEGATIVE",
    "Naive affine-logit scaling": "PRESENT_NEGATIVE",
    "Zero/one-inflated Beta or EMOS-like SIC postprocessing": "PRESENT_NEGATIVE",
    "Isotonic/quantile mapping": "PRESENT_NEGATIVE",
    "Conformal intervals": "MISSING",
    "ECC-Q/ECC-T or rank-preserving reconstruction": "PRESENT_PARTIAL_NEGATIVE",
    "Deterministic background and 3D-Var": "PRESENT_DEVELOPMENT_ONLY",
    "Probabilistic DA baseline such as EnKF/LETKF": "MISSING",
}
MINIMUM_TIER_EVIDENCE_CONTRACT = {
    "Raw ensemble": (
        "REPRODUCIBILITY.md",
        "compares raw with all three fixed candidates",
    ),
    "Physical-space bias/spread scaling": (
        "REPRODUCIBILITY.md",
        "completed fixed-contract joint calibration audit",
    ),
    "Naive affine-logit scaling": ("CLAIM_LEDGER.md", "| C7 |"),
    "Zero/one-inflated Beta or EMOS-like SIC postprocessing": (
        "REPRODUCIBILITY.md",
        "## Frozen ZOIB-EMOS/ECC-Q handoff",
    ),
    "Isotonic/quantile mapping": (
        "REPRODUCIBILITY.md",
        "fixed purged hurdle-isotonic/ECC-Q audit",
    ),
    "Conformal intervals": ("RESEARCH_PLAN.md", "- conformal intervals;"),
    "ECC-Q/ECC-T or rank-preserving reconstruction": (
        "REPRODUCIBILITY.md",
        "deterministic ECC-Q reconstruction",
    ),
    "Deterministic background and 3D-Var": ("CLAIM_LEDGER.md", "| C4 |"),
    "Probabilistic DA baseline such as EnKF/LETKF": (
        "RESEARCH_PLAN.md",
        "- probabilistic DA baseline such as EnKF/LETKF",
    ),
}
MINIMUM_TIER_ROW = re.compile(
    r"^\| (?P<comparison>[^|]+) \| (?P<presentation>[^|]+) \| "
    r"`(?P<source>[^`]+)` \| (?P<status>[A-Z_]+) \|$",
    re.MULTILINE,
)
FINAL_DIAGNOSTIC_ANCHORS = {
    "PAPER_DRAFT.md": (
        "0.055738",
        "0.059107",
        "0.071936",
        "final fixed open-logit diagnostic",
        "`locked_mc_dropout_p010_final_ema_ensemble` also has\n`overall_eligible=false`",
    ),
    "CLAIM_LEDGER.md": (
        "| C26 |",
        "| C27 |",
        "0.0557378",
        "0.0591071",
        "0.0719362",
        "No post-hoc tuning follows.",
        "| C38 | The frozen locked-MC-dropout candidate",
        "Rejected by the controller-recorded gate decision",
        "The complete quantitative compact payload is not present in this worktree",
        "so no failed family or effect size is claimed",
        "see `LOCKED_MC_DROPOUT_RESULT_RECONCILIATION.md`",
        "contractual prerequisite for the subsequently completed iid calendar global-bias fallback",
    ),
    "REPRODUCIBILITY.md": (
        "0.0557378026",
        "0.0591071355",
        "0.0719361803",
        "must not be tuned after this result",
        "gate identifies `locked_mc_dropout_p010_final_ema_ensemble` and records\n`overall_eligible=false`",
    ),
    "PUBLICATION_READINESS.md": (
        "0.0557378",
        "0.0591071",
        "0.0719362",
        "no post-hoc tuning is admissible",
        "locked-MC-dropout chain is complete and negative",
    ),
}
CALIBRATION_CONTRACT_ANCHORS = {
    "NEXT_BASELINE_CONTRACT.md": (
        "topology-preserving stratified transport",
        "lambda in {1.0,1.25,1.5,2.0,2.75,4.0}",
        "no_feasible_training_scale=true",
        "source_experiment=joint_full_condition_validation_2022",
        "no_compensation_across_families=true",
        "Design frozen locally; no experiment has been launched.",
    ),
    "RESEARCH_PLAN.md": (
        "NEXT_BASELINE_CONTRACT.md",
        "topology-preserving stratified transport",
        "reviewed run is complete and rejected",
    ),
    "CLAIM_LEDGER.md": (
        "| C28 |",
        "| C29 |",
        "| C30 |",
        "| C31 |",
        "0.0585570",
        "overall_eligible=false",
        "only operational validity passes at family level",
        "conformal and probabilistic-DA families",
        "raw and candidate fair CRPS are both `0.0584905850`",
        "raw and candidate spread-skill are both `0.7240662110`",
    ),
    "PAPER_DRAFT.md": (
        "| ZOIB-EMOS/ECC-Q diagnostic | Raw ensemble | Candidate | Interpretation |",
        "| Ordinary CRPS | 0.062108 | 0.064991 | 4.64% worse |",
        "| Inner-order attainable-coverage error | 0.175953 | 0.253947 | Reliability family fails |",
        "| Mean IIEE | 0.079600 | 0.087190 | Spatial/physical family fails |",
        "topology-preserving stratified transport",
        "completed fixed result rejects the hypothesis",
        "`0.0584905850` versus `0.0584905850`",
        "`0.7240662110` versus `0.7240662110`",
    ),
    "REPRODUCIBILITY.md": (
        "## Frozen ZOIB-EMOS/ECC-Q handoff",
        "exactly 13 fitted coefficients",
        "0.0642804809",
        "Only operational validity passes at family level",
        "overall_eligible=false",
        "## Completed topology-preserving transport handoff",
        "topology-preserving stratified transport",
        "all three finite-ensemble reliability criteria",
        "(`0.0584905850`/\n`0.0584905850`)",
        "(`0.7240662110`/`0.7240662110`)",
    ),
    "PUBLICATION_READINESS.md": (
        "No additional implemented mode",
        "0.0585570",
        "Only operational validity passes at family level",
        "topology-preserving stratified-transport contract",
        "completed compact result",
        "fair CRPS are both `0.0584905850`",
        "spread-skill are both\n`0.7240662110`",
        "Job completion alone is not scientific\nevidence",
        "base gate id and separately completed retry id remain distinct\nevidence units",
        "manuscript therefore\nmakes no positive or negative EMA claim",
    ),
}
NEXT_METHOD_ANCHORS = (
    "purged analog-residual ensemble dressing",
    "five contiguous eight-case holdouts",
    "six forecast-only raw-mean features",
    "they are not errors against the verifying field",
    "They are never",
    "used in a held-out feature, feature normalization, distance or tie decision.",
    "Select the ten training dates",
    "clip(m + r_j, 0, 1)",
    "overall_eligible=true",
    "source_experiment=joint_full_condition_validation_2022",
    "no experiment has been launched",
    "analog_residual_reference.py",
)
ANALOG_RESULT_ANCHORS = {
    "PAPER_DRAFT.md": (
        "0.064305",
        "0.067464",
        "0.107720",
        "0.753021",
        "fields therefore improve ranks and attainable coverage but transfer unsafe",
        "| Later frozen mechanism | Proper score | Reliability | Boundary | Spatial/physical | Operational | Overall |",
        "| Locked MC dropout | NR | NR | NR | NR | NR | Fail |",
    ),
    "CLAIM_LEDGER.md": (
        "| C31 |",
        "0.0643049265",
        "0.0674642030",
        "0.1077199457",
        "0.7530213545",
    ),
    "REPRODUCIBILITY.md": (
        "## Completed purged analog-residual handoff",
        "0.0643049265",
        "0.0674642030",
        "0.1077199457",
        "0.7530213545",
    ),
    "PUBLICATION_READINESS.md": (
        "exact frozen purged analog-residual contract",
        "worsens by 9.94%",
        "mean IIEE by 35.3%",
        "0.753021",
    ),
}
GUIDANCE_RESULT_ANCHORS = {
    "PAPER_DRAFT.md": (
        "frozen guidance-mixture test",
        "exact two-member allocation",
        "all three proper-score\ncriteria fail",
        "without retuning weights or member",
    ),
    "CLAIM_LEDGER.md": (
        "| C32 |",
        "equal-allocation mixture across five independent-CFG guidance settings",
        "every sampling-protocol check passes",
        "weights and two-members-per-setting allocation must not be retuned",
    ),
    "REPRODUCIBILITY.md": (
        "guidance-mixture result is complete and rejected",
        "proper-score, reliability and boundary criteria fail",
    ),
    "PUBLICATION_READINESS.md": (
        "subsequent frozen guidance-mixture contract has also been executed",
        "fails the\nproper-score, finite-ensemble reliability, boundary and spatial/physical",
    ),
}
NEXT_GENERATIVE_METHOD_ANCHORS = {
    "NEXT_GENERATIVE_METHOD_CONTRACT.md": (
        "clean-checkpoint deep ensemble",
        "training seeds `1701`, `1702`, and",
        "balanced as `14/13/13`",
        "latent member seeds `2401`, `2402`, `2403`",
        "common-random-number comparison",
        "overall_eligible=true",
        "summary_only",
        "validation_clean_checkpoint_deep_ensemble_sampling",
        "validation_clean_checkpoint_deep_ensemble_gate",
        "deep_ensemble_reference.py",
        "validate_admission_manifest",
        "Unknown or missing fields",
    ),
    "REPRODUCIBILITY.md": (
        "NEXT_GENERATIVE_METHOD_CONTRACT.md",
        "Archived clean-checkpoint deep-ensemble executable handoff",
        "reviewed trusted sampling and gate modes",
        "route is not selectable as a fast fallback",
        "months-long route",
        "not\na scientific result, an active dependency or authorization to launch",
        "Exactly three clean training seeds",
        "`2401`, `2402`, and `2403`",
        "`2404`",
        "`14/13/13`",
        "source_experiment=joint_full_condition_validation_2022",
        "python3 paper/deep_ensemble_reference.py",
        "checkpoint extras 14/13/13",
        "metadata admission fails closed",
        "exact compact pre-score manifest",
        "metadata-admission contract conformance",
    ),
    "PUBLICATION_READINESS.md": (
        "guidance-mixture result is now complete and rejected",
        "clean-checkpoint deep ensemble",
        "retained only as an archived\nreproducibility contract",
        "It is not selectable as a fast fallback",
        "no sampling or gate stage from that route is an active scientific dependency",
        "Both checkpoint-trajectory EMA stages have now completed",
        "clean-checkpoint sampling\nretry was cancelled rather than completed",
        "neither it nor its dependent CPU\ngate is evidence in flight",
        "NEXT_LATENT_TEMPERATURE_CONTRACT.md",
        "no second temperature may be\nselected post hoc",
    ),
}
STALE_READINESS_ANCHORS = (
    "Continuous calibration development therefore moves to the pre-implementation",
    "Its fixed ten-member guidance mixture",
    "no currently implemented trusted mode\nexecutes that contract",
    "unchanged no-compensation gate is queued as",
    "A positive\nvalidation-set latent-temperature result would establish",
    "Its dependent unchanged CPU gate is\nqueued.",
    "Until the respective compact gates are returned",
    "locked-MC-dropout sampler is in flight",
    "its dependent gate already queued",
    "locked-MC-dropout chain remains the next decision-bearing",
    "Until that exact compact gate is returned",
    "scientific outcome remains unknown until the dependent",
    "the only active\ndecision-bearing external evidence",
    "The sole\nactive external primary",
)
MANUSCRIPT_EVIDENCE_ANCHORS = (
    "# Auditing Reliability in Generative Data Assimilation under Sparse Spatial Observations",
    "None passes the common development gate",
    "The contribution is therefore an auditable\nfailure map and a fail-closed evaluation protocol",
    "not a successful calibrated\nensemble",
    "A mechanistically diverse negative-calibration suite.",
    "topology-preserving stratified transport preserves boundary and\n   spatial structure but selects no effective correction",
    "historical residual\n   dressing improves reliability while damaging proper scores and physical\n   fields",
    "a fixed guidance mixture fails every scientific family",
    "coherent\n   member offsets, with and without projection, select the null action",
    "not a\n   claim that all calibration families have been exhausted",
    "| Diagnostic | Raw ensemble | Cross-fitted correction | Interpretation |",
    "| Established-ice Brier score | 0.056973 | 0.058200 |",
    "| Purged hurdle-IDR/ECC-Q diagnostic | Raw ensemble | Candidate | Interpretation |",
    "| Fair CRPS | 0.058491 | 0.085581 | 46.3% worse; proper-score family fails |",
    "| Mean-preserving projected-spread diagnostic | Raw ensemble | Candidate | Interpretation |",
    "| Upper-cap mass | 0 | 0.167438 | Active-cap boundary failure remains |",
    "| Mean-preserving open-logit diagnostic | Raw ensemble | Candidate | Interpretation |",
    "| Established-ice Brier score | 0.056973 | 0.059107 | Worse beyond 1% tolerance; boundary family fails |",
    "| ZOIB-EMOS/ECC-Q diagnostic | Raw ensemble | Candidate | Interpretation |",
    "| Fair CRPS | 0.058491 | 0.058557 | No 3% improvement; proper-score family fails |",
    "| Paired diagnostic | Mean delta (corrected - raw) | Date-bootstrap 95% CI | Four-date-block 95% CI |",
    "| Mean IIEE | 0.004379 | [0.002065, 0.006716] | [0.001149, 0.007849] |",
    "| Independent primary diagnostic | Raw ensemble | Frozen candidate | Decision |",
    "| Fair CRPS | 0.05541808434196047 | 0.061833300537408264 | Worsens; proper scores fail |",
    "| Normalized mean rank | 0.23610946912844127 | approximately 0.225 | Worsens; absolute rank reliability fails |",
    "| Truth-relative boundary calibration | — | — | Fails |",
    "| Overall no-compensation gate | — | false | Independent primary is rejected |",
    "The single frozen independent primary is presented as a\nnegative generalization result",
)
FROZEN_EVALUATION_ANCHORS = (
    "Status: AUTHORIZED_ACTIVE_PENDING_COMPACT_RESULT",
    "sealed raw stage is complete",
    "stage\nstatus is not a scientific outcome",
    "immutable checkpoint identity, dataset-manifest digest, code revision",
    "No calibration family, coefficient, threshold,",
    "no-compensation families remain those in `RESEARCH_PLAN.md`",
    "Retrieval defaults to `summary_only`",
    "`overall_eligible=true`",
    "every recorded mandatory family is true",
)
FIGURE_TEXT_ANCHORS = {
    "figures/calibration_summary.svg": (
        "Cross-fitted global spread correction",
        "Aggregate validation diagnostics; 40 dates, 10 ensemble members",
        "0.0585",
        "0.0557",
        "0.7241",
        "1.0615",
        "0.8788",
    ),
    "figures/joint_gate_summary.svg": (
        "Mean-preserving spread: joint-gate diagnostics",
        "Candidate/raw ratios; lower is better; 40 dates, 10 members",
        "0.963x",
        "1.047x",
        "1.014x",
    ),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_external_primary_consistency(documents: dict[str, str]) -> None:
    """Keep decision-bearing external claims complete and reject invalid legacy text."""
    for name in EXTERNAL_PRIMARY_DECISION_FILES:
        require(name in documents, f"missing external-primary document: {name}")
        normalized = " ".join(documents[name].split()).lower()
        missing = [
            anchor
            for anchor in EXTERNAL_PRIMARY_DECISION_ANCHORS
            if anchor.lower() not in normalized
        ]
        require(
            not missing,
            f"{name} is missing external-primary decision anchors: "
            + ", ".join(missing),
        )
        invalid = [
            claim
            for claim in INVALID_EXTERNAL_PRIMARY_CLAIMS
            if claim.lower() in normalized
        ]
        require(
            not invalid,
            f"{name} inherits invalid external-primary claims: "
            + ", ".join(invalid),
        )


def validate_rank_coherent_immutable_identities(
    paper_dir: Path, handoff_text: str
) -> None:
    """Verify both file content and the documented immutable identity table."""
    for name, expected_digest in RANK_COHERENT_IMMUTABLE_DIGESTS.items():
        actual_digest = hashlib.sha256((paper_dir / name).read_bytes()).hexdigest()
        require(
            actual_digest == expected_digest,
            f"{name} immutable digest mismatch: "
            f"expected {expected_digest}, got {actual_digest}",
        )
        require(
            f"| `paper/{name}` | `{expected_digest}` |" in handoff_text,
            "RANK_COHERENT_CONTROLLER_HANDOFF.md does not record "
            f"the verified digest for {name}",
        )


def validate_empirical_traceability(manuscript: str, paper_dir: Path) -> None:
    """Fail closed when empirical claims lose presentation or compact evidence."""
    require(
        EMPIRICAL_TRACEABILITY_HEADING in manuscript,
        "manuscript contains no empirical evidence traceability section",
    )
    empirical_section = manuscript.split(
        EMPIRICAL_TRACEABILITY_HEADING, maxsplit=1
    )[1].split("\n### ", maxsplit=1)[0]
    empirical_rows = list(EMPIRICAL_TRACEABILITY_ROW.finditer(empirical_section))
    require(empirical_rows, "empirical evidence traceability contains no rows")
    empirical_claim_ids: list[int] = []
    for row in empirical_rows:
        row_claim_ids = [
            int(value) for value in TRACEABILITY_PATTERN.findall(row["claims"])
        ]
        empirical_claim_ids.extend(row_claim_ids)
        presentation = row["presentation"].strip()
        require(
            "Section " in presentation or "Figure " in presentation,
            "empirical traceability row has no concrete manuscript presentation: "
            + row["claims"],
        )
        source = (paper_dir / row["source"]).resolve()
        require(
            source.is_relative_to(paper_dir.resolve()) and source.is_file(),
            "empirical traceability source is missing or escapes paper/: "
            + row["source"],
        )
    require(
        len(empirical_claim_ids) == len(set(empirical_claim_ids)),
        "empirical evidence traceability contains duplicate claim IDs",
    )
    require(
        set(empirical_claim_ids) == EMPIRICAL_CLAIM_IDS,
        "empirical evidence traceability mismatch: traced="
        + ",".join(map(str, sorted(empirical_claim_ids)))
        + " expected="
        + ",".join(map(str, sorted(EMPIRICAL_CLAIM_IDS))),
    )

    require(
        DECISION_PRESENTATION_HEADING in manuscript,
        "manuscript contains no decision-bearing presentation audit",
    )
    decision_section = manuscript.split(
        DECISION_PRESENTATION_HEADING, maxsplit=1
    )[1].split("\n### ", maxsplit=1)[0]
    observed: dict[str, tuple[str, str, str]] = {}
    for row in DECISION_PRESENTATION_ROW.finditer(decision_section):
        unit = row["unit"]
        require(unit not in observed, f"duplicate decision presentation unit: {unit}")
        observed[unit] = tuple(
            row[name].strip() for name in ("presentation", "decision", "uncertainty")
        )
    require(
        observed == DECISION_PRESENTATION_ROWS,
        "decision-bearing presentation audit mismatch",
    )

    presentation_anchors = {
        "Frozen-mechanism table": "| Diagnostic | Raw ensemble | Cross-fitted correction |",
        "paired-diagnostic table": "| Paired diagnostic | Mean delta (corrected - raw) |",
        "Figure 1": "**Figure 1.** Aggregate validation diagnostics",
        "Purged hurdle-IDR/ECC-Q table": "| Purged hurdle-IDR/ECC-Q diagnostic |",
        "Projected-spread table": "| Mean-preserving projected-spread diagnostic |",
        "Figure 2": "**Figure 2.** Exact mean-preserving projected spread",
        "Open-logit table": "| Mean-preserving open-logit diagnostic |",
        "ZOIB-EMOS/ECC-Q table": "| ZOIB-EMOS/ECC-Q diagnostic |",
        "Later-mechanism family matrix": "| Later frozen mechanism | Proper score |",
        "Amended primary decision policy": "## Amended primary decision policy",
        "Independent-primary table": "| Independent primary diagnostic |",
    }
    named_presentations = " ".join(value[0] for value in observed.values())
    for name, anchor in presentation_anchors.items():
        if name in named_presentations:
            require(anchor in manuscript, f"decision presentation object is missing: {name}")

    negative_units = {
        unit for unit, (_, decision, _) in observed.items()
        if decision.startswith("Negative")
    }
    require(
        negative_units == {
            "global-spread", "hurdle-IDR/ECC-Q", "projected-spread",
            "open-logit", "ZOIB-EMOS/ECC-Q", "later-mechanism-family",
            "locked-MC-dropout", "independent-primary",
        },
        "decision-bearing audit does not preserve every negative result",
    )
    for unit in ("global-spread", "projected-spread"):
        require(
            "Date and four-date-block intervals" in observed[unit][2],
            f"mandatory paired uncertainty is missing for {unit}",
        )


def validate_minimum_tier_comparisons(paper_dir: Path) -> None:
    """Require a complete, fail-closed map of the baseline tier."""
    audit = (paper_dir / "MINIMUM_TIER_COMPARISON_AUDIT.md").read_text(
        encoding="utf-8"
    )
    rows = list(MINIMUM_TIER_ROW.finditer(audit))
    require(
        len(rows) == len(MINIMUM_TIER_COMPARISONS),
        "minimum-tier audit has an incomplete or duplicate comparison map",
    )
    observed: dict[str, str] = {}
    for row in rows:
        comparison = row["comparison"].strip()
        require(
            comparison not in observed,
            f"minimum-tier audit duplicates comparison: {comparison}",
        )
        observed[comparison] = row["status"]
        require(
            "Section " in row["presentation"] or "Figure " in row["presentation"],
            f"minimum-tier comparison lacks a manuscript anchor: {comparison}",
        )
        source = (paper_dir / row["source"]).resolve()
        require(
            source.is_relative_to(paper_dir.resolve()) and source.is_file(),
            f"minimum-tier compact source is missing or escapes paper/: {comparison}",
        )
        expected_source, evidence_anchor = MINIMUM_TIER_EVIDENCE_CONTRACT[comparison]
        require(
            row["source"] == expected_source,
            f"minimum-tier compact source differs from evidence contract: {comparison}",
        )
        require(
            evidence_anchor in source.read_text(encoding="utf-8"),
            f"minimum-tier compact source lacks evidence anchor: {comparison}",
        )
    require(
        observed == MINIMUM_TIER_COMPARISONS,
        "minimum-tier evidence-status map differs from the frozen required set",
    )
    require(
        audit.count("| MISSING |") == 2,
        "minimum-tier audit must retain exactly the two evidenced missing families",
    )
    conformal_contract = (
        paper_dir / "NEXT_CONFORMAL_BASELINE_CONTRACT.md"
    ).read_text(encoding="utf-8")
    conformal_anchors = (
        "Status: `FROZEN_NOT_EXECUTABLE`",
        "`s = max(L - y, y - U, 0)`",
        "`ceil((n + 1) * 0.90)`",
        "coverage of at least `0.85`",
        "no more than `1.50` times the raw mean width",
        "No currently admitted trusted mode implements this contract",
    )
    require(
        all(anchor in conformal_contract for anchor in conformal_anchors),
        "frozen conformal baseline contract is incomplete or weakened",
    )
    probabilistic_contract = (
        paper_dir / "NEXT_PROBABILISTIC_DA_COMPARISON_CONTRACT.md"
    ).read_text(encoding="utf-8")
    probabilistic_anchors = (
        "Status: `FROZEN_NOT_EXECUTABLE`",
        "observation-operator output must be\n  hash-identical",
        "`radius_km in {50,100,200,400}`",
        "`inflation in {1.00,1.05,1.10,1.20}`",
        "LETKF fair CRPS is no more than `1.10` times raw",
        "LETKF passes the frozen absolute randomized-rank uniformity criterion",
        "LETKF ensemble-mean RMSE is no more than `1.02` times 3D-Var RMSE",
        "No currently\nadmitted trusted mode implements this contract",
    )
    require(
        all(anchor in probabilistic_contract for anchor in probabilistic_anchors),
        "frozen probabilistic DA comparison contract is incomplete or weakened",
    )


def validate_minimum_tier_key_claims(manuscript: str, paper_dir: Path) -> None:
    """Keep missing baseline families explicit in decision-bearing sections."""
    audit = (paper_dir / "MINIMUM_TIER_COMPARISON_AUDIT.md").read_text(
        encoding="utf-8"
    )
    missing_families = {
        row["comparison"].strip().lower()
        for row in MINIMUM_TIER_ROW.finditer(audit)
        if row["status"] == "MISSING"
    }
    require(
        len(missing_families) == 2,
        "key-claim audit requires exactly two normative missing families",
    )
    section_patterns = {
        "Abstract": r"^## Abstract\n(?P<body>.*?)^## 1\.",
        "Contributions": (
            r"^### Contributions supported by the present evidence\n"
            r"(?P<body>.*?)^## 2\."
        ),
        "Conclusion": r"^## 9\. Conclusion\n(?P<body>.*?)^## Ethics",
    }
    for section, pattern in section_patterns.items():
        match = re.search(pattern, manuscript, re.MULTILINE | re.DOTALL)
        require(match is not None, f"manuscript {section} section is missing")
        normalized = " ".join(match.group("body").split()).lower()
        for family in missing_families:
            require(
                family in normalized,
                f"manuscript {section} omits missing family: {family}",
            )
        require(
            "remain missing" in normalized and "baseline tier remains incomplete" in normalized,
            f"manuscript {section} promotes the incomplete minimum tier",
        )


def validate_readiness_blockers(readiness: str, paper_dir: Path) -> None:
    """Derive every NOT_READY blocker from normative evidence inventories."""
    rows = list(READINESS_BLOCKER_ROW.finditer(readiness))
    observed = {
        row["blocker"].strip(): (row["source"], row["status"])
        for row in rows
    }
    require(
        len(rows) == len(observed) == len(READINESS_BLOCKERS),
        "readiness blocker matrix is incomplete or contains duplicates",
    )
    require(
        observed == READINESS_BLOCKERS,
        "readiness blocker matrix differs from the normative blocker set",
    )
    require(
        all(row["closure"].strip() for row in rows),
        "readiness blocker lacks an exact closure condition",
    )
    audit = (paper_dir / "MINIMUM_TIER_COMPARISON_AUDIT.md").read_text(
        encoding="utf-8"
    )
    manuscript = (paper_dir / "PAPER_DRAFT.md").read_text(encoding="utf-8")
    ledger = (paper_dir / "CLAIM_LEDGER.md").read_text(encoding="utf-8")
    validate_minimum_tier_evidence_guards(manuscript, ledger, readiness, audit)
    tier_states = {
        row["comparison"].strip(): row["status"]
        for row in MINIMUM_TIER_ROW.finditer(audit)
    }
    require(
        tier_states.get("Conformal intervals") == observed["Conformal intervals"][1]
        and tier_states.get("Probabilistic DA baseline such as EnKF/LETKF")
        == observed["Probabilistic DA baseline such as EnKF/LETKF"][1]
        and tier_states.get("Deterministic background and 3D-Var")
        == observed["Independent-strength deterministic background and 3D-Var"][1],
        "readiness blockers do not match minimum-tier evidence states",
    )
    route_rows = list(READINESS_CLOSURE_ROUTE_ROW.finditer(readiness))
    routes = {
        row["blocker"].strip(): (
            row["contract"],
            row["manuscript"].strip(),
            row["ledger"].strip(),
        )
        for row in route_rows
    }
    require(
        len(route_rows) == len(routes) == len(READINESS_CLOSURE_ROUTES),
        "readiness closure-route matrix is incomplete or contains duplicates",
    )
    require(
        routes == READINESS_CLOSURE_ROUTES,
        "readiness closure routes differ from the frozen cross-artifact contract",
    )
    require(
        len({route[0] for route in routes.values()}) == len(routes),
        "readiness blockers do not have unique frozen contracts",
    )
    for blocker, (contract, manuscript_target, ledger_target) in routes.items():
        require((paper_dir / contract).is_file(), f"closure contract is missing: {contract}")
        require(manuscript_target.startswith("PAPER_DRAFT.md:"), "invalid manuscript closure target")
        require(ledger_target.startswith("CLAIM_LEDGER.md:"), "invalid claim-ledger closure target")
        manuscript_anchor, ledger_anchor = READINESS_CLOSURE_FILE_ANCHORS[blocker]
        require(manuscript_anchor in manuscript, "manuscript closure target is missing")
        require(ledger_anchor in ledger, "claim-ledger closure target is missing")


def validate_publication_status(readiness: str, frozen_handoff: str) -> str:
    """Prevent a READY declaration while the normative blocker matrix is open."""
    status_lines = re.findall(r"^Publication status: (\S+)$", readiness, re.MULTILINE)
    blocker_lines = re.findall(
        r"^Required scientific blockers:(.*)$", readiness, re.MULTILINE
    )
    require(len(status_lines) == 1, "readiness must contain exactly one publication status")
    require(len(blocker_lines) == 1, "readiness must contain exactly one blocker declaration")
    status = status_lines[0]
    blockers = blocker_lines[0].strip()
    require(
        status in {"NOT_READY", "READY_FOR_HUMAN_REVIEW"},
        f"unknown publication status: {status}",
    )
    blocker_states = [row["status"] for row in READINESS_BLOCKER_ROW.finditer(readiness)]
    unresolved_states = {"MISSING", "MISSING_ELIGIBLE_RESULT", "PRESENT_DEVELOPMENT_ONLY"}
    if status == "READY_FOR_HUMAN_REVIEW":
        require(blockers == "none", "ready status requires no scientific blockers")
        require(
            not any(state in unresolved_states for state in blocker_states),
            "ready status contradicts unresolved normative blocker states",
        )
        require(
            "Status: AUTHORIZED_ACTIVE_PENDING_COMPACT_RESULT" not in frozen_handoff,
            "ready status contradicts active evaluation awaiting compact result",
        )
    else:
        require(blockers != "none", "not-ready status requires explicit scientific blockers")
        require(
            "Scientific primary reconciliation: COMPLETE_NEGATIVE" in readiness,
            "not-ready status requires an explicit reconciled primary decision",
        )
    return status


def main() -> int:
    missing = [name for name in REQUIRED_FILES if not (PAPER_DIR / name).is_file()]
    require(not missing, f"missing required publication files: {', '.join(missing)}")

    for name in PYTHON_FILES:
        source = (PAPER_DIR / name).read_text(encoding="utf-8")
        ast.parse(source, filename=name)

    rank_oracle = subprocess.run(
        [sys.executable, str(PAPER_DIR / "rank_coherent_reference.py")],
        cwd=PAPER_DIR.parent,
        check=False,
        capture_output=True,
        text=True,
    )
    require(
        rank_oracle.returncode == 0
        and rank_oracle.stdout.strip() == "rank-coherent reference checks: PASS",
        "rank-coherent executable oracle failed: "
        + (rank_oracle.stderr.strip() or rank_oracle.stdout.strip() or "no output"),
    )

    reproducibility = (PAPER_DIR / "REPRODUCIBILITY.md").read_text(encoding="utf-8")
    validate_documented_regression_suites(reproducibility)
    validate_server_only_command_inputs(reproducibility)
    validate_eligible_calibration_transition(
        (PAPER_DIR / "PAPER_DRAFT.md").read_text(encoding="utf-8"),
        (PAPER_DIR / "CLAIM_LEDGER.md").read_text(encoding="utf-8"),
        (PAPER_DIR / "PUBLICATION_READINESS.md").read_text(encoding="utf-8"),
        reproducibility,
    )
    regression_suite = subprocess.run(
        [sys.executable, "-m", "unittest", *REQUIRED_REGRESSION_SUITES],
        cwd=PAPER_DIR.parent,
        check=False,
        capture_output=True,
        text=True,
    )
    require(
        regression_suite.returncode == 0,
        "required publication regression suites failed: "
        + (
            regression_suite.stderr.strip()
            or regression_suite.stdout.strip()
            or "no output"
        ),
    )

    manuscript = (PAPER_DIR / "PAPER_DRAFT.md").read_text(encoding="utf-8")
    claim_ledger = (PAPER_DIR / "CLAIM_LEDGER.md").read_text(encoding="utf-8")
    readiness = (PAPER_DIR / "PUBLICATION_READINESS.md").read_text(encoding="utf-8")
    score_aware_reconciliation = (
        PAPER_DIR / "SCORE_AWARE_RESULT_RECONCILIATION.md"
    ).read_text(encoding="utf-8")
    missing_score_aware_reconciliation_anchors = [
        anchor
        for anchor in SCORE_AWARE_RECONCILIATION_ANCHORS
        if anchor not in score_aware_reconciliation
    ]
    require(
        not missing_score_aware_reconciliation_anchors,
        "SCORE_AWARE_RESULT_RECONCILIATION.md is missing anchors: "
        + ", ".join(missing_score_aware_reconciliation_anchors),
    )
    validate_readiness_blockers(readiness, PAPER_DIR)
    reference_traceability = (PAPER_DIR / "REFERENCE_TRACEABILITY.md").read_text(
        encoding="utf-8"
    )
    validate_reference_traceability(reference_traceability)
    limitation_traceability = (PAPER_DIR / "LIMITATION_TRACEABILITY.md").read_text(
        encoding="utf-8"
    )
    validate_limitation_traceability(manuscript, claim_ledger, limitation_traceability)
    validate_claim_status_consistency(manuscript, claim_ledger)
    validate_score_aware_reconciliation_consistency(
        manuscript,
        claim_ledger,
        readiness,
        reproducibility,
        score_aware_reconciliation,
    )
    validate_minimum_tier_comparisons(PAPER_DIR)
    validate_minimum_tier_key_claims(manuscript, PAPER_DIR)
    missing_outcome_anchors = [
        anchor
        for anchor in MINIMUM_TIER_OUTCOME_MATRIX_ANCHORS
        if anchor not in manuscript
    ]
    require(
        not missing_outcome_anchors,
        "manuscript minimum-tier outcome matrix is incomplete: "
        + ", ".join(missing_outcome_anchors),
    )
    require(
        "all four valid joint outcomes of the two outstanding minimum-tier contracts"
        in readiness
        and "baseline-row closure is invariantly separate from learned-joint\ncalibration eligibility"
        in reproducibility,
        "publication handoff does not preserve the pre-result outcome interpretation",
    )
    claim_consistency_documents = {
        "PAPER_DRAFT.md": manuscript,
        "CLAIM_LEDGER.md": claim_ledger,
        "PUBLICATION_READINESS.md": readiness,
        "REPRODUCIBILITY.md": reproducibility,
    }
    missing_claim_consistency_anchors = [
        f"{filename}: {anchor}"
        for filename, anchors in MINIMUM_TIER_CLAIM_CONSISTENCY_ANCHORS.items()
        for anchor in anchors
        if anchor not in claim_consistency_documents[filename]
    ]
    require(
        not missing_claim_consistency_anchors,
        "minimum-tier claim-level consistency boundary is incomplete: "
        + ", ".join(missing_claim_consistency_anchors),
    )
    manuscript_tables = validate_markdown_tables(manuscript, "PAPER_DRAFT.md")
    ledger_tables = validate_markdown_tables(claim_ledger, "CLAIM_LEDGER.md")
    require(manuscript_tables >= 8, "manuscript is missing required evidence tables")
    require(ledger_tables == 2, "claim ledger must contain exactly two normative tables")
    frozen_handoff = (PAPER_DIR / "FROZEN_EVALUATION_HANDOFF.md").read_text(
        encoding="utf-8"
    )
    reproducibility = (PAPER_DIR / "REPRODUCIBILITY.md").read_text(encoding="utf-8")

    require(
        f"audit passed with {len(REQUIRED_FILES)} required files" in readiness,
        "publication readiness has a stale required-file count",
    )
    require(
        "The completed\nindependent primary supplies a negative falsification "
        "for one frozen candidate,\nnot successful independent generalization"
        in manuscript,
        "manuscript scope does not distinguish the completed negative primary "
        "from successful generalization",
    )
    require(
        "Independent\nevaluation, multi-seed training and broader "
        "generalization remain outside the\npresent claim"
        not in manuscript,
        "manuscript retains the stale pre-primary scope statement",
    )
    require(
        "| Checkpoint and independent generalization evidence | One legacy "
        "checkpoint plus one completed frozen independent primary that is "
        "negative for its candidate | Blocking for a successful "
        "generalization claim: the completed primary falsifies one candidate "
        "but does not establish an eligible calibration or broader "
        "checkpoint/seed robustness |"
        in readiness,
        "publication readiness does not reconcile the completed negative "
        "primary with the remaining generalization blocker",
    )
    require(
        "| Clean checkpoint and frozen independent evaluation | One legacy "
        "checkpoint and reused development dates |"
        not in readiness,
        "publication readiness retains the stale pre-primary minimum-tier row",
    )

    section_positions = [
        reproducibility.find(heading) for heading in REPRODUCIBILITY_SECTION_ORDER
    ]
    require(
        all(position >= 0 for position in section_positions),
        "reproducibility handoff is missing a required section heading",
    )
    require(
        section_positions == sorted(section_positions),
        "reproducibility handoff sections are out of semantic order",
    )
    latent_start = section_positions[0]
    locked_start = section_positions[1]
    latent_section = reproducibility[latent_start:locked_start]
    require(
        "The compact payload `latent_temperature_1p30_gate_retry1`" in latent_section,
        "latent-temperature compact result is outside its recovery section",
    )
    analog_start = section_positions[2]
    coherent_start = section_positions[3]
    analog_section = reproducibility[analog_start:coherent_start]
    require(
        "latent_temperature_1p30_gate_retry1" not in analog_section,
        "analog-residual section contains latent-temperature evidence",
    )

    missing_current_decision = [
        anchor for anchor in CURRENT_DECISION_ANCHORS if anchor not in readiness
    ]
    require(
        not missing_current_decision,
        "publication readiness is missing current decision-chain anchors: "
        + ", ".join(missing_current_decision),
    )

    external_primary_texts = {
        "PUBLICATION_READINESS.md": readiness,
        "CLAIM_LEDGER.md": claim_ledger,
    }
    for name, anchors in EXTERNAL_PRIMARY_HANDOFF_ANCHORS.items():
        missing_anchors = [
            anchor for anchor in anchors if anchor not in external_primary_texts[name]
        ]
        require(
            not missing_anchors,
            f"{name} is missing frozen external-primary handoff anchors: "
            + ", ".join(missing_anchors),
        )

    validate_external_primary_consistency(
        {
            "PAPER_DRAFT.md": manuscript,
            "CLAIM_LEDGER.md": claim_ledger,
            "REPRODUCIBILITY.md": reproducibility,
        }
    )

    external_reconciliation = (
        PAPER_DIR / "EXTERNAL_PRIMARY_RESULT_RECONCILIATION.md"
    ).read_text(encoding="utf-8")
    missing_external_reconciliation_anchors = [
        anchor
        for anchor in EXTERNAL_PRIMARY_RECONCILIATION_ANCHORS
        if anchor not in external_reconciliation
    ]
    require(
        not missing_external_reconciliation_anchors,
        "EXTERNAL_PRIMARY_RESULT_RECONCILIATION.md is missing anchors: "
        + ", ".join(missing_external_reconciliation_anchors),
    )

    for name in EXTERNAL_PRIMARY_STATE_FILES:
        document = (PAPER_DIR / name).read_text(encoding="utf-8")
        require(
            document.count(EXTERNAL_PRIMARY_STATE_MARKER) == 1,
            f"{name} must contain exactly one pending external-primary state marker",
        )

    research_plan = (PAPER_DIR / "RESEARCH_PLAN.md").read_text(encoding="utf-8")
    missing_closed_primary_anchors = [
        anchor
        for anchor in RESEARCH_PLAN_CLOSED_PRIMARY_ANCHORS
        if anchor not in research_plan
    ]
    require(
        not missing_closed_primary_anchors,
        "research plan reopens the completed independent-primary policy: "
        + ", ".join(missing_closed_primary_anchors),
    )

    stale_readiness = [
        anchor for anchor in STALE_READINESS_ANCHORS if anchor in readiness
    ]
    require(
        not stale_readiness,
        "publication readiness contains stale pre-guidance-result narrative: "
        + ", ".join(stale_readiness),
    )

    missing_handoff_anchors = [
        anchor for anchor in FROZEN_EVALUATION_ANCHORS if anchor not in frozen_handoff
    ]
    require(
        not missing_handoff_anchors,
        "frozen evaluation handoff is missing anchors: "
        + ", ".join(missing_handoff_anchors),
    )

    for name, anchors in AMENDED_PRIMARY_ANCHORS.items():
        document = (PAPER_DIR / name).read_text(encoding="utf-8")
        missing_anchors = [anchor for anchor in anchors if anchor not in document]
        require(
            not missing_anchors,
            f"{name} is missing amended-primary anchors: "
            + ", ".join(missing_anchors),
        )

    claim_ids = [int(value) for value in CLAIM_PATTERN.findall(claim_ledger)]
    require(claim_ids, "claim ledger contains no claim rows")
    require(
        claim_ids == list(range(1, claim_ids[-1] + 1)),
        "claim IDs must be unique and contiguous from C1",
    )

    missing_table_anchors = [
        anchor for anchor in MANUSCRIPT_EVIDENCE_ANCHORS if anchor not in manuscript
    ]
    require(
        not missing_table_anchors,
        "manuscript is missing evidence-table anchors: "
        + ", ".join(missing_table_anchors),
    )

    for name, anchors in FINAL_DIAGNOSTIC_ANCHORS.items():
        document = (PAPER_DIR / name).read_text(encoding="utf-8")
        missing_anchors = [anchor for anchor in anchors if anchor not in document]
        require(
            not missing_anchors,
            f"{name} is missing final-diagnostic anchors: "
            + ", ".join(missing_anchors),
        )

    for name, anchors in CALIBRATION_CONTRACT_ANCHORS.items():
        document = (PAPER_DIR / name).read_text(encoding="utf-8")
        missing_anchors = [anchor for anchor in anchors if anchor not in document]
        require(
            not missing_anchors,
            f"{name} is missing next-baseline anchors: "
            + ", ".join(missing_anchors),
        )

    next_method = (PAPER_DIR / "NEXT_METHOD_CONTRACT.md").read_text(encoding="utf-8")
    missing_next_method = [
        anchor for anchor in NEXT_METHOD_ANCHORS if anchor not in next_method
    ]
    require(
        not missing_next_method,
        "NEXT_METHOD_CONTRACT.md is missing anchors: "
        + ", ".join(missing_next_method),
    )

    rank_coherent_contract = (
        PAPER_DIR / "NEXT_RANK_COHERENT_CONTRACT.md"
    ).read_text(encoding="utf-8")
    missing_rank_coherent = [
        anchor
        for anchor in RANK_COHERENT_CONTRACT_ANCHORS
        if anchor not in rank_coherent_contract
    ]
    require(
        not missing_rank_coherent,
        "NEXT_RANK_COHERENT_CONTRACT.md is missing anchors: "
        + ", ".join(missing_rank_coherent),
    )
    raw_member_reweighting_contract = (
        PAPER_DIR / "NEXT_RAW_MEMBER_REWEIGHTING_CONTRACT.md"
    ).read_text(encoding="utf-8")
    missing_raw_member_reweighting = [
        anchor
        for anchor in RAW_MEMBER_REWEIGHTING_CONTRACT_ANCHORS
        if anchor not in raw_member_reweighting_contract
    ]
    require(
        not missing_raw_member_reweighting,
        "NEXT_RAW_MEMBER_REWEIGHTING_CONTRACT.md is missing anchors: "
        + ", ".join(missing_raw_member_reweighting),
    )
    rank_coherent_checklist = (
        PAPER_DIR / "RANK_COHERENT_RUNNER_REVIEW_CHECKLIST.md"
    ).read_text(encoding="utf-8")
    missing_checklist = [
        anchor
        for anchor in RANK_COHERENT_REVIEW_CHECKLIST_ANCHORS
        if anchor not in rank_coherent_checklist
    ]
    require(
        not missing_checklist,
        "RANK_COHERENT_RUNNER_REVIEW_CHECKLIST.md is missing anchors: "
        + ", ".join(missing_checklist),
    )
    for name, anchors in RANK_COHERENT_HANDOFF_ANCHORS.items():
        document = (PAPER_DIR / name).read_text(encoding="utf-8")
        missing_anchors = [anchor for anchor in anchors if anchor not in document]
        require(
            not missing_anchors,
            f"{name} is missing rank-coherent handoff anchors: "
            + ", ".join(missing_anchors),
        )

    rank_coherent_handoff = (
        PAPER_DIR / "RANK_COHERENT_CONTROLLER_HANDOFF.md"
    ).read_text(encoding="utf-8")
    validate_rank_coherent_immutable_identities(PAPER_DIR, rank_coherent_handoff)

    for name, anchors in ANALOG_RESULT_ANCHORS.items():
        document = (PAPER_DIR / name).read_text(encoding="utf-8")
        missing_anchors = [anchor for anchor in anchors if anchor not in document]
        require(
            not missing_anchors,
            f"{name} is missing analog-result anchors: "
            + ", ".join(missing_anchors),
        )

    for name, anchors in GUIDANCE_RESULT_ANCHORS.items():
        document = (PAPER_DIR / name).read_text(encoding="utf-8")
        missing_anchors = [anchor for anchor in anchors if anchor not in document]
        require(
            not missing_anchors,
            f"{name} is missing guidance-result anchors: "
            + ", ".join(missing_anchors),
        )

    for name, anchors in NEXT_GENERATIVE_METHOD_ANCHORS.items():
        document = (PAPER_DIR / name).read_text(encoding="utf-8")
        missing_anchors = [anchor for anchor in anchors if anchor not in document]
        require(
            not missing_anchors,
            f"{name} is missing next-generative-method anchors: "
            + ", ".join(missing_anchors),
        )

    latent_reconciliation = (
        PAPER_DIR / "LATENT_TEMPERATURE_RESULT_RECONCILIATION.md"
    ).read_text(encoding="utf-8")
    missing_latent_anchors = [
        anchor
        for anchor in LATENT_RECONCILIATION_ANCHORS
        if anchor not in latent_reconciliation
    ]
    require(
        not missing_latent_anchors,
        "LATENT_TEMPERATURE_RESULT_RECONCILIATION.md is missing anchors: "
        + ", ".join(missing_latent_anchors),
    )

    locked_dropout_reconciliation = (
        PAPER_DIR / "LOCKED_MC_DROPOUT_RESULT_RECONCILIATION.md"
    ).read_text(encoding="utf-8")
    missing_locked_dropout_anchors = [
        anchor
        for anchor in LOCKED_DROPOUT_RECONCILIATION_ANCHORS
        if anchor not in locked_dropout_reconciliation
    ]
    require(
        not missing_locked_dropout_anchors,
        "LOCKED_MC_DROPOUT_RESULT_RECONCILIATION.md is missing anchors: "
        + ", ".join(missing_locked_dropout_anchors),
    )

    figures = FIGURE_PATTERN.findall(manuscript)
    require(figures, "manuscript contains no linked figures")
    for relative_name in figures:
        figure = (PAPER_DIR / relative_name).resolve()
        require(
            figure.is_relative_to(PAPER_DIR),
            f"figure escapes paper directory: {relative_name}",
        )
        require(figure.is_file(), f"linked figure does not exist: {relative_name}")
        if figure.suffix.lower() == ".svg":
            ET.parse(figure)
        anchors = FIGURE_TEXT_ANCHORS.get(relative_name, ())
        figure_text = figure.read_text(encoding="utf-8")
        missing_figure_anchors = [anchor for anchor in anchors if anchor not in figure_text]
        require(
            not missing_figure_anchors,
            f"{relative_name} is missing semantic anchors: "
            + ", ".join(missing_figure_anchors),
        )

    reference_heading = "## References\n"
    require(reference_heading in manuscript, "manuscript contains no References section")
    reference_section = manuscript.split(reference_heading, maxsplit=1)[1]
    references = [int(value) for value in REFERENCE_PATTERN.findall(reference_section)]
    require(references, "manuscript contains no numbered references")
    require(
        references == list(range(1, len(references) + 1)),
        "numbered references are not contiguous from 1",
    )
    manuscript_body = manuscript.split(reference_heading, maxsplit=1)[0]
    cited_references = {
        int(value)
        for citation in CITATION_PATTERN.findall(manuscript_body)
        for value in re.split(r"\s*,\s*", citation)
    }
    require(
        cited_references == set(references),
        "reference/citation mismatch: cited="
        + ",".join(map(str, sorted(cited_references)))
        + "; listed="
        + ",".join(map(str, references)),
    )

    require(
        TRACEABILITY_HEADING in manuscript,
        "manuscript contains no claim-ledger traceability section",
    )
    traceability_section = manuscript.split(TRACEABILITY_HEADING, maxsplit=1)[1]
    traceability_section = traceability_section.split("\n### ", maxsplit=1)[0]
    traced_claim_ids = [
        int(value) for value in TRACEABILITY_PATTERN.findall(traceability_section)
    ]
    require(
        len(traced_claim_ids) == len(set(traced_claim_ids)),
        "claim-ledger traceability contains duplicate claim IDs",
    )
    require(
        set(traced_claim_ids) == set(claim_ids),
        "claim-ledger traceability mismatch: traced="
        + ",".join(map(str, sorted(traced_claim_ids)))
        + "; ledger="
        + ",".join(map(str, claim_ids)),
    )

    validate_empirical_traceability(manuscript, PAPER_DIR)

    status = validate_publication_status(readiness, frozen_handoff)

    print(
        f"publication artifact audit passed: {len(REQUIRED_FILES)} files, "
        f"{len(figures)} figures, {len(references)} references, "
        f"{len(claim_ids)} claims, status={status}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ET.ParseError, ValueError) as error:
        print(f"publication artifact audit failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
