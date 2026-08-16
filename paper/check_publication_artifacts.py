#!/usr/bin/env python3
"""Fail closed on broken links and contradictory publication metadata."""

from __future__ import annotations

import ast
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


PAPER_DIR = Path(__file__).resolve().parent
REQUIRED_FILES = (
    "PAPER_DRAFT.md",
    "CLAIM_LEDGER.md",
    "REPRODUCIBILITY.md",
    "RESEARCH_PLAN.md",
    "PUBLICATION_READINESS.md",
    "NEXT_BASELINE_CONTRACT.md",
    "FROZEN_EVALUATION_HANDOFF.md",
    "check_publication_artifacts.py",
    "make_calibration_summary_figure.py",
    "make_case_level_artifacts.py",
    "make_joint_gate_figure.py",
)
PYTHON_FILES = tuple(name for name in REQUIRED_FILES if name.endswith(".py"))
FIGURE_PATTERN = re.compile(r"!\[[^]]*\]\(([^)]+)\)")
REFERENCE_PATTERN = re.compile(r"^(\d+)\. ", re.MULTILINE)
CITATION_PATTERN = re.compile(r"\[([1-9]\d*(?:\s*,\s*[1-9]\d*)*)\]")
CLAIM_PATTERN = re.compile(r"^\| C(\d+) \|", re.MULTILINE)
FINAL_DIAGNOSTIC_ANCHORS = {
    "PAPER_DRAFT.md": (
        "0.055738",
        "0.059107",
        "0.071936",
        "final fixed open-logit diagnostic",
    ),
    "CLAIM_LEDGER.md": (
        "| C26 |",
        "| C27 |",
        "0.0557378",
        "0.0591071",
        "0.0719362",
        "No post-hoc tuning follows.",
    ),
    "REPRODUCIBILITY.md": (
        "0.0557378026",
        "0.0591071355",
        "0.0719361803",
        "must not be tuned after this result",
    ),
    "PUBLICATION_READINESS.md": (
        "0.0557378",
        "0.0591071",
        "0.0719362",
        "no post-hoc tuning is admissible",
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
        "not executed",
    ),
    "CLAIM_LEDGER.md": (
        "| C28 |",
        "| C29 |",
        "| C30 |",
        "0.0585570",
        "overall_eligible=false",
        "only operational validity passes at family level",
        "conformal and probabilistic-DA families",
    ),
    "PAPER_DRAFT.md": (
        "| ZOIB-EMOS/ECC-Q diagnostic | Raw ensemble | Candidate | Interpretation |",
        "| Ordinary CRPS | 0.062108 | 0.064991 | 4.64% worse |",
        "| Inner-order attainable-coverage error | 0.175953 | 0.253947 | Reliability family fails |",
        "| Mean IIEE | 0.079600 | 0.087190 | Spatial/physical family fails |",
        "topology-preserving stratified transport",
        "planned falsifiable mechanism",
    ),
    "REPRODUCIBILITY.md": (
        "## Frozen ZOIB-EMOS/ECC-Q handoff",
        "exactly 13 fitted coefficients",
        "0.0642804809",
        "Only operational validity passes at family level",
        "overall_eligible=false",
        "## Next frozen runner contract",
        "topology-preserving stratified transport",
    ),
    "PUBLICATION_READINESS.md": (
        "No additional implemented mode",
        "0.0585570",
        "Only operational validity passes at family level",
        "topology-preserving stratified-transport contract",
        "No experiment was launched",
    ),
}
MANUSCRIPT_EVIDENCE_ANCHORS = (
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
)
FROZEN_EVALUATION_ANCHORS = (
    "Status: BLOCKED_PENDING_EXTERNAL_AUTHORIZATION",
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


def main() -> int:
    missing = [name for name in REQUIRED_FILES if not (PAPER_DIR / name).is_file()]
    require(not missing, f"missing required publication files: {', '.join(missing)}")

    for name in PYTHON_FILES:
        source = (PAPER_DIR / name).read_text(encoding="utf-8")
        ast.parse(source, filename=name)

    manuscript = (PAPER_DIR / "PAPER_DRAFT.md").read_text(encoding="utf-8")
    claim_ledger = (PAPER_DIR / "CLAIM_LEDGER.md").read_text(encoding="utf-8")
    readiness = (PAPER_DIR / "PUBLICATION_READINESS.md").read_text(encoding="utf-8")
    frozen_handoff = (PAPER_DIR / "FROZEN_EVALUATION_HANDOFF.md").read_text(
        encoding="utf-8"
    )

    missing_handoff_anchors = [
        anchor for anchor in FROZEN_EVALUATION_ANCHORS if anchor not in frozen_handoff
    ]
    require(
        not missing_handoff_anchors,
        "frozen evaluation handoff is missing anchors: "
        + ", ".join(missing_handoff_anchors),
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
    if status == "READY_FOR_HUMAN_REVIEW":
        require(blockers == "none", "ready status requires no scientific blockers")
        require(
            "Status: BLOCKED_PENDING_EXTERNAL_AUTHORIZATION" not in frozen_handoff,
            "ready status contradicts blocked frozen evaluation handoff",
        )
    else:
        require(blockers != "none", "not-ready status requires explicit scientific blockers")
        require(
            "Status: BLOCKED_PENDING_EXTERNAL_AUTHORIZATION" in frozen_handoff,
            "not-ready status requires the blocked frozen evaluation handoff",
        )

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
