#!/usr/bin/env python3
"""Fail closed on broken links and contradictory publication metadata."""

from __future__ import annotations

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
)
FIGURE_PATTERN = re.compile(r"!\[[^]]*\]\(([^)]+)\)")
REFERENCE_PATTERN = re.compile(r"^(\d+)\. ", re.MULTILINE)
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


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> int:
    missing = [name for name in REQUIRED_FILES if not (PAPER_DIR / name).is_file()]
    require(not missing, f"missing required publication files: {', '.join(missing)}")

    manuscript = (PAPER_DIR / "PAPER_DRAFT.md").read_text(encoding="utf-8")
    readiness = (PAPER_DIR / "PUBLICATION_READINESS.md").read_text(encoding="utf-8")

    for name, anchors in FINAL_DIAGNOSTIC_ANCHORS.items():
        document = (PAPER_DIR / name).read_text(encoding="utf-8")
        missing_anchors = [anchor for anchor in anchors if anchor not in document]
        require(
            not missing_anchors,
            f"{name} is missing final-diagnostic anchors: "
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

    reference_heading = "## References\n"
    require(reference_heading in manuscript, "manuscript contains no References section")
    reference_section = manuscript.split(reference_heading, maxsplit=1)[1]
    references = [int(value) for value in REFERENCE_PATTERN.findall(reference_section)]
    require(references, "manuscript contains no numbered references")
    require(
        references == list(range(1, len(references) + 1)),
        "numbered references are not contiguous from 1",
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
    else:
        require(blockers != "none", "not-ready status requires explicit scientific blockers")

    print(
        f"publication artifact audit passed: {len(REQUIRED_FILES)} files, "
        f"{len(figures)} figures, {len(references)} references, status={status}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ET.ParseError, ValueError) as error:
        print(f"publication artifact audit failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
