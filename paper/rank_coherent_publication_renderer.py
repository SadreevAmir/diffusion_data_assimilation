"""Render complete publication replacements from one admitted compact result."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from pathlib import Path

from .rank_coherent_result_reconciliation import FAMILIES, canonical_marker
from .validate_rank_coherent_admission import load_and_validate_combined

IDENTIFIER = re.compile(r"[a-z0-9_]{1,64}")
COMPACT_FAMILY = {
    "proper_score": "proper_score",
    "reliability": "finite_ensemble_reliability",
    "boundary": "boundary",
    "spatial_physical": "spatial_physical",
    "operational": "operational",
}
RESULT_BLOCK = re.compile(
    r"\n<!-- RANK_COHERENT_RENDERED_RESULT_START -->.*?"
    r"<!-- RANK_COHERENT_RENDERED_RESULT_END -->\n?",
    re.DOTALL,
)
PRE_RESULT_STATUS = "Status: PRE_RESULT_NO_TRUSTED_MODE"
GUARD_ROW = re.compile(
    r"^\| `eligible_calibration` \| `(?:MISSING_ELIGIBLE_RESULT|ELIGIBLE)` \| "
    r"`(?:NONE|[0-9a-f]{64})` \| `(?:BLOCKED|DECISION_BEARING)` \|$",
    re.MULTILINE,
)


def _load_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain one JSON object")
    return value


def _identity(metadata: Mapping[str, object]) -> tuple[str, str]:
    experiment_id = metadata.get("experiment_id")
    candidate = metadata.get("candidate")
    if not isinstance(experiment_id, str) or not IDENTIFIER.fullmatch(experiment_id):
        raise ValueError("metadata must contain a safe experiment_id")
    if not isinstance(candidate, str) or not IDENTIFIER.fullmatch(candidate):
        raise ValueError("metadata must contain a safe candidate")
    return experiment_id, candidate


def _result_block(marker: str, aggregate: Mapping[str, object]) -> str:
    metrics = aggregate.get("aggregate_metrics")
    paired = aggregate.get("paired_uncertainty")
    gate = aggregate.get("gate")
    if not isinstance(metrics, Mapping) or not isinstance(paired, Mapping) or not isinstance(gate, Mapping):
        raise ValueError("compact result omits publication sections")
    rows = []
    for name in sorted(metrics):
        record, uncertainty = metrics[name], paired.get(name)
        if not isinstance(record, Mapping) or not isinstance(uncertainty, Mapping):
            raise ValueError("every effect size requires paired uncertainty")
        required = (record.get("raw"), record.get("candidate"), record.get("delta"))
        intervals = (uncertainty.get("paired_date_95_ci"), uncertainty.get("four_case_block_95_ci"))
        if any(type(value) not in (int, float) or type(value) is bool or not math.isfinite(value) for value in required):
            raise ValueError("every effect size must be finite")
        if any(not isinstance(value, list) or len(value) != 2 for value in intervals):
            raise ValueError("every effect size requires both uncertainty intervals")
        rows.append(
            f"| `{name}` | {required[0]:.12g} | {required[1]:.12g} | {required[2]:+.12g} | "
            f"[{intervals[0][0]:.12g}, {intervals[0][1]:.12g}] | "
            f"[{intervals[1][0]:.12g}, {intervals[1][1]:.12g}] |"
        )
    decisions = {name: gate.get(COMPACT_FAMILY[name]) for name in FAMILIES}
    if any(type(value) is not bool for value in decisions.values()):
        raise ValueError("compact result omits literal family decisions")
    failed = [name for name in FAMILIES if not decisions[name]]
    interpretation = (
        "Ни одно обязательное семейство не провалено; результат допускает положительную ветвь."
        if not failed else
        "Отрицательная ветвь обязательна: провалены семейства " + ", ".join(f"`{name}`" for name in failed)
        + "; улучшения других метрик не компенсируют этот провал и blocker не закрывается."
    )
    decision_text = "; ".join(f"`{name}`={'true' if decisions[name] else 'false'}" for name in FAMILIES)
    return (
        "\n<!-- RANK_COHERENT_RENDERED_RESULT_START -->\n"
        "## Допущенный rank-coherent результат\n\n"
        f"{marker}\n\n"
        "| metric | raw | candidate | delta | paired date 95% CI | four-case block 95% CI |\n"
        "|---|---:|---:|---:|---|---|\n" + "\n".join(rows) + "\n\n"
        f"Решения семейств: {decision_text}.\n\n{interpretation}\n"
        "<!-- RANK_COHERENT_RENDERED_RESULT_END -->\n"
    )


def render_publication_documents(
    base_documents: Mapping[Path, str], *, reconciliation_path: Path,
    record_path: Path, compact_directory: Path,
) -> tuple[dict[Path, str], dict[str, object]]:
    """Return five full replacements derived only from re-admitted compact bytes."""
    admission = load_and_validate_combined(record_path, compact_directory)
    metadata = _load_object(compact_directory / "metadata.json")
    aggregate = _load_object(compact_directory / "aggregate_case_mean_metrics.json")
    experiment_id, candidate = _identity(metadata)
    gate = aggregate.get("gate")
    if not isinstance(gate, Mapping):
        raise ValueError("compact result omits gate")
    decisions = {name: gate.get(COMPACT_FAMILY[name]) for name in FAMILIES}
    overall = gate.get("overall_eligible")
    marker = canonical_marker(
        record_path, compact_directory, experiment_id=experiment_id, candidate=candidate,
        family_decisions=decisions, overall_eligible=overall,
    )
    required = {"PAPER_DRAFT.md", "CLAIM_LEDGER.md", "REPRODUCIBILITY.md",
                "PUBLICATION_READINESS.md", reconciliation_path.name}
    if len(base_documents) != 5 or {path.name for path in base_documents} != required:
        raise ValueError("renderer requires exactly five publication surfaces")
    block = _result_block(marker, aggregate)
    rendered = {}
    for path, text in base_documents.items():
        updated = RESULT_BLOCK.sub("\n", text)
        if path == reconciliation_path:
            if PRE_RESULT_STATUS not in updated:
                raise ValueError("reconciliation surface omits the pre-result status")
            status = "RECONCILED_POSITIVE" if overall else "RECONCILED_NEGATIVE"
            updated = updated.replace(PRE_RESULT_STATUS, f"Status: {status}", 1)
        if overall and path.name in {
            "PAPER_DRAFT.md", "CLAIM_LEDGER.md", "REPRODUCIBILITY.md",
            "PUBLICATION_READINESS.md",
        }:
            guard = (
                "| `eligible_calibration` | `ELIGIBLE` | "
                f"`{admission.admission_record_sha256}` | `DECISION_BEARING` |"
            )
            updated, count = GUARD_ROW.subn(guard, updated)
            if count != 1:
                raise ValueError(f"{path.name} omits one eligible-calibration guard")
        if overall and path.name == "PAPER_DRAFT.md":
            updated = updated.rstrip() + (
                "\n\nEligible calibration decision: `ELIGIBLE`; compact record: "
                f"`{admission.admission_record_sha256}`; claim role: `DECISION_BEARING`.\n"
            )
        if overall and path.name == "REPRODUCIBILITY.md":
            updated = updated.rstrip() + (
                "\n\nEligible calibration reproducibility identity: "
                f"`{admission.admission_record_sha256}`; verification: `HASH_VERIFIED`.\n"
            )
        if overall and path.name == "PUBLICATION_READINESS.md":
            updated = updated.replace(
                "| Eligible spatially preserving calibration | `RESEARCH_PLAN.md` | "
                "MISSING_ELIGIBLE_RESULT |",
                "| Eligible spatially preserving calibration | `RESEARCH_PLAN.md` | ELIGIBLE |",
                1,
            )
            updated = updated.replace(
                "Required scientific blockers: an eligible spatially preserving calibration and\n"
                "the remaining minimum-tier comparisons",
                "Required scientific blockers: the remaining minimum-tier comparisons",
                1,
            )
        rendered[path] = updated.rstrip() + "\n" + block
    return rendered, {
        "experiment_id": experiment_id, "candidate": candidate,
        "family_decisions": decisions, "overall_eligible": overall,
    }
