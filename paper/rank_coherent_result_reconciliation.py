"""Fail-closed publication reconciliation for an admitted rank-coherent result."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

from .atomic_publish import publish_text_artifacts
from .validate_rank_coherent_admission import load_and_validate_combined

FAMILIES = ("proper_score", "reliability", "boundary", "spatial_physical", "operational")
MARKER = re.compile(r"^RANK_COHERENT_RESULT: status=RECONCILED_(?:POSITIVE|NEGATIVE); .+$", re.MULTILINE)


def canonical_marker(record_path: Path, compact_directory: Path, *,
                     experiment_id: str, candidate: str,
                     family_decisions: Mapping[str, bool], overall_eligible: bool) -> str:
    """Build the only marker allowed downstream of combined admission."""
    admission = load_and_validate_combined(record_path, compact_directory)
    if not re.fullmatch(r"[a-z0-9_]{1,64}", experiment_id) or not re.fullmatch(r"[a-z0-9_]{1,64}", candidate):
        raise ValueError("experiment and candidate must be safe identifiers")
    if set(family_decisions) != set(FAMILIES) or any(type(value) is not bool for value in family_decisions.values()):
        raise ValueError("family decisions must contain exactly five literal Booleans")
    conjunction = all(family_decisions[name] for name in FAMILIES)
    if type(overall_eligible) is not bool or overall_eligible is not conjunction:
        raise ValueError("overall_eligible contradicts mandatory families")
    status = "RECONCILED_POSITIVE" if overall_eligible else "RECONCILED_NEGATIVE"
    families = "; ".join(f"{name}={'true' if family_decisions[name] else 'false'}" for name in FAMILIES)
    return (f"RANK_COHERENT_RESULT: status={status}; reviewed_mode={admission.reviewed_mode}; "
            f"experiment_id={experiment_id}; candidate={candidate}; "
            f"admission_record_sha256={admission.admission_record_sha256}; "
            f"compact_directory_sha256={admission.compact_directory_sha256}; "
            f"completed_cases=40; ensemble_size=10; {families}; "
            f"overall_eligible={'true' if overall_eligible else 'false'}")


def publish_reconciliation(
    documents: Mapping[Path, str], *, reconciliation_path: Path,
    record_path: Path, compact_directory: Path, experiment_id: str, candidate: str,
    family_decisions: Mapping[str, bool], overall_eligible: bool,
) -> None:
    """Validate a complete five-surface transition, then publish with rollback."""
    marker = canonical_marker(
        record_path, compact_directory, experiment_id=experiment_id, candidate=candidate,
        family_decisions=family_decisions, overall_eligible=overall_eligible,
    )
    resolved = {path.resolve(): text for path, text in documents.items()}
    required_names = {"PAPER_DRAFT.md", "CLAIM_LEDGER.md", "REPRODUCIBILITY.md",
                      "PUBLICATION_READINESS.md", reconciliation_path.name}
    if len(resolved) != 5 or {path.name for path in resolved} != required_names:
        raise ValueError("reconciliation requires exactly five publication surfaces")
    if reconciliation_path.resolve() not in resolved:
        raise ValueError("reconciliation surface is missing")
    for path, content in resolved.items():
        if MARKER.findall(content) != [marker]:
            raise ValueError(f"{path.name} must contain exactly the canonical marker")
    publish_text_artifacts(resolved.items())
