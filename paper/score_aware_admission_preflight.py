#!/usr/bin/env python3
"""Fail-closed local preflight for score-aware trusted-mode admission."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


REQUIRED_STATUS = {
    "admission": "NO_GO",
    "reviewed_mode": None,
    "proposal_authorized": False,
}


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def preflight(repository: Path, handoff_path: Path, visible_modes: frozenset[str]) -> dict:
    handoff = json.loads(handoff_path.read_text())
    reasons: list[str] = []

    for key, expected in REQUIRED_STATUS.items():
        if handoff.get(key) != expected:
            reasons.append(f"UNEXPECTED_HANDOFF_{key.upper()}")

    commit = handoff.get("publication_commit")
    if not isinstance(commit, str) or len(commit) != 40:
        reasons.append("INVALID_PUBLICATION_COMMIT")
    else:
        ancestry = subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
            cwd=repository,
            check=False,
            capture_output=True,
        )
        if ancestry.returncode != 0:
            reasons.append("PUBLICATION_COMMIT_NOT_ANCESTOR")

    artifact_sha256 = handoff.get("artifact_sha256")
    if not isinstance(artifact_sha256, dict) or not artifact_sha256:
        reasons.append("MISSING_ARTIFACT_INVENTORY")
    else:
        for filename, expected_digest in sorted(artifact_sha256.items()):
            path = handoff_path.parent / filename
            if not path.is_file() or _sha256(path.read_bytes()) != expected_digest:
                reasons.append(f"WORKTREE_ARTIFACT_DRIFT:{filename}")
                continue
            if isinstance(commit, str) and len(commit) == 40:
                frozen = subprocess.run(
                    ["git", "show", f"{commit}:paper/{filename}"],
                    cwd=repository,
                    check=False,
                    capture_output=True,
                )
                if frozen.returncode != 0 or _sha256(frozen.stdout) != expected_digest:
                    reasons.append(f"PUBLICATION_ARTIFACT_DRIFT:{filename}")

    reviewed_mode = handoff.get("reviewed_mode")
    if reviewed_mode is None:
        reasons.append("MISSING_LITERAL_REVIEWED_MODE")
    elif reviewed_mode not in visible_modes:
        reasons.append("REVIEWED_MODE_NOT_CONTROLLER_VISIBLE")

    local_reasons = [reason for reason in reasons if reason != "MISSING_LITERAL_REVIEWED_MODE"]
    return {
        "schema_version": 1,
        "preflight": "PASS_LOCAL_BOUNDARY" if not local_reasons else "FAIL_LOCAL_BOUNDARY",
        "proposal_authorized": False,
        "reasons": reasons,
        "next_operation": (
            "REGISTER_LITERAL_MODE_AND_RUN_ATOMIC_INDEPENDENT_ADMISSION"
            if reasons == ["MISSING_LITERAL_REVIEWED_MODE"]
            else "REPAIR_LISTED_LOCAL_BOUNDARY_FAILURES"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--controller-visible-mode", action="append", default=[])
    args = parser.parse_args()
    result = preflight(args.repository, args.handoff, frozenset(args.controller_visible_mode))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
