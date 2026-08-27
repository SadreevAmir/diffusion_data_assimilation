#!/usr/bin/env python3
"""Fail closed unless the complete local rank-coherent review package is frozen."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


PAPER_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = PAPER_DIR / "rank_coherent_admission_manifest.json"
SCHEMA_VERSION = "rank-coherent-admission-manifest-v1"
EXPECTED_FILES = frozenset({
    "NEXT_RANK_COHERENT_CONTRACT.md",
    "RANK_COHERENT_ADAPTER_SPEC.md",
    "RANK_COHERENT_CONTROLLER_HANDOFF.md",
    "RANK_COHERENT_RUNNER_REVIEW_CHECKLIST.md",
    "rank_coherent_adapter_parity.py",
    "rank_coherent_reference.py",
    "rank_coherent_runner_prototype.py",
    "test_rank_coherent_adapter_parity.py",
    "test_rank_coherent_runner_prototype.py",
    "test_validate_rank_coherent_admission.py",
    "test_validate_rank_coherent_manifest.py",
    "validate_rank_coherent_admission.py",
    "validate_rank_coherent_manifest.py",
})


def validate_manifest(
    path: Path = MANIFEST_PATH,
    root: Path = PAPER_DIR,
    expected_files: frozenset[str] = EXPECTED_FILES,
) -> int:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or set(manifest) != {"schema_version", "files"}:
        raise ValueError("manifest must have the exact top-level schema")
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("manifest schema_version is not frozen")
    files = manifest["files"]
    if not isinstance(files, dict) or not files:
        raise ValueError("manifest files must be a non-empty object")
    if set(files) != expected_files:
        raise ValueError("manifest files must equal the frozen package allowlist")
    for name, expected in files.items():
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or not isinstance(expected, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected) is None
        ):
            raise ValueError("manifest entry is malformed")
        candidate = root / name
        if not candidate.is_file():
            raise ValueError(f"manifest file is missing: {name}")
        observed = hashlib.sha256(candidate.read_bytes()).hexdigest()
        if observed != expected:
            raise ValueError(f"manifest digest mismatch: {name}")
    return len(files)


def main() -> None:
    count = validate_manifest()
    print(f"rank_coherent_admission_manifest=PASS files={count}")


if __name__ == "__main__":
    main()
