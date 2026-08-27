#!/usr/bin/env python3
"""Fail-closed downstream consumer for probabilistic-DA admission JSON."""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from typing import Mapping

OUTCOMES = {"PROBABILISTIC_DA_USEFUL", "PROBABILISTIC_DA_NEGATIVE"}
SHA256_RE = re.compile(r"[0-9a-f]{64}")

def reconcile_admission(payload: Mapping[str, object], expected_digest: str) -> str:
    if set(payload) != {"outcome", "compact_directory_sha256"}:
        raise ValueError("admission JSON must contain exactly outcome and compact_directory_sha256")
    outcome = payload["outcome"]
    digest = payload["compact_directory_sha256"]
    if outcome not in OUTCOMES:
        raise ValueError("admission outcome is not decision-bearing")
    if not isinstance(expected_digest, str) or SHA256_RE.fullmatch(expected_digest) is None:
        raise ValueError("expected compact_directory_sha256 is invalid")
    if digest != expected_digest:
        raise ValueError("compact_directory_sha256 does not match the admitted bundle")
    return str(outcome)

def main() -> None:
    parser = argparse.ArgumentParser(description="Consume admission for one expected compact digest.")
    parser.add_argument("admission_json", type=Path)
    parser.add_argument("expected_compact_directory_sha256")
    args = parser.parse_args()
    payload = json.loads(args.admission_json.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("admission JSON must be an object")
    print(reconcile_admission(payload, args.expected_compact_directory_sha256))

if __name__ == "__main__":
    main()
