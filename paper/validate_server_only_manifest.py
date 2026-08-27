#!/usr/bin/env python3
"""Validate one server-only compact artifact against its provenance sidecar."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "server_only_compact_manifest_v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_manifest(artifact: Path, manifest: Path, expected_experiment: str) -> None:
    """Fail closed unless the sidecar binds the exact producer, file and bytes."""
    require(artifact.is_file(), f"compact artifact is missing: {artifact.name}")
    require(manifest.is_file(), f"sidecar manifest is missing: {manifest.name}")
    try:
        payload: Any = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("sidecar manifest is not valid UTF-8 JSON") from exc
    require(isinstance(payload, dict), "sidecar manifest must be a JSON object")
    require(
        set(payload) == {"schema_version", "experiment_id", "artifacts"},
        "sidecar manifest has missing or extra top-level fields",
    )
    require(payload["schema_version"] == SCHEMA_VERSION, "sidecar schema mismatch")
    require(
        payload["experiment_id"] == expected_experiment,
        "sidecar producer experiment mismatch",
    )
    artifacts = payload["artifacts"]
    require(isinstance(artifacts, dict), "sidecar artifacts must be an object")
    require(artifact.name in artifacts, "sidecar hash entry is missing")
    require(
        set(artifacts) == {artifact.name},
        "sidecar must bind exactly the requested compact artifact",
    )
    digest = artifacts[artifact.name]
    require(
        isinstance(digest, str)
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest),
        "sidecar digest must be lowercase SHA-256",
    )
    observed = hashlib.sha256(artifact.read_bytes()).hexdigest()
    require(observed == digest, "compact artifact SHA-256 mismatch")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("expected_experiment")
    args = parser.parse_args()
    try:
        validate_manifest(args.artifact, args.manifest, args.expected_experiment)
    except ValueError as exc:
        parser.error(str(exc))
    print("server-only compact manifest: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
