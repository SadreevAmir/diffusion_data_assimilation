#!/usr/bin/env python3
"""Fail-closed validation of the two-file E3 engineering sentinel result."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


STATUS_KEYS = {"status", "selection_role", "cases", "scientific_gate"}
MANIFEST_KEYS = {
    "config_sha256", "training_interior_count", "unique_knot_count",
    "roundtrip_max_abs_error", "zero_atoms_preserved", "one_atoms_preserved",
    "clipping_calls", "raw_arrays",
}


def _digest(value: object) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("config_sha256 must be one lowercase SHA-256 digest")
    return value


def validate_result(status: Mapping[str, Any], manifest: Mapping[str, Any], config_bytes: bytes) -> str:
    if set(status) != STATUS_KEYS:
        raise ValueError("run_status keys must be exact")
    if status != {"status": "completed", "selection_role": "engineering_only", "cases": 8,
                  "scientific_gate": False}:
        raise ValueError("run_status must record the completed non-scientific eight-case sentinel")
    if set(manifest) != MANIFEST_KEYS:
        raise ValueError("artifact_manifest keys must be exact")
    if _digest(manifest["config_sha256"]) != hashlib.sha256(config_bytes).hexdigest():
        raise ValueError("config_sha256 does not bind the supplied frozen config")
    for key in ("training_interior_count", "unique_knot_count", "clipping_calls"):
        if isinstance(manifest[key], bool) or not isinstance(manifest[key], int):
            raise ValueError(f"{key} must be an integer")
    if manifest["training_interior_count"] != 7 or manifest["unique_knot_count"] != 6:
        raise ValueError("training support counts disagree with the frozen sentinel fixture")
    error = manifest["roundtrip_max_abs_error"]
    if isinstance(error, bool) or not isinstance(error, (int, float)) or not math.isfinite(float(error)):
        raise ValueError("roundtrip_max_abs_error must be finite numeric")
    if float(error) < 0 or float(error) > 1e-7:
        raise ValueError("roundtrip_max_abs_error exceeds the frozen tolerance")
    if manifest["zero_atoms_preserved"] is not True or manifest["one_atoms_preserved"] is not True:
        raise ValueError("both boundary atoms must be preserved")
    if manifest["clipping_calls"] != 0:
        raise ValueError("clipping_calls must equal zero")
    if manifest["raw_arrays"] != "not_persisted":
        raise ValueError("raw arrays must not be persisted by the engineering sentinel")
    return "E3_SENTINEL_PASS"


def _object(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: top-level JSON must be an object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("run_status", type=Path)
    parser.add_argument("artifact_manifest", type=Path)
    args = parser.parse_args()
    decision = validate_result(_object(args.run_status), _object(args.artifact_manifest),
                               args.config.read_bytes())
    print(json.dumps({"decision": decision, "launch_authorized": False}, sort_keys=True))


if __name__ == "__main__":
    main()
