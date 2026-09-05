#!/usr/bin/env python3
"""Fail-closed validation of the compact E1 engineering sentinel result."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


STATUS = {
    "status": "completed",
    "selection_role": "engineering_only",
    "cases": 8,
    "clearml_enabled": True,
    "scientific_gate": False,
}
MANIFEST_KEYS = {
    "config_sha256", "conditioning_shape", "target_shape", "sample_shape",
    "exact_one_policy", "sample_finite", "sample_in_unit_interval",
    "hard_clip_count", "masked_channel_leakage", "provenance_consistent",
    "orientation_landmarks", "candidate_track_imprint_ratio",
    "background_track_imprint_ratio", "panel_artifacts", "raw_arrays",
}
PANEL_ARTIFACTS = [f"case_{case:02d}_panel.png" for case in range(8)]


def _finite_numbers(value: Any, *, name: str, count: int) -> list[float]:
    if not isinstance(value, list) or len(value) != count:
        raise ValueError(f"{name} must contain exactly {count} values")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) or
           not math.isfinite(float(item)) for item in value):
        raise ValueError(f"{name} must contain finite numbers")
    return [float(item) for item in value]


def validate_result(status: Mapping[str, Any], manifest: Mapping[str, Any],
                    config_bytes: bytes) -> str:
    if status != STATUS or set(status) != set(STATUS):
        raise ValueError("run_status must record the completed non-scientific eight-case sentinel")
    if set(manifest) != MANIFEST_KEYS:
        raise ValueError("artifact_manifest keys must be exact")
    digest = manifest["config_sha256"]
    if (not isinstance(digest, str) or len(digest) != 64 or
            any(char not in "0123456789abcdef" for char in digest) or
            digest != hashlib.sha256(config_bytes).hexdigest()):
        raise ValueError("config_sha256 must bind the supplied frozen config")

    literals = {
        "conditioning_shape": [8, 24, 5, 4],
        "target_shape": [8, 2, 5, 4],
        "sample_shape": [8, 10, 1, 5, 4],
        "exact_one_policy": "explicit_exact_one_atom",
        "sample_finite": True,
        "sample_in_unit_interval": True,
        "hard_clip_count": 0,
        "provenance_consistent": [True] * 8,
        "orientation_landmarks": [[True, True] for _ in range(8)],
        "panel_artifacts": PANEL_ARTIFACTS,
        "raw_arrays": "not_persisted",
    }
    for key, expected in literals.items():
        if manifest[key] != expected:
            raise ValueError(f"{key} disagrees with the frozen E1 sentinel")
    if isinstance(manifest["hard_clip_count"], bool):
        raise ValueError("hard_clip_count must be an integer count")

    leakage = _finite_numbers(manifest["masked_channel_leakage"],
                              name="masked_channel_leakage", count=8)
    if any(value < 0 or value > 1e-7 for value in leakage):
        raise ValueError("masked_channel_leakage exceeds the frozen limit")
    candidate = _finite_numbers(manifest["candidate_track_imprint_ratio"],
                                name="candidate_track_imprint_ratio", count=8)
    background = _finite_numbers(manifest["background_track_imprint_ratio"],
                                 name="background_track_imprint_ratio", count=8)
    if any(value < 0 for value in candidate + background):
        raise ValueError("track-imprint ratios must be non-negative")
    if any(test > 1.05 * base for test, base in zip(candidate, background)):
        raise ValueError("candidate track-imprint ratio exceeds the frozen per-case limit")
    return "E1_SENTINEL_PASS"


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
