#!/usr/bin/env python3
"""Fail-closed validation of the two-file E4 engineering sentinel result."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


FEATURE_NAMES = [
    "domain_mean", "domain_standard_deviation", "ice_extent",
    "established_ice_area", "meridional_centroid", "zonal_centroid",
]
STATUS = {"status": "completed", "selection_role": "engineering_only", "cases": 8,
          "scientific_gate": False}
MANIFEST_KEYS = {
    "config_sha256", "feature_names", "training_records", "analog_indices",
    "analog_distances", "training_only_standardization", "chronological_ties",
    "complete_field_shape", "result_shape", "clipping_calls", "raw_arrays",
}


def validate_result(status: Mapping[str, Any], manifest: Mapping[str, Any],
                    config_bytes: bytes) -> str:
    if status != STATUS or set(status) != set(STATUS):
        raise ValueError("run_status must record the completed non-scientific eight-case sentinel")
    if set(manifest) != MANIFEST_KEYS:
        raise ValueError("artifact_manifest keys must be exact")
    digest = manifest["config_sha256"]
    if (not isinstance(digest, str) or len(digest) != 64 or
            any(c not in "0123456789abcdef" for c in digest)):
        raise ValueError("config_sha256 must be one lowercase SHA-256 digest")
    if digest != hashlib.sha256(config_bytes).hexdigest():
        raise ValueError("config_sha256 does not bind the supplied frozen config")
    if manifest["feature_names"] != FEATURE_NAMES:
        raise ValueError("feature_names differ from the frozen six-feature contract")
    if manifest["training_records"] != 12 or isinstance(manifest["training_records"], bool):
        raise ValueError("training_records must equal the frozen fixture size")

    indices, distances = manifest["analog_indices"], manifest["analog_distances"]
    if not isinstance(indices, list) or not isinstance(distances, list) or len(indices) != 8 or len(distances) != 8:
        raise ValueError("analog evidence must contain exactly eight cases")
    for case, (index_row, distance_row) in enumerate(zip(indices, distances)):
        if not isinstance(index_row, list) or not isinstance(distance_row, list) or len(index_row) != 10 or len(distance_row) != 10:
            raise ValueError(f"case {case}: analog evidence must contain exactly ten neighbors")
        if any(isinstance(i, bool) or not isinstance(i, int) or i < 0 or i >= 12 for i in index_row):
            raise ValueError(f"case {case}: analog index is outside the training inventory")
        if len(set(index_row)) != 10:
            raise ValueError(f"case {case}: analog indices must be distinct")
        if any(isinstance(d, bool) or not isinstance(d, (int, float)) or
               not math.isfinite(float(d)) or float(d) < 0 for d in distance_row):
            raise ValueError(f"case {case}: analog distances must be finite and non-negative")
        if any(float(a) > float(b) for a, b in zip(distance_row, distance_row[1:])):
            raise ValueError(f"case {case}: analog distances must be ordered")
        for pos in range(1, 10):
            if float(distance_row[pos - 1]) == float(distance_row[pos]) and index_row[pos - 1] > index_row[pos]:
                raise ValueError(f"case {case}: equal-distance ties must be chronological")

    literals = {
        "training_only_standardization": True, "chronological_ties": True,
        "complete_field_shape": [5, 4], "result_shape": [8, 10, 5, 4],
        "clipping_calls": 0, "raw_arrays": "not_persisted",
    }
    for key, expected in literals.items():
        if manifest[key] != expected or (key == "clipping_calls" and isinstance(manifest[key], bool)):
            raise ValueError(f"{key} disagrees with the frozen E4 sentinel")
    return "E4_SENTINEL_PASS"


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
