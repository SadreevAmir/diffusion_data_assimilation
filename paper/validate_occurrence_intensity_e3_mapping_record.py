#!/usr/bin/env python3
"""Fail-closed validation of a controller-visible E3 admission mapping record."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

from paper.validate_occurrence_intensity_e3_compact_result import validate_result

RECORD_KEYS = {
    "upstream_base", "upstream_decision", "reviewed_mode", "publication_commit",
    "runner_sha256", "contract_sha256", "config_sha256", "run_status_sha256",
    "artifact_manifest_sha256", "test_command", "test_sentinel",
    "decision_bearing_validation", "deviations",
}
SAFE_ID = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
SAFE_MODE = re.compile(r"^[a-z][a-z0-9_]{0,127}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _nonempty(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def validate_mapping_record(record: Mapping[str, Any], contract_bytes: bytes, config_bytes: bytes,
                            status_bytes: bytes, manifest_bytes: bytes) -> str:
    if set(record) != RECORD_KEYS:
        raise ValueError("mapping record keys must be exact")
    if not SAFE_ID.fullmatch(_nonempty(record["upstream_base"], "upstream_base")):
        raise ValueError("upstream_base must be one literal safe identifier")
    if record["upstream_decision"] not in {"E1_ACCEPTED", "TEMPORAL_MECHANISM_USEFUL"}:
        raise ValueError("upstream_decision must select an accepted frozen base")
    if not SAFE_MODE.fullmatch(_nonempty(record["reviewed_mode"], "reviewed_mode")):
        raise ValueError("reviewed_mode must be one literal safe mode identifier")
    if not isinstance(record["publication_commit"], str) or not HEX40.fullmatch(record["publication_commit"]):
        raise ValueError("publication_commit must be one lowercase 40-hex commit")
    bindings = {
        "contract_sha256": contract_bytes, "config_sha256": config_bytes,
        "run_status_sha256": status_bytes, "artifact_manifest_sha256": manifest_bytes,
    }
    for label in ("runner_sha256", *bindings):
        if not isinstance(record[label], str) or not HEX64.fullmatch(record[label]):
            raise ValueError(f"{label} must be one lowercase SHA-256 digest")
    for label, content in bindings.items():
        if record[label] != hashlib.sha256(content).hexdigest():
            raise ValueError(f"{label} does not bind the supplied artifact")
    status = json.loads(status_bytes); manifest = json.loads(manifest_bytes)
    if not isinstance(status, dict) or not isinstance(manifest, dict):
        raise ValueError("compact artifacts must be JSON objects")
    validate_result(status, manifest, config_bytes)
    if record["test_command"] != "python3 test/test_occurrence_intensity_e3.py":
        raise ValueError("test_command must name the unchanged E3 test package")
    if record["test_sentinel"] != "E3_SENTINEL_PASS":
        raise ValueError("test_sentinel must bind the semantic validator decision")
    if record["decision_bearing_validation"] != "PASS":
        raise ValueError("decision_bearing_validation must be PASS")
    if record["deviations"] != []:
        raise ValueError("deviations must be an empty list")
    return "READY_FOR_CONTROLLER_ADMISSION_REVIEW"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("mapping_record", "contract", "config", "run_status", "artifact_manifest"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    record = json.loads(args.mapping_record.read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise ValueError("mapping record must be a JSON object")
    decision = validate_mapping_record(record, args.contract.read_bytes(), args.config.read_bytes(),
                                       args.run_status.read_bytes(), args.artifact_manifest.read_bytes())
    print(json.dumps({"decision": decision, "launch_authorized": False}, sort_keys=True))


if __name__ == "__main__":
    main()
