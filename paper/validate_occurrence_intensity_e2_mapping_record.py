#!/usr/bin/env python3
"""Fail-closed validation of a controller-visible E2 mapping review record."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

from paper.validate_occurrence_intensity_e1_admission import validate_response
from paper.validate_occurrence_intensity_e2_compact_result import validate_result

RECORD_KEYS = {"e1_admission_decision", "reviewed_mode", "publication_commit", "runner_sha256", "contract_sha256", "synthetic_result_sha256", "test_command", "test_sentinel", "decision_bearing_validation", "deviations"}
SAFE_MODE = re.compile(r"^[a-z][a-z0-9_]{0,127}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _object(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: top-level JSON must be an object")
    return value


def _nonempty(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def validate_mapping_record(record: Mapping[str, Any], admission_request: Mapping[str, Any], admission_response: Mapping[str, Any], contract_bytes: bytes, synthetic_result_bytes: bytes) -> str:
    if set(record) != RECORD_KEYS:
        raise ValueError("mapping record keys must be exact")
    if validate_response(admission_request, admission_response) != "PASS":
        raise ValueError("E1 admission must be a literal PASS")
    if record["e1_admission_decision"] != "PASS":
        raise ValueError("mapping record must bind the literal E1 PASS")
    reviewed_mode = _nonempty(record["reviewed_mode"], "reviewed_mode")
    if not SAFE_MODE.fullmatch(reviewed_mode):
        raise ValueError("reviewed_mode must be one literal safe mode identifier")
    if not isinstance(record["publication_commit"], str) or not HEX40.fullmatch(record["publication_commit"]):
        raise ValueError("publication_commit must be one lowercase 40-hex commit")
    for label in ("runner_sha256", "contract_sha256", "synthetic_result_sha256"):
        value = record[label]
        if not isinstance(value, str) or not HEX64.fullmatch(value):
            raise ValueError(f"{label} must be one lowercase SHA-256 digest")
    if record["contract_sha256"] != hashlib.sha256(contract_bytes).hexdigest():
        raise ValueError("contract_sha256 does not bind the supplied frozen contract")
    if record["synthetic_result_sha256"] != hashlib.sha256(synthetic_result_bytes).hexdigest():
        raise ValueError("synthetic_result_sha256 does not bind the supplied result")
    synthetic_result = json.loads(synthetic_result_bytes)
    if not isinstance(synthetic_result, dict):
        raise ValueError("synthetic result must be a JSON object")
    validate_result(synthetic_result)
    _nonempty(record["test_command"], "test_command")
    _nonempty(record["test_sentinel"], "test_sentinel")
    if record["decision_bearing_validation"] != "PASS":
        raise ValueError("decision_bearing_validation must be PASS")
    if record["deviations"] != []:
        raise ValueError("deviations must be an empty list")
    return "GO"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("admission_request", type=Path)
    parser.add_argument("admission_response", type=Path)
    parser.add_argument("mapping_record", type=Path)
    parser.add_argument("contract", type=Path)
    parser.add_argument("synthetic_result", type=Path)
    args = parser.parse_args()
    decision = validate_mapping_record(_object(args.mapping_record), _object(args.admission_request), _object(args.admission_response), args.contract.read_bytes(), args.synthetic_result.read_bytes())
    print(json.dumps({"decision": decision, "launch_authorized": False}, sort_keys=True))


if __name__ == "__main__":
    main()
