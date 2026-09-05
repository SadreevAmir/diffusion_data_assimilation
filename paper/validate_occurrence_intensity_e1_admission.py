#!/usr/bin/env python3
"""Fail-closed validation of the independent E1 admission response."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


REQUEST_KEYS = {
    "decision", "reviewed_mode", "config", "runner", "implementation", "tests",
    "required_checks", "required_test_command", "accepted_response",
    "rejection_response", "pass_scope", "launch_authorized",
}
RESPONSE_KEYS = {"decision", "failed_checks", "failed_tests"}


def _object(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: top-level JSON must be an object")
    return value


def validate_request(request: Mapping[str, Any]) -> None:
    if set(request) != REQUEST_KEYS:
        raise ValueError("admission request keys must be exact")
    if request["decision"] != "PENDING_INDEPENDENT_REVIEW":
        raise ValueError("request is not pending independent review")
    if request["reviewed_mode"] != "occurrence_intensity_e1_engineering_sentinel":
        raise ValueError("unexpected reviewed_mode")
    if request["launch_authorized"] is not False:
        raise ValueError("admission request must not authorize launch")
    if request["pass_scope"] != "independent_admission_only_not_launch_authorization":
        raise ValueError("unexpected pass_scope")
    checks = request["required_checks"]
    if not isinstance(checks, list) or not checks or not all(isinstance(x, str) and x for x in checks):
        raise ValueError("required_checks must be a non-empty string list")
    if request["accepted_response"] != {"decision": "PASS", "failed_checks": [], "failed_tests": []}:
        raise ValueError("accepted_response contract drift")
    rejection = request["rejection_response"]
    if not isinstance(rejection, dict) or set(rejection) != RESPONSE_KEYS or rejection.get("decision") != "REJECT":
        raise ValueError("rejection_response contract drift")


def validate_response(request: Mapping[str, Any], response: Mapping[str, Any]) -> str:
    validate_request(request)
    if set(response) != RESPONSE_KEYS:
        raise ValueError("admission response keys must be exact")
    decision = response["decision"]
    failed_checks = response["failed_checks"]
    failed_tests = response["failed_tests"]
    if not isinstance(failed_checks, list) or not all(isinstance(x, str) and x for x in failed_checks):
        raise ValueError("failed_checks must be a string list")
    if not isinstance(failed_tests, list) or not all(isinstance(x, str) and x for x in failed_tests):
        raise ValueError("failed_tests must be a string list")
    if decision == "PASS":
        if failed_checks or failed_tests:
            raise ValueError("PASS cannot contain failures")
    elif decision == "REJECT":
        if not failed_checks and not failed_tests:
            raise ValueError("REJECT must identify at least one failure")
        unknown = set(failed_checks) - set(request["required_checks"])
        if unknown:
            raise ValueError(f"unknown failed_checks: {sorted(unknown)}")
        if any(":" not in item or not item.split(":", 1)[0].strip() or not item.split(":", 1)[1].strip() for item in failed_tests):
            raise ValueError("each failed_tests entry must contain an identifier and message separated by ':'")
    else:
        raise ValueError("decision must be PASS or REJECT")
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("response", type=Path)
    args = parser.parse_args()
    request_bytes = args.request.read_bytes()
    request = _object(args.request)
    response = _object(args.response)
    decision = validate_response(request, response)
    print(json.dumps({
        "decision": decision,
        "launch_authorized": False,
        "request_sha256": hashlib.sha256(request_bytes).hexdigest(),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
