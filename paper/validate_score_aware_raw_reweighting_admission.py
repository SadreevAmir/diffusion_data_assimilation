#!/usr/bin/env python3
"""Fail-closed semantic admission for a future score-aware trusted runner."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

try:
    from . import score_aware_raw_reweighting_reference as reference
    from .validate_score_aware_compact_outputs import directory_sha256, validate_directory
except ImportError:  # Keep direct script execution stable.
    import score_aware_raw_reweighting_reference as reference
    from validate_score_aware_compact_outputs import directory_sha256, validate_directory


PAPER_DIR = Path(__file__).resolve().parent
CONTRACT = PAPER_DIR / "NEXT_SCORE_AWARE_RAW_REWEIGHTING_CONTRACT.md"
REFERENCE = PAPER_DIR / "score_aware_raw_reweighting_reference.py"
REQUIRED_KEYS = {
    "schema_version",
    "reviewed_mode",
    "publication_commit",
    "runner_sha256",
    "contract_sha256",
    "reference_sha256",
    "compact_directory_sha256",
    "decision_bearing_validation",
    "deviations",
}
SELECTION_VECTORS = (
    [0.0] * 10,
    [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0],
    [1000.0, -1000.0, 50.0, -50.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0],
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_digest(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be one lowercase SHA-256 digest")
    return value


def validate_record(record: object, runner: Path, compact_directory: Path) -> str:
    if not isinstance(record, dict) or set(record) != REQUIRED_KEYS:
        raise ValueError("admission record must be one exact-schema JSON object")
    if record["schema_version"] != "score-aware-raw-reweighting-admission-v2":
        raise ValueError("unsupported admission schema_version")
    mode = record["reviewed_mode"]
    if not isinstance(mode, str) or not mode.startswith("validation_") or not mode.strip():
        raise ValueError("reviewed_mode must be one literal validation mode")
    commit = record["publication_commit"]
    if not isinstance(commit, str) or len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError("publication_commit must be one lowercase 40-hex identity")
    if record["deviations"] != []:
        raise ValueError("deviations must be an empty list")
    if record["decision_bearing_validation"] != "PASS":
        raise ValueError("decision_bearing_validation must be PASS")
    expected = {
        "runner_sha256": _sha256(runner),
        "contract_sha256": _sha256(CONTRACT),
        "reference_sha256": _sha256(REFERENCE),
        "compact_directory_sha256": directory_sha256(compact_directory),
    }
    for key, digest in expected.items():
        if _require_digest(record[key], key) != digest:
            raise ValueError(f"{key} does not bind the reviewed local artifact")
    return mode


def _load_runner(path: Path):
    spec = importlib.util.spec_from_file_location("score_aware_candidate_runner", path)
    if spec is None or spec.loader is None:
        raise ValueError("runner is not an importable Python module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assert_nested_close(actual, expected, label: str) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"{label} schema diverges from reference")
        for key in expected:
            _assert_nested_close(actual[key], expected[key], f"{label}.{key}")
    elif isinstance(expected, (list, tuple)) or hasattr(expected, "shape"):
        left, right = list(actual), list(expected)
        if len(left) != len(right):
            raise ValueError(f"{label} length diverges from reference")
        for index, (a, e) in enumerate(zip(left, right)):
            _assert_nested_close(a, e, f"{label}[{index}]")
    elif isinstance(expected, float):
        if not isinstance(actual, (int, float)) or not math.isclose(float(actual), expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"{label} diverges from reference")
    elif actual != expected:
        raise ValueError(f"{label} diverges from reference")


def validate_semantic_parity(runner: Path) -> None:
    module = _load_runner(runner)
    for name in ("MEMBERS", "PREDICTORS", "RIDGE", "SVD_RELATIVE_CUTOFF"):
        if getattr(module, name, None) != getattr(reference, name):
            raise ValueError(f"runner constant {name} diverges from reference")
    for name in ("fit_fold_ridge", "predict_member_risks", "select_raw_scenarios"):
        if not callable(getattr(module, name, None)):
            raise ValueError(f"runner lacks callable {name}")
    for index, risks in enumerate(SELECTION_VECTORS):
        _assert_nested_close(module.select_raw_scenarios(risks), reference.select_raw_scenarios(risks), f"selection[{index}]")
    if reference.np is None:
        raise RuntimeError("numpy is required for decision-bearing ridge parity")
    np = reference.np
    rows = 24
    design = np.asarray([[((i + 2) * (j + 3) % 29) / 7.0 + i * 0.01 for j in range(13)] for i in range(rows)])
    targets = np.asarray([0.2 + 0.03 * i + (i % 4) * 0.007 for i in range(rows)])
    heldout = np.asarray([[((i + 5) * (j + 1) % 31) / 9.0 for j in range(13)] for i in range(10)])
    expected_model = reference.fit_fold_ridge(design, targets)
    actual_model = module.fit_fold_ridge(design, targets)
    _assert_nested_close(actual_model, expected_model, "ridge")
    _assert_nested_close(module.predict_member_risks(actual_model, heldout), reference.predict_member_risks(expected_model, heldout), "prediction")


def load_and_validate(record_path: Path, runner: Path, compact_directory: Path) -> str:
    record = json.loads(record_path.read_text(encoding="utf-8"))
    admitted_digest = directory_sha256(compact_directory)
    mode = validate_record(record, runner, compact_directory)
    validate_semantic_parity(runner)
    validate_directory(compact_directory)
    if directory_sha256(compact_directory) != admitted_digest:
        raise ValueError("compact directory changed during combined admission")
    return mode


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail-closed score-aware runner admission.")
    parser.add_argument("record", type=Path)
    parser.add_argument("runner", type=Path)
    parser.add_argument("compact_directory", type=Path)
    args = parser.parse_args()
    print(f"reviewed_mode={load_and_validate(args.record, args.runner, args.compact_directory)}")


if __name__ == "__main__":
    main()
