#!/usr/bin/env python3
"""Fail-closed atomic admission for a reviewed casewise-selector runner."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import numbers
from dataclasses import dataclass
from pathlib import Path

try:
    from . import casewise_safety_selector_reference as reference
    from .validate_casewise_safety_selector_compact_outputs import validate_directory
except ImportError:
    import casewise_safety_selector_reference as reference
    from validate_casewise_safety_selector_compact_outputs import validate_directory


PAPER_DIR = Path(__file__).resolve().parent
FROZEN_ARTIFACTS = (
    "NEXT_CASEWISE_SAFETY_SELECTOR_CONTRACT.md",
    "casewise_safety_selector_reference.py",
    "casewise_safety_selector_runner_prototype.py",
    "validate_casewise_safety_selector_compact_outputs.py",
    "validate_casewise_safety_selector_semantics.py",
    "CASEWISE_SAFETY_SELECTOR_RUNNER_REVIEW_CHECKLIST.md",
)
REQUIRED_KEYS = {
    "schema_version", "reviewed_mode", "publication_commit", "runner_sha256",
    "frozen_artifact_sha256", "synthetic_result_sha256",
    "decision_bearing_validation", "deviations",
}


@dataclass(frozen=True)
class AdmissionResult:
    reviewed_mode: str
    admission_record_sha256: str
    synthetic_result_sha256: str


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def directory_sha256(directory: Path) -> str:
    names = sorted(path.name for path in directory.iterdir() if path.is_file())
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode("utf-8")); digest.update(b"\0")
        digest.update((directory / name).read_bytes()); digest.update(b"\0")
    return digest.hexdigest()


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{label} must be one lowercase SHA-256 digest")
    return value


def _assert_close(actual, expected, label: str) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"{label} schema diverges from reference")
        for key in expected:
            _assert_close(actual[key], expected[key], f"{label}.{key}")
    elif isinstance(expected, numbers.Real):
        if not isinstance(actual, numbers.Real) or not math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"{label} diverges from reference")
    elif isinstance(expected, (list, tuple)) or hasattr(expected, "shape"):
        left, right = list(actual), list(expected)
        if len(left) != len(right):
            raise ValueError(f"{label} length diverges from reference")
        for index, (a, e) in enumerate(zip(left, right)):
            _assert_close(a, e, f"{label}[{index}]")
    elif actual != expected:
        raise ValueError(f"{label} diverges from reference")


def validate_semantic_parity(runner: Path) -> None:
    spec = importlib.util.spec_from_file_location("casewise_reviewed_runner", runner)
    if spec is None or spec.loader is None:
        raise ValueError("runner is not an importable Python module")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    for name in ("CASES", "HOLDOUT_SIZE", "PURGE", "PREDICTORS", "RIDGE", "SVD_RELATIVE_CUTOFF", "ACTION_MARGIN", "ACTIONS"):
        if getattr(module, name, None) != getattr(reference, name):
            raise ValueError(f"runner constant {name} diverges from reference")
    case_ids = tuple(f"case-{index:02d}" for index in range(40))
    _assert_close(module.build_purged_folds(case_ids), reference.build_purged_folds(case_ids), "folds")
    diagnostics = {"fair_crps": .05, "rank_abs_error": .2, "inner_coverage_error": .1, "boundary_error": .1, "iiee_ratio": 1.01, "edge_ratio": 1.03, "variogram_ratio": 1.0}
    limits = {"rank_limit": .1, "inner_limit": .1, "boundary_limit": .1}
    _assert_close(module.scalar_loss(diagnostics, limits), reference.scalar_loss(diagnostics, limits), "loss")
    for values in ((.1, .098), (.1, .0980000001), (.1, .09)):
        _assert_close(module.select_action(*values), reference.select_action(*values), "selection")
    if reference.np is None:
        raise RuntimeError("numpy is required for decision-bearing ridge parity")
    np = reference.np
    design = np.asarray([[((i + 2) * (j + 3) % 29) / 7.0 + i * .01 for j in range(6)] for i in range(24)])
    targets = np.asarray([.2 + .03 * i + (i % 4) * .007 for i in range(24)])
    expected = reference.fit_ridge(design, targets)
    actual = module.fit_ridge(design, targets)
    _assert_close(actual, expected, "ridge")
    descriptor = np.asarray([.2, .4, .6, .8, 1., 1.2])
    _assert_close(module.predict_loss(actual, descriptor), reference.predict_loss(expected, descriptor), "prediction")


def validate_record(record: object, runner: Path, synthetic_directory: Path) -> str:
    if not isinstance(record, dict) or set(record) != REQUIRED_KEYS:
        raise ValueError("admission record must be one exact-schema JSON object")
    if record["schema_version"] != "casewise-safety-selector-admission-v1":
        raise ValueError("unsupported admission schema_version")
    mode = record["reviewed_mode"]
    if not isinstance(mode, str) or not mode.startswith("validation_") or not mode.strip():
        raise ValueError("reviewed_mode must be one literal validation mode")
    commit = record["publication_commit"]
    if not isinstance(commit, str) or len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError("publication_commit must be one lowercase 40-hex identity")
    if record["decision_bearing_validation"] != "PASS" or record["deviations"] != []:
        raise ValueError("decision-bearing validation must PASS with deviations=[]")
    if _digest(record["runner_sha256"], "runner_sha256") != sha256_file(runner):
        raise ValueError("runner_sha256 does not bind the reviewed runner")
    identities = record["frozen_artifact_sha256"]
    if not isinstance(identities, dict) or tuple(identities) != FROZEN_ARTIFACTS:
        raise ValueError("frozen artifact inventory drift")
    for name in FROZEN_ARTIFACTS:
        if _digest(identities[name], name) != sha256_file(PAPER_DIR / name):
            raise ValueError(f"{name} digest mismatch")
    if _digest(record["synthetic_result_sha256"], "synthetic_result_sha256") != directory_sha256(synthetic_directory):
        raise ValueError("synthetic_result_sha256 mismatch")
    return mode


def load_and_validate(record_path: Path, runner: Path, synthetic_directory: Path) -> AdmissionResult:
    record_bytes = record_path.read_bytes()
    synthetic_digest = directory_sha256(synthetic_directory)
    mode = validate_record(json.loads(record_bytes), runner, synthetic_directory)
    validate_semantic_parity(runner)
    validate_directory(synthetic_directory)
    if record_path.read_bytes() != record_bytes or directory_sha256(synthetic_directory) != synthetic_digest:
        raise ValueError("admission inputs changed during atomic validation")
    return AdmissionResult(mode, hashlib.sha256(record_bytes).hexdigest(), synthetic_digest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("record", type=Path); parser.add_argument("runner", type=Path)
    parser.add_argument("synthetic_directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(load_and_validate(args.record, args.runner, args.synthetic_directory).__dict__, sort_keys=True))


if __name__ == "__main__":
    main()
