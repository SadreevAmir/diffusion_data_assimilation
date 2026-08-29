#!/usr/bin/env python3
"""Fail-closed admission for the frozen raw-member reweighting runner review."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

from . import raw_member_reweighting_reference as reference


PAPER_DIR = Path(__file__).resolve().parent
CONTRACT = PAPER_DIR / "NEXT_RAW_MEMBER_REWEIGHTING_CONTRACT.md"
REQUIRED_KEYS = {
    "reviewed_mode",
    "publication_commit",
    "runner_sha256",
    "contract_sha256",
    "synthetic_result_sha256",
    "test_command",
    "test_sentinel",
    "decision_bearing_validation",
    "deviations",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_hex(value: object, length: int, name: str) -> str:
    if not isinstance(value, str) or len(value) != length or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be one lowercase {length}-hex identity")
    return value


def _load_runner(path: Path):
    spec = importlib.util.spec_from_file_location("raw_member_reweighting_runner", path)
    if spec is None or spec.loader is None:
        raise ValueError("runner is not an importable Python module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_semantic_parity(runner: Path) -> None:
    """Require the reviewed runner's pure construction surface to match the oracle."""
    module = _load_runner(runner)
    for name in (
        "analog_rank_probabilities",
        "systematic_rank_positions",
        "selection_diagnostics",
        "rank_positions_to_member_indices",
        "construct_selection",
    ):
        if not callable(getattr(module, name, None)):
            raise ValueError(f"runner lacks callable {name}")

    rank_vectors = (
        [(index + 0.5) / 10 for index in range(10)],
        [0.0] * 10,
        [1.0] * 10,
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0],
    )
    mean_vectors = (
        list(range(10)),
        list(reversed(range(10))),
        [0.5] * 10,
        [0.2, 0.1, 0.1, 0.8, 0.7, 0.7, 0.4, 0.3, 0.9, 0.0],
    )
    for ranks, means in zip(rank_vectors, mean_vectors):
        expected = reference.construct_selection(ranks, means)
        actual = module.construct_selection(ranks, means)
        if actual != expected:
            raise ValueError("runner construction diverges from frozen oracle")


def validate_record(
    record: object, runner: Path, synthetic_result: Path
) -> str:
    if not isinstance(record, dict) or set(record) != REQUIRED_KEYS:
        raise ValueError("review record must be one exact-schema JSON object")
    mode = record["reviewed_mode"]
    if (
        not isinstance(mode, str)
        or not mode.startswith("validation_")
        or mode.strip() != mode
        or "<" in mode
        or ">" in mode
    ):
        raise ValueError("reviewed_mode must be one literal validation mode")
    _require_hex(record["publication_commit"], 40, "publication_commit")
    for key, expected in (
        ("runner_sha256", _sha256(runner)),
        ("contract_sha256", _sha256(CONTRACT)),
        ("synthetic_result_sha256", _sha256(synthetic_result)),
    ):
        if _require_hex(record[key], 64, key) != expected:
            raise ValueError(f"{key} does not bind the reviewed local artifact")
    if record["decision_bearing_validation"] != "PASS":
        raise ValueError("decision_bearing_validation must be PASS")
    if record["deviations"] != []:
        raise ValueError("deviations must be an empty list")
    for key in ("test_command", "test_sentinel"):
        value = record[key]
        if not isinstance(value, str) or not value.strip() or "<" in value or ">" in value:
            raise ValueError(f"{key} must be one exact non-placeholder string")
    validate_semantic_parity(runner)
    return mode


def load_and_validate(record_path: Path, runner: Path, synthetic_result: Path) -> str:
    record_bytes = record_path.read_bytes()
    runner_digest = _sha256(runner)
    synthetic_digest = _sha256(synthetic_result)
    mode = validate_record(json.loads(record_bytes.decode("utf-8")), runner, synthetic_result)
    if record_path.read_bytes() != record_bytes:
        raise ValueError("review record changed during admission")
    if _sha256(runner) != runner_digest or _sha256(synthetic_result) != synthetic_digest:
        raise ValueError("reviewed artifact changed during admission")
    return mode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path)
    parser.add_argument("runner", type=Path)
    parser.add_argument("synthetic_result", type=Path)
    args = parser.parse_args()
    mode = load_and_validate(args.record, args.runner, args.synthetic_result)
    print(json.dumps({"reviewed_mode": mode, "admission": "GO"}, sort_keys=True))


if __name__ == "__main__":
    main()
