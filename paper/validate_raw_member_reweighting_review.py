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
        "purged_folds",
        "standardize_and_select_analogs",
        "analog_rank_probabilities",
        "systematic_rank_positions",
        "selection_diagnostics",
        "rank_positions_to_member_indices",
        "construct_selection",
    ):
        if not callable(getattr(module, name, None)):
            raise ValueError(f"runner lacks callable {name}")

    expected_folds = []
    for fold in range(5):
        start = fold * 8
        holdout = tuple(range(start, start + 8))
        excluded = set(holdout)
        excluded.update(range(max(0, start - 3), start))
        excluded.update(range(start + 8, min(40, start + 11)))
        training = tuple(index for index in range(40) if index not in excluded)
        expected_folds.append((holdout, training))
    if module.purged_folds() != tuple(expected_folds):
        raise ValueError("runner folds or non-circular purge diverge from frozen contract")

    # Each training row varies in every feature, while indices 7 and 11 are an
    # exact distance tie.  Population rather than sample scaling is exercised;
    # deterministic ordering must prefer the smaller case index.
    training_features = {
        index: [
            float(index),
            float(index * index),
            float(index % 3),
            float((index + 1) % 4),
            float(index % 5),
            float((2 * index + 1) % 7),
        ]
        for index in range(12)
    }
    training_features[11] = list(training_features[7])
    target_features = list(training_features[7])
    selected = module.standardize_and_select_analogs(target_features, training_features)
    if len(selected) != 10 or selected[:2] != [7, 11] or len(set(selected)) != 10:
        raise ValueError("runner analog selection, scaling or tie order diverges")

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


def validate_synthetic_result(runner: Path, synthetic_result: Path) -> None:
    """Require the bound dry run to be the runner's exact outcome-agnostic record."""
    module = _load_runner(runner)
    build_synthetic = getattr(module, "synthetic_result", None)
    if not callable(build_synthetic):
        raise ValueError("runner lacks callable synthetic_result")
    try:
        actual = json.loads(synthetic_result.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("synthetic result must be one UTF-8 JSON record") from error
    expected = build_synthetic()
    if actual != expected:
        raise ValueError("synthetic result diverges from reviewed runner dry run")
    if actual.get("decision_bearing") is not False:
        raise ValueError("synthetic result must not be decision-bearing")
    if actual.get("project_data_metrics_emitted") is not False:
        raise ValueError("synthetic result must not emit project-data metrics")


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
    validate_synthetic_result(runner, synthetic_result)
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


def build_admission_payload(
    record_path: Path, runner: Path, synthetic_result: Path
) -> dict[str, str]:
    """Return one self-contained admission object with every verified identity."""
    record_bytes = record_path.read_bytes()
    record = json.loads(record_bytes.decode("utf-8"))
    mode = load_and_validate(record_path, runner, synthetic_result)
    if record_path.read_bytes() != record_bytes:
        raise ValueError("review record changed while building admission payload")
    return {
        "reviewed_mode": mode,
        "admission": "GO",
        "review_record_sha256": hashlib.sha256(record_bytes).hexdigest(),
        "publication_commit": record["publication_commit"],
        "runner_sha256": record["runner_sha256"],
        "contract_sha256": record["contract_sha256"],
        "synthetic_result_sha256": record["synthetic_result_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path)
    parser.add_argument("runner", type=Path)
    parser.add_argument("synthetic_result", type=Path)
    args = parser.parse_args()
    payload = build_admission_payload(args.record, args.runner, args.synthetic_result)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
