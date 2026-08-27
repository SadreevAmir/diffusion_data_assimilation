#!/usr/bin/env python3
"""Validate one explicit controller-visible rank-coherent admission record."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

try:
    from .rank_coherent_adapter_parity import OUTPUTS, validate_result_directory
    from .rank_coherent_reference import validate_admission_record
except ImportError:  # Direct script execution keeps the documented CLI stable.
    from rank_coherent_adapter_parity import OUTPUTS, validate_result_directory
    from rank_coherent_reference import validate_admission_record


@dataclass(frozen=True)
class AdmissionResult:
    """Identities returned by the operation that admits both exact inputs."""

    reviewed_mode: str
    admission_record_sha256: str
    compact_directory_sha256: str


def directory_sha256(path: Path) -> str:
    """Hash the exact compact filenames and bytes with unambiguous framing."""
    if not path.is_dir() or {candidate.name for candidate in path.iterdir()} != OUTPUTS:
        raise ValueError("result directory must contain exactly four compact outputs")
    digest = hashlib.sha256()
    for name in sorted(OUTPUTS):
        encoded_name = name.encode("utf-8")
        payload = (path / name).read_bytes()
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def load_and_validate(path: Path) -> str:
    """Return only the literally reviewed mode from a valid JSON object."""
    with path.open("r", encoding="utf-8") as handle:
        record = json.load(handle)
    if not isinstance(record, dict):
        raise ValueError("admission record must be a JSON object")
    return validate_admission_record(record)


def load_and_validate_combined(record_path: Path, compact_directory: Path) -> AdmissionResult:
    """Atomically admit one record and its decision-bearing compact directory."""
    admitted_record_bytes = record_path.read_bytes()
    admitted_record_digest = hashlib.sha256(admitted_record_bytes).hexdigest()
    record = json.loads(admitted_record_bytes.decode("utf-8"))
    if not isinstance(record, dict):
        raise ValueError("admission record must be a JSON object")
    admitted_directory_digest = directory_sha256(compact_directory)
    reviewed_mode = validate_admission_record(record)
    validate_result_directory(compact_directory, decision_bearing=True)
    if directory_sha256(compact_directory) != admitted_directory_digest:
        raise ValueError("compact directory changed during combined admission")
    if record_path.read_bytes() != admitted_record_bytes:
        raise ValueError("admission record changed during combined admission")
    return AdmissionResult(
        reviewed_mode=reviewed_mode,
        admission_record_sha256=admitted_record_digest,
        compact_directory_sha256=admitted_directory_digest,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail-closed validation of a rank-coherent admission record."
    )
    parser.add_argument("record", type=Path, help="explicit controller-visible JSON file")
    parser.add_argument("compact_directory", type=Path, help="decision-bearing compact result")
    args = parser.parse_args()
    result = load_and_validate_combined(args.record, args.compact_directory)
    print(json.dumps(result.__dict__, sort_keys=True))


if __name__ == "__main__":
    main()
