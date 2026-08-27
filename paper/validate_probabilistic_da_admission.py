#!/usr/bin/env python3
"""Fail-closed combined admission for one probabilistic-DA compact bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    from .probabilistic_da_adapter_parity import validate_result_directory
    from .probabilistic_da_contract_oracle import EXPECTED_ARTIFACTS
except ImportError:  # Keep direct-script use at the reviewed boundary.
    from probabilistic_da_adapter_parity import validate_result_directory
    from probabilistic_da_contract_oracle import EXPECTED_ARTIFACTS


@dataclass(frozen=True)
class AdmissionResult:
    """Decision and exact identity admitted by one combined operation."""

    outcome: str
    compact_directory_sha256: str


def directory_sha256(path: Path) -> str:
    """Hash the exact allowed filenames and bytes with unambiguous framing."""
    expected = set(EXPECTED_ARTIFACTS)
    if not path.is_dir() or {candidate.name for candidate in path.iterdir()} != expected:
        raise ValueError("result directory must contain exactly three compact artifacts")
    digest = hashlib.sha256()
    for name in sorted(expected):
        encoded_name = name.encode("utf-8")
        payload = (path / name).read_bytes()
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def admit_result_directory(path: Path) -> AdmissionResult:
    """Validate semantics and reject any bundle substitution during admission."""
    admitted_digest = directory_sha256(path)
    outcome = validate_result_directory(path)
    if directory_sha256(path) != admitted_digest:
        raise ValueError("compact directory changed during combined admission")
    return AdmissionResult(
        outcome=outcome,
        compact_directory_sha256=admitted_digest,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail-closed combined admission of a probabilistic-DA bundle."
    )
    parser.add_argument("compact_directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(asdict(admit_result_directory(args.compact_directory)), sort_keys=True))


if __name__ == "__main__":
    main()
