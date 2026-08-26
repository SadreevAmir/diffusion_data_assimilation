#!/usr/bin/env python3
"""Validate one explicit controller-visible rank-coherent admission record."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .rank_coherent_reference import validate_admission_record
except ImportError:  # Direct script execution keeps the documented CLI stable.
    from rank_coherent_reference import validate_admission_record


def load_and_validate(path: Path) -> str:
    """Return only the literally reviewed mode from a valid JSON object."""
    with path.open("r", encoding="utf-8") as handle:
        record = json.load(handle)
    if not isinstance(record, dict):
        raise ValueError("admission record must be a JSON object")
    return validate_admission_record(record)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail-closed validation of a rank-coherent admission record."
    )
    parser.add_argument("record", type=Path, help="explicit controller-visible JSON file")
    args = parser.parse_args()
    reviewed_mode = load_and_validate(args.record)
    print(f"reviewed_mode={reviewed_mode}")


if __name__ == "__main__":
    main()
