#!/usr/bin/env python3
"""Fail-closed validator for a controller-visible conformal admission record."""
from __future__ import annotations
import argparse,json
from pathlib import Path
try:
    from .conformal_area_reference import validate_admission_record
except ImportError:
    from conformal_area_reference import validate_admission_record

def load_and_validate(path: Path) -> str:
    record=json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record,dict): raise ValueError("admission record must be a JSON object")
    return validate_admission_record(record)

def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("record",type=Path); args=parser.parse_args()
    print(json.dumps({"reviewed_mode":load_and_validate(args.record)},sort_keys=True))
if __name__=="__main__": main()
