#!/usr/bin/env python3
"""Fail-closed validator for conformal admission and compact evidence."""
from __future__ import annotations
import argparse,csv,hashlib,json,math
from dataclasses import dataclass
from pathlib import Path
try:
    from .conformal_area_reference import (
        EXPECTED_CASES,
        MAX_WIDTH_RATIO,
        MIN_USEFUL_COVERAGE,
        validate_admission_record,
    )
except ImportError:
    from conformal_area_reference import (
        EXPECTED_CASES,
        MAX_WIDTH_RATIO,
        MIN_USEFUL_COVERAGE,
        validate_admission_record,
    )

OUTPUTS = {"conformal_summary.json", "conformal_per_case.csv"}


@dataclass(frozen=True)
class AdmissionResult:
    reviewed_mode: str
    admission_record_sha256: str
    compact_directory_sha256: str


def directory_sha256(path: Path) -> str:
    if not path.is_dir() or {candidate.name for candidate in path.iterdir()} != OUTPUTS:
        raise ValueError("result directory must contain exactly two compact outputs")
    digest = hashlib.sha256()
    for name in sorted(OUTPUTS):
        encoded_name = name.encode("utf-8")
        payload = (path / name).read_bytes()
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def validate_result_directory(path: Path) -> str:
    if {candidate.name for candidate in path.iterdir()} != OUTPUTS:
        raise ValueError("result directory must contain exactly two compact outputs")
    summary = json.loads((path / "conformal_summary.json").read_text(encoding="utf-8"))
    required = {
        "decision", "coverage_count", "coverage", "raw_coverage_count",
        "raw_mean_width", "raw_median_width", "conformal_mean_width",
        "conformal_median_width", "width_ratio", "folds",
    }
    if not isinstance(summary, dict) or set(summary) != required:
        raise ValueError("conformal summary keys must be exact")
    decision = summary["decision"]
    if decision not in {"CONFORMAL_USEFUL", "CONFORMAL_NEGATIVE"}:
        raise ValueError("conformal decision is invalid")
    count = summary["coverage_count"]
    coverage = summary["coverage"]
    ratio = summary["width_ratio"]
    if not isinstance(count, int) or not 0 <= count <= EXPECTED_CASES:
        raise ValueError("coverage_count is invalid")
    if not math.isclose(float(coverage), count / EXPECTED_CASES, abs_tol=1e-12):
        raise ValueError("coverage is inconsistent with coverage_count")
    expected = "CONFORMAL_USEFUL" if coverage >= MIN_USEFUL_COVERAGE and ratio <= MAX_WIDTH_RATIO else "CONFORMAL_NEGATIVE"
    if decision != expected:
        raise ValueError("decision is inconsistent with frozen thresholds")
    with (path / "conformal_per_case.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != EXPECTED_CASES or [int(row["case_index"]) for row in rows] != list(range(EXPECTED_CASES)):
        raise ValueError("per-case CSV must contain the ordered forty-case envelope")
    csv_count = sum(row["conformal_covered"] == "True" for row in rows)
    if csv_count != count:
        raise ValueError("summary coverage_count disagrees with per-case CSV")
    return decision

def load_and_validate(path: Path) -> str:
    record=json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record,dict): raise ValueError("admission record must be a JSON object")
    return validate_admission_record(record)


def load_and_validate_combined(record_path: Path, compact_directory: Path) -> AdmissionResult:
    admitted_record_bytes = record_path.read_bytes()
    record = json.loads(admitted_record_bytes.decode("utf-8"))
    if not isinstance(record,dict): raise ValueError("admission record must be a JSON object")
    admitted_directory_digest = directory_sha256(compact_directory)
    reviewed_mode = validate_admission_record(record)
    validate_result_directory(compact_directory)
    if directory_sha256(compact_directory) != admitted_directory_digest:
        raise ValueError("compact directory changed during combined admission")
    if record_path.read_bytes() != admitted_record_bytes:
        raise ValueError("admission record changed during combined admission")
    return AdmissionResult(
        reviewed_mode=reviewed_mode,
        admission_record_sha256=hashlib.sha256(admitted_record_bytes).hexdigest(),
        compact_directory_sha256=admitted_directory_digest,
    )

def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("record",type=Path); parser.add_argument("compact_directory",type=Path); args=parser.parse_args()
    print(json.dumps(load_and_validate_combined(args.record,args.compact_directory).__dict__,sort_keys=True))
if __name__=="__main__": main()
