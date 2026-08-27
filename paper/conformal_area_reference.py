#!/usr/bin/env python3
"""Dependency-free oracle for the frozen block-conformal area baseline."""
from __future__ import annotations
import math
import hashlib
import re
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path

EXPECTED_CASES, EXPECTED_MEMBERS, HOLDOUT_SIZE, PURGE = 40, 10, 8, 3
NOMINAL_COVERAGE, MIN_USEFUL_COVERAGE, MAX_WIDTH_RATIO = 0.90, 0.85, 1.50
SOURCE_EXPERIMENT = "joint_full_condition_validation_2022"
ARTIFACT_POLICY = "summary_only"
ADMISSION_RECORD_KEYS = (
    "reviewed_mode", "publication_commit", "runner_sha256", "contract_sha256",
    "synthetic_result_sha256", "test_command", "test_sentinel",
    "decision_bearing_validation", "deviations",
)
FROZEN_CONTRACT_PATH = Path(__file__).with_name("NEXT_CONFORMAL_BASELINE_CONTRACT.md")

def frozen_contract_sha256() -> str:
    return hashlib.sha256(FROZEN_CONTRACT_PATH.read_bytes()).hexdigest()

def validate_admission_record(record: Mapping[str, object]) -> str:
    """Admit only an exact independently reviewed literal runner contract."""
    if set(record) != set(ADMISSION_RECORD_KEYS):
        raise ValueError("admission record keys must be exact")
    for key in ("reviewed_mode", "test_command", "test_sentinel"):
        value = record[key]
        if not isinstance(value, str) or not value.strip() or "<" in value or ">" in value:
            raise ValueError(f"{key} must be a non-placeholder string")
    if not re.fullmatch(r"[0-9a-f]{40}", str(record["publication_commit"])):
        raise ValueError("publication_commit must be lowercase 40-hex")
    for key in ("runner_sha256", "contract_sha256", "synthetic_result_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(record[key])):
            raise ValueError(f"{key} must be lowercase SHA-256")
    if record["contract_sha256"] != frozen_contract_sha256():
        raise ValueError("contract_sha256 must match the frozen local contract")
    if record["decision_bearing_validation"] != "PASS":
        raise ValueError("decision_bearing_validation must be PASS")
    if record["deviations"] != []:
        raise ValueError("deviations must be an empty list")
    return str(record["reviewed_mode"])

def purged_folds():
    folds = []
    for start in range(0, EXPECTED_CASES, HOLDOUT_SIZE):
        stop = start + HOLDOUT_SIZE
        holdout = tuple(range(start, stop))
        training = tuple(i for i in range(EXPECTED_CASES) if not max(0, start-PURGE) <= i < min(EXPECTED_CASES, stop+PURGE))
        folds.append((holdout, training))
    return tuple(folds)

def validate_interface(parameters: Mapping[str, object], artifact_policy: str) -> None:
    if dict(parameters) != {"source_experiment": SOURCE_EXPERIMENT}: raise ValueError("only the frozen source_experiment is permitted")
    if artifact_policy != ARTIFACT_POLICY: raise ValueError("retrieval must remain summary_only")

def _finite(values: Sequence[float], name: str):
    result = tuple(float(value) for value in values)
    if len(result) != EXPECTED_CASES or not all(math.isfinite(value) for value in result): raise ValueError(f"{name} must contain forty finite values")
    return result

def nonconformity(lower: float, upper: float, truth: float) -> float:
    if not lower <= upper: raise ValueError("raw interval lower endpoint exceeds upper endpoint")
    return max(lower-truth, truth-upper, 0.0)

def conformal_quantile(scores: Sequence[float]):
    if not scores or not all(math.isfinite(score) and score >= 0 for score in scores): raise ValueError("calibration scores must be finite and non-negative")
    rank = min(len(scores), math.ceil((len(scores)+1)*NOMINAL_COVERAGE))
    return sorted(scores)[rank-1], rank

def evaluate(lower: Sequence[float], upper: Sequence[float], truth: Sequence[float]):
    lower, upper, truth = _finite(lower,"lower"), _finite(upper,"upper"), _finite(truth,"truth")
    if any(left > right for left,right in zip(lower,upper)): raise ValueError("raw interval lower endpoint exceeds upper endpoint")
    rows, fold_records = [], []
    for fold_index,(holdout,training) in enumerate(purged_folds()):
        q,rank = conformal_quantile([nonconformity(lower[i],upper[i],truth[i]) for i in training])
        fold_records.append({"fold_index":fold_index,"training_size":len(training),"quantile_rank":rank,"quantile":q})
        for i in holdout:
            cl,cu = lower[i]-q,upper[i]+q
            rows.append({"case_index":i,"fold_index":fold_index,"raw_lower":lower[i],"raw_upper":upper[i],"conformal_lower":cl,"conformal_upper":cu,"truth_area":truth[i],"raw_width":upper[i]-lower[i],"conformal_width":cu-cl,"paired_width_change":2*q,"raw_covered":lower[i] <= truth[i] <= upper[i],"conformal_covered":cl <= truth[i] <= cu})
    rows.sort(key=lambda row: row["case_index"])
    raw_mean = sum(row["raw_upper"]-row["raw_lower"] for row in rows)/EXPECTED_CASES
    conformal_mean = sum(row["conformal_upper"]-row["conformal_lower"] for row in rows)/EXPECTED_CASES
    ratio = conformal_mean/raw_mean if raw_mean > 0 else math.inf
    count = sum(row["conformal_covered"] for row in rows); coverage = count/EXPECTED_CASES
    decision = "CONFORMAL_USEFUL" if coverage >= MIN_USEFUL_COVERAGE and ratio <= MAX_WIDTH_RATIO else "CONFORMAL_NEGATIVE"
    return {"decision":decision,"coverage_count":count,"coverage":coverage,"raw_coverage_count":sum(row["raw_covered"] for row in rows),"raw_mean_width":raw_mean,"raw_median_width":statistics.median(row["raw_width"] for row in rows),"conformal_mean_width":conformal_mean,"conformal_median_width":statistics.median(row["conformal_width"] for row in rows),"width_ratio":ratio,"folds":fold_records,"rows":rows}
