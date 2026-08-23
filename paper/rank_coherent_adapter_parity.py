#!/usr/bin/env python3
"""Fail-closed parity oracle for a rank-coherent compact result directory."""
from __future__ import annotations
import csv, json, math
from pathlib import Path
from rank_coherent_reference import (ARTIFACT_POLICY, CASE_STRIDE, DATASET_SPLIT,
    END_DATE, EXPECTED_CASES, EXPECTED_MEMBERS, SOURCE_EXPERIMENT, START_DATE,
    validate_compact_handoff)

OUTPUTS = {"run_status.json", "aggregate_case_mean_metrics.json", "per_case_metrics.csv", "metadata.json"}
METRICS = ("analysis_fair_crps", "analysis_crps", "analysis_mean_rmse")

def _load_json(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict): raise ValueError(f"{path.name} must contain one JSON object")
    return value

def validate_result_directory(root: Path, *, decision_bearing: bool) -> None:
    if not root.is_dir() or {p.name for p in root.iterdir()} != OUTPUTS:
        raise ValueError("result directory must contain exactly four compact outputs")
    aggregate = _load_json(root / "aggregate_case_mean_metrics.json")
    schema = aggregate.pop("schema_version", None)
    if not isinstance(schema, str) or not schema: raise ValueError("aggregate schema_version must be non-empty")
    validate_compact_handoff(aggregate)
    metadata = _load_json(root / "metadata.json")
    expected = {"source_experiment": SOURCE_EXPERIMENT, "artifact_policy": ARTIFACT_POLICY,
        "dataset_split": DATASET_SPLIT, "start_date": START_DATE, "end_date": END_DATE,
        "cases": EXPECTED_CASES, "ensemble_size": EXPECTED_MEMBERS, "case_stride": CASE_STRIDE}
    if any(metadata.get(k) != v for k, v in expected.items()): raise ValueError("metadata drifted from frozen envelope")
    if metadata.get("schema_version") != schema: raise ValueError("metadata and aggregate schema versions disagree")
    if metadata.get("decision_bearing") is not decision_bearing: raise ValueError("metadata decision_bearing disagrees with adapter role")
    status = _load_json(root / "run_status.json")
    if status.get("schema_version") != schema or status.get("status") != "complete": raise ValueError("run status is incomplete or has schema drift")
    if (status.get("completed_cases"), status.get("total_cases")) != (EXPECTED_CASES, EXPECTED_CASES): raise ValueError("run status has incomplete envelope")
    if status.get("overall_eligible") is not aggregate["gate"]["overall_eligible"]: raise ValueError("status and compact gate disagree")
    if set(status.get("compact_outputs", ())) != OUTPUTS: raise ValueError("status does not enumerate exact outputs")
    with (root / "per_case_metrics.csv").open(newline="", encoding="utf-8") as stream: rows = list(csv.DictReader(stream))
    if len(rows) != 2 * EXPECTED_CASES: raise ValueError("per-case CSV row count is incomplete")
    grouped = {(int(r["case_index"]), r["variant"]): r for r in rows}
    keys = {(i, v) for i in range(EXPECTED_CASES) for v in ("raw", "candidate")}
    if set(grouped) != keys: raise ValueError("per-case CSV keys are incomplete or duplicated")
    for metric in METRICS:
        for variant in ("raw", "candidate"):
            values = [float(grouped[(i, variant)][metric]) for i in range(EXPECTED_CASES)]
            if not all(math.isfinite(v) for v in values): raise ValueError("per-case CSV contains non-finite metric")
            if not math.isclose(sum(values) / EXPECTED_CASES, aggregate["aggregate_metrics"][metric][variant], rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError("per-case and aggregate metrics disagree")
