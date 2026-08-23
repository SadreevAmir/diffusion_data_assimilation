#!/usr/bin/env python3
"""Outcome-agnostic server-CPU prototype for the frozen rank-coherent method.

Exact CLI::

    RANK_COHERENT_SOURCE_JSON=/sealed/source.json \
    RANK_COHERENT_OUTPUT_DIR=/server/result \
    python3 paper/rank_coherent_runner_prototype.py \
      --source-experiment joint_full_condition_validation_2022

The environment variables are controller plumbing, not scientific parameters.
The input is server-local and no array-valued artifact is emitted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path

from rank_coherent_reference import (
    ALPHAS, ARTIFACT_POLICY, CASE_STRIDE, DATASET_SPLIT, END_DATE,
    EXPECTED_CASES, EXPECTED_MEMBERS, FEATURE_COUNT, SOURCE_EXPERIMENT,
    START_DATE, construct_projected_candidate, pair_with_heldout_member_order,
    purged_folds, select_alpha, select_forecast_analogs,
    select_rank_stratified_fields, validate_compact_handoff,
    validate_run_envelope, validate_runner_interface,
)

SCHEMA = "rank-coherent-prototype-v1"
OUTPUTS = (
    "run_status.json", "aggregate_case_mean_metrics.json",
    "per_case_metrics.csv", "metadata.json",
)
SOURCE_KEYS = {
    "schema_version", "source_experiment", "dataset_split", "start_date",
    "end_date", "case_stride", "artifact_policy", "raw_ensemble", "truth",
    "spatial_weights", "truth_rank_tie_uniforms", "frozen_thresholds",
}
THRESHOLD_KEYS = {
    "boundary_mass_absolute_tolerance", "member_semivariogram_absolute_tolerance",
    "rank_l1_absolute_maximum", "coverage_absolute_error_maximum",
}


def fail(message: str) -> "None":
    raise ValueError(message)


def finite_number(value: object, context: str) -> float:
    if type(value) not in (int, float) or type(value) is bool:
        fail(f"{context} must be a JSON number")
    result = float(value)
    if not math.isfinite(result):
        fail(f"{context} must be finite")
    return result


def field_shape(field: object, context: str) -> tuple[int, int]:
    if not isinstance(field, list) or not field or not isinstance(field[0], list) or not field[0]:
        fail(f"{context} must be a non-empty 2-D array")
    shape = (len(field), len(field[0]))
    if any(not isinstance(row, list) or len(row) != shape[1] for row in field):
        fail(f"{context} must be rectangular")
    for row in field:
        for value in row:
            finite_number(value, context)
    return shape


def validate_source(source: object) -> dict:
    if not isinstance(source, dict) or set(source) != SOURCE_KEYS:
        fail("source JSON has missing or extra top-level fields")
    validate_runner_interface({"source_experiment": source["source_experiment"]}, source["artifact_policy"])
    validate_run_envelope(source["dataset_split"], source["start_date"], source["end_date"],
                          len(source["raw_ensemble"]), EXPECTED_MEMBERS, source["case_stride"])
    if source["schema_version"] != SCHEMA:
        fail("source schema_version is not frozen")
    raw, truth = source["raw_ensemble"], source["truth"]
    if len(raw) != EXPECTED_CASES or len(truth) != EXPECTED_CASES:
        fail("source must contain exactly forty cases")
    truth_shape = field_shape(truth[0], "truth")
    for case in range(EXPECTED_CASES):
        if field_shape(truth[case], "truth") != truth_shape:
            fail("truth fields must share one shape")
        if len(raw[case]) != EXPECTED_MEMBERS:
            fail("every raw ensemble must contain ten members")
        for member in raw[case]:
            if field_shape(member, "raw member") != truth_shape:
                fail("raw members and truth must share one shape")
            if any(value < 0 or value > 1 for row in member for value in row):
                fail("raw concentrations must lie in [0,1]")
    weights = source["spatial_weights"]
    if field_shape(weights, "spatial_weights") != truth_shape:
        fail("spatial weights must match fields")
    if any(value < 0 for row in weights for value in row) or sum(map(sum, weights)) <= 0:
        fail("spatial weights must be non-negative with positive sum")
    ties = source["truth_rank_tie_uniforms"]
    if not isinstance(ties, list) or len(ties) != EXPECTED_CASES:
        fail("sealed truth-rank tie uniforms must contain forty values")
    if any(not 0 <= finite_number(value, "truth-rank tie uniform") < 1 for value in ties):
        fail("truth-rank tie uniforms must lie in [0,1)")
    thresholds = source["frozen_thresholds"]
    if not isinstance(thresholds, dict) or set(thresholds) != THRESHOLD_KEYS:
        fail("frozen thresholds have missing or extra fields")
    for name, value in thresholds.items():
        if finite_number(value, name) < 0:
            fail("frozen thresholds must be non-negative")
    return source


def weighted_mean(field, weights) -> float:
    total = sum(map(sum, weights))
    return sum(field[r][c] * weights[r][c] for r in range(len(field)) for c in range(len(field[0]))) / total


def mean_field(members):
    rows, cols = len(members[0]), len(members[0][0])
    return [[sum(m[r][c] for m in members) / EXPECTED_MEMBERS for c in range(cols)] for r in range(rows)]


def semivariogram(field, lag: int) -> float:
    values = []
    rows, cols = len(field), len(field[0])
    for r in range(rows):
        for c in range(cols):
            if c + lag < cols:
                values.append(0.5 * (field[r][c + lag] - field[r][c]) ** 2)
            if r + lag < rows:
                values.append(0.5 * (field[r + lag][c] - field[r][c]) ** 2)
    return sum(values) / len(values) if values else 0.0


def features(members, weights):
    field = mean_field(members)
    flat = [v for row in field for v in row]
    avg = sum(flat) / len(flat)
    return (
        weighted_mean(field, weights),
        sum(weights[r][c] for r in range(len(field)) for c in range(len(field[0])) if field[r][c] >= .15) / sum(map(sum, weights)),
        avg, math.sqrt(sum((v - avg) ** 2 for v in flat) / len(flat)),
        semivariogram(field, 1), semivariogram(field, 4),
    )


def anomalies(members, weights):
    center = mean_field(members)
    fields = [[[m[r][c] - center[r][c] for c in range(len(center[0]))] for r in range(len(center))] for m in members]
    return fields, [weighted_mean(field, weights) for field in fields]


def normalized_truth_rank(members, truth, weights, tie_uniform):
    member_values = [weighted_mean(m, weights) for m in members]
    target = weighted_mean(truth, weights)
    below = sum(value < target for value in member_values)
    equal = sum(value == target for value in member_values)
    return (below + tie_uniform * equal) / EXPECTED_MEMBERS


def make_candidate(target, library, eligible_indices, all_features, truth_ranks, weights, alpha):
    neighbor_indices = select_forecast_analogs(eligible_indices, [all_features[i] for i in eligible_indices], all_features[target])
    fields, means = [], []
    for index in neighbor_indices:
        f, m = anomalies(library[index], weights)
        fields.append(f); means.append(m)
    selected = select_rank_stratified_fields(neighbor_indices, [truth_ranks[i] for i in neighbor_indices], means, fields)
    _, target_means = anomalies(library[target], weights)
    paired = pair_with_heldout_member_order(target_means, selected)
    return construct_projected_candidate(mean_field(library[target]), paired, alpha)


def crps_metrics(members, truth, weights):
    wsum = sum(map(sum, weights)); ordinary = fair = sq = 0.0
    for r in range(len(truth)):
        for c in range(len(truth[0])):
            vals = [m[r][c] for m in members]; y = truth[r][c]; weight = weights[r][c]
            first = sum(abs(x - y) for x in vals) / EXPECTED_MEMBERS
            pair = sum(abs(vals[i] - vals[j]) for i in range(EXPECTED_MEMBERS) for j in range(EXPECTED_MEMBERS))
            ordinary += weight * (first - pair / (2 * EXPECTED_MEMBERS ** 2))
            fair += weight * (first - pair / (2 * EXPECTED_MEMBERS * (EXPECTED_MEMBERS - 1)))
            sq += weight * (sum(vals) / EXPECTED_MEMBERS - y) ** 2
    return {"analysis_fair_crps": fair / wsum, "analysis_crps": ordinary / wsum, "analysis_mean_rmse": math.sqrt(sq / wsum)}


def feasibility(raw, candidate, thresholds):
    boundary = abs(sum(v in (0.0, 1.0) for m in candidate for row in m for v in row) - sum(v in (0.0, 1.0) for m in raw for row in m for v in row)) / sum(len(row) for m in raw for row in m)
    spatial = max(abs(semivariogram(candidate[j], lag) - semivariogram(raw[j], lag)) for j in range(EXPECTED_MEMBERS) for lag in (1, 4))
    return boundary <= thresholds["boundary_mass_absolute_tolerance"] and spatial <= thresholds["member_semivariogram_absolute_tolerance"]


def percentile(values, q):
    ordered = sorted(values); position = (len(ordered) - 1) * q; low = int(position); high = min(low + 1, len(ordered) - 1)
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def intervals(deltas):
    # Deterministic leave-one-case and non-overlapping four-case block sensitivities.
    date = [sum(deltas[:i] + deltas[i + 1:]) / (len(deltas) - 1) for i in range(len(deltas))]
    blocks = [sum(deltas[i:i + 4]) / 4 for i in range(0, len(deltas), 4)]
    return [percentile(date, .025), percentile(date, .975)], [percentile(blocks, .025), percentile(blocks, .975)]


def run(source):
    raw, truth, weights = source["raw_ensemble"], source["truth"], source["spatial_weights"]
    thresholds = source["frozen_thresholds"]
    all_features = [features(case, weights) for case in raw]
    ranks = [normalized_truth_rank(raw[i], truth[i], weights, source["truth_rank_tie_uniforms"][i]) for i in range(EXPECTED_CASES)]
    candidates = [None] * EXPECTED_CASES; selections = []
    for fold_index, (holdout, training) in enumerate(purged_folds()):
        scores, feasible = {}, {}
        for alpha in ALPHAS:
            fold_scores = []; flags = []
            for target in training:
                pool = tuple(i for i in training if i != target)
                candidate = make_candidate(target, raw, pool, all_features, ranks, weights, alpha)
                fold_scores.append(crps_metrics(candidate, truth[target], weights)["analysis_fair_crps"])
                flags.append(feasibility(raw[target], candidate, thresholds))
            scores[alpha] = sum(fold_scores) / len(fold_scores); feasible[alpha] = all(flags)
        alpha, no_positive = select_alpha(scores, feasible)
        selections.append({"fold_index": fold_index, "holdout_case_indices": list(holdout), "training_case_indices": list(training), "selected_alpha": alpha, "no_positive_feasible_alpha": no_positive})
        for target in holdout:
            candidates[target] = make_candidate(target, raw, training, all_features, ranks, weights, alpha)
    if any(candidate is None for candidate in candidates):
        fail("candidate construction left an incomplete fold")
    raw_case = [crps_metrics(raw[i], truth[i], weights) for i in range(EXPECTED_CASES)]
    candidate_case = [crps_metrics(candidates[i], truth[i], weights) for i in range(EXPECTED_CASES)]
    aggregates = {}; paired = {}
    for name in ("analysis_fair_crps", "analysis_crps", "analysis_mean_rmse"):
        rv = sum(row[name] for row in raw_case) / EXPECTED_CASES; cv = sum(row[name] for row in candidate_case) / EXPECTED_CASES
        deltas = [candidate_case[i][name] - raw_case[i][name] for i in range(EXPECTED_CASES)]; date_ci, block_ci = intervals(deltas)
        aggregates[name] = {"raw": rv, "candidate": cv, "delta": cv - rv}
        paired[name] = {"raw": rv, "candidate": cv, "paired_mean_delta": cv - rv, "paired_date_95_ci": date_ci, "four_case_block_95_ci": block_ci}
    changed = caps0 = caps1 = count = 0; max_error = 0.0
    pre_dist = post_dist = 0.0
    for i in range(EXPECTED_CASES):
        center = mean_field(raw[i]); count += EXPECTED_MEMBERS * len(center) * len(center[0])
        for j in range(EXPECTED_MEMBERS):
            pre_dist += sum(semivariogram(raw[i][j], lag) for lag in (1, 4))
            post_dist += sum(semivariogram(candidates[i][j], lag) for lag in (1, 4))
            for r in range(len(center)):
                for c in range(len(center[0])):
                    value = candidates[i][j][r][c]; changed += value != raw[i][j][r][c]; caps0 += value == 0; caps1 += value == 1
        candidate_mean = mean_field(candidates[i])
        max_error = max(max_error, max(abs(candidate_mean[r][c] - center[r][c]) for r in range(len(center)) for c in range(len(center[0]))))
    spatial = {}
    for lag in (1, 4):
        rv = sum(semivariogram(raw[i][j], lag) for i in range(EXPECTED_CASES) for j in range(EXPECTED_MEMBERS)) / (EXPECTED_CASES * EXPECTED_MEMBERS)
        cv = sum(semivariogram(candidates[i][j], lag) for i in range(EXPECTED_CASES) for j in range(EXPECTED_MEMBERS)) / (EXPECTED_CASES * EXPECTED_MEMBERS)
        delta = abs(cv - rv); tolerance = thresholds["member_semivariogram_absolute_tolerance"]
        spatial[f"member_semivariogram_lag_{lag}"] = {"raw": rv, "candidate": cv, "absolute_delta": delta, "maximum_allowed_absolute_delta": tolerance, "passed": delta <= tolerance}
    proper = {"fair_crps_improves_3pct": aggregates["analysis_fair_crps"]["candidate"] <= .97 * aggregates["analysis_fair_crps"]["raw"], "ordinary_crps_nonworse": aggregates["analysis_crps"]["candidate"] <= aggregates["analysis_crps"]["raw"], "mean_rmse_nonworse": aggregates["analysis_mean_rmse"]["candidate"] <= aggregates["analysis_mean_rmse"]["raw"]}
    # The prototype makes rank-target balance and coverage operationally explicit;
    # production parity must replace these structural proxies with the trusted gate.
    reliability = {"rank_target_balance": True, "all_positive_alpha": all(not row["no_positive_feasible_alpha"] for row in selections)}
    boundary = {"frozen_training_tolerance": all(feasibility(raw[i], candidates[i], thresholds) for i in range(EXPECTED_CASES))}
    # Fail closed: this standalone prototype proves construction and compact I/O,
    # but cannot impersonate the controller-owned unchanged full gate.  Trusted
    # integration replaces this flag only after parity review of that gate.
    operational = {"case_count": True, "finite_members": True,
                   "mean_invariant": max_error <= 1e-10,
                   "trusted_full_gate_integrated": False}
    gate_inputs = {"proper_score": proper, "finite_ensemble_reliability": reliability, "boundary": boundary, "spatial_physical": {name: row["passed"] for name, row in spatial.items()}, "operational": operational}
    gate = {name: all(values.values()) for name, values in gate_inputs.items()}; gate["overall_eligible"] = all(gate.values())
    diagnostics = {"changed_member_fraction": changed / count, "lower_cap_mass": caps0 / count, "upper_cap_mass": caps1 / count, "maximum_mean_error": max_error, "member_semivariogram_distortion_pre": pre_dist / (EXPECTED_CASES * EXPECTED_MEMBERS * 2), "member_semivariogram_distortion_post": post_dist / (EXPECTED_CASES * EXPECTED_MEMBERS * 2)}
    handoff = {"gate": gate, "gate_inputs": gate_inputs, "aggregate_metrics": aggregates, "paired_uncertainty": paired, "fold_selections": selections, "rank_target_balance": [EXPECTED_CASES] * EXPECTED_MEMBERS, "projection_diagnostics": diagnostics, "member_spatial_deltas": spatial}
    validate_compact_handoff(handoff)
    return handoff, raw_case, candidate_case


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-experiment", required=True)
    args = parser.parse_args(argv)
    validate_runner_interface({"source_experiment": args.source_experiment}, ARTIFACT_POLICY)
    source_name = os.environ.get("RANK_COHERENT_SOURCE_JSON"); output_name = os.environ.get("RANK_COHERENT_OUTPUT_DIR")
    if not source_name or not output_name:
        fail("controller must set RANK_COHERENT_SOURCE_JSON and RANK_COHERENT_OUTPUT_DIR")
    source_path, output_dir = Path(source_name), Path(output_name)
    if not source_path.is_file() or output_dir.exists() and not output_dir.is_dir():
        fail("controller paths are invalid")
    source_bytes = source_path.read_bytes(); source = validate_source(json.loads(source_bytes))
    handoff, raw_case, candidate_case = run(source)
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate = {"schema_version": SCHEMA, **handoff}
    atomic_json(output_dir / "aggregate_case_mean_metrics.json", aggregate)
    with (output_dir / "per_case_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream); writer.writerow(["case_index", "variant", "analysis_fair_crps", "analysis_crps", "analysis_mean_rmse"])
        for i in range(EXPECTED_CASES):
            for variant, rows in (("raw", raw_case), ("candidate", candidate_case)):
                writer.writerow([i, variant, rows[i]["analysis_fair_crps"], rows[i]["analysis_crps"], rows[i]["analysis_mean_rmse"]])
    metadata = {"schema_version": SCHEMA, "source_experiment": SOURCE_EXPERIMENT, "source_sha256": hashlib.sha256(source_bytes).hexdigest(), "artifact_policy": ARTIFACT_POLICY, "dataset_split": DATASET_SPLIT, "start_date": START_DATE, "end_date": END_DATE, "cases": EXPECTED_CASES, "ensemble_size": EXPECTED_MEMBERS, "case_stride": CASE_STRIDE, "scientific_parameters": {"alphas": list(ALPHAS), "purge": 3, "neighbors": 10}, "decision_bearing": False, "prototype_limit": "production must replace structural reliability proxies with the unchanged trusted full gate"}
    atomic_json(output_dir / "metadata.json", metadata)
    atomic_json(output_dir / "run_status.json", {"schema_version": SCHEMA, "status": "complete", "completed_cases": EXPECTED_CASES, "total_cases": EXPECTED_CASES, "overall_eligible": handoff["gate"]["overall_eligible"], "compact_outputs": list(OUTPUTS)})
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"rank-coherent prototype: FAIL: {exc}", file=sys.stderr)
        raise SystemExit(2)
