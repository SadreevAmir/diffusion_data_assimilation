#!/usr/bin/env python3
"""Exact four-file validator for observation-likelihood compact outputs."""

import hashlib, json, math
from pathlib import Path

FILES = ("case_weights.json", "aggregate_weights.json", "paired_uncertainty.json", "gate_decision.json")
FAMILIES = ("proper_score", "reliability", "boundary", "spatial_physical", "operational")


def directory_sha256(directory):
    if {p.name for p in Path(directory).iterdir() if p.is_file()} != set(FILES):
        raise ValueError("compact directory must contain exactly four frozen files")
    digest = hashlib.sha256()
    for name in sorted(FILES):
        data = (Path(directory) / name).read_bytes(); encoded = name.encode()
        digest.update(len(encoded).to_bytes(4, "big") + encoded + len(data).to_bytes(8, "big") + data)
    return digest.hexdigest()


def validate_directory(directory):
    directory_sha256(directory)
    docs = {name: json.loads((Path(directory) / name).read_text()) for name in FILES}
    cases = docs["case_weights.json"]
    if set(cases) != {"schema_version", "cases"} or cases["schema_version"] != "observation-likelihood-case-weights-v1" or not isinstance(cases["cases"], list) or len(cases["cases"]) != 40:
        raise ValueError("case_weights.json has wrong schema")
    identifiers, ess, fair_deltas, crps_deltas = set(), [], [], []
    for case in cases["cases"]:
        if set(case) != {"case_id", "fold", "weights", "effective_sample_size", "source_hash_match", "mask_invariants_pass", "analysis_fair_crps_delta", "analysis_crps_delta"}:
            raise ValueError("case weight has wrong schema")
        if case["case_id"] in identifiers: raise ValueError("duplicate case_id")
        identifiers.add(case["case_id"])
        weights = case["weights"]
        if len(weights) != 10 or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0 for x in weights) or abs(sum(weights)-1) > 1e-12:
            raise ValueError("invalid weights")
        expected = 1 / sum(x*x for x in weights)
        if not math.isclose(case["effective_sample_size"], expected, rel_tol=1e-12, abs_tol=1e-12): raise ValueError("ESS mismatch")
        if not isinstance(case["fold"], int) or isinstance(case["fold"], bool) or not 0 <= case["fold"] < 5: raise ValueError("invalid fold")
        if not isinstance(case["source_hash_match"], bool) or not isinstance(case["mask_invariants_pass"], bool): raise ValueError("invariant flags must be Boolean")
        for key, destination in (("analysis_fair_crps_delta", fair_deltas), ("analysis_crps_delta", crps_deltas)):
            value = case[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value): raise ValueError("score delta must be finite")
            destination.append(float(value))
        ess.append(expected)
    aggregate = docs["aggregate_weights.json"]
    if set(aggregate) != {"schema_version", "num_cases", "median_effective_sample_size", "minimum_effective_sample_size", "all_source_hashes_match", "all_mask_invariants_pass"} or aggregate["schema_version"] != "observation-likelihood-aggregate-v1" or aggregate["num_cases"] != 40:
        raise ValueError("aggregate_weights.json has wrong schema")
    ordered = sorted(ess); median = (ordered[19] + ordered[20]) / 2
    if not math.isclose(aggregate["median_effective_sample_size"], median, rel_tol=1e-12) or not math.isclose(aggregate["minimum_effective_sample_size"], min(ess), rel_tol=1e-12): raise ValueError("aggregate ESS mismatch")
    hash_pass = all(case["source_hash_match"] for case in cases["cases"])
    mask_pass = all(case["mask_invariants_pass"] for case in cases["cases"])
    if aggregate["all_source_hashes_match"] is not hash_pass or aggregate["all_mask_invariants_pass"] is not mask_pass: raise ValueError("aggregate invariant mismatch")
    uncertainty = docs["paired_uncertainty.json"]
    if set(uncertainty) != {"schema_version", "analysis_fair_crps", "analysis_crps"} or uncertainty["schema_version"] != "observation-likelihood-paired-uncertainty-v1": raise ValueError("uncertainty schema mismatch")
    for metric, deltas in (("analysis_fair_crps", fair_deltas), ("analysis_crps", crps_deltas)):
        summary = uncertainty[metric]
        if not isinstance(summary, dict) or set(summary) != {"point_delta", "date_interval", "four_case_block_interval"}: raise ValueError("uncertainty metric schema mismatch")
        point = sum(deltas) / 40
        if not math.isclose(summary["point_delta"], point, rel_tol=1e-12, abs_tol=1e-12): raise ValueError("uncertainty point mismatch")
        for name in ("date_interval", "four_case_block_interval"):
            interval = summary[name]
            if not isinstance(interval, list) or len(interval) != 2 or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in interval) or interval[0] > point or interval[1] < point: raise ValueError("invalid uncertainty interval")
    gate = docs["gate_decision.json"]
    if set(gate) != {"schema_version", "families", "overall_eligible"} or gate["schema_version"] != "observation-likelihood-no-compensation-gate-v1" or set(gate["families"]) != set(FAMILIES) or any(not isinstance(v, bool) for v in gate["families"].values()) or gate["overall_eligible"] is not all(gate["families"].values()):
        raise ValueError("gate schema or conjunction mismatch")
    operational = hash_pass and mask_pass and median >= 3 and min(ess) >= 2
    if gate["families"]["operational"] != operational: raise ValueError("operational gate mismatch")
