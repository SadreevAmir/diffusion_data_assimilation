"""Fail-closed local oracle for the frozen probabilistic-DA compact contract.

This module validates a prospective trusted runner's compact manifest and
decision summary.  It deliberately does not implement LETKF or admit a mode.
"""
from __future__ import annotations

import math
import re
from typing import Any


CONTRACT_VERSION = "probabilistic_da_letkf_v1"
SOURCE_EXPERIMENT = "joint_full_condition_validation_2022"
RADII_KM = (50, 100, 200, 400)
INFLATIONS = (1.00, 1.05, 1.10, 1.20)
EXPECTED_ARTIFACTS = (
    "probabilistic_da_summary.json",
    "probabilistic_da_per_case.csv",
    "probabilistic_da_manifest.json",
)
CASE_COLUMNS = (
    "case_index", "target_date", "fold", "method", "fair_crps", "crps",
    "randomized_rank", "central_coverage", "central_width", "mean_rmse", "iiee",
    "ice_area_error", "ice_extent_error", "edge_error", "variogram_discrepancy",
)
_HASH = re.compile(r"[0-9a-f]{64}")


def purged_folds() -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    """Return literal held-out and non-circular three-case-purged train folds."""
    folds = []
    universe = set(range(40))
    for start in range(0, 40, 8):
        held_out = tuple(range(start, start + 8))
        excluded = set(range(max(0, start - 3), min(40, start + 11)))
        folds.append((held_out, tuple(sorted(universe - excluded))))
    return tuple(folds)


def _exact_keys(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{label} keys must be exact")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return float(value)


def _hash(value: Any, label: str) -> None:
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase sha256")


def _validate_fold(record: Any, fold_index: int) -> None:
    record = _exact_keys(
        record,
        {"fold", "held_out", "training", "candidates", "selected"},
        f"fold {fold_index}",
    )
    held_out, training = purged_folds()[fold_index]
    if record["fold"] != fold_index or record["held_out"] != list(held_out):
        raise ValueError(f"fold {fold_index} held-out block mismatch")
    if record["training"] != list(training):
        raise ValueError(f"fold {fold_index} purge mismatch")
    candidates = record["candidates"]
    if not isinstance(candidates, list) or len(candidates) != 16:
        raise ValueError(f"fold {fold_index} must report all 16 candidates")
    parsed: dict[tuple[int, float], tuple[bool, float | None]] = {}
    for candidate in candidates:
        candidate = _exact_keys(candidate, {"radius_km", "inflation", "feasible", "training_fair_crps"}, "candidate")
        key = (candidate["radius_km"], candidate["inflation"])
        if key not in {(r, i) for r in RADII_KM for i in INFLATIONS} or key in parsed:
            raise ValueError(f"fold {fold_index} candidate grid mismatch")
        feasible = candidate["feasible"]
        if not isinstance(feasible, bool):
            raise ValueError("candidate feasible flag must be boolean")
        score = candidate["training_fair_crps"]
        if feasible:
            score = _finite(score, "training_fair_crps")
        elif score is not None:
            raise ValueError("infeasible candidate score must be null")
        parsed[key] = (feasible, score)
    feasible = [(score, inflation, radius) for (radius, inflation), (ok, score) in parsed.items() if ok]
    if not feasible:
        raise ValueError(f"fold {fold_index} has no feasible candidate")
    _, expected_inflation, expected_radius = min(feasible)
    if record["selected"] != {"radius_km": expected_radius, "inflation": expected_inflation}:
        raise ValueError(f"fold {fold_index} selection or tie-break mismatch")


def validate_manifest(manifest: Any) -> None:
    manifest = _exact_keys(
        manifest,
        {
            "contract_version", "source_experiment", "artifact_policy", "case_count",
            "ensemble_size", "artifacts", "observation_operator", "localization",
            "assimilation", "background", "folds", "invariant_checks",
            "artifact_hashes",
        },
        "manifest",
    )
    if manifest["contract_version"] != CONTRACT_VERSION or manifest["source_experiment"] != SOURCE_EXPERIMENT:
        raise ValueError("contract or source identity mismatch")
    if manifest["artifact_policy"] != "summary_only" or tuple(manifest["artifacts"]) != EXPECTED_ARTIFACTS:
        raise ValueError("compact artifact boundary mismatch")
    artifact_hashes = _exact_keys(
        manifest["artifact_hashes"],
        {"probabilistic_da_summary.json", "probabilistic_da_per_case.csv"},
        "artifact_hashes",
    )
    for name, value in artifact_hashes.items():
        _hash(value, f"{name} artifact hash")
    if manifest["case_count"] != 40 or manifest["ensemble_size"] != 10:
        raise ValueError("case/member envelope mismatch")
    operator = _exact_keys(manifest["observation_operator"], {"name", "output_hashes"}, "observation_operator")
    if not isinstance(operator["name"], str) or not operator["name"]:
        raise ValueError("observation operator name is required")
    hashes = operator["output_hashes"]
    if not isinstance(hashes, list) or len(hashes) != 40:
        raise ValueError("observation output hashes must cover 40 cases")
    for index, record in enumerate(hashes):
        record = _exact_keys(record, {"case_index", "letkf", "learned_joint"}, "observation hash")
        _hash(record["letkf"], "LETKF observation hash"); _hash(record["learned_joint"], "learned-joint observation hash")
        if record["case_index"] != index or record["letkf"] != record["learned_joint"]:
            raise ValueError("observation-operator outputs are not hash-identical")
    localization = _exact_keys(manifest["localization"], {"taper", "radii_km"}, "localization")
    if localization != {"taper": "Gaspari-Cohn", "radii_km": list(RADII_KM)}:
        raise ValueError("localization contract mismatch")
    assimilation = _exact_keys(
        manifest["assimilation"],
        {"filter", "variable", "square_root", "projection", "observation_error_source", "inflations"},
        "assimilation",
    )
    expected_assimilation = {
        "filter": "LETKF", "variable": "physical_SIC", "square_root": "deterministic",
        "projection": "after_complete_analysis_update", "observation_error_source": "sealed_common_contract",
        "inflations": list(INFLATIONS),
    }
    if assimilation != expected_assimilation:
        raise ValueError("assimilation procedure mismatch")
    background = _exact_keys(manifest["background"], {"configuration_digest", "member_hashes"}, "background")
    _hash(background["configuration_digest"], "background configuration digest")
    if not isinstance(background["member_hashes"], list) or len(background["member_hashes"]) != 40:
        raise ValueError("background hashes must cover 40 cases")
    for row in background["member_hashes"]:
        if not isinstance(row, list) or len(row) != 10:
            raise ValueError("background hashes must have shape 40x10")
        for value in row: _hash(value, "background member hash")
    if not isinstance(manifest["folds"], list) or len(manifest["folds"]) != 5:
        raise ValueError("exactly five folds are required")
    for index, fold in enumerate(manifest["folds"]): _validate_fold(fold, index)
    checks = _exact_keys(
        manifest["invariant_checks"],
        {"common_information", "finite", "fold_leakage_absent", "background_independent", "solver_success"},
        "invariant_checks",
    )
    if any(value is not True for value in checks.values()):
        raise ValueError("all comparator invariant checks must pass")


def validate_case_rows(rows: Any) -> None:
    """Validate parsed rows from the single forty-case compact metric CSV."""
    if not isinstance(rows, list) or len(rows) != 120:
        raise ValueError("case metric CSV must contain 120 method-case rows")
    seen = set()
    methods = {"LETKF", "raw_learned_joint", "3D-Var"}
    for row in rows:
        row = _exact_keys(row, set(CASE_COLUMNS), "case metric row")
        case_index = row["case_index"]
        method = row["method"]
        if isinstance(case_index, bool) or not isinstance(case_index, int) or case_index not in range(40):
            raise ValueError("case_index must cover the frozen envelope")
        if method not in methods or (case_index, method) in seen:
            raise ValueError("method-case identities must be unique and exact")
        seen.add((case_index, method))
        if row["fold"] != case_index // 8:
            raise ValueError("case fold identity mismatch")
        if not isinstance(row["target_date"], str) or not row["target_date"]:
            raise ValueError("target_date is required")
        for metric in CASE_COLUMNS[4:]:
            _finite(row[metric], metric)
    if seen != {(index, method) for index in range(40) for method in methods}:
        raise ValueError("case metric CSV does not cover every method-case identity")


def evaluate_decision(summary: Any) -> str:
    summary = _exact_keys(
        summary,
        {"case_count", "letkf_fair_crps", "raw_learned_joint_fair_crps", "rank_uniformity_pass", "letkf_mean_rmse", "var3d_mean_rmse"},
        "summary",
    )
    if summary["case_count"] != 40 or not isinstance(summary["rank_uniformity_pass"], bool):
        raise ValueError("summary envelope or rank decision mismatch")
    letkf_crps = _finite(summary["letkf_fair_crps"], "letkf_fair_crps")
    raw_crps = _finite(summary["raw_learned_joint_fair_crps"], "raw_learned_joint_fair_crps")
    letkf_rmse = _finite(summary["letkf_mean_rmse"], "letkf_mean_rmse")
    var3d_rmse = _finite(summary["var3d_mean_rmse"], "var3d_mean_rmse")
    if raw_crps <= 0 or var3d_rmse <= 0:
        raise ValueError("reference metrics must be positive")
    useful = letkf_crps <= 1.10 * raw_crps and summary["rank_uniformity_pass"] and letkf_rmse <= 1.02 * var3d_rmse
    return "PROBABILISTIC_DA_USEFUL" if useful else "PROBABILISTIC_DA_NEGATIVE"
