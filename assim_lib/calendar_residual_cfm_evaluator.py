"""Fail-closed contracts for calendar-residual CFM sampling and calibration.

The trusted controller owns scheduling and server paths.  This module deliberately
accepts a completed dependency manifest instead of a run directory and returns a
typed invocation; it never constructs or executes a shell command.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

SAMPLING_MODE = "validation_siconc_calendar_residual_cfm_sampling"
GATE_MODE = "validation_siconc_calendar_residual_cfm_gate"
SOURCE_EXPERIMENT = "siconc_calendar_residual_cfm_training_retry1"
CHECKPOINT_NAME = "ema_last_model.pth"
CASE_COUNT = 40
MEMBER_COUNT = 10
SHA256_RE = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class CompletedCheckpoint:
    experiment_id: str
    run_dir: Path
    checkpoint_path: Path
    checkpoint_sha256: str
    publication_commit: str


def _require_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"completed dependency metadata requires string {key!r}")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_completed_checkpoint(
    metadata: Mapping[str, Any], *, dependency_root: Path
) -> CompletedCheckpoint:
    """Resolve and hash the sole permitted checkpoint from compact metadata."""
    if _require_string(metadata, "experiment_id") != SOURCE_EXPERIMENT:
        raise ValueError("unexpected source experiment")
    if metadata.get("status") != "completed":
        raise ValueError("source experiment is not completed")
    if _require_string(metadata, "checkpoint") != CHECKPOINT_NAME:
        raise ValueError("source did not publish the frozen final EMA checkpoint")
    expected_sha = _require_string(metadata, "checkpoint_sha256")
    if SHA256_RE.fullmatch(expected_sha) is None:
        raise ValueError("checkpoint_sha256 is not a lowercase SHA-256 digest")

    root = dependency_root.resolve(strict=True)
    recorded = Path(_require_string(metadata, "training_run_dir"))
    run_dir = recorded.resolve(strict=True)
    if run_dir.parent != root or run_dir.name != "training":
        raise ValueError("training_run_dir is outside the completed dependency root")
    checkpoint = (run_dir / CHECKPOINT_NAME).resolve(strict=True)
    if checkpoint.parent != run_dir or checkpoint.is_symlink() or not checkpoint.is_file():
        raise ValueError("checkpoint is not a regular in-root file")
    actual_sha = _file_sha256(checkpoint)
    if actual_sha != expected_sha:
        raise ValueError("checkpoint hash differs from completed compact metadata")
    return CompletedCheckpoint(
        experiment_id=SOURCE_EXPERIMENT,
        run_dir=run_dir,
        checkpoint_path=checkpoint,
        checkpoint_sha256=actual_sha,
        publication_commit=_require_string(metadata, "publication_commit"),
    )


def sampling_argv(checkpoint: CompletedCheckpoint, output_dir: Path) -> Sequence[str]:
    """Return a literal argv vector for the unchanged evaluator."""
    return (
        "python3",
        "-m",
        "assim_lib.evaluate",
        "--config",
        "config/experiments/evaluate_siconc_calendar_residual_cfm.json",
        "--run-dir",
        str(checkpoint.run_dir),
        "--output-dir",
        str(output_dir),
        "--checkpoint-name",
        CHECKPOINT_NAME,
    )


def frozen_gate_contract() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "sampling_mode": SAMPLING_MODE,
        "gate_mode": GATE_MODE,
        "source_experiment": SOURCE_EXPERIMENT,
        "schedule": {
            "dataset_split": "valid",
            "start_date": "2022-01-01",
            "end_date": "2022-07-15",
            "stride_days": 5,
            "cases": CASE_COUNT,
            "ensemble_size": MEMBER_COUNT,
            "seed": 1234,
            "iid_exchangeable": True,
        },
        "sampling": {
            "checkpoint": CHECKPOINT_NAME,
            "checkpoint_selection": "completed_metadata_sha256_only",
            "conditioning": "full",
            "solver": "dopri5",
            "num_timesteps": 25,
            "rtol": 1e-5,
            "atol": 1e-6,
            "cfg_mode": "none",
            "fallback_allowed": False,
        },
        "reliability": {
            "primary": "absolute_randomized_rank_histogram",
            "tie_seed": 20220831,
            "bins": 11,
            "aggregation": "equal_date",
            "tv_to_uniform_max": 0.10,
            "max_bin_deviation_max": 0.03,
            "normalized_mean_rank_range": [0.45, 0.55],
            "normalized_mean_rank_block_ci_contains": 0.5,
            "attainable_range_target": 9 / 11,
            "attainable_inner_target": 7 / 11,
            "coverage_absolute_error_max": 0.05,
        },
        "uncertainty": {
            "paired_date_bootstrap": True,
            "block_bootstrap": {"kind": "fixed_non_circular", "cases": 4},
        },
        "proper": ["analysis_fair_crps", "analysis_crps", "energy_score"],
        "mean_safety": ["analysis_mean_rmse"],
        "boundary": {
            "reference": "truth",
            "thresholds": [0.0, 0.15, 0.90, 0.95, 0.99],
            "metrics": ["date_balanced_brier", "reliability_l1", "marginal_frequency_error"],
            "diagnostic_only": ["exact_one_mass", "ge_0.999_mass"],
        },
        "spatial": {
            "metrics": ["IIEE", "edge", "area", "extent", "pooled_member_semivariograms", "local_variogram_score"],
            "worst_individual_member_gate": False,
            "finite_bounds": [0.0, 1.0],
        },
        "decision": {"no_compensation": True, "requires": "all_families_true"},
        "artifacts": {"policy": "summary_only", "raw_arrays": "server_only"},
    }


def load_completed_metadata(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("completed dependency metadata must be a JSON object")
    return payload
