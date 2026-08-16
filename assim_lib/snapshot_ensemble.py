"""Frozen server-side last/EMA-last checkpoint ensemble.

This is a bounded diagnostic for the existing learned-joint run.  It deliberately
avoids the ``best`` checkpoints because that legacy run evaluated 2023 while
training.  The two fixed last-epoch states are sampled with common random noise;
all raw arrays remain on the server.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from .deep_ensemble import (
    COMPACT_OUTPUTS,
    _atomic_csv,
    _atomic_json,
    _canonical_hash,
    _file_hash,
    _run_checked,
    _sample_manifest,
)

MODE = "validation_checkpoint_trajectory_ema_sampling"
SOURCE_EXPERIMENT = "joint_full_condition_validation_2022"
CASE_COUNT = 40
MEMBER_COUNT = 10
MEMBERS_PER_CHECKPOINT = 5
BASE_NOISE_SEED = 2401
EXPECTED_DATES = tuple(date(2022, 1, 1) + timedelta(days=5 * i) for i in range(CASE_COUNT))
SOURCE_CONFIG_SHA256 = "8623a048735b0f739807c90dd038cf52e9e834eae29fd32ff3b73ad4e8505634"
SOURCE_METADATA_SHA256 = "5fcb930c084d7bb7c6367a81bd5272e9bdee2614d949780d6d18c9d82553c95f"
SOURCE_METRICS_SHA256 = "c95dad7a279ec947b936a28b98b922b09222b438be97f8dc0839c793ff5b9ae3"
CHECKPOINTS = (
    (
        "last",
        "last_model.pth",
        "bf7b1248b0b310c4387a2bda1a309c211dc01e5cebd2c7348bbb22799e0880e1",
    ),
    (
        "ema_last",
        "ema_last_model.pth",
        "cd73cedc97a9f19d15c70ba31d28d3248a77dc6ef568edf8793326763810fdfe",
    ),
)


def noise_seed(case_index: int, member_index: int) -> int:
    if not 0 <= case_index < CASE_COUNT or not 0 <= member_index < MEMBERS_PER_CHECKPOINT:
        raise ValueError("case or member index outside frozen schedule")
    return BASE_NOISE_SEED + case_index * MEMBERS_PER_CHECKPOINT + member_index


def case_plan(case_index: int) -> tuple[tuple[str, int, int], ...]:
    """Interleave paired last/EMA-last members with exactly common noise."""
    return tuple(
        (label, member_index, noise_seed(case_index, member_index))
        for member_index in range(MEMBERS_PER_CHECKPOINT)
        for label, _, _ in CHECKPOINTS
    )


def _validate_source(run_dir: Path) -> list[dict[str, Any]]:
    expected = {
        "config.json": SOURCE_CONFIG_SHA256,
        "metadata.json": SOURCE_METADATA_SHA256,
        "metrics.json": SOURCE_METRICS_SHA256,
    }
    for name, digest in expected.items():
        path = run_dir / name
        if not path.is_file() or _file_hash(path) != digest:
            raise ValueError(f"source {name} differs from the frozen inventory")
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    training = metadata.get("training_config") or {}
    data = metadata.get("data_config") or {}
    if training.get("num_epochs") != 40 or training.get("seed") != 0:
        raise ValueError("legacy source training identity differs")
    if data.get("train", {}).get("obs_end_day") != "2021-12-31":
        raise ValueError("legacy source training period differs")
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    if not isinstance(metrics, list) or [row.get("epoch") for row in metrics] != list(range(15)):
        raise ValueError("legacy source epoch inventory differs")
    records: list[dict[str, Any]] = []
    for label, filename, digest in CHECKPOINTS:
        path = run_dir / filename
        if not path.is_file() or _file_hash(path) != digest:
            raise ValueError(f"checkpoint {label} differs from the frozen inventory")
        records.append(
            {
                "checkpoint_label": label,
                "checkpoint_name": filename,
                "checkpoint_sha256": digest,
                "fixed_available_epoch_zero_based": 14,
                "validation_selected": False,
            }
        )
    return records


def run(output_dir: Path, source_experiment: str, run_dir: Path) -> None:
    if source_experiment != SOURCE_EXPERIMENT:
        raise ValueError("source_experiment differs from the frozen contract")
    output_dir = output_dir.resolve()
    run_dir = run_dir.resolve()
    if output_dir.is_symlink() or run_dir.is_symlink():
        raise ValueError("input and output directories cannot be symlinks")
    output_dir.mkdir(parents=True, exist_ok=True)
    unexpected = {path.name for path in output_dir.iterdir()} - COMPACT_OUTPUTS - {
        "internal",
        "samples",
    }
    if unexpected:
        raise ValueError(f"output_dir contains unexpected entries: {sorted(unexpected)}")
    repo = Path(__file__).resolve().parents[1]
    internal = output_dir / "internal"
    internal.mkdir(exist_ok=True)
    try:
        checkpoint_records = _validate_source(run_dir)
        sampling_roots: dict[str, Path] = {}
        for order, (label, filename, _) in enumerate(CHECKPOINTS):
            sample_root = internal / f"samples_{label}"
            _run_checked(
                [
                    sys.executable,
                    "-m",
                    "assim_lib.compare_3dvar",
                    "--config",
                    str(repo / "config/experiments/experiment_m2m_flow_modes.json"),
                    "--mode",
                    "validation_clean_checkpoint_member_sampling",
                    "--run-dir",
                    str(run_dir),
                    "--checkpoint-name",
                    filename,
                    "--output-dir",
                    str(sample_root),
                    "--ensemble-size",
                    str(MEMBERS_PER_CHECKPOINT),
                    "--sample-batch-size",
                    str(MEMBERS_PER_CHECKPOINT),
                    "--seed",
                    str(BASE_NOISE_SEED),
                    "--save-ensembles",
                ],
                repo,
                output_dir,
                f"sampling_{label}",
                order,
                total_units=2 + CASE_COUNT,
            )
            sampling_roots[label] = sample_root

        source_cases = {
            label: json.loads((root / "cases.json").read_text(encoding="utf-8"))
            for label, root in sampling_roots.items()
        }
        samples_dir = output_dir / "samples"
        samples_dir.mkdir(exist_ok=True)
        output_paths: list[Path] = []
        manifest_cases: list[dict[str, Any]] = []
        per_case: list[dict[str, object]] = []
        checkpoint_hashes = {row[0]: row[2] for row in CHECKPOINTS}
        for case_index, expected_date in enumerate(EXPECTED_DATES):
            filename = f"{case_index:03d}_{expected_date.isoformat()}_h23.npz"
            payloads: dict[str, dict[str, np.ndarray]] = {}
            for label, root in sampling_roots.items():
                with np.load(root / "samples" / filename, allow_pickle=False) as payload:
                    payloads[label] = {key: payload[key] for key in payload.files}
            reference = payloads[CHECKPOINTS[0][0]]
            other = payloads[CHECKPOINTS[1][0]]
            for key in (
                "truth",
                "background",
                "obs_values",
                "conditioning_mask",
                "track_imitation_mask",
                "valid_mask",
            ):
                if not np.array_equal(reference[key], other[key], equal_nan=True):
                    raise ValueError(f"conditioning context differs for case {case_index}")

            members: list[np.ndarray] = []
            member_records: list[dict[str, object]] = []
            noise_by_member: dict[int, str] = {}
            for label, member_index, actual_seed in case_plan(case_index):
                member = np.asarray(payloads[label]["analysis_ensemble"][member_index], dtype=np.float32)
                if not np.all(np.isfinite(member)):
                    raise ValueError(f"non-finite member for case {case_index}")
                noise_hash = str(source_cases[label][case_index]["initial_noise_sha256"][member_index])
                prior = noise_by_member.setdefault(member_index, noise_hash)
                if prior != noise_hash:
                    raise ValueError("paired checkpoints did not receive common initial noise")
                members.append(member)
                member_records.append(
                    {
                        "member_id": f"case{case_index:02d}-{label}-z{actual_seed}",
                        "checkpoint_label": label,
                        "checkpoint_hash": checkpoint_hashes[label],
                        "noise_seed": actual_seed,
                        "initial_noise_hash": noise_hash,
                        "finite": True,
                    }
                )
            candidate = np.stack(members)
            target = samples_dir / filename
            np.savez_compressed(
                target,
                analysis_ensemble=candidate,
                analysis_mean=candidate.mean(axis=0, dtype=np.float64).astype(np.float32),
                truth=reference["truth"],
                background=reference["background"],
                obs_values=reference["obs_values"],
                conditioning_mask=reference["conditioning_mask"],
                track_imitation_mask=reference["track_imitation_mask"],
                valid_mask=reference["valid_mask"],
            )
            output_paths.append(target)
            manifest_cases.append({"case_index": case_index, "members": member_records})
            per_case.append(
                {
                    "case_index": case_index,
                    "target_date": expected_date.isoformat(),
                    "ensemble_size": MEMBER_COUNT,
                    "last_members": 5,
                    "ema_last_members": 5,
                    "finite_members": MEMBER_COUNT,
                    "sample_sha256": _file_hash(target),
                }
            )

        admission_manifest = {
            "schema_version": 1,
            "source_experiment": SOURCE_EXPERIMENT,
            "source_config_sha256": SOURCE_CONFIG_SHA256,
            "source_metadata_sha256": SOURCE_METADATA_SHA256,
            "source_metrics_sha256": SOURCE_METRICS_SHA256,
            "checkpoint_records": checkpoint_records,
            "cases": manifest_cases,
        }
        _atomic_csv(output_dir / "per_case_metrics.csv", per_case)
        metadata = {
            "status": "completed",
            "mode": MODE,
            "source_experiment": SOURCE_EXPERIMENT,
            "dataset_split": "valid",
            "date_range": ["2022-01-01", "2022-07-15"],
            "case_stride_days": 5,
            "num_cases": CASE_COUNT,
            "ensemble_size": MEMBER_COUNT,
            "member_allocation": {"last": 5, "ema_last": 5},
            "base_noise_seed": BASE_NOISE_SEED,
            "conditioning_mode": "full",
            "sampler": {"method": "dopri5", "num_timesteps": 25, "rtol": 1e-5, "atol": 1e-6},
            "admission_manifest": admission_manifest,
            "admission_manifest_sha256": _canonical_hash(admission_manifest),
            "sample_manifest_sha256": _sample_manifest(output_paths),
            "fallback_member_count": 0,
            "raw_arrays_retrieved": False,
            "test_data_used_for_this_selection": False,
            "gate_pending": True,
            "scientific_role": "fast frozen diagnostic before clean independent-seed retraining",
        }
        _atomic_json(output_dir / "metadata.json", metadata)
        _atomic_json(
            output_dir / "aggregate_case_mean_metrics.json",
            {
                "schema_version": 1,
                "status": "sampling_completed_gate_pending",
                "num_cases": CASE_COUNT,
                "ensemble_size": MEMBER_COUNT,
                "admission_manifest_sha256": metadata["admission_manifest_sha256"],
            },
        )
        _atomic_json(
            output_dir / "run_status.json",
            {
                "status": "completed",
                "experiment_id": output_dir.parent.name,
                "completed_cases": CASE_COUNT,
                "total_cases": CASE_COUNT,
                "progress_percent": 100.0,
                "gate_pending": True,
            },
        )
    except Exception as error:
        for name in COMPACT_OUTPUTS - {"run_status.json"}:
            (output_dir / name).unlink(missing_ok=True)
        _atomic_json(
            output_dir / "run_status.json",
            {
                "status": "failed",
                "experiment_id": output_dir.parent.name,
                "error_type": type(error).__name__,
            },
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-experiment", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.output_dir, args.source_experiment, args.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
