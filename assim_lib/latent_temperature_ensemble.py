"""Frozen server-side latent-temperature sampling for the learned-joint ensemble."""

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

MODE = "validation_latent_temperature_sampling"
SOURCE_EXPERIMENT = "joint_full_condition_validation_2022"
CASE_COUNT = 40
MEMBER_COUNT = 10
BASE_NOISE_SEED = 1234
INITIAL_NOISE_SCALE = 1.30
EXPECTED_DATES = tuple(date(2022, 1, 1) + timedelta(days=5 * i) for i in range(CASE_COUNT))
SOURCE_CONFIG_SHA256 = "8623a048735b0f739807c90dd038cf52e9e834eae29fd32ff3b73ad4e8505634"
SOURCE_METADATA_SHA256 = "5fcb930c084d7bb7c6367a81bd5272e9bdee2614d949780d6d18c9d82553c95f"
SOURCE_METRICS_SHA256 = "c95dad7a279ec947b936a28b98b922b09222b438be97f8dc0839c793ff5b9ae3"
CHECKPOINT_NAME = "ema_last_model.pth"
CHECKPOINT_SHA256 = "cd73cedc97a9f19d15c70ba31d28d3248a77dc6ef568edf8793326763810fdfe"


def noise_seed(case_index: int, member_index: int) -> int:
    if not 0 <= case_index < CASE_COUNT or not 0 <= member_index < MEMBER_COUNT:
        raise ValueError("case or member index outside frozen schedule")
    return BASE_NOISE_SEED + case_index * MEMBER_COUNT + member_index


def _validate_source(run_dir: Path) -> dict[str, Any]:
    expected = {
        "config.json": SOURCE_CONFIG_SHA256,
        "metadata.json": SOURCE_METADATA_SHA256,
        "metrics.json": SOURCE_METRICS_SHA256,
        CHECKPOINT_NAME: CHECKPOINT_SHA256,
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
    return {
        "checkpoint_label": "ema_last",
        "checkpoint_name": CHECKPOINT_NAME,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "fixed_available_epoch_zero_based": 14,
        "validation_selected": False,
    }


def _validate_sampling_metadata(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    expected = {
        "dataset_split": "valid",
        "start_date": "2022-01-01",
        "end_date": "2022-07-15",
        "num_cases": CASE_COUNT,
        "case_stride_days": 5,
        "ensemble_size": MEMBER_COUNT,
        "sample_batch_size": MEMBER_COUNT,
        "num_timesteps": 25,
        "method": "dopri5",
        "rtol": 1e-5,
        "atol": 1e-6,
        "inference_precision": "float32",
        "seed": BASE_NOISE_SEED,
        "initial_noise_scale": INITIAL_NOISE_SCALE,
        "conditioning_mode": "full",
        "cfg_mode": "none",
        "cfg_background_scale": 1.0,
        "cfg_observation_scale": 1.0,
        "save_ensembles": True,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"sampling metadata differs for {key}")
    if Path(str(metadata.get("checkpoint", ""))).name != CHECKPOINT_NAME:
        raise ValueError("sampling used an unexpected checkpoint")
    if metadata.get("checkpoint_selection") != (
        "fixed final epoch; legacy 2023 validation was not used for checkpoint selection"
    ):
        raise ValueError("checkpoint selection differs from the frozen contract")
    cases = metadata.get("cases")
    if not isinstance(cases, list) or len(cases) != CASE_COUNT:
        raise ValueError("sampling metadata does not contain exactly forty cases")
    return cases


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
    if (output_dir / "samples").exists():
        raise ValueError("candidate samples already exist")
    repo = Path(__file__).resolve().parents[1]
    internal = output_dir / "internal"
    sampling_root = internal / "sampling"
    internal.mkdir(exist_ok=True)
    try:
        checkpoint_record = _validate_source(run_dir)
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
                CHECKPOINT_NAME,
                "--output-dir",
                str(sampling_root),
                "--ensemble-size",
                str(MEMBER_COUNT),
                "--sample-batch-size",
                str(MEMBER_COUNT),
                "--seed",
                str(BASE_NOISE_SEED),
                "--initial-noise-scale",
                format(INITIAL_NOISE_SCALE, ".2f"),
                "--save-ensembles",
            ],
            repo,
            output_dir,
            "sampling_latent_temperature_1p30",
            0,
            total_units=CASE_COUNT,
        )
        sampling_metadata = json.loads(
            (sampling_root / "metadata.json").read_text(encoding="utf-8")
        )
        cases = _validate_sampling_metadata(sampling_metadata)
        sample_source = sampling_root / "samples"
        samples_dir = output_dir / "samples"
        sample_source.replace(samples_dir)

        output_paths: list[Path] = []
        manifest_cases: list[dict[str, Any]] = []
        per_case: list[dict[str, object]] = []
        for case_index, expected_date in enumerate(EXPECTED_DATES):
            filename = f"{case_index:03d}_{expected_date.isoformat()}_h23.npz"
            path = samples_dir / filename
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"missing frozen sample {filename}")
            with np.load(path, allow_pickle=False) as payload:
                if "analysis_ensemble" not in payload.files:
                    raise ValueError("candidate sample lacks analysis_ensemble")
                ensemble = np.asarray(payload["analysis_ensemble"])
                if ensemble.ndim != 3 or ensemble.shape[0] != MEMBER_COUNT:
                    raise ValueError("candidate sample does not contain exactly ten 2D members")
                if not np.all(np.isfinite(ensemble)):
                    raise ValueError("candidate sample contains non-finite values")
            case = cases[case_index]
            if case.get("case_order") != case_index or case.get("target_date") != expected_date.isoformat():
                raise ValueError("case identity differs from the frozen schedule")
            base_hashes = case.get("initial_noise_sha256")
            scaled_hashes = case.get("scaled_initial_noise_sha256")
            if not isinstance(base_hashes, list) or not isinstance(scaled_hashes, list):
                raise ValueError("initial-noise hashes are missing")
            if len(base_hashes) != MEMBER_COUNT or len(scaled_hashes) != MEMBER_COUNT:
                raise ValueError("initial-noise hash accounting is incomplete")
            members: list[dict[str, object]] = []
            for member_index, (base_hash, scaled_hash) in enumerate(
                zip(base_hashes, scaled_hashes, strict=True)
            ):
                if not isinstance(base_hash, str) or not isinstance(scaled_hash, str):
                    raise ValueError("initial-noise hash is not a string")
                if len(base_hash) != 64 or len(scaled_hash) != 64 or base_hash == scaled_hash:
                    raise ValueError("initial-noise hashes do not prove the frozen scaling")
                actual_seed = noise_seed(case_index, member_index)
                members.append(
                    {
                        "member_id": f"case{case_index:02d}-ema_last-z{actual_seed}",
                        "checkpoint_hash": CHECKPOINT_SHA256,
                        "noise_seed": actual_seed,
                        "initial_noise_hash": base_hash,
                        "scaled_initial_noise_hash": scaled_hash,
                        "finite": True,
                    }
                )
            sample_hash = _file_hash(path)
            output_paths.append(path)
            manifest_cases.append({"case_index": case_index, "members": members})
            per_case.append(
                {
                    "case_index": case_index,
                    "target_date": expected_date.isoformat(),
                    "ensemble_size": MEMBER_COUNT,
                    "initial_noise_scale": INITIAL_NOISE_SCALE,
                    "initial_noise_manifest_sha256": _canonical_hash(members),
                    "sample_sha256": sample_hash,
                }
            )
        if len(list(samples_dir.glob("*.npz"))) != CASE_COUNT:
            raise ValueError("sample directory contains files outside the frozen schedule")

        admission_manifest = {
            "schema_version": 1,
            "source_experiment": SOURCE_EXPERIMENT,
            "source_config_sha256": SOURCE_CONFIG_SHA256,
            "source_metadata_sha256": SOURCE_METADATA_SHA256,
            "source_metrics_sha256": SOURCE_METRICS_SHA256,
            "checkpoint_record": checkpoint_record,
            "initial_noise_scale": INITIAL_NOISE_SCALE,
            "base_noise_seed": BASE_NOISE_SEED,
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
            "checkpoint_record": checkpoint_record,
            "initial_noise_scale": INITIAL_NOISE_SCALE,
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
            "scientific_role": "frozen upstream variance calibration without output-space postprocessing",
        }
        _atomic_json(output_dir / "metadata.json", metadata)
        _atomic_json(
            output_dir / "aggregate_case_mean_metrics.json",
            {
                "schema_version": 1,
                "status": "sampling_completed_gate_pending",
                "num_cases": CASE_COUNT,
                "ensemble_size": MEMBER_COUNT,
                "initial_noise_scale": INITIAL_NOISE_SCALE,
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
