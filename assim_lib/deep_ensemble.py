"""Frozen server-side clean-checkpoint deep-ensemble orchestrator."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np

CHECKPOINT_SEEDS = (1701, 1702, 1703)
LATENT_SEEDS = (2401, 2402, 2403, 2404)
SOURCE_EXPERIMENT = "joint_full_condition_validation_2022"
MODE = "validation_clean_checkpoint_deep_ensemble_sampling"
CASE_COUNT = 40
MEMBER_COUNT = 10
EXPECTED_DATES = tuple(date(2022, 1, 1) + timedelta(days=5 * i) for i in range(CASE_COUNT))
COMPACT_OUTPUTS = {
    "run_status.json",
    "metadata.json",
    "aggregate_case_mean_metrics.json",
    "per_case_metrics.csv",
}

_active_process: subprocess.Popen[bytes] | None = None


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("per-case accounting cannot be empty")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def case_plan(case_index: int) -> tuple[tuple[int, int], ...]:
    if not 0 <= case_index < CASE_COUNT:
        raise ValueError("case index outside frozen schedule")
    extra = CHECKPOINT_SEEDS[case_index % len(CHECKPOINT_SEEDS)]
    plan = tuple(
        (checkpoint_seed, latent_seed)
        for checkpoint_seed in CHECKPOINT_SEEDS
        for latent_seed in (
            LATENT_SEEDS if checkpoint_seed == extra else LATENT_SEEDS[:3]
        )
    )
    if len(plan) != MEMBER_COUNT or len(set(plan)) != MEMBER_COUNT:
        raise AssertionError("invalid frozen member plan")
    return plan


def _write_status(output_dir: Path, stage: str, completed: int, **extra: object) -> None:
    _atomic_json(
        output_dir / "run_status.json",
        {
            "status": "running",
            "experiment_id": output_dir.parent.name,
            "stage": stage,
            "completed_units": completed,
            "total_units": 3 + 3 + CASE_COUNT,
            "updated_epoch_seconds": time.time(),
            **extra,
        },
    )


def _forward_signal(signum: int, _frame: object) -> None:
    process = _active_process
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
    raise SystemExit(128 + signum)


def _run_checked(command: list[str], cwd: Path, output_dir: Path, stage: str, completed: int) -> None:
    global _active_process
    _active_process = subprocess.Popen(command, cwd=cwd, stdin=subprocess.DEVNULL)
    try:
        while _active_process.poll() is None:
            _write_status(output_dir, stage, completed, child_pid=_active_process.pid)
            time.sleep(30)
        if _active_process.returncode:
            raise RuntimeError(f"trusted subprocess failed in {stage}: rc={_active_process.returncode}")
    finally:
        _active_process = None


def _checkpoint_is_finite(path: Path) -> bool:
    import torch

    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    tensors: list[Any] = []

    def collect(value: Any) -> None:
        if torch.is_tensor(value):
            tensors.append(value)
        elif isinstance(value, dict):
            for item in value.values():
                collect(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                collect(item)

    collect(payload)
    return bool(tensors) and all(bool(torch.isfinite(item).all()) for item in tensors)


def _training_configuration(repo: Path, seed: int) -> dict[str, Any]:
    base_path = repo / "config/experiments/train_clean_deep_ensemble.json"
    with base_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    config["data_config"] = str((repo / "config/data/m2m_2f_1y.json").resolve())
    config["model_config"] = str(
        (repo / "config/methods/concat_conditioning_diffusion_balanced_2f.json").resolve()
    )
    config["training"] = {**config["training"], "seed": seed}
    return config


def _validate_training(run_dir: Path, seed: int) -> dict[str, Any]:
    metrics_path = run_dir / "metrics.json"
    metadata_path = run_dir / "metadata.json"
    checkpoint = run_dir / "ema_best_model.pth"
    if not all(path.is_file() for path in (metrics_path, metadata_path, checkpoint)):
        raise ValueError(f"training seed {seed} did not produce the frozen artifacts")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metrics, list) or len(metrics) != 40:
        raise ValueError(f"training seed {seed} did not complete exactly 40 epochs")
    losses = [float(row["val_loss"]) for row in metrics]
    if not all(math.isfinite(value) for value in losses):
        raise ValueError(f"training seed {seed} contains non-finite validation loss")
    selected_epoch = min(range(len(losses)), key=losses.__getitem__)
    training = dict(metadata.get("training_config") or {})
    if training.get("seed") != seed or training.get("num_epochs") != 40:
        raise ValueError("effective training configuration differs from the frozen contract")
    if training.get("sample_every_n_epochs") != 0 or training.get("metric_every_n_epochs") != 0:
        raise ValueError("expensive in-training sampling was not disabled")
    data_config = metadata.get("data_config") or {}
    if data_config.get("split_protocol") != "3dvar_main_200d":
        raise ValueError("checkpoint was not trained on the corrected disjoint split")
    if not _checkpoint_is_finite(checkpoint):
        raise ValueError(f"selected checkpoint for seed {seed} is non-finite")
    non_seed_training = {key: value for key, value in training.items() if key != "seed"}
    non_seed_hash = _canonical_hash(
        {
            "training": non_seed_training,
            "data_config": data_config,
            "model_config": metadata.get("model_config"),
        }
    )
    return {
        "training_seed": seed,
        "non_seed_configuration_hash": non_seed_hash,
        "selected_checkpoint_hash": _file_hash(checkpoint),
        "training_completed_normally": True,
        "selected_checkpoint_finite": True,
        "selection_count": 1,
        "selected_epoch_zero_based": selected_epoch,
        "selected_validation_loss": losses[selected_epoch],
        "run_dir": str(run_dir),
    }


def _sample_manifest(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(_file_hash(path).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def run(output_dir: Path, source_experiment: str) -> None:
    if source_experiment != SOURCE_EXPERIMENT:
        raise ValueError("source_experiment differs from the frozen contract")
    output_dir = output_dir.resolve()
    if output_dir.is_symlink():
        raise ValueError("output_dir cannot be a symlink")
    output_dir.mkdir(parents=True, exist_ok=True)
    unexpected = {path.name for path in output_dir.iterdir()} - COMPACT_OUTPUTS - {"internal", "samples"}
    if unexpected:
        raise ValueError(f"output_dir contains unexpected entries: {sorted(unexpected)}")
    repo = Path(__file__).resolve().parents[1]
    internal = output_dir / "internal"
    internal.mkdir(exist_ok=True)
    training_runs: list[dict[str, Any]] = []

    try:
        for order, seed in enumerate(CHECKPOINT_SEEDS):
            seed_root = internal / f"seed_{seed}"
            seed_root.mkdir(parents=True, exist_ok=True)
            config_path = seed_root / "training_config.json"
            _atomic_json(config_path, _training_configuration(repo, seed))
            _run_checked(
                [sys.executable, "-m", "assim_lib.main", "--config", str(config_path)],
                seed_root,
                output_dir,
                f"training_seed_{seed}",
                order,
            )
            run_dir = seed_root / "training/clean_checkpoint"
            training_runs.append(_validate_training(run_dir, seed))
            _write_status(output_dir, f"training_seed_{seed}_complete", order + 1)

        non_seed_hashes = {row["non_seed_configuration_hash"] for row in training_runs}
        checkpoint_hashes = {row["selected_checkpoint_hash"] for row in training_runs}
        if len(non_seed_hashes) != 1 or len(checkpoint_hashes) != len(CHECKPOINT_SEEDS):
            raise ValueError("training configuration or checkpoint identity admission failed")

        sampling_roots: dict[int, Path] = {}
        for order, (seed, training_record) in enumerate(zip(CHECKPOINT_SEEDS, training_runs, strict=True)):
            sample_root = internal / f"samples_seed_{seed}"
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
                    str(training_record["run_dir"]),
                    "--checkpoint-name",
                    "ema_best_model.pth",
                    "--output-dir",
                    str(sample_root),
                    "--save-ensembles",
                ],
                repo,
                output_dir,
                f"sampling_seed_{seed}",
                len(CHECKPOINT_SEEDS) + order,
            )
            sampling_roots[seed] = sample_root

        source_cases = {
            seed: json.loads((root / "cases.json").read_text(encoding="utf-8"))
            for seed, root in sampling_roots.items()
        }
        samples_dir = output_dir / "samples"
        samples_dir.mkdir(exist_ok=True)
        output_paths: list[Path] = []
        manifest_cases: list[dict[str, Any]] = []
        per_case: list[dict[str, object]] = []
        for case_index, expected_date in enumerate(EXPECTED_DATES):
            filename = f"{case_index:03d}_{expected_date.isoformat()}_h23.npz"
            payloads: dict[int, dict[str, np.ndarray]] = {}
            for seed, root in sampling_roots.items():
                path = root / "samples" / filename
                with np.load(path, allow_pickle=False) as payload:
                    payloads[seed] = {key: payload[key] for key in payload.files}
            reference = payloads[CHECKPOINT_SEEDS[0]]
            for seed in CHECKPOINT_SEEDS[1:]:
                for key in (
                    "truth",
                    "background",
                    "obs_values",
                    "conditioning_mask",
                    "track_imitation_mask",
                    "valid_mask",
                ):
                    if not np.array_equal(reference[key], payloads[seed][key], equal_nan=True):
                        raise ValueError(f"conditioning context differs for case {case_index}")
            members: list[np.ndarray] = []
            member_records: list[dict[str, object]] = []
            for checkpoint_seed, latent_seed in case_plan(case_index):
                latent_index = LATENT_SEEDS.index(latent_seed)
                member = np.asarray(
                    payloads[checkpoint_seed]["analysis_ensemble"][latent_index],
                    dtype=np.float32,
                )
                if not np.all(np.isfinite(member)):
                    raise ValueError(f"non-finite generated member for case {case_index}")
                members.append(member)
                noise_hashes = source_cases[checkpoint_seed][case_index]["initial_noise_sha256"]
                noise_hash = str(noise_hashes[latent_index])
                member_records.append(
                    {
                        "member_id": f"case{case_index:02d}-ckpt{checkpoint_seed}-z{latent_seed}",
                        "checkpoint_seed": checkpoint_seed,
                        "latent_seed": latent_seed,
                        "checkpoint_hash": next(
                            row["selected_checkpoint_hash"]
                            for row in training_runs
                            if row["training_seed"] == checkpoint_seed
                        ),
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
                    "extra_checkpoint_seed": CHECKPOINT_SEEDS[case_index % 3],
                    "finite_members": MEMBER_COUNT,
                    "sample_sha256": _file_hash(target),
                }
            )
            _write_status(output_dir, "assembling", 6 + case_index + 1)

        admission_manifest = {
            "schema_version": 1,
            "source_experiment": SOURCE_EXPERIMENT,
            "non_seed_configuration_hash": next(iter(non_seed_hashes)),
            "training_runs": [
                {
                    key: row[key]
                    for key in (
                        "training_seed",
                        "non_seed_configuration_hash",
                        "selected_checkpoint_hash",
                        "training_completed_normally",
                        "selected_checkpoint_finite",
                        "selection_count",
                    )
                }
                for row in training_runs
            ],
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
            "checkpoint_seeds": list(CHECKPOINT_SEEDS),
            "latent_seeds": list(LATENT_SEEDS),
            "extra_checkpoint_counts": {"1701": 14, "1702": 13, "1703": 13},
            "conditioning_mode": "full",
            "sampler": {"method": "dopri5", "num_timesteps": 25, "rtol": 1e-5, "atol": 1e-6},
            "training_runs": training_runs,
            "admission_manifest": admission_manifest,
            "admission_manifest_sha256": _canonical_hash(admission_manifest),
            "sample_manifest_sha256": _sample_manifest(output_paths),
            "fallback_member_count": 0,
            "raw_arrays_retrieved": False,
            "test_data_used": False,
            "gate_pending": True,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-experiment", required=True)
    return parser.parse_args()


def main() -> int:
    signal.signal(signal.SIGTERM, _forward_signal)
    signal.signal(signal.SIGINT, _forward_signal)
    args = parse_args()
    run(args.output_dir, args.source_experiment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
