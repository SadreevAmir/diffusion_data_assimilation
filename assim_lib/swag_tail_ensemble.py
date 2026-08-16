"""Frozen training, SWAG construction, and sampling orchestration.

All large weights, matrices, and arrays remain beneath ``internal`` or
``samples`` on the server.  The top-level reporting contract is exactly four
compact files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import signal
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import compare_3dvar
from .deep_ensemble import (
    COMPACT_OUTPUTS,
    _atomic_csv,
    _atomic_json,
    _canonical_hash,
    _file_hash,
    _forward_signal,
    _run_checked,
    _sample_manifest,
)
from .evaluate import generate_ensemble as _base_generate_ensemble
from .swag_tail_training import EPOCHS, SNAPSHOT_EPOCHS, TRAINING_SEED

MODE = "validation_swag_tail_weight_posterior_sampling"
CANDIDATE_METHOD = "swag_tail_weight_posterior_ensemble"
SOURCE_EXPERIMENT = "joint_full_condition_validation_2022"
CASE_COUNT = 40
MEMBER_COUNT = 10
BASE_NOISE_SEED = 1234
WEIGHT_SAMPLE_SEEDS = tuple(170_100 + index for index in range(MEMBER_COUNT))
CONFIG_NAME = "config/experiments/train_swag_tail_weight_posterior.json"
EXPECTED_DATES = tuple(date(2022, 1, 1) + timedelta(days=5 * index) for index in range(CASE_COUNT))
WORKER_MEMBER_ENV = "SWAG_MEMBER_INDEX"
WORKER_MANIFEST_ENV = "SWAG_MEMBER_MANIFEST"
MATRIX_CHUNK = 262_144
DEPENDENT_GATE_MODE = "validation_swag_tail_weight_posterior_gate"

_active_member_index: int | None = None
_worker_cases: list[dict[str, Any]] = []


def noise_seed(case_index: int, member_index: int) -> int:
    if not 0 <= case_index < CASE_COUNT or not 0 <= member_index < MEMBER_COUNT:
        raise ValueError("case or member index outside frozen schedule")
    return BASE_NOISE_SEED + 10 * case_index + member_index


def _tensor_hash(value: torch.Tensor) -> str:
    canonical = value.detach().cpu().to(dtype=torch.float32).contiguous().numpy()
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _atomic_torch_save(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _load_state(path: Path) -> dict[str, torch.Tensor]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or not payload:
        raise ValueError("snapshot is not a non-empty state dict")
    state: dict[str, torch.Tensor] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not torch.is_tensor(value):
            raise ValueError("snapshot state schema differs")
        if not value.is_floating_point():
            raise ValueError("frozen UNet snapshot unexpectedly contains a non-floating tensor")
        canonical = value.detach().cpu().to(dtype=torch.float32).contiguous()
        if not bool(torch.isfinite(canonical).all()):
            raise ValueError("snapshot contains non-finite weights")
        state[key] = canonical
    return state


def build_snapshot_matrix(
    snapshot_paths: list[Path], matrix_path: Path
) -> tuple[list[dict[str, Any]], int, str]:
    """Materialize K flattened snapshots as one server-side float32 memmap."""
    if len(snapshot_paths) != len(SNAPSHOT_EPOCHS):
        raise ValueError("exactly ten SWAG snapshots are required")
    if matrix_path.exists() or matrix_path.is_symlink():
        raise ValueError("SWAG matrix target must be new")
    first = _load_state(snapshot_paths[0])
    inventory: list[dict[str, Any]] = []
    offset = 0
    for name, tensor in first.items():
        count = int(tensor.numel())
        inventory.append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "numel": count,
                "offset": offset,
            }
        )
        offset += count
    if offset <= 0:
        raise ValueError("SWAG parameter inventory is empty")
    matrix = np.memmap(matrix_path, mode="w+", dtype="<f4", shape=(len(snapshot_paths), offset))
    for snapshot_index, path in enumerate(snapshot_paths):
        state = first if snapshot_index == 0 else _load_state(path)
        if list(state) != [row["name"] for row in inventory]:
            raise ValueError("snapshot parameter order differs")
        for row in inventory:
            tensor = state[str(row["name"])]
            if list(tensor.shape) != row["shape"]:
                raise ValueError("snapshot parameter shape differs")
            start = int(row["offset"])
            stop = start + int(row["numel"])
            matrix[snapshot_index, start:stop] = tensor.numpy().reshape(-1)
        matrix.flush()
        if snapshot_index == 0:
            first = {}
    del matrix
    return inventory, offset, _file_hash(matrix_path)


def sample_swag_checkpoints(
    matrix_path: Path,
    inventory: list[dict[str, Any]],
    parameter_count: int,
    output_dir: Path,
    *,
    chunk_size: int = MATRIX_CHUNK,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Draw ten iid standard-SWAG weights using bounded-memory chunks."""
    if chunk_size <= 0 or parameter_count <= 0:
        raise ValueError("invalid SWAG chunk or parameter count")
    output_dir.mkdir(parents=True, exist_ok=False)
    snapshot_count = len(SNAPSHOT_EPOCHS)
    matrix = np.memmap(matrix_path, mode="r", dtype="<f4", shape=(snapshot_count, parameter_count))
    sample_paths = [output_dir / f"swag_member_{index:02d}.f32" for index in range(MEMBER_COUNT)]
    samples = [np.memmap(path, mode="w+", dtype="<f4", shape=(parameter_count,)) for path in sample_paths]
    generators: list[torch.Generator] = []
    coefficients: list[np.ndarray] = []
    coefficient_hashes: list[str] = []
    epsilon_hashes = [hashlib.sha256() for _ in range(MEMBER_COUNT)]
    for seed in WEIGHT_SAMPLE_SEEDS:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        coefficient = torch.randn(snapshot_count, dtype=torch.float32, generator=generator)
        generators.append(generator)
        coefficients.append(coefficient.numpy().astype(np.float64))
        coefficient_hashes.append(_tensor_hash(coefficient))
    mean_hash = hashlib.sha256()
    variance_hash = hashlib.sha256()
    positive_variance = 0
    for start in range(0, parameter_count, chunk_size):
        stop = min(parameter_count, start + chunk_size)
        block = np.asarray(matrix[:, start:stop], dtype=np.float64)
        mean = block.mean(axis=0, dtype=np.float64)
        variance = np.maximum(np.mean(block * block, axis=0) - mean * mean, 0.0)
        positive_variance += int(np.count_nonzero(variance > 0.0))
        mean32 = mean.astype("<f4")
        variance32 = variance.astype("<f4")
        mean_hash.update(mean32.tobytes())
        variance_hash.update(variance32.tobytes())
        deviations = block - mean[None, :]
        for member_index in range(MEMBER_COUNT):
            epsilon = torch.randn(
                stop - start, dtype=torch.float32, generator=generators[member_index]
            ).numpy()
            epsilon_hashes[member_index].update(epsilon.astype("<f4", copy=False).tobytes())
            diagonal = math.sqrt(0.5) * np.sqrt(variance) * epsilon.astype(np.float64)
            low_rank = (deviations.T @ coefficients[member_index]) / math.sqrt(2.0 * (snapshot_count - 1))
            sampled = mean + diagonal + low_rank
            if not np.all(np.isfinite(sampled)):
                raise ValueError("SWAG law produced non-finite sampled weights")
            samples[member_index][start:stop] = sampled.astype("<f4")
    if positive_variance <= 0:
        raise ValueError("SWAG tail covariance collapsed to zero")
    for sample in samples:
        sample.flush()
    del matrix

    records: list[dict[str, Any]] = []
    for member_index, (vector_path, vector) in enumerate(zip(sample_paths, samples, strict=True)):
        state: dict[str, torch.Tensor] = {}
        for row in inventory:
            start = int(row["offset"])
            stop = start + int(row["numel"])
            array = np.array(vector[start:stop], dtype=np.float32, copy=True).reshape(row["shape"])
            state[str(row["name"])] = torch.from_numpy(array)
        checkpoint = output_dir / f"swag_member_{member_index:02d}.pth"
        _atomic_torch_save(checkpoint, state)
        records.append(
            {
                "member_index": member_index,
                "weight_seed": WEIGHT_SAMPLE_SEEDS[member_index],
                "low_rank_coefficient_sha256": coefficient_hashes[member_index],
                "diagonal_epsilon_sha256": epsilon_hashes[member_index].hexdigest(),
                "sampled_weight_vector_sha256": _file_hash(vector_path),
                "sampled_checkpoint_name": checkpoint.name,
                "sampled_checkpoint_sha256": _file_hash(checkpoint),
                "finite": True,
            }
        )
    del samples
    statistics = {
        "schema_version": 1,
        "snapshot_count": snapshot_count,
        "parameter_count": parameter_count,
        "diagonal_mean_sha256": mean_hash.hexdigest(),
        "diagonal_variance_sha256": variance_hash.hexdigest(),
        "positive_diagonal_variance_count": positive_variance,
        "covariance_formula": (
            "theta_bar + sqrt(1/2)*sqrt(diag_variance)*epsilon + deviations*a/sqrt(2*(K-1))"
        ),
        "iid_weight_draws": True,
    }
    return records, statistics


def _worker_generate_ensemble(*args, **kwargs):
    if args:
        raise ValueError("trusted SWAG worker requires keyword sampling arguments")
    member_index = _active_member_index
    if member_index is None:
        raise RuntimeError("SWAG worker member index is unset")
    case_index = int(kwargs["case_order"])
    if (
        int(kwargs["ensemble_size"]) != 1
        or int(kwargs["sample_batch_size"]) != 1
        or int(kwargs["num_timesteps"]) != 25
        or str(kwargs["method"]) != "dopri5"
        or float(kwargs["rtol"]) != 1e-5
        or float(kwargs["atol"]) != 1e-6
        or float(kwargs["initial_noise_scale"]) != 1.0
        or kwargs.get("autocast_dtype") is not None
    ):
        raise ValueError("SWAG worker sampling arguments differ from the frozen contract")
    expected_seed = noise_seed(case_index, member_index)
    kwargs["seed"] = expected_seed - case_index
    before_base = len(kwargs["initial_noise_hashes"])
    before_scaled = len(kwargs["scaled_initial_noise_hashes"])
    result = _base_generate_ensemble(**kwargs)
    base_hashes = kwargs["initial_noise_hashes"][before_base:]
    scaled_hashes = kwargs["scaled_initial_noise_hashes"][before_scaled:]
    if len(base_hashes) != 1 or base_hashes != scaled_hashes:
        raise ValueError("SWAG worker latent hash accounting differs")
    _worker_cases.append(
        {
            "case_index": case_index,
            "member_index": member_index,
            "latent_seed": expected_seed,
            "initial_noise_sha256": base_hashes[0],
        }
    )
    return result


def _worker_main(member_index: int) -> int:
    global _active_member_index
    if not 0 <= member_index < MEMBER_COUNT:
        raise ValueError("SWAG worker member index outside frozen schedule")
    manifest_value = os.environ.get(WORKER_MANIFEST_ENV)
    if not manifest_value:
        raise ValueError("SWAG worker manifest path is missing")
    manifest_path = Path(manifest_value)
    if manifest_path.is_symlink():
        raise ValueError("SWAG worker manifest path cannot be a symlink")
    _active_member_index = member_index
    _worker_cases.clear()
    original = compare_3dvar.generate_ensemble
    compare_3dvar.generate_ensemble = _worker_generate_ensemble
    try:
        worker_args = compare_3dvar.parse_args()
        compare_3dvar.run(worker_args)
        if [row["case_index"] for row in _worker_cases] != list(range(CASE_COUNT)):
            raise ValueError("SWAG worker did not complete the forty-case schedule")
        sample_root = Path(worker_args.output_dir).resolve() / "samples"
        for case_index, expected_date in enumerate(EXPECTED_DATES):
            path = sample_root / f"{case_index:03d}_{expected_date.isoformat()}_h23.npz"
            if path.is_symlink() or not path.is_file():
                raise ValueError("SWAG worker sample is missing or unsafe")
            with np.load(path, allow_pickle=False) as payload:
                sample = np.asarray(payload["analysis_ensemble"])
                if sample.ndim != 3 or sample.shape[0] != 1 or not np.all(np.isfinite(sample)):
                    raise ValueError("SWAG worker sample shape or finiteness differs")
            _worker_cases[case_index]["sample_sha256"] = _file_hash(path)
        _atomic_json(
            manifest_path.resolve(),
            {
                "schema_version": 1,
                "member_index": member_index,
                "weight_seed": WEIGHT_SAMPLE_SEEDS[member_index],
                "cases": list(_worker_cases),
            },
        )
    finally:
        compare_3dvar.generate_ensemble = original
        _active_member_index = None
    return 0


def _validate_training(training_dir: Path) -> tuple[dict[str, Any], list[Path]]:
    manifest_path = training_dir / "swag_training_manifest.json"
    metadata_path = training_dir / "metadata.json"
    metrics_path = training_dir / "metrics.json"
    required = (manifest_path, metadata_path, metrics_path)
    if not all(path.is_file() and not path.is_symlink() for path in required):
        raise ValueError("SWAG training artifacts are incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("training_seed") != TRAINING_SEED
        or manifest.get("training_completed_normally") is not True
        or manifest.get("epochs_completed") != EPOCHS
        or manifest.get("validation_selected") is not False
        or manifest.get("test_data_used") is not False
        or manifest.get("weight_state") != "raw non-EMA end-of-epoch model"
        or manifest.get("resume_contract_sha256") is None
    ):
        raise ValueError("SWAG training manifest differs")
    repo = Path(__file__).resolve().parents[1]
    anchor_paths = {
        "contract_config_sha256": repo / CONFIG_NAME,
        "data_config_sha256": repo / "config/data/m2m_2f_1y.json",
        "model_config_sha256": (repo / "config/methods/concat_conditioning_diffusion_balanced_2f.json"),
        "training_module_sha256": repo / "assim_lib/swag_tail_training.py",
        "training_entrypoint_sha256": repo / "assim_lib/main.py",
        "base_trainer_sha256": repo / "assim_lib/trainer.py",
    }
    source_anchors = manifest.get("source_anchors")
    if not isinstance(source_anchors, dict):
        raise ValueError("SWAG training source anchors are missing")
    for name, path in anchor_paths.items():
        if path.is_symlink() or not path.is_file() or source_anchors.get(name) != _file_hash(path):
            raise ValueError(f"SWAG training source anchor differs for {name}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if (
        not isinstance(metrics, list)
        or [row.get("epoch") for row in metrics] != list(range(EPOCHS))
        or any(
            not math.isfinite(float(row.get(name, float("nan"))))
            for row in metrics
            for name in ("val_loss", "val_loss_full", "val_loss_obs")
        )
    ):
        raise ValueError("SWAG training metrics epoch accounting differs")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    training = metadata.get("training_config") or {}
    data = metadata.get("data_config") or {}
    if (
        training.get("seed") != TRAINING_SEED
        or training.get("num_epochs") != EPOCHS
        or training.get("num_workers_train") != 0
        or training.get("num_workers_val") != 0
        or training.get("sample_every_n_epochs") != 0
        or training.get("metric_every_n_epochs") != 0
    ):
        raise ValueError("effective SWAG training configuration differs")
    if (
        data.get("split_protocol") != "3dvar_main_200d"
        or data.get("train", {}).get("obs_end_day") != "2021-12-31"
        or data.get("valid", {}).get("obs_start_day") != "2022-01-01"
        or data.get("valid", {}).get("obs_end_day") != "2022-12-31"
    ):
        raise ValueError("SWAG train/validation temporal protocol differs")
    records = manifest.get("snapshots")
    if not isinstance(records, list) or len(records) != len(SNAPSHOT_EPOCHS):
        raise ValueError("SWAG training did not produce ten snapshots")
    paths: list[Path] = []
    scheduler = manifest.get("scheduler")
    if not isinstance(scheduler, dict):
        raise ValueError("SWAG scheduler provenance is missing")
    steps_per_epoch = scheduler.get("steps_per_epoch")
    if (
        type(steps_per_epoch) is not int
        or steps_per_epoch <= 0
        or scheduler.get("total_steps") != EPOCHS * steps_per_epoch
        or scheduler.get("tail_start_step") != 30 * steps_per_epoch
        or scheduler.get("warmup_steps") != 500
        or scheduler.get("optimizer")
        != {
            "class": "AdamW",
            "learning_rate": 1e-4,
            "betas": [0.9, 0.999],
            "epsilon": 1e-8,
            "weight_decay": 0.01,
            "amsgrad": False,
        }
        or manifest.get("optimizer") != scheduler.get("optimizer")
        or manifest.get("optimizer_state_retained_through_tail") is not True
    ):
        raise ValueError("SWAG frozen scheduler accounting differs")
    tail_lr = float(scheduler.get("tail_learning_rate", float("nan")))
    snapshot_dir = (training_dir / "swag_snapshots").resolve()
    for epoch, record in zip(SNAPSHOT_EPOCHS, records, strict=True):
        path = Path(str(record.get("path", "")))
        if (
            record.get("epoch_zero_based") != epoch
            or path.parent.resolve() != snapshot_dir
            or path.name != f"raw_epoch_{epoch:02d}.pth"
            or not path.is_file()
            or path.is_symlink()
            or _file_hash(path) != record.get("sha256")
            or not math.isclose(
                float(record.get("learning_rate", float("nan"))),
                tail_lr,
                rel_tol=1e-6,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("SWAG snapshot provenance differs")
        paths.append(path)
    return manifest, paths


def _validate_member_sampling(root: Path, member_index: int) -> tuple[list[Path], list[dict[str, Any]]]:
    metadata_path = root / "metadata.json"
    manifest_path = root / "swag_member_manifest.json"
    samples_root = root / "samples"
    if (
        metadata_path.is_symlink()
        or not metadata_path.is_file()
        or manifest_path.is_symlink()
        or not manifest_path.is_file()
        or samples_root.is_symlink()
        or not samples_root.is_dir()
    ):
        raise ValueError("SWAG member sampling paths are incomplete or unsafe")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {
        "dataset_split": "valid",
        "start_date": "2022-01-01",
        "end_date": "2022-07-15",
        "num_cases": CASE_COUNT,
        "case_stride_days": 5,
        "ensemble_size": 1,
        "sample_batch_size": 1,
        "num_timesteps": 25,
        "method": "dopri5",
        "rtol": 1e-5,
        "atol": 1e-6,
        "inference_precision": "float32",
        "conditioning_mode": "full",
        "cfg_mode": "none",
        "initial_noise_scale": 1.0,
        "save_ensembles": True,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"member sampling metadata differs for {key}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("member_index") != member_index
        or manifest.get("weight_seed") != WEIGHT_SAMPLE_SEEDS[member_index]
    ):
        raise ValueError("SWAG member sampling manifest identity differs")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or len(cases) != CASE_COUNT:
        raise ValueError("SWAG member sampling manifest is incomplete")
    paths: list[Path] = []
    for case_index, expected_date in enumerate(EXPECTED_DATES):
        row = cases[case_index]
        if (
            row.get("case_index") != case_index
            or row.get("member_index") != member_index
            or row.get("latent_seed") != noise_seed(case_index, member_index)
            or len(str(row.get("initial_noise_sha256", ""))) != 64
            or len(str(row.get("sample_sha256", ""))) != 64
        ):
            raise ValueError("SWAG latent schedule differs")
        path = samples_root / f"{case_index:03d}_{expected_date.isoformat()}_h23.npz"
        if not path.is_file() or path.is_symlink() or _file_hash(path) != row.get("sample_sha256"):
            raise ValueError("SWAG member sample is missing")
        paths.append(path)
    if len(list(samples_root.glob("*.npz"))) != CASE_COUNT:
        raise ValueError("SWAG member sample inventory differs")
    return paths, cases


def _validate_weight_stage(
    marker_path: Path, matrix_path: Path, checkpoint_dir: Path
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], int, str]:
    if marker_path.is_symlink() or not marker_path.is_file():
        raise ValueError("completed SWAG weight-stage marker has unsafe type")
    payload = json.loads(marker_path.read_text(encoding="utf-8"))
    inventory = payload.get("parameter_inventory")
    records = payload.get("weight_records")
    statistics = payload.get("statistics")
    parameter_count = payload.get("parameter_count")
    matrix_sha256 = payload.get("snapshot_matrix_sha256")
    if (
        payload.get("schema_version") != 1
        or not isinstance(inventory, list)
        or not inventory
        or not isinstance(records, list)
        or len(records) != MEMBER_COUNT
        or not isinstance(statistics, dict)
        or type(parameter_count) is not int
        or parameter_count <= 0
        or not isinstance(matrix_sha256, str)
        or len(matrix_sha256) != 64
        or statistics.get("parameter_inventory_sha256") != _canonical_hash(inventory)
        or statistics.get("snapshot_matrix_sha256") != matrix_sha256
    ):
        raise ValueError("completed SWAG weight-stage contract differs")
    if matrix_path.is_symlink() or not matrix_path.is_file() or _file_hash(matrix_path) != matrix_sha256:
        raise ValueError("completed SWAG snapshot matrix differs")
    for member_index, record in enumerate(records):
        checkpoint = checkpoint_dir / f"swag_member_{member_index:02d}.pth"
        if (
            record.get("member_index") != member_index
            or record.get("weight_seed") != WEIGHT_SAMPLE_SEEDS[member_index]
            or record.get("sampled_checkpoint_name") != checkpoint.name
            or checkpoint.is_symlink()
            or not checkpoint.is_file()
            or _file_hash(checkpoint) != record.get("sampled_checkpoint_sha256")
            or len(str(record.get("sampled_weight_vector_sha256", ""))) != 64
            or record.get("finite") is not True
        ):
            raise ValueError("completed SWAG sampled-weight provenance differs")
    return records, statistics, inventory, parameter_count, matrix_sha256


def _validate_completed_output(output_dir: Path, metadata: dict[str, Any]) -> None:
    if (
        metadata.get("status") != "completed"
        or metadata.get("mode") != MODE
        or metadata.get("candidate_method") != CANDIDATE_METHOD
        or metadata.get("source_experiment") != SOURCE_EXPERIMENT
        or metadata.get("num_cases") != CASE_COUNT
        or metadata.get("ensemble_size") != MEMBER_COUNT
        or metadata.get("gate_pending") is not True
        or metadata.get("test_data_used_for_this_selection") is not False
    ):
        raise ValueError("completed SWAG metadata contract differs")
    admission = metadata.get("admission_manifest")
    if not isinstance(admission, dict) or _canonical_hash(admission) != metadata.get(
        "admission_manifest_sha256"
    ):
        raise ValueError("completed SWAG admission manifest differs")
    aggregate_path = output_dir / "aggregate_case_mean_metrics.json"
    per_case_path = output_dir / "per_case_metrics.csv"
    samples_dir = output_dir / "samples"
    if (
        aggregate_path.is_symlink()
        or not aggregate_path.is_file()
        or per_case_path.is_symlink()
        or not per_case_path.is_file()
        or samples_dir.is_symlink()
        or not samples_dir.is_dir()
    ):
        raise ValueError("completed SWAG output inventory is incomplete or unsafe")
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    if aggregate.get("status") != "sampling_completed_gate_pending" or aggregate.get(
        "admission_manifest_sha256"
    ) != metadata.get("admission_manifest_sha256"):
        raise ValueError("completed SWAG aggregate artifact differs")
    with per_case_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != CASE_COUNT or [int(row["case_index"]) for row in rows] != list(range(CASE_COUNT)):
        raise ValueError("completed SWAG per-case accounting differs")
    sample_paths = [
        samples_dir / f"{case_index:03d}_{expected_date.isoformat()}_h23.npz"
        for case_index, expected_date in enumerate(EXPECTED_DATES)
    ]
    if (
        any(path.is_symlink() or not path.is_file() for path in sample_paths)
        or len(list(samples_dir.glob("*.npz"))) != CASE_COUNT
        or _sample_manifest(sample_paths) != metadata.get("sample_manifest_sha256")
    ):
        raise ValueError("completed SWAG sample manifest differs")


def run(output_dir: Path, source_experiment: str) -> None:
    if source_experiment != SOURCE_EXPERIMENT:
        raise ValueError("source_experiment differs from the frozen contract")
    if output_dir.is_symlink():
        raise ValueError("output_dir cannot be a symlink")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    unexpected = (
        {path.name for path in output_dir.iterdir()}
        - COMPACT_OUTPUTS
        - {
            "internal",
            "samples",
        }
    )
    if unexpected:
        raise ValueError(f"SWAG output_dir contains unexpected entries: {sorted(unexpected)}")
    completed_metadata = output_dir / "metadata.json"
    if completed_metadata.exists():
        if completed_metadata.is_symlink() or not completed_metadata.is_file():
            raise ValueError("SWAG metadata has unsafe type")
        metadata_payload = json.loads(completed_metadata.read_text(encoding="utf-8"))
        if metadata_payload.get("status") == "completed":
            _validate_completed_output(output_dir, metadata_payload)
            _atomic_json(
                output_dir / "run_status.json",
                {
                    "status": "completed",
                    "experiment_id": output_dir.parent.name,
                    "completed_cases": CASE_COUNT,
                    "total_cases": CASE_COUNT,
                    "progress_percent": 100.0,
                    "gate_pending": True,
                    "dependent_gate_mode": DEPENDENT_GATE_MODE,
                },
            )
            return
        raise ValueError("preexisting SWAG metadata is not a valid completion marker")
    repo = Path(__file__).resolve().parents[1]
    contract_path = repo / CONFIG_NAME
    if not contract_path.is_file() or contract_path.is_symlink():
        raise ValueError("frozen SWAG config is unavailable")
    internal = output_dir / "internal"
    if internal.is_symlink() or (internal.exists() and not internal.is_dir()):
        raise ValueError("SWAG internal path has unsafe type")
    internal.mkdir(exist_ok=True)
    try:
        training_root = internal / "training_run"
        if training_root.is_symlink() or (training_root.exists() and not training_root.is_dir()):
            raise ValueError("SWAG training root has unsafe type")
        training_root.mkdir(exist_ok=True)
        _run_checked(
            [sys.executable, "-m", "assim_lib.swag_tail_training", "--config", str(contract_path)],
            training_root,
            output_dir,
            "training_swag_tail",
            0,
            total_units=1 + MEMBER_COUNT + CASE_COUNT,
        )
        training_dir = training_root / "training" / "swag_tail"
        training_manifest, snapshot_paths = _validate_training(training_dir)
        matrix_path = internal / "swag_snapshot_matrix.f32"
        checkpoint_dir = training_dir / "swag_sampled_checkpoints"
        weight_marker = internal / "swag_weight_manifest.json"
        if weight_marker.exists() or weight_marker.is_symlink():
            weight_records, statistics, inventory, parameter_count, matrix_sha256 = _validate_weight_stage(
                weight_marker, matrix_path, checkpoint_dir
            )
        else:
            if matrix_path.is_symlink() or checkpoint_dir.is_symlink():
                raise ValueError("incomplete SWAG weight-stage paths cannot be symlinks")
            if matrix_path.exists():
                if not matrix_path.is_file():
                    raise ValueError("incomplete SWAG matrix has unsafe type")
                matrix_path.unlink()
            if checkpoint_dir.exists():
                if not checkpoint_dir.is_dir():
                    raise ValueError("incomplete SWAG checkpoint path has unsafe type")
                shutil.rmtree(checkpoint_dir)
            inventory, parameter_count, matrix_sha256 = build_snapshot_matrix(snapshot_paths, matrix_path)
            weight_records, statistics = sample_swag_checkpoints(
                matrix_path, inventory, parameter_count, checkpoint_dir
            )
            statistics.update(
                {
                    "snapshot_matrix_sha256": matrix_sha256,
                    "parameter_inventory_sha256": _canonical_hash(inventory),
                }
            )
            _atomic_json(
                weight_marker,
                {
                    "schema_version": 1,
                    "parameter_count": parameter_count,
                    "parameter_inventory": inventory,
                    "snapshot_matrix_sha256": matrix_sha256,
                    "statistics": statistics,
                    "weight_records": weight_records,
                },
            )
            for member_index in range(MEMBER_COUNT):
                vector = checkpoint_dir / f"swag_member_{member_index:02d}.f32"
                if vector.is_symlink() or not vector.is_file():
                    raise ValueError("SWAG transient sampled-weight vector has unsafe type")
                vector.unlink()

        sampling_roots: list[Path] = []
        member_case_records: list[list[dict[str, Any]]] = []
        for member_index, weight_record in enumerate(weight_records):
            sample_root = internal / f"sampling_member_{member_index:02d}"
            manifest_path = sample_root / "swag_member_manifest.json"
            if manifest_path.exists() or manifest_path.is_symlink():
                _, cases = _validate_member_sampling(sample_root, member_index)
                sampling_roots.append(sample_root)
                member_case_records.append(cases)
                continue
            if sample_root.exists() or sample_root.is_symlink():
                if sample_root.is_symlink() or not sample_root.is_dir():
                    raise ValueError("incomplete SWAG sampling path has unsafe type")
                shutil.rmtree(sample_root)
            old_member = os.environ.get(WORKER_MEMBER_ENV)
            old_manifest = os.environ.get(WORKER_MANIFEST_ENV)
            os.environ[WORKER_MEMBER_ENV] = str(member_index)
            os.environ[WORKER_MANIFEST_ENV] = str(manifest_path)
            try:
                _run_checked(
                    [
                        sys.executable,
                        "-m",
                        "assim_lib.swag_tail_ensemble",
                        "--worker",
                        "--member-index",
                        str(member_index),
                        "--config",
                        str(repo / "config/experiments/experiment_m2m_flow_modes.json"),
                        "--mode",
                        "validation_clean_checkpoint_member_sampling",
                        "--run-dir",
                        str(training_dir),
                        "--checkpoint-name",
                        str(checkpoint_dir / weight_record["sampled_checkpoint_name"]),
                        "--output-dir",
                        str(sample_root),
                        "--ensemble-size",
                        "1",
                        "--sample-batch-size",
                        "1",
                        "--seed",
                        str(BASE_NOISE_SEED),
                        "--initial-noise-scale",
                        "1.0",
                        "--save-ensembles",
                    ],
                    repo,
                    output_dir,
                    f"sampling_swag_member_{member_index:02d}",
                    1 + member_index,
                    total_units=1 + MEMBER_COUNT + CASE_COUNT,
                )
            finally:
                if old_member is None:
                    os.environ.pop(WORKER_MEMBER_ENV, None)
                else:
                    os.environ[WORKER_MEMBER_ENV] = old_member
                if old_manifest is None:
                    os.environ.pop(WORKER_MANIFEST_ENV, None)
                else:
                    os.environ[WORKER_MANIFEST_ENV] = old_manifest
            _, cases = _validate_member_sampling(sample_root, member_index)
            sampling_roots.append(sample_root)
            member_case_records.append(cases)

        samples_dir = output_dir / "samples"
        if samples_dir.exists() or samples_dir.is_symlink():
            if samples_dir.is_symlink() or not samples_dir.is_dir():
                raise ValueError("incomplete SWAG sample path has unsafe type")
            shutil.rmtree(samples_dir)
        samples_dir.mkdir()
        output_paths: list[Path] = []
        admission_cases: list[dict[str, Any]] = []
        per_case: list[dict[str, object]] = []
        context_keys = (
            "truth",
            "background",
            "obs_values",
            "conditioning_mask",
            "track_imitation_mask",
            "valid_mask",
        )
        for case_index, expected_date in enumerate(EXPECTED_DATES):
            filename = f"{case_index:03d}_{expected_date.isoformat()}_h23.npz"
            payloads: list[dict[str, np.ndarray]] = []
            for root in sampling_roots:
                with np.load(root / "samples" / filename, allow_pickle=False) as payload:
                    payloads.append({key: payload[key] for key in payload.files})
            reference = payloads[0]
            members: list[np.ndarray] = []
            member_records: list[dict[str, Any]] = []
            for member_index, payload in enumerate(payloads):
                if any(
                    not np.array_equal(reference[key], payload[key], equal_nan=True) for key in context_keys
                ):
                    raise ValueError("SWAG member conditioning contexts differ")
                ensemble = np.asarray(payload["analysis_ensemble"], dtype=np.float32)
                if ensemble.shape[0] != 1 or not np.all(np.isfinite(ensemble)):
                    raise ValueError("SWAG member sample is non-finite or has wrong shape")
                members.append(ensemble[0])
                member_records.append(
                    {
                        "member_id": f"swag-weight-{member_index:02d}",
                        "member_index": member_index,
                        "weight_seed": WEIGHT_SAMPLE_SEEDS[member_index],
                        "sampled_checkpoint_sha256": weight_records[member_index][
                            "sampled_checkpoint_sha256"
                        ],
                        "latent_seed": noise_seed(case_index, member_index),
                        "initial_noise_sha256": member_case_records[member_index][case_index][
                            "initial_noise_sha256"
                        ],
                        "finite": True,
                    }
                )
            candidate = np.stack(members)
            target = samples_dir / filename
            np.savez_compressed(
                target,
                analysis_ensemble=candidate,
                analysis_mean=candidate.mean(axis=0, dtype=np.float64).astype(np.float32),
                **{key: reference[key] for key in context_keys},
            )
            output_paths.append(target)
            admission_cases.append({"case_index": case_index, "members": member_records})
            per_case.append(
                {
                    "case_index": case_index,
                    "target_date": expected_date.isoformat(),
                    "ensemble_size": MEMBER_COUNT,
                    "finite_members": MEMBER_COUNT,
                    "member_manifest_sha256": _canonical_hash(member_records),
                    "sample_sha256": _file_hash(target),
                }
            )
        if len(list(samples_dir.glob("*.npz"))) != CASE_COUNT:
            raise ValueError("SWAG sample directory differs from the forty-case schedule")

        admission_manifest = {
            "schema_version": 1,
            "candidate_method": CANDIDATE_METHOD,
            "source_experiment": SOURCE_EXPERIMENT,
            "contract_config_sha256": _file_hash(contract_path),
            "sampling_source_anchors": {
                name: _file_hash(repo / relative)
                for name, relative in {
                    "orchestrator_sha256": "assim_lib/swag_tail_ensemble.py",
                    "comparison_runner_sha256": "assim_lib/compare_3dvar.py",
                    "ensemble_generator_sha256": "assim_lib/evaluate.py",
                    "model_loader_sha256": "assim_lib/model_io.py",
                    "ode_sampler_sha256": "assim_lib/sampler.py",
                    "sampling_config_sha256": ("config/experiments/experiment_m2m_flow_modes.json"),
                }.items()
            },
            "training_manifest": training_manifest,
            "snapshot_matrix_sha256": matrix_sha256,
            "parameter_inventory_sha256": statistics["parameter_inventory_sha256"],
            "swag_statistics": statistics,
            "weight_records": weight_records,
            "iid_weight_draws": True,
            "iid_latent_draws": True,
            "conditionally_iid_exchangeable_members": True,
            "library_identity": {
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "numpy": np.__version__,
            },
            "cases": admission_cases,
        }
        _atomic_csv(output_dir / "per_case_metrics.csv", per_case)
        metadata = {
            "status": "completed",
            "mode": MODE,
            "candidate_method": CANDIDATE_METHOD,
            "source_experiment": SOURCE_EXPERIMENT,
            "dataset_split": "valid",
            "date_range": ["2022-01-01", "2022-07-15"],
            "case_stride_days": 5,
            "num_cases": CASE_COUNT,
            "ensemble_size": MEMBER_COUNT,
            "training_seed": TRAINING_SEED,
            "snapshot_epochs_zero_based": list(SNAPSHOT_EPOCHS),
            "weight_sample_seeds": list(WEIGHT_SAMPLE_SEEDS),
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
            "dependent_gate": {
                "mode": DEPENDENT_GATE_MODE,
                "implementation": "existing deep_ensemble_gate full no-compensation gate",
                "families": ["proper", "reliability", "boundary", "spatial", "operational"],
                "paired_bootstrap_draws": 20_000,
                "block_sensitivity_is_diagnostic_only": True,
                "no_compensation": True,
            },
            "scientific_limitations": [
                "SWAG is a Gaussian approximation to one optimizer trajectory, not an exact posterior.",
                "The canonical covariance scale and fixed last-quarter tail are not tuned "
                "on validation scores.",
                "The reused 2022 development period is not independent test evidence.",
            ],
        }
        _atomic_json(output_dir / "metadata.json", metadata)
        _atomic_json(
            output_dir / "aggregate_case_mean_metrics.json",
            {
                "schema_version": 1,
                "status": "sampling_completed_gate_pending",
                "candidate_method": CANDIDATE_METHOD,
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
                "dependent_gate_mode": DEPENDENT_GATE_MODE,
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
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        if len(sys.argv) < 4 or sys.argv[2] != "--member-index":
            raise ValueError("trusted SWAG worker requires --member-index")
        member_index = int(sys.argv[3])
        del sys.argv[1:4]
        return _worker_main(member_index)
    args = parse_args()
    run(args.output_dir, args.source_experiment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
