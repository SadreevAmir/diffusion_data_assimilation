"""Bounded, fail-closed GPU admission smoke for the structured joint run.

This module is deliberately separate from the production trainer.  It admits
one immutable experiment/config/data snapshot, executes exactly one real
full-resolution optimization step, and then one truth-free sampler smoke.  It
never initializes ClearML, never writes a checkpoint, and never changes the
training configuration.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import platform
import re
import signal
import subprocess
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import torch
from torch.utils.data import default_collate

from .config import TrainingConfig, jsonable, load_json, resolve_path
from .data import M2MForecastDataset, build_dataset
from .runtime import make_normalized_xy_grid
from .structured_joint_preflight import run_preflight
from .structured_joint_state import canonical_mapping_sha256, validate_conditioning_normalization
from .structured_joint_state import (
    StructuredDecodeSaturationError,
    decode_structured_joint_trajectory,
)


SCHEMA_VERSION = "structured_joint_gpu_admission_smoke_v1"
RESULT_FILENAME = "result.json"
PROTOCOL_RELATIVE_PATH = Path("config/admission/structured_joint_gpu_smoke_v1.json")
EXPECTED_GPU_UUID = "GPU-40ff1cbb-07cc-992e-23cb-8fe181972b76"
EXPECTED_IMAGE_ID = "sha256:6461c778a994df485f968da62fb9992a800f91ad58d455e0db20bd31d8ecc9e5"
EXPECTED_OUTPUT_ROOT = "/home/autoresearch_results/structured_joint_gpu_admission_smoke"
EXPECTED_HOST_OUTPUT_ROOT = (
    "/home/a.madreev/diffusion_data_assimilation/autoresearch_results/"
    "structured_joint_gpu_admission_smoke"
)
EXPECTED_SERVER_WORKTREE = (
    "/home/a.madreev/diffusion_data_assimilation/autoresearch_worktrees/"
    "structured_joint_forecast_v2_20260907_0255"
)
SERVER_SNAPSHOT_RELATIVE_PATH = Path("paper/STRUCTURED_JOINT_FORECAST_SNAPSHOT_SHA256.json")
SERVER_SNAPSHOT_SHA256 = "9eff2a34ad52bff934f0979fb03e3794480d9754eada91d9cc7c645a3641dd73"
REQUIRED_OFFLINE_ENVIRONMENT = {
    "CLEARML_OFFLINE_MODE": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "WANDB_MODE": "offline",
    "WANDB_DISABLED": "true",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
REQUIRED_EXTENSION_FILES = {
    "assim_lib/structured_joint_gpu_admission.py",
    "scripts/run_structured_joint_gpu_admission_container.sh",
    "scripts/run_structured_joint_gpu_admission_smoke.sh",
    "tests/test_structured_joint_gpu_admission.py",
}
IDENTITY_NAMES = ("protocol", "experiment", "data_config", "method_config", "audit", "stats")
SERVER_SNAPSHOT_SCHEMA = "structured_joint_forecast_snapshot_sha256_v1"
EXTENSION_SNAPSHOT_SCHEMA = "structured_joint_gpu_smoke_extension_sha256_v1"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(jsonable(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _ordinary_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be one ordinary file: {path}")
    return path.resolve()


def _prepare_output_directory(output_dir: Path, required_root: Path) -> Path:
    output_dir = Path(os.path.abspath(output_dir.expanduser()))
    required_root = Path(os.path.abspath(required_root.expanduser()))
    if output_dir.parent != required_root:
        raise ValueError(f"output must be one new direct child of the isolated root {required_root}")
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"admission output already exists: {output_dir}")
    required_root.mkdir(parents=True, exist_ok=True)
    if required_root.is_symlink() or not required_root.is_dir():
        raise ValueError("admission output root must be an ordinary directory")
    output_dir.mkdir(mode=0o700)
    return output_dir


@contextmanager
def _exclusive_lock(path: Path):
    path = path.expanduser().absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError(f"admission lock cannot be a symlink: {path}")
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"structured GPU admission lock is already held: {path}") from error
        os.ftruncate(descriptor, 0)
        os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        os.fsync(descriptor)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


@contextmanager
def _walltime_limit(seconds: float, stage: str):
    seconds = float(seconds)
    if seconds <= 0.0:
        raise ValueError(f"{stage} walltime limit must be positive")
    if not hasattr(signal, "setitimer"):
        raise RuntimeError("bounded admission requires POSIX setitimer support")

    def _timeout(_signum, _frame):
        raise TimeoutError(f"{stage} exceeded its {seconds:g}s walltime limit")

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    effective_seconds = (
        min(seconds, previous_timer[0]) if previous_timer[0] > 0.0 else seconds
    )
    started = time.monotonic()
    signal.signal(signal.SIGALRM, _timeout)
    signal.setitimer(signal.ITIMER_REAL, effective_seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer != (0.0, 0.0):
            elapsed = time.monotonic() - started
            remaining = max(previous_timer[0] - elapsed, 1e-6)
            signal.setitimer(signal.ITIMER_REAL, remaining, previous_timer[1])


def _validate_offline_environment(required: Mapping[str, str], environment: Mapping[str, str]) -> None:
    mismatches = {
        key: (environment.get(key), expected)
        for key, expected in required.items()
        if environment.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"offline environment is not fail-closed: {mismatches}")


def _parse_nvidia_csv(output: str, columns: int) -> list[list[str]]:
    rows = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line or line.lower().startswith("no running processes"):
            continue
        values = [value.strip() for value in line.split(",")]
        if len(values) != columns:
            raise RuntimeError(f"unexpected nvidia-smi row: {raw_line!r}")
        rows.append(values)
    return rows


def _run_nvidia_smi(query: str) -> str:
    completed = subprocess.run(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    return completed.stdout


def _run_nvidia_compute_apps() -> str:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    return completed.stdout


def _canonical_torch_gpu_uuid(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("ascii")
    text = str(value).strip()
    # PyTorch 2.5 exposes `_CUuuid`: its string form is the canonical UUID
    # body, while nvidia-smi prefixes the same identity with `GPU-`.
    if len(text) == 36 and text.count("-") == 4:
        text = f"GPU-{text}"
    return text


def _validate_cuda_admission(protocol: Mapping[str, Any]) -> dict[str, Any]:
    expected_uuid = str(protocol["expected_gpu_uuid"])
    resource = protocol["resource_gates"]
    inventory = _parse_nvidia_csv(
        _run_nvidia_smi("uuid,index,memory.used,utilization.gpu"), 4
    )
    if len(inventory) != int(resource["visible_cuda_device_count"]):
        raise RuntimeError(
            f"nvidia-smi exposes {len(inventory)} GPUs; exactly one is required"
        )
    uuid, physical_index, memory_mib, utilization_percent = inventory[0]
    if uuid != expected_uuid:
        raise RuntimeError(f"visible GPU UUID {uuid!r} != admitted {expected_uuid!r}")
    memory_value = int(memory_mib)
    utilization_value = int(utilization_percent)
    if memory_value > int(resource["max_preexisting_memory_mib"]):
        raise RuntimeError(f"visible GPU already uses {memory_value} MiB")
    if utilization_value > int(resource["max_preexisting_utilization_percent"]):
        raise RuntimeError(f"visible GPU utilization is already {utilization_value}%")
    compute_apps = _parse_nvidia_csv(_run_nvidia_compute_apps(), 3)
    target_compute_apps = [row for row in compute_apps if row[1] == expected_uuid]
    if target_compute_apps:
        raise RuntimeError(
            f"visible GPU has pre-existing compute processes: {target_compute_apps}"
        )

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"torch must expose exactly one CUDA device, got {torch.cuda.device_count()}"
        )
    torch.cuda.set_device(0)
    if torch.cuda.current_device() != 0:
        raise RuntimeError("container CUDA remapping must expose the admitted GPU as logical cuda:0")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("the admitted GPU does not support configured bf16 autocast")
    properties = torch.cuda.get_device_properties(0)
    raw_torch_uuid = getattr(properties, "uuid", None)
    torch_uuid = _canonical_torch_gpu_uuid(raw_torch_uuid)
    if torch_uuid is not None and torch_uuid != expected_uuid:
        raise RuntimeError(
            f"torch logical cuda:0 UUID {torch_uuid!r} != nvidia-smi UUID {expected_uuid!r}"
        )
    return {
        "logical_device": "cuda:0",
        "uuid": expected_uuid,
        "torch_uuid": torch_uuid,
        "nvidia_smi_index_inside_container": physical_index,
        "name": properties.name,
        "compute_capability": [properties.major, properties.minor],
        "total_memory_bytes": int(properties.total_memory),
        "preexisting_memory_mib": memory_value,
        "preexisting_utilization_percent": utilization_value,
        "cuda_visible_devices_environment": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _load_protocol(protocol_path: Path, repo_root: Path) -> dict[str, Any]:
    protocol_path = _ordinary_file(protocol_path, "GPU admission protocol")
    frozen_protocol_path = _ordinary_file(
        repo_root / PROTOCOL_RELATIVE_PATH, "frozen GPU admission protocol"
    )
    if protocol_path != frozen_protocol_path:
        raise ValueError("runner was given a different GPU admission protocol")
    protocol = load_json(protocol_path)
    if protocol.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"expected GPU admission protocol schema {SCHEMA_VERSION!r}")
    if protocol.get("paths", {}).get("protocol") != PROTOCOL_RELATIVE_PATH.as_posix():
        raise ValueError("GPU admission protocol does not bind its frozen repository path")
    if protocol.get("expected_gpu_uuid") != EXPECTED_GPU_UUID:
        raise ValueError(f"GPU admission is frozen to UUID {EXPECTED_GPU_UUID}")
    resource = protocol.get("resource_gates", {})
    if (
        resource.get("visible_cuda_device_count") != 1
        or resource.get("logical_cuda_device") != 0
        or resource.get("require_zero_preexisting_compute_processes") is not True
        or int(resource.get("max_preexisting_memory_mib", 10**9)) > 512
        or int(resource.get("max_preexisting_utilization_percent", 10**9)) > 5
    ):
        raise ValueError("GPU admission must require one idle device remapped to logical cuda:0")
    if protocol.get("required_output_root") != EXPECTED_OUTPUT_ROOT:
        raise ValueError("GPU admission output root differs from its isolated fixed root")
    if protocol.get("required_environment") != REQUIRED_OFFLINE_ENVIRONMENT:
        raise ValueError("GPU admission protocol weakened the required offline environment")
    if set(protocol.get("smoke_extension_required_files", ())) != REQUIRED_EXTENSION_FILES:
        raise ValueError("GPU admission protocol changed the required extension file set")
    outer = protocol.get("outer_container_contract", {})
    if (
        outer.get("launcher") != "scripts/run_structured_joint_gpu_admission_container.sh"
        or outer.get("server_worktree") != EXPECTED_SERVER_WORKTREE
        or outer.get("host_output_root") != EXPECTED_HOST_OUTPUT_ROOT
        or outer.get("container_output_root") != EXPECTED_OUTPUT_ROOT
        or outer.get("expected_local_image_id") != EXPECTED_IMAGE_ID
        or outer.get("image_run_reference") != "expected_local_image_id"
        or outer.get("entrypoint") != "/bin/bash"
        or outer.get("pull_policy") != "never"
        or outer.get("network") != "none"
        or outer.get("gpu_device") != EXPECTED_GPU_UUID
        or outer.get("repository_mount") != "read_only"
        or outer.get("data_mount") != "read_only"
        or outer.get("host_gpu_lock") is not True
        or outer.get("no_new_privileges") is not True
        or outer.get("run_as_host_uid_gid") is not True
        or outer.get("cpu_limit") != 4
        or outer.get("detached_unique_container_name") is not True
        or outer.get("python_total_wall_seconds") != 1800
        or outer.get("host_watchdog_grace_seconds") != 30
        or outer.get("host_watchdog_total_seconds") != 1830
        or outer.get("docker_start_timeout_seconds") != 60
        or outer.get("docker_control_timeout_seconds") != 30
        or outer.get("timeout_cleanup_scope")
        != "exact_verified_owned_container_id_and_name_only"
        or outer.get("confirm_container_removal_before_unlock") is not True
        or outer.get("preserve_normal_container_exit_status") is not True
        or outer.get("auto_full_training_permitted") is not False
    ):
        raise ValueError("outer container isolation contract differs")
    limits = protocol.get("limits", {})
    maximum_limits = {
        "config_preflight_wall_seconds": 900,
        "data_materialization_wall_seconds": 300,
        "train_step_wall_seconds": 600,
        "sampler_wall_seconds": 600,
        "sampler_max_nfe": 256,
        "total_wall_seconds": 1800,
    }
    if set(limits) != set(maximum_limits) or any(
        float(limits[name]) <= 0 or float(limits[name]) > maximum
        for name, maximum in maximum_limits.items()
    ):
        raise ValueError("GPU admission time/NFE limits are absent or weakened")
    if protocol.get("clearml_permitted") is not False or protocol.get("network_permitted") is not False:
        raise ValueError("GPU admission protocol must forbid ClearML and network access")
    server_snapshot = protocol.get("validated_server_snapshot", {})
    if (
        server_snapshot.get("manifest_path") != SERVER_SNAPSHOT_RELATIVE_PATH.as_posix()
        or server_snapshot.get("manifest_sha256") != SERVER_SNAPSHOT_SHA256
    ):
        raise ValueError("GPU admission protocol changed the validated server snapshot binding")
    return protocol


def _resolve_identity_paths(
    protocol: Mapping[str, Any], repo_root: Path, experiment_path: Path
) -> dict[str, Path]:
    declared = protocol["paths"]
    paths = {
        name: _ordinary_file(repo_root / declared[name], name)
        for name in IDENTITY_NAMES
        if name != "experiment"
    }
    paths["experiment"] = _ordinary_file(experiment_path, "experiment")
    expected_experiment = _ordinary_file(repo_root / declared["experiment"], "frozen experiment")
    if paths["experiment"] != expected_experiment:
        raise ValueError("runner was given a different experiment")

    experiment = load_json(paths["experiment"])
    if experiment.get("data_overrides") not in (None, {}):
        raise ValueError("GPU admission forbids experiment data overrides")
    if experiment.get("training") not in (None, {}):
        raise ValueError("GPU admission requires the frozen method without training overrides")
    referenced_data = resolve_path(
        experiment["data_config"], paths["experiment"].parent
    ).resolve()
    referenced_method = resolve_path(
        experiment["model_config"], paths["experiment"].parent
    ).resolve()
    if referenced_data != paths["data_config"] or referenced_method != paths["method_config"]:
        raise ValueError("experiment data/method references differ from the admission protocol")
    data = load_json(paths["data_config"])
    method = load_json(paths["method_config"])
    referenced_audit = resolve_path(
        data["archive_semantics_audit_path"], paths["data_config"].parent
    ).resolve()
    referenced_stats = resolve_path(
        method["structured_state_stats_path"], paths["method_config"].parent
    ).resolve()
    if referenced_audit != paths["audit"] or referenced_stats != paths["stats"]:
        raise ValueError("data audit/stats references differ from the admission protocol")
    return paths


def _take_identity_snapshot(paths: Mapping[str, Path]) -> dict[str, Any]:
    files = {
        name: {
            "path": str(path),
            "sha256": _file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for name, path in sorted(paths.items())
    }
    return {"files": files, "snapshot_sha256": _canonical_sha256(files)}


def _assert_identity_snapshot(snapshot: Mapping[str, Any]) -> None:
    for name, record in snapshot["files"].items():
        path = Path(record["path"])
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"identity file {name} changed type during admission")
        if path.stat().st_size != record["size_bytes"] or _file_sha256(path) != record["sha256"]:
            raise RuntimeError(f"identity file {name} changed during admission")


def _validate_hash_manifest_files(
    manifest: Mapping[str, Any], repo_root: Path, *, schema_version: str
) -> dict[str, str]:
    if manifest.get("schema_version") != schema_version:
        raise ValueError(f"expected SHA manifest schema {schema_version!r}")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or not files:
        raise ValueError("SHA manifest must contain a nonempty files mapping")
    verified = {}
    for relative, expected_sha in files.items():
        if not isinstance(relative, str) or not isinstance(expected_sha, str) or len(expected_sha) != 64:
            raise ValueError("SHA manifest file entries must be relative path/SHA-256 pairs")
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError(f"SHA manifest contains an unsafe path: {relative!r}")
        path = _ordinary_file(repo_root / candidate, f"SHA-bound file {relative}")
        actual_sha = _file_sha256(path)
        if actual_sha != expected_sha:
            raise ValueError(
                f"server-validated SHA differs for {relative}: "
                f"expected {expected_sha}, got {actual_sha}"
            )
        verified[relative] = actual_sha
    return verified


def _validate_server_snapshot_manifest(
    protocol: Mapping[str, Any], repo_root: Path
) -> dict[str, Any]:
    binding = protocol.get("validated_server_snapshot")
    if not isinstance(binding, Mapping):
        raise ValueError("protocol does not bind the validated server snapshot")
    path = _ordinary_file(
        repo_root / str(binding.get("manifest_path", "")),
        "validated server snapshot manifest",
    )
    actual_sha = _file_sha256(path)
    if actual_sha != binding.get("manifest_sha256"):
        raise ValueError("validated server snapshot manifest SHA-256 differs")
    manifest = load_json(path)
    verification = manifest.get("verification", {})
    focused_evidence = str(verification.get("server_cpu_focused_tests", ""))
    full_evidence = str(verification.get("server_cpu_tests_directory", ""))
    focused_match = re.fullmatch(r"(\d+)/\1 PASS(?: .*)?", focused_evidence)
    full_match = re.fullmatch(r"(\d+)/\1 PASS(?: .*)?", full_evidence)
    if (
        manifest.get("server_worktree") != EXPECTED_SERVER_WORKTREE
        or verification.get("local_server_byte_identical") is not True
        or focused_match is None
        or int(focused_match.group(1)) < 33
        or full_match is None
        or int(full_match.group(1)) < 80
    ):
        raise ValueError("server snapshot manifest lacks the admitted CPU verification evidence")
    verified_files = _validate_hash_manifest_files(
        manifest, repo_root, schema_version=SERVER_SNAPSHOT_SCHEMA
    )
    return {
        "manifest_path": str(path),
        "manifest_sha256": actual_sha,
        "verified_file_count": len(verified_files),
        "verified_files": verified_files,
    }


def _validate_smoke_extension_manifest(
    protocol: Mapping[str, Any], repo_root: Path
) -> dict[str, Any]:
    binding = protocol.get("smoke_extension_validation")
    if not isinstance(binding, Mapping) or binding.get("status") != "server_cpu_verified":
        raise RuntimeError(
            "GPU admission remains blocked until the smoke extension has server CPU test evidence"
        )
    manifest_path = binding.get("manifest_path")
    manifest_sha = binding.get("manifest_sha256")
    if not isinstance(manifest_path, str) or not isinstance(manifest_sha, str):
        raise ValueError("verified smoke extension must bind a manifest path and SHA-256")
    path = _ordinary_file(repo_root / manifest_path, "smoke extension SHA manifest")
    actual_sha = _file_sha256(path)
    if actual_sha != manifest_sha:
        raise ValueError("smoke extension manifest SHA-256 differs")
    manifest = load_json(path)
    if manifest.get("server_cpu_tests_status") != "passed":
        raise ValueError("smoke extension manifest lacks passing server CPU tests")
    verified_files = _validate_hash_manifest_files(
        manifest, repo_root, schema_version=EXTENSION_SNAPSHOT_SCHEMA
    )
    required = set(protocol["smoke_extension_required_files"])
    if set(verified_files) != required:
        raise ValueError("smoke extension manifest does not bind the exact required file set")
    return {
        "manifest_path": str(path),
        "manifest_sha256": actual_sha,
        "verified_file_count": len(verified_files),
        "verified_files": verified_files,
        "server_cpu_tests": manifest.get("server_cpu_tests"),
    }


def _validate_frozen_contract(
    protocol: Mapping[str, Any],
    data: Mapping[str, Any],
    method: Mapping[str, Any],
    stats: Mapping[str, Any],
) -> TrainingConfig:
    expected = protocol["expected_effective_contract"]
    actual = {
        "dataset_name": data.get("dataset_name"),
        "conditioning_layout": data.get("conditioning_layout"),
        "future_horizon_days": data.get("future_horizon_days"),
        "dynamic_forcing_indices": data.get("dynamic_forcing_indices"),
        "training_objective": method.get("training_objective"),
        "image_size": method.get("image_size"),
        "in_channels": method.get("in_channels"),
        "out_channels": method.get("out_channels"),
        "train_batch_size": method.get("train_batch_size"),
        "mixed_precision": method.get("mixed_precision"),
        "timestep_sampler": method.get("timestep_sampler"),
        "diagnostic_min_optimizer_steps": method.get("diagnostic_min_optimizer_steps"),
        "sample_method": method.get("sample_method"),
        "num_sample_timesteps": method.get("num_sample_timesteps"),
        "sample_rtol": method.get("sample_rtol"),
        "sample_atol": method.get("sample_atol"),
        "sample_end_time": method.get("sample_end_time"),
    }
    if actual != expected:
        raise ValueError(f"effective structured GPU smoke contract differs: {actual} != {expected}")
    config_payload = dict(method)
    config_payload["structured_state_stats"] = dict(stats)
    config = TrainingConfig.from_dict(config_payload)
    if config.mixed_precision != "bf16":
        raise ValueError("GPU admission is frozen to bf16 training/sampling")
    return config


def _move_named_tensors(
    raw_batch: Mapping[str, Any], device: torch.device, keys: tuple[str, ...]
) -> dict[str, torch.Tensor]:
    missing = [key for key in keys if key not in raw_batch]
    if missing:
        raise KeyError(f"structured batch is missing {missing}")
    return {
        key: raw_batch[key].to(device=device, dtype=torch.float32, non_blocking=False)
        for key in keys
    }


def _move_training_batch(
    raw_batch: Mapping[str, Any], device: torch.device
) -> dict[str, torch.Tensor]:
    keys = (
        "truth",
        "background",
        "obs_values",
        "obs_mask",
        "valid_mask",
        "water_mask",
        "structured_conditioning",
        "structured_physical_truth",
        "structured_physical_background",
        "structured_flow_mask",
        "structured_lag0_mask",
        "structured_lag0_physical_values",
    )
    return _move_named_tensors(raw_batch, device, keys)


def _move_forecast_batch(
    raw_batch: Mapping[str, Any], device: torch.device
) -> dict[str, torch.Tensor]:
    keys = (
        "background",
        "valid_mask",
        "structured_conditioning",
        "structured_flow_mask",
        "structured_lag0_mask",
        "structured_lag0_physical_values",
    )
    return _move_named_tensors(raw_batch, device, keys)


def _stratified_uniform_timesteps(
    batch_size: int, device: torch.device, generator: torch.Generator
) -> torch.Tensor:
    jitter = torch.rand(batch_size, device=device, generator=generator)
    timesteps = (torch.arange(batch_size, device=device) + jitter) / float(batch_size)
    return timesteps[torch.randperm(batch_size, device=device, generator=generator)]


def _gradient_norm(parameters) -> float:
    squares = []
    for parameter in parameters:
        if parameter.grad is not None:
            if not bool(torch.isfinite(parameter.grad).all()):
                raise FloatingPointError("non-finite gradient in GPU admission step")
            norm = parameter.grad.detach().norm(2)
            if not bool(torch.isfinite(norm)):
                raise FloatingPointError("non-finite global gradient norm in GPU admission step")
            squares.append(norm.to(torch.float64).square())
    if not squares:
        raise FloatingPointError("GPU admission step produced no gradients")
    return float(torch.stack(squares).sum().sqrt().item())


class _NFECountingModel(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, max_nfe: int):
        super().__init__()
        self.model = model
        self.max_nfe = int(max_nfe)
        self.nfe = 0
        if self.max_nfe <= 0:
            raise ValueError("max_nfe must be positive")

    def forward(self, *args, **kwargs):
        self.nfe += 1
        if self.nfe > self.max_nfe:
            raise RuntimeError(f"sampler exceeded the predeclared max_nfe={self.max_nfe}")
        return self.model(*args, **kwargs)


def _expected_fixed_nfe(method: str, num_timepoints: int) -> int:
    evaluations_per_step = {
        "euler": 1,
        "midpoint": 2,
        "heun3": 3,
        "rk4": 4,
    }
    if method not in evaluations_per_step:
        raise ValueError(f"GPU admission requires a fixed-step solver, got {method!r}")
    if int(num_timepoints) < 2:
        raise ValueError("fixed-step admission requires at least two timepoints")
    return evaluations_per_step[method] * (int(num_timepoints) - 1)


def _validate_truth_free_sample(
    item: Mapping[str, Any], sample: torch.Tensor, *, sic_cap: float
) -> dict[str, Any]:
    forbidden = {"truth", "structured_physical_truth"} & set(item)
    if forbidden or item.get("meta", {}).get("target_trajectory_paths") != []:
        raise ValueError(f"forecast smoke is not truth-free: forbidden={sorted(forbidden)}")
    if sample.ndim != 4 or sample.shape[0] != 1 or sample.shape[1] % 2:
        raise ValueError("structured sample must be [1,2*T,H,W]")
    valid = item["valid_mask"][:, :1].to(device=sample.device) > 0
    valid_pairs = valid.expand(-1, sample.shape[1] // 2, -1, -1)
    sic = sample[:, 0::2]
    sit = sample[:, 1::2]
    support_bad = (
        ~torch.isfinite(sic)
        | ~torch.isfinite(sit)
        | (sic < 0)
        | (sic > float(sic_cap))
        | (sit < 0)
        | ((sic > 0) != (sit > 0))
    ) & valid_pairs
    if bool(torch.any(support_bad)):
        raise ValueError("sampler smoke violates finite joint SIC/SIT support")
    invalid = ~valid.expand_as(sample)
    if bool(torch.any(sample[invalid] != 0)):
        raise ValueError("sampler smoke is nonzero on land/padding")
    lag0_mask = item["structured_lag0_mask"].to(device=sample.device) > 0
    observed_count = int(lag0_mask.sum().item())
    if observed_count <= 0:
        raise ValueError("sampler smoke anchor has no lag0 observations to enforce")
    exact = lag0_mask.expand(-1, 2, -1, -1)
    expected = item["structured_lag0_physical_values"].to(device=sample.device)
    if not torch.equal(sample[:, :2][exact], expected[exact]):
        raise ValueError("sampler smoke failed exact paired d0 observation enforcement")
    return {
        "truth_free": True,
        "lag0_observed_pixel_count": observed_count,
        "lag0_exact_bitwise": True,
        "support_violation_count": 0,
        "land_padding_nonzero_count": 0,
    }


def _memory_record(device: torch.device) -> dict[str, int]:
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def _run_smoke(
    protocol: dict[str, Any], paths: Mapping[str, Path], snapshot: Mapping[str, Any]
) -> dict[str, Any]:
    limits = protocol["limits"]
    stage_times = {}
    start_total = time.perf_counter()
    data = load_json(paths["data_config"])
    method = load_json(paths["method_config"])
    stats = load_json(paths["stats"])
    config = _validate_frozen_contract(protocol, data, method, stats)
    validate_conditioning_normalization(data, stats)
    if canonical_mapping_sha256(data) != stats["data_config_sha256"]:
        raise ValueError("train-only stats/data config identity differs")

    stage_start = time.perf_counter()
    with _walltime_limit(limits["config_preflight_wall_seconds"], "config preflight"):
        preflight = run_preflight(paths["experiment"], build_model=False)
    stage_times["config_preflight_seconds"] = time.perf_counter() - stage_start
    if preflight.get("status") != "ready":
        raise RuntimeError(f"structured CPU preflight did not admit the run: {preflight}")
    if preflight.get("cuda_initialized") is not False:
        raise RuntimeError("structured CPU preflight initialized CUDA before GPU admission")
    _assert_identity_snapshot(snapshot)

    stage_start = time.perf_counter()
    with _walltime_limit(limits["data_materialization_wall_seconds"], "data materialization"):
        # The immediately preceding preflight cryptographically revalidated
        # this exact audit.  Reloading the immutable JSON avoids a second full
        # archive scan while preserving the runtime source/availability check.
        archive_audit = load_json(paths["audit"])
        train_dataset = build_dataset(data, split="train")
        train_dataset.validate_structured_sral_audit_contract(archive_audit)
        if hasattr(train_dataset, "set_epoch"):
            train_dataset.set_epoch(0)
        batch_size = int(config.train_batch_size)
        if len(train_dataset) < batch_size:
            raise ValueError("train dataset cannot provide one full configured batch")
        train_lag0_dates = archive_audit["sral_provenance"][
            "availability_by_split_and_lag"
        ]["train"]["lag0"]["usable_nonempty_geometry_dates"]
        if not train_lag0_dates:
            raise ValueError("train split has no audited usable lag0 date")
        first_observed_date = train_lag0_dates[0]
        matching_indices = [
            index
            for index, (_, target) in enumerate(train_dataset.calendar_pairs)
            if target.date.isoformat() == first_observed_date
        ]
        if len(matching_indices) != 1:
            raise ValueError("audited first train lag0 date does not map to one dataset item")
        batch_start = matching_indices[0]
        if batch_start + batch_size > len(train_dataset):
            raise ValueError("first observed train date cannot start one complete configured batch")
        train_indices = list(range(batch_start, batch_start + batch_size))
        train_items = [train_dataset[index] for index in train_indices]
        train_case_ids = [item["meta"]["case_id"] for item in train_items]
        raw_batch = default_collate(train_items)
        if int(raw_batch["structured_lag0_mask"].sum().item()) <= 0:
            raise ValueError("selected full training batch contains no lag0 observation pixels")

        valid_lag0 = archive_audit["sral_provenance"][
            "availability_by_split_and_lag"
        ]["valid"]["lag0"]["usable_nonempty_geometry_dates"]
        if not valid_lag0:
            raise ValueError("validation split has no audited usable lag0 forecast anchor")
        forecast_anchor = valid_lag0[0]
        forecast_item = M2MForecastDataset.build_structured_forecast_item(
            data, forecast_anchor
        )
        if not forecast_item["meta"]["lag_available"][0]:
            raise ValueError("selected forecast smoke anchor unexpectedly lacks lag0 observations")
    stage_times["data_materialization_seconds"] = time.perf_counter() - stage_start
    _assert_identity_snapshot(snapshot)
    if torch.cuda.is_initialized():
        raise RuntimeError("data materialization initialized CUDA before GPU admission")

    device_identity = _validate_cuda_admission(protocol)
    device = torch.device("cuda:0")
    from diffusers.optimization import get_cosine_schedule_with_warmup
    from diffusers.training_utils import EMAModel

    from .model_io import build_unet
    from .sampler import Sampler
    from .trainer import UNetTrainer

    torch.manual_seed(int(protocol["seeds"]["model_initialization"]))
    torch.cuda.manual_seed(int(protocol["seeds"]["model_initialization"]))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    model = build_unet(config).to(device)
    xformers_enabled = False
    if hasattr(model, "enable_xformers_memory_efficient_attention"):
        model.enable_xformers_memory_efficient_attention()
        xformers_enabled = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(
            int(config.lr_scheduler_total_steps)
            if config.lr_scheduler_total_steps > 0
            else preflight["planned_optimizer_steps"]
        ),
    )
    ema = EMAModel(
        model.parameters(),
        decay=float(config.ema_decay),
        min_decay=float(config.ema_min_decay),
        update_after_step=int(config.ema_update_after_step),
        use_ema_warmup=bool(config.ema_use_warmup),
    )
    ema.to(device)
    model.train()
    batch = _move_training_batch(raw_batch, device)
    helper = SimpleNamespace(
        config=config,
        _grid=make_normalized_xy_grid(*config.image_size, device=device),
    )
    generator = torch.Generator(device=device).manual_seed(int(protocol["seeds"]["train_step"]))
    timesteps = _stratified_uniform_timesteps(batch_size, device, generator)
    state, target = UNetTrainer._make_training_pair(
        helper,
        batch["truth"],
        batch,
        timesteps,
        generator=generator,
    )
    from .flow_parameterization import (
        GAUSSIAN_PATH_PRECONDITIONED,
        reconstruct_velocity,
        velocity_model_state,
    )

    model_state = velocity_model_state(
        state, timesteps, config.structured_velocity_parameterization
    )
    model_input = UNetTrainer._make_model_input(helper, model_state, batch)
    if tuple(model_input.shape) != (
        batch_size,
        config.in_channels,
        *config.image_size,
    ):
        raise ValueError(f"full-resolution model input shape differs: {tuple(model_input.shape)}")
    if not bool(torch.isfinite(model_input).all()) or not bool(torch.isfinite(target).all()):
        raise FloatingPointError("non-finite full-resolution training input/target")
    active_flow_coordinates = int(batch["structured_flow_mask"].sum().item())
    if active_flow_coordinates <= 0:
        raise ValueError("full-resolution training batch has an empty flow-loss domain")
    model_setup_memory = _memory_record(device)

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    stage_start = time.perf_counter()
    with _walltime_limit(limits["train_step_wall_seconds"], "full-resolution train step"):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            model_output = model(
                model_input, timesteps * 1000, return_dict=False
            )[0]
            prediction = reconstruct_velocity(
                model_output,
                state,
                timesteps,
                config.structured_velocity_parameterization,
            )
            loss = UNetTrainer._masked_mse(
                prediction, target, batch["structured_flow_mask"]
            )
        if model_output.dtype != torch.bfloat16:
            raise RuntimeError(
                f"bf16 autocast produced {model_output.dtype} raw model output"
            )
        expected_velocity_dtype = (
            torch.float32
            if config.structured_velocity_parameterization
            == GAUSSIAN_PATH_PRECONDITIONED
            else torch.bfloat16
        )
        if prediction.dtype != expected_velocity_dtype:
            raise RuntimeError(
                "reconstructed full velocity dtype differs: "
                f"expected {expected_velocity_dtype}, got {prediction.dtype}"
            )
        if loss.numel() != 1 or not bool(torch.isfinite(loss)):
            raise FloatingPointError("non-finite scalar loss in GPU admission step")
        loss.backward()
        gradient_norm_before_clip = _gradient_norm(model.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        gradient_norm_after_clip = _gradient_norm(model.parameters())
        learning_rate_before_step = float(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
        ema.step(model.parameters())
        for name, parameter in model.named_parameters():
            if not bool(torch.isfinite(parameter).all()):
                raise FloatingPointError(f"non-finite parameter after optimizer step: {name}")
        torch.cuda.synchronize(device)
    train_step_seconds = time.perf_counter() - stage_start
    stage_times["train_step_seconds"] = train_step_seconds
    if train_step_seconds > float(limits["train_step_wall_seconds"]):
        raise TimeoutError("full-resolution train step exceeded its post-sync walltime gate")
    train_memory = _memory_record(device)
    train_result = {
        "batch_indices": train_indices,
        "case_ids": train_case_ids,
        "batch_shape": list(model_input.shape),
        "target_shape": list(target.shape),
        "configured_batch_size": batch_size,
        "configured_image_size": list(config.image_size),
        "configured_mixed_precision": config.mixed_precision,
        "xformers_memory_efficient_attention": xformers_enabled,
        "active_flow_coordinate_count": active_flow_coordinates,
        "timestep_min": float(timesteps.min().item()),
        "timestep_max": float(timesteps.max().item()),
        "autocast_dtype": str(model_output.dtype),
        "full_velocity_dtype": str(prediction.dtype),
        "loss_dtype": str(loss.dtype),
        "loss": float(loss.detach().item()),
        "gradient_norm_before_clip": gradient_norm_before_clip,
        "gradient_norm_after_clip": gradient_norm_after_clip,
        "learning_rate_before_step": learning_rate_before_step,
        "learning_rate_after_scheduler": float(optimizer.param_groups[0]["lr"]),
        "optimizer_steps": 1,
        "ema_updates": 1,
        "wall_seconds": train_step_seconds,
        "memory": train_memory,
    }

    model.eval()
    forecast_batch = _move_forecast_batch(
        default_collate([forecast_item]), device
    )
    initial_noise = torch.randn(
        (1, config.out_channels, *config.image_size),
        device=device,
        dtype=torch.float32,
        generator=torch.Generator(device=device).manual_seed(
            int(protocol["seeds"]["sampler_noise"])
        ),
    )
    counting_model = _NFECountingModel(model, int(limits["sampler_max_nfe"]))
    sampler = Sampler(
        counting_model,
        structured_velocity_parameterization=(
            config.structured_velocity_parameterization
        ),
    )
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    stage_start = time.perf_counter()
    ema.store(model.parameters())
    ema.copy_to(model.parameters())
    try:
        with _walltime_limit(limits["sampler_wall_seconds"], "truth-free sampler smoke"):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                zeros = torch.zeros_like(forecast_batch["background"])
                latent_sample = sampler.sample_conditioned(
                    background=forecast_batch["background"],
                    background_mask=forecast_batch["valid_mask"][:, :1].expand_as(
                        forecast_batch["background"]
                    ),
                    obs_values=zeros,
                    obs_mask=zeros,
                    water_mask=forecast_batch["valid_mask"][:, :1],
                    model_conditioning=forecast_batch["structured_conditioning"],
                    valid_mask=forecast_batch["valid_mask"],
                    state_mask=forecast_batch["structured_flow_mask"],
                    size=config.image_size,
                    num_timesteps=config.num_sample_timesteps,
                    device=device,
                    method=config.sample_method,
                    rtol=config.sample_rtol,
                    atol=config.sample_atol,
                    start_mode="noise",
                    sample_target="state",
                    state_channels=config.out_channels,
                    end_time=0.0,
                    initial_noise=initial_noise,
                )
            torch.cuda.synchronize(device)
    finally:
        ema.restore(model.parameters())
    sampler_seconds = time.perf_counter() - stage_start
    stage_times["sampler_seconds"] = sampler_seconds
    if sampler_seconds > float(limits["sampler_wall_seconds"]):
        raise TimeoutError("truth-free sampler exceeded its post-sync walltime gate")
    expected_nfe = _expected_fixed_nfe(
        config.sample_method, config.num_sample_timesteps
    )
    if counting_model.nfe != expected_nfe:
        raise RuntimeError(
            f"fixed-step sampler used nfe={counting_model.nfe}, expected exactly {expected_nfe}"
        )
    if tuple(latent_sample.shape) != (1, config.out_channels, *config.image_size):
        raise RuntimeError(f"actual-model latent sample shape differs: {tuple(latent_sample.shape)}")
    if latent_sample.dtype != torch.float32 or not bool(torch.isfinite(latent_sample).all()):
        raise FloatingPointError("actual-model RK4 endpoint latent must be finite float32")
    flow_active = forecast_batch["structured_flow_mask"] > 0
    if bool(torch.any(latent_sample[~flow_active] != 0)):
        raise ValueError("actual-model RK4 endpoint is nonzero outside the flow subspace")

    actual_model_decode: dict[str, Any]
    try:
        actual_physical = decode_structured_joint_trajectory(latent_sample, stats)
        exact_mask = forecast_batch["structured_lag0_mask"] > 0
        actual_physical[:, :2] = torch.where(
            exact_mask,
            forecast_batch["structured_lag0_physical_values"],
            actual_physical[:, :2],
        )
        actual_physical = torch.where(
            forecast_batch["valid_mask"][:, :1] > 0,
            actual_physical,
            torch.zeros_like(actual_physical),
        )
        actual_model_decode = {
            "status": "passed_not_a_quality_claim",
            **_validate_truth_free_sample(
                {**forecast_batch, "meta": forecast_item["meta"]},
                actual_physical,
                sic_cap=float(stats["sic_cap"]),
            ),
        }
    except StructuredDecodeSaturationError as error:
        actual_model_decode = {
            "status": "not_applicable_near_untrained_decode_saturation",
            "error": str(error),
            "diagnostics": error.diagnostics,
            "quality_claim_permitted": False,
        }

    class _ZeroVelocity(torch.nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            self.channels = int(channels)

        def forward(self, value, timestep):
            return (
                torch.zeros(
                    (value.shape[0], self.channels, *value.shape[-2:]),
                    device=value.device,
                    dtype=value.dtype,
                ),
            )

    controlled_sample = Sampler(
        _ZeroVelocity(config.out_channels),
        structured_velocity_parameterization=(
            config.structured_velocity_parameterization
        ),
    ).sample_structured_trajectory(
        background_trajectory=forecast_batch["background"],
        model_conditioning=forecast_batch["structured_conditioning"],
        valid_mask=forecast_batch["valid_mask"],
        flow_mask=forecast_batch["structured_flow_mask"],
        lag0_physical_values=forecast_batch["structured_lag0_physical_values"],
        lag0_mask=forecast_batch["structured_lag0_mask"],
        stats=stats,
        size=config.image_size,
        num_timesteps=2,
        device=device,
        method="euler",
        initial_noise=torch.zeros_like(initial_noise),
    )
    controlled_support = _validate_truth_free_sample(
        {**forecast_batch, "meta": forecast_item["meta"]},
        controlled_sample,
        sic_cap=float(stats["sic_cap"]),
    )
    active_values = latent_sample[flow_active]
    sampler_result = {
        "forecast_anchor": forecast_anchor,
        "case_id": forecast_item["meta"]["case_id"],
        "endpoint_latent_shape": list(latent_sample.shape),
        "controlled_physical_shape": list(controlled_sample.shape),
        "solver": config.sample_method,
        "requested_timepoints": config.num_sample_timesteps,
        "start_time": 1.0,
        "end_time": config.sample_end_time,
        "rtol": config.sample_rtol,
        "atol": config.sample_atol,
        "nfe": counting_model.nfe,
        "expected_nfe": expected_nfe,
        "max_nfe": int(limits["sampler_max_nfe"]),
        "weights": "EMA after the single admission optimizer step; capacity only",
        "wall_seconds": sampler_seconds,
        "memory": _memory_record(device),
        "endpoint_latent_dtype": str(latent_sample.dtype),
        "endpoint_latent_finite": True,
        "endpoint_latent_active_min": float(active_values.min().item()),
        "endpoint_latent_active_max": float(active_values.max().item()),
        "endpoint_latent_active_abs_mean": float(active_values.abs().mean().item()),
        "endpoint_latent_inactive_nonzero_count": 0,
        "actual_model_physical_decode": actual_model_decode,
        "controlled_decode": {
            "status": "passed_synthetic_api_fixture_not_a_model_forecast",
            "solver": "euler",
            "timepoints": 2,
            **controlled_support,
        },
    }

    _assert_identity_snapshot(snapshot)
    total_seconds = time.perf_counter() - start_total
    if total_seconds > float(limits["total_wall_seconds"]):
        raise TimeoutError("GPU admission exceeded total walltime gate")
    peak_allocated = max(
        model_setup_memory["peak_allocated_bytes"],
        train_memory["peak_allocated_bytes"],
        sampler_result["memory"]["peak_allocated_bytes"],
    )
    peak_reserved = max(
        model_setup_memory["peak_reserved_bytes"],
        train_memory["peak_reserved_bytes"],
        sampler_result["memory"]["peak_reserved_bytes"],
    )
    return {
        "status": "passed",
        "schema_version": SCHEMA_VERSION,
        "identity_snapshot": snapshot,
        "device": device_identity,
        "runtime_versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        },
        "preflight": preflight,
        "effective_training_config": jsonable(asdict(config)),
        "train_step": train_result,
        "sampler": sampler_result,
        "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "model_and_ema_setup_memory": model_setup_memory,
        "overall_peak_allocated_bytes": peak_allocated,
        "overall_peak_reserved_bytes": peak_reserved,
        "stage_times": stage_times,
        "total_wall_seconds": total_seconds,
        "clearml_initialized": False,
        "network_calls_permitted": False,
    }


def run_admission(
    experiment_path: str | Path,
    protocol_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    protocol_path = Path(protocol_path).expanduser().absolute()
    protocol = _load_protocol(protocol_path, repo_root)
    _validate_offline_environment(protocol["required_environment"], os.environ)
    server_snapshot = _validate_server_snapshot_manifest(protocol, repo_root)
    extension_snapshot = _validate_smoke_extension_manifest(protocol, repo_root)
    paths = _resolve_identity_paths(protocol, repo_root, Path(experiment_path).expanduser().absolute())
    snapshot_paths = dict(paths)
    snapshot_paths["validated_server_snapshot_manifest"] = Path(
        server_snapshot["manifest_path"]
    )
    snapshot_paths["validated_smoke_extension_manifest"] = Path(
        extension_snapshot["manifest_path"]
    )
    for relative in server_snapshot["verified_files"]:
        snapshot_paths[f"server-validated:{relative}"] = repo_root / relative
    for relative in extension_snapshot["verified_files"]:
        snapshot_paths[f"extension-validated:{relative}"] = repo_root / relative
    for relative in protocol["sha_snapshot_code_paths"]:
        path = _ordinary_file(repo_root / relative, f"snapshot code {relative}")
        snapshot_paths[f"code:{relative}"] = path
    snapshot = _take_identity_snapshot(snapshot_paths)
    lock_path = Path(protocol["lock_path"])
    with _exclusive_lock(lock_path):
        destination = _prepare_output_directory(
            Path(output_dir), Path(protocol["required_output_root"])
        )
        result_path = destination / RESULT_FILENAME
        try:
            with _walltime_limit(
                protocol["limits"]["total_wall_seconds"], "complete GPU admission"
            ):
                result = _run_smoke(protocol, paths, snapshot)
            result["validated_server_snapshot"] = server_snapshot
            result["validated_smoke_extension"] = extension_snapshot
            result["protocol_sha256"] = snapshot["files"]["protocol"]["sha256"]
            result["completed_epoch_seconds"] = time.time()
            _atomic_json(result_path, result)
            return result
        except Exception as error:
            failure = {
                "status": "failed",
                "schema_version": SCHEMA_VERSION,
                "error_type": type(error).__name__,
                "error": str(error),
                "identity_snapshot": snapshot,
                "completed_epoch_seconds": time.time(),
                "clearml_initialized": False,
                "network_calls_permitted": False,
            }
            _atomic_json(result_path, failure)
            raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_admission(args.experiment, args.protocol, args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
