"""Fail-closed identity and single-GPU guard for the structured full training run.

This module intentionally imports only the Python standard library.  It is run
both on the host and inside the container before importing torch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping


SNAPSHOT_RELATIVE_PATH = Path("paper/STRUCTURED_JOINT_FORECAST_SNAPSHOT_SHA256.json")
SNAPSHOT_SHA256 = "9eff2a34ad52bff934f0979fb03e3794480d9754eada91d9cc7c645a3641dd73"
EXTENSION_RELATIVE_PATH = Path("paper/STRUCTURED_JOINT_GPU_SMOKE_EXTENSION_SHA256.json")
EXTENSION_SHA256 = "e409bd39b0fb059aa2795fcff2c2bb5c17c149f191ec59ee5be853db2f2a1605"
PROTOCOL_RELATIVE_PATH = Path("config/admission/structured_joint_gpu_smoke_v1.json")
PROTOCOL_SHA256 = "e5e7b8c0e372631bf47a7a2113823ebee0537fab27406ff577ce57ab6f501179"
SMOKE_RESULT_SHA256 = "3ee4d71fdf9ab04a3f04cd03bb304a59bdd9f6a3a09363bb78f38b8b8a2bab11"
SMOKE_IDENTITY_SNAPSHOT_SHA256 = (
    "e0a4ee5fee858117a6556fb54af9bf0be5d80b1c6b2cd8f3311ee6b9b5832e91"
)
SMOKE_SCHEMA = "structured_joint_gpu_admission_smoke_v1"
SNAPSHOT_SCHEMA = "structured_joint_forecast_snapshot_sha256_v1"
EXTENSION_SCHEMA = "structured_joint_gpu_smoke_extension_sha256_v1"
EXPECTED_GPU_UUID = "GPU-40ff1cbb-07cc-992e-23cb-8fe181972b76"
MAX_PREEXISTING_MEMORY_MIB = 512
MAX_PREEXISTING_UTILIZATION_PERCENT = 5
REQUIRED_ADMITTED_CODE = {
    "assim_lib/config.py",
    "assim_lib/data.py",
    "assim_lib/forecast.py",
    "assim_lib/model_io.py",
    "assim_lib/runtime.py",
    "assim_lib/sampler.py",
    "assim_lib/structured_archive_audit.py",
    "assim_lib/structured_joint_gpu_admission.py",
    "assim_lib/structured_joint_preflight.py",
    "assim_lib/structured_joint_state.py",
    "assim_lib/trainer.py",
    "assim_lib/transforms.py",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ordinary_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be an ordinary file: {path}")
    return path


def _load_bound_json(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    path = _ordinary_file(path, label)
    actual = sha256(path)
    if actual != expected_sha256:
        raise ValueError(f"{label} SHA-256 differs: expected={expected_sha256}, actual={actual}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _validate_manifest_files(root: Path, manifest: Mapping[str, Any], label: str) -> None:
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError(f"{label} has no file identity mapping")
    resolved_root = root.resolve()
    for relative, expected in files.items():
        candidate = _ordinary_file(root / relative, f"{label} file {relative}")
        try:
            candidate.resolve().relative_to(resolved_root)
        except ValueError as error:
            raise ValueError(f"{label} path escapes the worktree: {relative}") from error
        actual = sha256(candidate)
        if actual != expected:
            raise ValueError(
                f"{label} file differs: {relative}, expected={expected}, actual={actual}"
            )


def _verified_files(value: Any, label: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} is missing")
    files = value.get("verified_files")
    if not isinstance(files, dict):
        raise ValueError(f"{label}.verified_files is missing")
    if value.get("verified_file_count") != len(files):
        raise ValueError(f"{label}.verified_file_count differs")
    return files


def validate_launch_inputs(
    worktree: str | Path,
    smoke_result_path: str | Path,
    *,
    launch_payload_manifest: str | Path | None = None,
    launch_payload_sha256: str | None = None,
) -> dict[str, Any]:
    root = Path(worktree).expanduser().absolute()
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"worktree must be an ordinary directory: {root}")

    snapshot = _load_bound_json(root / SNAPSHOT_RELATIVE_PATH, SNAPSHOT_SHA256, "snapshot")
    extension = _load_bound_json(
        root / EXTENSION_RELATIVE_PATH, EXTENSION_SHA256, "smoke extension"
    )
    protocol = _load_bound_json(root / PROTOCOL_RELATIVE_PATH, PROTOCOL_SHA256, "protocol")
    smoke = _load_bound_json(Path(smoke_result_path), SMOKE_RESULT_SHA256, "smoke result")

    if snapshot.get("schema_version") != SNAPSHOT_SCHEMA:
        raise ValueError("snapshot schema differs")
    if extension.get("schema_version") != EXTENSION_SCHEMA:
        raise ValueError("smoke extension schema differs")
    if protocol.get("schema_version") != SMOKE_SCHEMA:
        raise ValueError("protocol schema differs")
    if smoke.get("schema_version") != SMOKE_SCHEMA or smoke.get("status") != "passed":
        raise ValueError("smoke result is not a passed result of the bound schema")

    _validate_manifest_files(root, snapshot, "snapshot")
    _validate_manifest_files(root, extension, "smoke extension")

    server_binding = protocol.get("validated_server_snapshot", {})
    extension_binding = protocol.get("smoke_extension_validation", {})
    if server_binding.get("manifest_path") != SNAPSHOT_RELATIVE_PATH.as_posix():
        raise ValueError("protocol snapshot path differs")
    if server_binding.get("manifest_sha256") != SNAPSHOT_SHA256:
        raise ValueError("protocol snapshot SHA-256 differs")
    if extension_binding.get("manifest_path") != EXTENSION_RELATIVE_PATH.as_posix():
        raise ValueError("protocol extension path differs")
    if extension_binding.get("manifest_sha256") != EXTENSION_SHA256:
        raise ValueError("protocol extension SHA-256 differs")
    if protocol.get("expected_gpu_uuid") != EXPECTED_GPU_UUID:
        raise ValueError("protocol GPU UUID differs")

    if smoke.get("protocol_sha256") != PROTOCOL_SHA256:
        raise ValueError("smoke protocol SHA-256 differs")
    smoke_server = smoke.get("validated_server_snapshot")
    smoke_extension = smoke.get("validated_smoke_extension")
    if _verified_files(smoke_server, "smoke validated snapshot") != snapshot["files"]:
        raise ValueError("smoke validated snapshot files differ")
    if smoke_server.get("manifest_sha256") != SNAPSHOT_SHA256:
        raise ValueError("smoke validated snapshot SHA-256 differs")
    if _verified_files(smoke_extension, "smoke validated extension") != extension["files"]:
        raise ValueError("smoke validated extension files differ")
    if smoke_extension.get("manifest_sha256") != EXTENSION_SHA256:
        raise ValueError("smoke validated extension SHA-256 differs")

    identity = smoke.get("identity_snapshot")
    if not isinstance(identity, dict):
        raise ValueError("smoke identity snapshot is missing")
    if identity.get("snapshot_sha256") != SMOKE_IDENTITY_SNAPSHOT_SHA256:
        raise ValueError("smoke identity snapshot SHA-256 differs")
    identity_files = identity.get("files")
    if not isinstance(identity_files, dict):
        raise ValueError("smoke identity file mapping is missing")

    admitted_code: set[str] = set()
    for name, record in identity_files.items():
        if not isinstance(record, dict):
            raise ValueError(f"invalid smoke identity record: {name}")
        smoke_path = record.get("path")
        expected = record.get("sha256")
        if not isinstance(smoke_path, str) or not smoke_path.startswith("/workspace/"):
            raise ValueError(f"smoke identity path is outside /workspace: {name}")
        relative = smoke_path.removeprefix("/workspace/")
        candidate = _ordinary_file(root / relative, f"smoke identity {name}")
        if sha256(candidate) != expected:
            raise ValueError(f"smoke-admitted file differs: {relative}")
        if name.startswith("code:"):
            admitted_code.add(relative)
    if not REQUIRED_ADMITTED_CODE.issubset(admitted_code):
        missing = sorted(REQUIRED_ADMITTED_CODE - admitted_code)
        raise ValueError(f"smoke identity lacks required admitted code: {missing}")

    sampler = smoke.get("sampler", {})
    if (
        smoke.get("device", {}).get("uuid") != EXPECTED_GPU_UUID
        or sampler.get("nfe") != 128
        or sampler.get("expected_nfe") != 128
        or sampler.get("endpoint_latent_finite") is not True
        or sampler.get("endpoint_latent_dtype") != "torch.float32"
        or sampler.get("endpoint_latent_inactive_nonzero_count") != 0
        or sampler.get("controlled_decode", {}).get("status")
        != "passed_synthetic_api_fixture_not_a_model_forecast"
    ):
        raise ValueError("smoke result lacks required GPU capacity/codec evidence")

    payload_sha = None
    if launch_payload_manifest is not None:
        if not launch_payload_sha256:
            raise ValueError("launch payload SHA-256 is required")
        payload_path = Path(launch_payload_manifest)
        if not payload_path.is_absolute():
            payload_path = root / payload_path
        payload = _load_bound_json(payload_path, launch_payload_sha256, "launch payload")
        if payload.get("schema_version") != "structured_joint_full_launch_payload_v1":
            raise ValueError("launch payload schema differs")
        _validate_manifest_files(root, payload, "launch payload")
        payload_sha = launch_payload_sha256

    return {
        "status": "passed",
        "snapshot_sha256": SNAPSHOT_SHA256,
        "extension_sha256": EXTENSION_SHA256,
        "protocol_sha256": PROTOCOL_SHA256,
        "smoke_result_sha256": SMOKE_RESULT_SHA256,
        "smoke_identity_snapshot_sha256": SMOKE_IDENTITY_SNAPSHOT_SHA256,
        "launch_payload_sha256": payload_sha,
        "verified_identity_records": len(identity_files),
        "verified_admitted_code": sorted(admitted_code),
    }


def validate_idle_gpu(expected_gpu_uuid: str = EXPECTED_GPU_UUID) -> dict[str, Any]:
    rows = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=uuid,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    ).stdout.splitlines()
    matches = [row for row in rows if row.split(",", 1)[0].strip() == expected_gpu_uuid]
    if len(matches) != 1:
        raise RuntimeError("the admitted GPU UUID is not uniquely visible")
    uuid, memory, utilization = [part.strip() for part in matches[0].split(",")]
    if int(memory) > MAX_PREEXISTING_MEMORY_MIB or int(utilization) > MAX_PREEXISTING_UTILIZATION_PERCENT:
        raise RuntimeError(
            f"admitted GPU is not idle: uuid={uuid} memory_mib={memory} utilization={utilization}"
        )
    apps = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    ).stdout.splitlines()
    if any(row.split(",", 1)[0].strip() == expected_gpu_uuid for row in apps if row.strip()):
        raise RuntimeError("the admitted GPU already has a compute process")
    return {
        "uuid": uuid,
        "preexisting_memory_mib": int(memory),
        "preexisting_utilization_percent": int(utilization),
        "preexisting_compute_processes": 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktree", required=True)
    parser.add_argument("--smoke-result", required=True)
    parser.add_argument("--launch-payload-manifest")
    parser.add_argument("--launch-payload-sha256")
    parser.add_argument("--check-gpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = validate_launch_inputs(
        args.worktree,
        args.smoke_result,
        launch_payload_manifest=args.launch_payload_manifest,
        launch_payload_sha256=args.launch_payload_sha256,
    )
    if args.check_gpu:
        result["gpu"] = validate_idle_gpu()
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
