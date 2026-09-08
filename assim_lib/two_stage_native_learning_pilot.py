"""Run the reviewed assimilation and dynamics learning pilots sequentially."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path


MIN_SHM_BYTES = 32 * 1024**3
CONTRACT_SHA256S = {
    "assimilation_data.json": "7b6d05fcd690c9670d9dd5c0821225bb2e09ca75ff7ab76471507529c483f9d8",
    "assimilation_stats.json": "5b87df05d93930b05a0f52625bd9293b553dfb187c8396e32a53c3c1bd3ead14",
    "assimilation_archive_audit.json": "b65c122488f3a9b27f8a4eaa9d88f2239dbd074233c672d9aa0341209a647959",
    "dynamics_data.json": "cb1737273a80a2dc58363dc1216556bdc0b6f6d556ae8b0b119a267e67d9f668",
    "dynamics_stats.json": "d1603a841b8126aa5f8d6499076d2a6f3197a998ab114dbc1e4caeda7f7ac7bc",
    "dynamics_archive_audit.json": "00155552aac6a9e7439e5e7d8143eb710182e8af8562d88ae91dee311077e526",
}
ROLE_METHODS = {
    "assimilation": "two_stage_assimilation_s_native.json",
    "dynamics": "two_stage_dynamics_s_native.json",
}


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_contracts(contract_dir: Path) -> None:
    for name, expected in CONTRACT_SHA256S.items():
        path = contract_dir / name
        if path.is_symlink() or not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"trusted contract differs: {name}")


def _available_shm_bytes() -> int:
    stats = os.statvfs("/dev/shm")
    return int(stats.f_bavail * stats.f_frsize)


def _build_experiment(role: str, root: Path, output_dir: Path, contract_dir: Path) -> dict:
    if role not in ROLE_METHODS:
        raise ValueError(f"unknown learning role: {role}")
    run_root = output_dir / role / "training"
    return {
        "project_name": "sea_ice_two_stage",
        "task_name": f"two_stage_{role}_s_native_pilot_v1",
        "data_config": str((contract_dir / f"{role}_data.json").resolve()),
        "model_config": str((root / "config/methods" / ROLE_METHODS[role]).resolve()),
        "training": {
            "base_output_dir": str(run_root.resolve()),
            "run_name": "seed1701",
            "structured_state_stats_path": str(
                (contract_dir / f"{role}_stats.json").resolve()
            ),
            "num_workers_train": 4,
            "num_workers_val": 2,
            "activation_checkpointing": False,
            "sample_every_n_epochs": 1,
            "metric_every_n_epochs": 2,
            "metric_num_cases": 4,
            "metric_num_ensemble": 5,
            "metric_num_timesteps": 17,
            "metric_stride_days": 30,
            "metric_save_ensemble_samples": True,
        },
        "clearml": {
            "enabled": True,
            "tags": [
                "two-stage",
                role,
                "learning-pilot",
                "one-gpu",
                "no-checkpointing",
                "validation-samples",
            ],
            "upload_checkpoints": False,
        },
    }


def _tensor_tree_bytes(value: object) -> int:
    import torch

    if torch.is_tensor(value):
        return int(value.numel() * value.element_size())
    if isinstance(value, dict):
        return sum(_tensor_tree_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_tree_bytes(item) for item in value)
    return 0


def _preflight_role(role: str, experiment: dict, available_shm: int) -> dict:
    import torch
    from torch.utils.data import default_collate

    from .config import load_json
    from .data import build_dataset
    from .runtime import build_dataloader
    from .structured_archive_audit import validate_bound_archive_audit
    from .structured_joint_state import (
        canonical_mapping_sha256,
        validate_conditioning_normalization,
        validate_structured_state_stats,
    )

    data_path = Path(experiment["data_config"])
    stats_path = Path(experiment["training"]["structured_state_stats_path"])
    data = load_json(data_path)
    stats = load_json(stats_path)
    audit = validate_bound_archive_audit(data, data_path)
    validate_structured_state_stats(stats)
    validate_conditioning_normalization(data, stats)
    if canonical_mapping_sha256(data) != stats.get("data_config_sha256"):
        raise ValueError(f"{role} data/stats binding differs")
    train = build_dataset(data, split="train")
    valid = build_dataset(data, split="valid")
    train.validate_structured_sral_audit_contract(audit)
    valid.validate_structured_sral_audit_contract(audit)
    if train.provenance().get("pair_manifest_sha256") != stats.get("pair_manifest_sha256"):
        raise ValueError(f"{role} train pair manifest differs")

    training = experiment["training"]
    train_batch_size = 8
    valid_batch_size = 4
    prefetch = 4
    train_sample_bytes = _tensor_tree_bytes(default_collate([train[0]]))
    valid_sample_bytes = _tensor_tree_bytes(default_collate([valid[0]]))
    train_estimate = train_sample_bytes * train_batch_size * (1 + 4 * prefetch)
    valid_estimate = valid_sample_bytes * valid_batch_size * (1 + 2 * prefetch)
    if max(train_estimate, valid_estimate) > 0.70 * available_shm:
        raise RuntimeError(
            f"{role} full-batch IPC estimate exceeds shared-memory budget: "
            f"train={train_estimate}, valid={valid_estimate}, available={available_shm}"
        )
    train_loader = build_dataloader(
        train,
        train_batch_size,
        int(training["num_workers_train"]),
        shuffle=True,
        prefetch_factor=prefetch,
        generator=torch.Generator().manual_seed(1701),
    )
    valid_loader = build_dataloader(
        valid,
        valid_batch_size,
        int(training["num_workers_val"]),
        shuffle=False,
        prefetch_factor=prefetch,
    )
    train_iterator = iter(train_loader)
    observed_train = [next(train_iterator), next(train_iterator)]
    valid_iterator = iter(valid_loader)
    observed_valid = next(valid_iterator)
    if any(batch["truth"].shape[0] != train_batch_size for batch in observed_train):
        raise RuntimeError(f"{role} full train loader returned a partial batch")
    if observed_valid["truth"].shape[0] != valid_batch_size:
        raise RuntimeError(f"{role} full validation loader returned a partial first batch")
    observed_train_bytes = max(_tensor_tree_bytes(batch) for batch in observed_train)
    observed_valid_bytes = _tensor_tree_bytes(observed_valid)
    del observed_train, observed_valid, train_iterator, valid_iterator, train_loader, valid_loader
    del train, valid
    gc.collect()
    return {
        "role": role,
        "train_sample_bytes": train_sample_bytes,
        "valid_sample_bytes": valid_sample_bytes,
        "estimated_train_peak_ipc_bytes": train_estimate,
        "estimated_valid_peak_ipc_bytes": valid_estimate,
        "observed_train_batch_bytes": observed_train_bytes,
        "observed_valid_batch_bytes": observed_valid_bytes,
        "workers_train": 4,
        "workers_val": 2,
        "prefetch_factor": prefetch,
    }


def _summarize_run(role: str, run_dir: Path) -> dict:
    metrics_path = run_dir / "metrics.json"
    for required in (metrics_path, run_dir / "last_model.pth", run_dir / "ema_last_model.pth"):
        if required.is_symlink() or not required.is_file():
            raise RuntimeError(f"{role} training output is incomplete: {required.name}")
    history = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not isinstance(history, list) or len(history) != 4:
        raise RuntimeError(f"{role} validation history must contain four epochs")
    final = history[-1]
    diagnostic = final.get("structured_sampling_diagnostic", {})
    if diagnostic.get("status") != "passed" or diagnostic.get("epoch") != 3:
        raise RuntimeError(f"{role} final EMA sampling diagnostic did not pass")
    rank_counts = final.get("lead0_sic_rank_counts")
    if (
        not isinstance(rank_counts, list)
        or len(rank_counts) != 6
        or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in rank_counts)
    ):
        raise RuntimeError(f"{role} final rank diagnostic is absent or invalid")
    samples = sorted((run_dir / "samples").glob("epoch_*"))
    required_samples = {
        run_dir / "samples" / "epoch_0003_structured_trajectory.png",
        run_dir / "samples" / "epoch_0003_structured_trajectory.pt",
        run_dir / "samples" / "epoch_0003_structured_ensemble.pt",
    }
    if any(path.is_symlink() or not path.is_file() for path in required_samples):
        raise RuntimeError(f"{role} final validation samples are incomplete")
    return {
        "role": role,
        "status": "completed",
        "run_dir": str(run_dir),
        "validation_epochs": len(history),
        "last_validation": history[-1],
        "final_sampling_binding": {
            "epoch": int(diagnostic["epoch"]),
            "weight_source": "ema",
            "ema_checkpoint_sha256": _sha256(run_dir / "ema_last_model.pth"),
        },
        "ema_checkpoint_sha256": _sha256(run_dir / "ema_last_model.pth"),
        "sample_artifacts": [str(path.relative_to(run_dir)) for path in samples],
    }


def run(output_dir: Path, contract_dir: Path) -> dict:
    import torch

    if torch.cuda.device_count() != 1:
        raise RuntimeError("learning pilot requires exactly one visible GPU")
    available_shm = _available_shm_bytes()
    if available_shm < MIN_SHM_BYTES:
        raise RuntimeError(
            f"learning pilot requires at least {MIN_SHM_BYTES} shared-memory bytes; "
            f"available={available_shm}"
        )
    _validate_contracts(contract_dir)
    root = Path(__file__).resolve().parents[1]
    completed: list[dict] = []
    preflights: list[dict] = []
    status_path = output_dir / "run_status.json"
    try:
        for role in ("assimilation", "dynamics"):
            _atomic_json(
                status_path,
                {
                    "schema_version": "two_stage_native_learning_pilot_v1",
                    "status": "running",
                    "phase": role,
                    "completed": completed,
                },
            )
            experiment = _build_experiment(role, root, output_dir, contract_dir)
            preflights.append(_preflight_role(role, experiment, available_shm))
            _atomic_json(output_dir / "full_loader_preflight.json", {"roles": preflights})
            config_path = output_dir / "control" / f"{role}_experiment.json"
            _atomic_json(config_path, experiment)
            subprocess.run(
                [sys.executable, "-m", "assim_lib.main", "--config", str(config_path)],
                cwd=root,
                stdin=subprocess.DEVNULL,
                check=True,
            )
            completed.append(
                _summarize_run(
                    role,
                    output_dir / role / "training" / "seed1701",
                )
            )
        result = {
            "schema_version": "two_stage_native_learning_pilot_v1",
            "status": "completed",
            "shared_memory_available_bytes": available_shm,
            "full_loader_preflight": preflights,
            "models": completed,
        }
        _atomic_json(output_dir / "learning_pilot.json", result)
        _atomic_json(status_path, result)
        return result
    except Exception as error:
        _atomic_json(
            status_path,
            {
                "schema_version": "two_stage_native_learning_pilot_v1",
                "status": "failed",
                "phase": ("assimilation" if not completed else "dynamics"),
                "completed": completed,
                "error_type": type(error).__name__,
                "error": str(error)[:1000],
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.output_dir.resolve(), arguments.contract_dir.resolve()), indent=2))


if __name__ == "__main__":
    main()
