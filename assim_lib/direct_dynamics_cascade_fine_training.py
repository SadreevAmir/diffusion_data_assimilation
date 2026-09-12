"""Bounded, audited mechanics pilot for the fine stage of the dynamics cascade."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Subset, default_collate

from .config import TrainingConfig, load_json, merge_config_overrides, resolve_path
from .data import build_dataset
from .direct_dynamics_cascade import masked_block_average, residual_flow_pair
from .direct_dynamics_cascade_fine import (
    FINE_INPUT_CHANNELS,
    FineCascadeDynamicsTrainer,
    ProjectedDetailModel,
    teacher_coarse_condition,
)
from .direct_dynamics_training import validate_direct_dataset
from .model_io import build_unet
from .runtime import add_noise, build_dataloader, make_normalized_xy_grid, seed_everything
from .trainer import _atomic_json

EXPECTED_SPLIT_LENGTHS = {"train": 51_792, "valid": 8_544}
EXPECTED_INVENTORY_SHA256 = {
    "train": "514249bfed2b98e7b70ca80f7749e3ed4086b3364b647719cea1c90b781117ea",
    "valid": "0298710a8b4cbb031dd78864281e4b049a60251bc8fccb664056118a3d1e1c97",
}
FINE_BATCH_KEYS = (
    "truth",
    "background",
    "obs_values",
    "obs_mask",
    "valid_mask",
    "water_mask",
    "structured_conditioning",
    "structured_physical_truth",
    "structured_physical_background",
)
PREFETCH_FACTOR = 4
SHM_SAFETY_FRACTION = 0.70


@dataclass
class _Lifecycle:
    run_id: str = "unknown"
    code_commit: str = "unknown"
    phase: str = "configuration"
    output_root: Path | None = None
    output_writable: bool = False
    trainer: Any = None


def _launch_status(status: str, **details: Any) -> None:
    raw_path = os.environ.get("FINE_CASCADE_STATUS_PATH", "").strip()
    if not raw_path:
        return
    path = Path(raw_path)
    if not path.is_absolute() or path.name != "status.json":
        raise ValueError("FINE_CASCADE_STATUS_PATH must be an absolute status.json path")
    _atomic_json(path, {"status": status, **details})


def _write_preflight_evidence(payload: dict[str, Any]) -> Path | None:
    """Persist full preflight evidence in the unique controller launch directory."""
    raw_path = os.environ.get("FINE_CASCADE_STATUS_PATH", "").strip()
    if not raw_path:
        return None
    status_path = Path(raw_path)
    if not status_path.is_absolute() or status_path.name != "status.json":
        raise ValueError("FINE_CASCADE_STATUS_PATH must be an absolute status.json path")
    path = status_path.parent / "preflight.json"
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to replace fine cascade preflight evidence: {path}")
    _atomic_json(path, payload)
    return path


def _record_failure(lifecycle: _Lifecycle, error: BaseException) -> None:
    """Persist the primary failure while cleanup remains strictly best-effort."""
    failure = {
        "status": "failed",
        "phase": lifecycle.phase,
        "error_type": type(error).__name__,
        "error": str(error),
        "code_commit": lifecycle.code_commit,
        "run_id": lifecycle.run_id,
        "output_dir": None if lifecycle.output_root is None else str(lifecycle.output_root),
        "cleanup_errors": [],
    }
    cleanup_errors = failure["cleanup_errors"]
    if lifecycle.output_writable and lifecycle.output_root is not None:
        try:
            _atomic_json(lifecycle.output_root / "fine_cascade_failure.json", failure)
        except Exception as cleanup_error:  # noqa: BLE001 - preserve primary failure
            cleanup_errors.append(f"initial_failure_artifact: {cleanup_error}")
    try:
        _launch_status(**failure)
    except Exception as cleanup_error:  # noqa: BLE001 - preserve primary failure
        cleanup_errors.append(f"initial_failure_status: {cleanup_error}")
    trainer = lifecycle.trainer
    tracker = getattr(trainer, "clearml", None)
    if tracker is not None:
        try:
            tracker.task.mark_failed(
                status_reason=type(error).__name__, status_message=str(error)[:1000]
            )
        except Exception as cleanup_error:  # noqa: BLE001 - preserve primary failure
            cleanup_errors.append(f"clearml_mark_failed: {cleanup_error}")
        try:
            tracker.close()
        except Exception as cleanup_error:  # noqa: BLE001 - preserve primary failure
            cleanup_errors.append(f"clearml_close: {cleanup_error}")
    accelerator = getattr(trainer, "accelerator", None)
    if accelerator is not None:
        try:
            accelerator.end_training()
        except Exception as cleanup_error:  # noqa: BLE001 - preserve primary failure
            cleanup_errors.append(f"accelerator_end_training: {cleanup_error}")
    if lifecycle.output_writable and lifecycle.output_root is not None:
        try:
            _atomic_json(lifecycle.output_root / "fine_cascade_failure.json", failure)
        except Exception as cleanup_error:  # noqa: BLE001 - preserve primary failure
            cleanup_errors.append(f"failure_artifact: {cleanup_error}")
    try:
        _launch_status(**failure)
    except Exception as cleanup_error:  # noqa: BLE001 - preserve primary failure
        cleanup_errors.append(f"failure_status: {cleanup_error}")


def _initialize_trainer(
    lifecycle: _Lifecycle,
    *,
    trainer_class=None,
    **kwargs: Any,
) -> FineCascadeDynamicsTrainer:
    """Expose a partially initialized trainer to failure cleanup immediately."""
    trainer_class = FineCascadeDynamicsTrainer if trainer_class is None else trainer_class
    trainer = trainer_class.__new__(trainer_class)
    lifecycle.trainer = trainer
    trainer_class.__init__(trainer, **kwargs)
    return trainer


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _clean_code_identity(repo_root: Path) -> dict[str, str]:
    clean_environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        env=clean_environment,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repo_root,
        env=clean_environment,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise RuntimeError("fine cascade launch requires an immutable clean git worktree")
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise RuntimeError("fine cascade launch could not resolve a full git commit")
    return {"git_commit": commit, "git_worktree": "clean"}


def _fine_collate(samples: list[dict]) -> dict:
    if not samples:
        raise ValueError("fine cascade collate requires at least one sample")
    missing = [key for key in FINE_BATCH_KEYS if any(key not in sample for sample in samples)]
    if missing:
        raise KeyError(f"fine cascade samples miss required keys: {missing}")
    batch = {key: default_collate([sample[key] for sample in samples]) for key in FINE_BATCH_KEYS}
    batch["meta"] = {"case_id": [str(sample["meta"]["case_id"]) for sample in samples]}
    return batch


def _batch_tensor_bytes(batch: dict) -> int:
    return sum(value.numel() * value.element_size() for value in batch.values() if torch.is_tensor(value))


def _ipc_preflight(train_subset, validation_subset, config: TrainingConfig) -> dict[str, int | float]:
    train_single = _batch_tensor_bytes(_fine_collate([train_subset[0]]))
    validation_single = _batch_tensor_bytes(_fine_collate([validation_subset[0]]))
    train_retained = 1 + config.num_workers_train * PREFETCH_FACTOR
    validation_retained = 1 + config.num_workers_val * PREFETCH_FACTOR
    estimated = (
        train_single * config.train_batch_size * train_retained
        + validation_single * config.eval_batch_size * validation_retained
    )
    stats = os.statvfs("/dev/shm")
    available = int(stats.f_bavail * stats.f_frsize)
    budget = int(SHM_SAFETY_FRACTION * available)
    if estimated > budget:
        raise RuntimeError(
            "simultaneous persistent train/validation DataLoader IPC estimate exceeds "
            f"shared-memory safety budget: estimated={estimated}, budget={budget}, available={available}"
        )
    return {
        "train_single_sample_bytes": train_single,
        "validation_single_sample_bytes": validation_single,
        "train_retained_batches": train_retained,
        "validation_retained_batches": validation_retained,
        "estimated_peak_ipc_bytes": estimated,
        "shared_memory_available_bytes": available,
        "safety_fraction": SHM_SAFETY_FRACTION,
        "prefetch_factor": PREFETCH_FACTOR,
    }


def _calendar_inventory(dataset, *, split: str) -> dict[str, Any]:
    bounds = {
        "train": (date(2016, 1, 1), date(2021, 12, 31)),
        "valid": (date(2022, 1, 1), date(2022, 12, 31)),
    }
    lower, upper = bounds[split]
    digest = hashlib.sha256()
    first_d0 = None
    last_target = None
    for _, target in dataset.calendar_pairs:
        dates = [target.date, *(target.date + timedelta(days=lead) for lead in (3, 6, 9))]
        if any(value < lower or value > upper for value in dates):
            raise ValueError(f"{split} trajectory date escapes its declared split")
        if any(value not in dataset.records_by_date for value in dates):
            raise FileNotFoundError(f"{split} calendar inventory lacks a trajectory target")
        paths = [dataset.records_by_date[value].path.name for value in dates]
        digest.update(("|".join([*(value.isoformat() for value in dates), *paths]) + "\n").encode())
        first_d0 = dates[0] if first_d0 is None else min(first_d0, dates[0])
        last_target = dates[-1] if last_target is None else max(last_target, dates[-1])
    fingerprint = digest.hexdigest()
    if fingerprint != EXPECTED_INVENTORY_SHA256[split]:
        raise ValueError(f"{split} ordered calendar inventory fingerprint differs")
    return {
        "calendar_pairs": len(dataset.calendar_pairs),
        "first_d0": first_d0.isoformat(),
        "last_target": last_target.isoformat(),
        "ordered_inventory_sha256": fingerprint,
    }


def _case_id(dataset, index: int) -> str:
    day_index, archive_slice = divmod(index, 24)
    target = dataset.calendar_pairs[day_index][1]
    return f"{target.date.isoformat()}_slice{archive_slice:02d}"


def _evenly_spaced_indices(length: int, count: int) -> list[int]:
    if not 0 < count <= length:
        raise ValueError(f"subset count must lie in [1,{length}], got {count}")
    if count == 1:
        return [length // 2]
    indices = torch.linspace(0, length - 1, count, dtype=torch.float64).round().to(torch.int64).tolist()
    if len(set(indices)) != count:
        raise RuntimeError("deterministic subset construction produced duplicate indices")
    return indices


def _indices_sha256(indices: list[int]) -> str:
    return hashlib.sha256(",".join(str(value) for value in indices).encode()).hexdigest()


def _gpu_admission_smoke(
    model,
    raw: dict,
    config: TrainingConfig,
    base_noise_channel_scales: tuple[float, ...] | None = None,
    preconditioning_scales: tuple[tuple[float, ...], tuple[float, ...]] | None = None,
    colored_base: tuple[float, tuple[float, ...]] | None = None,
) -> dict:
    """Run the exact 29→6 projected training path without an optimizer step."""
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    projected_model = ProjectedDetailModel(model.to(device)).train()
    truth = raw["truth"].to(device=device, dtype=torch.float32)
    valid_mask = raw["valid_mask"][:, :1].to(device=device, dtype=torch.float32)
    causal = raw["structured_conditioning"].to(device=device, dtype=torch.float32)
    batch_size = truth.shape[0]
    time = torch.linspace(0.05, 0.95, batch_size, device=device)
    noise = torch.randn_like(truth)
    if colored_base is not None:
        if base_noise_channel_scales is not None:
            raise ValueError("GPU smoke cannot combine matched and colored base laws")
        from .direct_dynamics_cascade_fine_colored import colored_projected_gaussian

        noise = colored_projected_gaussian(
            noise,
            valid_mask,
            blend=colored_base[0],
            channel_scales=colored_base[1],
        )
    if base_noise_channel_scales is None:
        state, target, _, _ = residual_flow_pair(truth, noise, valid_mask, time)
    else:
        from .direct_dynamics_cascade_fine_matched_scale import matched_residual_flow_pair

        state, target, _, _ = matched_residual_flow_pair(
            truth, noise, valid_mask, time, base_noise_channel_scales
        )
    condition, _, _ = teacher_coarse_condition(truth, causal, valid_mask)
    grid = make_normalized_xy_grid(*config.image_size, device=device).expand(batch_size, -1, -1, -1)
    model_state = state
    if preconditioning_scales is not None:
        from .direct_dynamics_cascade_fine_preconditioned import (
            preconditioned_model_state,
            reconstruct_preconditioned_velocity,
        )

        target_residual_rms, projected_base_rms = preconditioning_scales
        model_state = preconditioned_model_state(
            state, time, target_residual_rms, projected_base_rms
        )
    model_input = torch.cat((model_state, grid, condition), dim=1)
    if model_input.shape[1] != FINE_INPUT_CHANNELS:
        raise ValueError("GPU smoke constructed the wrong fine-stage input")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        prediction = projected_model(model_input, time * 1000)[0]
    neural_prediction = prediction
    if preconditioning_scales is not None:
        prediction, neural_scale = reconstruct_preconditioned_velocity(
            neural_prediction,
            state,
            time,
            target_residual_rms,
            projected_base_rms,
        )
    mask = valid_mask.expand_as(prediction)
    error = prediction - target
    if preconditioning_scales is not None:
        error = error / neural_scale
    loss = (error.square() * mask).sum() / mask.sum().clamp(min=1)
    if not torch.isfinite(loss):
        raise FloatingPointError("fine-stage GPU smoke produced a non-finite loss")
    loss.backward()
    for parameter in projected_model.parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise FloatingPointError("fine-stage GPU smoke produced a non-finite gradient")
    coarse_velocity, coarse_fraction = masked_block_average(prediction, valid_mask)
    active = coarse_fraction > 0
    max_coarse_velocity = float(coarse_velocity[active].abs().max().detach())
    if max_coarse_velocity > 3e-6:
        raise RuntimeError("projected fine velocity escaped the coarse nullspace")
    result = {
        "status": "passed",
        "batch_size": batch_size,
        "model_input_shape": list(model_input.shape),
        "model_output_shape": list(prediction.shape),
        "loss": float(loss.detach()),
        "max_abs_coarse_velocity": max_coarse_velocity,
        "peak_gpu_memory_mib": float(torch.cuda.max_memory_allocated(device) / 2**20),
        "activation_checkpointing": config.activation_checkpointing,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "base_noise_channel_scales": (
            None if base_noise_channel_scales is None else list(base_noise_channel_scales)
        ),
        "colored_base": (
            None
            if colored_base is None
            else {"blend": colored_base[0], "channel_scales": list(colored_base[1])}
        ),
        "preconditioning": (
            None
            if preconditioning_scales is None
            else {
                "target_residual_rms": list(preconditioning_scales[0]),
                "projected_base_rms": list(preconditioning_scales[1]),
            }
        ),
    }
    for parameter in projected_model.parameters():
        parameter.grad = None
    projected_model.model.to("cpu")
    del projected_model, loss, prediction, model_input, state, target, condition, truth, raw
    torch.cuda.empty_cache()
    return result


def _run_impl(config_path: Path, *, preflight_only: bool, lifecycle: _Lifecycle) -> dict:
    experiment = load_json(config_path)
    config_dir = config_path.resolve().parent
    data_path = resolve_path(experiment["data_config"], config_dir)
    model_path = resolve_path(experiment["model_config"], config_dir)
    repo_root = Path(__file__).resolve().parents[1]
    code_identity = _clean_code_identity(repo_root)
    lifecycle.code_commit = code_identity["git_commit"]
    source_config_identity = {
        "experiment": {"path": str(config_path.resolve()), "sha256": _sha256_file(config_path)},
        "data": {"path": str(data_path.resolve()), "sha256": _sha256_file(data_path)},
        "method": {"path": str(model_path.resolve()), "sha256": _sha256_file(model_path)},
    }
    data_config = merge_config_overrides(load_json(data_path), experiment.get("data_overrides"))
    model_config = {**load_json(model_path), **experiment.get("training", {})}
    launch_id = os.environ.get("FINE_CASCADE_LAUNCH_ID", "").strip()
    if not launch_id or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", launch_id) is None:
        raise ValueError("FINE_CASCADE_LAUNCH_ID must be a safe non-empty identifier")
    lifecycle.run_id = launch_id
    _launch_status("python_preflight", run_id=launch_id, code_commit=code_identity["git_commit"])
    base_run_name = str(model_config.get("run_name") or "fine-cascade")
    model_config["run_name"] = f"{base_run_name}-{launch_id}"
    clearml = experiment.get("clearml", {})
    model_config.update(
        {
            "clearml_project_name": experiment["project_name"],
            "clearml_task_name": f"{experiment['task_name']}-{launch_id}",
            "clearml_enabled": bool(clearml.get("enabled", True)),
            "clearml_tags": clearml.get("tags", []),
            "clearml_env_path": clearml.get("env_path"),
            "clearml_upload_checkpoints": bool(clearml.get("upload_checkpoints", False)),
        }
    )
    config = TrainingConfig.from_dict(model_config)
    if not config.clearml_enabled:
        raise ValueError("bounded fine pilot requires online ClearML")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("fine cascade training requires exactly one visible GPU")

    pilot = experiment.get("pilot", {})
    if pilot.get("kind") not in {
        None,
        "mechanics_512",
        "compact_architecture_screen_512",
        "compact_undertraining_test_2048",
        "compact_full_data_one_epoch_6474",
        "compact_matched_base_scale_2048",
        "compact_variance_preconditioned_512",
        "compact_colored_preconditioned_512",
        "compact_colored_preconditioned_2048",
    }:
        raise ValueError("fine cascade experiment declares an unsupported pilot kind")
    train_case_count = int(pilot.get("train_case_count", 4096))
    validation_case_count = int(pilot.get("validation_case_count", 48))
    declared_updates = int(pilot.get("optimizer_updates", -1))
    if declared_updates not in {512, 2048, 6474}:
        raise ValueError("fine cascade experiment must declare 512, 2048, or 6474 optimizer updates")
    if pilot.get("kind") == "compact_colored_preconditioned_512" and declared_updates != 512:
        raise ValueError("colored 512 pilot must declare exactly 512 optimizer updates")
    if pilot.get("kind") == "compact_colored_preconditioned_2048" and declared_updates != 2048:
        raise ValueError("colored 2048 pilot must declare exactly 2048 optimizer updates")
    lifecycle.phase = "dataset_preflight"
    seed_everything(config.seed)
    train_dataset = build_dataset(data_config, split="train")
    validation_dataset = build_dataset(data_config, split="valid")
    for split, dataset in (("train", train_dataset), ("valid", validation_dataset)):
        if len(dataset) != EXPECTED_SPLIT_LENGTHS[split]:
            raise ValueError(
                f"{split} all-hour archive length differs: {len(dataset)} != {EXPECTED_SPLIT_LENGTHS[split]}"
            )
    sentinel = {
        "train": validate_direct_dataset(train_dataset),
        "valid": validate_direct_dataset(validation_dataset),
        "train_calendar": _calendar_inventory(train_dataset, split="train"),
        "valid_calendar": _calendar_inventory(validation_dataset, split="valid"),
    }
    train_indices = _evenly_spaced_indices(len(train_dataset), train_case_count)
    validation_indices = _evenly_spaced_indices(len(validation_dataset), validation_case_count)
    train_subset = Subset(train_dataset, train_indices)
    validation_subset = Subset(validation_dataset, validation_indices)
    ipc_preflight = _ipc_preflight(train_subset, validation_subset, config)
    train_loader = build_dataloader(
        train_subset,
        config.train_batch_size,
        config.num_workers_train,
        shuffle=True,
        collate_fn=_fine_collate,
        prefetch_factor=PREFETCH_FACTOR,
    )
    validation_loader = build_dataloader(
        validation_subset,
        config.eval_batch_size,
        config.num_workers_val,
        shuffle=False,
        collate_fn=_fine_collate,
        prefetch_factor=PREFETCH_FACTOR,
    )
    planned_updates = len(train_loader) * config.num_epochs
    if planned_updates != declared_updates:
        raise ValueError(
            f"fine cascade planned {planned_updates} updates but declared {declared_updates}"
        )
    subset_provenance = {
        "selection": "deterministic_even_spacing_over_full_all-hour_split",
        "train_case_count": train_case_count,
        "train_indices_sha256": _indices_sha256(train_indices),
        "train_indices": train_indices,
        "train_case_ids": [_case_id(train_dataset, index) for index in train_indices],
        "validation_case_count": validation_case_count,
        "validation_indices_sha256": _indices_sha256(validation_indices),
        "validation_indices": validation_indices,
        "validation_case_ids": [_case_id(validation_dataset, index) for index in validation_indices],
        "planned_optimizer_updates": planned_updates,
    }
    sentinel["pilot_subset"] = subset_provenance
    sentinel["ipc_preflight"] = ipc_preflight
    sentinel["source_config_identity"] = source_config_identity
    sentinel["effective_config_sha256"] = {
        "data": _canonical_sha256(data_config),
        "model": _canonical_sha256(model_config),
        "experiment": _canonical_sha256(experiment),
    }
    sentinel["code_identity"] = code_identity
    output_root = Path(config.base_output_dir) / config.run_name
    lifecycle.output_root = output_root
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"refusing to reuse fine cascade output: {output_root}")
    lifecycle.output_writable = True

    model = build_unet(config)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    smoke_batch = _fine_collate([train_subset[index] for index in range(config.train_batch_size)])
    lifecycle.phase = "gpu_smoke"
    trainer_class = FineCascadeDynamicsTrainer
    base_noise_channel_scales = None
    preconditioning_scales = None
    colored_base = None
    if pilot.get("kind") == "compact_matched_base_scale_2048":
        from .direct_dynamics_cascade_fine_matched_scale import (
            MatchedScaleFineCascadeDynamicsTrainer,
            validate_channel_scales,
        )

        law = experiment.get("fine_base_law")
        if not isinstance(law, dict):
            raise ValueError("matched-scale pilot requires fine_base_law")
        base_noise_channel_scales = validate_channel_scales(law.get("channel_scales"))
        trainer_class = MatchedScaleFineCascadeDynamicsTrainer
    elif pilot.get("kind") == "compact_variance_preconditioned_512":
        from .direct_dynamics_cascade_fine_preconditioned import (
            VariancePreconditionedFineCascadeDynamicsTrainer,
            validate_preconditioning_contract,
        )

        contract = validate_preconditioning_contract(experiment.get("fine_preconditioning"))
        preconditioning_scales = (
            tuple(contract["target_residual_rms"]),
            tuple(contract["projected_base_rms"]),
        )
        trainer_class = VariancePreconditionedFineCascadeDynamicsTrainer
    elif pilot.get("kind") in {
        "compact_colored_preconditioned_512",
        "compact_colored_preconditioned_2048",
    }:
        from .direct_dynamics_cascade_fine_colored import (
            ColoredVariancePreconditionedFineCascadeDynamicsTrainer,
            validate_colored_base_contract,
        )
        from .direct_dynamics_cascade_fine_preconditioned import (
            validate_preconditioning_contract,
        )

        precondition = validate_preconditioning_contract(experiment.get("fine_preconditioning"))
        color = validate_colored_base_contract(experiment.get("fine_colored_base"))
        preconditioning_scales = (
            tuple(precondition["target_residual_rms"]),
            tuple(precondition["projected_base_rms"]),
        )
        colored_base = (float(color["blend"]), tuple(color["channel_scales"]))
        trainer_class = ColoredVariancePreconditionedFineCascadeDynamicsTrainer
    smoke = _gpu_admission_smoke(
        model,
        smoke_batch,
        config,
        base_noise_channel_scales=base_noise_channel_scales,
        preconditioning_scales=preconditioning_scales,
        colored_base=colored_base,
    )
    result = {
        "status": "preflight_passed" if preflight_only else "training_pending",
        "parameter_count": parameter_count,
        "steps_per_epoch": len(train_loader),
        "planned_optimizer_updates": planned_updates,
        "dataset_sentinel": sentinel,
        "gpu_batch_smoke": smoke,
    }
    if preflight_only:
        result["executed_optimizer_updates"] = 0
        preflight_evidence = {
            **result,
            "status": "preflight_passed",
            "run_id": launch_id,
            "code_identity": code_identity,
            "source_config_identity": source_config_identity,
            "effective_config_sha256": sentinel["effective_config_sha256"],
        }
        preflight_path = _write_preflight_evidence(preflight_evidence)
        lifecycle.phase = "preflight_complete"
        _launch_status(
            "preflight_passed",
            run_id=launch_id,
            code_commit=code_identity["git_commit"],
            candidate_optimizer_updates=planned_updates,
            executed_optimizer_updates=0,
            gpu_smoke_status=smoke["status"],
            preflight_evidence=None if preflight_path is None else str(preflight_path),
        )
        return result

    _atomic_json(output_root / "fine_cascade_dataset_sentinel.json", sentinel)
    _atomic_json(output_root / "fine_cascade_gpu_smoke.json", smoke)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    from diffusers.optimization import get_cosine_schedule_with_warmup

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=planned_updates,
    )
    lifecycle.phase = "clearml_initialization"
    trainer = _initialize_trainer(
        lifecycle,
        trainer_class=trainer_class,
        config=config,
        fine_code_identity=code_identity,
        model=model,
        optimizer=optimizer,
        data_loader_train=train_loader,
        data_loader_val=validation_loader,
        lr_scheduler=scheduler,
        add_noise_func=add_noise,
        experiment_config=experiment,
        model_config=model_config,
        data_config=data_config,
        dataset_provenance={
            "train": train_dataset.provenance(),
            "valid": validation_dataset.provenance(),
            "pilot_subset": subset_provenance,
        },
        dashboard_dataset=None,
    )
    _launch_status(
        "training",
        run_id=launch_id,
        code_commit=code_identity["git_commit"],
        output_dir=str(output_root),
        planned_optimizer_updates=planned_updates,
    )
    lifecycle.phase = "training_or_diagnostics"
    output_dir = trainer.train_loop()
    lifecycle.phase = "terminal_finalization"
    result.update(
        {
            "status": "complete_pending_generated_coarse_e2e_and_independent_review",
            "output_dir": str(Path(output_dir).resolve()),
            "executed_optimizer_updates": planned_updates,
        }
    )
    _atomic_json(Path(output_dir) / "fine_cascade_training_completion.json", result)
    _launch_status(
        result["status"],
        run_id=launch_id,
        code_commit=code_identity["git_commit"],
        output_dir=str(Path(output_dir).resolve()),
        executed_optimizer_updates=planned_updates,
    )
    return result


def run(config_path: Path, *, preflight_only: bool = False) -> dict:
    lifecycle = _Lifecycle()
    try:
        return _run_impl(config_path, preflight_only=preflight_only, lifecycle=lifecycle)
    except BaseException as error:
        _record_failure(lifecycle, error)
        raise


def main() -> None:
    def _terminate(signum, _frame):
        raise TimeoutError(f"fine cascade received termination signal {signum}")

    signal.signal(signal.SIGTERM, _terminate)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.config.resolve(), preflight_only=arguments.preflight_only), indent=2))


if __name__ == "__main__":
    main()
