"""Bounded server runner for the audited 160x128 coarse cascade mechanics pilot."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Subset

from .config import TrainingConfig, load_json, merge_config_overrides, resolve_path
from .data import build_dataset
from .direct_dynamics_cascade_coarse import (
    COARSE_INPUT_CHANNELS,
    CoarseCascadeDynamicsTrainer,
    coarse_flow_pair,
    coarse_model_input_for_test,
    ocean_fraction_weighted_mse,
)
from .direct_dynamics_cascade_fine_training import (
    EXPECTED_SPLIT_LENGTHS,
    PREFETCH_FACTOR,
    _calendar_inventory,
    _canonical_sha256,
    _clean_code_identity,
    _evenly_spaced_indices,
    _fine_collate,
    _indices_sha256,
    _ipc_preflight,
    _sha256_file,
)
from .direct_dynamics_training import validate_direct_dataset
from .model_io import build_unet
from .runtime import add_noise, build_dataloader, seed_everything
from .trainer import _atomic_json


@dataclass
class _Lifecycle:
    run_id: str = "unknown"
    code_commit: str = "unknown"
    phase: str = "configuration"
    output_root: Path | None = None
    output_writable: bool = False
    trainer: Any = None


def _record_failure(lifecycle: _Lifecycle, error: BaseException) -> None:
    """Best-effort failure containment which never replaces the primary error."""
    failure = {
        "status": "failed",
        "phase": lifecycle.phase,
        "error_type": type(error).__name__,
        "error": str(error),
        "code_commit": lifecycle.code_commit,
        "run_id": lifecycle.run_id,
        "output_dir": None if lifecycle.output_root is None else str(lifecycle.output_root),
    }
    cleanup_errors = []
    failure["cleanup_errors"] = cleanup_errors
    if lifecycle.output_writable and lifecycle.output_root is not None:
        try:
            _atomic_json(lifecycle.output_root / "coarse_cascade_failure.json", failure)
        except Exception as cleanup_error:
            cleanup_errors.append(f"initial_failure_artifact: {cleanup_error}")
    try:
        _launch_status(**failure)
    except Exception as cleanup_error:
        cleanup_errors.append(f"initial_failure_status: {cleanup_error}")

    trainer = lifecycle.trainer
    tracker = getattr(trainer, "clearml", None)
    if tracker is not None:
        try:
            tracker.task.mark_failed(status_reason=type(error).__name__, status_message=str(error)[:1000])
        except Exception as cleanup_error:
            cleanup_errors.append(f"clearml_mark_failed: {cleanup_error}")
        try:
            tracker.close()
        except Exception as cleanup_error:
            cleanup_errors.append(f"clearml_close: {cleanup_error}")
    accelerator = getattr(trainer, "accelerator", None)
    if accelerator is not None:
        try:
            accelerator.end_training()
        except Exception as cleanup_error:
            cleanup_errors.append(f"accelerator_end_training: {cleanup_error}")
    if lifecycle.output_writable and lifecycle.output_root is not None:
        try:
            _atomic_json(lifecycle.output_root / "coarse_cascade_failure.json", failure)
        except Exception as cleanup_error:
            cleanup_errors.append(f"failure_artifact: {cleanup_error}")
    try:
        _launch_status(**failure)
    except Exception as cleanup_error:
        cleanup_errors.append(f"failure_status: {cleanup_error}")
    if lifecycle.output_writable and lifecycle.output_root is not None and cleanup_errors:
        try:
            _atomic_json(lifecycle.output_root / "coarse_cascade_failure.json", failure)
        except Exception:
            pass


def _launch_status(status: str, **details) -> None:
    raw_path = os.environ.get("COARSE_CASCADE_STATUS_PATH", "").strip()
    if not raw_path:
        return
    path = Path(raw_path)
    if not path.is_absolute() or path.name != "status.json":
        raise ValueError("COARSE_CASCADE_STATUS_PATH must be an absolute status.json path")
    _atomic_json(path, {"status": status, **details})


def _gpu_admission_smoke(model, raw: dict, config: TrainingConfig) -> dict:
    """Exercise the exact 56-to-6 coarse path with a real backward pass."""
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    model = model.to(device).train()
    truth = raw["truth"].to(device=device, dtype=torch.float32)
    valid = raw["valid_mask"][:, :1].to(device=device, dtype=torch.float32)
    condition = raw["structured_conditioning"].to(device=device, dtype=torch.float32)
    time = torch.linspace(0.05, 0.95, truth.shape[0], device=device)
    coarse_height, coarse_width = config.image_size
    noise = torch.randn(
        (truth.shape[0], 6, coarse_height, coarse_width),
        device=device,
        dtype=torch.float32,
    )
    state, target, _, fraction = coarse_flow_pair(truth, noise, valid, time)
    model_input = coarse_model_input_for_test(state, condition, valid)
    if tuple(model_input.shape[1:]) != (COARSE_INPUT_CHANNELS, coarse_height, coarse_width):
        raise ValueError("GPU smoke constructed the wrong coarse model input")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        prediction = model(model_input, time * 1000, return_dict=False)[0]
    loss = ocean_fraction_weighted_mse(prediction.float(), target.float(), fraction.float())
    loss.backward()
    if not torch.isfinite(loss):
        raise FloatingPointError("coarse GPU smoke produced a non-finite loss")
    for parameter in model.parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise FloatingPointError("coarse GPU smoke produced a non-finite gradient")
    result = {
        "status": "passed",
        "batch_size": truth.shape[0],
        "model_input_shape": list(model_input.shape),
        "model_output_shape": list(prediction.shape),
        "loss": float(loss.detach()),
        "peak_gpu_memory_mib": float(torch.cuda.max_memory_allocated(device) / 2**20),
        "activation_checkpointing": config.activation_checkpointing,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "state_dtype": str(state.dtype),
        "loss_dtype": str(loss.dtype),
    }
    for parameter in model.parameters():
        parameter.grad = None
    model.to("cpu")
    del model, prediction, loss, model_input, state, target, truth, raw
    torch.cuda.empty_cache()
    return result


def _seasonal_diagnostic_batch(validation_subset: Subset) -> tuple[dict, list[int]]:
    if len(validation_subset) != 48:
        raise ValueError("seasonal diagnostic selection requires the declared 48-case validation subset")
    positions = [4, 16, 28, 40]
    batch = _fine_collate([validation_subset[position] for position in positions])
    case_ids = batch["meta"]["case_id"]
    months = {int(case_id[5:7]) for case_id in case_ids}
    if len(months) != len(case_ids):
        raise ValueError("seasonal diagnostic cases must occupy distinct calendar months")
    return batch, positions


def _run_impl(config_path: Path, *, preflight_only: bool, lifecycle: _Lifecycle) -> dict:
    experiment = load_json(config_path)
    config_dir = config_path.resolve().parent
    data_path = resolve_path(experiment["data_config"], config_dir)
    model_path = resolve_path(experiment["model_config"], config_dir)
    repo_root = Path(__file__).resolve().parents[1]
    code_identity = _clean_code_identity(repo_root)
    lifecycle.code_commit = code_identity["git_commit"]
    source_identity = {
        "experiment": {"path": str(config_path.resolve()), "sha256": _sha256_file(config_path)},
        "data": {"path": str(data_path.resolve()), "sha256": _sha256_file(data_path)},
        "method": {"path": str(model_path.resolve()), "sha256": _sha256_file(model_path)},
        "coarse_module": _sha256_file(Path(__file__).with_name("direct_dynamics_cascade_coarse.py")),
        "runner": _sha256_file(Path(__file__)),
    }
    data_config = merge_config_overrides(load_json(data_path), experiment.get("data_overrides"))
    model_config = {**load_json(model_path), **experiment.get("training", {})}
    launch_id = os.environ.get("COARSE_CASCADE_LAUNCH_ID", "").strip()
    if not launch_id or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", launch_id) is None:
        raise ValueError("COARSE_CASCADE_LAUNCH_ID must be a safe non-empty identifier")
    lifecycle.run_id = launch_id
    _launch_status("python_preflight", run_id=launch_id, code_commit=code_identity["git_commit"])
    model_config["run_name"] = f"{model_config.get('run_name', 'coarse-cascade')}-{launch_id}"
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
        raise ValueError("bounded coarse pilot requires online ClearML")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("coarse cascade training requires exactly one visible GPU")
    pilot = experiment.get("pilot", {})
    train_case_count = int(pilot.get("train_case_count", -1))
    validation_case_count = int(pilot.get("validation_case_count", -1))
    if (train_case_count, validation_case_count, int(pilot.get("optimizer_updates", -1))) != (
        4096,
        48,
        512,
    ):
        raise ValueError("coarse mechanics pilot requires exactly 4096 train, 48 validation, 512 updates")

    output_root = Path(config.base_output_dir) / config.run_name
    lifecycle.output_root = output_root
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"refusing to reuse coarse cascade output: {output_root}")
    lifecycle.output_writable = True

    lifecycle.phase = "dataset_preflight"
    seed_everything(config.seed)
    train_dataset = build_dataset(data_config, split="train")
    validation_dataset = build_dataset(data_config, split="valid")
    for split, dataset in (("train", train_dataset), ("valid", validation_dataset)):
        if len(dataset) != EXPECTED_SPLIT_LENGTHS[split]:
            raise ValueError(f"{split} archive length differs from its audited inventory")
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
    diagnostic_batch, diagnostic_positions = _seasonal_diagnostic_batch(validation_subset)
    ipc = _ipc_preflight(train_subset, validation_subset, config)
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
    if planned_updates != 512:
        raise ValueError(f"coarse mechanics pilot must plan 512 updates, got {planned_updates}")
    subset_provenance = {
        "selection": "deterministic_even_spacing_over_full_all-hour_split",
        "train_indices": train_indices,
        "train_indices_sha256": _indices_sha256(train_indices),
        "train_case_ids": [_case_id(train_dataset, index) for index in train_indices],
        "validation_indices": validation_indices,
        "validation_indices_sha256": _indices_sha256(validation_indices),
        "validation_case_ids": [_case_id(validation_dataset, index) for index in validation_indices],
        "diagnostic_subset_positions": diagnostic_positions,
        "diagnostic_case_ids": diagnostic_batch["meta"]["case_id"],
        "planned_optimizer_updates": planned_updates,
    }
    sentinel.update(
        {
            "pilot_subset": subset_provenance,
            "ipc_preflight": ipc,
            "source_identity": source_identity,
            "effective_config_sha256": {
                "data": _canonical_sha256(data_config),
                "model": _canonical_sha256(model_config),
                "experiment": _canonical_sha256(experiment),
            },
            "code_identity": code_identity,
        }
    )
    model = build_unet(config)
    smoke_batch = _fine_collate([train_subset[index] for index in range(config.train_batch_size)])
    lifecycle.phase = "gpu_smoke"
    smoke = _gpu_admission_smoke(model, smoke_batch, config)
    result = {
        "status": "preflight_passed" if preflight_only else "training_pending",
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "planned_optimizer_updates": planned_updates,
        "dataset_sentinel": sentinel,
        "gpu_batch_smoke": smoke,
    }
    if preflight_only:
        return result
    _atomic_json(output_root / "coarse_cascade_dataset_sentinel.json", sentinel)
    _atomic_json(output_root / "coarse_cascade_gpu_smoke.json", smoke)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    from diffusers.optimization import get_cosine_schedule_with_warmup

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=planned_updates,
    )
    lifecycle.phase = "clearml_initialization"
    trainer = CoarseCascadeDynamicsTrainer.__new__(CoarseCascadeDynamicsTrainer)
    lifecycle.trainer = trainer
    CoarseCascadeDynamicsTrainer.__init__(
        trainer,
        config=config,
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
        coarse_diagnostic_batch=diagnostic_batch,
        coarse_code_identity=code_identity,
    )
    _launch_status(
        "clearml_online",
        run_id=launch_id,
        code_commit=code_identity["git_commit"],
        output_dir=str(output_root),
    )
    lifecycle.phase = "training_or_diagnostics"
    _launch_status(
        "training",
        run_id=launch_id,
        code_commit=code_identity["git_commit"],
        output_dir=str(output_root),
        planned_optimizer_updates=planned_updates,
    )
    output_dir = trainer.train_loop()
    lifecycle.phase = "terminal_finalization"
    gate_path = Path(output_dir) / "coarse_mechanics_gate.json"
    if not gate_path.is_file():
        raise FileNotFoundError("coarse mechanics completion lacks its terminal gate")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    result.update(
        {
            "status": "complete_pending_independent_review",
            "output_dir": str(Path(output_dir).resolve()),
            "mechanics_gate": gate,
        }
    )
    _atomic_json(Path(output_dir) / "coarse_cascade_training_completion.json", result)
    _launch_status(
        "complete_pending_independent_review",
        run_id=launch_id,
        code_commit=code_identity["git_commit"],
        output_dir=str(Path(output_dir).resolve()),
        mechanics_gate_status=gate.get("status"),
        mechanics_gate_decision=gate.get("decision"),
    )
    return result


def run(config_path: Path, *, preflight_only: bool = False) -> dict:
    lifecycle = _Lifecycle()
    try:
        return _run_impl(config_path, preflight_only=preflight_only, lifecycle=lifecycle)
    except BaseException as error:
        _record_failure(lifecycle, error)
        raise


def _case_id(dataset, index: int) -> str:
    day_index, archive_slice = divmod(index, 24)
    target = dataset.calendar_pairs[day_index][1]
    return f"{target.date.isoformat()}_slice{archive_slice:02d}"


def main() -> None:
    def _terminate(signum, _frame):
        raise TimeoutError(f"coarse cascade received termination signal {signum}")

    signal.signal(signal.SIGTERM, _terminate)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.config.resolve(), preflight_only=arguments.preflight_only), indent=2))


if __name__ == "__main__":
    main()
