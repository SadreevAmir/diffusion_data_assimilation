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
    CoarseLearningCurveEarlyStop,
    coarse_flow_pair,
    coarse_model_input_for_test,
    coarse_target,
    ocean_fraction_weighted_mse,
)
from .direct_dynamics_cascade_coarse_residual import (
    CoarsePersistenceResidualTrainer,
    CoarseStandardizedPersistenceResidualTrainer,
    coarse_residual_flow_pair,
    coarse_standardized_residual_flow_pair,
    residual_model_input_for_test,
)
from .direct_dynamics_cascade_coarse_mean import (
    CoarseConditionalMeanTrainer,
    coarse_mean_model_input,
    coarse_mean_target,
)
from .direct_dynamics_cascade_residual_stats import _tensor_sha256
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


def _gpu_admission_smoke(
    model,
    raw: dict,
    config: TrainingConfig,
    *,
    persistence_residual: bool = False,
    residual_statistics: dict[str, Any] | None = None,
    deterministic_mean: bool = False,
) -> dict:
    """Exercise the configured coarse path with a real CUDA backward pass."""
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    model = model.to(device).train()
    truth = raw["truth"].to(device=device, dtype=torch.float32)
    valid = raw["valid_mask"][:, :1].to(device=device, dtype=torch.float32)
    condition = raw["structured_conditioning"].to(device=device, dtype=torch.float32)
    time = (
        torch.zeros(truth.shape[0], device=device)
        if deterministic_mean
        else torch.linspace(0.05, 0.95, truth.shape[0], device=device)
    )
    coarse_height, coarse_width = config.image_size
    noise = torch.randn(
        (truth.shape[0], 6, coarse_height, coarse_width),
        device=device,
        dtype=torch.float32,
    )
    if deterministic_mean:
        target, persistence, _, fraction = coarse_mean_target(
            truth, condition, valid
        )
        background, _, _ = coarse_target(raw["background"].to(device), valid)
        if not torch.equal(persistence, background):
            raise ValueError("mean smoke persistence differs from dataset baseline")
        state = torch.zeros_like(target)
        model_input, _, _ = coarse_mean_model_input(condition, valid)
        state_coordinate = "deterministic_conditional_mean"
    elif persistence_residual:
        if residual_statistics is None:
            state, target, _, _, fraction = coarse_residual_flow_pair(
                truth, noise, condition, valid, time
            )
            state_coordinate = "persistence_residual"
        else:
            state, target, _, _, fraction = coarse_standardized_residual_flow_pair(
                truth, noise, condition, valid, time, residual_statistics
            )
            state_coordinate = "standardized_persistence_residual"
        model_input = residual_model_input_for_test(state, condition, valid)
    else:
        state, target, _, fraction = coarse_flow_pair(truth, noise, valid, time)
        model_input = coarse_model_input_for_test(state, condition, valid)
        state_coordinate = "absolute_coarse"
    if tuple(model_input.shape[1:]) != (
        config.in_channels,
        coarse_height,
        coarse_width,
    ):
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
        "state_coordinate": state_coordinate,
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
        "mean_module": _sha256_file(
            Path(__file__).with_name("direct_dynamics_cascade_coarse_mean.py")
        ),
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
    pilot_kind = str(pilot.get("kind", "mechanics_512"))
    expected_updates = {
        "mechanics_512": 512,
        "compact_architecture_screen_512": 512,
        "learning_curve_2048": 2048,
        "learning_curve_4096": 4096,
        "full_train_6_epochs": 19422,
        "persistence_residual_2048": 2048,
        "standardized_persistence_residual_2048": 2048,
        "deterministic_mean_2048": 2048,
    }.get(pilot_kind)
    expected_train_cases = 51_792 if pilot_kind == "full_train_6_epochs" else 4096
    if expected_updates is None or (
        train_case_count,
        validation_case_count,
        int(pilot.get("optimizer_updates", -1)),
    ) != (expected_train_cases, 48, expected_updates):
        raise ValueError(
            "coarse pilot must declare a supported exact bounded protocol"
        )

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
    train_static_valid_mask_sha256 = _tensor_sha256(train_subset[0]["valid_mask"][:1])
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
    if planned_updates != expected_updates:
        raise ValueError(f"coarse pilot must plan {expected_updates} updates, got {planned_updates}")
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
    persistence_residual = pilot_kind in {
        "persistence_residual_2048",
        "standardized_persistence_residual_2048",
    }
    standardized_residual = pilot_kind == "standardized_persistence_residual_2048"
    deterministic_mean = pilot_kind == "deterministic_mean_2048"
    residual_statistics_binding = None
    residual_statistics = None
    if standardized_residual:
        residual_statistics_binding = experiment.get("residual_statistics")
        if not isinstance(residual_statistics_binding, dict):
            raise ValueError("standardized residual pilot requires residual_statistics")
        statistics_path = Path(str(residual_statistics_binding.get("path", "")))
        statistics_sha256 = str(residual_statistics_binding.get("sha256", ""))
        if not statistics_path.is_absolute() or statistics_path.name != "statistics.json":
            raise ValueError("residual statistics path must be an absolute statistics.json")
        if not statistics_path.is_file() or _sha256_file(statistics_path) != statistics_sha256:
            raise ValueError("residual statistics artifact is missing or differs")
        residual_statistics = json.loads(statistics_path.read_text(encoding="utf-8"))
        expected_statistics_binding = {
            "status": "complete",
            "split": "train",
            "train_case_count": train_case_count,
            "train_indices_sha256": subset_provenance["train_indices_sha256"],
            "selected_case_ids": subset_provenance["train_case_ids"],
            "ordered_inventory_sha256": sentinel["train_calendar"][
                "ordered_inventory_sha256"
            ],
            "static_valid_mask_sha256": train_static_valid_mask_sha256,
            "data_config_sha256": _canonical_sha256(data_config),
        }
        for key, expected in expected_statistics_binding.items():
            if residual_statistics.get(key) != expected:
                raise ValueError(f"residual statistics binding differs for {key}")
    smoke = _gpu_admission_smoke(
        model,
        smoke_batch,
        config,
        persistence_residual=persistence_residual,
        residual_statistics=residual_statistics,
        deterministic_mean=deterministic_mean,
    )
    result = {
        "status": "preflight_passed" if preflight_only else "training_pending",
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "planned_optimizer_updates": planned_updates,
        "dataset_sentinel": sentinel,
        "gpu_batch_smoke": smoke,
    }
    if preflight_only:
        result["executed_optimizer_updates"] = 0
        lifecycle.phase = "preflight_complete"
        _launch_status(
            "preflight_passed",
            run_id=launch_id,
            code_commit=code_identity["git_commit"],
            candidate_optimizer_updates=planned_updates,
            executed_optimizer_updates=0,
            gpu_smoke_status=smoke["status"],
        )
        return result
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    from diffusers.optimization import get_cosine_schedule_with_warmup

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=planned_updates,
    )
    lifecycle.phase = "clearml_initialization"
    if deterministic_mean:
        trainer_class = CoarseConditionalMeanTrainer
    elif standardized_residual:
        trainer_class = CoarseStandardizedPersistenceResidualTrainer
    elif persistence_residual:
        trainer_class = CoarsePersistenceResidualTrainer
    else:
        trainer_class = CoarseCascadeDynamicsTrainer
    trainer = trainer_class.__new__(trainer_class)
    lifecycle.trainer = trainer
    trainer_kwargs = {}
    if persistence_residual:
        baseline = experiment.get("residual_baseline")
        if not isinstance(baseline, dict):
            raise ValueError("persistence-residual pilot requires residual_baseline")
        trainer_kwargs["residual_baseline"] = baseline
    if standardized_residual:
        trainer_kwargs["residual_statistics"] = residual_statistics_binding
    trainer_class.__init__(
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
            "train_calendar": sentinel["train_calendar"],
            "static_valid_mask_sha256": train_static_valid_mask_sha256,
            "pilot_subset": subset_provenance,
        },
        dashboard_dataset=None,
        coarse_diagnostic_batch=diagnostic_batch,
        coarse_code_identity=code_identity,
        **trainer_kwargs,
    )
    _atomic_json(output_root / "coarse_cascade_dataset_sentinel.json", sentinel)
    _atomic_json(output_root / "coarse_cascade_gpu_smoke.json", smoke)
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
    early_stop = None
    try:
        output_dir = trainer.train_loop()
    except CoarseLearningCurveEarlyStop as error:
        early_stop = str(error)
        output_dir = trainer.output_dir
        if trainer.clearml is not None:
            trainer.clearml.report_single_value("coarse_learning_curve_early_stop", 1.0)
            trainer.clearml.close()
        trainer.accelerator.end_training()
    lifecycle.phase = "terminal_finalization"
    gate_path = Path(output_dir) / "coarse_mechanics_gate.json"
    if not gate_path.is_file():
        raise FileNotFoundError("coarse mechanics completion lacks its terminal gate")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    completion_status = (
        "stopped_pending_independent_review"
        if early_stop is not None
        else "complete_pending_independent_review"
    )
    result.update(
        {
            "status": completion_status,
            "output_dir": str(Path(output_dir).resolve()),
            "mechanics_gate": gate,
            "early_stop_reason": early_stop,
            "executed_optimizer_updates": int(gate["optimizer_updates"]),
        }
    )
    _atomic_json(Path(output_dir) / "coarse_cascade_training_completion.json", result)
    _launch_status(
        completion_status,
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
