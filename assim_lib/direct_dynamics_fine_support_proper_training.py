"""Bounded 64-update support-aware refinement of the terminal fine interval."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import signal
import time
from pathlib import Path
from typing import Any

import torch

from .clearml_tracking import ClearMLTracker
from .config import load_json, merge_config_overrides, resolve_path
from .data import build_dataset
from .direct_dynamics_cascade import masked_block_average
from .direct_dynamics_cascade_coarse_fine_boundary_audit import (
    _sha256,
    _strict_atomic_json,
    _terminate,
)
from .direct_dynamics_cascade_contract import (
    forecast_contract_from_data_config,
    forecast_contract_sha256,
)
from .direct_dynamics_cascade_coarse_proper_refinement import _atomic_torch_save
from .direct_dynamics_cascade_fine import teacher_coarse_condition
from .direct_dynamics_cascade_fine_colored import (
    load_colored_variance_preconditioned_fine_cascade_sampler,
)
from .direct_dynamics_cascade_fine_training import (
    EXPECTED_SPLIT_LENGTHS,
    _calendar_inventory,
    _canonical_sha256,
    _clean_code_identity,
    _evenly_spaced_indices,
    _sha256_file,
)
from .direct_dynamics_fine_support_proper_admission import (
    support_aware_physical_objective,
)
from .direct_dynamics_fine_support_proper_gpu_preflight import (
    EXPECTED_PROTOCOL as PREFLIGHT_PROTOCOL,
    _load_normalization_binding,
    _validate_source_files,
    canonical_truth_and_teacher_coarse,
)
from .direct_dynamics_fine_support_proper_integration import (
    _repeat_members,
    fine_frozen_prefix,
    terminal_fine_residual,
)
from .direct_dynamics_training import validate_direct_dataset
from .runtime import make_normalized_xy_grid, seed_everything


EXPECTED_PROTOCOL = {
    "split": "train",
    "updates": 64,
    "batch_size": 1,
    "members": 4,
    "rk4_intervals": 32,
    "frozen_prefix_intervals": 31,
    "trainable_terminal_intervals": 1,
    "learning_rate": 1e-5,
    "weight_decay": 0.0,
    "network_precision": "bf16",
    "state_precision": "fp32",
    "score_precision": "fp64",
    "crps_weight": 0.75,
    "joint_energy_weight": 0.25,
    "gradient_clipping": False,
    "activation_checkpointing": False,
    "additional_ema": False,
    "noise_schedule": "deterministic_seed_plus_update",
    "seed": 74831,
    "test_2023": "closed",
    "max_gpu_minutes": 30,
}


def _load_bound_preflight(
    config: dict[str, Any], repo: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_spec = config["preflight_config"]
    source_path = (
        repo / source_spec["path"]
        if source_spec.get("repo_relative")
        else Path(source_spec["path"])
    )
    if not source_path.is_file() or _sha256(source_path) != source_spec["sha256"]:
        raise ValueError("bound preflight config SHA mismatch")
    preflight = load_json(source_path)
    if (
        preflight.get("schema_version")
        != "fine_support_proper_actual_gpu_preflight_v1"
        or preflight.get("protocol") != PREFLIGHT_PROTOCOL
    ):
        raise ValueError("bound preflight config differs from reviewed contract")
    result_spec = config["preflight_result"]
    status_path = Path(result_spec["status_path"])
    metrics_path = Path(result_spec["metrics_path"])
    if not status_path.is_file() or _sha256(status_path) != result_spec["status_sha256"]:
        raise ValueError("bound preflight terminal status SHA mismatch")
    if not metrics_path.is_file() or _sha256(metrics_path) != result_spec["metrics_sha256"]:
        raise ValueError("bound preflight metrics SHA mismatch")
    status = load_json(status_path)
    metrics = load_json(metrics_path)
    if (
        status.get("status") != "complete"
        or status.get("code_identity", {}).get("git_commit") != result_spec["code_commit"]
        or status.get("clearml_task_id") != result_spec["clearml_task_id"]
        or status.get("metrics_sha256") != result_spec["metrics_sha256"]
        or metrics.get("status") != "preflight_passed_pending_astra_review"
        or metrics.get("code_identity", {}).get("git_commit") != result_spec["code_commit"]
        or metrics.get("training_performed") is not False
        or metrics.get("optimizer_steps") != 0
        or metrics.get("test_2023_used") is not False
        or metrics.get("step0", {}).get("production_replay_max_abs") != 0.0
        or metrics.get("step0", {}).get("candidate_control_max_abs") != 0.0
    ):
        raise ValueError("bound preflight result differs from reviewed PASS")
    return preflight, {
        "config_path": str(source_path),
        "config_sha256": source_spec["sha256"],
        "status_path": str(status_path),
        "status_sha256": result_spec["status_sha256"],
        "metrics_path": str(metrics_path),
        "metrics_sha256": result_spec["metrics_sha256"],
        "code_commit": result_spec["code_commit"],
        "clearml_task_id": result_spec["clearml_task_id"],
    }


def _resume_payload(
    terminal_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    completed_updates: int,
    history: list[dict[str, Any]],
    config_sha256: str,
    code_identity: dict[str, str],
    source_checkpoint_sha256: str,
    training_started_unix: float,
    deadline_unix: float,
) -> dict[str, Any]:
    return {
        "schema_version": "fine_support_proper_resume_v1",
        "completed_updates": completed_updates,
        "optimizer_steps": completed_updates,
        "history": history,
        "config_sha256": config_sha256,
        "code_identity": code_identity,
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "training_started_unix": training_started_unix,
        "deadline_unix": deadline_unix,
        "protocol": EXPECTED_PROTOCOL,
        "terminal_model_state": terminal_model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }


def _restore_resume(
    path: Path,
    terminal_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    config_sha256: str,
    code_identity: dict[str, str],
    source_checkpoint_sha256: str,
    expected_sha256: str,
    expected_completed_updates: int,
    expected_indices: list[int],
    training_started_unix: float,
    deadline_unix: float,
    device: torch.device,
) -> tuple[int, list[dict[str, Any]], str]:
    expected_sha = _sha256(path)
    if expected_sha != expected_sha256:
        raise ValueError("resume checkpoint SHA differs from durable status binding")
    payload = torch.load(path, map_location=device, weights_only=True)
    if (
        payload.get("schema_version") != "fine_support_proper_resume_v1"
        or payload.get("config_sha256") != config_sha256
        or payload.get("code_identity") != code_identity
        or payload.get("source_checkpoint_sha256") != source_checkpoint_sha256
        or payload.get("training_started_unix") != training_started_unix
        or payload.get("deadline_unix") != deadline_unix
        or payload.get("protocol") != EXPECTED_PROTOCOL
    ):
        raise ValueError("resume checkpoint identity differs from current training contract")
    completed = int(payload.get("completed_updates", -1))
    if completed < 1 or completed > EXPECTED_PROTOCOL["updates"]:
        raise ValueError("resume checkpoint has invalid completed update count")
    if completed != expected_completed_updates:
        raise ValueError("resume checkpoint update count differs from durable status")
    if payload.get("optimizer_steps") != completed:
        raise ValueError("resume optimizer accounting differs from completed updates")
    history = payload.get("history")
    if not isinstance(history, list) or len(history) != completed:
        raise ValueError("resume history differs from completed updates")
    for position, row in enumerate(history, start=1):
        if (
            not isinstance(row, dict)
            or row.get("update") != position
            or row.get("train_index") != int(expected_indices[position - 1])
            or row.get("noise_seed") != EXPECTED_PROTOCOL["seed"] + position
        ):
            raise ValueError("resume history sequence differs from deterministic schedule")
    terminal_model.load_state_dict(payload["terminal_model_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    optimizer_steps = {
        int(value["step"].item() if torch.is_tensor(value["step"]) else value["step"])
        for value in optimizer.state.values()
        if "step" in value
    }
    if not optimizer_steps or optimizer_steps != {completed}:
        raise ValueError("resume optimizer state does not contain the exact committed step")
    return completed, history, expected_sha


def _write_step_pending(
    status_path: Path,
    reservation: dict[str, Any],
    *,
    update: int,
    committed_updates: int,
    latest_resume_sha256: str | None,
    code_identity: dict[str, str],
    clearml_task_id: str,
) -> None:
    """Durably close the SIGKILL ambiguity window before optimizer.step()."""
    if update != committed_updates + 1:
        raise ValueError("pending optimizer update is not the next committed update")
    _strict_atomic_json(
        status_path,
        {
            **reservation,
            "status": "step_pending",
            "completed_updates": committed_updates,
            "optimizer_steps": committed_updates,
            "attempted_optimizer_steps": update,
            "executed_optimizer_steps": committed_updates,
            "latest_resume_sha256": latest_resume_sha256,
            "code_identity": code_identity,
            "clearml_task_id": clearml_task_id,
        },
    )


def run(config_path: Path, output_dir: Path, *, resume: bool = False) -> dict[str, Any]:
    config = load_json(config_path)
    if config.get("schema_version") != "fine_support_proper_training_v1":
        raise ValueError("unreviewed fine support training schema")
    if config.get("protocol") != EXPECTED_PROTOCOL:
        raise ValueError("fine support training protocol differs from review")
    status_path = output_dir / "status.json"
    progress_path = output_dir / "progress.json"
    resume_path = output_dir / "resume_latest.pth"
    final_path = output_dir / "terminal_fine_model_update_64.pth"
    record_path = output_dir / "training.json"
    config_sha = _sha256(config_path)
    previous_status_binding = None
    expected_resume_sha = None
    expected_resume_updates = 0
    now_unix = time.time()
    if resume:
        if not output_dir.is_dir() or not status_path.is_file() or not resume_path.is_file():
            raise FileNotFoundError("resume requires an existing failed/interrupted run and checkpoint")
        previous = load_json(status_path)
        if previous.get("status") not in {
            "failed",
            "training",
            "resuming",
            "step_pending",
        }:
            raise ValueError("resume requires failed or interrupted training status")
        expected_resume_updates = int(previous.get("completed_updates", -1))
        counters = {
            int(previous.get(name, -1))
            for name in (
                "optimizer_steps",
                "attempted_optimizer_steps",
                "executed_optimizer_steps",
            )
        }
        if counters != {expected_resume_updates}:
            raise ValueError("resume forbidden after an ambiguous uncommitted optimizer step")
        expected_resume_sha = previous.get("latest_resume_sha256")
        if not isinstance(expected_resume_sha, str) or len(expected_resume_sha) != 64:
            raise ValueError("resume status lacks an exact checkpoint SHA binding")
        training_started_unix = float(previous.get("training_started_unix", math.nan))
        deadline_unix = float(previous.get("deadline_unix", math.nan))
        if (
            not math.isfinite(training_started_unix)
            or not math.isfinite(deadline_unix)
            or deadline_unix != training_started_unix + 60 * EXPECTED_PROTOCOL["max_gpu_minutes"]
            or training_started_unix > now_unix
        ):
            raise ValueError("resume status has an invalid immutable wall-clock budget")
        if expected_resume_updates < EXPECTED_PROTOCOL["updates"] and now_unix >= deadline_unix:
            raise TimeoutError("immutable 30-minute training budget already exhausted")
        previous_status_binding = {
            "sha256": _sha256(status_path),
            "status": previous.get("status"),
            "completed_updates": previous.get("completed_updates"),
            "optimizer_steps": previous.get("optimizer_steps"),
            "attempted_optimizer_steps": previous.get("attempted_optimizer_steps"),
            "executed_optimizer_steps": previous.get("executed_optimizer_steps"),
            "latest_resume_sha256": expected_resume_sha,
        }
    else:
        if output_dir.exists() or output_dir.is_symlink():
            raise FileExistsError(f"refusing to reuse output directory {output_dir}")
        training_started_unix = now_unix
        deadline_unix = training_started_unix + 60 * EXPECTED_PROTOCOL["max_gpu_minutes"]
    if torch.cuda.device_count() != 1:
        raise RuntimeError("fine support training requires exactly one visible GPU")
    if not resume:
        output_dir.mkdir(parents=True)
    reservation = {
        "status": "resuming" if resume else "reserved",
        "config_path": str(config_path),
        "config_sha256": config_sha,
        "requested_updates": EXPECTED_PROTOCOL["updates"],
        "resume": resume,
        "training_started_unix": training_started_unix,
        "deadline_unix": deadline_unix,
    }
    if previous_status_binding is not None:
        reservation["previous_status"] = previous_status_binding
        reservation.update(
            {
                "completed_updates": expected_resume_updates,
                "optimizer_steps": expected_resume_updates,
                "attempted_optimizer_steps": expected_resume_updates,
                "executed_optimizer_steps": expected_resume_updates,
                "latest_resume_sha256": expected_resume_sha,
            }
        )
    _strict_atomic_json(status_path, reservation)
    tracker = None
    completed_updates = expected_resume_updates if resume else 0
    attempted_optimizer_steps = completed_updates
    executed_optimizer_steps = completed_updates
    latest_resume_sha = expected_resume_sha if resume else None
    final_sha = None
    history: list[dict[str, Any]] = []
    try:
        repo = Path(__file__).resolve().parents[1]
        code_identity = _clean_code_identity(repo)
        preflight, preflight_binding = _load_bound_preflight(config, repo)
        source = preflight["source"]
        source_files = _validate_source_files(source)
        means, stds, normalization = _load_normalization_binding(preflight, repo)
        config_dir = (repo / config["preflight_config"]["path"]).resolve().parent
        data_path = resolve_path(preflight["data_config"], config_dir)
        model_path = resolve_path(preflight["model_config"], config_dir)
        if _sha256_file(data_path) != preflight["data_config_sha256"]:
            raise ValueError("training data config SHA differs from preflight")
        if _sha256_file(model_path) != preflight["model_config_sha256"]:
            raise ValueError("training model config SHA differs from preflight")
        data_config = merge_config_overrides(
            load_json(data_path), preflight.get("data_overrides")
        )
        model_config = load_json(model_path)
        effective_contract = forecast_contract_from_data_config(data_config)
        effective_contract_sha = forecast_contract_sha256(effective_contract)
        if effective_contract_sha != source["forecast_contract_sha256"]:
            raise ValueError("training forecast contract differs from fine2048")
        seed_everything(int(EXPECTED_PROTOCOL["seed"]))
        dataset = build_dataset(data_config, split="train")
        if len(dataset) != EXPECTED_SPLIT_LENGTHS["train"]:
            raise ValueError("train archive length differs from audited inventory")
        indices = _evenly_spaced_indices(len(dataset), EXPECTED_PROTOCOL["updates"])
        if len(indices) != EXPECTED_PROTOCOL["updates"] or len(set(indices)) != len(indices):
            raise RuntimeError("training schedule must contain 64 unique train cases")
        sentinel = {
            "dataset": validate_direct_dataset(dataset),
            "calendar": _calendar_inventory(dataset, split="train"),
            "data_config_sha256": _canonical_sha256(data_config),
        }
        device = torch.device("cuda:0")
        tracker = ClearMLTracker(
            config["project_name"],
            f"{config['task_name']}-{output_dir.name}-{'resume' if resume else 'fresh'}",
            tags=[*config["clearml"]["tags"], "resume" if resume else "fresh"],
            env_path=config["clearml"]["env_path"],
        )
        tracker.connect("training_contract", config)
        implementation = source["implementation_sha256"]
        loaded = load_colored_variance_preconditioned_fine_cascade_sampler(
            source["run_dir"],
            source["checkpoint"],
            model_config,
            source_files[source["checkpoint"]],
            source["code_commit"],
            source["forecast_contract_sha256"],
            device=device,
            expected_colored_implementation_sha256=implementation["fine_colored"],
            expected_preconditioning_implementation_sha256=implementation["fine_preconditioning"],
            expected_conditioning_implementation_sha256=implementation["fine_conditioning"],
        )
        frozen_model = loaded.sampler.model.eval()
        for parameter in frozen_model.parameters():
            parameter.requires_grad_(False)
        terminal_model = copy.deepcopy(frozen_model).train()
        for parameter in terminal_model.parameters():
            parameter.requires_grad_(True)
        optimizer = torch.optim.AdamW(
            terminal_model.parameters(),
            lr=EXPECTED_PROTOCOL["learning_rate"],
            weight_decay=EXPECTED_PROTOCOL["weight_decay"],
        )
        if resume:
            completed_updates, history, latest_resume_sha = _restore_resume(
                resume_path,
                terminal_model,
                optimizer,
                config_sha256=config_sha,
                code_identity=code_identity,
                source_checkpoint_sha256=source_files[source["checkpoint"]],
                expected_sha256=expected_resume_sha,
                expected_completed_updates=expected_resume_updates,
                expected_indices=indices,
                training_started_unix=training_started_unix,
                deadline_unix=deadline_unix,
                device=device,
            )
            attempted_optimizer_steps = completed_updates
            executed_optimizer_steps = completed_updates
        else:
            frozen_state = frozen_model.state_dict()
            terminal_state = terminal_model.state_dict()
            if frozen_state.keys() != terminal_state.keys() or any(
                not torch.equal(frozen_state[name], terminal_state[name])
                for name in frozen_state
            ):
                raise RuntimeError("fresh terminal model differs from fine2048")
        _strict_atomic_json(
            status_path,
            {
                **reservation,
                "status": "training",
                "completed_updates": completed_updates,
                "optimizer_steps": completed_updates,
                "attempted_optimizer_steps": attempted_optimizer_steps,
                "executed_optimizer_steps": executed_optimizer_steps,
                "latest_resume_sha256": latest_resume_sha,
                "code_identity": code_identity,
                "clearml_task_id": str(tracker.task.id),
            },
        )
        grid = None
        torch.cuda.reset_peak_memory_stats(device)
        members = EXPECTED_PROTOCOL["members"]
        means_gpu = means.to(device=device).reshape(1, 1, 6, 1, 1)
        stds_gpu = stds.to(device=device).reshape(1, 1, 6, 1, 1)
        for update in range(completed_updates + 1, EXPECTED_PROTOCOL["updates"] + 1):
            if time.time() >= deadline_unix:
                raise TimeoutError("immutable 30-minute training budget exhausted")
            update_started = time.monotonic()
            raw = dataset[indices[update - 1]]
            case_id = str(raw.get("meta", {}).get("case_id", f"train_index_{indices[update - 1]}"))
            truth = raw["truth"].unsqueeze(0).to(device=device, dtype=torch.float32)
            valid = raw["valid_mask"].unsqueeze(0)[:, :1].to(device=device, dtype=torch.float32)
            structured = raw["structured_conditioning"].unsqueeze(0).to(device=device, dtype=torch.float32)
            condition, coarse, _ = teacher_coarse_condition(truth, structured, valid)
            condition_m = _repeat_members(condition, members)
            valid_m = _repeat_members(valid, members)
            generator = torch.Generator(device=device).manual_seed(
                EXPECTED_PROTOCOL["seed"] + update
            )
            white = torch.randn(
                (members, 6, *truth.shape[-2:]),
                generator=generator,
                device=device,
                dtype=torch.float32,
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                colored = loaded.project_initial_noise(white, valid_m)
            if grid is None:
                grid = make_normalized_xy_grid(
                    *truth.shape[-2:], device=device, dtype=torch.float32
                )
            optimizer.zero_grad(set_to_none=True)
            prefix = fine_frozen_prefix(
                frozen_model, colored, condition_m, valid_m, grid
            )
            candidate_residual = terminal_fine_residual(
                terminal_model, prefix, condition_m, valid_m, grid
            )
            candidate = (
                condition_m[:, -6:] + candidate_residual
            ).unflatten(0, (1, members))
            candidate_physical = candidate.double() * stds_gpu + means_gpu
            truth_physical, teacher_physical, teacher_consistency = (
                canonical_truth_and_teacher_coarse(
                    truth,
                    coarse,
                    valid,
                    means.to(device=device),
                    stds.to(device=device),
                )
            )
            coarse_physical = teacher_physical[:, None].expand(
                -1, members, -1, -1, -1
            )
            objective, crps, energy, decoded = support_aware_physical_objective(
                candidate_physical,
                coarse_physical,
                truth_physical,
                valid.double(),
                stds.to(device=device),
            )
            if not all(torch.isfinite(value) for value in (objective, crps, energy)):
                raise FloatingPointError("fine support refinement score is non-finite")
            residual_coarse, residual_fraction = masked_block_average(
                candidate_residual, valid_m
            )
            residual_error = float(
                residual_coarse[
                    residual_fraction.expand_as(residual_coarse) > 0
                ].abs().max()
            )
            if residual_error > 3e-6:
                raise RuntimeError("trained terminal residual left the detail nullspace")
            recovered, decoded_fraction = masked_block_average(
                decoded.flatten(0, 1)[:, 0::2], valid_m
            )
            expected_coarse = coarse_physical.flatten(0, 1)[:, 0::2].clamp(0, 1)
            decoded_coarse_error = float(
                (recovered - expected_coarse)[
                    decoded_fraction.expand_as(recovered) > 0
                ].abs().max()
            )
            if decoded_coarse_error > 3e-14:
                raise RuntimeError("trained decoded SIC lost teacher coarse consistency")
            objective.backward()
            gradients = [
                parameter.grad
                for parameter in terminal_model.parameters()
                if parameter.grad is not None
            ]
            if not gradients or not all(torch.isfinite(value).all() for value in gradients):
                raise FloatingPointError("fine support refinement produced invalid gradients")
            gradient_norm = float(
                torch.sqrt(sum(value.float().square().sum() for value in gradients)).cpu()
            )
            if not math.isfinite(gradient_norm) or gradient_norm <= 0:
                raise FloatingPointError("fine support refinement gradient is dead")
            if any(parameter.grad is not None for parameter in frozen_model.parameters()):
                raise RuntimeError("gradient leaked through frozen fine2048 prefix")
            attempted_optimizer_steps = update
            _write_step_pending(
                status_path,
                reservation,
                update=update,
                committed_updates=completed_updates,
                latest_resume_sha256=latest_resume_sha,
                code_identity=code_identity,
                clearml_task_id=str(tracker.task.id),
            )
            optimizer.step()
            executed_optimizer_steps = update
            if not all(torch.isfinite(parameter).all() for parameter in terminal_model.parameters()):
                raise FloatingPointError("fine support refinement produced invalid parameters")
            row = {
                "update": update,
                "train_index": int(indices[update - 1]),
                "case_id": case_id,
                "noise_seed": EXPECTED_PROTOCOL["seed"] + update,
                "objective": float(objective.detach().cpu()),
                "fair_crps": float(crps.detach().cpu()),
                "joint_energy": float(energy.detach().cpu()),
                "gradient_norm": gradient_norm,
                "canonical_teacher_normalized_coarse_max_abs": teacher_consistency,
                "residual_nullspace_max_abs": residual_error,
                "decoded_sic_coarse_consistency_max_abs": decoded_coarse_error,
                "elapsed_seconds": time.time() - training_started_unix,
                "update_seconds": time.monotonic() - update_started,
                "peak_gpu_memory_mib": float(torch.cuda.max_memory_allocated(device) / 2**20),
            }
            next_history = [*history, row]
            next_resume_sha = _atomic_torch_save(
                _resume_payload(
                    terminal_model,
                    optimizer,
                    completed_updates=update,
                    history=next_history,
                    config_sha256=config_sha,
                    code_identity=code_identity,
                    source_checkpoint_sha256=source_files[source["checkpoint"]],
                    training_started_unix=training_started_unix,
                    deadline_unix=deadline_unix,
                ),
                resume_path,
            )
            completed_updates = update
            history = next_history
            latest_resume_sha = next_resume_sha
            _strict_atomic_json(
                progress_path,
                {
                    "status": "training",
                    "completed_updates": completed_updates,
                    "optimizer_steps": completed_updates,
                    "attempted_optimizer_steps": attempted_optimizer_steps,
                    "executed_optimizer_steps": executed_optimizer_steps,
                    "latest_resume_sha256": latest_resume_sha,
                    "history": history,
                },
            )
            _strict_atomic_json(
                status_path,
                {
                    **reservation,
                    "status": "training",
                    "completed_updates": completed_updates,
                    "optimizer_steps": completed_updates,
                    "attempted_optimizer_steps": attempted_optimizer_steps,
                    "executed_optimizer_steps": executed_optimizer_steps,
                    "latest_resume_sha256": latest_resume_sha,
                    "code_identity": code_identity,
                    "clearml_task_id": str(tracker.task.id),
                },
            )
            for name in ("objective", "fair_crps", "joint_energy", "gradient_norm"):
                tracker.report_scalar("train", name, row[name], update)

        if completed_updates != EXPECTED_PROTOCOL["updates"] or len(history) != completed_updates:
            raise RuntimeError("fine support refinement did not complete exactly 64 updates")
        final_sha = _atomic_torch_save(
            {
                "schema_version": "terminal_fine_support_model_v1",
                "completed_updates": completed_updates,
                "optimizer_steps": completed_updates,
                "terminal_model_state": terminal_model.state_dict(),
                "source_checkpoint_sha256": source_files[source["checkpoint"]],
                "config_sha256": config_sha,
                "code_identity": code_identity,
                "protocol": EXPECTED_PROTOCOL,
            },
            final_path,
        )
        result = {
            "status": "training_complete_pending_generated_coarse_validation",
            "scientific_role": "train_only_terminal_fine_support_refinement",
            "training_performed": True,
            "optimizer_steps": completed_updates,
            "attempted_optimizer_steps": attempted_optimizer_steps,
            "executed_optimizer_steps": executed_optimizer_steps,
            "completed_updates": completed_updates,
            "test_2023_used": False,
            "training_started_unix": training_started_unix,
            "deadline_unix": deadline_unix,
            "code_identity": code_identity,
            "preflight": preflight_binding,
            "source": source,
            "source_files_sha256": source_files,
            "normalization": normalization,
            "protocol": EXPECTED_PROTOCOL,
            "train_indices": indices,
            "dataset_sentinel": sentinel,
            "effective_forecast_contract": effective_contract,
            "effective_forecast_contract_sha256": effective_contract_sha,
            "history": history,
            "terminal_checkpoint": str(final_path),
            "terminal_checkpoint_sha256": final_sha,
            "resume_checkpoint": str(resume_path),
            "resume_checkpoint_sha256": latest_resume_sha,
            "peak_gpu_memory_mib": float(torch.cuda.max_memory_allocated(device) / 2**20),
            "elapsed_seconds": time.time() - training_started_unix,
            "clearml_task_id": str(tracker.task.id),
        }
        _strict_atomic_json(record_path, result)
        tracker.connect("training_result", result)
        tracker.upload_artifact("terminal_fine_support_model_update_64", final_path)
        tracker.upload_artifact("fine_support_training_record", record_path)
        tracker.close()
        tracker = None
        _strict_atomic_json(
            status_path,
            {
                **reservation,
                "status": "complete",
                "completed_updates": completed_updates,
                "optimizer_steps": completed_updates,
                "attempted_optimizer_steps": attempted_optimizer_steps,
                "executed_optimizer_steps": executed_optimizer_steps,
                "code_identity": code_identity,
                "clearml_task_id": result["clearml_task_id"],
                "training_record": str(record_path),
                "training_record_sha256": _sha256(record_path),
                "terminal_checkpoint": str(final_path),
                "terminal_checkpoint_sha256": final_sha,
                "resume_checkpoint_sha256": latest_resume_sha,
            },
        )
        return result
    except BaseException as error:
        failure = {
            **reservation,
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error)[:2000],
            "completed_updates": completed_updates,
            "optimizer_steps": completed_updates,
            "attempted_optimizer_steps": attempted_optimizer_steps,
            "executed_optimizer_steps": executed_optimizer_steps,
        }
        try:
            if latest_resume_sha is not None:
                failure["resume_checkpoint"] = str(resume_path)
                failure["latest_resume_sha256"] = latest_resume_sha
            if final_sha is not None and final_path.is_file():
                failure["terminal_checkpoint"] = str(final_path)
                failure["terminal_checkpoint_sha256"] = final_sha
            if record_path.is_file():
                failure["training_record"] = str(record_path)
                failure["training_record_sha256"] = _sha256(record_path)
            _strict_atomic_json(status_path, failure)
        except BaseException:
            pass
        if tracker is not None:
            try:
                tracker.task.mark_failed(
                    status_reason=f"{type(error).__name__}: {str(error)[:1000]}"
                )
            except BaseException:
                pass
        raise
    finally:
        if tracker is not None:
            try:
                tracker.close()
            except BaseException:
                pass


def main() -> None:
    signal.signal(signal.SIGTERM, _terminate)
    signal.signal(signal.SIGINT, _terminate)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.config.resolve(), args.output.resolve(), resume=args.resume),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
