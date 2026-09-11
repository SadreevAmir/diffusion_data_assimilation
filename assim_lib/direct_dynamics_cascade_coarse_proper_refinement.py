"""Bounded proper-score refinement of the terminal coarse-flow interval.

The first 15 RK4 intervals are evaluated by the immutable EMA9711 model under
``no_grad``.  Only the final interval, from t=1/16 to t=0, is trainable.  The
same hybrid generator is used for training and evaluation, so the gradient is
exact for the declared candidate rather than a one-step denoising surrogate.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import signal
import time
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import Subset

from .clearml_tracking import ClearMLTracker
from .config import load_json, merge_config_overrides, resolve_path
from .data import build_dataset
from .direct_dynamics_cascade_coarse import (
    DIRECT_OUTPUT_CHANNELS,
    coarse_target,
    load_coarse_cascade_sampler,
    lossless_coarse_condition,
)
from .direct_dynamics_cascade_contract import (
    forecast_contract_from_data_config,
    forecast_contract_sha256,
)
from .direct_dynamics_cascade_fine_training import (
    EXPECTED_SPLIT_LENGTHS,
    _calendar_inventory,
    _canonical_sha256,
    _clean_code_identity,
    _evenly_spaced_indices,
    _sha256_file,
)
from .direct_dynamics_training import validate_direct_dataset
from .runtime import make_normalized_xy_grid, seed_everything
from .trainer import _atomic_json


def _model_output(value: Any) -> torch.Tensor:
    if hasattr(value, "sample"):
        return value.sample
    if isinstance(value, (tuple, list)):
        return value[0]
    return value


def _velocity(
    model: torch.nn.Module,
    state: torch.Tensor,
    encoded_condition: torch.Tensor,
    active: torch.Tensor,
    grid: torch.Tensor,
    time: float,
) -> torch.Tensor:
    batch = state.shape[0]
    timestamp = torch.full((batch,), 1000.0 * float(time), device=state.device)
    model_input = torch.cat((state, grid.expand(batch, -1, -1, -1), encoded_condition), dim=1)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        velocity = _model_output(model(model_input, timestamp, return_dict=False))
    velocity = velocity.float()
    support = active.expand_as(velocity) > 0
    return torch.where(support, velocity, torch.zeros_like(velocity))


def rk4_interval(
    model: torch.nn.Module,
    state: torch.Tensor,
    encoded_condition: torch.Tensor,
    active: torch.Tensor,
    grid: torch.Tensor,
    t0: float,
    t1: float,
) -> torch.Tensor:
    """One torchdiffeq-compatible RK4 3/8 interval.

    ``torchdiffeq`` maps ``method='rk4'`` to ``rk4_alt_step_func`` rather than
    the classical RK4 tableau.  Keeping the exact 1/3, 2/3 and 3/8 rule is
    necessary for a valid step-zero replay against the production sampler.
    """
    step = float(t1) - float(t0)
    k1 = _velocity(model, state, encoded_condition, active, grid, t0)
    k2 = _velocity(
        model,
        state + (step / 3.0) * k1,
        encoded_condition,
        active,
        grid,
        float(t0) + step / 3.0,
    )
    k3 = _velocity(
        model,
        state + step * (k2 - k1 / 3.0),
        encoded_condition,
        active,
        grid,
        float(t0) + 2.0 * step / 3.0,
    )
    k4 = _velocity(
        model,
        state + step * (k1 - k2 + k3),
        encoded_condition,
        active,
        grid,
        t1,
    )
    result = state + (step / 8.0) * (k1 + 3.0 * k2 + 3.0 * k3 + k4)
    support = active.expand_as(result) > 0
    result = torch.where(support, result, torch.zeros_like(result))
    if not torch.isfinite(result[support]).all():
        raise FloatingPointError("hybrid RK4 interval produced NaN/Inf")
    return result


def frozen_prefix(
    model: torch.nn.Module,
    noise: torch.Tensor,
    encoded_condition: torch.Tensor,
    active: torch.Tensor,
    grid: torch.Tensor,
    *,
    intervals: int = 16,
) -> torch.Tensor:
    if intervals != 16:
        raise ValueError("proper refinement requires exactly sixteen RK4 intervals")
    support = active.expand_as(noise) > 0
    state = torch.where(support, noise.float(), torch.zeros_like(noise.float()))
    with torch.no_grad():
        for interval in range(intervals - 1):
            t0 = 1.0 - interval / intervals
            t1 = 1.0 - (interval + 1) / intervals
            state = rk4_interval(model, state, encoded_condition, active, grid, t0, t1)
    return state.detach()


def hybrid_terminal_sample(
    trainable_model: torch.nn.Module,
    prefix_state: torch.Tensor,
    encoded_condition: torch.Tensor,
    active: torch.Tensor,
    grid: torch.Tensor,
) -> torch.Tensor:
    return rk4_interval(
        trainable_model,
        prefix_state,
        encoded_condition,
        active,
        grid,
        1.0 / 16.0,
        0.0,
    )


def _validate_score_inputs(
    members: torch.Tensor, truth: torch.Tensor, ocean_fraction: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if members.ndim != 5 or members.shape[1] < 2:
        raise ValueError("proper scores require members [B,M,C,H,W] with M>=2")
    if truth.shape != members.shape[:1] + members.shape[2:]:
        raise ValueError("truth shape differs from ensemble")
    if ocean_fraction.shape != truth.shape[:1] + (1,) + truth.shape[-2:]:
        raise ValueError("ocean-fraction shape differs from truth")
    if members.shape[2] != DIRECT_OUTPUT_CHANNELS:
        raise ValueError("proper refinement requires all six outputs jointly")
    if not torch.isfinite(ocean_fraction).all() or torch.any(
        (ocean_fraction < 0) | (ocean_fraction > 1)
    ):
        raise ValueError("ocean fractions must lie in [0,1]")
    weight = ocean_fraction[:, None].to(device=members.device, dtype=members.dtype)
    support = weight.expand_as(members) > 0
    denominator = members.shape[2] * weight[:, 0].sum(dim=(1, 2, 3))
    if torch.any(denominator <= 0):
        raise ValueError("every proper-score case must contain positive ocean weight")
    if not torch.isfinite(members[support]).all():
        raise FloatingPointError("ensemble is empty or non-finite on active ocean")
    truth_support = ocean_fraction.expand_as(truth) > 0
    if not torch.isfinite(truth[truth_support]).all():
        raise FloatingPointError("truth is non-finite on active ocean")
    safe_members = torch.where(support, members, torch.zeros_like(members))
    safe_truth = torch.where(truth_support, truth, torch.zeros_like(truth))
    return weight, safe_members, safe_truth


def standardized_fair_crps(
    members: torch.Tensor, truth: torch.Tensor, ocean_fraction: torch.Tensor
) -> torch.Tensor:
    """Case-equal ocean-fraction-weighted unbiased ensemble CRPS."""
    weight, safe_members, safe_truth = _validate_score_inputs(
        members, truth, ocean_fraction
    )
    count = safe_members.shape[1]
    observation = (safe_members - safe_truth[:, None]).abs().mean(dim=1)
    pair_sum = torch.zeros_like(observation)
    for left in range(count):
        for right in range(left + 1, count):
            pair_sum = pair_sum + (
                safe_members[:, left] - safe_members[:, right]
            ).abs()
    score = observation - pair_sum / (count * (count - 1))
    denominator = safe_members.shape[2] * weight[:, 0].sum(dim=(1, 2, 3))
    result = ((score * weight[:, 0]).sum(dim=(1, 2, 3)) / denominator).mean()
    if not torch.isfinite(result):
        raise FloatingPointError("standardized fair CRPS is non-finite")
    return result


def standardized_joint_energy(
    members: torch.Tensor, truth: torch.Tensor, ocean_fraction: torch.Tensor
) -> torch.Tensor:
    """Case-equal unbiased joint energy score in all-six standardized space."""
    weight, safe_members, safe_truth = _validate_score_inputs(
        members, truth, ocean_fraction
    )
    count = safe_members.shape[1]
    denominator = safe_members.shape[2] * weight[:, 0].sum(dim=(1, 2, 3))
    root_weight = torch.sqrt(weight[:, 0])

    def distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        weighted = (left - right) * root_weight
        return torch.linalg.vector_norm(weighted.flatten(1), dim=1) / torch.sqrt(
            denominator
        )

    observation = torch.stack(
        [distance(safe_members[:, member], safe_truth) for member in range(count)], dim=1
    ).mean(dim=1)
    pair_sum = torch.zeros_like(observation)
    for left in range(count):
        for right in range(left + 1, count):
            pair_sum = pair_sum + distance(
                safe_members[:, left], safe_members[:, right]
            )
    result = (observation - pair_sum / (count * (count - 1))).mean()
    if not torch.isfinite(result):
        raise FloatingPointError("standardized joint energy is non-finite")
    return result


def proper_objective(
    members: torch.Tensor, truth: torch.Tensor, ocean_fraction: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    crps = standardized_fair_crps(members, truth, ocean_fraction)
    energy = standardized_joint_energy(members, truth, ocean_fraction)
    return 0.75 * crps + 0.25 * energy, crps, energy


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repeat_for_members(value: torch.Tensor, members: int) -> torch.Tensor:
    return value[:, None].expand(-1, members, *value.shape[1:]).flatten(0, 1)


REVIEWED_PROTOCOL = {
    "prefix_intervals": 15,
    "trainable_intervals": 1,
    "rk4_intervals": 16,
    "members": 4,
    "updates": 64,
    "batch_size": 1,
    "learning_rate": 1e-5,
    "crps_weight": 0.75,
    "joint_energy_weight": 0.25,
    "network_precision": "bf16",
    "state_precision": "fp32",
    "score_precision": "fp32",
    "seed": 69173,
    "training_split": "train-2016-2021",
    "evaluation_generator": "same frozen-prefix plus trainable-terminal hybrid",
    "rank_objective": False,
    "clipping": False,
    "activation_checkpointing": False,
    "additional_ema": False,
    "max_gpu_minutes": 30,
}


def _validate_reviewed_protocol(protocol: dict[str, Any]) -> None:
    if any(protocol.get(key) != value for key, value in REVIEWED_PROTOCOL.items()):
        raise ValueError("proper-refinement protocol differs from the reviewed bounded contract")


def _atomic_torch_save(payload: Any, path: Path) -> str:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return _sha256(path)


def _persist_training_update_then_report(
    output_dir: Path,
    progress: dict[str, Any],
    report: Callable[[], None],
    *,
    final_checkpoint: dict[str, Any] | None = None,
) -> str | None:
    """Make local progress durable before any fallible ClearML reporting."""
    _atomic_json(output_dir / "progress.json", progress)
    checkpoint_sha256 = None
    if final_checkpoint is not None:
        checkpoint_sha256 = _atomic_torch_save(
            final_checkpoint, output_dir / "terminal_model_update_64.pth"
        )
    report()
    return checkpoint_sha256


def run_admission(config_path: Path, output_dir: Path) -> dict[str, Any]:
    experiment = load_json(config_path)
    protocol = experiment["protocol"]
    _validate_reviewed_protocol(protocol)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("proper refinement requires exactly one visible GPU")
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"refusing to reuse output directory {output_dir}")
    output_dir.mkdir(parents=True)
    status_path = output_dir / "status.json"
    _atomic_json(status_path, {"status": "initializing"})
    tracker = None
    try:
        repo_root = Path(__file__).resolve().parents[1]
        code_identity = _clean_code_identity(repo_root)
        config_dir = config_path.resolve().parent
        data_path = resolve_path(experiment["data_config"], config_dir)
        model_path = resolve_path(experiment["model_config"], config_dir)
        data_config = merge_config_overrides(
            load_json(data_path), experiment.get("data_overrides")
        )
        model_config = load_json(model_path)
        source = experiment["source"]
        effective_contract = forecast_contract_from_data_config(data_config)
        effective_contract_sha256 = forecast_contract_sha256(effective_contract)
        if effective_contract_sha256 != source["forecast_contract_sha256"]:
            raise ValueError("effective data/conditioning contract differs from EMA9711")
        seed_everything(int(protocol["seed"]))
        dataset = build_dataset(data_config, split="train")
        if len(dataset) != EXPECTED_SPLIT_LENGTHS["train"]:
            raise ValueError("train archive length differs from audited inventory")
        sentinel = {
            "dataset": validate_direct_dataset(dataset),
            "calendar": _calendar_inventory(dataset, split="train"),
            "data_config_sha256": _canonical_sha256(data_config),
        }
        device = torch.device("cuda:0")
        tracker = ClearMLTracker(
            experiment["project_name"],
            f"{experiment['task_name']}-{output_dir.name}",
            tags=experiment["clearml"]["tags"],
            env_path=experiment["clearml"]["env_path"],
        )
        _atomic_json(
            status_path,
            {"status": "gpu_admission_running", "clearml_task_id": str(tracker.task.id)},
        )
        loaded = load_coarse_cascade_sampler(
            source["run_dir"],
            source["checkpoint"],
            model_config,
            source["checkpoint_sha256"],
            source["code_commit"],
            source["forecast_contract_sha256"],
            device=device,
        )
        frozen_model = loaded.sampler.model.eval()
        for parameter in frozen_model.parameters():
            parameter.requires_grad_(False)
        trainable_model = copy.deepcopy(frozen_model).train()
        for parameter in trainable_model.parameters():
            parameter.requires_grad_(True)
        source_state = frozen_model.state_dict()
        candidate_state = trainable_model.state_dict()
        if source_state.keys() != candidate_state.keys() or any(
            not torch.equal(source_state[key], candidate_state[key]) for key in source_state
        ):
            raise RuntimeError("step-zero trainable model differs from EMA9711")

        indices = _evenly_spaced_indices(len(dataset), int(protocol["updates"]))
        raw = Subset(dataset, indices)[0]
        truth = raw["truth"].unsqueeze(0).to(device=device, dtype=torch.float32)
        valid = raw["valid_mask"].unsqueeze(0)[:, :1].to(device=device, dtype=torch.float32)
        condition = raw["structured_conditioning"].unsqueeze(0).to(
            device=device, dtype=torch.float32
        )
        coarse_truth, active, ocean_fraction = coarse_target(truth, valid)
        encoded, encoded_active, encoded_fraction = lossless_coarse_condition(condition, valid)
        if not torch.equal(active, encoded_active) or not torch.equal(
            ocean_fraction, encoded_fraction
        ):
            raise RuntimeError("coarse truth and condition support differ")
        members = int(protocol["members"])
        generator = torch.Generator(device=device).manual_seed(int(protocol["seed"]) + 1)
        noise = torch.randn(
            (members, DIRECT_OUTPUT_CHANNELS, *coarse_truth.shape[-2:]),
            generator=generator,
            device=device,
        )
        encoded_members = _repeat_for_members(encoded, members)
        active_members = _repeat_for_members(active, members)
        grid = make_normalized_xy_grid(
            *coarse_truth.shape[-2:], device=device, dtype=torch.float32
        )
        torch.cuda.reset_peak_memory_stats(device)
        prefix = frozen_prefix(
            frozen_model, noise, encoded_members, active_members, grid
        )
        candidate = hybrid_terminal_sample(
            trainable_model, prefix, encoded_members, active_members, grid
        ).unflatten(0, (1, members))
        with torch.no_grad():
            control = hybrid_terminal_sample(
                frozen_model, prefix, encoded_members, active_members, grid
            ).unflatten(0, (1, members))
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                production = loaded.sample_conditioned(
                    structured_conditioning=_repeat_for_members(condition, members),
                    valid_mask=_repeat_for_members(valid, members),
                    initial_noise=noise,
                    num_timesteps=17,
                    device=device,
                    method="rk4",
                    rtol=1e-5,
                    atol=1e-6,
                    end_time=0.0,
                ).unflatten(0, (1, members))
        candidate_control_max_abs = float((candidate.detach() - control).abs().max().cpu())
        production_replay_max_abs = float((control - production).abs().max().cpu())
        if candidate_control_max_abs > 1e-6:
            raise RuntimeError("step-zero trainable terminal differs from frozen terminal")
        if production_replay_max_abs > 2e-5:
            raise RuntimeError("custom hybrid does not replay the production RK4 sampler")
        objective, crps, energy = proper_objective(
            candidate, coarse_truth, ocean_fraction
        )
        objective.backward()
        gradients = [
            parameter.grad.detach()
            for parameter in trainable_model.parameters()
            if parameter.grad is not None
        ]
        if not gradients or not all(torch.isfinite(value).all() for value in gradients):
            raise FloatingPointError("proper-refinement admission produced invalid gradients")
        gradient_norm = float(
            torch.sqrt(sum(value.float().square().sum() for value in gradients)).cpu()
        )
        if not math.isfinite(gradient_norm) or gradient_norm <= 0:
            raise FloatingPointError("proper-refinement gradient is dead")
        result = {
            "status": "admission_passed_pending_astra_review",
            "training_performed": False,
            "source": source,
            "code_identity": code_identity,
            "source_identity": {
                "experiment_sha256": _sha256(config_path),
                "data_sha256": _sha256_file(data_path),
                "model_sha256": _sha256_file(model_path),
            },
            "protocol": protocol,
            "train_indices": indices,
            "dataset_sentinel": sentinel,
            "effective_forecast_contract": effective_contract,
            "effective_forecast_contract_sha256": effective_contract_sha256,
            "step0": {
                "candidate_control_max_abs": candidate_control_max_abs,
                "production_replay_max_abs": production_replay_max_abs,
                "objective": float(objective.detach().cpu()),
                "fair_crps": float(crps.detach().cpu()),
                "joint_energy": float(energy.detach().cpu()),
                "gradient_norm": gradient_norm,
                "peak_gpu_memory_mib": float(
                    torch.cuda.max_memory_allocated(device) / 2**20
                ),
                "candidate_shape": list(candidate.shape),
            },
        }
        result["clearml_task_id"] = str(tracker.task.id)
        _atomic_json(output_dir / "admission.json", result)
        tracker.connect("proper_refinement_admission", result)
        for key, value in result["step0"].items():
            if isinstance(value, (int, float)):
                tracker.report_single_value(f"admission_{key}", float(value))
        tracker.close()
        tracker = None
        _atomic_json(status_path, {"status": result["status"], "clearml_task_id": result["clearml_task_id"]})
        return result
    except BaseException as error:
        failure = {
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
        }
        _atomic_json(status_path, failure)
        raise
    finally:
        if tracker is not None:
            try:
                tracker.close()
            except Exception:
                pass


def run_training(config_path: Path, output_dir: Path) -> dict[str, Any]:
    """Run the single Astra-approved 64-update train-only refinement."""
    experiment = load_json(config_path)
    protocol = experiment["protocol"]
    _validate_reviewed_protocol(protocol)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("proper refinement requires exactly one visible GPU")
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"refusing to reuse output directory {output_dir}")
    output_dir.mkdir(parents=True)
    status_path = output_dir / "status.json"
    _atomic_json(status_path, {"status": "initializing"})
    tracker = None
    history: list[dict[str, Any]] = []
    attempted_update = 0
    checkpoint_sha256 = None
    try:
        started = time.monotonic()
        repo_root = Path(__file__).resolve().parents[1]
        code_identity = _clean_code_identity(repo_root)
        config_dir = config_path.resolve().parent
        data_path = resolve_path(experiment["data_config"], config_dir)
        model_path = resolve_path(experiment["model_config"], config_dir)
        data_config = merge_config_overrides(
            load_json(data_path), experiment.get("data_overrides")
        )
        model_config = load_json(model_path)
        source = experiment["source"]
        effective_contract = forecast_contract_from_data_config(data_config)
        effective_contract_sha256 = forecast_contract_sha256(effective_contract)
        if effective_contract_sha256 != source["forecast_contract_sha256"]:
            raise ValueError("effective data/conditioning contract differs from EMA9711")

        seed_everything(int(protocol["seed"]))
        dataset = build_dataset(data_config, split="train")
        if len(dataset) != EXPECTED_SPLIT_LENGTHS["train"]:
            raise ValueError("train archive length differs from audited inventory")
        sentinel = {
            "dataset": validate_direct_dataset(dataset),
            "calendar": _calendar_inventory(dataset, split="train"),
            "data_config_sha256": _canonical_sha256(data_config),
        }
        indices = _evenly_spaced_indices(len(dataset), int(protocol["updates"]))
        if len(indices) != int(protocol["updates"]) or len(set(indices)) != len(indices):
            raise RuntimeError("reviewed training schedule must contain 64 unique cases")

        device = torch.device("cuda:0")
        tracker = ClearMLTracker(
            experiment["project_name"],
            f"direct_dynamics_cascade_coarse_proper_refinement_train_v1-{output_dir.name}",
            tags=[*experiment["clearml"]["tags"], "train-64-updates"],
            env_path=experiment["clearml"]["env_path"],
        )
        tracker.connect("proper_refinement_protocol", protocol)
        _atomic_json(
            status_path,
            {"status": "training", "clearml_task_id": str(tracker.task.id), "update": 0},
        )

        loaded = load_coarse_cascade_sampler(
            source["run_dir"],
            source["checkpoint"],
            model_config,
            source["checkpoint_sha256"],
            source["code_commit"],
            source["forecast_contract_sha256"],
            device=device,
        )
        frozen_model = loaded.sampler.model.eval()
        for parameter in frozen_model.parameters():
            parameter.requires_grad_(False)
        trainable_model = copy.deepcopy(frozen_model).train()
        for parameter in trainable_model.parameters():
            parameter.requires_grad_(True)
        optimizer = torch.optim.AdamW(
            trainable_model.parameters(),
            lr=float(protocol["learning_rate"]),
            weight_decay=0.0,
        )
        generator = torch.Generator(device=device).manual_seed(int(protocol["seed"]) + 1)
        grid = None
        torch.cuda.reset_peak_memory_stats(device)

        for update, index in enumerate(indices, start=1):
            attempted_update = update
            raw = dataset[index]
            truth = raw["truth"].unsqueeze(0).to(device=device, dtype=torch.float32)
            valid = raw["valid_mask"].unsqueeze(0)[:, :1].to(
                device=device, dtype=torch.float32
            )
            condition = raw["structured_conditioning"].unsqueeze(0).to(
                device=device, dtype=torch.float32
            )
            coarse_truth, active, ocean_fraction = coarse_target(truth, valid)
            encoded, encoded_active, encoded_fraction = lossless_coarse_condition(
                condition, valid
            )
            if not torch.equal(active, encoded_active) or not torch.equal(
                ocean_fraction, encoded_fraction
            ):
                raise RuntimeError("coarse truth and condition support differ")
            if grid is None:
                grid = make_normalized_xy_grid(
                    *coarse_truth.shape[-2:], device=device, dtype=torch.float32
                )

            members = int(protocol["members"])
            noise = torch.randn(
                (members, DIRECT_OUTPUT_CHANNELS, *coarse_truth.shape[-2:]),
                generator=generator,
                device=device,
            )
            encoded_members = _repeat_for_members(encoded, members)
            active_members = _repeat_for_members(active, members)
            prefix = frozen_prefix(
                frozen_model, noise, encoded_members, active_members, grid
            )
            candidate = hybrid_terminal_sample(
                trainable_model, prefix, encoded_members, active_members, grid
            ).unflatten(0, (1, members))
            objective, crps, energy = proper_objective(
                candidate, coarse_truth, ocean_fraction
            )
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            gradients = [
                parameter.grad.detach()
                for parameter in trainable_model.parameters()
                if parameter.grad is not None
            ]
            if not gradients or not all(torch.isfinite(value).all() for value in gradients):
                raise FloatingPointError(f"invalid gradients at update {update}")
            gradient_norm = torch.sqrt(
                sum(value.float().square().sum() for value in gradients)
            )
            if not torch.isfinite(gradient_norm) or gradient_norm <= 0:
                raise FloatingPointError(f"dead gradient at update {update}")
            optimizer.step()
            if not all(
                torch.isfinite(parameter).all() for parameter in trainable_model.parameters()
            ):
                raise FloatingPointError(f"non-finite model at update {update}")

            record = {
                "update": update,
                "train_index": int(index),
                "objective": float(objective.detach().cpu()),
                "fair_crps": float(crps.detach().cpu()),
                "joint_energy": float(energy.detach().cpu()),
                "gradient_norm": float(gradient_norm.detach().cpu()),
                "elapsed_seconds": time.monotonic() - started,
            }
            history.append(record)
            progress = {
                "status": "training",
                "clearml_task_id": str(tracker.task.id),
                "attempted_update": attempted_update,
                "completed_updates": len(history),
                "history": history,
            }
            final_checkpoint = None
            if update == int(protocol["updates"]):
                final_checkpoint = {
                    "model": trainable_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "source": source,
                    "protocol": protocol,
                    "train_indices": indices,
                    "code_identity": code_identity,
                    "effective_forecast_contract_sha256": effective_contract_sha256,
                    "completed_updates": len(history),
                }

            def report_update() -> None:
                for name in ("objective", "fair_crps", "joint_energy", "gradient_norm"):
                    tracker.report_scalar(
                        "proper_refinement_train", name, record[name], update
                    )

            saved_sha256 = _persist_training_update_then_report(
                output_dir,
                progress,
                report_update,
                final_checkpoint=final_checkpoint,
            )
            if saved_sha256 is not None:
                checkpoint_sha256 = saved_sha256
            _atomic_json(status_path, {**progress, **record})

        checkpoint_path = output_dir / "terminal_model_update_64.pth"
        if checkpoint_sha256 is None or not checkpoint_path.is_file():
            raise RuntimeError("final checkpoint was not durably saved at update 64")
        result = {
            "status": "training_complete_pending_paired_gate",
            "source": source,
            "code_identity": code_identity,
            "source_identity": {
                "experiment_sha256": _sha256(config_path),
                "data_sha256": _sha256_file(data_path),
                "model_sha256": _sha256_file(model_path),
            },
            "protocol": protocol,
            "train_indices": indices,
            "dataset_sentinel": sentinel,
            "effective_forecast_contract_sha256": effective_contract_sha256,
            "history": history,
            "checkpoint": checkpoint_path.name,
            "checkpoint_sha256": checkpoint_sha256,
            "peak_gpu_memory_mib": float(torch.cuda.max_memory_allocated(device) / 2**20),
            "elapsed_seconds": time.monotonic() - started,
            "clearml_task_id": str(tracker.task.id),
        }
        _atomic_json(output_dir / "training.json", result)
        tracker.upload_artifact("terminal_model_update_64", checkpoint_path)
        tracker.upload_artifact("proper_refinement_training", output_dir / "training.json")
        tracker.close()
        tracker = None
        _atomic_json(
            status_path,
            {"status": result["status"], "clearml_task_id": result["clearml_task_id"]},
        )
        return result
    except BaseException as error:
        failure = {
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "attempted_update": attempted_update,
            "completed_updates": len(history),
            "history": history,
            "checkpoint": "terminal_model_update_64.pth"
            if (output_dir / "terminal_model_update_64.pth").is_file()
            else None,
        }
        try:
            _atomic_json(output_dir / "failure.json", failure)
            _atomic_json(status_path, failure)
        except Exception:
            pass
        raise
    finally:
        if tracker is not None:
            try:
                tracker.close()
            except Exception:
                pass


def main() -> None:
    def _terminate(signum, _frame):
        raise TimeoutError(f"proper refinement received termination signal {signum}")

    signal.signal(signal.SIGTERM, _terminate)
    signal.signal(signal.SIGINT, _terminate)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("admission", "train"), default="admission")
    args = parser.parse_args()
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", args.output.name) is None:
        raise ValueError("output directory name is unsafe")
    runner = run_admission if args.mode == "admission" else run_training
    print(json.dumps(runner(args.config.resolve(), args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
