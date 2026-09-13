"""Matched 256+256 continuation from EMA6 with native versus geometry proper risk."""

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

import numpy as np
import torch

from .clearml_tracking import ClearMLTracker
from .config import TrainingConfig, load_json
from .data import build_dataset
from .direct_dynamics_cascade_coarse_proper_refinement import proper_objective
from .direct_dynamics_geometry_preflight import (
    SOURCE_SHA_KEYS,
    _cell_area_proxy,
    _schedule_indices,
    _sha256,
    _tensor_sha256,
)
from .direct_dynamics_geometry_score import geometry_energy_score
from .direct_dynamics_suffix import (
    FROZEN_INTERVALS,
    TOTAL_INTERVALS,
    exact_two_pass_score_vjp,
    frozen_prefix,
)
from .direct_dynamics_training import DIRECT_OUTPUT_CHANNELS, _repeat_field_stats, validate_direct_dataset
from .model_io import load_sampler
from .runtime import make_normalized_xy_grid, seed_everything
from .trainer import _atomic_json
from .transforms import channel_denormalize


SCHEMA_VERSION = "direct_dynamics_geometry_objective_ab_training_v1"


def _atomic_torch_save(payload: Any, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.incomplete")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _repeat_members(value: torch.Tensor, members: int) -> torch.Tensor:
    return value[:, None].expand(-1, members, *value.shape[1:]).flatten(0, 1)


def _check_protocol(protocol: dict[str, Any]) -> None:
    expected = {
        "arms": ["control", "treatment"],
        "members": 4,
        "updates_per_arm": 256,
        "batch_size": 1,
        "total_rk4_intervals": TOTAL_INTERVALS,
        "frozen_prefix_intervals": FROZEN_INTERVALS,
        "trainable_suffix_intervals": TOTAL_INTERVALS - FROZEN_INTERVALS,
        "learning_rate": 1e-5,
        "weight_decay": 0.0,
        "optimizer": "AdamW",
        "checkpoint_every_updates": 16,
        "diagnostic_updates": [1, 64, 128, 192, 256],
        "network_precision": "bf16",
        "state_precision": "fp32",
        "score_precision": "fp32",
        "model_mode": "eval_dropout_disabled",
        "control_objective": "0.75_fair_native_crps_plus_0.25_unbiased_native_joint_energy",
        "treatment_addition": "0.25_unbiased_geometry_observable_energy",
        "geometry_weight": 0.25,
        "product_scale": 0.3980696029516097,
        "product_semantics": "mean_sic_times_mean_sit_proxy_not_verified_hourly_volume",
        "seed": 81373,
        "noise_seed_rule": "seed_plus_update_one_based",
        "prefix_cache": "control_computes_treatment_reuses_exact_fp32",
        "rank_or_hard_event_training_loss": False,
        "new_support_decoder": False,
        "additional_ema": False,
        "gradient_clipping": False,
        "test_2023_accessed": False,
    }
    if protocol != expected:
        mismatches = {
            key: (protocol.get(key), value)
            for key, value in expected.items()
            if protocol.get(key) != value
        }
        extras = sorted(set(protocol) - set(expected))
        raise ValueError(f"unreviewed training protocol: mismatches={mismatches}, extras={extras}")


def _load_item(dataset, index: int, device: torch.device) -> dict[str, Any]:
    raw = dataset[index]
    paths = tuple(raw["meta"]["target_trajectory_paths"])
    if raw["meta"].get("split") != "train" or not paths:
        raise RuntimeError("training item lacks train provenance")
    if any("2023" in Path(value).name for value in paths):
        raise RuntimeError("test-2023 path entered training")
    return {
        "raw": raw,
        "condition": raw["structured_conditioning"].unsqueeze(0).to(device, torch.float32),
        "valid": raw["valid_mask"].unsqueeze(0)[:, :1].to(device, torch.float32),
        "truth": raw["truth"].unsqueeze(0).to(device, torch.float32),
        "physical_truth": raw["structured_physical_truth"].unsqueeze(0).to(device, torch.float32),
        "initial_sic": raw["structured_physical_background"][0:1].unsqueeze(0).to(
            device, torch.float32
        ),
    }


def _gradient_norm(model: torch.nn.Module) -> float:
    gradients = [parameter.grad.detach().float() for parameter in model.parameters() if parameter.grad is not None]
    if not gradients or not all(torch.isfinite(value).all() for value in gradients):
        raise FloatingPointError("missing or non-finite training gradient")
    norm = torch.sqrt(sum(value.square().sum() for value in gradients))
    if not torch.isfinite(norm) or norm <= 0:
        raise FloatingPointError("dead training gradient")
    return float(norm.cpu())


def _input_fingerprints(
    batch: dict[str, Any], noise: torch.Tensor
) -> dict[str, str]:
    return {
        "condition": _tensor_sha256(batch["condition"]),
        "valid": _tensor_sha256(batch["valid"]),
        "truth": _tensor_sha256(batch["truth"]),
        "physical_truth": _tensor_sha256(batch["physical_truth"]),
        "initial_sic": _tensor_sha256(batch["initial_sic"]),
        "raw_noise": _tensor_sha256(noise),
    }


def _validate_prefix_cache_binding(
    *,
    payload: dict[str, Any],
    manifest_entry: dict[str, Any],
    prefix_path: Path,
    update: int,
    train_index: int,
    noise_seed: int,
    input_sha256: dict[str, str],
    source_checkpoint_sha256: str,
    code_identity: dict[str, str],
) -> None:
    expected = {
        "train_index": train_index,
        "noise_seed": noise_seed,
        "input_sha256": input_sha256,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise RuntimeError("treatment prefix payload input binding mismatch")
    if (
        payload.get("source_checkpoint_sha256") != source_checkpoint_sha256
        or payload.get("code_identity") != code_identity
        or manifest_entry.get("update") != update
        or any(manifest_entry.get(key) != value for key, value in expected.items())
        or manifest_entry.get("file_sha256") != _sha256(prefix_path)
        or manifest_entry.get("prefix_sha256") != payload.get("prefix_sha256")
    ):
        raise RuntimeError("treatment prefix manifest binding mismatch")


def _durable_optimizer_step(
    optimizer: torch.optim.Optimizer,
    status_path: Path,
    lifecycle: dict[str, Any],
    status_context: dict[str, Any],
    *,
    update: int,
    committed_updates: int,
    train_index: int,
    noise_seed: int,
) -> None:
    lifecycle.update(
        phase="step_pending",
        attempted_update=update,
        executed_update=update - 1,
        committed_update=committed_updates,
        train_index=train_index,
        noise_seed=noise_seed,
    )
    _atomic_json(status_path, {"status": "training", **lifecycle, **status_context})
    optimizer.step()
    lifecycle.update(phase="step_executed_uncommitted", executed_update=update)
    _atomic_json(status_path, {"status": "training", **lifecycle, **status_context})


def _durable_commit_update(
    *,
    status_path: Path,
    history_path: Path,
    history: list[dict[str, Any]],
    lifecycle: dict[str, Any],
    status_context: dict[str, Any],
    update: int,
) -> None:
    _atomic_json(history_path, {"history": history, "committed_updates": update})
    lifecycle.update(phase="update_committed", committed_update=update)
    _atomic_json(status_path, {"status": "training", **lifecycle, **status_context})


def _failure_status(
    lifecycle: dict[str, Any],
    histories: dict[str, list[dict[str, Any]]],
    latest_checkpoints: dict[str, dict[str, Any]],
    final_checkpoints: dict[str, dict[str, Any]],
    error: BaseException,
) -> dict[str, Any]:
    return {
        "status": "failed_terminal_non_resumable",
        **lifecycle,
        "completed_updates": {arm: len(history) for arm, history in histories.items()},
        "latest_checkpoints": latest_checkpoints,
        "final_checkpoints": final_checkpoints,
        "error_type": type(error).__name__,
        "error": str(error),
    }


def run(config_path: Path, output_dir: Path) -> dict[str, Any]:
    experiment = load_json(config_path)
    if experiment.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("training schema mismatch")
    protocol = experiment["protocol"]
    _check_protocol(protocol)
    if torch.cuda.device_count() != 1 or os.environ.get("CLEARML_REQUIRE_ONLINE") != "1":
        raise RuntimeError("matched A/B requires exactly one visible GPU and online ClearML")
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"refusing to reuse training output {output_dir}")
    output_dir.mkdir(parents=True)
    status_path = output_dir / "status.json"
    _atomic_json(status_path, {"status": "initializing", "completed_updates": 0})
    tracker = None
    histories: dict[str, list[dict[str, Any]]] = {}
    final_checkpoints: dict[str, dict[str, Any]] = {}
    latest_checkpoint_bindings: dict[str, dict[str, Any]] = {}
    lifecycle: dict[str, Any] = {
        "phase": "initializing",
        "arm": None,
        "attempted_update": 0,
        "executed_update": 0,
        "committed_update": 0,
    }
    previous_handlers: dict[int, Any] = {}

    def terminate(signum, _frame):
        raise InterruptedError(f"received termination signal {signum}")

    for signum in (signal.SIGTERM, signal.SIGINT):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, terminate)
    try:
        source = experiment["source"]
        if source.get("checkpoint") != "epoch_snapshots/epoch_0006/ema_last_model.pth":
            raise ValueError("training source is not reviewed EMA6")
        if set(source.get("sha256", {})) != SOURCE_SHA_KEYS:
            raise ValueError("source SHA map is incomplete or has unreviewed entries")
        run_dir = Path(source["run_dir"])
        for relative, expected in source["sha256"].items():
            if _sha256(run_dir / relative) != expected:
                raise ValueError(f"source SHA mismatch for {relative}")
        preflight = experiment["preflight"]
        preflight_path = Path(preflight["path"])
        if _sha256(preflight_path) != preflight["sha256"]:
            raise ValueError("actual preflight binding SHA mismatch")
        preflight_payload = json.loads(preflight_path.read_text())
        if (
            preflight_payload.get("status") != "preflight_passed"
            or preflight_payload.get("optimizer_steps") != 0
            or preflight_payload.get("test_2023_accessed") is not False
            or preflight_payload.get("verified_sha256") != source["sha256"]
            or preflight_payload.get("production_replay_max_abs") != 0.0
        ):
            raise ValueError("actual preflight did not establish the required zero-update law")

        repository_root = config_path.parents[2]
        audit = experiment["product_scale_audit"]
        for key, sha_key in (("path", "sha256"), ("script_path", "script_sha256")):
            if _sha256(repository_root / audit[key]) != audit[sha_key]:
                raise ValueError(f"product scale binding mismatch for {key}")
        code_identity = {
            "config_sha256": _sha256(config_path),
            "training_sha256": _sha256(Path(__file__)),
            "geometry_score_sha256": _sha256(repository_root / "assim_lib/direct_dynamics_geometry_score.py"),
            "suffix_sha256": _sha256(repository_root / "assim_lib/direct_dynamics_suffix.py"),
            "preflight_result_sha256": preflight["sha256"],
        }
        if (
            preflight_payload["code_identity"].get("geometry_score_sha256")
            != code_identity["geometry_score_sha256"]
            or preflight_payload["code_identity"].get("suffix_sha256")
            != code_identity["suffix_sha256"]
        ):
            raise ValueError("training score/suffix code differs from actual preflight")

        metadata = json.loads((run_dir / "metadata.json").read_text())
        training_config = metadata["training_config"]
        model_config = TrainingConfig.from_dict(training_config)
        if (
            tuple(model_config.image_size) != (320, 256)
            or model_config.in_channels != 23
            or model_config.out_channels != 6
            or model_config.training_objective != "flow"
            or model_config.structured_velocity_parameterization != "raw"
        ):
            raise ValueError("EMA6 model contract mismatch")
        seed_everything(int(protocol["seed"]))
        dataset = build_dataset(metadata["data_config"], split="train")
        sentinel = validate_direct_dataset(dataset)
        schedule = _schedule_indices(len(dataset), int(protocol["updates_per_arm"]))
        if dataset.split != "train":
            raise RuntimeError("training dataset escaped train split")

        device = torch.device("cuda:0")
        tracker = ClearMLTracker(
            experiment["project_name"],
            f"{experiment['task_name']}-{output_dir.name}",
            tags=experiment["clearml"]["tags"],
            env_path=experiment["clearml"]["env_path"],
        )
        tracker.connect("matched_ab_contract", {"source": source, "protocol": protocol, "code": code_identity})
        sampler = load_sampler(str(run_dir), source["checkpoint"], training_config, device=device)
        frozen_model = sampler.model.eval()
        for parameter in frozen_model.parameters():
            parameter.requires_grad_(False)
        source_state = frozen_model.state_dict()
        grid = make_normalized_xy_grid(*model_config.image_size, device=device, dtype=torch.float32)
        mask = torch.from_numpy(~np.load(metadata["data_config"]["mask_path"])).unsqueeze(0)
        cell_area, cell_area_evidence = _cell_area_proxy(
            Path(experiment["cell_geometry"]["latitude_path"]),
            Path(experiment["cell_geometry"]["longitude_path"]),
            mask,
            tuple(model_config.image_size),
            device,
        )
        if any(
            cell_area_evidence.get(key) != preflight_payload["cell_area"].get(key)
            for key in ("latitude_sha256", "longitude_sha256", "kind")
        ):
            raise ValueError("cell-area geometry differs from admitted preflight")
        means, stds = _repeat_field_stats(dataset.means), _repeat_field_stats(dataset.stds)
        prefix_root = output_dir / "prefix_cache"
        diagnostics_root = output_dir / "training_samples"
        prefix_root.mkdir()
        diagnostics_root.mkdir()
        prefix_manifest_path = output_dir / "prefix_manifest.json"
        prefix_manifest = {
            "schema_version": "matched_ab_prefix_cache_v1",
            "source_checkpoint_sha256": source["sha256"][source["checkpoint"]],
            "code_identity": code_identity,
            "entries": [],
        }
        _atomic_json(prefix_manifest_path, prefix_manifest)
        started = time.monotonic()
        shared_prefix_seconds = 0.0
        arm_suffix_seconds = {arm: 0.0 for arm in protocol["arms"]}
        torch.cuda.reset_peak_memory_stats(device)

        for arm in protocol["arms"]:
            lifecycle.update(
                phase="arm_initializing",
                arm=arm,
                attempted_update=0,
                executed_update=0,
                committed_update=0,
            )
            _atomic_json(status_path, {"status": "training", **lifecycle})
            candidate = copy.deepcopy(frozen_model).eval()
            for parameter in candidate.parameters():
                parameter.requires_grad_(True)
            candidate_state = candidate.state_dict()
            if source_state.keys() != candidate_state.keys() or any(
                not torch.equal(source_state[key], candidate_state[key]) for key in source_state
            ):
                raise RuntimeError(f"{arm} did not start exactly from EMA6")
            optimizer = torch.optim.AdamW(
                candidate.parameters(),
                lr=float(protocol["learning_rate"]),
                weight_decay=float(protocol["weight_decay"]),
            )
            history = []
            if arm == "treatment":
                prefix_manifest = json.loads(prefix_manifest_path.read_text())
                if (
                    prefix_manifest.get("schema_version") != "matched_ab_prefix_cache_v1"
                    or prefix_manifest.get("source_checkpoint_sha256")
                    != source["sha256"][source["checkpoint"]]
                    or prefix_manifest.get("code_identity") != code_identity
                    or len(prefix_manifest.get("entries", [])) != len(schedule)
                ):
                    raise RuntimeError("control prefix manifest is incomplete or mismatched")
            for update, train_index in enumerate(schedule, start=1):
                batch = _load_item(dataset, train_index, device)
                condition, valid, truth = batch["condition"], batch["valid"], batch["truth"]
                noise_seed = int(protocol["seed"]) + update
                generator = torch.Generator(device=device).manual_seed(noise_seed)
                noise = torch.randn(
                    (int(protocol["members"]), DIRECT_OUTPUT_CHANNELS, *model_config.image_size),
                    generator=generator,
                    device=device,
                    dtype=torch.float32,
                )
                input_sha256 = _input_fingerprints(batch, noise)
                prefix_path = prefix_root / f"update_{update:04d}.pth"
                if arm == "control":
                    prefix_started = time.monotonic()
                    prefix = torch.stack(
                        [
                            frozen_prefix(
                                frozen_model,
                                noise[member : member + 1],
                                condition,
                                valid,
                                grid,
                            )[0]
                            for member in range(int(protocol["members"]))
                        ],
                        dim=0,
                    ).unsqueeze(0)
                    prefix_seconds = time.monotonic() - prefix_started
                    shared_prefix_seconds += prefix_seconds
                    prefix_sha256 = _tensor_sha256(prefix)
                    _atomic_torch_save(
                        {
                            "prefix": prefix.detach().cpu(),
                            "prefix_sha256": prefix_sha256,
                            "train_index": int(train_index),
                            "noise_seed": noise_seed,
                            "input_sha256": input_sha256,
                            "source_checkpoint_sha256": source["sha256"][source["checkpoint"]],
                            "code_identity": code_identity,
                        },
                        prefix_path,
                    )
                    prefix_manifest["entries"].append(
                        {
                            "update": update,
                            "train_index": int(train_index),
                            "noise_seed": noise_seed,
                            "input_sha256": input_sha256,
                            "prefix_sha256": prefix_sha256,
                            "file_sha256": _sha256(prefix_path),
                        }
                    )
                    _atomic_json(prefix_manifest_path, prefix_manifest)
                else:
                    prefix_seconds = 0.0
                    payload = torch.load(prefix_path, map_location="cpu", weights_only=True)
                    manifest_entry = prefix_manifest["entries"][update - 1]
                    _validate_prefix_cache_binding(
                        payload=payload,
                        manifest_entry=manifest_entry,
                        prefix_path=prefix_path,
                        update=update,
                        train_index=int(train_index),
                        noise_seed=noise_seed,
                        input_sha256=input_sha256,
                        source_checkpoint_sha256=source["sha256"][source["checkpoint"]],
                        code_identity=code_identity,
                    )
                    prefix = payload["prefix"].to(device=device, dtype=torch.float32)
                    if (
                        _tensor_sha256(prefix) != payload["prefix_sha256"]
                        or payload["prefix_sha256"] != manifest_entry["prefix_sha256"]
                    ):
                        raise RuntimeError("treatment prefix cache tensor hash mismatch")

                def physical(outputs: torch.Tensor) -> torch.Tensor:
                    return channel_denormalize(outputs.float(), means, stds)

                def score(outputs: torch.Tensor) -> torch.Tensor:
                    native, _, _ = proper_objective(outputs, truth, valid)
                    if arm == "control":
                        return native
                    geometry, _ = geometry_energy_score(
                        physical(outputs),
                        batch["physical_truth"],
                        valid,
                        batch["initial_sic"],
                        cell_area,
                        sic_scale=float(dataset.stds[0]),
                        sit_scale=float(dataset.stds[1]),
                        product_scale=float(protocol["product_scale"]),
                    )
                    return native + float(protocol["geometry_weight"]) * geometry

                optimizer.zero_grad(set_to_none=True)
                suffix_started = time.monotonic()
                outputs, objective, replay = exact_two_pass_score_vjp(
                    candidate,
                    prefix,
                    condition,
                    valid,
                    grid,
                    score,
                    replay_atol=0.0,
                )
                suffix_seconds = time.monotonic() - suffix_started
                arm_suffix_seconds[arm] += suffix_seconds
                gradient_norm = _gradient_norm(candidate)
                status_context = {
                    "completed_arms": list(histories),
                    "latest_checkpoints": latest_checkpoint_bindings,
                    "clearml_task_id": str(tracker.task.id),
                }
                _durable_optimizer_step(
                    optimizer,
                    status_path,
                    lifecycle,
                    status_context,
                    update=update,
                    committed_updates=len(history),
                    train_index=int(train_index),
                    noise_seed=noise_seed,
                )
                if not all(torch.isfinite(parameter).all() for parameter in candidate.parameters()):
                    raise FloatingPointError(f"non-finite {arm} parameter at update {update}")
                with torch.no_grad():
                    native, crps, energy = proper_objective(outputs, truth, valid)
                    geometry = geometry_energy_score(
                        physical(outputs), batch["physical_truth"], valid, batch["initial_sic"], cell_area,
                        sic_scale=float(dataset.stds[0]), sit_scale=float(dataset.stds[1]),
                        product_scale=float(protocol["product_scale"]),
                    )[0]
                record = {
                    "update": update,
                    "measurement_phase": "pre_step",
                    "model_completed_updates": update - 1,
                    "train_index": int(train_index),
                    "noise_seed": noise_seed,
                    "objective": float(objective.cpu()),
                    "native_objective": float(native.cpu()),
                    "fair_crps": float(crps.cpu()),
                    "joint_energy": float(energy.cpu()),
                    "geometry_energy": float(geometry.cpu()),
                    "gradient_norm": gradient_norm,
                    "suffix_replay_max_abs": replay["suffix_replay_max_abs"],
                    "shared_prefix_seconds": prefix_seconds,
                    "arm_suffix_seconds": suffix_seconds,
                    "elapsed_seconds": time.monotonic() - started,
                }
                if not all(math.isfinite(value) for value in record.values() if isinstance(value, float)):
                    raise FloatingPointError("non-finite training record")
                history.append(record)
                if update in protocol["diagnostic_updates"]:
                    _atomic_torch_save(
                        {
                            "arm": arm,
                            "update": update,
                            "measurement_phase": "pre_step",
                            "model_completed_updates": update - 1,
                            "train_index": int(train_index),
                            "members_normalized": outputs.cpu(),
                            "members_physical": physical(outputs).cpu(),
                            "truth_physical": batch["physical_truth"].cpu(),
                            "initial_sic": batch["initial_sic"].cpu(),
                            "valid": valid.cpu(),
                        },
                        diagnostics_root / f"{arm}_update_{update:04d}.pth",
                    )
                if update % int(protocol["checkpoint_every_updates"]) == 0:
                    checkpoint_path = output_dir / f"{arm}_latest_checkpoint.pth"
                    _atomic_torch_save(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "arm": arm,
                            "completed_updates": update,
                            "model": candidate.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "source": source,
                            "protocol": protocol,
                            "code_identity": code_identity,
                            "history": history,
                        },
                        checkpoint_path,
                    )
                    latest_checkpoint_bindings[arm] = {
                        "path": str(checkpoint_path),
                        "sha256": _sha256(checkpoint_path),
                        "completed_updates": update,
                    }
                status_context["latest_checkpoints"] = latest_checkpoint_bindings
                _durable_commit_update(
                    status_path=status_path,
                    history_path=output_dir / f"{arm}_history.json",
                    history=history,
                    lifecycle=lifecycle,
                    status_context=status_context,
                    update=update,
                )
                for key in (
                    "objective",
                    "native_objective",
                    "fair_crps",
                    "joint_energy",
                    "geometry_energy",
                    "gradient_norm",
                ):
                    tracker.report_scalar(
                        "matched_ab_training", f"{arm}_{key}", record[key], update
                    )
            histories[arm] = history
            final_path = output_dir / f"{arm}_final_checkpoint.pth"
            _atomic_torch_save(
                {
                    "schema_version": SCHEMA_VERSION,
                    "arm": arm,
                    "completed_updates": len(history),
                    "model": candidate.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "source": source,
                    "protocol": protocol,
                    "code_identity": code_identity,
                    "history": history,
                    "inference_law": "frozen_ema6_prefix12_plus_arm_candidate_suffix4",
                },
                final_path,
            )
            final_checkpoints[arm] = {"path": str(final_path), "sha256": _sha256(final_path)}
            final_checkpoints[arm]["completed_updates"] = len(history)
            lifecycle.update(phase="arm_complete")
            _atomic_json(
                status_path,
                {
                    "status": "training",
                    **lifecycle,
                    "completed_arms": list(histories),
                    "latest_checkpoints": latest_checkpoint_bindings,
                    "final_checkpoints": final_checkpoints,
                    "clearml_task_id": str(tracker.task.id),
                },
            )
            del optimizer, candidate
            torch.cuda.empty_cache()

        result = {
            "schema_version": SCHEMA_VERSION,
            "status": "training_complete_pending_paired_validation",
            "source": source,
            "preflight": preflight,
            "protocol": protocol,
            "code_identity": code_identity,
            "dataset_sentinel": sentinel,
            "schedule": schedule,
            "cell_area": cell_area_evidence,
            "completed_updates": {arm: len(history) for arm, history in histories.items()},
            "final_checkpoints": final_checkpoints,
            "peak_gpu_memory_mib": float(torch.cuda.max_memory_allocated(device) / 2**20),
            "elapsed_seconds": time.monotonic() - started,
            "timing": {
                "shared_prefix_seconds": shared_prefix_seconds,
                "arm_suffix_seconds": arm_suffix_seconds,
                "raw_arm_wall_times_not_a_matched_compute_claim": True,
            },
            "clearml_task_id": str(tracker.task.id),
            "test_2023_accessed": False,
        }
        _atomic_json(output_dir / "training_result.json", result)
        tracker.connect("matched_ab_training_result", result)
        tracker.upload_artifact("matched_ab_training_result", output_dir / "training_result.json")
        tracker.close()
        tracker = None
        _atomic_json(status_path, {"status": result["status"], "clearml_task_id": result["clearml_task_id"]})
        return result
    except BaseException as error:
        try:
            _atomic_json(
                status_path,
                _failure_status(
                    lifecycle,
                    histories,
                    latest_checkpoint_bindings,
                    final_checkpoints,
                    error,
                ),
            )
        except Exception:
            pass
        if tracker is not None:
            try:
                tracker.task.mark_failed(status_reason=f"{type(error).__name__}: {error}")
            except Exception:
                pass
        raise
    finally:
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)
        if tracker is not None:
            try:
                tracker.close()
            except Exception:
                pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve(), args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
