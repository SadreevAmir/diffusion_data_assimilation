"""Transactional matched full-CFM training admitted by a bound zero-step preflight.

The scientific forward/loss path remains in
``direct_dynamics_geometry_cfm_training``.  This module owns only lifecycle:
write-ahead state, durable per-update evidence, checkpoint verification,
ClearML ordering, failure cleanup, and terminal completion.
"""

from __future__ import annotations

import argparse
import json
import signal
from pathlib import Path
from typing import Any

import torch
from diffusers.training_utils import EMAModel

from .clearml_tracking import ClearMLTracker
from .config import TrainingConfig, load_json
from .data import build_dataset
from .direct_dynamics_training import (
    DIRECT_INPUT_CHANNELS,
    DIRECT_OUTPUT_CHANNELS,
    validate_direct_dataset,
)
from .direct_dynamics_geometry_cfm_training import (
    ARMS,
    SCHEMA_VERSION,
    _atomic_json,
    _batch_to_device,
    _gradient_norm,
    _make_loader,
    _merge_status,
    _one_forward,
    _prepare_static_contract,
    _record_terminal_failure,
    _save_checkpoint,
    _sha256,
    _tensor_sha256,
    _utc_now,
    make_matched_schedule,
    training_contract_sha256,
)
from .model_io import load_sampler
from .runtime import make_normalized_xy_grid


TRANSACTIONAL_SCHEMA_VERSION = f"{SCHEMA_VERSION}_transactional_training_v1"
SCIENTIFIC_RUNNER = Path(__file__).with_name("direct_dynamics_geometry_cfm_training.py")
OBJECTIVE = Path(__file__).with_name("direct_dynamics_geometry_cfm.py")


def _load_source_contract_without_model(experiment: dict[str, Any]):
    source = experiment["source"]
    run_dir = Path(source["run_dir"])
    expected_sha = source["sha256"]
    if set(expected_sha) != {
        "metadata.json",
        "config.json",
        "epoch_snapshots/epoch_0006/ema_last_model.pth",
    }:
        raise ValueError("source SHA map is incomplete")
    for relative, digest in expected_sha.items():
        if _sha256(run_dir / relative) != digest:
            raise ValueError(f"source SHA mismatch: {relative}")
    metadata = load_json(run_dir / "metadata.json")
    config = TrainingConfig.from_dict(metadata["training_config"])
    if (
        tuple(config.image_size) != (320, 256)
        or config.in_channels != DIRECT_INPUT_CHANNELS
        or config.out_channels != DIRECT_OUTPUT_CHANNELS
        or config.training_objective != "flow"
        or config.timestep_sampler != "beta"
        or tuple(config.timestep_beta_params) != (1.0, 1.5)
        or config.activation_checkpointing
    ):
        raise ValueError("EMA6 source training contract mismatch")
    dataset = build_dataset(metadata["data_config"], split="train")
    sentinel = validate_direct_dataset(dataset)
    return config, dataset, sentinel, metadata


def _validate_bound_preflight(
    experiment: dict[str, Any], schedule: dict[str, Any]
) -> dict[str, Any]:
    binding = experiment["required_preflight"]
    path = Path(binding["path"])
    if _sha256(path) != binding["sha256"]:
        raise ValueError("required zero-update preflight SHA mismatch")
    payload = json.loads(path.read_text())
    manifest = payload.get("implementation_manifest", {})
    selected_checkpoint = experiment["source"]["checkpoint"]
    selected_sha = experiment["source"]["sha256"].get(selected_checkpoint)
    expected = {
        "status": payload.get("status") == "passed",
        "optimizer_steps": payload.get("optimizer_steps") == 0,
        "test_2023_accessed": payload.get("test_2023_accessed") is False,
        "parameters_unchanged": payload.get("parameters_unchanged") is True,
        "contract": payload.get("training_contract_sha256")
        == training_contract_sha256(experiment),
        "schedule": payload.get("schedule_sha256") == schedule["sha256"],
        "runner": manifest.get("runner_sha256") == _sha256(SCIENTIFIC_RUNNER),
        "objective": manifest.get("objective_sha256") == _sha256(OBJECTIVE),
        "metric": manifest.get("metric_config_sha256")
        == experiment["metric_admission"]["sha256"],
        "checkpoint_path": manifest.get("source_checkpoint_relative_path")
        == selected_checkpoint,
        "checkpoint_sha": manifest.get("source_checkpoint_sha256") == selected_sha,
        "checkpoint_map": manifest.get("source_checkpoint_sha_map_entry") == selected_sha,
        "closed": payload.get("clearml_closed_before_terminal_success") is True,
    }
    failed = sorted(key for key, passed in expected.items() if not passed)
    if failed:
        raise ValueError(f"required preflight binding failed: {failed}")
    return payload


def _attempt_fingerprint(
    raw_batch: dict[str, Any],
    timesteps: torch.Tensor,
    noise_seed: int,
    dropout_seed: int,
) -> dict[str, Any]:
    return {
        "case_ids": list(raw_batch["meta"]["case_id"]),
        "timesteps_sha256": _tensor_sha256(timesteps),
        "noise_seed": int(noise_seed),
        "dropout_seed": int(dropout_seed),
    }


def _write_ahead_update(
    status_path: Path,
    arm: str,
    update: int,
    committed: dict[str, int],
    attempt: dict[str, Any],
) -> None:
    _merge_status(
        status_path,
        status="training_update_attempted_before_model_mutation",
        arm=arm,
        attempted_update=int(update),
        committed_updates_by_arm=dict(committed),
        optimizer_step_state="not_started",
        ema_step_state="not_started",
        mutation_evidence_committed=False,
        resume_permitted=False,
        attempt=attempt,
    )


def _run_optimizer_and_ema_with_boundaries(
    status_path: Path,
    arm: str,
    update: int,
    committed: dict[str, int],
    optimizer: torch.optim.Optimizer,
    ema: EMAModel,
    model: torch.nn.Module,
) -> None:
    _merge_status(
        status_path,
        status="optimizer_step_pending_unknown",
        arm=arm,
        attempted_update=int(update),
        committed_updates_by_arm=dict(committed),
        optimizer_step_state="pending_unknown",
        ema_step_state="not_started",
        mutation_evidence_committed=False,
        resume_permitted=False,
    )
    optimizer.step()
    _merge_status(
        status_path,
        status="optimizer_completed_ema_step_pending_unknown",
        arm=arm,
        attempted_update=int(update),
        committed_updates_by_arm=dict(committed),
        optimizer_step_state="completed",
        ema_step_state="pending_unknown",
        mutation_evidence_committed=False,
        resume_permitted=False,
    )
    ema.step(model.parameters())
    _merge_status(
        status_path,
        status="optimizer_and_ema_completed_pending_durable_evidence",
        arm=arm,
        attempted_update=int(update),
        committed_updates_by_arm=dict(committed),
        optimizer_step_state="completed",
        ema_step_state="completed",
        mutation_evidence_committed=False,
        resume_permitted=False,
    )


def _commit_update_evidence(
    status_path: Path,
    history_path: Path,
    arm: str,
    history: list[dict[str, Any]],
    committed: dict[str, int],
    checkpoints: dict[str, dict[str, dict[str, str]]],
) -> None:
    _atomic_json(history_path, history)
    committed[arm] = int(history[-1]["update"])
    _merge_status(
        status_path,
        status="training_update_durably_recorded_before_clearml",
        arm=arm,
        attempted_update=int(history[-1]["update"]),
        committed_updates_by_arm=dict(committed),
        optimizer_step_state="completed",
        ema_step_state="completed",
        mutation_evidence_committed=True,
        resume_permitted=False,
        latest_record=history[-1],
        checkpoints=checkpoints,
    )


def _verify_checkpoint_files(
    output_dir: Path,
    checkpoints: dict[str, dict[str, dict[str, str]]],
    expected_updates: list[int],
) -> None:
    expected_keys = {str(value) for value in expected_updates}
    for arm in ARMS:
        if set(checkpoints.get(arm, {})) != expected_keys:
            raise RuntimeError(f"{arm} checkpoint set is incomplete")
        for update, files in checkpoints[arm].items():
            root = output_dir / arm / f"update_{int(update):04d}"
            expected_names = {
                "model": "model.pth",
                "ema_model": "ema_model.pth",
                "optimizer_recovery": "optimizer_recovery.pth",
            }
            if set(files) != set(expected_names):
                raise RuntimeError(f"{arm} update {update} checkpoint manifest is incomplete")
            for key, name in expected_names.items():
                if _sha256(root / name) != files[key]:
                    raise RuntimeError(f"{arm} update {update} checkpoint SHA mismatch: {key}")


def _validate_completion(
    output_dir: Path,
    histories: dict[str, list[dict[str, Any]]],
    checkpoints: dict[str, dict[str, dict[str, str]]],
    protocol: dict[str, Any],
    committed: dict[str, int],
) -> None:
    expected_count = int(protocol["updates_per_arm"])
    expected_sequence = list(range(1, expected_count + 1))
    if set(histories) != set(ARMS) or set(committed) != set(ARMS):
        raise RuntimeError("matched completion is missing an arm")
    for arm in ARMS:
        if [int(record["update"]) for record in histories[arm]] != expected_sequence:
            raise RuntimeError(f"{arm} update sequence is incomplete or noncanonical")
        if committed[arm] != expected_count:
            raise RuntimeError(f"{arm} durable committed count is incomplete")
    for control, treatment in zip(
        histories["control"], histories["treatment"], strict=True
    ):
        for key in ("case_ids", "input_sha256", "attempt"):
            if control[key] != treatment[key]:
                raise RuntimeError(f"matched arm evidence diverged: {key}")
    _verify_checkpoint_files(output_dir, checkpoints, protocol["checkpoint_updates"])


def _close_and_mark_success(
    result_path: Path,
    status_path: Path,
    tracker: ClearMLTracker,
    result: dict[str, Any],
) -> dict[str, Any]:
    pending = dict(result, status="completed_pending_clearml_close")
    _atomic_json(result_path, pending)
    _merge_status(status_path, **pending)
    tracker.close()
    completed = dict(
        result,
        status="completed_pending_paired_validation",
        clearml_closed_before_terminal_success=True,
    )
    _atomic_json(result_path, completed)
    _merge_status(status_path, **completed)
    return completed


def run_training(config_path: Path, output_dir: Path) -> dict[str, Any]:
    (
        experiment,
        protocol,
        spec,
        treatment_weight,
        admission,
        _repository_root,
        _metric_path,
    ) = _prepare_static_contract(config_path)
    output_dir.mkdir(parents=True, exist_ok=False)
    status_path = output_dir / "status.json"
    _merge_status(
        status_path,
        status="initializing_before_gpu",
        test_2023_accessed=False,
        committed_updates_by_arm={arm: 0 for arm in ARMS},
    )
    tracker: ClearMLTracker | None = None
    previous = {
        number: signal.getsignal(number) for number in (signal.SIGINT, signal.SIGTERM)
    }

    def terminate(signum, _frame):
        raise InterruptedError(f"matched training received signal {signum}")

    for number in previous:
        signal.signal(number, terminate)
    try:
        tracker = ClearMLTracker(
            experiment["project_name"],
            f"{experiment['task_name']}-{output_dir.name}",
            tags=[*experiment["clearml"]["tags"], "transactional-lifecycle"],
            env_path=experiment["clearml"]["env_path"],
        )
        tracker.connect(
            "matched_full_cfm_contract",
            {
                "source": experiment["source"],
                "protocol": protocol,
                "metric": admission,
                "transactional_training_sha256": _sha256(Path(__file__)),
            },
        )
        _merge_status(
            status_path,
            status="tracker_reserved_before_gpu",
            clearml_task_id=str(getattr(tracker.task, "id", "")),
        )
        model_config, dataset, sentinel, metadata = _load_source_contract_without_model(
            experiment
        )
        schedule = make_matched_schedule(len(dataset), protocol)
        _validate_bound_preflight(experiment, schedule)
        _atomic_json(output_dir / "schedule.json", schedule)
        tracker.connect("matched_schedule", schedule)
        if torch.cuda.device_count() != 1:
            raise RuntimeError("matched training requires exactly one visible GPU")
        device = torch.device("cuda:0")
        grid = make_normalized_xy_grid(
            *model_config.image_size, device=device, dtype=torch.float32
        )
        histories: dict[str, list[dict[str, Any]]] = {}
        checkpoints: dict[str, dict[str, dict[str, str]]] = {}
        committed = {arm: 0 for arm in ARMS}

        for arm in ARMS:
            weight = 0.0 if arm == "control" else treatment_weight
            sampler = load_sampler(
                experiment["source"]["run_dir"],
                experiment["source"]["checkpoint"],
                metadata["training_config"],
                device=device,
            )
            model = sampler.model.train()
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(protocol["learning_rate"]),
                weight_decay=float(protocol["weight_decay"]),
            )
            ema = EMAModel(model.parameters(), decay=float(protocol["ema_decay"]))
            ema.to(device)
            loader = _make_loader(dataset, schedule, protocol)
            history: list[dict[str, Any]] = []
            checkpoints[arm] = {}
            arm_dir = output_dir / arm
            arm_dir.mkdir(parents=True, exist_ok=False)
            history_path = arm_dir / "history.json"
            _atomic_json(history_path, history)

            for update_index, raw_batch in enumerate(loader, start=1):
                timesteps = torch.tensor(
                    schedule["timesteps"][update_index - 1],
                    dtype=torch.float32,
                    device=device,
                )
                attempt = _attempt_fingerprint(
                    raw_batch,
                    timesteps,
                    schedule["noise_seeds"][update_index - 1],
                    schedule["dropout_seeds"][update_index - 1],
                )
                _write_ahead_update(
                    status_path, arm, update_index, committed, attempt
                )
                batch = _batch_to_device(raw_batch, device)
                optimizer.zero_grad(set_to_none=True)
                loss, diagnostics, evidence = _one_forward(
                    model,
                    batch,
                    timesteps,
                    schedule["noise_seeds"][update_index - 1],
                    schedule["dropout_seeds"][update_index - 1],
                    grid,
                    weight,
                    spec,
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite matched full-CFM training loss")
                loss.backward()
                unclipped_norm = _gradient_norm(model)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(protocol["gradient_clip_norm"])
                )
                _run_optimizer_and_ema_with_boundaries(
                    status_path,
                    arm,
                    update_index,
                    committed,
                    optimizer,
                    ema,
                    model,
                )
                record = {
                    "update": update_index,
                    "loss": float(loss.detach().cpu()),
                    "native": float(diagnostics["native"].detach().cpu()),
                    "geometry": float(diagnostics["geometry"].detach().cpu()),
                    "gradient_norm_before_clip": unclipped_norm,
                    "case_ids": list(raw_batch["meta"]["case_id"]),
                    "attempt": attempt,
                    "input_sha256": {
                        "truth": _tensor_sha256(evidence["truth"]),
                        "noise": _tensor_sha256(evidence["noise"]),
                        "timesteps": _tensor_sha256(evidence["timesteps"]),
                        "condition": _tensor_sha256(evidence["condition"]),
                        "valid": _tensor_sha256(evidence["valid"]),
                        "initial_sic": _tensor_sha256(evidence["initial_sic"]),
                        "initial_sit": _tensor_sha256(evidence["initial_sit"]),
                        "model_input": _tensor_sha256(evidence["model_input"]),
                        "target_velocity": _tensor_sha256(evidence["target_velocity"]),
                    },
                }
                history.append(record)
                if update_index in protocol["checkpoint_updates"]:
                    checkpoints[arm][str(update_index)] = _save_checkpoint(
                        output_dir, arm, update_index, model, ema, optimizer
                    )
                _commit_update_evidence(
                    status_path,
                    history_path,
                    arm,
                    history,
                    committed,
                    checkpoints,
                )
                # External telemetry is deliberately last: it can never lead
                # the durable scientific record.
                tracker.report_scalar(
                    "matched_full_cfm/loss", arm, record["loss"], update_index
                )
                tracker.report_scalar(
                    "matched_full_cfm/native", arm, record["native"], update_index
                )
                tracker.report_scalar(
                    "matched_full_cfm/geometry", arm, record["geometry"], update_index
                )
            histories[arm] = history
            del ema, optimizer, model, sampler, loader
            torch.cuda.empty_cache()

        _validate_completion(
            output_dir, histories, checkpoints, protocol, committed
        )
        result = {
            "schema_version": TRANSACTIONAL_SCHEMA_VERSION,
            "resume_supported": False,
            "checkpoint_semantics": "bounded_nonresumable_forensic_snapshots_only",
            "completed_updates_per_arm": int(protocol["updates_per_arm"]),
            "committed_updates_by_arm": committed,
            "schedule_sha256": schedule["sha256"],
            "checkpoints": checkpoints,
            "dataset_sentinel": sentinel,
            "test_2023_accessed": False,
            "training_contract_sha256": training_contract_sha256(experiment),
            "config_sha256": _sha256(config_path),
            "scientific_runner_sha256": _sha256(SCIENTIFIC_RUNNER),
            "objective_sha256": _sha256(OBJECTIVE),
            "transactional_training_sha256": _sha256(Path(__file__)),
            "completed_at": _utc_now(),
        }
        completed = _close_and_mark_success(
            output_dir / "training_result.json",
            status_path,
            tracker,
            result,
        )
        tracker = None
        return completed
    except BaseException as error:
        _record_terminal_failure(status_path, tracker, error)
        raise
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = run_training(args.config.resolve(), args.output_dir.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
