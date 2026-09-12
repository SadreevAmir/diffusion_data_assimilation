"""Bounded 64-update coarse-budget refinement on the proved cascade.

Only the terminal coarse RK4 interval is trainable.  For every fixed train
case, the frozen production cascade first supplies C0 and the raw colored
fine2048 draw.  The candidate changes C0 through the reviewed anchored chart;
the fine allocation remains detached and is never resampled at Ctheta.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import signal
import time
from pathlib import Path
from typing import Any

import torch

from .clearml_tracking import ClearMLTracker
from .config import load_json, merge_config_overrides, resolve_path
from .data import build_dataset
from .direct_dynamics_cascade_coarse import lossless_coarse_condition
from .direct_dynamics_cascade_coarse_fine_boundary_audit import (
    _finish_success,
    _sha256,
    _strict_atomic_json,
    _terminate,
)
from .direct_dynamics_cascade_coarse_proper_refinement import (
    _atomic_torch_save,
    proper_objective,
)
from .direct_dynamics_cascade_contract import (
    forecast_contract_from_data_config,
    forecast_contract_sha256,
)
from .direct_dynamics_cascade_end_to_end import load_cascade_predictor
from .direct_dynamics_cascade_fine_training import (
    EXPECTED_SPLIT_LENGTHS,
    _calendar_inventory,
    _canonical_sha256,
    _clean_code_identity,
    _sha256_file,
)
from .direct_dynamics_coarse_budget_actual_gpu_preflight import (
    _load_normalization_binding,
    _repeat_case,
    _validate_config as _validate_base_config,
    _validate_files,
    _validate_proved_contract,
)
from .direct_dynamics_coarse_budget_refinement_integration import (
    anchored_canonical_physical_coarse,
    frozen_allocation_physical_law,
)
from .direct_dynamics_coarse_budget_production_replay import (
    sequential_terminal_candidate_control,
)
from .direct_dynamics_fine_support_proper_admission import apply_training_sic_decoder
from .direct_dynamics_sic_support_decoder_scoring import canonical_physical_decode
from .direct_dynamics_training import validate_direct_dataset
from .runtime import make_normalized_xy_grid, seed_everything


EXPECTED_PROTOCOL = {
    "split": "train",
    "updates": 64,
    "cases_per_update": 1,
    "member_indices": [0, 1, 2, 3],
    "learning_rate": 1e-5,
    "weight_decay": 0.0,
    "optimizer": "AdamW",
    "coarse_rk4_timepoints": 17,
    "fine_rk4_timepoints": 33,
    "network_precision": "bf16",
    "state_precision": "fp32",
    "score_precision": "fp64",
    "objective": "0.75_fair_crps_plus_0.25_joint_energy_all_water",
    "fine_allocation": "raw_colored_fine2048_at_frozen_C0_detached",
    "coarse_chart": "anchored_canonical_physical",
    "noise_rule": "case_id_member_index_stage_hash",
    "test_2023": "closed",
    "extension": "forbidden",
    "max_gpu_minutes": 180,
}


def _indices_sha256(indices: list[int]) -> str:
    return hashlib.sha256(
        ",".join(str(value) for value in indices).encode()
    ).hexdigest()


def _validate_training_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != "coarse_budget_train64_v1":
        raise ValueError("unreviewed coarse-budget training schema")
    if config.get("protocol") != EXPECTED_PROTOCOL:
        raise ValueError("coarse-budget training protocol differs from review")
    indices = config.get("train_indices")
    if (
        not isinstance(indices, list)
        or len(indices) != 64
        or len(set(indices)) != 64
        or any(not isinstance(index, int) or index < 0 for index in indices)
    ):
        raise ValueError("training requires 64 explicit unique nonnegative indices")
    if _indices_sha256(indices) != config.get("train_indices_sha256"):
        raise ValueError("explicit train-index schedule SHA mismatch")
    if config.get("checkpoint_updates") != [16, 32, 48, 64]:
        raise ValueError("unreviewed checkpoint schedule")


def _load_base_config(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    spec = config["base_preflight_config"]
    path = (config_path.parent / spec["path"]).resolve()
    if not path.is_file() or _sha256(path) != spec["sha256"]:
        raise ValueError("base preflight config SHA mismatch")
    base = load_json(path)
    _validate_base_config(base)
    return base


def _validate_actual_admission(config: dict[str, Any]) -> dict[str, Any]:
    spec = config["actual_admission"]
    for key in ("metrics", "status", "exit"):
        path = Path(spec[f"{key}_path"])
        if not path.is_file() or _sha256(path) != spec[f"{key}_sha256"]:
            raise ValueError(f"actual admission {key} binding differs")
    metrics = load_json(Path(spec["metrics_path"]))
    status = load_json(Path(spec["status_path"]))
    exit_record = load_json(Path(spec["exit_path"]))
    step0 = metrics.get("step0", {})
    if (
        metrics.get("status") != "preflight_passed_pending_astra_review"
        or metrics.get("code_identity", {}).get("git_commit") != spec["code_commit"]
        or metrics.get("clearml_task_id") != spec["clearml_task_id"]
        or metrics.get("optimizer_steps") != 0
        or metrics.get("test_2023_used") is not False
        or metrics.get("fine_resampled_for_candidate") is not False
        or any(
            step0.get(name) != 0.0
            for name in (
                "candidate_control_max_abs",
                "production_coarse_replay_max_abs",
                "scored_law_replay_max_abs",
            )
        )
        or not math.isfinite(float(step0.get("terminal_parameter_gradient_norm", math.nan)))
        or float(step0.get("terminal_parameter_gradient_norm", 0.0)) <= 0
        or status.get("status") != "complete"
        or exit_record.get("controller_exit_code") != 0
    ):
        raise ValueError("actual zero-update admission did not pass its frozen contract")
    return {key: spec[key] for key in sorted(spec)}


def _scored_candidate(
    production_forecast: torch.Tensor,
    production_coarse: torch.Tensor,
    control_coarse: torch.Tensor,
    candidate_coarse: torch.Tensor,
    truth: torch.Tensor,
    valid: torch.Tensor,
    means: torch.Tensor,
    stds: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    anchored_candidate = production_coarse.detach() + (
        candidate_coarse - control_coarse.detach()
    )
    base_coarse, candidate_physical_coarse = anchored_canonical_physical_coarse(
        production_coarse.detach(), anchored_candidate, means, stds
    )
    base_physical = canonical_physical_decode(
        production_forecast.detach(), means, stds
    ).detach()
    truth_physical = canonical_physical_decode(truth, means, stds)
    candidate_physical = frozen_allocation_physical_law(
        base_physical, base_coarse, candidate_physical_coarse, valid
    )
    scale = stds.double().reshape(1, 1, 6, 1, 1)
    objective, crps, energy = proper_objective(
        candidate_physical / scale,
        truth_physical / scale[:, 0],
        valid.double(),
    )
    return candidate_physical, objective, crps, energy


def _mark_failed(
    tracker: ClearMLTracker | None,
    status_path: Path,
    payload: dict[str, Any],
    error: BaseException,
) -> None:
    failure = {
        **payload,
        "status": "failed",
        "error_type": type(error).__name__,
        "error": str(error)[:2000],
    }
    try:
        _strict_atomic_json(status_path, failure)
    except Exception:
        pass
    if tracker is not None:
        try:
            tracker.task.mark_failed(status_reason=str(error)[:1000])
        except Exception:
            pass
        try:
            tracker.close()
        except Exception:
            pass


def run(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = load_json(config_path)
    _validate_training_config(config)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("coarse-budget training requires one visible GPU")
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"refusing to reuse output directory {output_dir}")
    output_dir.mkdir(parents=True)
    status_path = output_dir / "status.json"
    reservation = {"status": "reserved", "completed_updates": 0}
    _strict_atomic_json(status_path, reservation)
    tracker: ClearMLTracker | None = None
    history: list[dict[str, Any]] = []
    checkpoint_sha256: dict[str, str] = {}
    try:
        started = time.monotonic()
        repo = Path(__file__).resolve().parents[1]
        code_identity = _clean_code_identity(repo)
        base = _load_base_config(config, config_path)
        admission = _validate_actual_admission(config)
        parent = _validate_proved_contract(base)
        coarse_spec = base["source"]["coarse"]
        fine_spec = base["source"]["fine"]
        source_files = {
            "coarse": _validate_files(coarse_spec, "coarse"),
            "fine": _validate_files(fine_spec, "fine"),
        }
        means, stds, normalization = _load_normalization_binding(base, repo)
        data_path = resolve_path(base["data_config"], config_path.parent)
        if _sha256_file(data_path) != base["data_config_sha256"]:
            raise ValueError("data config SHA mismatch")
        data_config = merge_config_overrides(
            load_json(data_path), base.get("data_overrides")
        )
        contract_sha = forecast_contract_sha256(
            forecast_contract_from_data_config(data_config)
        )
        if contract_sha != base["source"]["forecast_contract_sha256"]:
            raise ValueError("effective forecast contract differs from source")
        seed_everything(73127)
        dataset = build_dataset(data_config, split="train")
        if len(dataset) != EXPECTED_SPLIT_LENGTHS["train"]:
            raise ValueError("train archive length differs from audited inventory")
        indices = config["train_indices"]
        if indices[-1] >= len(dataset):
            raise ValueError("train-index schedule exceeds the audited archive")
        sentinel = {
            "dataset": validate_direct_dataset(dataset),
            "calendar": _calendar_inventory(dataset, split="train"),
            "data_config_sha256": _canonical_sha256(data_config),
        }
        device = torch.device("cuda:0")
        tracker = ClearMLTracker(
            config["project_name"],
            f"{config['task_name']}-{output_dir.name}",
            tags=config["clearml"]["tags"],
            env_path=config["clearml"]["env_path"],
        )
        tracker.connect("training_contract", config)
        _strict_atomic_json(
            status_path,
            {
                **reservation,
                "status": "running",
                "code_identity": code_identity,
                "clearml_task_id": str(tracker.task.id),
            },
        )
        coarse_root = Path(coarse_spec["run_dir"])
        fine_root = Path(fine_spec["run_dir"])
        predictor = load_cascade_predictor(
            coarse_run_dir=str(coarse_root),
            coarse_checkpoint_name=coarse_spec["checkpoint"],
            coarse_model_config=load_json(coarse_root / "config.json"),
            coarse_checkpoint_sha256=coarse_spec["checkpoint_sha256"],
            fine_run_dir=str(fine_root),
            fine_checkpoint_name=fine_spec["checkpoint"],
            fine_model_config=load_json(fine_root / "config.json"),
            fine_checkpoint_sha256=fine_spec["checkpoint_sha256"],
            expected_coarse_code_commit=coarse_spec["code_commit"],
            expected_fine_code_commit=fine_spec["code_commit"],
            replay_code_commit=code_identity["git_commit"],
            expected_forecast_contract_sha256=contract_sha,
            device=device,
            expected_fine_conditioning_implementation_sha256=fine_spec[
                "implementation_sha256"
            ]["fine_conditioning"],
            expected_fine_preconditioning_implementation_sha256=fine_spec[
                "implementation_sha256"
            ]["fine_preconditioning"],
            expected_fine_colored_implementation_sha256=fine_spec[
                "implementation_sha256"
            ]["fine_colored"],
        )
        frozen_coarse = predictor.coarse_sampler.sampler.model.eval()
        frozen_fine = predictor.fine_sampler.sampler.model.eval()
        for parameter in (*frozen_coarse.parameters(), *frozen_fine.parameters()):
            parameter.requires_grad_(False)
        terminal_coarse = copy.deepcopy(frozen_coarse).train()
        for parameter in terminal_coarse.parameters():
            parameter.requires_grad_(True)
        frozen_state = frozen_coarse.state_dict()
        terminal_state = terminal_coarse.state_dict()
        if frozen_state.keys() != terminal_state.keys() or any(
            not torch.equal(frozen_state[name], terminal_state[name])
            for name in frozen_state
        ):
            raise RuntimeError("terminal model does not start exactly at coarse EMA9711")
        optimizer = torch.optim.AdamW(
            terminal_coarse.parameters(),
            lr=EXPECTED_PROTOCOL["learning_rate"],
            weight_decay=0.0,
        )
        means = means.to(device)
        stds = stds.to(device)
        grid: torch.Tensor | None = None
        members = tuple(EXPECTED_PROTOCOL["member_indices"])
        torch.cuda.reset_peak_memory_stats(device)

        for update, index in enumerate(indices, start=1):
            raw = dataset[index]
            condition = raw["structured_conditioning"].unsqueeze(0).float()
            valid = raw["valid_mask"].unsqueeze(0)[:, :1].float()
            truth = raw["truth"].unsqueeze(0).float()
            case_id = str(raw.get("meta", {}).get("case_id", f"train_index_{index}"))
            with torch.no_grad(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16
            ):
                production = predictor.sample_ensemble(
                    member_indices=members,
                    storage_device=device,
                    structured_conditioning=condition.to(device),
                    valid_mask=valid.to(device),
                    case_ids=(case_id,),
                    coarse_num_timesteps=17,
                    fine_num_timesteps=33,
                    device=device,
                    method="rk4",
                    rtol=1e-5,
                    atol=1e-6,
                    end_time=0.0,
                )
            condition_m = _repeat_case(condition.to(device), len(members))
            valid_m = _repeat_case(valid.to(device), len(members))
            encoded, active, _ = lossless_coarse_condition(condition_m, valid_m)
            if grid is None:
                grid = make_normalized_xy_grid(
                    *encoded.shape[-2:], device=device, dtype=torch.float32
                )
            candidate_flat, control_flat, prefix = sequential_terminal_candidate_control(
                frozen_coarse,
                terminal_coarse,
                production["raw_coarse_noise"].flatten(0, 1),
                encoded,
                active,
                grid,
            )
            if prefix.requires_grad:
                raise RuntimeError("frozen prefix unexpectedly retained a graph")
            candidate = candidate_flat.unflatten(0, (1, len(members)))
            control = control_flat.unflatten(0, (1, len(members)))
            production_coarse = production["coarse"]
            replay_error = float((control - production_coarse).abs().max())
            if replay_error != 0.0:
                raise RuntimeError(f"production coarse replay changed at update {update}")
            candidate_physical, objective, crps, energy = _scored_candidate(
                production["forecast"],
                production_coarse,
                control,
                candidate,
                truth.to(device),
                valid.to(device),
                means,
                stds,
            )
            if update == 1:
                with torch.no_grad():
                    base_coarse, _ = anchored_canonical_physical_coarse(
                        production_coarse.detach(), production_coarse.detach(), means, stds
                    )
                    base_physical = canonical_physical_decode(
                        production["forecast"], means, stds
                    )
                    control_physical = apply_training_sic_decoder(
                        base_physical, base_coarse, valid.to(device)
                    )
                if not torch.equal(candidate_physical.detach(), control_physical):
                    raise RuntimeError("training step zero changed the scored law")
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            gradients = [
                parameter.grad for parameter in terminal_coarse.parameters()
                if parameter.grad is not None
            ]
            if not gradients or not all(torch.isfinite(value).all() for value in gradients):
                raise FloatingPointError(f"invalid terminal gradient at update {update}")
            gradient_norm = float(
                torch.sqrt(sum(value.double().square().sum() for value in gradients))
            )
            if not math.isfinite(gradient_norm) or gradient_norm <= 0:
                raise FloatingPointError(f"dead terminal gradient at update {update}")
            if any(parameter.grad is not None for parameter in frozen_coarse.parameters()):
                raise RuntimeError("gradient leaked into frozen coarse model")
            if any(parameter.grad is not None for parameter in frozen_fine.parameters()):
                raise RuntimeError("gradient leaked into frozen fine model")
            optimizer.step()
            if not all(torch.isfinite(parameter).all() for parameter in terminal_coarse.parameters()):
                raise FloatingPointError(f"non-finite terminal model at update {update}")
            record = {
                "update": update,
                "train_index": index,
                "case_id": case_id,
                "objective": float(objective.detach()),
                "fair_crps": float(crps.detach()),
                "joint_energy": float(energy.detach()),
                "gradient_norm": gradient_norm,
                "production_replay_max_abs": replay_error,
                "elapsed_seconds": time.monotonic() - started,
            }
            history.append(record)
            if update in config["checkpoint_updates"]:
                name = f"terminal_model_update_{update}.pth"
                checkpoint_sha256[name] = _atomic_torch_save(
                    {
                        "model": terminal_coarse.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "completed_updates": update,
                        "protocol": config["protocol"],
                        "train_indices": indices,
                        "code_identity": code_identity,
                        "actual_admission": admission,
                        "source": base["source"],
                    },
                    output_dir / name,
                )
            _strict_atomic_json(
                status_path,
                {
                    "status": "running",
                    "completed_updates": update,
                    "last": record,
                    "checkpoint_sha256": checkpoint_sha256,
                    "clearml_task_id": str(tracker.task.id),
                },
            )
            for name in ("objective", "fair_crps", "joint_energy", "gradient_norm"):
                tracker.report_scalar("coarse_budget_train64", name, record[name], update)

        final_name = "terminal_model_update_64.pth"
        if final_name not in checkpoint_sha256:
            raise RuntimeError("bounded final checkpoint was not saved")
        result = {
            "status": "training_complete_pending_paired_validation",
            "scientific_role": "bounded_train_only_coarse_budget_refinement",
            "completed_updates": 64,
            "extension_allowed": False,
            "test_2023_used": False,
            "fine_resampled_for_candidate": False,
            "code_identity": code_identity,
            "actual_admission": admission,
            "proved_source_contract": parent,
            "source_files_sha256": source_files,
            "normalization": normalization,
            "dataset_sentinel": sentinel,
            "effective_forecast_contract_sha256": contract_sha,
            "protocol": config["protocol"],
            "train_indices": indices,
            "train_indices_sha256": config["train_indices_sha256"],
            "history": history,
            "checkpoint_sha256": checkpoint_sha256,
            "peak_gpu_memory_mib": float(torch.cuda.max_memory_allocated(device) / 2**20),
            "elapsed_seconds": time.monotonic() - started,
            "clearml_task_id": str(tracker.task.id),
        }
        metrics_path = output_dir / "training.json"
        _strict_atomic_json(metrics_path, result)
        tracker.connect("coarse_budget_train64_result", result)
        tracker.upload_artifact("terminal_model_update_64", output_dir / final_name)
        tracker.upload_artifact("coarse_budget_train64_result", metrics_path)
        terminal = {
            "completed_updates": 64,
            "checkpoint": final_name,
            "checkpoint_sha256": checkpoint_sha256[final_name],
            "clearml_task_id": str(tracker.task.id),
            "scientific_role": result["scientific_role"],
        }
        _finish_success(tracker, status_path, metrics_path, terminal)
        tracker = None
        return result
    except BaseException as error:
        _mark_failed(
            tracker,
            status_path,
            {
                "completed_updates": len(history),
                "history": history,
                "checkpoint_sha256": checkpoint_sha256,
            },
            error,
        )
        tracker = None
        raise
    finally:
        if tracker is not None:
            try:
                tracker.close()
            except Exception:
                pass


def main() -> None:
    signal.signal(signal.SIGTERM, _terminate)
    signal.signal(signal.SIGINT, _terminate)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve(), args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
