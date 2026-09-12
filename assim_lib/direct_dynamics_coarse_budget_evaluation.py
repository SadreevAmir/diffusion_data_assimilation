"""Paired validation of the bounded terminal coarse-budget refinement.

The control cascade is sampled exactly once.  The candidate reuses the same
coarse noise and the already generated colored fine draw; only the final
coarse RK4 interval is replaced.  Fine noise, residuals, conditioning and the
within-cell allocation therefore remain frozen at the control draw C0.
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import signal
from pathlib import Path
from typing import Any

import torch

from .clearml_tracking import ClearMLTracker
from .config import load_json, merge_config_overrides, resolve_path
from .data import build_dataset
from .direct_dynamics_affine_calibration import boundary_event_metrics
from .direct_dynamics_cascade_checkpoint_evaluation import _save_rank_histograms
from .direct_dynamics_cascade_coarse import lossless_coarse_condition
from .direct_dynamics_cascade_coarse_fine_boundary_audit import (
    _sha256,
    _strict_atomic_json,
)
from .direct_dynamics_cascade_end_to_end import load_cascade_predictor
from .direct_dynamics_cascade_e2e_evaluation import _add_joint_consistency, _summary
from .direct_dynamics_cascade_fine_training import (
    EXPECTED_SPLIT_LENGTHS,
    _clean_code_identity,
    _sha256_file,
)
from .direct_dynamics_cascade_paired_evaluation import (
    _require_finite_scalars,
    _save_contact_sheet,
    score_ensemble,
)
from .direct_dynamics_cascade_proper_refinement_evaluation import (
    _atomic_torch_save,
    _paired_primary_confirmation,
    _primary_standardized_fair_crps,
)
from .direct_dynamics_coarse_budget_actual_gpu_preflight import (
    _load_normalization_binding,
    _repeat_case,
    _validate_config as _validate_base_config,
    _validate_files,
    _validate_proved_contract,
)
from .direct_dynamics_coarse_budget_production_replay import (
    sequential_terminal_candidate_control,
)
from .direct_dynamics_coarse_budget_refinement_integration import (
    anchored_canonical_physical_coarse,
    frozen_allocation_physical_law,
)
from .direct_dynamics_coarse_budget_training import EXPECTED_PROTOCOL
from .direct_dynamics_fine_support_proper_admission import apply_training_sic_decoder
from .direct_dynamics_sic_support_decoder_scoring import canonical_physical_decode
from .direct_dynamics_training import validate_direct_dataset
from .runtime import make_normalized_xy_grid, seed_everything


CHECKPOINT_KEYS = {
    "model",
    "optimizer",
    "completed_updates",
    "protocol",
    "train_indices",
    "code_identity",
    "actual_admission",
    "source",
}
_ACTIVE_TRACKER: ClearMLTracker | None = None


def _terminate(signum: int, _frame: Any) -> None:
    raise TimeoutError(f"coarse-budget paired evaluation received signal {signum}")


def _training_anchored_candidate(
    production_coarse: torch.Tensor,
    control_coarse: torch.Tensor,
    candidate_coarse: torch.Tensor,
) -> torch.Tensor:
    """Reproduce the exact FP32 anchoring expression used during train64."""
    return production_coarse.detach() + (
        candidate_coarse - control_coarse.detach()
    )


def _standardize_physical(
    physical: torch.Tensor, stds: torch.Tensor
) -> torch.Tensor:
    shape = (1,) * (physical.ndim - 3) + (6, 1, 1)
    return physical.double() / stds.double().reshape(shape)


def _record_evidence(
    output: Path,
    manifest: dict[str, Any],
    name: str,
    payload: Any,
) -> str:
    """Durably save one stage and immediately bind it into the manifest."""
    path = output / "evidence" / f"{name}.pt"
    path.parent.mkdir(exist_ok=True)
    digest = _atomic_torch_save(payload, path)
    manifest["artifacts"][name] = {
        "path": str(path),
        "sha256": digest,
    }
    _strict_atomic_json(output / "evidence_manifest.json", manifest)
    return digest


def _finalize_success(
    status_path: Path,
    reservation: dict[str, Any],
    result_path: Path,
    uploaded_result_sha256: str,
    evidence_manifest_path: Path,
    clearml_task_id: str,
) -> dict[str, Any]:
    """Write terminal lifecycle state without mutating uploaded numerical bytes."""
    if _sha256(result_path) != uploaded_result_sha256:
        raise RuntimeError("numerical result changed after ClearML upload")
    status = {
        **reservation,
        "status": "complete_pending_astra_and_visual_review",
        "clearml_task_id": clearml_task_id,
        "result_path": str(result_path),
        "result_sha256": uploaded_result_sha256,
        "evidence_manifest_sha256": _sha256(evidence_manifest_path),
    }
    _strict_atomic_json(status_path, status)
    return status


def _load_base_config(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    spec = config["base_preflight_config"]
    path = (config_path.parent / spec["path"]).resolve()
    if not path.is_file() or _sha256(path) != spec["sha256"]:
        raise ValueError("base preflight config SHA mismatch")
    base = load_json(path)
    _validate_base_config(base)
    return base


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != "coarse_budget_paired_validation_v1":
        raise ValueError("unreviewed coarse-budget evaluation schema")
    if config.get("split") != "valid" or config.get("test_2023") != "closed":
        raise ValueError("paired evaluation is restricted to validation-2022")
    if config.get("cases") != 12 or config.get("members") != 8:
        raise ValueError("paired evaluation requires exactly 12 cases x 8 members")
    if config.get("coarse_rk4_timepoints") != 17 or config.get(
        "fine_rk4_timepoints"
    ) != 33:
        raise ValueError("paired evaluation changed the reviewed RK4 grids")
    indices = config.get("case_indices")
    case_ids = config.get("case_ids")
    if (
        not isinstance(indices, list)
        or not isinstance(case_ids, list)
        or len(indices) != 12
        or len(case_ids) != 12
        or len(set(indices)) != 12
        or len(set(case_ids)) != 12
    ):
        raise ValueError("paired evaluation panel is not twelve unique cases")
    gate = config.get("decision_gate", {})
    if (
        gate.get("primary")
        != "case_equal_six_channel_train_standardized_fair_crps"
        or gate.get("bootstrap_draws") != 100000
    ):
        raise ValueError("paired decision gate differs from the frozen protocol")
    labels = config.get("labels", {})
    if set(labels) != {"control", "candidate", "raw_secondary"}:
        raise ValueError("paired evaluation labels omit the raw secondary reference")
    if len(set(labels.values())) != 3:
        raise ValueError("paired evaluation labels must be unique")


def _load_training_record(config: dict[str, Any]) -> dict[str, Any]:
    spec = config["trained_terminal"]
    path = Path(spec["training_record"])
    if not path.is_file() or _sha256(path) != spec["training_record_sha256"]:
        raise ValueError("bounded training record SHA mismatch")
    record = load_json(path)
    if (
        record.get("status") != "training_complete_pending_paired_validation"
        or record.get("completed_updates") != 64
        or record.get("protocol") != EXPECTED_PROTOCOL
        or record.get("extension_allowed") is not False
        or record.get("fine_resampled_for_candidate") is not False
        or record.get("test_2023_used") is not False
        or record.get("clearml_task_id") != spec["training_clearml_task_id"]
        or record.get("code_identity", {}).get("git_commit") != spec["code_commit"]
        or record.get("checkpoint_sha256", {}).get(spec["checkpoint_name"])
        != spec["checkpoint_sha256"]
    ):
        raise ValueError("bounded training record differs from the admitted result")
    return record


def load_checkpoint_cpu_gate(
    checkpoint_path: Path,
    expected_sha256: str,
    training_record: dict[str, Any],
    expected_source: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load and exhaustively validate the terminal checkpoint before CUDA use."""
    if not checkpoint_path.is_file() or _sha256(checkpoint_path) != expected_sha256:
        raise ValueError("terminal checkpoint SHA mismatch")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or set(payload) != CHECKPOINT_KEYS:
        raise ValueError("terminal checkpoint top-level keys differ")
    for key in (
        "completed_updates",
        "protocol",
        "train_indices",
        "code_identity",
        "actual_admission",
    ):
        if payload.get(key) != training_record.get(key):
            raise ValueError(f"checkpoint/training record binding differs: {key}")
    if payload["completed_updates"] != 64 or payload["protocol"] != EXPECTED_PROTOCOL:
        raise ValueError("terminal checkpoint is not the reviewed 64-update protocol")
    if payload.get("source") != expected_source:
        raise ValueError("terminal checkpoint source differs from the frozen cascade")
    if training_record.get("effective_forecast_contract_sha256") != expected_source.get(
        "forecast_contract_sha256"
    ):
        raise ValueError("training forecast contract differs from checkpoint source")
    expected_files = {
        stage: expected_source[stage]["files_sha256"] for stage in ("coarse", "fine")
    }
    if training_record.get("source_files_sha256") != expected_files:
        raise ValueError("training source file inventory differs from checkpoint source")

    model = payload["model"]
    if not isinstance(model, dict) or not model:
        raise ValueError("terminal checkpoint model state is empty")
    model_tensor_count = 0
    model_numel = 0
    for name, value in model.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise TypeError("terminal model state must contain named tensors only")
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"terminal model tensor is non-finite: {name}")
        model_tensor_count += 1
        model_numel += value.numel()

    optimizer = payload["optimizer"]
    if not isinstance(optimizer, dict) or set(optimizer) != {"state", "param_groups"}:
        raise ValueError("terminal optimizer state structure differs")
    states = optimizer["state"]
    groups = optimizer["param_groups"]
    if not isinstance(states, dict) or not states or not isinstance(groups, list) or len(groups) != 1:
        raise ValueError("terminal optimizer must contain one non-empty AdamW group")
    group = groups[0]
    if (
        group.get("lr") != EXPECTED_PROTOCOL["learning_rate"]
        or group.get("weight_decay") != EXPECTED_PROTOCOL["weight_decay"]
        or group.get("amsgrad") is not False
    ):
        raise ValueError("terminal AdamW hyperparameters differ from training protocol")
    parameter_ids = list(group.get("params", []))
    if len(parameter_ids) != len(set(parameter_ids)) or set(parameter_ids) != set(states):
        raise ValueError("terminal optimizer parameter/state inventory differs")
    steps: list[int] = []
    moment_tensor_count = 0
    for parameter_id, state in states.items():
        if not isinstance(state, dict) or not {"step", "exp_avg", "exp_avg_sq"}.issubset(state):
            raise ValueError(f"incomplete AdamW state for parameter {parameter_id}")
        step = state["step"]
        step_value = float(step.item() if isinstance(step, torch.Tensor) else step)
        if not math.isfinite(step_value) or not step_value.is_integer():
            raise ValueError("non-integral AdamW step in terminal checkpoint")
        steps.append(int(step_value))
        for name in ("exp_avg", "exp_avg_sq"):
            value = state[name]
            if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
                raise FloatingPointError(f"non-finite AdamW {name}")
            moment_tensor_count += 1
    if set(steps) != {64}:
        raise ValueError("terminal optimizer states did not all complete update 64")
    report = {
        "checkpoint_sha256": expected_sha256,
        "top_level_keys": sorted(payload),
        "completed_updates": payload["completed_updates"],
        "model_tensor_count": model_tensor_count,
        "model_numel": model_numel,
        "optimizer_state_count": len(states),
        "optimizer_step_min": min(steps),
        "optimizer_step_max": max(steps),
        "optimizer_step_unique": sorted(set(steps)),
        "optimizer_moment_tensor_count": moment_tensor_count,
        "all_model_and_optimizer_tensors_finite": True,
        "protocol_source_and_training_bindings_exact": True,
    }
    return payload, report


@torch.no_grad()
def _paired_case(
    predictor: Any,
    terminal_model: torch.nn.Module,
    condition: torch.Tensor,
    valid: torch.Tensor,
    case_id: str,
    means: torch.Tensor,
    stds: torch.Tensor,
    device: torch.device,
    output: Path,
    evidence_manifest: dict[str, Any],
    case_position: int,
) -> dict[str, torch.Tensor | tuple | dict | None]:
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        production = predictor.sample_ensemble(
            member_indices=tuple(range(8)),
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
    _record_evidence(
        output,
        evidence_manifest,
        f"case{case_position:02d}_production",
        {
            "case_id": case_id,
            "member_indices": tuple(range(8)),
            "production": {
                key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
                for key, value in production.items()
            },
        },
    )
    condition_m = _repeat_case(condition.to(device), 8)
    valid_m = _repeat_case(valid.to(device), 8)
    encoded, active, _ = lossless_coarse_condition(condition_m, valid_m)
    grid = make_normalized_xy_grid(
        *encoded.shape[-2:], device=device, dtype=torch.float32
    )
    candidate_flat, control_flat, prefix = sequential_terminal_candidate_control(
        predictor.coarse_sampler.sampler.model,
        terminal_model,
        production["raw_coarse_noise"].flatten(0, 1),
        encoded,
        active,
        grid,
    )
    if prefix.requires_grad:
        raise RuntimeError("evaluation prefix retained a gradient graph")
    candidate = candidate_flat.unflatten(0, (1, 8))
    control = control_flat.unflatten(0, (1, 8))
    anchored_candidate = _training_anchored_candidate(
        production["coarse"], control, candidate
    )
    _record_evidence(
        output,
        evidence_manifest,
        f"case{case_position:02d}_candidate_control",
        {
            "case_id": case_id,
            "production_coarse": production["coarse"].detach().cpu(),
            "control_coarse": control.detach().cpu(),
            "terminal_candidate_coarse": candidate.detach().cpu(),
            "training_anchored_candidate_coarse": anchored_candidate.detach().cpu(),
        },
    )
    if not torch.equal(control, production["coarse"]):
        raise RuntimeError(
            "terminal evaluation control did not replay production coarse exactly"
        )
    base_coarse, candidate_coarse = anchored_canonical_physical_coarse(
        production["coarse"], anchored_candidate, means, stds
    )
    base_physical = canonical_physical_decode(production["forecast"], means, stds)
    control_physical = apply_training_sic_decoder(
        base_physical, base_coarse, valid.to(device)
    )
    candidate_physical = frozen_allocation_physical_law(
        base_physical.detach(), base_coarse.detach(), candidate_coarse, valid.to(device)
    )
    _record_evidence(
        output,
        evidence_manifest,
        f"case{case_position:02d}_scored_physical",
        {
            "case_id": case_id,
            "raw_production_physical": base_physical.detach().cpu(),
            "control_physical": control_physical.detach().cpu(),
            "candidate_physical": candidate_physical.detach().cpu(),
        },
    )
    return {
        **production,
        "terminal_candidate_coarse_normalized": candidate,
        "candidate_coarse_normalized": anchored_candidate,
        "raw_production_physical": base_physical,
        "control_physical": control_physical,
        "candidate_physical": candidate_physical,
    }


def _run_impl(
    config_path: Path, output: Path, lifecycle: dict[str, Any]
) -> dict[str, Any]:
    global _ACTIVE_TRACKER
    if os.environ.get("CLEARML_REQUIRE_ONLINE") != "1":
        raise RuntimeError("paired coarse-budget evaluation requires online ClearML")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to reuse output {output}")
    output.mkdir(parents=True)
    lifecycle["output"] = output
    lifecycle["status_path"] = output / "status.json"
    reservation = {
        "status": "reserved",
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "test_2023_used": False,
        "optimizer_steps": 0,
    }
    lifecycle["reservation"] = reservation
    _strict_atomic_json(lifecycle["status_path"], reservation)
    config = load_json(config_path)
    _validate_config(config)
    base = _load_base_config(config, config_path)
    _validate_proved_contract(base)
    source = base["source"]
    source_files = {
        stage: _validate_files(source[stage], stage) for stage in ("coarse", "fine")
    }
    record = _load_training_record(config)
    terminal_spec = config["trained_terminal"]
    payload, checkpoint_probe = load_checkpoint_cpu_gate(
        Path(terminal_spec["checkpoint"]),
        terminal_spec["checkpoint_sha256"],
        record,
        source,
    )
    if torch.cuda.device_count() != 1:
        raise RuntimeError("paired coarse-budget evaluation requires one visible GPU")

    repo = Path(__file__).resolve().parents[1]
    code_identity = _clean_code_identity(repo)
    data_path = resolve_path(base["data_config"], config_path.parent)
    if _sha256_file(data_path) != base["data_config_sha256"]:
        raise ValueError("data config SHA mismatch")
    data_config = merge_config_overrides(load_json(data_path), base.get("data_overrides"))
    dataset = build_dataset(data_config, split="valid")
    if len(dataset) != EXPECTED_SPLIT_LENGTHS["valid"]:
        raise ValueError("validation archive length differs from the audited inventory")
    validate_direct_dataset(dataset)
    selected = [dataset[index] for index in config["case_indices"]]
    case_ids = [str(item["meta"]["case_id"]) for item in selected]
    if case_ids != config["case_ids"]:
        raise ValueError("frozen validation case identities differ from the archive")
    means, stds, normalization = _load_normalization_binding(base, repo)
    seed_everything(73127)

    _strict_atomic_json(output / "checkpoint_probe.json", checkpoint_probe)
    tracker = ClearMLTracker(
        config["project_name"],
        f"{config['task_name']}-{output.name}",
        tags=config["clearml"]["tags"],
        env_path=config["clearml"]["env_path"],
    )
    _ACTIVE_TRACKER = tracker
    lifecycle["tracker"] = tracker
    tracker.connect("evaluation_contract", config)
    device = torch.device("cuda:0")
    coarse_spec, fine_spec = source["coarse"], source["fine"]
    predictor = load_cascade_predictor(
        coarse_run_dir=coarse_spec["run_dir"],
        coarse_checkpoint_name=coarse_spec["checkpoint"],
        coarse_model_config=load_json(Path(coarse_spec["run_dir"]) / "config.json"),
        coarse_checkpoint_sha256=coarse_spec["checkpoint_sha256"],
        fine_run_dir=fine_spec["run_dir"],
        fine_checkpoint_name=fine_spec["checkpoint"],
        fine_model_config=load_json(Path(fine_spec["run_dir"]) / "config.json"),
        fine_checkpoint_sha256=fine_spec["checkpoint_sha256"],
        expected_coarse_code_commit=coarse_spec["code_commit"],
        expected_fine_code_commit=fine_spec["code_commit"],
        replay_code_commit=code_identity["git_commit"],
        expected_forecast_contract_sha256=source["forecast_contract_sha256"],
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
    terminal_model = copy.deepcopy(predictor.coarse_sampler.sampler.model)
    terminal_model.load_state_dict(payload["model"], strict=True)
    terminal_model.to(device=device, dtype=torch.float32).eval()
    delta_square = torch.zeros((), device=device, dtype=torch.float64)
    base_square = torch.zeros((), device=device, dtype=torch.float64)
    delta_max_abs = 0.0
    for base_parameter, terminal_parameter in zip(
        predictor.coarse_sampler.sampler.model.parameters(),
        terminal_model.parameters(),
        strict=True,
    ):
        delta = terminal_parameter.double() - base_parameter.double()
        delta_square += delta.square().sum()
        base_square += base_parameter.double().square().sum()
        delta_max_abs = max(delta_max_abs, float(delta.abs().max().cpu()))
    if delta_max_abs == 0.0:
        raise RuntimeError("terminal64 checkpoint is identical to control EMA9711")

    evidence_manifest: dict[str, Any] = {
        "schema_version": "coarse_budget_paired_evidence_manifest_v1",
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "code_identity": code_identity,
        "normalization": normalization,
        "source": source,
        "trained_terminal": terminal_spec,
        "seed": 73127,
        "case_indices": config["case_indices"],
        "case_ids": case_ids,
        "member_indices": list(range(8)),
        "artifacts": {},
    }
    lifecycle["evidence_manifest"] = evidence_manifest
    _record_evidence(
        output,
        evidence_manifest,
        "fixed_inputs",
        {
            "structured_conditioning": torch.stack(
                [row["structured_conditioning"].float() for row in selected]
            ),
            "valid_mask": torch.stack(
                [row["valid_mask"][:1].float() for row in selected]
            ),
            "truth": torch.stack([row["truth"].float() for row in selected]),
            "background": torch.stack(
                [row["background"].float() for row in selected]
            ),
            "case_indices": config["case_indices"],
            "case_ids": case_ids,
            "member_indices": tuple(range(8)),
            "seed": 73127,
            "normalization": normalization,
            "config_sha256": _sha256(config_path),
            "code_identity": code_identity,
        },
    )

    stored: list[dict[str, Any]] = []
    for case_position, (item, case_id) in enumerate(
        zip(selected, case_ids, strict=True)
    ):
        stored.append(
            {
                key: (
                    value.detach().cpu()
                    if isinstance(value, torch.Tensor)
                    else value
                )
                for key, value in _paired_case(
                    predictor,
                    terminal_model,
                    item["structured_conditioning"].unsqueeze(0).float(),
                    item["valid_mask"].unsqueeze(0)[:, :1].float(),
                    case_id,
                    means.to(device),
                    stds.to(device),
                    device,
                    output,
                    evidence_manifest,
                    case_position,
                ).items()
            }
        )
        torch.cuda.empty_cache()

    tensor_keys = (
        "forecast",
        "coarse",
        "residual",
        "raw_coarse_noise",
        "raw_fine_noise",
        "projected_fine_noise",
        "terminal_candidate_coarse_normalized",
        "candidate_coarse_normalized",
        "raw_production_physical",
        "control_physical",
        "candidate_physical",
    )
    combined = {key: torch.cat([row[key] for row in stored], dim=0) for key in tensor_keys}
    truth_normalized = torch.stack([row["truth"].float() for row in selected])
    persistence_normalized = torch.stack([row["background"].float() for row in selected])
    valid = torch.stack([row["valid_mask"][:1].float() for row in selected])
    truth_physical = canonical_physical_decode(truth_normalized, means, stds)
    persistence_physical = canonical_physical_decode(
        persistence_normalized, means, stds
    )
    truth_standardized = _standardize_physical(truth_physical, stds)
    labels = config["labels"]
    variants = {
        labels["control"]: (
            _standardize_physical(combined["control_physical"], stds),
            combined["control_physical"],
        ),
        labels["candidate"]: (
            _standardize_physical(combined["candidate_physical"], stds),
            combined["candidate_physical"],
        ),
        labels["raw_secondary"]: (
            _standardize_physical(combined["raw_production_physical"], stds),
            combined["raw_production_physical"],
        ),
    }
    result: dict[str, Any] = {
        "status": "numerical_results_complete_lifecycle_in_status_json",
        "publication_claim_permitted": False,
        "clearml_task_id": str(tracker.task.id),
        "code_identity": code_identity,
        "checkpoint_probe": checkpoint_probe,
        "terminal_parameter_change": {
            "l2": float(torch.sqrt(delta_square).cpu()),
            "relative_l2": float(torch.sqrt(delta_square / base_square).cpu()),
            "max_abs": delta_max_abs,
        },
        "fine_resampled_for_candidate": False,
        "raw_secondary_role": "unbounded_production_reference_not_primary",
        "test_2023_used": False,
        "source_files_sha256": source_files,
        "normalization": normalization,
        "candidates": {},
    }
    evidence = {
        **combined,
        "truth_normalized": truth_normalized,
        "persistence_normalized": persistence_normalized,
        "valid_mask": valid,
        "case_indices": config["case_indices"],
        "case_ids": case_ids,
        "member_indices": tuple(range(8)),
        "source": source,
        "trained_terminal": terminal_spec,
        "fine_resampled_for_candidate": False,
    }
    evidence_sha = _record_evidence(
        output, evidence_manifest, "paired_raw_evidence", evidence
    )
    result["raw_evidence_sha256"] = evidence_sha
    for label, (normalized, physical) in variants.items():
        metrics = score_ensemble(
            normalized,
            physical,
            truth_standardized,
            truth_physical,
            persistence_physical,
            valid,
            valid,
        )
        _add_joint_consistency(metrics, physical, valid)
        metrics["boundary_events"] = boundary_event_metrics(
            physical, truth_physical, valid
        )
        metrics["primary_standardized_fair_crps"] = _primary_standardized_fair_crps(
            normalized, truth_standardized, valid
        )
        metrics["summary"] = _summary(metrics)
        _require_finite_scalars(metrics)
        result["candidates"][label] = {"metrics": metrics}
    control_metrics = result["candidates"][labels["control"]]["metrics"]
    candidate_metrics = result["candidates"][labels["candidate"]]["metrics"]
    result["paired_development_diagnostic"] = _paired_primary_confirmation(
        control_metrics["primary_standardized_fair_crps"],
        candidate_metrics["primary_standardized_fair_crps"],
        config["decision_gate"],
    )
    result["paired_development_diagnostic"]["claim_status"] = (
        "DEVELOPMENT_ONLY_NOT_INDEPENDENT_CONFIRMATION"
    )
    result["paired_summary_delta_candidate_minus_control"] = {
        key: float(
            candidate_metrics["summary"][key] - control_metrics["summary"][key]
        )
        for key in control_metrics["summary"]
    }
    _require_finite_scalars(result)
    result_path = output / "coarse_budget_paired_validation.json"
    _strict_atomic_json(result_path, result)

    ranks_dir = output / "rank_histograms"
    visuals_dir = output / "fixed_scale_members"
    ranks_dir.mkdir()
    visuals_dir.mkdir()
    for label in variants:
        rank_path = ranks_dir / f"{label}.png"
        _save_rank_histograms(rank_path, label, result["candidates"][label]["metrics"])
        tracker.report_image("rank_histograms", label, rank_path, 0)
    for position in (0, 4, 7, 11):
        for label, (_, physical) in variants.items():
            visual_path = visuals_dir / f"{label}_case{position:02d}_all8_fixed.png"
            _save_contact_sheet(
                visual_path,
                label,
                case_ids[position],
                physical[position],
                truth_physical[position],
                persistence_physical[position],
                valid[position],
                {"sic": (0.0, 1.0), "sit": (0.0, 3.5)},
            )
            tracker.report_image(
                "fixed_scale_members", f"{label}/case{position:02d}", visual_path, 0
            )
    uploaded_result_sha256 = _sha256(result_path)
    tracker.upload_artifact("coarse_budget_paired_validation", result_path)
    tracker.upload_artifact("checkpoint_probe", output / "checkpoint_probe.json")
    tracker.close()
    lifecycle["tracker"] = None
    _ACTIVE_TRACKER = None
    _finalize_success(
        lifecycle["status_path"],
        reservation,
        result_path,
        uploaded_result_sha256,
        output / "evidence_manifest.json",
        result["clearml_task_id"],
    )
    return result


def _finalize_failure(
    lifecycle: dict[str, Any], error: BaseException
) -> dict[str, Any]:
    failure = {
        **lifecycle.get("reservation", {}),
        "status": "failed",
        "error_type": type(error).__name__,
        "error": str(error)[:2000],
        "cleanup_errors": [],
    }
    output = lifecycle.get("output")
    manifest = lifecycle.get("evidence_manifest")
    if isinstance(output, Path) and isinstance(manifest, dict):
        manifest_path = output / "evidence_manifest.json"
        if manifest_path.is_file():
            try:
                failure["evidence_manifest_sha256"] = _sha256(manifest_path)
                failure["evidence_artifacts"] = copy.deepcopy(
                    manifest.get("artifacts", {})
                )
            except Exception as cleanup_error:
                failure["cleanup_errors"].append(
                    f"evidence_manifest_read: {cleanup_error}"
                )
    status_path = lifecycle.get("status_path")
    if isinstance(status_path, Path):
        try:
            _strict_atomic_json(status_path, failure)
        except Exception as cleanup_error:
            failure["cleanup_errors"].append(f"status_write: {cleanup_error}")
    tracker = lifecycle.get("tracker")
    if tracker is not None:
        try:
            tracker.task.mark_failed(
                status_reason=type(error).__name__,
                status_message=str(error)[:1000],
            )
        except Exception as cleanup_error:
            failure["cleanup_errors"].append(
                f"clearml_mark_failed: {cleanup_error}"
            )
        try:
            tracker.close()
        except Exception as cleanup_error:
            failure["cleanup_errors"].append(f"clearml_close: {cleanup_error}")
    if isinstance(status_path, Path):
        try:
            _strict_atomic_json(status_path, failure)
        except Exception:
            pass
    return failure


def run(config_path: Path, output: Path) -> dict[str, Any]:
    global _ACTIVE_TRACKER
    lifecycle: dict[str, Any] = {
        "output": None,
        "status_path": None,
        "reservation": {},
        "tracker": None,
        "evidence_manifest": None,
    }
    try:
        return _run_impl(config_path, output, lifecycle)
    except BaseException as error:
        failure = _finalize_failure(lifecycle, error)
        created = lifecycle.get("output")
        if isinstance(created, Path) and created.is_dir():
            try:
                _strict_atomic_json(created / "failure.json", failure)
            except Exception:
                pass
        _ACTIVE_TRACKER = None
        raise


def main() -> None:
    signal.signal(signal.SIGTERM, _terminate)
    signal.signal(signal.SIGINT, _terminate)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.config.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
