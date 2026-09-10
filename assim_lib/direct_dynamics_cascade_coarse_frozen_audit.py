"""Zero-optimizer audit of the frozen coarse dynamics checkpoint.

This module measures evidence relevant to three possible sources of the
remaining pixel-scale texture: fixed-step ODE error, translation/stride
sensitivity, and the learned velocity field itself.  The evidence is explicitly
non-causal.  The module never creates an optimizer or changes model weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch

from .clearml_tracking import ClearMLTracker
from .config import load_json
from .data import build_dataset
from .direct_dynamics_cascade_coarse import (
    coarse_model_input_for_test,
    coarse_target,
    load_coarse_cascade_sampler,
)
from .direct_dynamics_cascade_fine_training import _clean_code_identity, _fine_collate, _sha256_file
from .direct_dynamics_training import DIRECT_LEADS, _repeat_field_stats, validate_direct_dataset
from .structured_trajectory_evaluation import make_structured_trajectory_figure
from .trainer import _atomic_json
from .transforms import channel_denormalize


SOLVER_TIMEPOINTS = (17, 33, 65)
PROBE_TIME = 0.5
COARSE_SHIFTS = (1, 2, 4, 8)
SHIFT_DIRECTIONS = ((0, 1), (1, 0))
INTERIOR_MARGIN = 16
SELECTED_SOURCE_CASES = (0, 2)


@dataclass
class _Lifecycle:
    phase: str = "configuration"
    output_dir: Path | None = None
    tracker: ClearMLTracker | None = None


def _launch_status(status: str, **details: Any) -> None:
    raw = os.environ.get("COARSE_FROZEN_AUDIT_STATUS_PATH", "").strip()
    if not raw:
        return
    path = Path(raw)
    if not path.is_absolute() or path.name != "status.json":
        raise ValueError("COARSE_FROZEN_AUDIT_STATUS_PATH must be an absolute status.json path")
    _atomic_json(path, {"status": status, **details})


def _atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.incomplete")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _record_failure(lifecycle: _Lifecycle, error: BaseException) -> None:
    failure = {
        "status": "failed",
        "phase": lifecycle.phase,
        "error_type": type(error).__name__,
        "error": str(error),
        "cleanup_errors": [],
    }
    if lifecycle.output_dir is not None and lifecycle.output_dir.is_dir():
        try:
            _atomic_json(lifecycle.output_dir / "failure.json", failure)
        except Exception as cleanup_error:
            failure["cleanup_errors"].append(f"failure_artifact: {cleanup_error}")
    try:
        _launch_status(**failure)
    except Exception as cleanup_error:
        failure["cleanup_errors"].append(f"failure_status: {cleanup_error}")
    if lifecycle.tracker is not None:
        try:
            lifecycle.tracker.task.mark_failed(
                status_reason=type(error).__name__, status_message=str(error)[:1000]
            )
        except Exception as cleanup_error:
            failure["cleanup_errors"].append(f"clearml_mark_failed: {cleanup_error}")
        try:
            lifecycle.tracker.close()
        except Exception as cleanup_error:
            failure["cleanup_errors"].append(f"clearml_close: {cleanup_error}")
    if lifecycle.output_dir is not None and lifecycle.output_dir.is_dir():
        try:
            _atomic_json(lifecycle.output_dir / "failure.json", failure)
        except Exception:
            pass


def _close_tracker_for_success(lifecycle: _Lifecycle) -> None:
    """Close online tracking before publishing terminal success.

    The tracker stays attached to the lifecycle until close returns, so any
    RuntimeError or signal-derived TimeoutError reaches the public failure
    boundary and can still mark the task failed.
    """
    if lifecycle.tracker is None:
        raise RuntimeError("successful audit finalization requires an active tracker")
    lifecycle.tracker.close()
    lifecycle.tracker = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_finite_scalars(value: Any, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _require_finite_scalars(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _require_finite_scalars(child, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise FloatingPointError(f"{path} is NaN/Inf")


def _model_output(value: Any) -> torch.Tensor:
    if hasattr(value, "sample"):
        return value.sample
    if isinstance(value, (tuple, list)) and value and torch.is_tensor(value[0]):
        return value[0]
    if torch.is_tensor(value):
        return value
    raise TypeError("model output does not contain a tensor")


def _masked_rms(value: torch.Tensor, mask: torch.Tensor) -> float:
    expanded = mask.expand_as(value) > 0
    selected = value[expanded]
    if selected.numel() == 0 or not torch.isfinite(selected).all():
        raise FloatingPointError("masked RMS received empty or non-finite values")
    return float(torch.sqrt(selected.square().mean()).item())


def _channel_differences(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    active: torch.Tensor,
    case_ids: list[str] | None = None,
) -> dict[str, Any]:
    if candidate.shape != reference.shape or candidate.shape[1] != 6:
        raise ValueError("solver comparison requires matching six-channel trajectories")
    if case_ids is None:
        case_ids = [f"case{index:02d}" for index in range(candidate.shape[0])]
    if len(case_ids) != candidate.shape[0]:
        raise ValueError("solver comparison case identities differ")
    result: dict[str, Any] = {"aggregate": {}, "cases": {}}
    for lead_index, lead in enumerate(DIRECT_LEADS):
        for offset, field in enumerate(("sic", "sit")):
            channel = 2 * lead_index + offset
            error = candidate[:, channel : channel + 1] - reference[:, channel : channel + 1]
            key = f"d{lead}_{field}"
            selected = error[active.expand_as(error) > 0].abs()
            result["aggregate"][key] = {
                "normalized_rmse": _masked_rms(error, active),
                "normalized_abs_p95": float(torch.quantile(selected, 0.95).item()),
                "normalized_abs_max": float(selected.max().item()),
            }
            for case, case_id in enumerate(case_ids):
                case_error = error[case : case + 1]
                case_active = active[case : case + 1]
                case_selected = case_error[case_active.expand_as(case_error) > 0].abs()
                result["cases"].setdefault(case_id, {})[key] = {
                    "normalized_rmse": _masked_rms(case_error, case_active),
                    "normalized_abs_p95": float(torch.quantile(case_selected, 0.95).item()),
                    "normalized_abs_max": float(case_selected.max().item()),
                }
    return result


def _support_tail_metrics(
    physical: torch.Tensor,
    active: torch.Tensor,
    case_ids: list[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {"cases": {}}
    for case, case_id in enumerate(case_ids):
        ocean = active[case, 0] > 0
        fields: dict[str, Any] = {}
        for lead_index, lead in enumerate(DIRECT_LEADS):
            sic = physical[case, 2 * lead_index][ocean]
            sit = physical[case, 2 * lead_index + 1][ocean]
            for field, excess in (
                ("sic", torch.maximum((-sic).clamp_min(0), (sic - 1).clamp_min(0))),
                ("sit", (-sit).clamp_min(0)),
            ):
                fields[f"d{lead}_{field}"] = {
                    "support_excess_mean_all": float(excess.mean().item()),
                    "support_excess_p95_all": float(torch.quantile(excess, 0.95).item()),
                    "support_excess_max": float(excess.max().item()),
                    "support_violation_fraction": float((excess > 0).float().mean().item()),
                }
        result["cases"][case_id] = fields
    return result


def _region_masks(active: torch.Tensor, margin: int = INTERIOR_MARGIN) -> tuple[torch.Tensor, torch.Tensor]:
    if active.ndim != 4 or active.shape[1] != 1:
        raise ValueError("active mask must be [B,1,H,W]")
    if 2 * margin >= min(active.shape[-2:]):
        raise ValueError("interior margin removes the entire field")
    interior_window = torch.zeros_like(active, dtype=torch.bool)
    interior_window[..., margin:-margin, margin:-margin] = True
    ocean = active > 0
    interior = ocean & interior_window
    padding_control = ocean & ~interior_window
    if not torch.any(interior) or not torch.any(padding_control):
        raise ValueError("shift audit requires both interior and padding-control ocean")
    return interior, padding_control


def _phase_rms(error: torch.Tensor, mask: torch.Tensor, period: int) -> list[list[float | None]]:
    if error.ndim != 4 or mask.shape != (error.shape[0], 1, *error.shape[-2:]):
        raise ValueError("phase audit shapes differ")
    scalar = error.square().mean(dim=1, keepdim=True)
    rows: list[list[float | None]] = []
    for y_phase in range(period):
        row: list[float | None] = []
        for x_phase in range(period):
            phase_mask = torch.zeros_like(mask, dtype=torch.bool)
            phase_mask[..., y_phase::period, x_phase::period] = True
            selected = scalar[phase_mask & (mask > 0)]
            row.append(None if selected.numel() == 0 else float(torch.sqrt(selected.mean()).item()))
        rows.append(row)
    return rows


def _phase_contrast(table: list[list[float | None]]) -> float:
    values = [value for row in table for value in row if value is not None]
    if not values:
        raise ValueError("phase table is empty")
    mean = sum(values) / len(values)
    return 0.0 if mean == 0 else (max(values) - min(values)) / mean


def _tensor_summary(value: torch.Tensor) -> dict[str, float | list[int]]:
    value = value.detach().float()
    if value.numel() == 0 or not torch.isfinite(value).all():
        raise FloatingPointError("activation is empty or non-finite")
    rms = torch.sqrt(value.square().mean())
    scale = max(float(rms.item()), torch.finfo(torch.float32).tiny)
    result: dict[str, float | list[int]] = {
        "shape": list(value.shape),
        "mean": float(value.mean().item()),
        "rms": scale,
        "abs_p99": float(torch.quantile(value.abs().flatten(), 0.99).item()),
        "near_zero_fraction": float((value.abs() < scale * 1e-4).float().mean().item()),
        "large_fraction_gt_8rms": float((value.abs() > scale * 8).float().mean().item()),
    }
    if value.ndim == 4:
        phase = _phase_rms(value, torch.ones_like(value[:, :1]), 2)
        result["phase2_rms_contrast"] = _phase_contrast(phase)
    return result


def _iter_tensors(value: Any, prefix: str) -> Iterable[tuple[str, torch.Tensor]]:
    if torch.is_tensor(value):
        yield prefix, value
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            yield from _iter_tensors(child, f"{prefix}.{index}")


class _ActivationAudit:
    """Capture transition and skip-merge tensors for one explicitly scoped forward."""

    def __init__(self, model: torch.nn.Module):
        self.records: dict[str, dict[str, Any]] = {}
        self.spatial_maps: dict[str, dict[str, torch.Tensor]] = {}
        self.handles = []
        interesting = {"Downsample2D", "Upsample2D", "UpBlock2D"}
        for name, module in model.named_modules():
            if module.__class__.__name__ not in interesting:
                continue
            self.handles.append(module.register_forward_pre_hook(self._pre(name)))
            self.handles.append(module.register_forward_hook(self._post(name)))

    def _pre(self, name: str):
        def hook(_module, args):
            for label, tensor in _iter_tensors(args, "input"):
                key = f"{name}.before.{label}"
                self.records[key] = _tensor_summary(tensor)
                self._capture_spatial(key, tensor)

        return hook

    def _post(self, name: str):
        def hook(_module, _args, output):
            for label, tensor in _iter_tensors(output, "output"):
                key = f"{name}.after.{label}"
                self.records[key] = _tensor_summary(tensor)
                self._capture_spatial(key, tensor)

        return hook

    def _capture_spatial(self, key: str, tensor: torch.Tensor) -> None:
        if tensor.ndim != 4:
            return
        value = tensor.detach().float()
        self.spatial_maps[key] = {
            "channel_rms": torch.sqrt(value.square().mean(dim=1)).cpu(),
            "fixed_channel0": value[:, 0].cpu(),
        }

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


@torch.no_grad()
def _velocity(model, model_input: torch.Tensor, time_value: float) -> torch.Tensor:
    times = torch.full(
        (model_input.shape[0],),
        float(time_value) * 1000.0,
        device=model_input.device,
        dtype=torch.float32,
    )
    output = _model_output(model(model_input.float(), times, return_dict=False)).float()
    if output.shape != (model_input.shape[0], 6, *model_input.shape[-2:]):
        raise ValueError("velocity probe returned the wrong shape")
    if not torch.isfinite(output).all():
        raise FloatingPointError("velocity probe returned NaN/Inf")
    return output


@torch.no_grad()
def _shift_audit(
    model,
    *,
    state: torch.Tensor,
    condition: torch.Tensor,
    fine_valid: torch.Tensor,
    coarse_active: torch.Tensor,
    shifts_to_test: tuple[int, ...] = COARSE_SHIFTS,
    interior_margins: tuple[int, int] = (16, 32),
) -> tuple[dict[str, Any], dict[str, torch.Tensor], dict[str, dict[str, torch.Tensor]]]:
    base_input = coarse_model_input_for_test(state, condition, fine_valid)
    activation = _ActivationAudit(model)
    try:
        base_velocity = _velocity(model, base_input, PROBE_TIME)
    finally:
        activation.close()
    regions = {
        f"margin{margin}": _region_masks(coarse_active, margin) for margin in interior_margins
    }
    phase_interior = regions[f"margin{interior_margins[0]}"][0]
    shifts: dict[str, Any] = {}
    raw_errors: dict[str, torch.Tensor] = {}
    for amount in shifts_to_test:
        for dy_unit, dx_unit in SHIFT_DIRECTIONS:
            shift = (amount * dy_unit, amount * dx_unit)
            shifted_input = torch.roll(base_input, shifts=shift, dims=(-2, -1))
            shifted_velocity = _velocity(model, shifted_input, PROBE_TIME)
            restored = torch.roll(shifted_velocity, shifts=(-shift[0], -shift[1]), dims=(-2, -1))
            error = restored - base_velocity
            raw_errors[f"dy{shift[0]}_dx{shift[1]}"] = error.detach().cpu()
            phase2 = _phase_rms(error, phase_interior, 2)
            phase4 = _phase_rms(error, phase_interior, 4)
            phase8 = _phase_rms(error, phase_interior, 8)
            label = f"dy{shift[0]}_dx{shift[1]}"
            shifts[label] = {
                "shift_cells": amount,
                "aligned_with_total_stride8": amount % 8 == 0,
                "phase2_rms": phase2,
                "phase4_rms": phase4,
                "phase8_rms": phase8,
                "phase2_contrast": _phase_contrast(phase2),
                "phase4_contrast": _phase_contrast(phase4),
                "phase8_contrast": _phase_contrast(phase8),
            }
            for region_name, (region_interior, region_padding) in regions.items():
                interior_rms = _masked_rms(error, region_interior)
                base_rms = _masked_rms(base_velocity, region_interior)
                shifts[label][region_name] = {
                    "interior_absolute_rms": interior_rms,
                    "interior_relative_rms": interior_rms / max(base_rms, 1e-12),
                    "padding_control_absolute_rms": _masked_rms(error, region_padding),
                }
    if not raw_errors:
        raise RuntimeError("shift audit did not execute")
    return {"activations": activation.records, "shifts": shifts}, raw_errors, activation.spatial_maps


def _save_phase_error(
    error: torch.Tensor,
    active: torch.Tensor,
    case: int,
    label: str,
    output: Path,
) -> None:
    magnitude = torch.sqrt(error[case].square().mean(dim=0))
    magnitude = torch.where(active[case, 0].cpu() > 0, magnitude, torch.nan)
    figure, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
    image = axis.imshow(magnitude.numpy(), origin="upper", cmap="magma")
    axis.set_title(f"Ошибка эквивариантности скорости: {label}; case {case}")
    figure.colorbar(image, ax=axis, label="RMS по 6 каналам (норм.)")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _decision(solver: dict[str, Any], shift: dict[str, Any]) -> dict[str, Any]:
    rk33 = solver["rk4_33_vs_65"]
    solver_rmse_max = max(
        field["normalized_rmse"]
        for case in rk33["cases"].values()
        for field in case.values()
    )
    solver_tail_max = max(
        field["normalized_abs_max"]
        for case in rk33["cases"].values()
        for field in case.values()
    )
    shift_relative = max(
        record[margin]["interior_relative_rms"]
        for record in shift["shifts"].values()
        for margin in ("margin16", "margin32")
    )
    phase_contrast = max(
        record[key]
        for record in shift["shifts"].values()
        for key in ("phase2_contrast", "phase4_contrast", "phase8_contrast")
    )
    return {
        "status": "descriptive_inconclusive_pending_astra_review",
        "solver_rk4_33_vs_65_case_channel_rmse_below_0p001_observation": solver_rmse_max < 1e-3,
        "solver_rk4_33_vs_65_max_case_channel_rmse": solver_rmse_max,
        "solver_rk4_33_vs_65_max_pointwise_abs_tail": solver_tail_max,
        "max_shift_interior_relative_velocity_rms": shift_relative,
        "max_phase_rms_contrast": phase_contrast,
        "shift_relative_rms_5pct_observation_flag": shift_relative >= 0.05,
        "causal_diagnosis_permitted": False,
        "limitations": [
            "teacher-forced velocity was probed only at t=0.5",
            "negative probes do not exclude artifacts at other times or trajectory states",
            "GroupNorm and zero padding can propagate boundary effects beyond either crop margin",
            "the 5 percent shift flag is descriptive and not a causal threshold",
            "activation maps are exploratory and do not by themselves localize causation",
        ],
        "permits_training": False,
        "next_authority": "Astra review required before any new training",
    }


def _run_impl(config_path: Path, output_dir: Path, lifecycle: _Lifecycle) -> dict[str, Any]:
    if torch.cuda.device_count() != 1:
        raise RuntimeError("coarse frozen audit requires exactly one visible GPU")
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"refusing to reuse frozen-audit output: {output_dir}")
    experiment = load_json(config_path)
    repo_root = Path(__file__).resolve().parents[1]
    code_identity = _clean_code_identity(repo_root)
    audit_identity = {
        **code_identity,
        "audit_module_sha256": _sha256_file(Path(__file__)),
        "audit_config_sha256": _sha256_file(config_path),
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    lifecycle.output_dir = output_dir
    lifecycle.phase = "source_validation"
    _launch_status(
        "source_validation",
        code_commit=code_identity["git_commit"],
        output_dir=str(output_dir),
    )
    source = Path(experiment["source_run"])
    required = {
        "checkpoint": source / "coarse_update_2048.pth",
        "samples": source / "coarse_diagnostics/update_2048_samples.pt",
        "manifest": source / "coarse_cascade_manifest.json",
        "sentinel": source / "coarse_cascade_dataset_sentinel.json",
        "metadata": source / "metadata.json",
        "model_config": source / "config.json",
        "gate": source / "coarse_mechanics_gate.json",
    }
    expected_sha = experiment["source_sha256"]
    for name, path in required.items():
        if not path.is_file() or _sha256(path) != expected_sha[name]:
            raise ValueError(f"frozen source missing or SHA mismatch: {name}")
    manifest = load_json(required["manifest"])
    gate = load_json(required["gate"])
    metadata = load_json(required["metadata"])
    if manifest.get("code_commit") != experiment["source_code_commit"]:
        raise ValueError("source code commit differs from audit contract")
    if gate.get("decision") != "reject_mechanics_candidate":
        raise ValueError("source mechanics gate is not the expected rejection")

    lifecycle.phase = "dataset_reconstruction"
    _launch_status("dataset_reconstruction", code_commit=code_identity["git_commit"])
    torch.set_num_threads(6)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dataset = build_dataset(metadata["data_config"], split="valid")
    validate_direct_dataset(dataset)
    sentinel = load_json(required["sentinel"])["pilot_subset"]
    source_positions = sentinel["diagnostic_subset_positions"]
    source_indices = [sentinel["validation_indices"][position] for position in source_positions]
    indices = [source_indices[position] for position in SELECTED_SOURCE_CASES]
    batch = _fine_collate([dataset[index] for index in indices])
    payload = torch.load(required["samples"], map_location="cpu", weights_only=True)
    case_ids = [payload["validation_case_ids"][position] for position in SELECTED_SOURCE_CASES]
    if list(batch["meta"]["case_id"]) != case_ids:
        raise ValueError("reconstructed validation cases differ from saved diagnostic identities")

    condition_cpu = batch["structured_conditioning"].float()
    valid_cpu = batch["valid_mask"][:, :1].float()
    truth_normalized_cpu, active_cpu, fraction_cpu = coarse_target(batch["truth"], valid_cpu)
    persistence_normalized_cpu, _, _ = coarse_target(batch["background"], valid_cpu)
    means = _repeat_field_stats(dataset.means)
    stds = _repeat_field_stats(dataset.stds)
    truth_physical_cpu = channel_denormalize(truth_normalized_cpu, means, stds)
    persistence_physical_cpu = channel_denormalize(persistence_normalized_cpu, means, stds)
    selected = list(SELECTED_SOURCE_CASES)
    if not torch.equal(active_cpu, payload["coarse_active_mask"][selected]):
        raise ValueError("reconstructed coarse active mask differs from saved payload")
    if not torch.equal(fraction_cpu, payload["coarse_ocean_fraction"][selected]):
        raise ValueError("reconstructed ocean fraction differs from saved payload")
    if not torch.equal(truth_physical_cpu, payload["coarse_truth_physical"][selected]):
        raise ValueError("reconstructed physical truth differs from saved payload")
    if not torch.equal(
        persistence_physical_cpu, payload["coarse_persistence_physical"][selected]
    ):
        raise ValueError("reconstructed physical persistence differs from saved payload")
    raw_noise_cpu = payload["raw_coarse_noise_normalized"][selected, 0].float()
    support_cpu = active_cpu.expand_as(raw_noise_cpu) > 0
    if not torch.isfinite(raw_noise_cpu[support_cpu]).all():
        raise FloatingPointError("saved noise is non-finite on active coarse ocean")
    noise_cpu = torch.where(support_cpu, raw_noise_cpu, torch.zeros_like(raw_noise_cpu))
    input_artifact = output_dir / "fixed_inputs.pt"
    _atomic_torch_save(
        {
            "case_ids": case_ids,
            "dataset_indices": indices,
            "structured_conditioning": condition_cpu,
            "fine_valid_mask": valid_cpu,
            "coarse_truth_normalized": truth_normalized_cpu,
            "coarse_truth_physical": truth_physical_cpu,
            "coarse_persistence_normalized": persistence_normalized_cpu,
            "coarse_persistence_physical": persistence_physical_cpu,
            "coarse_active_mask": active_cpu,
            "coarse_ocean_fraction": fraction_cpu,
            "original_saved_noise_before_support_mask": raw_noise_cpu,
            "canonical_masked_noise": noise_cpu,
        },
        input_artifact,
    )

    contract = {
        "source_run": str(source),
        "source_task_id": experiment["source_task_id"],
        "source_code_commit": experiment["source_code_commit"],
        "source_sha256": expected_sha,
        "audit_code_identity": audit_identity,
        "fixed_inputs_sha256": _sha256(input_artifact),
        "rebuilt_truth_persistence_fraction_exactly_match_source_payload": True,
        "selected_source_case_positions": selected,
        "dataset_indices": indices,
        "case_ids": case_ids,
        "solver_timepoints": list(SOLVER_TIMEPOINTS),
        "probe_time": PROBE_TIME,
        "coarse_shifts": list(COARSE_SHIFTS),
        "shift_directions": [list(value) for value in SHIFT_DIRECTIONS],
        "interior_margins": [16, 32],
        "weights": "raw_update_2048",
        "network_precision": "fp32",
        "ode_precision": "fp32",
        "tf32": False,
        "optimizer_objects": 0,
        "optimizer_steps": 0,
        "generative_sampling_future_truth_used": False,
        "velocity_probe_uses_future_truth": True,
        "velocity_probe_role": "teacher_forced_mechanistic_diagnostic_only",
        "velocity_probe_forecast_claim_permitted": False,
    }
    lifecycle.phase = "clearml_initialization"
    tracker = ClearMLTracker(
        experiment["project_name"],
        f"{experiment['task_name']}-{output_dir.name}",
        tags=experiment["clearml"]["tags"],
        env_path=experiment["clearml"].get("env_path"),
    )
    lifecycle.tracker = tracker
    tracker.connect("coarse_frozen_audit_contract", contract)
    _atomic_json(output_dir / "contract.json", contract)
    _launch_status(
        "solver_replay",
        code_commit=code_identity["git_commit"],
        output_dir=str(output_dir),
        clearml_task_id=str(tracker.task.id),
    )

    device = torch.device("cuda:0")
    lifecycle.phase = "audited_model_reload"
    sampler = load_coarse_cascade_sampler(
        str(source),
        "coarse_update_2048.pth",
        load_json(required["model_config"]),
        expected_sha["checkpoint"],
        experiment["source_code_commit"],
        device=device,
    )
    model = sampler.sampler.model.to(dtype=torch.float32).eval()
    condition = condition_cpu.to(device)
    valid = valid_cpu.to(device)
    truth_normalized = truth_normalized_cpu.to(device)
    persistence_normalized = persistence_normalized_cpu.to(device)
    active = active_cpu.to(device)
    noise = noise_cpu.to(device)

    samples: dict[int, torch.Tensor] = {}
    raw_dir = output_dir / "raw"
    visual_dir = output_dir / "visuals"
    raw_dir.mkdir()
    visual_dir.mkdir()
    lifecycle.phase = "solver_replay"
    for timepoints in SOLVER_TIMEPOINTS:
        samples[timepoints] = sampler.sample_conditioned(
            structured_conditioning=condition,
            valid_mask=valid,
            initial_noise=noise,
            num_timesteps=timepoints,
            device=device,
            method="rk4",
            end_time=0.0,
        )
        _atomic_torch_save(
            {
                "sample_normalized": samples[timepoints].cpu(),
                "fixed_inputs_sha256": contract["fixed_inputs_sha256"],
                "checkpoint_sha256": expected_sha["checkpoint"],
                "timepoints": timepoints,
                "optimizer_steps": 0,
            },
            raw_dir / f"rk4_{timepoints - 1}_sample.pt",
        )

    solver = {
        "rk4_17_vs_65": _channel_differences(samples[17], samples[65], active, case_ids),
        "rk4_33_vs_65": _channel_differences(samples[33], samples[65], active, case_ids),
    }
    physical_samples = {
        timepoints: channel_denormalize(sample, means, stds)
        for timepoints, sample in samples.items()
    }
    support_tails = {
        f"rk4_{timepoints - 1}": _support_tail_metrics(physical, active, case_ids)
        for timepoints, physical in physical_samples.items()
    }

    lifecycle.phase = "shift_and_activation_probe"
    probe_state = (1.0 - PROBE_TIME) * truth_normalized + PROBE_TIME * noise
    probe_state = torch.where(
        active.expand_as(probe_state) > 0, probe_state, torch.zeros_like(probe_state)
    )
    shift, raw_shift_errors, activation_maps = _shift_audit(
        model,
        state=probe_state,
        condition=condition,
        fine_valid=valid,
        coarse_active=active,
    )
    shift_artifact = raw_dir / "raw_shift_errors.pt"
    activation_artifact = raw_dir / "activation_spatial_maps.pt"
    _atomic_torch_save(raw_shift_errors, shift_artifact)
    _atomic_torch_save(activation_maps, activation_artifact)
    shift["raw_shift_errors_sha256"] = _sha256(shift_artifact)
    shift["activation_spatial_maps_sha256"] = _sha256(activation_artifact)
    decision = _decision(solver, shift)

    lifecycle.phase = "visualization"
    for timepoints, physical in physical_samples.items():
        for case in range(len(case_ids)):
            figure = make_structured_trajectory_figure(
                truth_physical_cpu[case],
                persistence_physical_cpu[case],
                physical[case],
                active[case],
                title=f"frozen RK4-{timepoints - 1}: raw member; {case_ids[case]}",
                origin="upper",
                lead_days=DIRECT_LEADS,
            )
            path = visual_dir / f"rk4_{timepoints - 1}_case{case:02d}_raw_member.png"
            figure.savefig(path, dpi=180)
            plt.close(figure)
            tracker.report_image("coarse_frozen_audit/raw_samples", path.stem, path, timepoints)
    for label, error in raw_shift_errors.items():
        for case in range(len(case_ids)):
            phase_path = visual_dir / f"{label}_case{case:02d}_velocity_error.png"
            _save_phase_error(error, active_cpu, case, label, phase_path)
            tracker.report_image("coarse_frozen_audit/shift", phase_path.stem, phase_path, 0)

    result = {
        "status": "complete_pending_astra_review",
        "clearml_task_id": str(tracker.task.id),
        "contract": contract,
        "solver": solver,
        "raw_support_tails": support_tails,
        "shift_and_activation": shift,
        "decision": decision,
    }
    _require_finite_scalars(result)
    _atomic_json(output_dir / "result.json", result)
    tracker.connect("coarse_frozen_audit_result", result)
    tracker.upload_artifact("coarse_frozen_audit_result", output_dir / "result.json")
    for label, value in decision.items():
        if isinstance(value, (int, float, bool)):
            tracker.report_single_value(label, float(value))
    clearml_task_id = str(tracker.task.id)
    lifecycle.phase = "clearml_close_before_terminal_success"
    _close_tracker_for_success(lifecycle)
    _launch_status(
        "complete_pending_astra_review",
        code_commit=code_identity["git_commit"],
        output_dir=str(output_dir),
        clearml_task_id=clearml_task_id,
    )
    return result


def run(config_path: Path, output_dir: Path) -> dict[str, Any]:
    lifecycle = _Lifecycle()
    try:
        return _run_impl(config_path, output_dir, lifecycle)
    except BaseException as error:
        _record_failure(lifecycle, error)
        raise


def main() -> None:
    def _terminate(signum, _frame):
        raise TimeoutError(f"coarse frozen audit received termination signal {signum}")

    signal.signal(signal.SIGTERM, _terminate)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.config.resolve(), arguments.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
