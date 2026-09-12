"""Zero-update actual-checkpoint GPU preflight for fine proper refinement."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import signal
from pathlib import Path
from typing import Any

import torch

from .clearml_tracking import ClearMLTracker
from .config import load_json, merge_config_overrides, resolve_path
from .data import build_dataset
from .direct_dynamics_cascade import masked_block_average
from .direct_dynamics_cascade_coarse_fine_boundary_audit import (
    _finish_success,
    _load_normalization,
    _sha256,
    _strict_atomic_json,
    _terminate,
)
from .direct_dynamics_cascade_coarse_proper_refinement import _atomic_torch_save
from .direct_dynamics_cascade_contract import (
    forecast_contract_from_data_config,
    forecast_contract_sha256,
)
from .direct_dynamics_cascade_fine import teacher_coarse_condition
from .direct_dynamics_cascade_fine_colored import (
    load_colored_variance_preconditioned_fine_cascade_sampler,
)
from .direct_dynamics_cascade_fine_training import (
    EXPECTED_SPLIT_LENGTHS,
    _calendar_inventory,
    _canonical_sha256,
    _clean_code_identity,
    _sha256_file,
)
from .direct_dynamics_fine_support_proper_admission import (
    support_aware_physical_objective,
)
from .direct_dynamics_fine_support_proper_integration import (
    _repeat_members,
    fine_frozen_prefix,
    terminal_fine_residual,
)
from .direct_dynamics_training import validate_direct_dataset
from .direct_dynamics_sic_support_decoder_scoring import canonical_physical_decode
from .runtime import make_normalized_xy_grid, seed_everything


EXPECTED_PROTOCOL = {
    "split": "train",
    "case_index": 0,
    "members": 4,
    "rk4_intervals": 32,
    "frozen_prefix_intervals": 31,
    "trainable_terminal_intervals": 1,
    "network_precision": "bf16",
    "state_precision": "fp32",
    "score_precision": "fp64",
    "optimizer_steps": 0,
    "seed": 71693,
    "test_2023": "closed",
    "max_gpu_minutes": 30,
}


def _validate_source_files(source: dict[str, Any]) -> dict[str, str]:
    root = Path(source["run_dir"])
    if not root.is_dir():
        raise ValueError("fine2048 source run directory is missing")
    bound: dict[str, str] = {}
    for name, expected in source["files_sha256"].items():
        path = root / name
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"fine2048 source SHA mismatch: {name}")
        bound[name] = expected
    return bound


def _load_normalization_binding(
    config: dict[str, Any], repo: Path
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    spec = config["normalization_source_config"]
    path = repo / spec["path"] if spec.get("repo_relative") else Path(spec["path"])
    if not path.is_file() or _sha256(path) != spec["sha256"]:
        raise ValueError("normalization source config SHA mismatch")
    source_config = load_json(path)
    means, stds, binding = _load_normalization(source_config)
    binding["source_config_path"] = str(path)
    binding["source_config_sha256"] = spec["sha256"]
    return means, stds, binding


def _validate_parent_gate(config: dict[str, Any]) -> dict[str, Any]:
    spec = config["parent_cpu_gate"]
    path = Path(spec["metrics_path"])
    if not path.is_file() or _sha256(path) != spec["metrics_sha256"]:
        raise ValueError("parent support-aware CPU gate SHA mismatch")
    payload = load_json(path)
    if (
        payload.get("status") != "admission_passed_pending_astra_review"
        or payload.get("code_identity", {}).get("git_commit") != spec["commit"]
        or payload.get("training_performed") is not False
        or payload.get("gpu_used") is not False
    ):
        raise ValueError("parent support-aware CPU gate contract differs")
    return {
        "path": str(path),
        "sha256": spec["metrics_sha256"],
        "commit": spec["commit"],
    }


def _production_fine_sample(
    sampler: Any,
    white: torch.Tensor,
    condition: torch.Tensor,
    mask: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    zeros = torch.zeros_like(white)
    with torch.no_grad(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16
    ):
        return sampler.sample_conditioned(
            background=zeros,
            background_mask=torch.ones_like(zeros),
            obs_values=zeros[:, :2],
            obs_mask=zeros[:, :2],
            water_mask=mask,
            size=tuple(white.shape[-2:]),
            num_timesteps=33,
            device=device,
            method="rk4",
            rtol=1e-5,
            atol=1e-6,
            start_mode="noise",
            initial_noise=white,
            sample_target="state",
            model_conditioning=condition,
            state_channels=6,
            end_time=0.0,
        )


def canonical_truth_and_teacher_coarse(
    truth_normalized: torch.Tensor,
    coarse_normalized: torch.Tensor,
    valid: torch.Tensor,
    means: torch.Tensor,
    stds: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Restore fixed physical atoms before deriving the teacher budget."""
    truth_physical = canonical_physical_decode(
        truth_normalized, means, stds
    )
    teacher_physical, fraction = masked_block_average(
        truth_physical, valid.double()
    )
    decoded_normalized_coarse = canonical_physical_decode(
        coarse_normalized, means, stds
    )
    active = fraction.expand_as(teacher_physical) > 0
    consistency = float(
        (teacher_physical - decoded_normalized_coarse)[active].abs().max()
    )
    tolerance = 8 * torch.finfo(torch.float32).eps
    if consistency > tolerance:
        raise RuntimeError("canonical physical teacher budget differs from normalized coarse")
    return truth_physical, teacher_physical, consistency


def _failure_payload(
    reservation: dict[str, Any],
    error: BaseException,
    *,
    fixed_inputs_sha256: str | None,
    step0_samples_sha256: str | None,
) -> dict[str, Any]:
    result = {
        **reservation,
        "status": "failed",
        "error_type": type(error).__name__,
        "error": str(error)[:2000],
    }
    if fixed_inputs_sha256 is not None:
        result["fixed_inputs_sha256"] = fixed_inputs_sha256
    if step0_samples_sha256 is not None:
        result["step0_samples_sha256"] = step0_samples_sha256
    return result


def run(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = load_json(config_path)
    if config.get("schema_version") != "fine_support_proper_actual_gpu_preflight_v1":
        raise ValueError("unreviewed actual fine-support preflight schema")
    if config.get("protocol") != EXPECTED_PROTOCOL:
        raise ValueError("actual fine-support preflight protocol differs from review")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("actual fine-support preflight requires exactly one visible GPU")
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"refusing to reuse output directory {output_dir}")
    output_dir.mkdir(parents=True)
    status_path = output_dir / "status.json"
    metrics_path = output_dir / "preflight.json"
    reservation = {
        "status": "reserved",
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "optimizer_steps": 0,
    }
    _strict_atomic_json(status_path, reservation)
    tracker = None
    fixed_inputs_sha = None
    evidence_sha = None
    try:
        repo = Path(__file__).resolve().parents[1]
        code_identity = _clean_code_identity(repo)
        parent_gate = _validate_parent_gate(config)
        source = config["source"]
        source_files = _validate_source_files(source)
        means, stds, normalization = _load_normalization_binding(config, repo)
        config_dir = config_path.resolve().parent
        data_path = resolve_path(config["data_config"], config_dir)
        model_path = resolve_path(config["model_config"], config_dir)
        if _sha256_file(data_path) != config["data_config_sha256"]:
            raise ValueError("data config SHA mismatch")
        if _sha256_file(model_path) != config["model_config_sha256"]:
            raise ValueError("model config SHA mismatch")
        data_config = merge_config_overrides(
            load_json(data_path), config.get("data_overrides")
        )
        model_config = load_json(model_path)
        effective_contract = forecast_contract_from_data_config(data_config)
        effective_contract_sha = forecast_contract_sha256(effective_contract)
        if effective_contract_sha != source["forecast_contract_sha256"]:
            raise ValueError("effective forecast contract differs from fine2048")
        seed_everything(int(config["protocol"]["seed"]))
        dataset = build_dataset(data_config, split="train")
        if len(dataset) != EXPECTED_SPLIT_LENGTHS["train"]:
            raise ValueError("train archive length differs from audited inventory")
        sentinel = {
            "dataset": validate_direct_dataset(dataset),
            "calendar": _calendar_inventory(dataset, split="train"),
            "data_config_sha256": _canonical_sha256(data_config),
        }
        index = int(config["protocol"]["case_index"])
        raw = dataset[index]
        case_id = str(raw.get("meta", {}).get("case_id", f"train_index_{index}"))
        device = torch.device("cuda:0")
        tracker = ClearMLTracker(
            config["project_name"],
            f"{config['task_name']}-{output_dir.name}",
            tags=config["clearml"]["tags"],
            env_path=config["clearml"]["env_path"],
        )
        tracker.connect("preflight_contract", config)
        _strict_atomic_json(
            status_path,
            {
                **reservation,
                "status": "running",
                "code_identity": code_identity,
                "clearml_task_id": str(tracker.task.id),
            },
        )
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
            expected_preconditioning_implementation_sha256=implementation[
                "fine_preconditioning"
            ],
            expected_conditioning_implementation_sha256=implementation[
                "fine_conditioning"
            ],
        )
        frozen_model = loaded.sampler.model.eval()
        for parameter in frozen_model.parameters():
            parameter.requires_grad_(False)
        terminal_model = copy.deepcopy(frozen_model).train()
        for parameter in terminal_model.parameters():
            parameter.requires_grad_(True)
        frozen_state = frozen_model.state_dict()
        terminal_state = terminal_model.state_dict()
        if frozen_state.keys() != terminal_state.keys() or any(
            not torch.equal(frozen_state[name], terminal_state[name])
            for name in frozen_state
        ):
            raise RuntimeError("step-zero terminal model differs from fine2048")

        truth = raw["truth"].unsqueeze(0).to(device=device, dtype=torch.float32)
        valid = raw["valid_mask"].unsqueeze(0)[:, :1].to(
            device=device, dtype=torch.float32
        )
        structured = raw["structured_conditioning"].unsqueeze(0).to(
            device=device, dtype=torch.float32
        )
        condition, coarse, _ = teacher_coarse_condition(
            truth, structured, valid
        )
        recovered_teacher, teacher_fraction = masked_block_average(truth, valid)
        if not torch.equal(coarse, recovered_teacher) or not torch.any(teacher_fraction > 0):
            raise RuntimeError("teacher coarse is not exactly D(truth)")
        if tuple(truth.shape[-2:]) != (320, 256):
            raise ValueError("actual fine2048 preflight requires full 320x256 fields")
        members = int(config["protocol"]["members"])
        condition_m = _repeat_members(condition, members)
        valid_m = _repeat_members(valid, members)
        coarse_m = _repeat_members(coarse, members).unflatten(0, (1, members))
        if not torch.equal(coarse_m, coarse_m[:, :1].expand_as(coarse_m)):
            raise RuntimeError("teacher coarse differs across ensemble members")
        generator = torch.Generator(device=device).manual_seed(
            int(config["protocol"]["seed"]) + 1
        )
        white = torch.randn(
            (members, 6, *truth.shape[-2:]),
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            colored = loaded.project_initial_noise(white, valid_m)
        fixed_inputs_path = output_dir / "fixed_inputs.pth"
        fixed_inputs_sha = _atomic_torch_save(
            {
                "truth_normalized": truth.detach().cpu(),
                "teacher_coarse_normalized": coarse.detach().cpu(),
                "valid_mask": valid.detach().cpu(),
                "fine_condition": condition.detach().cpu(),
                "white_initial_noise": white.detach().cpu(),
                "colored_initial_state": colored.detach().cpu(),
                "case_index": index,
                "case_id": case_id,
                "source": source,
                "code_identity": code_identity,
            },
            fixed_inputs_path,
        )
        grid = make_normalized_xy_grid(
            *truth.shape[-2:], device=device, dtype=torch.float32
        )
        torch.cuda.reset_peak_memory_stats(device)
        prefix = fine_frozen_prefix(
            frozen_model, colored, condition_m, valid_m, grid
        )
        if prefix.requires_grad or prefix.grad_fn is not None:
            raise RuntimeError("actual fine2048 prefix retained an autograd graph")
        candidate_residual = terminal_fine_residual(
            terminal_model, prefix, condition_m, valid_m, grid
        )
        with torch.no_grad():
            control_residual = terminal_fine_residual(
                frozen_model, prefix, condition_m, valid_m, grid
            )
            loaded.capture_evidence = True
            production = _production_fine_sample(
                loaded, white, condition_m, valid_m, device
            )
        lift = condition_m[:, -6:]
        candidate = (lift + candidate_residual).unflatten(0, (1, members))
        control = (lift + control_residual).unflatten(0, (1, members))
        production_colored = (
            loaded.evidence[0].get("projected_initial_noise")
            if len(loaded.evidence) == 1
            else None
        )
        colored_equal = torch.is_tensor(production_colored) and torch.equal(
            colored.detach().cpu(), production_colored
        )
        evidence_path = output_dir / "step0_samples.pth"
        evidence_sha = _atomic_torch_save(
            {
                "candidate_normalized": candidate.detach().cpu(),
                "control_normalized": control.detach().cpu(),
                "production_normalized": production.detach().cpu(),
                "truth_normalized": truth.detach().cpu(),
                "teacher_coarse_normalized": coarse.detach().cpu(),
                "valid_mask": valid.detach().cpu(),
                "white_initial_noise": white.detach().cpu(),
                "custom_colored_initial_state": colored.detach().cpu(),
                "production_colored_initial_state": production_colored,
                "case_index": index,
                "case_id": case_id,
                "source": source,
                "code_identity": code_identity,
                "fixed_inputs_sha256": fixed_inputs_sha,
            },
            evidence_path,
        )
        if len(loaded.evidence) != 1 or not torch.is_tensor(production_colored):
            raise RuntimeError("production fine2048 did not capture its initial state")
        if not colored_equal:
            raise RuntimeError("custom and production colored initial states differ")
        candidate_control = float((candidate.detach() - control).abs().max())
        production_replay = float((control.flatten(0, 1) - production).abs().max())
        if candidate_control > 1e-6:
            raise RuntimeError("actual terminal copy differs from frozen fine2048")
        if production_replay > 2e-5:
            raise RuntimeError("actual 31+1 path does not replay production fine2048")

        means_gpu = means.to(device=device).reshape(1, 1, 6, 1, 1)
        stds_gpu = stds.to(device=device).reshape(1, 1, 6, 1, 1)
        candidate_physical = candidate.double() * stds_gpu + means_gpu
        truth_physical, teacher_coarse_physical, teacher_consistency = (
            canonical_truth_and_teacher_coarse(
                truth,
                coarse,
                valid,
                means.to(device=device),
                stds.to(device=device),
            )
        )
        coarse_physical = teacher_coarse_physical[:, None].expand(
            -1, members, -1, -1, -1
        )
        objective, crps, energy, decoded = support_aware_physical_objective(
            candidate_physical,
            coarse_physical,
            truth_physical,
            valid.double(),
            stds.to(device=device),
        )
        residual_coarse, residual_fraction = masked_block_average(
            candidate_residual, valid_m
        )
        residual_error = float(
            residual_coarse[residual_fraction.expand_as(residual_coarse) > 0]
            .abs()
            .max()
        )
        if residual_error > 3e-6:
            raise RuntimeError("actual terminal residual left the detail nullspace")
        objective.backward()
        gradients = [
            parameter.grad
            for parameter in terminal_model.parameters()
            if parameter.grad is not None
        ]
        if not gradients or not all(torch.isfinite(value).all() for value in gradients):
            raise FloatingPointError("actual terminal fine parameters lack finite gradients")
        gradient_norm = float(
            torch.sqrt(sum(value.float().square().sum() for value in gradients)).cpu()
        )
        if not math.isfinite(gradient_norm) or gradient_norm <= 0:
            raise FloatingPointError("actual terminal fine parameter gradient is dead")
        if any(parameter.grad is not None for parameter in frozen_model.parameters()):
            raise RuntimeError("gradient leaked through actual frozen fine prefix")
        recovered, fraction = masked_block_average(
            decoded.flatten(0, 1)[:, 0::2], valid_m
        )
        expected = coarse_physical.flatten(0, 1)[:, 0::2].clamp(0, 1)
        coarse_error = float((recovered - expected)[fraction > 0].abs().max())
        if coarse_error > 3e-14:
            raise RuntimeError("actual decoded SIC lost coarse consistency")
        result = {
            "status": "preflight_passed_pending_astra_review",
            "scientific_role": "actual_checkpoint_zero_update_gpu_preflight_not_training_or_evaluation",
            "training_performed": False,
            "optimizer_created": False,
            "optimizer_steps": 0,
            "checkpoint_sampling_performed": True,
            "test_2023_used": False,
            "code_identity": code_identity,
            "source_files_sha256": source_files,
            "parent_cpu_gate": parent_gate,
            "parent_integration_commit": config["parent_integration_commit"],
            "normalization": normalization,
            "source": source,
            "protocol": config["protocol"],
            "dataset_sentinel": sentinel,
            "effective_forecast_contract": effective_contract,
            "effective_forecast_contract_sha256": effective_contract_sha,
            "case_index": index,
            "case_id": case_id,
            "step0": {
                "candidate_control_max_abs": candidate_control,
                "production_replay_max_abs": production_replay,
                "colored_initial_states_bitwise_equal": colored_equal,
                "objective": float(objective.detach().cpu()),
                "fair_crps": float(crps.detach().cpu()),
                "joint_energy": float(energy.detach().cpu()),
                "terminal_parameter_gradient_norm": gradient_norm,
                "frozen_prefix_has_no_graph": True,
                "frozen_parameter_gradients_absent": True,
                "decoded_sic_coarse_consistency_max_abs": coarse_error,
                "canonical_teacher_normalized_coarse_max_abs": teacher_consistency,
                "residual_nullspace_max_abs": residual_error,
                "peak_gpu_memory_mib": float(
                    torch.cuda.max_memory_allocated(device) / 2**20
                ),
                "candidate_shape": list(candidate.shape),
                "evidence_sha256": evidence_sha,
            },
            "clearml_task_id": str(tracker.task.id),
        }
        _strict_atomic_json(metrics_path, result)
        tracker.connect("actual_fine_support_preflight", result)
        for name, value in result["step0"].items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                tracker.report_single_value(f"preflight/{name}", value)
        tracker.upload_artifact("actual_fine_support_step0", evidence_path)
        tracker.upload_artifact("actual_fine_support_preflight", metrics_path)
        terminal = {
            **reservation,
            "code_identity": code_identity,
            "clearml_task_id": str(tracker.task.id),
            "scientific_role": result["scientific_role"],
            "evidence_path": str(evidence_path),
            "evidence_sha256": evidence_sha,
            "fixed_inputs_path": str(fixed_inputs_path),
            "fixed_inputs_sha256": fixed_inputs_sha,
        }
        _finish_success(tracker, status_path, metrics_path, terminal)
        tracker = None
        return result
    except BaseException as error:
        try:
            _strict_atomic_json(
                status_path,
                _failure_payload(
                    reservation,
                    error,
                    fixed_inputs_sha256=fixed_inputs_sha,
                    step0_samples_sha256=evidence_sha,
                ),
            )
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
    signal.signal(signal.SIGTERM, _terminate)
    signal.signal(signal.SIGINT, _terminate)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve(), args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
