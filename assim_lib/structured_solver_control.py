"""Paired validation-only solver/precision control for a frozen structured CFM.

This diagnostic never trains or selects a checkpoint.  It holds checkpoint,
validation cases and initial noise fixed while refining the RK4 grid and while
running one genuine FP32 forward control.  Raw members remain on the server;
only compact metrics and fixed visual panels are emitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
from diffusers.training_utils import EMAModel
from torch.utils.data._utils.collate import default_collate

from .config import TrainingConfig, load_json, merge_config_overrides, resolve_path
from .data import build_dataset
from .model_io import build_unet
from .structured_archive_audit import validate_bound_archive_audit
from .structured_joint_state import (
    StructuredDecodeSaturationError,
    canonical_mapping_sha256,
    validate_conditioning_normalization,
)
from .structured_trajectory_evaluation import (
    make_structured_trajectory_figure,
    sample_structured_batch,
    structured_trajectory_metrics,
)
from .sampler import Sampler


SCHEMA_VERSION = "structured_solver_control_v1"
CASE_INDICES = (0, 90)
MEMBER_SEEDS = (16012880, 16012981)
RK4_TIMEPOINTS = (33, 65, 129)
FP32_TIMEPOINTS = 65
SCORE_RELATIVE_TOLERANCE = 0.01
EVENT_PROBABILITY_ABSOLUTE_TOLERANCE = 0.01
TEXTURE_RELATIVE_TOLERANCE = 0.01
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_contract(
    experiment_path: Path,
) -> tuple[dict, dict, TrainingConfig, dict[str, Any]]:
    experiment = load_json(experiment_path)
    config_dir = experiment_path.resolve().parent
    data_path = resolve_path(experiment["data_config"], config_dir).resolve()
    method_path = resolve_path(experiment["model_config"], config_dir).resolve()
    data_config = merge_config_overrides(
        load_json(data_path), experiment.get("data_overrides")
    )
    method = {**load_json(method_path), **experiment.get("training", {})}
    if method.get("training_objective") != "structured_joint_state_flow":
        raise ValueError("solver control requires structured_joint_state_flow")
    audit = validate_bound_archive_audit(data_config, data_path)
    stats_path = resolve_path(
        method["structured_state_stats_path"], method_path.parent
    ).resolve()
    stats = load_json(stats_path)
    method["structured_state_stats"] = stats
    validate_conditioning_normalization(data_config, stats)
    if stats.get("data_config_sha256") != canonical_mapping_sha256(data_config):
        raise ValueError("structured stats/data identity mismatch")
    config = TrainingConfig.from_dict(method)
    if config.structured_velocity_parameterization != "raw":
        raise ValueError("reference solver control is bound to raw velocity")
    return data_config, stats, config, audit


def _load_epoch_ema(
    model: torch.nn.Module,
    config: TrainingConfig,
    run_dir: Path,
    recovery_epoch: int,
    *,
    expected_metadata_sha256: str,
    expected_ema_sha256: str,
    expected_resume_sha256: str,
) -> tuple[Path, Path, Path]:
    metadata_path = run_dir / "metadata.json"
    recovery_dir = (
        run_dir
        / config.recovery_checkpoint_name
        / f"epoch_{int(recovery_epoch):04d}"
    )
    ema_path = recovery_dir / "ema_state.pth"
    resume_path = recovery_dir / "resume.json"
    for path in (metadata_path, ema_path, resume_path):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"required frozen reference artifact is unsafe or absent: {path}")
    expected_hashes = {
        metadata_path: expected_metadata_sha256,
        ema_path: expected_ema_sha256,
        resume_path: expected_resume_sha256,
    }
    for path, expected in expected_hashes.items():
        if _SHA256_RE.fullmatch(str(expected)) is None:
            raise ValueError(f"invalid frozen SHA-256 for {path.name}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(
                f"frozen reference SHA-256 differs for {path}: "
                f"expected {expected}, got {actual}"
            )
    metadata = load_json(metadata_path)
    training = metadata.get("training_config")
    if not isinstance(training, dict):
        raise ValueError("reference metadata has no training_config")
    if training.get("training_objective") != "structured_joint_state_flow":
        raise ValueError("reference metadata objective mismatch")
    if training.get("structured_velocity_parameterization", "raw") != "raw":
        raise ValueError("reference checkpoint is not the raw-velocity baseline")
    resume = load_json(resume_path)
    if resume.get("next_epoch") != int(recovery_epoch):
        raise ValueError("reference recovery epoch metadata mismatch")
    if resume.get("global_step") != 2128 or int(recovery_epoch) != 16:
        raise ValueError("solver control is frozen to epoch 15 / 2128 optimizer steps")
    try:
        state = torch.load(ema_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(ema_path, map_location="cpu")
    ema = EMAModel(
        model.parameters(),
        decay=float(config.ema_decay),
        min_decay=float(config.ema_min_decay),
        update_after_step=int(config.ema_update_after_step),
        use_ema_warmup=bool(config.ema_use_warmup),
    )
    ema.load_state_dict(state)
    ema.copy_to(model.parameters())
    return metadata_path, ema_path, resume_path


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device=device, dtype=torch.float32)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


def _texture_energy(field: torch.Tensor, valid: torch.Tensor) -> float:
    value = torch.where(valid > 0, field, torch.zeros_like(field))
    dy = value[..., 1:, :] - value[..., :-1, :]
    dx = value[..., :, 1:] - value[..., :, :-1]
    dy_mask = ((valid[..., 1:, :] > 0) & (valid[..., :-1, :] > 0)).expand_as(dy)
    dx_mask = ((valid[..., :, 1:] > 0) & (valid[..., :, :-1] > 0)).expand_as(dx)
    numerator = dy[dy_mask].square().sum(dtype=torch.float64) + dx[
        dx_mask
    ].square().sum(dtype=torch.float64)
    denominator = dy_mask.sum(dtype=torch.float64) + dx_mask.sum(dtype=torch.float64)
    if denominator <= 0:
        raise ValueError("texture diagnostic has no valid neighbor pairs")
    return float((numerator / denominator).item())


def _sic_neighbor_energy_by_lead(
    ensemble: torch.Tensor,
    valid: torch.Tensor,
    lag0: torch.Tensor,
) -> dict[str, float]:
    sic = ensemble[:, :, 0::2]
    spatial_valid = valid[:, :1] > 0
    result = {}
    for lead in range(sic.shape[2]):
        domain = spatial_valid
        if lead == 0:
            domain = domain & ~(lag0[:, :1] > 0)
        result[f"lead{lead}"] = _texture_energy(
            sic[:, :, lead : lead + 1], domain[:, None]
        )
    return result


@contextmanager
def _precision_context(use_bf16: bool):
    if use_bf16:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            yield
        return
    previous_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    previous_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    previous_matmul_precision = torch.get_float32_matmul_precision()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    try:
        with torch.autocast(device_type="cuda", enabled=False):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_matmul_tf32
        torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32
        torch.set_float32_matmul_precision(previous_matmul_precision)


def _sample_variant(
    *,
    sampler: Sampler,
    batches: list[dict[str, Any]],
    stats: dict,
    config: TrainingConfig,
    timepoints: int,
    use_bf16: bool,
) -> tuple[torch.Tensor | None, dict[str, Any]]:
    members_by_case: list[torch.Tensor] = []
    failures: list[dict[str, Any]] = []
    for case_position, batch in enumerate(batches):
        members = []
        for member_position, seed in enumerate(MEMBER_SEEDS):
            generator = torch.Generator(device=batch["background"].device)
            generator.manual_seed(int(seed))
            initial_noise = torch.randn(
                (1, config.out_channels, *config.image_size),
                device=batch["background"].device,
                dtype=torch.float32,
                generator=generator,
            )
            try:
                with _precision_context(use_bf16):
                    sample = sample_structured_batch(
                        sampler,
                        batch,
                        stats=stats,
                        size=config.image_size,
                        num_timesteps=int(timepoints),
                        device=batch["background"].device,
                        method="rk4",
                        rtol=config.sample_rtol,
                        atol=config.sample_atol,
                        initial_noise=initial_noise,
                    )
            except StructuredDecodeSaturationError as error:
                failures.append(
                    {
                        "case_position": case_position,
                        "member_position": member_position,
                        "member_seed": int(seed),
                        "error": str(error),
                        "diagnostics": error.diagnostics,
                    }
                )
                continue
            members.append(sample.detach().to(device="cpu", dtype=torch.float32))
        if len(members) != len(MEMBER_SEEDS):
            continue
        members_by_case.append(torch.stack(members, dim=1))
    if failures or len(members_by_case) != len(batches):
        return None, {
            "status": "failed_decode_saturation",
            "partial_metrics_permitted": False,
            "failures": failures,
        }
    return torch.cat(members_by_case, dim=0), {
        "status": "passed",
        "partial_metrics_permitted": False,
        "precision": (
            {
                "network_autocast": "bf16",
                "ode_state_dtype": "float32",
                "tf32_disabled": False,
            }
            if use_bf16
            else {
                "network_autocast": "disabled",
                "model_parameter_dtype": "float32",
                "input_and_ode_state_dtype": "float32",
                "matmul_tf32": False,
                "cudnn_tf32": False,
                "float32_matmul_precision": "highest",
            }
        ),
    }


def _paired_field_differences(
    left: torch.Tensor,
    right: torch.Tensor,
    valid: torch.Tensor,
    lag0: torch.Tensor,
    *,
    sic_cap: float,
) -> dict[str, Any]:
    if left.shape != right.shape or left.ndim != 5:
        raise ValueError("paired solver ensembles must have identical [B,M,2*T,H,W] shape")
    if left.shape[2] % 2:
        raise ValueError("paired solver trajectories require SIC/SIT channel pairs")
    spatial_valid = valid[:, :1] > 0
    result: dict[str, Any] = {"fields": {}, "events": {}}
    for lead in range(left.shape[2] // 2):
        domain = spatial_valid
        if lead == 0:
            domain = domain & ~(lag0[:, :1] > 0)
        if not torch.any(domain):
            raise ValueError(f"paired solver domain is empty at lead {lead}")
        for offset, field in enumerate(("sic", "sit")):
            channel = 2 * lead + offset
            delta = (left[:, :, channel : channel + 1] - right[:, :, channel : channel + 1]).abs()
            selector = domain[:, None].expand_as(delta)
            selected = delta[selector]
            result["fields"][f"lead{lead}_{field}"] = {
                "mean_abs_difference": float(selected.to(torch.float64).mean().item()),
                "max_abs_difference": float(selected.max().item()),
                "evaluated_member_points": int(selected.numel()),
                "domain": "unobserved_valid_ocean" if lead == 0 else "valid_ocean",
            }
        left_sic = left[:, :, 2 * lead : 2 * lead + 1]
        right_sic = right[:, :, 2 * lead : 2 * lead + 1]
        for event_name, left_event, right_event in (
            ("occurrence", left_sic > 0, right_sic > 0),
            ("cap", left_sic == sic_cap, right_sic == sic_cap),
        ):
            probability_delta = (
                left_event.to(torch.float64).mean(dim=1)
                - right_event.to(torch.float64).mean(dim=1)
            ).abs()
            selected = probability_delta[domain]
            result["events"][f"lead{lead}_{event_name}"] = {
                "mean_absolute_probability_difference": float(selected.mean().item()),
                "max_absolute_probability_difference": float(selected.max().item()),
                "evaluated_points": int(selected.numel()),
                "domain": "unobserved_valid_ocean" if lead == 0 else "valid_ocean",
            }
    return result


def _relative_difference(left: float, right: float) -> float:
    denominator = max(abs(float(left)), abs(float(right)), 1e-12)
    return abs(float(left) - float(right)) / denominator


def _validate_reference_semantics(
    metadata_path: Path,
    data_config: dict[str, Any],
    stats: dict[str, Any],
    config: TrainingConfig,
) -> None:
    metadata = load_json(metadata_path)
    stored_data = metadata.get("data_config")
    training = metadata.get("training_config")
    if not isinstance(stored_data, dict) or not isinstance(training, dict):
        raise ValueError("reference metadata lacks frozen data/training configuration")
    if canonical_mapping_sha256(stored_data) != canonical_mapping_sha256(data_config):
        raise ValueError("reference metadata data config differs from solver contract")
    stored_stats = training.get("structured_state_stats")
    if not isinstance(stored_stats, dict) or canonical_mapping_sha256(
        stored_stats
    ) != canonical_mapping_sha256(stats):
        raise ValueError("reference metadata structured statistics differ")
    fields = (
        "image_size",
        "in_channels",
        "out_channels",
        "trajectory_horizon_days",
        "block_out_channels",
        "layers_per_block",
        "down_block_types",
        "up_block_types",
        "norm_num_groups",
    )
    stored_architecture = {name: training.get(name) for name in fields}
    current_architecture = {name: config.__dict__[name] for name in fields}
    if canonical_mapping_sha256(stored_architecture) != canonical_mapping_sha256(
        current_architecture
    ):
        raise ValueError("reference metadata architecture differs from solver contract")


def _solver_gate(
    variants: dict[str, Any],
    comparisons: dict[str, Any]
) -> dict[str, Any]:
    required_pairs = (
        "rk4_64_intervals_bf16_vs_rk4_128_intervals_bf16",
        "rk4_64_intervals_bf16_vs_fp32",
    )
    required_variants = (
        "rk4_32_intervals_bf16",
        "rk4_64_intervals_bf16",
        "rk4_128_intervals_bf16",
        "rk4_64_intervals_fp32",
    )
    if any(variants.get(name, {}).get("status") != "passed" for name in required_variants):
        return {
            "status": "failed",
            "reason": "one or more required variants failed strict physical decode",
            "pilot_permitted": False,
        }
    if any(name not in comparisons for name in required_pairs):
        return {
            "status": "indeterminate",
            "reason": "a required paired comparison is absent",
            "pilot_permitted": False,
        }

    checks: dict[str, Any] = {}
    passed = True
    for pair in required_pairs:
        comparison = comparisons[pair]
        left_name, right_name = comparison["variants"]
        score_differences = {}
        for key, left_value in variants[left_name]["metrics"].items():
            if not (key.endswith("_fair_crps") or key.endswith("_mean_rmse")):
                continue
            right_value = variants[right_name]["metrics"][key]
            score_differences[key] = _relative_difference(left_value, right_value)
        maximum_score_change = max(score_differences.values(), default=float("inf"))
        event_changes = [
            row["mean_absolute_probability_difference"]
            for row in comparison["events"].values()
        ]
        maximum_event_change = max(event_changes, default=float("inf"))
        texture_differences = {
            lead: _relative_difference(
                value,
                variants[right_name]["sic_neighbor_energy_by_lead"][lead],
            )
            for lead, value in variants[left_name][
                "sic_neighbor_energy_by_lead"
            ].items()
        }
        texture_change = max(texture_differences.values(), default=float("inf"))
        pair_passed = (
            maximum_score_change <= SCORE_RELATIVE_TOLERANCE
            and maximum_event_change <= EVENT_PROBABILITY_ABSOLUTE_TOLERANCE
            and texture_change <= TEXTURE_RELATIVE_TOLERANCE
        )
        passed = passed and pair_passed
        checks[pair] = {
            "passed": pair_passed,
            "per_score_relative_differences": score_differences,
            "maximum_score_relative_difference": maximum_score_change,
            "maximum_mean_event_probability_difference": maximum_event_change,
            "sic_neighbor_energy_relative_difference": texture_change,
            "per_lead_sic_neighbor_energy_relative_differences": (
                texture_differences
            ),
        }
    return {
        "status": "converged" if passed else "failed",
        "pilot_permitted": passed,
        "thresholds": {
            "maximum_score_relative_difference": SCORE_RELATIVE_TOLERANCE,
            "maximum_mean_event_probability_difference": (
                EVENT_PROBABILITY_ABSOLUTE_TOLERANCE
            ),
            "maximum_sic_neighbor_energy_relative_difference": (
                TEXTURE_RELATIVE_TOLERANCE
            ),
        },
        "checks": checks,
    }


def run_control(
    experiment_path: Path,
    run_dir: Path,
    output_dir: Path,
    recovery_epoch: int = 16,
    *,
    expected_metadata_sha256: str,
    expected_ema_sha256: str,
    expected_resume_sha256: str,
) -> dict[str, Any]:
    attempt_token = os.environ.get("STRUCTURED_SOLVER_CONTROL_ATTEMPT", "")
    if not attempt_token:
        raise ValueError("missing structured solver-control ownership token")
    owner_path = output_dir / ".structured_solver_control_owner"
    if (
        output_dir.is_symlink()
        or not output_dir.is_dir()
        or owner_path.is_symlink()
        or not owner_path.is_file()
        or owner_path.read_text(encoding="utf-8").strip() != attempt_token
        or {item.name for item in output_dir.iterdir()}
        != {owner_path.name}
    ):
        raise FileExistsError(
            "solver-control output was not atomically created by this attempt"
        )
    _atomic_json(
        output_dir / "run_status.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": "running",
            "completed_cases": 0,
            "expected_cases": len(CASE_INDICES),
            "attempt_token": attempt_token,
        },
    )
    if torch.cuda.device_count() != 1:
        raise RuntimeError("solver control requires exactly one visible CUDA device")
    device = torch.device("cuda:0")
    data_config, stats, config, archive_audit = _load_contract(experiment_path)
    dataset = build_dataset(data_config, split="valid")
    dataset.validate_structured_sral_audit_contract(archive_audit)
    if max(CASE_INDICES) >= len(dataset):
        raise ValueError("frozen solver-control case index exceeds validation dataset")
    batches = [
        _move_batch(default_collate([dataset[index]]), device)
        for index in CASE_INDICES
    ]
    case_ids = [str(batch["meta"]["case_id"][0]) for batch in batches]

    model = build_unet(config)
    metadata_path, ema_path, resume_path = _load_epoch_ema(
        model,
        config,
        run_dir,
        recovery_epoch,
        expected_metadata_sha256=expected_metadata_sha256,
        expected_ema_sha256=expected_ema_sha256,
        expected_resume_sha256=expected_resume_sha256,
    )
    _validate_reference_semantics(
        metadata_path, data_config, stats, config
    )
    model.to(device).eval()
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise RuntimeError("frozen reference model parameters must be float32")
    if hasattr(model, "enable_xformers_memory_efficient_attention"):
        model.enable_xformers_memory_efficient_attention()
    sampler = Sampler(model, structured_velocity_parameterization="raw")

    truth = torch.cat(
        [batch["structured_physical_truth"].cpu() for batch in batches], dim=0
    )
    background = torch.cat(
        [batch["structured_physical_background"].cpu() for batch in batches], dim=0
    )
    valid = torch.cat([batch["valid_mask"].cpu() for batch in batches], dim=0)
    lag0 = torch.cat(
        [batch["structured_lag0_mask"].cpu() for batch in batches], dim=0
    )

    variants: dict[str, Any] = {}
    successful: dict[str, torch.Tensor] = {}
    for timepoints in RK4_TIMEPOINTS:
        label = f"rk4_{timepoints - 1}_intervals_bf16"
        ensemble, status = _sample_variant(
            sampler=sampler,
            batches=batches,
            stats=stats,
            config=config,
            timepoints=timepoints,
            use_bf16=True,
        )
        variants[label] = status
        if ensemble is not None:
            successful[label] = ensemble
            variants[label]["metrics"] = structured_trajectory_metrics(
                ensemble,
                truth,
                background,
                valid,
                lag0_mask=lag0,
                sic_cap=float(stats["sic_cap"]),
                exclude_lag0_from_day0_scores=True,
            )
            variants[label]["sic_neighbor_energy_by_lead"] = (
                _sic_neighbor_energy_by_lead(ensemble, valid, lag0)
            )

    fp32_batch = batches
    fp32_ensemble, fp32_status = _sample_variant(
        sampler=sampler,
        batches=fp32_batch,
        stats=stats,
        config=config,
        timepoints=FP32_TIMEPOINTS,
        use_bf16=False,
    )
    fp32_label = "rk4_64_intervals_fp32"
    variants[fp32_label] = fp32_status
    if fp32_ensemble is not None:
        successful[fp32_label] = fp32_ensemble
        variants[fp32_label]["metrics"] = structured_trajectory_metrics(
            fp32_ensemble,
            truth,
            background,
            valid,
            lag0_mask=lag0,
            sic_cap=float(stats["sic_cap"]),
            exclude_lag0_from_day0_scores=True,
        )
        variants[fp32_label]["sic_neighbor_energy_by_lead"] = (
            _sic_neighbor_energy_by_lead(fp32_ensemble, valid, lag0)
        )

    comparisons: dict[str, Any] = {}
    for left, right in (
        ("rk4_32_intervals_bf16", "rk4_64_intervals_bf16"),
        ("rk4_64_intervals_bf16", "rk4_128_intervals_bf16"),
    ):
        if left in successful and right in successful:
            comparisons[f"{left}_vs_{right}"] = {
                "variants": [left, right],
                **_paired_field_differences(
                    successful[left],
                    successful[right],
                    valid,
                    lag0,
                    sic_cap=float(stats["sic_cap"]),
                ),
            }
    bf16_key = "rk4_64_intervals_bf16"
    if bf16_key in successful and fp32_label in successful:
        comparisons["rk4_64_intervals_bf16_vs_fp32"] = {
            "variants": [bf16_key, fp32_label],
            **_paired_field_differences(
                successful[bf16_key],
                successful[fp32_label],
                valid,
                lag0,
                sic_cap=float(stats["sic_cap"]),
            ),
        }

    visual_dir = output_dir / "visual_qc"
    visual_dir.mkdir()
    for label, ensemble in successful.items():
        figure = make_structured_trajectory_figure(
            truth[0],
            background[0],
            ensemble[0, 0],
            valid[0],
            title=f"frozen raw-velocity epoch 15: {label}",
        )
        figure.savefig(
            visual_dir / f"{label}.png", dpi=180, bbox_inches="tight"
        )
        import matplotlib.pyplot as plt

        plt.close(figure)

    gate = _solver_gate(variants, comparisons)
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": gate["status"],
        "selection_permitted": False,
        "training_performed": False,
        "test_2023_used": False,
        "reference": {
            "experiment_sha256": _sha256(experiment_path),
            "metadata_sha256": _sha256(metadata_path),
            "ema_checkpoint_sha256": _sha256(ema_path),
            "resume_metadata_sha256": _sha256(resume_path),
            "recovery_epoch": int(recovery_epoch),
            "optimizer_steps": 2128,
            "velocity_parameterization": "raw",
        },
        "case_indices": list(CASE_INDICES),
        "case_ids": case_ids,
        "member_seeds": list(MEMBER_SEEDS),
        "variants": variants,
        "paired_comparisons": comparisons,
        "solver_gate": gate,
        "decision_rule": (
            "If physical structure materially changes with RK4 refinement or "
            "FP32, correct sampling before any retraining; otherwise the speckle "
            "is a model/target-coordinate failure and the preconditioned pilot "
            "may proceed."
        ),
    }
    _atomic_json(output_dir / "solver_control.json", result)
    gate_artifact = {
        "schema_version": "structured_solver_gate_v1",
        "status": gate["status"],
        "pilot_permitted": gate.get("pilot_permitted", False),
        "solver_control_sha256": _sha256(output_dir / "solver_control.json"),
        "reference": result["reference"],
        "thresholds": gate.get("thresholds"),
    }
    _atomic_json(output_dir / "solver_gate.json", gate_artifact)
    _atomic_json(
        output_dir / "run_status.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": "completed",
            "completed_cases": len(CASE_INDICES),
            "expected_cases": len(CASE_INDICES),
            "scientific_status": gate["status"],
            "pilot_permitted": gate.get("pilot_permitted", False),
            "selection_permitted": False,
            "training_performed": False,
            "attempt_token": attempt_token,
        },
    )
    owner_path.unlink()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--recovery-epoch", type=int, default=16)
    parser.add_argument("--expected-metadata-sha256", required=True)
    parser.add_argument("--expected-ema-sha256", required=True)
    parser.add_argument("--expected-resume-sha256", required=True)
    args = parser.parse_args()
    result = run_control(
        args.experiment.resolve(),
        args.run_dir.resolve(),
        args.output_dir.resolve(),
        args.recovery_epoch,
        expected_metadata_sha256=args.expected_metadata_sha256,
        expected_ema_sha256=args.expected_ema_sha256,
        expected_resume_sha256=args.expected_resume_sha256,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
