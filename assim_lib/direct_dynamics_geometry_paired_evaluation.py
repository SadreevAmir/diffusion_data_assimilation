"""Frozen three-way validation of EMA6 and two geometry-score continuations.

This is deliberately a narrow development evaluator.  It samples the public
EMA6 law and the two admitted hybrid laws with common cases and common random
numbers, writes the complete raw ensembles before scoring, and never touches
the test split or performs fitting/selection on validation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import signal
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .clearml_tracking import ClearMLTracker
from .config import TrainingConfig, load_json
from .data import build_dataset
from .direct_dynamics_cascade_checkpoint_evaluation import _save_rank_histograms
from .direct_dynamics_cascade_paired_evaluation import (
    _joint_energy_score,
    _roughness,
    _weighted_case_fair_crps,
    _weighted_case_rmse,
    _weighted_case_rms,
    _weighted_fractional_rank,
)
from .direct_dynamics_geometry_preflight import _cell_area_proxy
from .direct_dynamics_geometry_score import (
    geometry_observable_vectors,
    unbiased_energy_score_vectors,
)
from .direct_dynamics_geometry_training import SCHEMA_VERSION as TRAINING_SCHEMA
from .direct_dynamics_suffix import frozen_prefix, trainable_suffix
from .direct_dynamics_training import (
    DIRECT_LEADS,
    DIRECT_OUTPUT_CHANNELS,
    _repeat_field_stats,
    validate_direct_dataset,
)
from .model_io import build_unet, load_sampler
from .runtime import make_normalized_xy_grid
from .structured_trajectory_evaluation import make_structured_trajectory_figure
from .trainer import _atomic_json
from .transforms import channel_denormalize


SCHEMA_VERSION = "direct_dynamics_geometry_paired_validation_v1"
LABELS = ("ema6", "control256", "treatment256")
OUTPUT_KEYS = tuple(
    f"d{lead}_{field}" for lead in DIRECT_LEADS for field in ("sic", "sit")
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_torch_save(payload: Any, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.incomplete")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _check_contract(experiment: dict[str, Any]) -> dict[str, Any]:
    if experiment.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unreviewed paired-validation schema")
    protocol = experiment["protocol"]
    expected = {
        "split": "valid",
        "validation_year": 2022,
        "development_reuse": True,
        "case_indices": [0, 777, 1553, 2330, 3107, 3883, 4660, 5436, 6213, 6990, 7766, 8543],
        "members": 8,
        "rk4_timepoints": 17,
        "network_precision": "bf16",
        "state_precision": "fp32",
        "raw_unclipped": True,
        "optimizer_steps": 0,
        "noise_seed": 314159,
        "bootstrap_replicates": 100000,
        "bootstrap_seed": 20260912,
        "visual_case_orders": [0, 3, 6, 9],
        "test_2023_accessed": False,
    }
    if protocol != expected:
        mismatch = {key: (protocol.get(key), value) for key, value in expected.items() if protocol.get(key) != value}
        raise ValueError(f"unreviewed paired-validation protocol: {mismatch}")
    expected_gate = {
        "geometry_effect": "treatment-control geometry-ES paired bootstrap upper95 < 0 and native CRPS ratio upper95 <= 1.01",
        "replacement": "treatment-EMA6 primary CRPS paired bootstrap upper95 < 0 plus all frozen safety margins",
        "per_output_crps_rmse_ratio_max": 1.01,
        "native_energy_ratio_max": 1.01,
        "rank_tv_increase_max": 0.01,
        "coverage_absolute_error_increase_max": 0.01,
        "adjusted_ssr_absolute_error_increase_max": 0.02,
        "roughness_ratio_max": 1.02,
        "manual_visual_review_required": True,
    }
    if experiment.get("gate") != expected_gate:
        raise ValueError("configured gate differs from executable frozen gate")
    return protocol


def _finite_tree(value: Any, path: str = "result") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _finite_tree(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _finite_tree(child, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise FloatingPointError(f"non-finite value at {path}")


def _merge_status(status_path: Path, **updates: Any) -> dict[str, Any]:
    current = json.loads(status_path.read_text()) if status_path.is_file() else {}
    current.update(updates)
    _atomic_json(status_path, current)
    return current


def _record_evidence_binding(bindings_path: Path, name: str, path: Path) -> dict[str, Any]:
    payload = json.loads(bindings_path.read_text()) if bindings_path.is_file() else {"artifacts": {}}
    payload.setdefault("artifacts", {})[name] = {"path": str(path), "sha256": _sha256(path)}
    _atomic_json(bindings_path, payload)
    return payload["artifacts"][name]


def _record_terminal_failure(status_path: Path, tracker: Any, error: BaseException) -> None:
    """Persist the primary failure before best-effort tracking cleanup."""
    try:
        _merge_status(
            status_path,
            status="failed_terminal_non_resumable",
            error_type=type(error).__name__,
            error=str(error),
            test_2023_accessed=False,
        )
    except Exception:
        pass
    if tracker is not None:
        try:
            tracker.task.mark_failed(status_reason=f"{type(error).__name__}: {error}")
        except Exception:
            pass
        try:
            tracker.close()
        except Exception:
            pass


def _close_and_mark_complete(status_path: Path, tracker: Any, clearml_task_id: str) -> None:
    tracker.close()
    _merge_status(
        status_path,
        status="complete_pending_manual_visual_gate",
        clearml_task_id=clearml_task_id,
        test_2023_accessed=False,
    )


def _verify_final_checkpoint(path: Path, expected_sha: str, arm: str) -> dict[str, Any]:
    if _sha256(path) != expected_sha:
        raise ValueError(f"{arm} final checkpoint SHA mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        payload.get("schema_version") != TRAINING_SCHEMA
        or payload.get("arm") != arm
        or payload.get("completed_updates") != 256
        or payload.get("inference_law") != "frozen_ema6_prefix12_plus_arm_candidate_suffix4"
        or len(payload.get("history", [])) != 256
    ):
        raise ValueError(f"{arm} final checkpoint lifecycle mismatch")
    if any(not torch.isfinite(value).all() for value in payload["model"].values() if torch.is_tensor(value)):
        raise FloatingPointError(f"{arm} model contains NaN/Inf")
    _finite_tree(payload["history"], f"{arm}.history")
    return payload


def _verify_optimizer_against_model(payload: dict[str, Any], model: torch.nn.Module, arm: str) -> None:
    optimizer = payload.get("optimizer", {})
    groups = optimizer.get("param_groups")
    states = optimizer.get("state")
    if not isinstance(groups, list) or len(groups) != 1 or not isinstance(states, dict):
        raise ValueError(f"{arm} optimizer structure mismatch")
    group = groups[0]
    if group.get("lr") != 1e-5 or group.get("weight_decay") != 0.0:
        raise ValueError(f"{arm} optimizer hyperparameters changed")
    identifiers = group.get("params")
    if not isinstance(identifiers, list) or len(identifiers) != len(set(identifiers)):
        raise ValueError(f"{arm} optimizer parameter identifiers are malformed")
    if set(identifiers) != set(states):
        raise ValueError(f"{arm} optimizer state coverage is incomplete")
    parameters = list(model.parameters())
    if len(parameters) != len(identifiers):
        raise ValueError(f"{arm} optimizer/model parameter coverage differs")
    for identifier, parameter in zip(identifiers, parameters, strict=True):
        state = states[identifier]
        if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError(f"{arm} AdamW state keys differ for parameter {identifier}")
        step = state["step"]
        if not torch.is_tensor(step) or step.numel() != 1 or not torch.isfinite(step).all() or float(step.item()) != 256.0:
            raise ValueError(f"{arm} AdamW step is not exactly 256")
        for name in ("exp_avg", "exp_avg_sq"):
            value = state[name]
            if not torch.is_tensor(value) or value.shape != parameter.shape or not torch.isfinite(value).all():
                raise ValueError(f"{arm} AdamW {name} shape/finite mismatch")


def _verify_matched_training(
    experiment: dict[str, Any], training_result: dict[str, Any], payloads: dict[str, dict[str, Any]]
) -> None:
    if (
        training_result.get("status") != "training_complete_pending_paired_validation"
        or training_result.get("test_2023_accessed") is not False
        or training_result.get("completed_updates") != {"control": 256, "treatment": 256}
    ):
        raise ValueError("matched training did not finish the admitted lifecycle")
    source = training_result.get("source")
    protocol = training_result.get("protocol")
    code_identity = training_result.get("code_identity")
    schedule = training_result.get("schedule")
    if not isinstance(schedule, list) or len(schedule) != 256 or len(set(schedule)) != 256:
        raise ValueError("matched training schedule is malformed")
    if source != experiment.get("source"):
        raise ValueError("evaluation source differs from matched training source")
    if training_result.get("final_checkpoints") != {
        arm: {
            "path": experiment["checkpoints"][arm]["path"],
            "sha256": experiment["checkpoints"][arm]["sha256"],
            "completed_updates": 256,
        }
        for arm in ("control", "treatment")
    }:
        raise ValueError("training-result final checkpoint binding mismatch")
    for arm, payload in payloads.items():
        if payload.get("source") != source or payload.get("protocol") != protocol or payload.get("code_identity") != code_identity:
            raise ValueError(f"{arm} source/protocol/code identity differs from training result")
        for update, (record, train_index) in enumerate(zip(payload["history"], schedule), start=1):
            if (
                record.get("update") != update
                or record.get("measurement_phase") != "pre_step"
                or record.get("model_completed_updates") != update - 1
                or record.get("train_index") != train_index
                or record.get("noise_seed") != int(protocol["seed"]) + update
                or record.get("suffix_replay_max_abs") != 0.0
            ):
                raise ValueError(f"{arm} history binding mismatch at update {update}")
    for relative in ("direct_dynamics_geometry_score.py", "direct_dynamics_suffix.py"):
        key = "geometry_score_sha256" if relative.startswith("direct_dynamics_geometry") else "suffix_sha256"
        if _sha256(Path(__file__).with_name(relative)) != code_identity.get(key):
            raise ValueError(f"current {relative} differs from admitted training code")
    preflight = training_result.get("preflight", {})
    if _sha256(Path(preflight["path"])) != preflight.get("sha256"):
        raise ValueError("training preflight artifact binding mismatch")
    preflight_payload = json.loads(Path(preflight["path"]).read_text())
    if (
        preflight_payload.get("status") != "preflight_passed"
        or preflight_payload.get("optimizer_steps") != 0
        or preflight_payload.get("test_2023_accessed") is not False
        or preflight_payload.get("verified_sha256") != source["sha256"]
        or preflight_payload.get("production_replay_max_abs") != 0.0
    ):
        raise ValueError("bound preflight no longer proves the admitted law")
    audit = experiment["product_scale_audit"]
    repository_root = Path(__file__).parents[1]
    for key, sha_key in (("path", "sha256"), ("script_path", "script_sha256")):
        if _sha256(repository_root / audit[key]) != audit[sha_key]:
            raise ValueError(f"product-scale audit binding mismatch: {key}")
    if (
        preflight_payload["code_identity"].get("product_scale_audit_sha256") != audit["sha256"]
        or preflight_payload["code_identity"].get("product_scale_script_sha256") != audit["script_sha256"]
    ):
        raise ValueError("product-scale audit differs from admitted preflight")
    area = training_result.get("cell_area", {})
    expected_geometry = experiment["cell_geometry"]
    if _sha256(Path(expected_geometry["latitude_path"])) != area.get("latitude_sha256"):
        raise ValueError("latitude geometry differs from admitted training")
    if _sha256(Path(expected_geometry["longitude_path"])) != area.get("longitude_sha256"):
        raise ValueError("longitude geometry differs from admitted training")
    if float(experiment["product_scale"]) != float(protocol["product_scale"]):
        raise ValueError("product scale differs from admitted training")
    if float(experiment["product_scale"]) != float(preflight_payload["protocol"]["product_scale"]):
        raise ValueError("product scale differs from admitted preflight")


def _case_identity(item: dict[str, Any], index: int) -> dict[str, Any]:
    paths = list(item["meta"]["target_trajectory_paths"])
    if item["meta"].get("split") != "valid" or not paths:
        raise ValueError("validation case lacks valid provenance")
    if any("2022" not in Path(path).name or "2023" in Path(path).name for path in paths):
        raise ValueError("case escaped frozen validation-2022")
    return {
        "dataset_index": index,
        "case_id": str(item["meta"]["case_id"]),
        "archive_slice_index": int(item["meta"]["archive_slice_index"]),
        "target_paths": paths,
    }


def _noise(case_order: int, member: int, image_size: tuple[int, int]) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(314159 + 1000003 * case_order + 1009 * member)
    return torch.randn((1, DIRECT_OUTPUT_CHANNELS, *image_size), generator=generator)


def _field_stats_tensor(values: Any) -> torch.Tensor:
    result = torch.as_tensor(_repeat_field_stats(values), dtype=torch.float32)
    if result.shape != (DIRECT_OUTPUT_CHANNELS,) or not torch.isfinite(result).all():
        raise ValueError("direct field statistics must be six finite values")
    return result


@torch.no_grad()
def _public_sample(sampler, item: dict[str, Any], noise: torch.Tensor, image_size: tuple[int, int], device: torch.device) -> torch.Tensor:
    condition = item["structured_conditioning"].unsqueeze(0).to(device)
    valid = item["valid_mask"].unsqueeze(0)[:, :1].to(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        return sampler.sample_conditioned(
            background=item["background"].unsqueeze(0).to(device),
            background_mask=torch.ones_like(item["background"].unsqueeze(0)).to(device),
            obs_values=item["obs_values"].unsqueeze(0).to(device),
            obs_mask=item["obs_mask"].unsqueeze(0).to(device),
            water_mask=item["water_mask"].unsqueeze(0).to(device),
            size=image_size,
            num_timesteps=17,
            device=device,
            method="rk4",
            rtol=1e-5,
            atol=1e-6,
            start_mode="noise",
            initial_noise=noise.to(device),
            sample_target="state",
            model_conditioning=condition,
            state_channels=DIRECT_OUTPUT_CHANNELS,
            end_time=0.0,
            state_mask=valid.expand(-1, DIRECT_OUTPUT_CHANNELS, -1, -1),
        ).float()


@torch.no_grad()
def _sample_all(
    sampler,
    candidates: dict[str, torch.nn.Module],
    cases: list[tuple[int, dict[str, Any]]],
    image_size: tuple[int, int],
    members: int,
    device: torch.device,
    fixed_noise: torch.Tensor,
    evidence_root: Path,
    identities: list[dict[str, Any]],
    status_path: Path,
    clearml_task_id: str,
    bindings_path: Path,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], float]:
    source = sampler.model.eval()
    grid = make_normalized_xy_grid(*image_size, device=device, dtype=torch.float32)
    ensembles: dict[str, list[torch.Tensor]] = {label: [] for label in LABELS}
    truth, persistence, valid, initial_sic, conditions, noises = [], [], [], [], [], []
    replay_max_abs = 0.0
    case_manifest: list[dict[str, Any]] = []
    for case_order, (_, item) in enumerate(cases):
        print(f"[geometry-paired-eval] sampling case {case_order + 1}/{len(cases)}", flush=True)
        condition = item["structured_conditioning"].unsqueeze(0).to(device, torch.float32)
        ocean = item["valid_mask"].unsqueeze(0)[:, :1].to(device, torch.float32)
        branch_members: dict[str, list[torch.Tensor]] = {label: [] for label in LABELS}
        case_noises = []
        for member in range(members):
            raw_noise = fixed_noise[case_order, member : member + 1]
            public = _public_sample(sampler, item, raw_noise, image_size, device)
            prefix = frozen_prefix(source, raw_noise.to(device), condition, ocean, grid)
            if case_order == 0 and member == 0:
                hybrid_source = trainable_suffix(source, prefix, condition, ocean, grid)
                step0_path = evidence_root / "step0_public_hybrid_parity.pt"
                _atomic_torch_save(
                    {
                        "identity": identities[0],
                        "member": 0,
                        "initial_noise": raw_noise.cpu(),
                        "condition": condition.cpu(),
                        "valid": ocean.cpu(),
                        "prefix12": prefix.cpu(),
                        "public_ema6": public.cpu(),
                        "hybrid_ema6": hybrid_source.cpu(),
                    },
                    step0_path,
                )
                step0_binding = _record_evidence_binding(
                    bindings_path, "step0_public_hybrid_parity", step0_path
                )
                _merge_status(
                    status_path,
                    status="sampling_parity_check",
                    completed_cases=0,
                    step0_parity_sha256=step0_binding["sha256"],
                    evidence_bindings_sha256=_sha256(bindings_path),
                    clearml_task_id=clearml_task_id,
                    test_2023_accessed=False,
                )
                replay_max_abs = float((hybrid_source - public).abs().max().cpu())
                if replay_max_abs != 0.0:
                    raise RuntimeError(f"public EMA6 versus unmodified hybrid mismatch: {replay_max_abs}")
            branch_members["ema6"].append(public[0].cpu())
            for label in ("control256", "treatment256"):
                branch_members[label].append(
                    trainable_suffix(candidates[label], prefix, condition, ocean, grid)[0].cpu()
                )
            case_noises.append(raw_noise[0])
        case_payload = {
            "identity": identities[case_order],
            "normalized": {label: torch.stack(branch_members[label]) for label in LABELS},
            "truth_standardized": item["truth"].float(),
            "persistence_physical": item["structured_physical_background"].float(),
            "valid": item["valid_mask"][:1].float(),
            "initial_sic": item["structured_physical_background"][0:1].float(),
            "condition": item["structured_conditioning"].float(),
            "initial_noise": torch.stack(case_noises),
        }
        case_path = evidence_root / f"completed_case_{case_order:02d}.pt"
        _atomic_torch_save(case_payload, case_path)
        case_manifest.append(
            {
                "case_order": case_order,
                "identity": identities[case_order],
                "path": str(case_path),
                "sha256": _sha256(case_path),
            }
        )
        _atomic_json(evidence_root / "completed_cases_manifest.json", {"completed_cases": case_manifest})
        manifest_binding = _record_evidence_binding(
            bindings_path, "completed_cases_manifest", evidence_root / "completed_cases_manifest.json"
        )
        _atomic_json(
            status_path,
            {
                "status": "sampling",
                "completed_cases": case_order + 1,
                "latest_case_sha256": case_manifest[-1]["sha256"],
                "completed_cases_manifest_sha256": manifest_binding["sha256"],
                "evidence_bindings_sha256": _sha256(bindings_path),
                "clearml_task_id": clearml_task_id,
                "test_2023_accessed": False,
            },
        )
        for label in LABELS:
            ensembles[label].append(torch.stack(branch_members[label]))
        truth.append(item["truth"].float())
        persistence.append(item["structured_physical_background"].float())
        valid.append(item["valid_mask"][:1].float())
        initial_sic.append(item["structured_physical_background"][0:1].float())
        conditions.append(item["structured_conditioning"].float())
        noises.append(torch.stack(case_noises))
    evidence = {
        "truth_standardized": torch.stack(truth),
        "persistence_physical": torch.stack(persistence),
        "valid": torch.stack(valid),
        "initial_sic": torch.stack(initial_sic),
        "condition": torch.stack(conditions),
        "initial_noise": torch.stack(noises),
    }
    return {key: torch.stack(value) for key, value in ensembles.items()}, evidence, replay_max_abs


def _coverage(members: torch.Tensor, truth: torch.Tensor, valid: torch.Tensor) -> dict[str, Any]:
    less = (members < truth[:, None]).sum(dim=1)
    equal = (members == truth[:, None]).sum(dim=1)
    definitions = {"inner_2_7": (2, 6, 5.0 / 9.0), "outer_1_8": (1, 7, 7.0 / 9.0)}
    result = {}
    for name, (first_rank, last_rank, reference) in definitions.items():
        per_case = []
        for case in range(members.shape[0]):
            selected = valid[case] > 0
            probability = torch.zeros_like(truth[case], dtype=torch.float64)
            for rank in range(first_rank, last_rank + 1):
                possible = (rank >= less[case]) & (rank <= less[case] + equal[case])
                probability += possible.to(torch.float64) / (equal[case].to(torch.float64) + 1.0)
            per_case.append(float(probability[selected].mean()))
        result[name] = {
            "coverage": float(np.mean(per_case)),
            "finite_m_reference": reference,
            "absolute_error": abs(float(np.mean(per_case)) - reference),
            "case_values": per_case,
        }
    return result


def _support_diagnostics(members: torch.Tensor, valid: torch.Tensor, field: str) -> dict[str, float]:
    selected = members[valid[:, None].expand_as(members) > 0].to(torch.float64)
    if selected.numel() == 0 or not torch.isfinite(selected).all():
        raise FloatingPointError(f"{field} has empty or non-finite valid support")
    if field == "sic":
        excess = torch.relu(-selected) + torch.relu(selected - 1.0)
    elif field == "sit":
        excess = torch.relu(-selected)
    else:
        raise ValueError(field)
    positive = excess[excess > 0]
    return {
        "violation_fraction": float((excess > 0).double().mean()),
        "mean_excess_all": float(excess.mean()),
        "p95_positive_excess": float(torch.quantile(positive, 0.95)) if positive.numel() else 0.0,
        "max_excess": float(positive.max()) if positive.numel() else 0.0,
    }


def _score_branch(
    normalized: torch.Tensor,
    physical: torch.Tensor,
    evidence: dict[str, torch.Tensor],
    means: torch.Tensor,
    stds: torch.Tensor,
    cell_area: torch.Tensor,
    product_scale: float,
) -> dict[str, Any]:
    truth_n = evidence["truth_standardized"]
    valid = evidence["valid"]
    means6 = means.view(1, 6, 1, 1)
    stds6 = stds.view(1, 6, 1, 1)
    truth_p = truth_n * stds6 + means6
    outputs: dict[str, Any] = {}
    for channel, key in enumerate(OUTPUT_KEYS):
        members_n = normalized[:, :, channel : channel + 1]
        target_n = truth_n[:, channel : channel + 1]
        members_p = physical[:, :, channel : channel + 1]
        target_p = truth_p[:, channel : channel + 1]
        persistence_p = evidence["persistence_physical"][:, channel : channel + 1]
        persistence_n = (persistence_p - means6[:, channel : channel + 1]) / stds6[:, channel : channel + 1]
        crps_n, crps_n_cases = _weighted_case_fair_crps(members_n, target_n, valid)
        crps_p, crps_p_cases = _weighted_case_fair_crps(members_p, target_p, valid)
        rmse_n, rmse_n_cases = _weighted_case_rmse(members_n.mean(1), target_n, valid)
        rmse_p, rmse_p_cases = _weighted_case_rmse(members_p.mean(1), target_p, valid)
        spread, spread_cases = _weighted_case_rms(members_n.std(1, unbiased=True), valid)
        persistence_rmse_n, persistence_rmse_n_cases = _weighted_case_rmse(persistence_n, target_n, valid)
        persistence_rmse_p, persistence_rmse_p_cases = _weighted_case_rmse(persistence_p, target_p, valid)
        point_members_n = persistence_n[:, None].expand(-1, normalized.shape[1], -1, -1, -1)
        persistence_crps_n, persistence_crps_n_cases = _weighted_case_fair_crps(point_members_n, target_n, valid)
        raw_ssr = None if rmse_n <= 0 else spread / rmse_n
        coverage = _coverage(members_n, target_n, valid)
        outputs[key] = {
            "standardized_fair_crps": crps_n,
            "standardized_fair_crps_case_values": crps_n_cases,
            "physical_fair_crps": crps_p,
            "physical_fair_crps_case_values": crps_p_cases,
            "standardized_rmse": rmse_n,
            "standardized_rmse_case_values": rmse_n_cases,
            "physical_rmse": rmse_p,
            "physical_rmse_case_values": rmse_p_cases,
            "persistence_standardized_rmse": persistence_rmse_n,
            "persistence_standardized_rmse_case_values": persistence_rmse_n_cases,
            "persistence_physical_rmse": persistence_rmse_p,
            "persistence_physical_rmse_case_values": persistence_rmse_p_cases,
            "persistence_standardized_point_mass_crps": persistence_crps_n,
            "persistence_standardized_point_mass_crps_case_values": persistence_crps_n_cases,
            "spread": spread,
            "spread_case_values": spread_cases,
            "raw_spread_skill_ratio": raw_ssr,
            "adjusted_spread_skill_ratio": None if raw_ssr is None else math.sqrt(9.0 / 8.0) * raw_ssr,
            "coverage": coverage,
            "roughness": _roughness(members_p, valid),
            "raw_support": _support_diagnostics(members_p, valid, key.rsplit("_", 1)[1]),
            **_weighted_fractional_rank(members_n, target_n, valid),
        }
    native_es, native_es_cases = _joint_energy_score(normalized, truth_n, valid)
    member_vectors, truth_vectors, slices = geometry_observable_vectors(
        physical,
        truth_p,
        valid,
        evidence["initial_sic"],
        cell_area,
        sic_scale=float(stds[0]),
        sit_scale=float(stds[1]),
        product_scale=product_scale,
    )
    geometry_cases = []
    for case in range(normalized.shape[0]):
        geometry_cases.append(float(unbiased_energy_score_vectors(member_vectors[case : case + 1], truth_vectors[case : case + 1])))
    geometry_groups = {}
    for name, (start, stop) in slices.items():
        values = []
        for case in range(normalized.shape[0]):
            values.append(float(unbiased_energy_score_vectors(member_vectors[case : case + 1, :, start:stop], truth_vectors[case : case + 1, start:stop])))
        geometry_groups[name] = {"mean": float(np.mean(values)), "case_values": values}
    return {
        "outputs": outputs,
        "primary_standardized_fair_crps": float(np.mean([outputs[key]["standardized_fair_crps"] for key in OUTPUT_KEYS])),
        "primary_standardized_fair_crps_case_values": np.mean(np.asarray([outputs[key]["standardized_fair_crps_case_values"] for key in OUTPUT_KEYS]), axis=0).tolist(),
        "native_joint_energy": native_es,
        "native_joint_energy_case_values": native_es_cases,
        "geometry_energy": float(np.mean(geometry_cases)),
        "geometry_energy_case_values": geometry_cases,
        "geometry_groups": geometry_groups,
    }


def _paired_bootstrap(left: list[float], right: list[float], indices: np.ndarray) -> dict[str, Any]:
    left_array, right_array = np.asarray(left), np.asarray(right)
    differences = (left_array - right_array)[indices].mean(axis=1)
    denominator = right_array[indices].mean(axis=1)
    ratios = None if np.any(denominator <= 0) or right_array.mean() <= 0 else left_array[indices].mean(axis=1) / denominator
    return {
        "difference": float(left_array.mean() - right_array.mean()),
        "difference_ci95": np.quantile(differences, [0.025, 0.975]).tolist(),
        "ratio": None if ratios is None else float(left_array.mean() / right_array.mean()),
        "ratio_ci95": None if ratios is None else np.quantile(ratios, [0.025, 0.975]).tolist(),
    }


def _paired_bootstrap_rmse(left: list[float], right: list[float], indices: np.ndarray) -> dict[str, Any]:
    left_mse, right_mse = np.square(np.asarray(left)), np.square(np.asarray(right))
    left_values = np.sqrt(left_mse[indices].mean(axis=1))
    right_values = np.sqrt(right_mse[indices].mean(axis=1))
    differences = left_values - right_values
    ratios = None if np.any(right_values <= 0) or right_mse.mean() <= 0 else left_values / right_values
    return {
        "difference": float(math.sqrt(left_mse.mean()) - math.sqrt(right_mse.mean())),
        "difference_ci95": np.quantile(differences, [0.025, 0.975]).tolist(),
        "ratio": None if ratios is None else float(math.sqrt(left_mse.mean() / right_mse.mean())),
        "ratio_ci95": None if ratios is None else np.quantile(ratios, [0.025, 0.975]).tolist(),
    }


def _comparisons(scores: dict[str, Any], replicates: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, 12, size=(replicates, 12))
    result = {}
    for left, right in (("treatment256", "control256"), ("control256", "ema6"), ("treatment256", "ema6")):
        pair = {}
        for metric in ("primary_standardized_fair_crps", "native_joint_energy", "geometry_energy"):
            pair[metric] = _paired_bootstrap(scores[left][f"{metric}_case_values"], scores[right][f"{metric}_case_values"], indices)
        pair["outputs"] = {}
        for key in OUTPUT_KEYS:
            pair["outputs"][key] = {}
            for metric in ("standardized_fair_crps", "standardized_rmse"):
                bootstrap = _paired_bootstrap_rmse if metric == "standardized_rmse" else _paired_bootstrap
                pair["outputs"][key][metric] = bootstrap(
                    scores[left]["outputs"][key][f"{metric}_case_values"],
                    scores[right]["outputs"][key][f"{metric}_case_values"],
                    indices,
                )
        result[f"{left}_vs_{right}"] = pair
    return result


def _gate(scores: dict[str, Any], comparisons: dict[str, Any]) -> dict[str, Any]:
    tc = comparisons["treatment256_vs_control256"]
    te = comparisons["treatment256_vs_ema6"]
    tc_ratio_ci = tc["primary_standardized_fair_crps"]["ratio_ci95"]
    geometry_effect = tc_ratio_ci is not None and tc["geometry_energy"]["difference_ci95"][1] < 0 and tc_ratio_ci[1] <= 1.01
    replacement_checks: dict[str, bool] = {
        "treatment_primary_crps_better_than_ema6": te["primary_standardized_fair_crps"]["difference_ci95"][1] < 0,
    }
    for reference in ("control256", "ema6"):
        comparison = comparisons[f"treatment256_vs_{reference}"]
        for key in OUTPUT_KEYS:
            trial = scores["treatment256"]["outputs"][key]
            base = scores[reference]["outputs"][key]
            crps_ratio = comparison["outputs"][key]["standardized_fair_crps"]["ratio"]
            rmse_ratio = comparison["outputs"][key]["standardized_rmse"]["ratio"]
            replacement_checks[f"{reference}_{key}_crps"] = crps_ratio is not None and crps_ratio <= 1.01
            replacement_checks[f"{reference}_{key}_rmse"] = rmse_ratio is not None and rmse_ratio <= 1.01
            replacement_checks[f"{reference}_{key}_rank_tv"] = trial["rank_tv_to_uniform"] - base["rank_tv_to_uniform"] <= 0.01
            coverage_delta = max(
                trial["coverage"][name]["absolute_error"] - base["coverage"][name]["absolute_error"]
                for name in trial["coverage"]
            )
            replacement_checks[f"{reference}_{key}_coverage"] = coverage_delta <= 0.01
            trial_ssr, base_ssr = trial["adjusted_spread_skill_ratio"], base["adjusted_spread_skill_ratio"]
            replacement_checks[f"{reference}_{key}_adjusted_ssr"] = trial_ssr is not None and base_ssr is not None and abs(trial_ssr - 1.0) - abs(base_ssr - 1.0) <= 0.02
            replacement_checks[f"{reference}_{key}_roughness"] = base["roughness"] > 0 and trial["roughness"] / base["roughness"] <= 1.02
        base_energy = scores[reference]["native_joint_energy"]
        replacement_checks[f"{reference}_native_energy"] = base_energy > 0 and scores["treatment256"]["native_joint_energy"] / base_energy <= 1.01
    statistical = geometry_effect and all(replacement_checks.values())
    return {
        "geometry_effect_pass": geometry_effect,
        "replacement_checks": replacement_checks,
        "statistical_replacement_pass": statistical,
        "replacement_status": "pending_visual_review" if statistical else "hold",
        "claim_boundary": "endpoint-only continuation without FM anchor; development reuse, not independent confirmation",
    }


def _save_visuals(output: Path, label: str, ensemble: torch.Tensor, truth: torch.Tensor, persistence: torch.Tensor, valid: torch.Tensor, case_orders: list[int]) -> list[str]:
    paths = []
    root = output / "individual_members" / label
    root.mkdir(parents=True)
    for case_order in case_orders:
        for member in range(ensemble.shape[1]):
            figure = make_structured_trajectory_figure(
                truth[case_order], persistence[case_order], ensemble[case_order, member], valid[case_order],
                title=f"{label}: case order {case_order}, member {member}", origin="upper", lead_days=DIRECT_LEADS,
            )
            # The shared helper auto-scales SIT per figure.  Override every SIT
            # panel to the frozen 0--4 m display range so apparent texture is
            # comparable across branches/members.  Raw support remains scored.
            main_axes = figure.axes[: 3 * 6]
            for row in range(3):
                for column in range(3, 6):
                    for image in main_axes[row * 6 + column].images:
                        image.set_clim(0.0, 4.0)
            path = root / f"case_{case_order:02d}_member_{member:02d}.png"
            figure.savefig(path, dpi=180)
            import matplotlib.pyplot as plt
            plt.close(figure)
            paths.append(str(path))
    return paths


def run(config_path: Path, output: Path) -> dict[str, Any]:
    experiment = load_json(config_path)
    protocol = _check_contract(experiment)
    if torch.cuda.device_count() != 1 or os.environ.get("CLEARML_REQUIRE_ONLINE") != "1":
        raise RuntimeError("paired validation requires exactly one visible GPU and online ClearML")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to reuse output {output}")
    output.mkdir(parents=True)
    status_path = output / "status.json"
    _atomic_json(status_path, {"status": "initializing", "test_2023_accessed": False})
    tracker = None
    prior_handlers: dict[int, Any] = {}

    def terminate(signum, _frame):
        raise InterruptedError(f"received termination signal {signum}")

    for signum in (signal.SIGTERM, signal.SIGINT):
        prior_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, terminate)
    try:
        source = experiment["source"]
        run_dir = Path(source["run_dir"])
        for relative, expected in source["sha256"].items():
            if _sha256(run_dir / relative) != expected:
                raise ValueError(f"source SHA mismatch: {relative}")
        completion = experiment["training_completion"]
        completion_payloads = {}
        for name, spec in completion.items():
            path = Path(spec["path"])
            if _sha256(path) != spec["sha256"]:
                raise ValueError(f"training completion SHA mismatch: {name}")
            completion_payloads[name] = json.loads(path.read_text())
        if completion_payloads["status"].get("status") != "training_complete_pending_paired_validation":
            raise ValueError("training status is not complete")
        if completion_payloads["launch"].get("status") != "complete" or completion_payloads["launch"].get("exit_code") != 0:
            raise ValueError("training launcher did not finish cleanly")
        if completion_payloads["launch"].get("code_commit") != experiment["training_commit"]:
            raise ValueError("training launcher commit mismatch")
        training_result_path = Path(experiment["training_result"]["path"])
        if _sha256(training_result_path) != experiment["training_result"]["sha256"]:
            raise ValueError("training result SHA mismatch")
        training_result = json.loads(training_result_path.read_text())
        checkpoint_payloads = {
            arm: _verify_final_checkpoint(Path(spec["path"]), spec["sha256"], arm)
            for arm, spec in experiment["checkpoints"].items()
        }
        _verify_matched_training(experiment, training_result, checkpoint_payloads)
        metadata = json.loads((run_dir / "metadata.json").read_text())
        config = metadata["training_config"]
        model_config = TrainingConfig.from_dict(config)
        if tuple(model_config.image_size) != (320, 256) or model_config.out_channels != 6:
            raise ValueError("unexpected EMA6 model geometry")
        optimizer_audit_model = build_unet(model_config).eval()
        for arm in ("control", "treatment"):
            optimizer_audit_model.load_state_dict(checkpoint_payloads[arm]["model"], strict=True)
            _verify_optimizer_against_model(checkpoint_payloads[arm], optimizer_audit_model, arm)
        del optimizer_audit_model
        dataset = build_dataset(metadata["data_config"], split="valid")
        if len(dataset) != 8544 or dataset.split != "valid":
            raise ValueError("validation-2022 dataset identity mismatch")
        cache = {index: dataset[index] for index in sorted(set(protocol["case_indices"]) | {0, 12, 23})}
        sentinel = validate_direct_dataset(dataset, item_cache=cache)
        cases = [(index, cache[index]) for index in protocol["case_indices"]]
        identities = [_case_identity(item, index) for index, item in cases]
        if identities != experiment["case_identities"]:
            raise ValueError("frozen validation case identity/slice/path mismatch")
        image_size = tuple(model_config.image_size)
        fixed_noise = torch.stack(
            [torch.cat([_noise(case, member, image_size) for member in range(protocol["members"])]) for case in range(len(cases))]
        )
        fixed_inputs_path = output / "fixed_inputs_before_ode.pt"
        _atomic_torch_save(
            {
                "case_identities": identities,
                "initial_noise": fixed_noise,
                "condition": torch.stack([item["structured_conditioning"].float() for _, item in cases]),
                "truth_standardized": torch.stack([item["truth"].float() for _, item in cases]),
                "persistence_physical": torch.stack([item["structured_physical_background"].float() for _, item in cases]),
                "valid": torch.stack([item["valid_mask"][:1].float() for _, item in cases]),
            },
            fixed_inputs_path,
        )
        fixed_inputs_sha = _sha256(fixed_inputs_path)
        run_contract_path = output / "run_contract_before_gpu.json"
        _atomic_json(
            run_contract_path,
            {
                "schema_version": SCHEMA_VERSION,
                "protocol": protocol,
                "gate": experiment["gate"],
                "case_identities": identities,
                "source_sha256": source["sha256"],
                "training_result_sha256": experiment["training_result"]["sha256"],
                "checkpoint_sha256": {
                    arm: spec["sha256"] for arm, spec in experiment["checkpoints"].items()
                },
                "evaluation_code_sha256": _sha256(Path(__file__)),
                "config_sha256": _sha256(config_path),
                "suffix_sha256": _sha256(Path(__file__).with_name("direct_dynamics_suffix.py")),
                "geometry_score_sha256": _sha256(Path(__file__).with_name("direct_dynamics_geometry_score.py")),
                "fixed_inputs_sha256": fixed_inputs_sha,
                "normalization_means": list(_repeat_field_stats(dataset.means)),
                "normalization_stds": list(_repeat_field_stats(dataset.stds)),
                "checkpoint_model_optimizer_audit": "passed_exact_step256_full_coverage",
                "optimizer_steps": 0,
                "test_2023_accessed": False,
            },
        )
        run_contract_sha = _sha256(run_contract_path)
        bindings_path = output / "evidence_bindings.json"
        _record_evidence_binding(bindings_path, "run_contract_before_gpu", run_contract_path)
        _record_evidence_binding(bindings_path, "fixed_inputs_before_ode", fixed_inputs_path)
        _merge_status(
            status_path,
            status="inputs_bound_before_gpu",
            run_contract_sha256=run_contract_sha,
            fixed_inputs_sha256=fixed_inputs_sha,
            evidence_bindings_sha256=_sha256(bindings_path),
            test_2023_accessed=False,
        )
        tracker = ClearMLTracker(
            experiment["project_name"], f"{experiment['task_name']}-{output.name}",
            tags=experiment["clearml"]["tags"], env_path=experiment["clearml"]["env_path"],
        )
        tracker.connect("paired_validation_contract", {"protocol": protocol, "gate": experiment["gate"], "case_identities": identities})
        _merge_status(status_path, status="sampling", completed_cases=0, clearml_task_id=str(tracker.task.id), test_2023_accessed=False)
        device = torch.device("cuda:0")
        sampler = load_sampler(str(run_dir), source["checkpoint"], config, device=device)
        candidates = {}
        for arm, label in (("control", "control256"), ("treatment", "treatment256")):
            model = copy.deepcopy(sampler.model).eval()
            model.load_state_dict(checkpoint_payloads[arm]["model"], strict=True)
            candidates[label] = model.to(device)
        evidence_root = output / "sampling_evidence"
        evidence_root.mkdir()
        normalized, evidence, replay_max_abs = _sample_all(
            sampler, candidates, cases, image_size, protocol["members"], device,
            fixed_noise, evidence_root, identities, status_path, str(tracker.task.id), bindings_path,
        )
        means = _field_stats_tensor(dataset.means)
        stds = _field_stats_tensor(dataset.stds)
        physical = {label: channel_denormalize(values.float(), means, stds) for label, values in normalized.items()}
        tensor_path = output / "complete_paired_ensembles.pt"
        _atomic_torch_save({"normalized": normalized, "physical": physical, "evidence": evidence, "case_identities": identities}, tensor_path)
        tensor_sha = _sha256(tensor_path)
        complete_binding = _record_evidence_binding(bindings_path, "complete_paired_ensembles", tensor_path)
        _merge_status(status_path, status="scoring", completed_cases=12, complete_tensor_sha256=complete_binding["sha256"], evidence_bindings_sha256=_sha256(bindings_path), clearml_task_id=str(tracker.task.id), test_2023_accessed=False)
        mask = torch.from_numpy(~np.load(metadata["data_config"]["mask_path"])).unsqueeze(0)
        cell_area, area_evidence = _cell_area_proxy(
            Path(experiment["cell_geometry"]["latitude_path"]), Path(experiment["cell_geometry"]["longitude_path"]),
            mask, image_size, torch.device("cpu"),
        )
        scores = {
            label: _score_branch(normalized[label], physical[label], evidence, means, stds, cell_area, experiment["product_scale"])
            for label in LABELS
        }
        comparisons = _comparisons(scores, protocol["bootstrap_replicates"], protocol["bootstrap_seed"])
        result: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION, "status": "scored_pending_visual_review",
            "split": "validation-2022", "development_reuse": True, "optimizer_steps": 0,
            "test_2023_accessed": False, "protocol": protocol, "case_identities": identities,
            "source_sha256": source["sha256"],
            "checkpoint_sha256": {arm: spec["sha256"] for arm, spec in experiment["checkpoints"].items()},
            "training_result_sha256": experiment["training_result"]["sha256"],
            "training_commit": experiment["training_commit"],
            "evaluation_code_sha256": _sha256(Path(__file__)), "config_sha256": _sha256(config_path),
            "suffix_sha256": _sha256(Path(__file__).with_name("direct_dynamics_suffix.py")),
            "geometry_score_sha256": _sha256(Path(__file__).with_name("direct_dynamics_geometry_score.py")),
            "fixed_inputs": {"path": str(fixed_inputs_path), "sha256": fixed_inputs_sha},
            "run_contract": {"path": str(run_contract_path), "sha256": run_contract_sha},
            "evidence_bindings": {"path": str(bindings_path), "sha256": _sha256(bindings_path)},
            "complete_tensor": {"path": str(tensor_path), "sha256": tensor_sha},
            "sampling_evidence_manifest_sha256": _sha256(evidence_root / "completed_cases_manifest.json"),
            "step0_parity_tensor_sha256": _sha256(evidence_root / "step0_public_hybrid_parity.pt"),
            "public_hybrid_replay_max_abs": replay_max_abs, "dataset_sentinel": sentinel,
            "cell_area": area_evidence, "scores": scores, "comparisons": comparisons,
            "clearml_task_id": str(tracker.task.id),
        }
        result["gate"] = _gate(scores, comparisons)
        _finite_tree(result)
        numerical_path = output / "paired_numerical_result.json"
        _atomic_json(numerical_path, result)
        rank_root = output / "rank_histograms"
        rank_root.mkdir()
        truth_physical = evidence["truth_standardized"] * stds.view(1, 6, 1, 1) + means.view(1, 6, 1, 1)
        for label in LABELS:
            rank_path = rank_root / f"{label}.png"
            _save_rank_histograms(rank_path, label, scores[label])
            tracker.report_image("rank_histograms", label, rank_path, 0)
            for path in _save_visuals(output, label, physical[label], truth_physical, evidence["persistence_physical"], evidence["valid"], protocol["visual_case_orders"]):
                tracker.report_image("individual_members", f"{label}/{Path(path).stem}", path, 0)
        _atomic_json(output / "paired_evaluation.json", result)
        tracker.upload_artifact("paired_evaluation", output / "paired_evaluation.json")
        tracker.upload_artifact("paired_numerical_result", numerical_path)
        tracker.upload_artifact("complete_paired_ensembles", tensor_path)
        _close_and_mark_complete(status_path, tracker, result["clearml_task_id"])
        tracker = None
        return result
    except BaseException as error:
        _record_terminal_failure(status_path, tracker, error)
        raise
    finally:
        for signum, handler in prior_handlers.items():
            signal.signal(signum, handler)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve(), args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
