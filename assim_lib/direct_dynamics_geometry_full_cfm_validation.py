"""Frozen three-law validation of full-CFM continuations from EMA6.

This evaluator is development-only (validation 2022).  It compares the exact
public EMA6 sampler with raw full-model control512 and treatment512 using
common cases and common initial noise.  Sampling artifacts are made durable
before any score or plot is computed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from .clearml_tracking import ClearMLTracker
from .config import TrainingConfig, load_json
from .data import build_dataset
from .direct_dynamics_geometry_paired_evaluation import (
    OUTPUT_KEYS,
    _case_identity,
    _cell_area_proxy,
    _close_and_mark_complete,
    _comparisons as _legacy_comparisons,
    _field_stats_tensor,
    _finite_tree,
    _merge_status,
    _noise,
    _public_sample,
    _record_evidence_binding,
    _record_terminal_failure,
    _save_visuals,
    _score_branch as _base_score_branch,
)
from .direct_dynamics_geometry_cfm_training import training_contract_sha256
from .direct_dynamics_geometry_cfm_transactional_training import (
    OBJECTIVE,
    SCIENTIFIC_RUNNER,
)
from .direct_dynamics_training import DIRECT_LEADS, validate_direct_dataset
from .model_io import build_unet, load_sampler
from .trainer import _atomic_json
from .transforms import channel_denormalize


SCHEMA_VERSION = "direct_dynamics_geometry_full_cfm_paired_validation_v1"
LABELS = ("ema6", "control512", "treatment512")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_torch_save(payload: Any, path: Path) -> str:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.incomplete")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return _sha256(path)


def _check_contract(experiment: dict[str, Any]) -> dict[str, Any]:
    if experiment.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unreviewed full-CFM validation schema")
    protocol = experiment["protocol"]
    expected_protocol = {
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
        "bootstrap_unit": "whole_date",
        "bootstrap_replicates": 100000,
        "bootstrap_seed": 20260912,
        "visual_case_orders": [0, 3, 6, 9],
        "test_2023_accessed": False,
    }
    if protocol != expected_protocol:
        mismatch = {
            key: (protocol.get(key), value)
            for key, value in expected_protocol.items()
            if protocol.get(key) != value
        }
        raise ValueError(f"unreviewed full-CFM validation protocol: {mismatch}")
    if [entry.get("label") for entry in experiment["laws"]] != list(LABELS):
        raise ValueError("validation laws must be frozen as EMA6/control512/treatment512")
    if [entry.get("kind") for entry in experiment["laws"]] != [
        "source_public_sampler",
        "raw_full_model_state",
        "raw_full_model_state",
    ]:
        raise ValueError("validation law kinds differ from full public samplers")
    expected_gate = {
        "geometry_effect": "treatment512-control512 geometry-ES paired date-bootstrap upper95 < 0 and primary CRPS ratio upper95 <= 1.01",
        "replacement": "treatment512-ema6 primary CRPS paired date-bootstrap upper95 < 0 plus all frozen safety margins against ema6 and control512",
        "per_output_crps_rmse_ratio_max": 1.01,
        "native_energy_ratio_max": 1.01,
        "rank_tv_increase_max": 0.01,
        "coverage_absolute_error_increase_max": 0.01,
        "adjusted_ssr_absolute_error_increase_max": 0.02,
        "roughness_ratio_max": 1.02,
        "manual_visual_review_required": True,
    }
    if experiment.get("gate") != expected_gate:
        raise ValueError("validation gate differs from the frozen gate")
    return protocol


def _verify_json_binding(spec: dict[str, str], label: str) -> dict[str, Any]:
    path = Path(spec["path"])
    if _sha256(path) != spec["sha256"]:
        raise ValueError(f"{label} SHA mismatch")
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _verify_optimizer_state(
    payload: dict[str, Any], model: torch.nn.Module, arm: str, expected_step: int
) -> None:
    if payload.get("arm") != arm or payload.get("completed_update") != expected_step:
        raise ValueError(f"{arm} optimizer checkpoint lifecycle mismatch")
    optimizer = payload.get("optimizer", {})
    groups, states = optimizer.get("param_groups"), optimizer.get("state")
    if not isinstance(groups, list) or len(groups) != 1 or not isinstance(states, dict):
        raise ValueError(f"{arm} optimizer structure mismatch")
    group = groups[0]
    if group.get("lr") != 1e-5 or group.get("weight_decay") != 0.0:
        raise ValueError(f"{arm} optimizer hyperparameters changed")
    identifiers = group.get("params")
    if (
        not isinstance(identifiers, list)
        or len(identifiers) != len(set(identifiers))
        or set(identifiers) != set(states)
    ):
        raise ValueError(f"{arm} optimizer state coverage is incomplete")
    parameters = list(model.parameters())
    if len(parameters) != len(identifiers):
        raise ValueError(f"{arm} optimizer/model parameter counts differ")
    for identifier, parameter in zip(identifiers, parameters, strict=True):
        state = states[identifier]
        if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError(f"{arm} optimizer state keys changed")
        step = state["step"]
        if not torch.is_tensor(step) or step.numel() != 1 or float(step) != expected_step:
            raise ValueError(f"{arm} optimizer step is not exactly {expected_step}")
        for name in ("exp_avg", "exp_avg_sq"):
            value = state[name]
            if value.shape != parameter.shape or not torch.isfinite(value).all():
                raise ValueError(f"{arm} optimizer {name} is malformed")


def _verify_training_and_models(
    experiment: dict[str, Any], model_config: TrainingConfig
) -> tuple[dict[str, Any], dict[str, dict[str, torch.Tensor]]]:
    evidence = experiment["training_evidence"]
    result = _verify_json_binding(evidence["training_result"], "training result")
    status = _verify_json_binding(evidence["status"], "training status")
    schedule = _verify_json_binding(evidence["schedule"], "training schedule")
    launch_status = _verify_json_binding(evidence["launch_status"], "training launch status")
    launch_exit = _verify_json_binding(evidence["launch_exit"], "training launch exit")
    if (
        result.get("status") != "completed_pending_paired_validation"
        or status.get("status") != "completed_pending_paired_validation"
        or result.get("committed_updates_by_arm") != {"control": 512, "treatment": 512}
        or status.get("committed_updates_by_arm") != {"control": 512, "treatment": 512}
        or result.get("test_2023_accessed") is not False
        or result.get("clearml_closed_before_terminal_success") is not True
        or result.get("resume_supported") is not False
    ):
        raise ValueError("training result is not the admitted terminal lifecycle")
    if (
        launch_status.get("status") != "complete"
        or launch_exit.get("controller_exit_code") != 0
        or launch_status.get("code_commit") != evidence["commit"]
        or launch_exit.get("code_commit") != evidence["commit"]
    ):
        raise ValueError("training launcher completion mismatch")
    if (
        schedule.get("sha256") != result.get("schedule_sha256")
        or len(schedule.get("indices", [])) != 4096
        or len(set(schedule.get("indices", []))) != 4096
        or result.get("scientific_runner_sha256") != _sha256(SCIENTIFIC_RUNNER)
        or result.get("objective_sha256") != _sha256(OBJECTIVE)
    ):
        raise ValueError("training schedule or scientific implementation binding mismatch")
    scientific_experiment = load_json(
        Path(__file__).parents[1]
        / "config/experiments/train_direct_dynamics_geometry_full_cfm_ab_v1.json"
    )
    if result.get("training_contract_sha256") != training_contract_sha256(
        scientific_experiment
    ):
        raise ValueError("training scientific-contract hash mismatch")

    checkpoint_manifest = result["checkpoints"]
    for arm in ("control", "treatment"):
        if set(checkpoint_manifest.get(arm, {})) != {"128", "256", "512"}:
            raise ValueError(f"{arm} checkpoint update set is incomplete")
        for update in (128, 256, 512):
            if set(checkpoint_manifest[arm][str(update)]) != {
                "model",
                "ema_model",
                "optimizer_recovery",
            }:
                raise ValueError(f"{arm} update {update} manifest is incomplete")

    candidate_states: dict[str, dict[str, torch.Tensor]] = {}
    audit_model = build_unet(model_config).eval()
    for law, arm in zip(experiment["laws"][1:], ("control", "treatment"), strict=True):
        path = Path(law["path"])
        if _sha256(path) != law["sha256"]:
            raise ValueError(f"{arm} raw model SHA mismatch")
        if law["sha256"] != checkpoint_manifest[arm]["512"]["model"]:
            raise ValueError(f"{arm} law does not select raw update512")
        state = torch.load(path, map_location="cpu", weights_only=True)
        audit_model.load_state_dict(state, strict=True)
        if any(not torch.isfinite(value).all() for value in state.values()):
            raise FloatingPointError(f"{arm} model state contains NaN/Inf")
        optimizer_path = path.with_name("optimizer_recovery.pth")
        if (
            _sha256(optimizer_path)
            != checkpoint_manifest[arm]["512"]["optimizer_recovery"]
        ):
            raise ValueError(f"{arm} loaded optimizer is not the exact update512 artifact")
        optimizer_payload = torch.load(optimizer_path, map_location="cpu", weights_only=True)
        _verify_optimizer_state(optimizer_payload, audit_model, arm, 512)
        candidate_states[f"{arm}512"] = state
    return result, candidate_states


def _fractional_rank(
    members: torch.Tensor, truth: torch.Tensor, selected: torch.Tensor
) -> dict[str, Any]:
    member_count = int(members.shape[1])
    case_masses, point_counts = [], []
    for case in range(members.shape[0]):
        mask = selected[case].bool()
        count = int(mask.sum())
        if count == 0:
            continue
        case_members = members[case, :, 0][:, mask[0]].to(torch.float64)
        case_truth = truth[case, 0][mask[0]].to(torch.float64)
        less = (case_members < case_truth[None]).sum(dim=0)
        equal = (case_members == case_truth[None]).sum(dim=0)
        mass = torch.zeros(member_count + 1, dtype=torch.float64)
        for rank in range(member_count + 1):
            possible = (rank >= less) & (rank <= less + equal)
            mass[rank] = (possible / (equal + 1).to(torch.float64)).sum()
        case_masses.append(mass / count)
        point_counts.append(count)
    if not case_masses:
        return {"case_count": 0, "point_count": 0, "frequencies": None, "rank_tv_to_uniform": None}
    probabilities = torch.stack(case_masses).mean(0)
    uniform = torch.full_like(probabilities, 1.0 / (member_count + 1))
    return {
        "case_count": len(case_masses),
        "point_count": int(sum(point_counts)),
        "frequencies": probabilities.tolist(),
        "rank_tv_to_uniform": float(0.5 * (probabilities - uniform).abs().sum()),
        "normalized_mean_rank": float(
            (probabilities * torch.arange(member_count + 1, dtype=torch.float64)).sum()
            / member_count
        ),
    }


def _binary_calibration(
    member_event: torch.Tensor,
    truth_event: torch.Tensor,
    valid: torch.Tensor,
    bins: int = 10,
) -> dict[str, Any]:
    probability = member_event.to(torch.float64).mean(dim=1)
    truth = truth_event.to(torch.float64)
    case_brier, bin_weight, bin_probability, bin_frequency = [], [], [], []
    for case in range(probability.shape[0]):
        mask = valid[case].bool()
        case_brier.append(float((probability[case][mask] - truth[case][mask]).square().mean()))
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        selected = (probability >= low) & (probability < high if index < bins - 1 else probability <= high) & valid.bool()
        count = int(selected.sum())
        bin_weight.append(count)
        bin_probability.append(None if count == 0 else float(probability[selected].mean()))
        bin_frequency.append(None if count == 0 else float(truth[selected].mean()))
    total = sum(bin_weight)
    ece = sum(
        count * abs(predicted - observed)
        for count, predicted, observed in zip(bin_weight, bin_probability, bin_frequency, strict=True)
        if count and predicted is not None and observed is not None
    ) / max(total, 1)
    return {
        "case_equal_brier": float(np.mean(case_brier)),
        "case_values": case_brier,
        "ece_10_bins": float(ece),
        "bin_counts": bin_weight,
        "bin_mean_probability": bin_probability,
        "bin_observed_frequency": bin_frequency,
    }


def _stratified_diagnostics(
    normalized: torch.Tensor,
    truth_standardized: torch.Tensor,
    truth_physical: torch.Tensor,
    valid: torch.Tensor,
    identities: list[dict[str, Any]],
    means: torch.Tensor,
    stds: torch.Tensor,
) -> dict[str, Any]:
    months = [int(identity["case_id"][5:7]) for identity in identities]
    seasons = {"DJF": {12, 1, 2}, "MAM": {3, 4, 5}, "JJA": {6, 7, 8}, "SON": {9, 10, 11}}
    result: dict[str, Any] = {"overall": {}, "seasonal": {}, "truth_support": {}, "binary": {}}
    for channel, key in enumerate(OUTPUT_KEYS):
        # Keep predictions and truth in the original encoded FP32 law.  A
        # physical decode/re-encode round trip can move exact ties and 0.01 /
        # 0.99 threshold events by one ULP.  Physical truth remains the sole
        # source of atom/support selectors.
        members = normalized[:, :, channel : channel + 1]
        truth = truth_standardized[:, channel : channel + 1]
        raw_truth = truth_physical[:, channel : channel + 1]
        field = key.rsplit("_", 1)[1]
        result["overall"][key] = _fractional_rank(members, truth, valid)
        result["seasonal"][key] = {}
        for season, season_months in seasons.items():
            case_mask = torch.tensor([month in season_months for month in months]).view(-1, 1, 1, 1)
            result["seasonal"][key][season] = _fractional_rank(
                members, truth, valid.bool() & case_mask
            )
        occurrence_threshold = (
            torch.tensor(0.01, dtype=torch.float32) - means[channel]
        ) / stds[channel]
        if field == "sic":
            groups = {
                "exact_zero": raw_truth == 0.0,
                "interior": (raw_truth > 0.0) & (raw_truth < 1.0),
                "exact_one": raw_truth == 1.0,
            }
            cap_threshold = (
                torch.tensor(0.99, dtype=torch.float32) - means[channel]
            ) / stds[channel]
            result["binary"][key] = {
                "open_water_le_0p01": _binary_calibration(
                    members <= occurrence_threshold, raw_truth <= 0.01, valid
                ),
                "sic_cap_ge_0p99": _binary_calibration(
                    members >= cap_threshold, raw_truth >= 0.99, valid
                ),
            }
        else:
            groups = {"exact_zero": raw_truth == 0.0, "positive": raw_truth > 0.0}
            result["binary"][key] = {
                "open_water_le_0p01": _binary_calibration(
                    members <= occurrence_threshold, raw_truth <= 0.01, valid
                )
            }
        result["truth_support"][key] = {
            name: _fractional_rank(members, truth, valid.bool() & selector)
            for name, selector in groups.items()
        }
    return result


def _comparisons(scores: dict[str, Any], replicates: int, seed: int) -> dict[str, Any]:
    # Reuse the already-audited date-bootstrap implementation by translating
    # only its legacy labels.
    translated = {
        "ema6": scores["ema6"],
        "control256": scores["control512"],
        "treatment256": scores["treatment512"],
    }
    legacy = _legacy_comparisons(translated, replicates, seed)
    return {
        key.replace("treatment256", "treatment512").replace("control256", "control512"): value
        for key, value in legacy.items()
    }


def _gate(scores: dict[str, Any], comparisons: dict[str, Any]) -> dict[str, Any]:
    treatment_control = comparisons["treatment512_vs_control512"]
    treatment_ema6 = comparisons["treatment512_vs_ema6"]
    primary_ratio_ci = treatment_control["primary_standardized_fair_crps"]["ratio_ci95"]
    geometry_effect = (
        primary_ratio_ci is not None
        and treatment_control["geometry_energy"]["difference_ci95"][1] < 0
        and primary_ratio_ci[1] <= 1.01
    )
    checks: dict[str, bool] = {
        "treatment_primary_better_than_ema6": treatment_ema6[
            "primary_standardized_fair_crps"
        ]["difference_ci95"][1]
        < 0
    }
    for reference in ("control512", "ema6"):
        comparison = comparisons[f"treatment512_vs_{reference}"]
        for key in OUTPUT_KEYS:
            trial = scores["treatment512"]["outputs"][key]
            base = scores[reference]["outputs"][key]
            crps_ratio = comparison["outputs"][key]["standardized_fair_crps"]["ratio"]
            rmse_ratio = comparison["outputs"][key]["standardized_rmse"]["ratio"]
            checks[f"{reference}_{key}_crps"] = crps_ratio is not None and crps_ratio <= 1.01
            checks[f"{reference}_{key}_rmse"] = rmse_ratio is not None and rmse_ratio <= 1.01
            checks[f"{reference}_{key}_rank_tv"] = (
                trial["rank_tv_to_uniform"] - base["rank_tv_to_uniform"] <= 0.01
            )
            coverage_delta = max(
                trial["coverage"][name]["absolute_error"]
                - base["coverage"][name]["absolute_error"]
                for name in trial["coverage"]
            )
            checks[f"{reference}_{key}_coverage"] = coverage_delta <= 0.01
            trial_ssr, base_ssr = (
                trial["adjusted_spread_skill_ratio"],
                base["adjusted_spread_skill_ratio"],
            )
            checks[f"{reference}_{key}_adjusted_ssr"] = (
                trial_ssr is not None
                and base_ssr is not None
                and abs(trial_ssr - 1.0) - abs(base_ssr - 1.0) <= 0.02
            )
            checks[f"{reference}_{key}_roughness"] = (
                base["roughness"] > 0
                and trial["roughness"] / base["roughness"] <= 1.02
            )
        checks[f"{reference}_native_energy"] = (
            scores[reference]["native_joint_energy"] > 0
            and scores["treatment512"]["native_joint_energy"]
            / scores[reference]["native_joint_energy"]
            <= 1.01
        )
    statistical = geometry_effect and all(checks.values())
    return {
        "geometry_effect_pass": geometry_effect,
        "replacement_checks": checks,
        "statistical_replacement_pass": statistical,
        "replacement_status": "pending_manual_visual_review" if statistical else "hold",
        "conditional_rank_uniformity_used_as_gate": False,
        "claim_boundary": "validation-2022 development reuse; not independent confirmation",
    }


@torch.no_grad()
def _sample_frozen_laws(
    sampler: Any,
    candidate_states: dict[str, dict[str, torch.Tensor]],
    cases: list[tuple[int, dict[str, Any]]],
    identities: list[dict[str, Any]],
    fixed_noise: torch.Tensor,
    image_size: tuple[int, int],
    members: int,
    device: torch.device,
    evidence_root: Path,
    status_path: Path,
    bindings_path: Path,
    clearml_task_id: str,
) -> dict[str, torch.Tensor]:
    """Sample each complete public law, committing every case before scoring."""
    ensembles: dict[str, torch.Tensor] = {}
    manifests: dict[str, list[dict[str, Any]]] = {}
    for law_order, label in enumerate(LABELS):
        if label != "ema6":
            sampler.model.load_state_dict(candidate_states[label], strict=True)
        sampler.model.eval()
        law_cases: list[torch.Tensor] = []
        manifest: list[dict[str, Any]] = []
        law_root = evidence_root / label
        law_root.mkdir(parents=True, exist_ok=False)
        for case_order, (_, item) in enumerate(cases):
            print(
                f"[full-cfm-validation] law={label} "
                f"case={case_order + 1}/{len(cases)}",
                flush=True,
            )
            member_samples = [
                _public_sample(
                    sampler,
                    item,
                    fixed_noise[case_order, member : member + 1],
                    image_size,
                    device,
                )[0].cpu()
                for member in range(members)
            ]
            case_tensor = torch.stack(member_samples)
            case_path = law_root / f"case_{case_order:02d}.pt"
            case_sha = _atomic_torch_save(
                {
                    "label": label,
                    "identity": identities[case_order],
                    "normalized": case_tensor,
                    "initial_noise": fixed_noise[case_order],
                },
                case_path,
            )
            manifest.append(
                {
                    "case_order": case_order,
                    "identity": identities[case_order],
                    "path": str(case_path),
                    "sha256": case_sha,
                }
            )
            manifest_path = law_root / "manifest.json"
            _atomic_json(manifest_path, {"label": label, "completed_cases": manifest})
            _record_evidence_binding(bindings_path, f"{label}_manifest", manifest_path)
            _merge_status(
                status_path,
                status="sampling",
                active_law=label,
                completed_laws=law_order,
                completed_cases_in_active_law=case_order + 1,
                clearml_task_id=clearml_task_id,
                evidence_bindings_sha256=_sha256(bindings_path),
                test_2023_accessed=False,
            )
            if not torch.isfinite(case_tensor).all():
                raise FloatingPointError(f"non-finite sample in {label} case {case_order}")
            law_cases.append(case_tensor)
        ensembles[label] = torch.stack(law_cases)
        manifests[label] = manifest
    return ensembles


def _rank_figure(
    path: Path,
    label: str,
    diagnostics: dict[str, Any],
    kind: str,
) -> None:
    uniform = 1.0 / 9.0
    if kind == "overall":
        figure, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
        for index, key in enumerate(OUTPUT_KEYS):
            axis = axes[index % 2, index // 2]
            values = diagnostics["overall"][key]["frequencies"]
            axis.bar(range(9), values, color="#276dc3")
            axis.axhline(uniform, color="#e67e22", linestyle="--")
            axis.set_title(key)
            axis.set_xlabel("rank")
        axes[0, 0].set_ylabel("case-equal frequency")
        axes[1, 0].set_ylabel("case-equal frequency")
    elif kind == "seasonal":
        seasons = ("DJF", "MAM", "JJA", "SON")
        figure, axes = plt.subplots(4, 6, figsize=(18, 11), sharex=True, sharey=True)
        for column, key in enumerate(OUTPUT_KEYS):
            for row, season in enumerate(seasons):
                axis = axes[row, column]
                values = diagnostics["seasonal"][key][season]["frequencies"]
                if values is not None:
                    axis.bar(range(9), values, color="#276dc3")
                axis.axhline(uniform, color="#e67e22", linestyle="--", linewidth=1)
                if row == 0:
                    axis.set_title(key)
                if column == 0:
                    axis.set_ylabel(season)
        for axis in axes[-1]:
            axis.set_xlabel("rank")
    elif kind == "truth_support":
        figure, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
        colors = ("#276dc3", "#2ca25f", "#d95f0e")
        for index, key in enumerate(OUTPUT_KEYS):
            axis = axes[index % 2, index // 2]
            for color, (group, values) in zip(
                colors, diagnostics["truth_support"][key].items(), strict=False
            ):
                frequencies = values["frequencies"]
                if frequencies is not None:
                    axis.plot(range(9), frequencies, marker="o", color=color, label=group)
            axis.axhline(uniform, color="#777777", linestyle="--", linewidth=1)
            axis.set_title(key)
            axis.set_xlabel("rank")
            axis.legend(fontsize=7)
        axes[0, 0].set_ylabel("case-equal frequency")
        axes[1, 0].set_ylabel("case-equal frequency")
    else:
        raise ValueError(kind)
    figure.suptitle(f"{label}: {kind} tie-aware fractional ranks")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _save_paired_delta_figures(
    output: Path,
    treatment: torch.Tensor,
    control: torch.Tensor,
    valid: torch.Tensor,
    case_orders: list[int],
) -> list[str]:
    root = output / "paired_deltas"
    root.mkdir(parents=True, exist_ok=False)
    paths: list[str] = []
    for case_order in case_orders:
        delta = (treatment[case_order].mean(0) - control[case_order].mean(0)).clone()
        delta[:, valid[case_order, 0] <= 0] = float("nan")
        figure, axes = plt.subplots(2, 3, figsize=(12, 7))
        for channel, key in enumerate(OUTPUT_KEYS):
            axis = axes[channel % 2, channel // 2]
            limit = 0.25 if key.endswith("_sic") else 0.75
            image = axis.imshow(
                delta[channel], origin="upper", cmap="RdBu_r", vmin=-limit, vmax=limit
            )
            axis.set_title(f"treatment-control {key}")
            axis.set_xticks([])
            axis.set_yticks([])
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        figure.suptitle(f"case order {case_order}: ensemble-mean paired delta")
        figure.tight_layout()
        path = root / f"case_{case_order:02d}.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        paths.append(str(path))
    return paths


def run(config_path: Path, output: Path) -> dict[str, Any]:
    experiment = load_json(config_path)
    protocol = _check_contract(experiment)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to reuse output {output}")
    output.mkdir(parents=True)
    status_path = output / "status.json"
    _atomic_json(status_path, {"status": "initializing", "test_2023_accessed": False})
    tracker = None
    prior_handlers: dict[int, Any] = {}

    def terminate(signum: int, _frame: Any) -> None:
        raise InterruptedError(f"received termination signal {signum}")

    for signum in (signal.SIGTERM, signal.SIGINT):
        prior_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, terminate)
    try:
        root = Path(__file__).parents[1]
        source = experiment["source"]
        run_dir = Path(source["run_dir"])
        source_files = {
            "metadata.json": source["metadata_sha256"],
            "config.json": source["config_sha256"],
            source["checkpoint"]: source["checkpoint_sha256"],
        }
        for relative, expected in source_files.items():
            if _sha256(run_dir / relative) != expected:
                raise ValueError(f"source SHA mismatch: {relative}")
        metadata = load_json(run_dir / "metadata.json")
        source_config = load_json(run_dir / "config.json")
        model_config = TrainingConfig.from_dict(metadata["training_config"])
        if tuple(model_config.image_size) != (320, 256) or model_config.out_channels != 6:
            raise ValueError("unexpected EMA6 model geometry")
        training_result, candidate_states = _verify_training_and_models(
            experiment, model_config
        )

        panel_spec = experiment["panel_source"]
        panel_path = root / panel_spec["path"]
        if _sha256(panel_path) != panel_spec["sha256"]:
            raise ValueError("frozen panel source SHA mismatch")
        panel = load_json(panel_path)
        expected_identities = panel[panel_spec["identity_key"]]
        dataset = build_dataset(metadata["data_config"], split="valid")
        if dataset.split != "valid" or len(dataset) != 8544:
            raise ValueError("validation-2022 dataset identity mismatch")
        cache = {
            index: dataset[index]
            for index in sorted(set(protocol["case_indices"]) | {0, 12, 23})
        }
        sentinel = validate_direct_dataset(dataset, item_cache=cache)
        cases = [(index, cache[index]) for index in protocol["case_indices"]]
        identities = [_case_identity(item, index) for index, item in cases]
        if identities != expected_identities:
            raise ValueError("frozen panel case identities differ")

        image_size = tuple(model_config.image_size)
        fixed_noise = torch.stack(
            [
                torch.cat(
                    [_noise(case, member, image_size) for member in range(protocol["members"])]
                )
                for case in range(len(cases))
            ]
        )
        fixed_inputs_path = output / "fixed_inputs_before_gpu.pt"
        fixed_inputs_sha = _atomic_torch_save(
            {
                "case_identities": identities,
                "initial_noise": fixed_noise,
                "condition": torch.stack(
                    [item["structured_conditioning"].float() for _, item in cases]
                ),
                "truth_standardized": torch.stack([item["truth"].float() for _, item in cases]),
                "truth_physical_exact": torch.stack(
                    [item["structured_physical_truth"].float() for _, item in cases]
                ),
                "persistence_physical": torch.stack(
                    [item["structured_physical_background"].float() for _, item in cases]
                ),
                "valid": torch.stack([item["valid_mask"][:1].float() for _, item in cases]),
            },
            fixed_inputs_path,
        )
        contract_path = output / "run_contract_before_gpu.json"
        _atomic_json(
            contract_path,
            {
                "schema_version": SCHEMA_VERSION,
                "protocol": protocol,
                "scores": experiment["scores"],
                "rank_strata": experiment["rank_strata"],
                "gate": experiment["gate"],
                "case_identities": identities,
                "source_files": source_files,
                "training_result_sha256": experiment["training_evidence"]["training_result"]["sha256"],
                "raw_law_sha256": {
                    law["label"]: law.get("sha256") for law in experiment["laws"]
                },
                "evaluation_code_sha256": _sha256(Path(__file__)),
                "config_sha256": _sha256(config_path),
                "fixed_inputs_sha256": fixed_inputs_sha,
                "checkpoint_model_optimizer_audit": "passed_exact_raw_update512_step512",
                "optimizer_steps": 0,
                "test_2023_accessed": False,
            },
        )
        bindings_path = output / "evidence_bindings.json"
        _record_evidence_binding(bindings_path, "fixed_inputs_before_gpu", fixed_inputs_path)
        _record_evidence_binding(bindings_path, "run_contract_before_gpu", contract_path)
        _merge_status(
            status_path,
            status="cpu_evidence_bound_before_gpu",
            evidence_bindings_sha256=_sha256(bindings_path),
            test_2023_accessed=False,
        )

        if torch.cuda.device_count() != 1 or os.environ.get("CLEARML_REQUIRE_ONLINE") != "1":
            raise RuntimeError("validation requires exactly one visible GPU and online ClearML")
        tracker = ClearMLTracker(
            experiment["project_name"],
            f"{experiment['task_name']}-{output.name}",
            tags=experiment["clearml"]["tags"],
            env_path=experiment["clearml"]["env_path"],
        )
        tracker.connect(
            "frozen_validation_contract",
            {"protocol": protocol, "gate": experiment["gate"], "identities": identities},
        )
        task_id = str(tracker.task.id)
        _merge_status(
            status_path,
            status="sampling",
            clearml_task_id=task_id,
            completed_laws=0,
            test_2023_accessed=False,
        )
        device = torch.device("cuda:0")
        sampler = load_sampler(
            str(run_dir), source["checkpoint"], source_config, device=device
        )
        evidence_root = output / "sampling_evidence"
        evidence_root.mkdir()
        normalized = _sample_frozen_laws(
            sampler,
            candidate_states,
            cases,
            identities,
            fixed_noise,
            image_size,
            protocol["members"],
            device,
            evidence_root,
            status_path,
            bindings_path,
            task_id,
        )
        del sampler, candidate_states
        torch.cuda.empty_cache()

        fixed = torch.load(fixed_inputs_path, map_location="cpu", weights_only=True)
        means = _field_stats_tensor(dataset.means)
        stds = _field_stats_tensor(dataset.stds)
        physical = {
            label: channel_denormalize(values.float(), means, stds)
            for label, values in normalized.items()
        }
        complete_path = output / "complete_three_law_ensembles_before_scoring.pt"
        complete_sha = _atomic_torch_save(
            {
                "normalized": normalized,
                "physical": physical,
                "truth_standardized": fixed["truth_standardized"],
                "truth_physical_exact": fixed["truth_physical_exact"],
                "persistence_physical": fixed["persistence_physical"],
                "valid": fixed["valid"],
                "case_identities": identities,
                "initial_noise_sha256": fixed_inputs_sha,
            },
            complete_path,
        )
        _record_evidence_binding(
            bindings_path, "complete_three_law_ensembles_before_scoring", complete_path
        )
        _merge_status(
            status_path,
            status="scoring",
            complete_tensor_sha256=complete_sha,
            evidence_bindings_sha256=_sha256(bindings_path),
            clearml_task_id=task_id,
            test_2023_accessed=False,
        )

        evidence = {
            "truth_standardized": fixed["truth_standardized"],
            "persistence_physical": fixed["persistence_physical"],
            "valid": fixed["valid"],
            "initial_sic": fixed["persistence_physical"][:, 0:1],
        }
        mask = torch.from_numpy(~np.load(metadata["data_config"]["mask_path"])).unsqueeze(0)
        cell_area, area_evidence = _cell_area_proxy(
            Path(experiment["cell_geometry"]["latitude_path"]),
            Path(experiment["cell_geometry"]["longitude_path"]),
            mask,
            image_size,
            torch.device("cpu"),
        )
        scores: dict[str, Any] = {}
        diagnostics: dict[str, Any] = {}
        for label in LABELS:
            scores[label] = _base_score_branch(
                normalized[label],
                physical[label],
                evidence,
                means,
                stds,
                cell_area,
                experiment["product_scale"],
            )
            diagnostics[label] = _stratified_diagnostics(
                normalized[label],
                fixed["truth_standardized"],
                fixed["truth_physical_exact"],
                fixed["valid"],
                identities,
                means,
                stds,
            )
        comparisons = _comparisons(
            scores, protocol["bootstrap_replicates"], protocol["bootstrap_seed"]
        )
        result: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": "scored_pending_manual_visual_review",
            "split": "validation-2022",
            "development_reuse": True,
            "optimizer_steps": 0,
            "test_2023_accessed": False,
            "protocol": protocol,
            "case_identities": identities,
            "source": source,
            "training_commit": experiment["training_evidence"]["commit"],
            "training_result_sha256": experiment["training_evidence"]["training_result"][
                "sha256"
            ],
            "raw_law_sha256": {
                law["label"]: law.get("sha256") for law in experiment["laws"]
            },
            "evaluation_code_sha256": _sha256(Path(__file__)),
            "config_sha256": _sha256(config_path),
            "fixed_inputs": {"path": str(fixed_inputs_path), "sha256": fixed_inputs_sha},
            "run_contract": {"path": str(contract_path), "sha256": _sha256(contract_path)},
            "complete_tensor": {"path": str(complete_path), "sha256": complete_sha},
            "evidence_bindings": {
                "path": str(bindings_path),
                "sha256": _sha256(bindings_path),
            },
            "dataset_sentinel": sentinel,
            "cell_area": area_evidence,
            "scores": scores,
            "rank_diagnostics": diagnostics,
            "comparisons": comparisons,
            "clearml_task_id": task_id,
        }
        result["gate"] = _gate(scores, comparisons)
        _finite_tree(result)
        numerical_path = output / "frozen_numerical_result_before_plots.json"
        _atomic_json(numerical_path, result)

        rank_root = output / "rank_histograms"
        rank_root.mkdir()
        for label in LABELS:
            for kind in ("overall", "seasonal", "truth_support"):
                path = rank_root / f"{label}_{kind}.png"
                _rank_figure(path, label, diagnostics[label], kind)
                tracker.report_image("rank_histograms", f"{label}/{kind}", path, 0)
            for path_string in _save_visuals(
                output,
                label,
                physical[label],
                fixed["truth_physical_exact"],
                fixed["persistence_physical"],
                fixed["valid"],
                protocol["visual_case_orders"],
            ):
                path = Path(path_string)
                tracker.report_image("individual_members", f"{label}/{path.stem}", path, 0)
        for path_string in _save_paired_delta_figures(
            output,
            physical["treatment512"],
            physical["control512"],
            fixed["valid"],
            protocol["visual_case_orders"],
        ):
            path = Path(path_string)
            tracker.report_image("paired_deltas", path.stem, path, 0)

        final_path = output / "paired_validation.json"
        _atomic_json(final_path, result)
        tracker.upload_artifact("paired_validation", final_path)
        tracker.upload_artifact("frozen_numerical_result", numerical_path)
        tracker.upload_artifact("complete_three_law_ensembles", complete_path)
        _close_and_mark_complete(status_path, tracker, task_id)
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
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.config.resolve(), arguments.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
