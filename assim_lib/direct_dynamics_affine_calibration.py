"""Frozen persistence-centred affine calibration for the EMA6 dynamics flow.

The candidate changes the predictive law member by member in train-standardized
coordinates.  It uses one positive scale for the complete six-channel
trajectory and one deterministic offset per physical field.  No pixelwise
parameters, filtering, projection, retraining, or new stochastic variables are
introduced.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch

from .clearml_tracking import ClearMLTracker
from .data import build_dataset
from .direct_dynamics_evaluation import EXPECTED_INPUT_SHA256, _require_finite_scalars, _save_visuals, _sha256
from .direct_dynamics_tail_diagnostic import EMA6_CHECKPOINT
from .direct_dynamics_temperature_calibration import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CALIBRATION_INDICES,
    CONFIRMATION_INDICES,
    STRESS_DATASET_INDICES,
    _development_candidate_gate,
    bootstrap_J_upper95,
    comparison_summary,
    sample_panel,
    score_cases_equal,
    tail_gate,
    tail_metrics,
    validate_panel_indices,
)
from .direct_dynamics_training import DIRECT_LEADS, _repeat_field_stats, validate_direct_dataset
from .model_io import load_sampler
from .structured_trajectory_evaluation import _fair_crps
from .trainer import _atomic_json

SCALE_BOUNDS = (0.8, 1.5)
OFFSET_BOUNDS = (-0.1, 0.1)
GOLDEN_ITERATIONS = 18
MEMBERS = 8
SOLVER_STEPS = 33
_ACTIVE_TRACKER: ClearMLTracker | None = None
_ACTIVE_STAGE = "preflight"


def apply_persistence_affine(
    ensemble: torch.Tensor,
    persistence: torch.Tensor,
    means: torch.Tensor,
    stds: torch.Tensor,
    *,
    scale: float,
    sic_offset: float,
    sit_offset: float,
) -> torch.Tensor:
    """Apply the same coherent affine map to every member and spatial point."""
    if ensemble.ndim != 5 or persistence.ndim != 4:
        raise ValueError("expected ensemble [case,member,channel,y,x] and persistence [case,channel,y,x]")
    if ensemble.shape[0] != persistence.shape[0] or ensemble.shape[2:] != persistence.shape[1:]:
        raise ValueError("ensemble and persistence shapes are incompatible")
    if ensemble.shape[2] != 6 or means.numel() != 6 or stds.numel() != 6:
        raise ValueError("affine calibration requires the six-channel d3/d6/d9 trajectory")
    if not SCALE_BOUNDS[0] <= scale <= SCALE_BOUNDS[1] or not math.isfinite(scale):
        raise ValueError("scale is outside the frozen admissible interval")
    if any(not OFFSET_BOUNDS[0] <= value <= OFFSET_BOUNDS[1] for value in (sic_offset, sit_offset)):
        raise ValueError("offset is outside the frozen admissible interval")
    shape = (1, 1, 6, 1, 1)
    means = means.to(dtype=ensemble.dtype, device=ensemble.device).reshape(shape)
    stds = stds.to(dtype=ensemble.dtype, device=ensemble.device).reshape(shape)
    if not torch.isfinite(stds).all() or torch.any(stds <= 0):
        raise ValueError("normalization standard deviations must be finite and positive")
    standardized = (ensemble - means) / stds
    centre = (persistence[:, None] - means) / stds
    offsets = ensemble.new_tensor([sic_offset, sit_offset, sic_offset, sit_offset, sic_offset, sit_offset])
    calibrated = centre + scale * (standardized - centre) + offsets.reshape(shape)
    physical = calibrated * stds + means
    if not torch.isfinite(physical).all():
        raise FloatingPointError("affine calibration produced NaN/Inf")
    return physical


def _fit_arrays(
    ensemble: torch.Tensor,
    truth: torch.Tensor,
    persistence: torch.Tensor,
    mask: torch.Tensor,
    means: torch.Tensor,
    stds: torch.Tensor,
) -> list[tuple[torch.Tensor, torch.Tensor, float]]:
    """Return point-major member deltas, targets, and fair-CRPS spread constants."""
    valid_counts = mask.flatten(1).sum(dim=1)
    if not torch.equal(valid_counts, valid_counts[:1].expand_as(valid_counts)):
        raise ValueError("case-equal affine fit requires identical static valid-ocean masks")
    means = means.reshape(1, 1, 6, 1, 1).to(torch.float32)
    stds = stds.reshape(1, 1, 6, 1, 1).to(torch.float32)
    standardized = (ensemble.float() - means) / stds
    centre = (persistence[:, None].float() - means) / stds
    target = (truth.float() - means[:, 0]) / stds[:, 0]
    result = []
    member_count = ensemble.shape[1]
    weights = torch.arange(member_count, dtype=torch.float32) * 2 - member_count + 1
    for field_offset in (0, 1):
        channels = slice(field_offset, 6, 2)
        deltas = standardized[:, :, channels] - centre[:, :, channels]
        target_delta = target[:, channels] - centre[:, 0, channels]
        point_mask = mask.expand(-1, 3, -1, -1) > 0
        point_deltas = deltas.permute(0, 2, 3, 4, 1)[point_mask].contiguous()
        point_targets = target_delta[point_mask].contiguous()
        ordered = point_deltas.sort(dim=1).values
        pairwise = ((ordered * weights).sum(dim=1) / (member_count * (member_count - 1))).mean()
        result.append((point_deltas, point_targets, float(pairwise.item())))
    return result


def fit_persistence_affine(
    ensemble: torch.Tensor,
    truth: torch.Tensor,
    persistence: torch.Tensor,
    mask: torch.Tensor,
    means: torch.Tensor,
    stds: torch.Tensor,
) -> dict[str, Any]:
    """Fit the three frozen coefficients to exact case/output-equal fair CRPS."""
    arrays = _fit_arrays(ensemble, truth, persistence, mask, means, stds)
    cache: dict[float, tuple[float, tuple[float, float], tuple[float, float]]] = {}

    def objective(scale: float) -> float:
        key = round(float(scale), 12)
        if key in cache:
            return cache[key][0]
        field_scores = []
        offsets = []
        first_terms = []
        for member_delta, target_delta, pairwise in arrays:
            residual = target_delta[:, None] - float(scale) * member_delta
            offset = float(torch.median(residual.reshape(-1)).clamp(*OFFSET_BOUNDS).item())
            first = float((float(scale) * member_delta + offset - target_delta[:, None]).abs().mean().item())
            field_scores.append(first - float(scale) * pairwise)
            offsets.append(offset)
            first_terms.append(first)
        value = sum(field_scores) / len(field_scores)
        cache[key] = (value, (offsets[0], offsets[1]), (first_terms[0], first_terms[1]))
        return value

    lo, hi = SCALE_BOUNDS
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    left = hi - ratio * (hi - lo)
    right = lo + ratio * (hi - lo)
    f_left, f_right = objective(left), objective(right)
    for _ in range(GOLDEN_ITERATIONS):
        if f_left <= f_right:
            hi, right, f_right = right, left, f_left
            left = hi - ratio * (hi - lo)
            f_left = objective(left)
        else:
            lo, left, f_left = left, right, f_right
            right = lo + ratio * (hi - lo)
            f_right = objective(right)
    candidates = [lo, hi, left, right, (lo + hi) / 2.0, 1.0, *SCALE_BOUNDS]
    scale = min(candidates, key=objective)
    value = objective(scale)
    offsets = cache[round(float(scale), 12)][1]
    identity_field_scores = []
    for member_delta, target_delta, pairwise in arrays:
        first = float((member_delta - target_delta[:, None]).abs().mean().item())
        identity_field_scores.append(first - pairwise)
    return {
        "scale": float(scale),
        "sic_offset": offsets[0],
        "sit_offset": offsets[1],
        "objective": value,
        "objective_identity": sum(identity_field_scores) / len(identity_field_scores),
        "objective_scale1_profiled_offsets": objective(1.0),
        "scale_bounds": list(SCALE_BOUNDS),
        "offset_bounds": list(OFFSET_BOUNDS),
        "golden_iterations": GOLDEN_ITERATIONS,
        "evaluations": len(cache),
        "fit_score": "case-equal six-output-equal fair CRPS in train-standardized coordinates",
    }


def coverage_metrics(ensemble: torch.Tensor, truth: torch.Tensor, mask: torch.Tensor) -> dict[str, Any]:
    result = {}
    references = {"outer_1_8": 7.0 / 9.0, "inner_2_7": 5.0 / 9.0}
    for lead_index, lead in enumerate(DIRECT_LEADS):
        for offset, field in enumerate(("sic", "sit")):
            channel = 2 * lead_index + offset
            members = ensemble[:, :, channel : channel + 1]
            target = truth[:, channel : channel + 1]
            valid = mask > 0
            values = {}
            less = (members < target[:, None]).sum(dim=1)
            equal = (members == target[:, None]).sum(dim=1)
            for name, lower_rank, upper_rank in (("outer_1_8", 1, 7), ("inner_2_7", 2, 6)):
                first = torch.maximum(less, torch.full_like(less, lower_rank))
                last = torch.minimum(less + equal, torch.full_like(less, upper_rank))
                fractional_inside = torch.clamp(last - first + 1, min=0).float() / (equal + 1).float()
                per_case = [
                    float(fractional_inside[case][valid[case]].mean().item())
                    for case in range(len(fractional_inside))
                ]
                coverage = sum(per_case) / len(per_case)
                values[name] = {
                    "coverage": coverage,
                    "iid_reference": references[name],
                    "absolute_error": abs(coverage - references[name]),
                    "per_case": per_case,
                }
            result[f"d{lead}_{field}"] = values
    return result


def mean_coverage_error(metrics: dict[str, Any]) -> float:
    return sum(interval["absolute_error"] for output in metrics.values() for interval in output.values()) / (
        2 * len(metrics)
    )


def boundary_event_metrics(ensemble: torch.Tensor, truth: torch.Tensor, mask: torch.Tensor) -> dict[str, Any]:
    """Score fixed near-boundary events without defining training regimes."""
    result: dict[str, Any] = {}
    for lead_index, lead in enumerate(DIRECT_LEADS):
        for offset, field in enumerate(("sic", "sit")):
            channel = 2 * lead_index + offset
            members = ensemble[:, :, channel : channel + 1]
            target = truth[:, channel : channel + 1]
            valid = mask > 0
            events = (
                (
                    ("sic_le_0p01", members <= 0.01, target <= 0.01),
                    ("sic_ge_0p99", members >= 0.99, target >= 0.99),
                )
                if field == "sic"
                else (("sit_le_0p01", members <= 0.01, target <= 0.01),)
            )
            event_result = {}
            for name, member_event, truth_event in events:
                probability = member_event.float().mean(dim=1)
                per_case = [
                    float((probability[i][valid[i]] - truth_event[i][valid[i]].float()).square().mean())
                    for i in range(len(ensemble))
                ]
                reliability = {}
                per_case_ece = []
                for probability_index in range(MEMBERS + 1):
                    forecast_probability = probability_index / MEMBERS
                    per_case_bin = []
                    for case in range(len(ensemble)):
                        selected_bin = (probability[case] == forecast_probability) & valid[case]
                        count = int(selected_bin.sum().item())
                        if count == 0:
                            per_case_bin.append(None)
                            continue
                        per_case_bin.append(
                            {
                                "mass": count / int(valid[case].sum().item()),
                                "observed_frequency": float(
                                    truth_event[case][selected_bin].float().mean().item()
                                ),
                            }
                        )
                    nonempty = [value for value in per_case_bin if value is not None]
                    reliability[f"{probability_index}/{MEMBERS}"] = {
                        "forecast_probability": forecast_probability,
                        "case_count": len(nonempty),
                        "mean_case_mass": (
                            sum(value["mass"] for value in nonempty) / len(ensemble) if nonempty else 0.0
                        ),
                        "case_equal_observed_frequency": (
                            sum(value["observed_frequency"] for value in nonempty) / len(nonempty)
                            if nonempty
                            else None
                        ),
                        "per_case": per_case_bin,
                    }
                for case in range(len(ensemble)):
                    case_error = 0.0
                    for probability_index in range(MEMBERS + 1):
                        entry = reliability[f"{probability_index}/{MEMBERS}"]["per_case"][case]
                        if entry is not None:
                            case_error += entry["mass"] * abs(
                                entry["observed_frequency"] - probability_index / MEMBERS
                            )
                    per_case_ece.append(case_error)
                event_result[name] = {
                    "brier": sum(per_case) / len(per_case),
                    "per_case_brier": per_case,
                    "forecast_rate": float(probability[valid].mean().item()),
                    "truth_rate": float(truth_event[valid].float().mean().item()),
                    "reliability_ece": sum(per_case_ece) / len(per_case_ece),
                    "reliability": reliability,
                }
            if field == "sit":
                truth_open = (target <= 0.01) & valid
                per_case_haze = []
                for case in range(len(ensemble)):
                    selected = members[case][truth_open[case][None].expand_as(members[case])]
                    if selected.numel() == 0:
                        per_case_haze.append(None)
                        continue
                    positive = selected.clamp_min(0)
                    per_case_haze.append(
                        {
                            "mean_positive_sit": float(positive.mean().item()),
                            "p95_positive_sit": float(torch.quantile(positive, 0.95).item()),
                            "fraction_gt_0p01": float((positive > 0.01).float().mean().item()),
                        }
                    )
                nonempty_haze = [value for value in per_case_haze if value is not None]
                event_result["open_water_haze"] = {
                    "definition": "positive SIT on valid points with truth SIT <= 0.01 m; no upper cutoff",
                    "case_count": len(nonempty_haze),
                    "mean_positive_sit": (
                        sum(value["mean_positive_sit"] for value in nonempty_haze) / len(nonempty_haze)
                        if nonempty_haze
                        else None
                    ),
                    "p95_positive_sit": (
                        sum(value["p95_positive_sit"] for value in nonempty_haze) / len(nonempty_haze)
                        if nonempty_haze
                        else None
                    ),
                    "fraction_gt_0p01": (
                        sum(value["fraction_gt_0p01"] for value in nonempty_haze) / len(nonempty_haze)
                        if nonempty_haze
                        else None
                    ),
                    "per_case": per_case_haze,
                }
            result[f"d{lead}_{field}"] = event_result
    return result


def joint_energy_score(
    ensemble: torch.Tensor,
    truth: torch.Tensor,
    mask: torch.Tensor,
    stds: torch.Tensor,
) -> float:
    scaled_members = ensemble.float() / stds.reshape(1, 1, 6, 1, 1)
    scaled_truth = truth.float() / stds.reshape(1, 6, 1, 1)
    scores = []
    for case in range(len(ensemble)):
        valid = mask[case].expand(6, -1, -1) > 0
        members = scaled_members[case, :, valid]
        target = scaled_truth[case, valid]
        normalization = math.sqrt(float(target.numel()))
        first = torch.linalg.vector_norm(members - target, dim=1).mean() / normalization
        pair_sum = ensemble.new_zeros(())
        for left in range(MEMBERS):
            for right in range(left + 1, MEMBERS):
                pair_sum += torch.linalg.vector_norm(members[left] - members[right]) / normalization
        scores.append(first - pair_sum / (MEMBERS * (MEMBERS - 1)))
    return float(torch.stack(scores).mean().item())


def standardized_fair_crps(
    ensemble: torch.Tensor,
    truth: torch.Tensor,
    mask: torch.Tensor,
    stds: torch.Tensor,
) -> dict[str, Any]:
    """Primary case/output-equal CRPS in train-standardized units."""
    standardized_members = ensemble.float() / stds.reshape(1, 1, 6, 1, 1)
    standardized_truth = truth.float() / stds.reshape(1, 6, 1, 1)
    per_case = []
    for case in range(len(ensemble)):
        values = []
        for channel in range(6):
            values.append(
                _fair_crps(
                    standardized_members[case : case + 1, :, channel : channel + 1],
                    standardized_truth[case : case + 1, channel : channel + 1],
                    mask[case : case + 1],
                )
            )
        per_case.append(values)
    tensor = torch.tensor(per_case, dtype=torch.float64)
    return {
        "aggregate": float(tensor.mean().item()),
        "per_case_output": [[float(value) for value in row] for row in tensor],
    }


def primary_score_comparison(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    candidate_values = torch.tensor(candidate["per_case_output"], dtype=torch.float64)
    baseline_values = torch.tensor(baseline["per_case_output"], dtype=torch.float64)
    ratio = candidate_values.mean() / baseline_values.mean().clamp(min=1e-12)
    generator = torch.Generator(device="cpu").manual_seed(BOOTSTRAP_SEED)
    indices = torch.randint(
        len(candidate_values),
        (BOOTSTRAP_REPLICATES, len(candidate_values)),
        generator=generator,
    )
    boot = candidate_values[indices].mean(dim=(1, 2)) / baseline_values[indices].mean(dim=(1, 2)).clamp(
        min=1e-12
    )
    return {
        "ratio": float(ratio.item()),
        "paired_date_bootstrap_upper95": float(torch.quantile(boot, 0.95).item()),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    }


def _calibration_diagnostics(
    ensemble: torch.Tensor,
    truth: torch.Tensor,
    persistence: torch.Tensor,
    mask: torch.Tensor,
    stds: torch.Tensor,
) -> dict[str, Any]:
    return {
        "scores": score_cases_equal(ensemble, truth, persistence, mask),
        "tails": tail_metrics(ensemble, mask),
        "coverage": coverage_metrics(ensemble, truth, mask),
        "boundary_events": boundary_event_metrics(ensemble, truth, mask),
        "joint_energy_score": joint_energy_score(ensemble, truth, mask, stds),
        "primary_standardized_fair_crps": standardized_fair_crps(ensemble, truth, mask, stds),
    }


def _structural_comparison(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    roughness_ratios = {}
    temporal_rmse_ratios = {}
    for lead in ("d3", "d6", "d9"):
        for field in ("sic", "sit"):
            key = f"{lead}_{field}"
            cand_values = [
                case["leads"][lead][field]["member_roughness"] for case in candidate["scores"]["per_case"]
            ]
            base_values = [
                case["leads"][lead][field]["member_roughness"] for case in baseline["scores"]["per_case"]
            ]
            roughness_ratios[key] = (sum(cand_values) / len(cand_values)) / max(
                sum(base_values) / len(base_values), 1e-12
            )
    temporal_keys = baseline["scores"]["per_case"][0]["temporal_change"]
    for key in temporal_keys:
        cand_values = [case["temporal_change"][key] for case in candidate["scores"]["per_case"]]
        base_values = [case["temporal_change"][key] for case in baseline["scores"]["per_case"]]
        candidate_rmse = math.sqrt(sum(value * value for value in cand_values) / len(cand_values))
        baseline_rmse = math.sqrt(sum(value * value for value in base_values) / len(base_values))
        temporal_rmse_ratios[key] = candidate_rmse / max(baseline_rmse, 1e-12)
    return {"roughness_ratios": roughness_ratios, "temporal_rmse_ratios": temporal_rmse_ratios}


def _extra_gate(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    failures = []
    candidate_coverage = mean_coverage_error(candidate["coverage"])
    baseline_coverage = mean_coverage_error(baseline["coverage"])
    if candidate_coverage > baseline_coverage + 1e-6:
        failures.append("mean attainable-coverage error increased")
    if candidate["joint_energy_score"] > 1.01 * baseline["joint_energy_score"]:
        failures.append("joint energy score worsened by more than 1%")
    boundary_ratios = {}
    haze_differences = {}
    for output, events in baseline["boundary_events"].items():
        for event, base in events.items():
            candidate_event = candidate["boundary_events"][output][event]
            key = f"{output}_{event}"
            if event == "open_water_haze":
                if candidate_event["case_count"] != base["case_count"]:
                    failures.append(f"{key} support case count changed")
                    continue
                for metric, tolerance in (
                    ("mean_positive_sit", 0.005),
                    ("p95_positive_sit", 0.01),
                    ("fraction_gt_0p01", 0.005),
                ):
                    candidate_value = candidate_event[metric]
                    baseline_value = base[metric]
                    haze_differences[f"{key}_{metric}"] = (
                        None
                        if candidate_value is None or baseline_value is None
                        else candidate_value - baseline_value
                    )
                    if (candidate_value is None) != (baseline_value is None):
                        failures.append(f"{key} {metric} changed between empty and non-empty")
                    elif candidate_value is not None and candidate_value > baseline_value + tolerance:
                        failures.append(f"{key} {metric} increased by more than {tolerance}")
            else:
                boundary_ratios[key] = candidate_event["brier"] / max(base["brier"], 1e-12)
                if candidate_event["brier"] > 1.01 * base["brier"] + 1e-5:
                    failures.append(f"{key} Brier worsened by more than 1%")
                if candidate_event["reliability_ece"] > 1.01 * base["reliability_ece"] + 1e-5:
                    failures.append(f"{key} reliability ECE worsened by more than 1%")
    structural = _structural_comparison(candidate, baseline)
    for key, ratio in structural["roughness_ratios"].items():
        if not 0.95 <= ratio <= 1.05:
            failures.append(f"{key} member roughness changed by more than 5%")
    for key, ratio in structural["temporal_rmse_ratios"].items():
        if ratio > 1.05:
            failures.append(f"{key} temporal-change RMSE worsened by more than 5%")
    return {
        "passed": not failures,
        "failures": failures,
        "mean_coverage_error_candidate": candidate_coverage,
        "mean_coverage_error_baseline": baseline_coverage,
        "joint_energy_ratio": candidate["joint_energy_score"] / max(baseline["joint_energy_score"], 1e-12),
        "boundary_brier_ratios": boundary_ratios,
        "open_water_haze_fraction_differences": haze_differences,
        "structural_comparison": structural,
    }


def _save_raw_bundle(
    path: Path,
    ensemble: torch.Tensor,
    truth: torch.Tensor,
    persistence: torch.Tensor,
    mask: torch.Tensor,
    identities: list[dict[str, Any]],
    coefficients: dict[str, Any] | None,
    checkpoint_sha256: str,
    means: torch.Tensor,
    stds: torch.Tensor,
    noise_orders: list[int],
) -> None:
    temporary = path.with_name(f".{path.name}.incomplete")
    torch.save(
        {
            "raw_ensemble": ensemble,
            "truth": truth,
            "persistence": persistence,
            "valid_mask": mask,
            "case_identities": identities,
            "affine_coefficients": coefficients,
            "checkpoint": EMA6_CHECKPOINT,
            "checkpoint_sha256": checkpoint_sha256,
            "normalization_means": [float(value) for value in means],
            "normalization_stds": [float(value) for value in stds],
            "noise_orders": noise_orders,
            "member_noise_seed_rule": "314159 + 1000003*noise_order + 1009*member_index",
            "solver": "NN/ODE FP32; TF32 off; RK4-33",
            "projection_role": "none_raw_evidence",
        },
        temporary,
    )
    os.replace(temporary, path)


def run(run_dir: Path, output: Path) -> dict[str, Any]:
    global _ACTIVE_STAGE, _ACTIVE_TRACKER
    _ACTIVE_TRACKER = None
    _ACTIVE_STAGE = "preflight"
    if torch.cuda.device_count() != 1:
        raise RuntimeError("affine calibration sampling requires exactly one visible GPU")
    if os.environ.get("CLEARML_REQUIRE_ONLINE") != "1":
        raise RuntimeError("affine calibration requires CLEARML_REQUIRE_ONLINE=1")
    torch.set_num_threads(6)
    torch.set_num_interop_threads(1)
    if output.exists():
        raise FileExistsError(f"refusing to reuse affine calibration output {output}")
    verified = {}
    for relative, expected in EXPECTED_INPUT_SHA256.items():
        actual = _sha256(run_dir / relative)
        if actual != expected:
            raise ValueError(f"frozen input SHA-256 mismatch for {relative}")
        verified[relative] = actual
    output.mkdir(parents=True)
    metadata = json.loads((run_dir / "metadata.json").read_text())
    training_config = metadata["training_config"]
    dataset = build_dataset(metadata["data_config"], split="valid")
    panel_contract = validate_panel_indices(len(dataset))
    means = torch.tensor(_repeat_field_stats(dataset.means), dtype=torch.float32)
    stds = torch.tensor(_repeat_field_stats(dataset.stds), dtype=torch.float32)
    tracker = ClearMLTracker(
        project_name="sea_ice_two_stage",
        task_name=f"direct_dynamics_ema6_affine_{output.name}",
        tags=["ema6", "persistence-affine", "post-hoc", "rk4-33", "raw-unclipped", "one-gpu"],
        env_path="/home/.env",
    )
    _ACTIVE_TRACKER = tracker
    _ACTIVE_STAGE = "clearml_connect"
    tracker.connect(
        "protocol",
        {
            **panel_contract,
            "members": MEMBERS,
            "solver": "FP32 TF32-off RK4-33",
            "optimizer_steps": 0,
            "scale_bounds": SCALE_BOUNDS,
            "offset_bounds": OFFSET_BOUNDS,
            "fit": "three-coefficient persistence-centred affine fair-CRPS fit",
        },
    )
    result: dict[str, Any] = {
        "schema_version": "direct_dynamics_ema6_persistence_affine_v1",
        "status": "running",
        "split": "valid",
        "selection_uses_confirmation": False,
        "optimizer_steps": 0,
        "primary_scores_use_raw_unclipped_samples": True,
        "projection_role": "display_only",
        "checkpoint": EMA6_CHECKPOINT,
        "checkpoint_sha256": verified[EMA6_CHECKPOINT],
        "verified_input_sha256": verified,
        "panel_contract": panel_contract,
        "clearml_task_id": str(tracker.task.id),
    }
    cache = {}
    _ACTIVE_STAGE = "calibration_data_prepare"
    calibration_indices = sorted(set(CALIBRATION_INDICES) | {12, 23})
    for position, index in enumerate(calibration_indices):
        print(
            f"[affine] preparing calibration case={position + 1}/{len(calibration_indices)} index={index}",
            flush=True,
        )
        cache[index] = dataset[index]
    result["dataset_sentinel"] = validate_direct_dataset(dataset, item_cache=cache)
    calibration = [(order, index, cache[index]) for order, index in enumerate(CALIBRATION_INDICES)]
    stress_positions = [CALIBRATION_INDICES.index(index) for index in STRESS_DATASET_INDICES]
    old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sampler = None
    try:
        sampler = load_sampler(str(run_dir), EMA6_CHECKPOINT, training_config, device=torch.device("cuda"))
        _ACTIVE_STAGE = "calibration_sampling"
        ensemble, truth, persistence, mask, identities = sample_panel(
            sampler, dataset, calibration, members=MEMBERS, temperature=1.0, steps=SOLVER_STEPS
        )
        _save_raw_bundle(
            output / "calibration_baseline_raw.pt",
            ensemble,
            truth,
            persistence,
            mask,
            identities,
            None,
            verified[EMA6_CHECKPOINT],
            means,
            stds,
            list(range(len(CALIBRATION_INDICES))),
        )
        _ACTIVE_STAGE = "coefficient_fit"
        coefficients = fit_persistence_affine(ensemble, truth, persistence, mask, means, stds)
        candidate = apply_persistence_affine(
            ensemble,
            persistence,
            means,
            stds,
            **{key: coefficients[key] for key in ("scale", "sic_offset", "sit_offset")},
        )
        _save_raw_bundle(
            output / "calibration_candidate_raw.pt",
            candidate,
            truth,
            persistence,
            mask,
            identities,
            coefficients,
            verified[EMA6_CHECKPOINT],
            means,
            stds,
            list(range(len(CALIBRATION_INDICES))),
        )
        _ACTIVE_STAGE = "calibration_scoring"
        baseline_diagnostics = _calibration_diagnostics(ensemble, truth, persistence, mask, stds)
        candidate_diagnostics = _calibration_diagnostics(candidate, truth, persistence, mask, stds)
        comparison = comparison_summary(candidate_diagnostics["scores"], baseline_diagnostics["scores"])
        primary_comparison = primary_score_comparison(
            candidate_diagnostics["primary_standardized_fair_crps"],
            baseline_diagnostics["primary_standardized_fair_crps"],
        )
        raw_tail_gate = tail_gate(candidate_diagnostics["tails"], baseline_diagnostics["tails"])
        stress_baseline = tail_metrics(ensemble[stress_positions], mask[stress_positions])
        stress_candidate = tail_metrics(candidate[stress_positions], mask[stress_positions])
        stress_tail_gate = tail_gate(stress_candidate, stress_baseline)
        extra_gate = _extra_gate(candidate_diagnostics, baseline_diagnostics)
        _require_finite_scalars(
            {
                "baseline": baseline_diagnostics,
                "candidate": candidate_diagnostics,
                "comparison": comparison,
                "primary_comparison": primary_comparison,
                "raw_tail_gate": raw_tail_gate,
                "stress_tail_gate": stress_tail_gate,
                "extra_gate": extra_gate,
            }
        )
        development_passed = (
            _development_candidate_gate(comparison, raw_tail_gate, stress_tail_gate)
            and extra_gate["passed"]
            and primary_comparison["ratio"] < 1.0
        )
        result["coefficients"] = coefficients
        result["calibration"] = {
            "case_identities": identities,
            "baseline": baseline_diagnostics,
            "candidate": candidate_diagnostics,
            "comparison": comparison,
            "primary_comparison": primary_comparison,
            "raw_tail_gate": raw_tail_gate,
            "stress_tail_gate": stress_tail_gate,
            "extra_gate": extra_gate,
            "development_passed": development_passed,
        }
        for label, values in (("baseline", ensemble), ("candidate", candidate)):
            for path in _save_visuals(
                output, f"calibration_{label}_raw", values[:2], truth[:2], persistence[:2], mask[:2]
            ):
                tracker.report_image(
                    "affine/raw_individual_samples", f"calibration/{label}/{path.stem}", path, 0
                )
            for path in _save_visuals(
                output,
                f"calibration_{label}_stress_raw",
                values[stress_positions],
                truth[stress_positions],
                persistence[stress_positions],
                mask[stress_positions],
            ):
                tracker.report_image("affine/raw_stress_samples", f"calibration/{label}/{path.stem}", path, 0)
        _atomic_json(output / "progress.json", result)
        if not development_passed:
            result["status"] = "complete_no_candidate"
            result["decision"] = "keep_ema6_identity"
        else:
            _ACTIVE_STAGE = "confirmation_data_prepare"
            for position, index in enumerate(CONFIRMATION_INDICES):
                print(
                    "[affine] preparing confirmation "
                    f"case={position + 1}/{len(CONFIRMATION_INDICES)} index={index}",
                    flush=True,
                )
                cache[index] = dataset[index]
            confirmation = [
                (100 + order, index, cache[index]) for order, index in enumerate(CONFIRMATION_INDICES)
            ]
            _ACTIVE_STAGE = "confirmation_sampling"
            base, confirm_truth, confirm_persistence, confirm_mask, confirm_identities = sample_panel(
                sampler, dataset, confirmation, members=MEMBERS, temperature=1.0, steps=SOLVER_STEPS
            )
            confirmation_noise_orders = [100 + order for order in range(len(CONFIRMATION_INDICES))]
            _save_raw_bundle(
                output / "confirmation_baseline_raw.pt",
                base,
                confirm_truth,
                confirm_persistence,
                confirm_mask,
                confirm_identities,
                None,
                verified[EMA6_CHECKPOINT],
                means,
                stds,
                confirmation_noise_orders,
            )
            _ACTIVE_STAGE = "confirmation_scoring"
            calibrated = apply_persistence_affine(
                base,
                confirm_persistence,
                means,
                stds,
                **{key: coefficients[key] for key in ("scale", "sic_offset", "sit_offset")},
            )
            _save_raw_bundle(
                output / "confirmation_candidate_raw.pt",
                calibrated,
                confirm_truth,
                confirm_persistence,
                confirm_mask,
                confirm_identities,
                coefficients,
                verified[EMA6_CHECKPOINT],
                means,
                stds,
                confirmation_noise_orders,
            )
            base_diagnostics = _calibration_diagnostics(
                base, confirm_truth, confirm_persistence, confirm_mask, stds
            )
            calibrated_diagnostics = _calibration_diagnostics(
                calibrated, confirm_truth, confirm_persistence, confirm_mask, stds
            )
            confirm_comparison = comparison_summary(
                calibrated_diagnostics["scores"], base_diagnostics["scores"]
            )
            confirm_primary = primary_score_comparison(
                calibrated_diagnostics["primary_standardized_fair_crps"],
                base_diagnostics["primary_standardized_fair_crps"],
            )
            upper95 = bootstrap_J_upper95(calibrated_diagnostics["scores"], base_diagnostics["scores"])
            confirm_tail_gate = tail_gate(calibrated_diagnostics["tails"], base_diagnostics["tails"])
            confirm_extra_gate = _extra_gate(calibrated_diagnostics, base_diagnostics)
            _require_finite_scalars(
                {
                    "baseline": base_diagnostics,
                    "candidate": calibrated_diagnostics,
                    "comparison": confirm_comparison,
                    "primary_comparison": confirm_primary,
                    "paired_date_bootstrap_J_upper95": upper95,
                    "raw_tail_gate": confirm_tail_gate,
                    "extra_gate": confirm_extra_gate,
                }
            )
            failures = []
            if confirm_primary["ratio"] > 0.99:
                failures.append("primary standardized fair-CRPS ratio > 0.99")
            if confirm_primary["paired_date_bootstrap_upper95"] >= 1.0:
                failures.append("primary standardized fair-CRPS bootstrap upper95 >= 1")
            if confirm_comparison["J_mean_fair_crps_ratio"] > 0.99:
                failures.append("mean fair-CRPS ratio > 0.99")
            if upper95 >= 1.0:
                failures.append("paired date-bootstrap upper95 >= 1")
            if any(value > 1.01 for value in confirm_comparison["fair_crps_ratios"].values()):
                failures.append("an individual fair CRPS ratio > 1.01")
            if any(value > 1.01 for value in confirm_comparison["rmse_ratios"].values()):
                failures.append("an individual RMSE ratio > 1.01")
            if confirm_comparison["mean_abs_ssr_error_reduction"] < 0.01:
                failures.append("mean abs SSR error reduction < 0.01")
            if confirm_comparison["mean_rank_tv_difference"] > 0:
                failures.append("mean rank-TV increased")
            if any(value > 0.01 for value in confirm_comparison["rank_tv_differences"].values()):
                failures.append("an individual rank-TV increased by > 0.01")
            if not confirm_tail_gate["passed"]:
                failures.append("confirmation raw-tail gate failed")
            if not confirm_extra_gate["passed"]:
                failures.extend(confirm_extra_gate["failures"])
            result["confirmation"] = {
                "case_identities": confirm_identities,
                "baseline": base_diagnostics,
                "candidate": calibrated_diagnostics,
                "comparison": confirm_comparison,
                "primary_comparison": confirm_primary,
                "paired_date_bootstrap_J_upper95": upper95,
                "raw_tail_gate": confirm_tail_gate,
                "extra_gate": confirm_extra_gate,
                "acceptance_gate": {"passed": not failures, "failures": failures},
            }
            for label, values in (("baseline", base), ("candidate", calibrated)):
                for path in _save_visuals(
                    output,
                    f"confirmation_{label}_raw",
                    values[:2],
                    confirm_truth[:2],
                    confirm_persistence[:2],
                    confirm_mask[:2],
                ):
                    tracker.report_image(
                        "affine/raw_individual_samples", f"confirmation/{label}/{path.stem}", path, 0
                    )
            result["visual_review_status"] = "pending_independent_visual_review"
            result["status"] = (
                "complete_candidate_pending_visual_review" if not failures else "complete_candidate_rejected"
            )
            result["decision"] = "keep_ema6_identity_pending_review"
    finally:
        if sampler is not None:
            del sampler
        torch.cuda.empty_cache()
        torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
        torch.backends.cudnn.allow_tf32 = old_cudnn_tf32
    result["finished_at_unix"] = time.time()
    _ACTIVE_STAGE = "finalization"
    _require_finite_scalars(result)
    _atomic_json(output / "affine_calibration.json", result)
    tracker.upload_artifact("affine_calibration", output / "affine_calibration.json")
    tracker.report_scalar(
        "affine/calibration", "J", result["calibration"]["comparison"]["J_mean_fair_crps_ratio"], 0
    )
    tracker.close()
    _ACTIVE_TRACKER = None
    _ACTIVE_STAGE = "completed"
    _atomic_json(
        output / "run_status.json",
        {"schema_version": result["schema_version"], "status": "completed", "decision": result["decision"]},
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    existed_before = output.exists()
    try:
        print(json.dumps(run(args.run_dir.resolve(), output), indent=2))
    except BaseException as error:
        close_error = None
        if _ACTIVE_TRACKER is not None:
            try:
                _ACTIVE_TRACKER.close()
            except BaseException as tracker_error:
                close_error = {
                    "type": type(tracker_error).__name__,
                    "message": str(tracker_error),
                }
        if not existed_before and output.exists():
            failure = {
                "schema_version": "direct_dynamics_ema6_persistence_affine_v1",
                "status": "failed",
                "stage": _ACTIVE_STAGE,
                "error_type": type(error).__name__,
                "error": str(error),
                "clearml_close_error": close_error,
                "finished_at_unix": time.time(),
            }
            _atomic_json(output / "failure.json", failure)
            _atomic_json(output / "run_status.json", failure)
        raise


if __name__ == "__main__":
    main()
