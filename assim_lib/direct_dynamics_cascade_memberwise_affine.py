"""OOF memberwise censored-affine calibration of the frozen cascade champion."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import traceback
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import minimize

from .clearml_tracking import ClearMLTracker
from .config import load_json
from .direct_dynamics_affine_calibration import boundary_event_metrics, coverage_metrics
from .direct_dynamics_cascade import smooth_right_inverse
from .direct_dynamics_cascade_fine_training import _clean_code_identity
from .direct_dynamics_cascade_paired_evaluation import (
    _roughness,
    _support_metrics,
    _weighted_case_rmse,
    _weighted_fractional_rank,
)
from .transforms import channel_denormalize


CHANNELS = ("d3_sic", "d3_sit", "d6_sic", "d6_sit", "d9_sic", "d9_sit")
FIELDS = ("sic", "sit")


def _atomic_strict_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.incomplete")
    try:
        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _require_finite_tree(value: Any, label: str = "result") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _require_finite_tree(child, f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _require_finite_tree(child, f"{label}[{index}]")
    elif isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        raise FloatingPointError(f"nonfinite value at {label}")


def _validate_inputs(raw: torch.Tensor, truth: torch.Tensor, valid: torch.Tensor) -> None:
    if raw.shape != (12, 8, 6, 320, 256):
        raise ValueError(f"unexpected ensemble shape: {tuple(raw.shape)}")
    if truth.shape != (12, 6, 320, 256) or valid.shape != (12, 1, 320, 256):
        raise ValueError("unexpected truth or mask shape")
    if not torch.isfinite(raw).all() or not torch.isfinite(truth).all() or not torch.isfinite(valid).all():
        raise FloatingPointError("frozen tensors contain NaN/Inf")
    if not torch.all((valid == 0) | (valid == 1)):
        raise ValueError("valid mask must be binary")
    if torch.any(valid.sum((1, 2, 3)) == 0):
        raise ValueError("every case must have nonempty ocean support")
    if not torch.equal(valid, valid[:1].expand_as(valid)):
        raise ValueError("the frozen ocean support must be static")


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


def validate_folds(folds: list[list[int]], case_count: int) -> None:
    flattened = [index for fold in folds for index in fold]
    if len(folds) != 3 or any(len(fold) != 4 for fold in folds):
        raise ValueError("calibration requires three four-date folds")
    if sorted(flattened) != list(range(case_count)) or len(set(flattened)) != case_count:
        raise ValueError("folds must partition every frozen case exactly once")


def apply_memberwise(
    ensemble: torch.Tensor, field_stds: torch.Tensor, parameters: dict[str, float]
) -> torch.Tensor:
    """Apply a fixed map to each member without using the other members."""
    if ensemble.ndim != 5 or ensemble.shape[2] != 6:
        raise ValueError("memberwise calibration expects [B,M,6,H,W]")
    if tuple(field_stds.shape) != (2,) or torch.any(field_stds <= 0):
        raise ValueError("field standard deviations must be positive SIC/SIT values")
    output = ensemble.clone()
    for offset, field in enumerate(FIELDS):
        scale = float(parameters[f"{field}_scale"])
        standardized_offset = float(parameters[f"{field}_offset"])
        if not math.isfinite(scale) or not math.isfinite(standardized_offset) or scale <= 0:
            raise ValueError("affine parameters must be finite with positive scales")
        transformed = scale * ensemble[:, :, offset::2] + field_stds[offset] * standardized_offset
        output[:, :, offset::2] = (
            transformed.clamp(0.0, 1.0) if field == "sic" else transformed.clamp_min(0.0)
        )
    if not torch.isfinite(output).all():
        raise FloatingPointError("memberwise calibration produced NaN/Inf")
    return output


def fair_crps_cases(
    members: torch.Tensor, truth: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    """Case-equal fair CRPS for one channel, retaining the date axis."""
    if members.ndim != 5 or members.shape[2] != 1 or truth.shape != members[:, 0].shape:
        raise ValueError("fair CRPS requires [B,M,1,H,W] and [B,1,H,W]")
    member_count = members.shape[1]
    if member_count < 2 or valid.shape != (members.shape[0], 1, *members.shape[-2:]):
        raise ValueError("invalid fair CRPS member count or mask")
    members = members.double()
    truth = truth.double()
    weight = valid.double()
    accuracy = (members - truth[:, None]).abs().mean(1)
    ordered = members.sort(dim=1).values
    coefficients = torch.arange(member_count, dtype=torch.float64, device=members.device)
    coefficients = 2 * coefficients - member_count + 1
    pair_sum = (ordered * coefficients.reshape(1, member_count, 1, 1, 1)).sum(1)
    pointwise = accuracy - pair_sum / (member_count * (member_count - 1))
    return (pointwise * weight).sum((1, 2, 3)) / weight.sum((1, 2, 3))


def _fit_arrays(
    ensemble: torch.Tensor,
    truth: torch.Tensor,
    valid: torch.Tensor,
    train_indices: list[int],
    field_offset: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = valid[train_indices, 0] > 0
    if not torch.equal(mask, mask[:1].expand_as(mask)):
        raise ValueError("the frozen ocean support must be static")
    values = ensemble[train_indices, :, field_offset::2]
    targets = truth[train_indices, field_offset::2]
    expanded = mask[:, None].expand(-1, 3, -1, -1)
    point_values = values.permute(0, 2, 3, 4, 1)[expanded].contiguous()
    point_targets = targets[expanded].contiguous()
    return point_values, point_targets


def _field_objective(
    values: torch.Tensor,
    targets: torch.Tensor,
    std: float,
    field: str,
    scale: float,
    offset: float,
    ordered_values: torch.Tensor | None = None,
) -> float:
    transformed = scale * values + std * offset
    transformed = transformed.clamp(0.0, 1.0) if field == "sic" else transformed.clamp_min(0.0)
    accuracy = (transformed - targets[:, None]).abs().mean()
    if ordered_values is None:
        ordered = transformed.sort(dim=1).values
    else:
        ordered = scale * ordered_values + std * offset
        ordered = ordered.clamp(0.0, 1.0) if field == "sic" else ordered.clamp_min(0.0)
    count = values.shape[1]
    coefficients = torch.arange(count, dtype=transformed.dtype, device=transformed.device)
    coefficients = 2 * coefficients - count + 1
    pair = (ordered * coefficients).sum(1).mean() / (count * (count - 1))
    return float(((accuracy - pair) / std).item())


def _fit_field(
    values: torch.Tensor,
    targets: torch.Tensor,
    std: float,
    field: str,
    spec: dict[str, Any],
) -> dict[str, Any]:
    scale_bounds = tuple(float(value) for value in spec["scale_bounds"])
    offset_bounds = tuple(float(value) for value in spec["offset_bounds_standardized"])
    stride = int(spec["coarse_pixel_stride"])
    preview_values, preview_targets = values[::stride], targets[::stride]
    ordered_values = values.sort(dim=1).values
    preview_ordered = ordered_values[::stride]
    scales = np.arange(scale_bounds[0], scale_bounds[1] + 0.5 * spec["coarse_scale_step"], spec["coarse_scale_step"])
    offsets = np.arange(offset_bounds[0], offset_bounds[1] + 0.5 * spec["coarse_offset_step"], spec["coarse_offset_step"])
    preview = []
    for scale in scales:
        for offset in offsets:
            preview.append((
                _field_objective(preview_values, preview_targets, std, field, float(scale), float(offset), preview_ordered),
                float(scale), float(offset),
            ))
    starts = sorted(preview)[:3]
    exact_cache: dict[tuple[float, float], float] = {}

    def objective(vector: np.ndarray) -> float:
        scale = float(np.clip(vector[0], *scale_bounds))
        offset = float(np.clip(vector[1], *offset_bounds))
        key = (round(scale, 8), round(offset, 8))
        if key not in exact_cache:
            exact_cache[key] = _field_objective(values, targets, std, field, scale, offset, ordered_values)
        return exact_cache[key]

    candidates = []
    for _, scale, offset in starts:
        fitted = minimize(
            objective,
            np.array([scale, offset]),
            method="Powell",
            bounds=(scale_bounds, offset_bounds),
            options={
                "maxfev": int(spec["powell_maxfev"]),
                "xtol": float(spec["powell_xtol"]),
                "ftol": float(spec["powell_ftol"]),
            },
        )
        candidates.append((objective(fitted.x), float(fitted.x[0]), float(fitted.x[1]), bool(fitted.success)))
    converged = [candidate for candidate in candidates if candidate[3]]
    if converged:
        score, scale, offset, success = min(converged)
    else:
        _, scale, offset = min(starts)
        score, success = objective(np.array([scale, offset])), False
    tolerance = 2 * max(float(spec["powell_xtol"]), 1e-6)
    boundary = (
        abs(scale - scale_bounds[0]) <= tolerance
        or abs(scale - scale_bounds[1]) <= tolerance
        or abs(offset - offset_bounds[0]) <= tolerance
        or abs(offset - offset_bounds[1]) <= tolerance
    )
    return {
        "scale": scale,
        "offset": offset,
        "objective": score,
        "optimizer_success": success,
        "optimizer_attempts": len(candidates),
        "optimum_on_boundary": boundary,
        "exact_evaluations": len(exact_cache),
        "preview_evaluations": len(preview),
    }


def fit_oof(
    raw: torch.Tensor,
    truth: torch.Tensor,
    valid: torch.Tensor,
    field_stds: torch.Tensor,
    folds: list[list[int]],
    fit_spec: dict[str, Any],
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    candidate = torch.empty_like(raw)
    rows = []
    all_indices = set(range(raw.shape[0]))
    for fold_number, heldout in enumerate(folds):
        train = sorted(all_indices - set(heldout))
        parameters: dict[str, float] = {}
        field_fits = {}
        for offset, field in enumerate(FIELDS):
            values, targets = _fit_arrays(raw, truth, valid, train, offset)
            fitted = _fit_field(values, targets, float(field_stds[offset]), field, fit_spec)
            parameters[f"{field}_scale"] = fitted["scale"]
            parameters[f"{field}_offset"] = fitted["offset"]
            field_fits[field] = fitted
        candidate[heldout] = apply_memberwise(raw[heldout], field_stds, parameters)
        rows.append({
            "fold": fold_number,
            "fit_indices": train,
            "heldout_indices": heldout,
            "parameters": parameters,
            "field_fits": field_fits,
        })
    return candidate, rows


def _temporal_metrics(ensemble: torch.Tensor, truth: torch.Tensor, valid: torch.Tensor) -> dict[str, float]:
    result = {}
    for start, label in ((0, "d3_to_d6"), (2, "d6_to_d9")):
        for offset, field in enumerate(FIELDS):
            channel = start + offset
            member_increment = ensemble[:, :, channel + 2 : channel + 3] - ensemble[:, :, channel : channel + 1]
            truth_increment = truth[:, channel + 2 : channel + 3] - truth[:, channel : channel + 1]
            rmse, _ = _weighted_case_rmse(member_increment.mean(1), truth_increment, valid)
            crps = fair_crps_cases(member_increment, truth_increment, valid).mean()
            result[f"{label}_{field}_mean_rmse"] = rmse
            result[f"{label}_{field}_fair_crps"] = float(crps)
    return result


def _cross_field_inconsistency(ensemble: torch.Tensor, valid: torch.Tensor) -> dict[str, float]:
    result = {}
    mask = valid[:, None] > 0
    for lead_index, lead in enumerate((3, 6, 9)):
        sic = ensemble[:, :, 2 * lead_index : 2 * lead_index + 1]
        sit = ensemble[:, :, 2 * lead_index + 1 : 2 * lead_index + 2]
        bad = (sic <= 0.01) & (sit > 0.01) & mask
        per_case = bad.double().sum((1, 2, 3, 4)) / mask.expand_as(bad).double().sum((1, 2, 3, 4))
        result[f"d{lead}"] = float(per_case.mean())
    return result


def _exact_atom_metrics(
    ensemble: torch.Tensor, truth: torch.Tensor, valid: torch.Tensor
) -> dict[str, dict[str, float]]:
    """Exact physical atoms after the support map, scored as ensemble events."""
    result: dict[str, dict[str, float]] = {}
    ocean = valid > 0
    for channel, name in enumerate(CHANNELS):
        members = ensemble[:, :, channel : channel + 1]
        target = truth[:, channel : channel + 1]
        atoms = (("zero", 0.0), ("one", 1.0)) if channel % 2 == 0 else (("zero", 0.0),)
        rows = {}
        for atom_name, atom in atoms:
            probability = (members == atom).double().mean(1)
            observed = (target == atom).double()
            per_case = [
                float((probability[case][ocean[case]] - observed[case][ocean[case]]).square().mean())
                for case in range(len(ensemble))
            ]
            rows[atom_name] = {
                "brier": float(np.mean(per_case)),
                "forecast_rate": float(probability[ocean].mean()),
                "truth_rate": float(observed[ocean].mean()),
            }
        if channel % 2 == 1:
            truth_open = (target == 0) & ocean
            per_case_zero = []
            for case in range(len(ensemble)):
                selected = members[case][truth_open[case][None].expand_as(members[case])]
                if selected.numel():
                    positive = selected.clamp_min(0)
                    per_case_zero.append({
                        "exact_zero_fraction": float((selected == 0).double().mean()),
                        "mean_positive_sit": float(positive.mean()),
                        "p95_positive_sit": float(torch.quantile(positive, 0.95)),
                        "fraction_gt_0p01": float((positive > 0.01).double().mean()),
                    })
                else:
                    per_case_zero.append(None)
            nonempty = [value for value in per_case_zero if value is not None]
            rows["open_water_exact_zero"] = {
                **({
                    key: float(np.mean([value[key] for value in nonempty]))
                    for key in ("exact_zero_fraction", "mean_positive_sit", "p95_positive_sit", "fraction_gt_0p01")
                } if nonempty else {
                    "exact_zero_fraction": None, "mean_positive_sit": None,
                    "p95_positive_sit": None, "fraction_gt_0p01": None,
                }),
                "per_case": per_case_zero,
            }
        result[name] = rows
    return result


def diagnostics(
    ensemble: torch.Tensor,
    truth: torch.Tensor,
    valid: torch.Tensor,
    channel_stds: torch.Tensor,
) -> dict[str, Any]:
    standardized_score = []
    outputs = {}
    for channel, name in enumerate(CHANNELS):
        members = ensemble[:, :, channel : channel + 1]
        target = truth[:, channel : channel + 1]
        crps_cases = fair_crps_cases(members / channel_stds[channel], target / channel_stds[channel], valid)
        standardized_score.append(crps_cases)
        mean_rmse, case_rmse = _weighted_case_rmse(members.mean(1), target, valid)
        spread = members.std(1, unbiased=True)
        spread_rms, _ = _weighted_case_rmse(spread, torch.zeros_like(spread), valid)
        adjusted_ssr = math.sqrt(1.0 + 1.0 / members.shape[1]) * spread_rms / max(mean_rmse, 1e-12)
        outputs[name] = {
            "standardized_fair_crps": float(crps_cases.mean()),
            "standardized_fair_crps_cases": crps_cases.tolist(),
            "ensemble_mean_rmse": mean_rmse,
            "ensemble_mean_rmse_cases": case_rmse,
            "spread": spread_rms,
            "adjusted_ssr": adjusted_ssr,
            **_weighted_fractional_rank(members, target, valid),
            "member_roughness": _roughness(members, valid),
            "truth_roughness": _roughness(target[:, None], valid),
            "support": _support_metrics(members, valid, "sic" if channel % 2 == 0 else "sit"),
        }
    score_cases = torch.stack(standardized_score, dim=1)
    coverages = coverage_metrics(ensemble, truth, valid)
    return {
        "primary_standardized_fair_crps": float(score_cases.mean()),
        "primary_standardized_fair_crps_cases_outputs": score_cases.tolist(),
        "mean_adjusted_ssr_absolute_error": float(np.mean([abs(v["adjusted_ssr"] - 1.0) for v in outputs.values()])),
        "mean_rank_tv": float(np.mean([v["rank_tv_to_uniform"] for v in outputs.values()])),
        "joint_normalized_energy": _joint_energy(ensemble, truth, valid, channel_stds),
        "coverage": coverages,
        "mean_coverage_absolute_error": float(np.mean([
            interval["absolute_error"]
            for output in coverages.values()
            for interval in output.values()
        ])),
        "boundary_events": boundary_event_metrics(ensemble, truth, valid),
        "exact_atoms": _exact_atom_metrics(ensemble, truth, valid),
        "temporal_increments": _temporal_metrics(ensemble, truth, valid),
        "cross_field_inconsistency": _cross_field_inconsistency(ensemble, valid),
        "outputs": outputs,
    }


def _joint_energy(
    ensemble: torch.Tensor, truth: torch.Tensor, valid: torch.Tensor, stds: torch.Tensor
) -> float:
    scaled_members = ensemble.double() / stds.reshape(1, 1, 6, 1, 1)
    scaled_truth = truth.double() / stds.reshape(1, 6, 1, 1)
    scores = []
    for case in range(ensemble.shape[0]):
        selected = valid[case].expand(6, -1, -1) > 0
        members = scaled_members[case, :, selected]
        target = scaled_truth[case, selected]
        norm = math.sqrt(target.numel())
        accuracy = torch.linalg.vector_norm(members - target, dim=1).mean() / norm
        pair = ensemble.new_zeros((), dtype=torch.float64)
        for left in range(ensemble.shape[1]):
            for right in range(left + 1, ensemble.shape[1]):
                pair += torch.linalg.vector_norm(members[left] - members[right]) / norm
        scores.append(accuracy - pair / (ensemble.shape[1] * (ensemble.shape[1] - 1)))
    return float(torch.stack(scores).mean())


def _paired_delta(candidate: list[list[float]], control: list[list[float]], spec: dict[str, Any], seed_offset: int) -> dict[str, float]:
    left = torch.tensor(candidate, dtype=torch.float64)
    right = torch.tensor(control, dtype=torch.float64)
    delta = left.mean(1) - right.mean(1)
    generator = torch.Generator().manual_seed(int(spec["bootstrap_seed"]) + seed_offset)
    indices = torch.randint(len(delta), (int(spec["bootstrap_draws"]), len(delta)), generator=generator)
    sampled = delta[indices].mean(1)
    return {
        "estimate": float(delta.mean()),
        "paired_date_bootstrap_95_ci_low": float(torch.quantile(sampled, 0.025)),
        "paired_date_bootstrap_95_ci_high": float(torch.quantile(sampled, 0.975)),
    }


def _positive_ratio(numerator: float, denominator: float, label: str) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator <= 0:
        raise FloatingPointError(f"invalid ratio inputs for {label}")
    return numerator / denominator


def compare(candidate: dict[str, Any], control: dict[str, Any], gate: dict[str, Any], seed_offset: int) -> dict[str, Any]:
    comparison = {
        "primary_ratio": _positive_ratio(candidate["primary_standardized_fair_crps"], control["primary_standardized_fair_crps"], "primary CRPS"),
        "joint_energy_ratio": _positive_ratio(candidate["joint_normalized_energy"], control["joint_normalized_energy"], "joint energy"),
        "ssr_error_reduction": 1.0 - _positive_ratio(candidate["mean_adjusted_ssr_absolute_error"], control["mean_adjusted_ssr_absolute_error"], "SSR error"),
        "mean_rank_tv_difference": candidate["mean_rank_tv"] - control["mean_rank_tv"],
        "mean_coverage_error_difference": candidate["mean_coverage_absolute_error"] - control["mean_coverage_absolute_error"],
        "paired_primary_delta": _paired_delta(
            candidate["primary_standardized_fair_crps_cases_outputs"],
            control["primary_standardized_fair_crps_cases_outputs"], gate, seed_offset,
        ),
        "outputs": {},
    }
    for name in CHANNELS:
        cand, base = candidate["outputs"][name], control["outputs"][name]
        comparison["outputs"][name] = {
            "fair_crps_ratio": _positive_ratio(cand["standardized_fair_crps"], base["standardized_fair_crps"], f"{name} CRPS"),
            "rmse_ratio": _positive_ratio(cand["ensemble_mean_rmse"], base["ensemble_mean_rmse"], f"{name} RMSE"),
            "rank_tv_difference": cand["rank_tv_to_uniform"] - base["rank_tv_to_uniform"],
            "roughness_ratio": _positive_ratio(cand["member_roughness"], base["member_roughness"], f"{name} roughness"),
        }
    return comparison


def _haze_failures(row: dict[str, Any], base: dict[str, Any], spec: dict[str, Any], prefix: str, exact: bool) -> list[str]:
    failures = []
    checks = [
        ("mean_positive_sit", "maximum", "haze_mean_tolerance_m"),
        ("p95_positive_sit", "maximum", "haze_p95_tolerance_m"),
        ("fraction_gt_0p01", "maximum", "haze_fraction_tolerance"),
    ]
    if exact:
        checks.insert(0, ("exact_zero_fraction", "minimum", "exact_zero_fraction_tolerance"))
    for metric, direction, tolerance_key in checks:
        value, reference = row[metric], base[metric]
        if value is None or reference is None:
            if value != reference:
                failures.append(f"{prefix} {metric} selector mismatch")
        elif direction == "minimum" and value + float(spec[tolerance_key]) < reference:
            failures.append(f"{prefix} {metric} decreased")
        elif direction == "maximum" and value > reference + float(spec[tolerance_key]):
            failures.append(f"{prefix} {metric} increased")
    return failures


def _gate(
    candidate: dict[str, Any], raw: dict[str, Any], support: dict[str, Any],
    versus_raw: dict[str, Any], versus_support: dict[str, Any], folds: list[dict[str, Any]], spec: dict[str, Any]
) -> dict[str, Any]:
    failures = []
    for label, comparison in (("raw", versus_raw), ("support", versus_support)):
        if comparison["primary_ratio"] > 1.0 - float(spec["minimum_crps_improvement"]):
            failures.append(f"primary CRPS improvement versus {label} is below 1%")
        if comparison["paired_primary_delta"]["paired_date_bootstrap_95_ci_high"] >= 0:
            failures.append(f"paired-date primary CRPS CI versus {label} includes zero")
        if comparison["joint_energy_ratio"] > 1.0 + float(spec["per_output_score_tolerance"]):
            failures.append(f"joint energy worsened versus {label}")
        if comparison["ssr_error_reduction"] < float(spec["minimum_ssr_error_reduction"]):
            failures.append(f"adjusted SSR error reduction versus {label} is below 5%")
        if comparison["mean_rank_tv_difference"] > 0:
            failures.append(f"mean rank TV worsened versus {label}")
        if comparison["mean_coverage_error_difference"] > 0:
            failures.append(f"mean interval coverage error worsened versus {label}")
        for name, row in comparison["outputs"].items():
            tolerance = 1.0 + float(spec["per_output_score_tolerance"])
            if row["fair_crps_ratio"] > tolerance or row["rmse_ratio"] > tolerance:
                failures.append(f"{name} score worsened versus {label}")
            if row["rank_tv_difference"] > float(spec["per_output_rank_tv_tolerance"]):
                failures.append(f"{name} rank TV worsened versus {label}")
            if label == "support" and row["roughness_ratio"] > 1.0 + float(spec["roughness_tolerance"]):
                failures.append(f"{name} member roughness worsened versus support")
    if any(not field["optimizer_success"] for fold in folds for field in fold["field_fits"].values()):
        failures.append("at least one fold fit did not converge")
    if any(field["optimum_on_boundary"] for fold in folds for field in fold["field_fits"].values()):
        failures.append("at least one fold optimum reached a frozen parameter boundary")
    for name in CHANNELS:
        for event, row in candidate["boundary_events"][name].items():
            if event == "open_water_haze":
                base = support["boundary_events"][name][event]
                failures.extend(_haze_failures(row, base, spec, f"{name} open-water", False))
            else:
                base = support["boundary_events"][name][event]
                if row["brier"] > 1.01 * base["brier"] + 1e-8:
                    failures.append(f"{name} {event} Brier worsened")
                if row["reliability_ece"] > 1.01 * base["reliability_ece"] + 1e-8:
                    failures.append(f"{name} {event} reliability worsened")
    for name in CHANNELS:
        for atom, row in candidate["exact_atoms"][name].items():
            if atom == "open_water_exact_zero":
                base = support["exact_atoms"][name][atom]
                failures.extend(_haze_failures(row, base, spec, f"{name} exact-zero open-water", True))
                continue
            for label, control in (("raw", raw), ("support", support)):
                base = control["exact_atoms"][name][atom]
                if row["brier"] > 1.01 * base["brier"] + 1e-8:
                    failures.append(f"{name} exact-{atom} Brier worsened versus {label}")
    for key, value in candidate["temporal_increments"].items():
        if value > 1.01 * support["temporal_increments"][key] + 1e-12:
            failures.append(f"temporal diagnostic worsened: {key}")
    for key, value in candidate["cross_field_inconsistency"].items():
        if value > support["cross_field_inconsistency"][key] + 1e-12:
            failures.append(f"cross-field inconsistency worsened: {key}")
    support_passed = all(
        row["support"]["global_max_excess"] == 0
        for row in candidate["outputs"].values()
    )
    if not support_passed:
        failures.append("candidate violates exact physical support")
    return {"passed_pending_visual_review": not failures, "failures": failures, "exact_support_passed": support_passed}


def _save_panels(
    output: Path, case_ids: list[str], truth: torch.Tensor,
    raw: torch.Tensor, support: torch.Tensor, candidate: torch.Tensor, valid: torch.Tensor,
) -> list[str]:
    directory = output / "visual_qc"
    directory.mkdir()
    paths = []
    for case, case_id in enumerate(case_ids):
        figure, axes = plt.subplots(18, 9, figsize=(20, 32), squeeze=False)
        for channel, name in enumerate(CHANNELS):
            field = "sic" if channel % 2 == 0 else "sit"
            low, high = (0.0, 1.0) if field == "sic" else (0.0, 3.5)
            for method_index, (method, values) in enumerate((
                ("raw", raw), ("support", support), ("calibrated", candidate)
            )):
                row = 3 * channel + method_index
                panels = [truth[case, channel], *values[case, :, channel]]
                for column, value in enumerate(panels):
                    shown = value.clone(); shown[valid[case, 0] <= 0] = torch.nan
                    axes[row, column].imshow(shown, origin="upper", cmap="viridis", vmin=low, vmax=high)
                    axes[row, column].set_xticks([]); axes[row, column].set_yticks([])
                    if row == 0: axes[row, column].set_title("truth" if column == 0 else f"member {column - 1}")
                    if column == 0: axes[row, column].set_ylabel(f"{name}\n{method}")
        figure.suptitle(f"Matched memberwise calibration; {case_id}")
        figure.tight_layout(rect=(0, 0, 1, .995))
        path = directory / f"case{case:02d}_all_outputs_all_members.png"
        figure.savefig(path, dpi=90); plt.close(figure); paths.append(str(path))
    return paths


def _validate_source_contract(payload: dict[str, Any], config: dict[str, Any], repository: Path) -> None:
    source = config["source"]
    for key in ("schema_version", "candidate", "replay_commit", "fine_training_commit", "fine_checkpoint_sha256", "solver"):
        if payload.get(key) != source[key]:
            raise ValueError(f"frozen source identity mismatch: {key}")
    if list(payload.get("case_ids", ())) != list(config["case_ids"]):
        raise ValueError("frozen case/date window differs from protocol")
    if tuple(payload.get("member_indices", ())) != tuple(range(8)):
        raise ValueError("source is not the frozen eight-member ensemble")
    expected_shapes = {
        "coarse": (12, 8, 6, 160, 128),
        "residual": (12, 8, 6, 320, 256),
        "truth": (12, 6, 320, 256),
        "valid": (12, 1, 320, 256),
    }
    for key, shape in expected_shapes.items():
        value = payload.get(key)
        if not torch.is_tensor(value) or tuple(value.shape) != shape:
            raise ValueError(f"frozen tensor contract mismatch: {key}")
    normalization_source = repository / config["normalization"]["source_path"]
    if _sha256(normalization_source) != config["normalization"]["source_sha256"]:
        raise ValueError("normalization source manifest differs")


def _finalize_failed_tracker(tracker: Any, error: BaseException, failure_path: Path) -> None:
    try:
        tracker.task.mark_failed(status_reason=type(error).__name__, status_message=str(error)[:1000])
    except Exception:
        pass
    try:
        tracker.upload_artifact("memberwise_affine_failure", failure_path)
    except Exception:
        pass
    try:
        tracker.close()
    except Exception:
        pass


def run(config_path: Path, output: Path) -> dict[str, Any]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.device_count() != 0:
        raise RuntimeError("memberwise calibration gate must be CPU-only")
    if os.environ.get("CLEARML_REQUIRE_ONLINE") != "1":
        raise RuntimeError("memberwise calibration gate requires online ClearML")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to replace {output}")
    config = load_json(config_path)
    output.mkdir(parents=True)
    tracker = None
    previous_handler = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(TimeoutError("received SIGTERM")))
    try:
        torch.set_num_threads(6); torch.set_num_interop_threads(1)
        tracker = ClearMLTracker(config["project_name"], config["task_name"], tags=config["clearml"]["tags"], env_path=config["clearml"]["env_path"])
        tracker.connect("memberwise_affine_protocol", config)
        _atomic_strict_json(output / "status.json", {"status": "running", "clearml_task_id": str(tracker.task.id)})
        repository = Path(__file__).resolve().parents[1]
        source = Path(config["source"]["path"])
        if not source.is_file() or _sha256(source) != config["source"]["sha256"]:
            raise ValueError("frozen champion artifact differs")
        identity = _clean_code_identity(repository)
        payload = torch.load(source, map_location="cpu", weights_only=True)
        _validate_source_contract(payload, config, repository)
        raw_normalized = payload["residual"].float()
        valid = payload["valid"].float()
        mask_members = valid[:, None].expand(-1, 8, -1, -1, -1).flatten(0, 1)
        raw_normalized = raw_normalized + smooth_right_inverse(
            payload["coarse"].float().flatten(0, 1), mask_members
        ).unflatten(0, (12, 8))
        truth_normalized = payload["truth"].float()
        field_means = torch.tensor(config["normalization"]["means"], dtype=torch.float32)
        field_stds = torch.tensor(config["normalization"]["stds"], dtype=torch.float32)
        if not torch.isfinite(field_means).all() or not torch.isfinite(field_stds).all() or torch.any(field_stds <= 0):
            raise ValueError("normalization parameters must be finite with positive stds")
        channel_means = field_means.repeat(3)
        channel_stds = field_stds.repeat(3)
        raw = channel_denormalize(raw_normalized.flatten(0, 1), channel_means, channel_stds).unflatten(0, (12, 8))
        truth = channel_denormalize(truth_normalized, channel_means, channel_stds)
        _validate_inputs(raw, truth, valid)
        folds = config["folds"]
        validate_folds(folds, 12)
        support_parameters = {"sic_scale": 1.0, "sic_offset": 0.0, "sit_scale": 1.0, "sit_offset": 0.0}
        support = apply_memberwise(raw, field_stds, support_parameters)
        candidate, fold_fits = fit_oof(raw, truth, valid, field_stds, folds, config["fit"])
        provenance = {
            "source": config["source"], "normalization": config["normalization"],
            "case_ids": list(payload["case_ids"]), "folds": folds, "code_identity": identity,
        }
        tensor_hashes = {
            "support_only.pt": _atomic_torch_save({"ensemble": support, "parameters": support_parameters, **provenance}, output / "support_only.pt"),
            "oof_calibrated.pt": _atomic_torch_save({"ensemble": candidate, "fold_fits": fold_fits, **provenance}, output / "oof_calibrated.pt"),
        }
        _atomic_strict_json(output / "fold_fits.json", {"status": "fit_complete", "fold_fits": fold_fits, **provenance})
        raw_diagnostics = diagnostics(raw, truth, valid, channel_stds)
        support_diagnostics = diagnostics(support, truth, valid, channel_stds)
        candidate_diagnostics = diagnostics(candidate, truth, valid, channel_stds)
        _require_finite_tree({"raw": raw_diagnostics, "support": support_diagnostics, "candidate": candidate_diagnostics})
        versus_raw = compare(candidate_diagnostics, raw_diagnostics, config["gate"], 0)
        versus_support = compare(candidate_diagnostics, support_diagnostics, config["gate"], 1)
        _require_finite_tree({"versus_raw": versus_raw, "versus_support": versus_support})
        gate = _gate(candidate_diagnostics, raw_diagnostics, support_diagnostics, versus_raw, versus_support, fold_fits, config["gate"])
        result = {
            "status": "scored_pending_visual_review", "training_performed": False,
            **provenance, "config_sha256": _sha256(config_path), "fold_fits": fold_fits,
            "controls": {"raw": raw_diagnostics, "support_only": support_diagnostics},
            "candidate": candidate_diagnostics,
            "comparison": {"versus_raw": versus_raw, "versus_support_only": versus_support},
            "gate": {**gate, "training_or_replacement_permitted": False},
            "tensor_sha256": tensor_hashes, "figures": [], "clearml_task_id": str(tracker.task.id),
        }
        _atomic_strict_json(output / "memberwise_affine_gate.json", result)
        tracker.upload_artifact("memberwise_affine_gate_scored", output / "memberwise_affine_gate.json")
        figures = _save_panels(output, list(payload["case_ids"]), truth, raw, support, candidate, valid)
        for path in figures:
            tracker.report_image("memberwise_affine_visual_qc", Path(path).stem, Path(path), 0)
        tracker.close(); tracker = None
        result["status"] = "complete_pending_astra_review"
        result["figures"] = figures
        _atomic_strict_json(output / "memberwise_affine_gate.json", result)
        _atomic_strict_json(output / "status.json", {"status": result["status"], "clearml_task_id": result["clearml_task_id"]})
        return result
    except BaseException as error:
        failure = {"status": "failed", "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()}
        _atomic_strict_json(output / "failure.json", failure)
        _atomic_strict_json(output / "status.json", {"status": "failed", "error_type": type(error).__name__, "error": str(error)})
        if tracker is not None:
            _finalize_failed_tracker(tracker, error, output / "failure.json")
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.config.resolve(), arguments.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
