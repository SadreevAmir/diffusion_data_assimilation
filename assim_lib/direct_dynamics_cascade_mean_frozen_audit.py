"""One-shot CPU-only audit of the frozen coarse conditional-mean checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import signal
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch

from .clearml_tracking import ClearMLTracker
from .config import TrainingConfig, load_json, merge_config_overrides
from .data import build_dataset
from .direct_dynamics_cascade import masked_block_average
from .direct_dynamics_cascade_coarse import coarse_target
from .direct_dynamics_cascade_coarse_mean import CoarseConditionalMeanPredictor
from .direct_dynamics_cascade_coarse_residual import coarse_persistence_from_condition
from .direct_dynamics_cascade_residual_stats import _tensor_sha256 as _source_tensor_sha256
from .direct_dynamics_cascade_fine_training import (
    _calendar_inventory,
    _clean_code_identity,
    _fine_collate,
    _indices_sha256,
)
from .direct_dynamics_training import DIRECT_LEADS, _repeat_field_stats, validate_direct_dataset
from .model_io import build_unet
from .trainer import _atomic_json
from .transforms import channel_denormalize


UPDATES = (512, 1024, 1536, 2048)
OUTPUT_KEYS = tuple(
    f"d{lead}_{field}" for lead in DIRECT_LEADS for field in ("sic", "sit")
)
SAVED_EQUAL_TENSORS = (
    "structured_conditioning",
    "fine_valid_mask",
    "coarse_truth_physical",
    "coarse_persistence_physical",
    "coarse_active_mask",
    "coarse_ocean_fraction",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _atomic_torch_save(payload: Any, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.incomplete")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _require_cpu_envelope() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("frozen mean audit requires CUDA_VISIBLE_DEVICES='' exactly")
    if torch.cuda.device_count() != 0 or torch.cuda.is_available():
        raise RuntimeError("frozen mean audit must not see a CUDA device")
    if os.environ.get("CLEARML_REQUIRE_ONLINE") != "1":
        raise RuntimeError("frozen mean audit requires online ClearML")
    torch.set_num_threads(6)
    torch.set_num_interop_threads(1)


def _verify_hash_inventory(root: Path, inventory: dict[str, str]) -> None:
    for relative, expected in inventory.items():
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts or len(expected) != 64:
            raise ValueError(f"unsafe or malformed frozen binding: {relative}")
        path = root / candidate
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen source: {path}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"frozen source hash mismatch: {relative}: {actual}")


def _verify_checkpoint_identity(run_dir: Path) -> dict[str, Any]:
    best = torch.load(run_dir / "best_model.pth", map_location="cpu", weights_only=True)
    raw = torch.load(run_dir / "coarse_update_0512.pth", map_location="cpu", weights_only=True)
    if list(best) != list(raw):
        raise ValueError("best_model and raw update512 state keys differ")
    for key in raw:
        left, right = best[key], raw[key]
        if left.shape != right.shape or left.dtype != right.dtype or not torch.equal(left, right):
            raise ValueError(f"best_model and raw update512 differ at {key}")
    return {
        "tensor_count": len(raw),
        "parameter_and_buffer_values_bitwise_equal": True,
        "best_model_file_sha256": _sha256(run_dir / "best_model.pth"),
        "raw_update512_file_sha256": _sha256(run_dir / "coarse_update_0512.pth"),
    }


def _weighted_cases(
    estimate: torch.Tensor, truth: torch.Tensor, fraction: torch.Tensor
) -> dict[str, Any]:
    if estimate.shape != truth.shape or estimate.ndim != 4 or estimate.shape[1] != 1:
        raise ValueError("weighted metrics require matching [B,1,H,W] fields")
    weight = fraction.to(torch.float64)
    error = estimate.to(torch.float64) - truth.to(torch.float64)
    denominator = weight.sum((1, 2, 3))
    if torch.any(denominator <= 0):
        raise ValueError("every audit case must contain active ocean")
    mse = (error.square() * weight).sum((1, 2, 3)) / denominator
    mae = (error.abs() * weight).sum((1, 2, 3)) / denominator
    bias = (error * weight).sum((1, 2, 3)) / denominator
    return {
        "aggregate_rmse": float(torch.sqrt(mse.mean()).item()),
        "aggregate_mae": float(mae.mean().item()),
        "aggregate_signed_bias": float(bias.mean().item()),
        "case_mse": mse.tolist(),
        "case_rmse": torch.sqrt(mse).tolist(),
        "case_mae": mae.tolist(),
        "case_signed_bias": bias.tolist(),
    }


def _roughness_cases(value: torch.Tensor, active: torch.Tensor) -> dict[str, Any]:
    results: list[float] = []
    for case in range(value.shape[0]):
        field = value[case, 0].to(torch.float64)
        mask = active[case, 0] > 0
        dy_mask = mask[1:] & mask[:-1]
        dx_mask = mask[:, 1:] & mask[:, :-1]
        differences = torch.cat(
            ((field[1:] - field[:-1]).abs()[dy_mask],
             (field[:, 1:] - field[:, :-1]).abs()[dx_mask])
        )
        if not differences.numel():
            raise ValueError("roughness selector is empty")
        results.append(float(differences.mean().item()))
    return {"case_equal_mean": sum(results) / len(results), "case_values": results}


def _excess(values: torch.Tensor, field: str) -> torch.Tensor:
    if field == "sic":
        return torch.maximum((-values).clamp_min(0), (values - 1).clamp_min(0))
    if field == "sit":
        return (-values).clamp_min(0)
    raise ValueError(field)


def _support_cases(values: torch.Tensor, active: torch.Tensor, field: str) -> dict[str, Any]:
    cases = []
    global_max = 0.0
    total_bad = total_count = 0
    for index in range(values.shape[0]):
        selected = values[index, 0][active[index, 0] > 0].to(torch.float64)
        excess = _excess(selected, field)
        bad = excess > 0
        row = {
            "count": selected.numel(),
            "violation_count": int(bad.sum().item()),
            "violation_frequency": float(bad.double().mean().item()),
            "mean_excess": float(excess.mean().item()),
            "p95_excess": float(torch.quantile(excess, 0.95).item()),
            "max_excess": float(excess.max().item()),
        }
        cases.append(row)
        global_max = max(global_max, row["max_excess"])
        total_bad += row["violation_count"]
        total_count += row["count"]
    return {
        "case_values": cases,
        "pooled_violation_frequency": total_bad / total_count,
        "case_equal_mean_violation_frequency": sum(x["violation_frequency"] for x in cases) / len(cases),
        "case_equal_mean_excess": sum(x["mean_excess"] for x in cases) / len(cases),
        "case_equal_mean_p95_excess": sum(x["p95_excess"] for x in cases) / len(cases),
        "global_max_excess": global_max,
    }


def _selector_cases(
    prediction: torch.Tensor,
    truth_sit: torch.Tensor,
    selector: torch.Tensor,
    active: torch.Tensor,
) -> dict[str, Any]:
    rows = []
    for case in range(prediction.shape[0]):
        chosen = selector[case, 0] & (active[case, 0] > 0)
        count = int(chosen.sum().item())
        if count == 0:
            rows.append({"count": 0, "metrics": None})
            continue
        pred = prediction[case, 0][chosen].to(torch.float64)
        truth = truth_sit[case, 0][chosen].to(torch.float64)
        error = pred - truth
        positive = pred.clamp_min(0)
        negative = (-pred).clamp_min(0)
        rows.append({
            "count": count,
            "metrics": {
                "truth_rms": float(torch.sqrt(truth.square().mean()).item()),
                "prediction_rms": float(torch.sqrt(pred.square().mean()).item()),
                "rmse": float(torch.sqrt(error.square().mean()).item()),
                "signed_bias": float(error.mean().item()),
                "positive_mean": float(positive.mean().item()),
                "positive_p95": float(torch.quantile(positive, 0.95).item()),
                "positive_max": float(positive.max().item()),
                "fraction_above_0p01m": float((pred > 0.01).double().mean().item()),
                "negative_excess_mean": float(negative.mean().item()),
                "negative_excess_p95": float(torch.quantile(negative, 0.95).item()),
                "negative_excess_max": float(negative.max().item()),
            },
        })
    return {"case_values": rows, "total_count": sum(row["count"] for row in rows)}


def score_point_forecast(
    prediction_normalized: torch.Tensor,
    prediction_physical: torch.Tensor,
    truth_normalized: torch.Tensor,
    truth_physical: torch.Tensor,
    persistence_normalized: torch.Tensor,
    persistence_physical: torch.Tensor,
    active: torch.Tensor,
    fraction: torch.Tensor,
) -> dict[str, Any]:
    weight = fraction.to(torch.float64)
    normalized_error = prediction_normalized.to(torch.float64) - truth_normalized.to(torch.float64)
    normalized_persistence_error = persistence_normalized.to(torch.float64) - truth_normalized.to(torch.float64)
    denominator = 6 * weight.sum((1, 2, 3))
    case_objective = (normalized_error.square() * weight).sum((1, 2, 3)) / denominator
    case_persistence_objective = (
        normalized_persistence_error.square() * weight
    ).sum((1, 2, 3)) / denominator
    result: dict[str, Any] = {
        "normalized_objective": {
            "case_equal_mse": float(case_objective.mean().item()),
            "case_values": case_objective.tolist(),
            "persistence_case_equal_mse": float(case_persistence_objective.mean().item()),
            "persistence_case_values": case_persistence_objective.tolist(),
        },
        "outputs": {},
    }
    for lead_index, lead in enumerate(DIRECT_LEADS):
        for offset, field in enumerate(("sic", "sit")):
            channel = 2 * lead_index + offset
            key = f"d{lead}_{field}"
            forecast = prediction_physical[:, channel : channel + 1]
            truth = truth_physical[:, channel : channel + 1]
            persistence = persistence_physical[:, channel : channel + 1]
            scored = _weighted_cases(forecast, truth, fraction)
            baseline = _weighted_cases(persistence, truth, fraction)
            scored["persistence"] = baseline
            if baseline["aggregate_rmse"] > 0:
                scored["rmse_ratio_to_persistence"] = (
                    scored["aggregate_rmse"] / baseline["aggregate_rmse"]
                )
                scored["rmse_skill_over_persistence"] = (
                    1.0 - scored["rmse_ratio_to_persistence"]
                )
            else:
                scored["rmse_ratio_to_persistence"] = None
                scored["rmse_skill_over_persistence"] = None
            scored["roughness"] = {
                "prediction": _roughness_cases(forecast, active),
                "truth": _roughness_cases(truth, active),
                "persistence": _roughness_cases(persistence, active),
            }
            scored["raw_support"] = _support_cases(forecast, active, field)
            if field == "sit":
                truth_sic = truth_physical[:, 2 * lead_index : 2 * lead_index + 1]
                scored["truth_sit_exact_zero"] = _selector_cases(
                    forecast, truth, truth == 0, active
                )
                scored["truth_sic_below_0p01"] = _selector_cases(
                    forecast, truth, truth_sic < 0.01, active
                )
            result["outputs"][key] = scored
    return result


def _historical_metrics(payload: dict[str, Any]) -> dict[str, float]:
    forecast = payload["coarse_mean_physical"]
    truth = payload["coarse_truth_physical"]
    persistence = payload["coarse_persistence_physical"]
    active = payload["coarse_active_mask"]
    fraction = payload["coarse_ocean_fraction"]
    sic = forecast[:, None, 0::2]
    sit = forecast[:, None, 1::2]
    support = active[:, None].expand_as(sic) > 0
    sic_bad = ((sic < 0) | (sic > 1)) & support
    sit_bad = (sit < 0) & support
    denominator = int(support.sum().item())
    result = {
        "step": float(payload["optimizer_updates"]),
        "diagnostic_case_count": float(forecast.shape[0]),
        "sic_raw_support_violation_fraction": float(sic_bad.sum().item() / denominator),
        "sit_raw_support_violation_fraction": float(sit_bad.sum().item() / denominator),
        "joint_raw_support_violation_fraction": float((sic_bad | sit_bad).sum().item() / denominator),
    }
    for field, values in (("sic", sic), ("sit", sit)):
        excess = _excess(values, field)[support]
        positive = excess[excess > 0]
        result.update({
            f"{field}_raw_support_excess_mean_all": float(excess.mean().item()),
            f"{field}_raw_support_excess_mean_violating": float(positive.mean().item()) if positive.numel() else 0.0,
            f"{field}_raw_support_excess_p95_all": float(torch.quantile(excess, 0.95).item()),
            f"{field}_raw_support_excess_max": float(excess.max().item()),
        })
    for lead_index, lead in enumerate(DIRECT_LEADS):
        for offset, field in enumerate(("sic", "sit")):
            channel = 2 * lead_index + offset
            key = f"d{lead}_{field}"
            result[f"{key}_raw_mean_rmse"] = _weighted_cases(
                forecast[:, channel : channel + 1], truth[:, channel : channel + 1], fraction
            )["aggregate_rmse"]
            result[f"{key}_persistence_rmse"] = _weighted_cases(
                persistence[:, channel : channel + 1], truth[:, channel : channel + 1], fraction
            )["aggregate_rmse"]
            result[f"{key}_member_roughness"] = _legacy_roughness(
                forecast[:, channel : channel + 1], active
            )
            result[f"{key}_truth_roughness"] = _legacy_roughness(
                truth[:, channel : channel + 1], active
            )
    return result


def _legacy_roughness(value: torch.Tensor, active: torch.Tensor) -> float:
    mask_y = active[..., 1:, :] * active[..., :-1, :]
    mask_x = active[..., :, 1:] * active[..., :, :-1]
    dy = (value[..., 1:, :] - value[..., :-1, :]).abs()
    dx = (value[..., :, 1:] - value[..., :, :-1]).abs()
    return float(((dy * mask_y).sum() + (dx * mask_x).sum()).div(
        (mask_y.sum() + mask_x.sum()).clamp(min=1)
    ).item())


def _compare_metric_dict(actual: dict[str, float], expected: dict[str, float], atol: float, rtol: float) -> None:
    if set(actual) != set(expected):
        raise ValueError("historical diagnostic metric keys differ")
    for key, wanted in expected.items():
        if not math.isclose(actual[key], wanted, abs_tol=atol, rel_tol=rtol):
            raise ValueError(f"historical metric mismatch {key}: {actual[key]} != {wanted}")


def _verify_saved4(run_dir: Path, protocol: dict[str, Any], means, stds) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    payloads: dict[int, dict[str, Any]] = {}
    reference = None
    historical = {}
    for update in UPDATES:
        path = run_dir / f"coarse_mean_diagnostics/update_{update:04d}_mean.pt"
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["optimizer_updates"] != update or payload["diagnostic_status"] != "complete":
            raise ValueError(f"saved diagnostic {update} is incomplete")
        if reference is None:
            reference = payload
        else:
            if payload["validation_case_ids"] != reference["validation_case_ids"]:
                raise ValueError("saved diagnostic case IDs/order differ")
            for key in SAVED_EQUAL_TENSORS:
                if not torch.equal(payload[key], reference[key]):
                    raise ValueError(f"saved diagnostics differ in frozen tensor {key}")
        fine = payload["fine_valid_mask"][:, :1].float()
        _, recomputed_fraction = masked_block_average(fine, fine, 2)
        recomputed_fraction = recomputed_fraction[:, :1]
        if not torch.equal(recomputed_fraction, payload["coarse_ocean_fraction"]):
            raise ValueError("saved coarse fraction differs from D(fine_mask)")
        if not torch.equal((recomputed_fraction > 0).float(), payload["coarse_active_mask"]):
            raise ValueError("saved active mask differs from fraction>0")
        roundtrip = channel_denormalize(
            (payload["coarse_truth_physical"] - torch.tensor(means).view(1, 6, 1, 1))
            / torch.tensor(stds).view(1, 6, 1, 1), means, stds
        )
        torch.testing.assert_close(roundtrip, payload["coarse_truth_physical"], rtol=1e-6, atol=1e-6)
        recomputed = _historical_metrics(payload)
        stored = load_json(run_dir / f"coarse_mean_diagnostics/update_{update:04d}_metrics.json")
        _compare_metric_dict(
            recomputed, stored,
            float(protocol["historical_metric_atol"]), float(protocol["historical_metric_rtol"]),
        )
        historical[str(update)] = recomputed
        payloads[update] = payload
    assert reference is not None
    return {
        "case_ids": reference["validation_case_ids"],
        "exact_equal_tensors": list(SAVED_EQUAL_TENSORS),
        "historical_metrics_reproduced": True,
        "historical_metrics": historical,
    }, payloads


def _save_panels(
    output: Path,
    case_ids: list[str],
    truth: torch.Tensor,
    persistence: torch.Tensor,
    prediction512: torch.Tensor,
    prediction2048: torch.Tensor,
    active: torch.Tensor,
) -> list[Path]:
    visual = output / "matched_512_vs_2048"
    visual.mkdir()
    paths: list[Path] = []
    for case, case_id in enumerate(case_ids):
        # One raw state sheet with scales shared across both checkpoints.
        figure, axes = plt.subplots(6, 4, figsize=(15, 18), squeeze=False)
        for channel, key in enumerate(OUTPUT_KEYS):
            field = "sic" if channel % 2 == 0 else "sit"
            values = [truth[case, channel], persistence[case, channel],
                      prediction512[case, channel], prediction2048[case, channel]]
            if field == "sic":
                lower = min(0.0, *(float(value.min()) for value in values))
                upper = max(1.0, *(float(value.max()) for value in values))
            else:
                lower = min(0.0, *(float(value.min()) for value in values))
                upper = max(float(value.max()) for value in values)
            for column, (label, value) in enumerate(zip(
                ("truth", "persistence", "update512", "update2048"), values, strict=True
            )):
                shown = value.clone()
                shown[active[case, 0] == 0] = torch.nan
                image = axes[channel, column].imshow(
                    shown, origin="upper", cmap="viridis", vmin=lower, vmax=upper
                )
                axes[channel, column].set_title(label)
                axes[channel, column].set_ylabel(key)
                axes[channel, column].set_xticks([]); axes[channel, column].set_yticks([])
                figure.colorbar(image, ax=axes[channel, column], fraction=0.046)
        figure.suptitle(f"raw truth / persistence / checkpoint comparison; {case_id}")
        figure.tight_layout(rect=(0, 0, 1, 0.98))
        path = visual / f"case{case:02d}_raw_states.png"
        figure.savefig(path, dpi=135); plt.close(figure); paths.append(path)

        figure, axes = plt.subplots(6, 2, figsize=(9, 18), squeeze=False)
        for channel, key in enumerate(OUTPUT_KEYS):
            errors = [prediction512[case, channel] - truth[case, channel],
                      prediction2048[case, channel] - truth[case, channel]]
            limit = max(float(error.abs().max()) for error in errors)
            limit = max(limit, 1e-8)
            for column, (label, error) in enumerate(zip(("update512", "update2048"), errors, strict=True)):
                shown = error.clone(); shown[active[case, 0] == 0] = torch.nan
                image = axes[channel, column].imshow(
                    shown, origin="upper", cmap="coolwarm", vmin=-limit, vmax=limit
                )
                axes[channel, column].set_title(label)
                axes[channel, column].set_ylabel(f"{key} signed error")
                axes[channel, column].set_xticks([]); axes[channel, column].set_yticks([])
                figure.colorbar(image, ax=axes[channel, column], fraction=0.046)
        figure.suptitle(f"raw signed errors, shared checkpoint scale; {case_id}")
        figure.tight_layout(rect=(0, 0, 1, 0.98))
        path = visual / f"case{case:02d}_signed_errors.png"
        figure.savefig(path, dpi=135); plt.close(figure); paths.append(path)

        figure, axes = plt.subplots(3, 4, figsize=(15, 9), squeeze=False)
        for lead_index, lead in enumerate(DIRECT_LEADS):
            channel = 2 * lead_index + 1
            for column, (label, value) in enumerate(zip(
                ("truth", "persistence", "update512", "update2048"),
                (truth[case, channel], persistence[case, channel],
                 prediction512[case, channel], prediction2048[case, channel]), strict=True
            )):
                shown = value.clone(); shown[active[case, 0] == 0] = torch.nan
                image = axes[lead_index, column].imshow(
                    shown, origin="upper", cmap="magma", vmin=0.0, vmax=0.05
                )
                axes[lead_index, column].set_title(label)
                axes[lead_index, column].set_ylabel(f"d{lead} SIT; display saturated outside 0–0.05m")
                axes[lead_index, column].set_xticks([]); axes[lead_index, column].set_yticks([])
                figure.colorbar(image, ax=axes[lead_index, column], fraction=0.046)
        figure.suptitle(f"SIT low-range view; raw tensors unchanged; {case_id}")
        figure.tight_layout(rect=(0, 0, 1, 0.97))
        path = visual / f"case{case:02d}_sit_0_0p05m.png"
        figure.savefig(path, dpi=135); plt.close(figure); paths.append(path)
    return paths


def _fp32_saved_comparison(
    fp32_normalized: torch.Tensor,
    fp32_physical: torch.Tensor,
    saved: dict[str, Any],
    positions: list[int],
    fraction: torch.Tensor,
) -> dict[str, Any]:
    result = {}
    for space, current, prior in (
        ("normalized", fp32_normalized[positions], saved["coarse_mean_normalized"]),
        ("physical", fp32_physical[positions], saved["coarse_mean_physical"]),
    ):
        delta = current.to(torch.float64) - prior.to(torch.float64)
        weight = fraction[positions].to(torch.float64)
        denominator = 6 * weight.sum((1, 2, 3))
        case_mse = (delta.square() * weight).sum((1, 2, 3)) / denominator
        case_bias = (delta * weight).sum((1, 2, 3)) / denominator
        result[space] = {
            "case_equal_rms_difference": float(torch.sqrt(case_mse.mean()).item()),
            "case_equal_signed_bias_difference": float(case_bias.mean().item()),
            "global_max_absolute_difference": float(delta.abs().max().item()),
            "case_rms_difference": torch.sqrt(case_mse).tolist(),
            "case_signed_bias_difference": case_bias.tolist(),
        }
    return result


def _launch_status(status: str, **details: Any) -> None:
    raw = os.environ.get("COARSE_MEAN_AUDIT_STATUS_PATH", "").strip()
    if not raw:
        return
    path = Path(raw)
    if not path.is_absolute() or path.name != "status.json":
        raise ValueError("COARSE_MEAN_AUDIT_STATUS_PATH must be an absolute status.json")
    _atomic_json(path, {"status": status, **details})


@torch.no_grad()
def _run_impl(config_path: Path, output: Path, lifecycle: dict[str, Any]) -> dict[str, Any]:
    _require_cpu_envelope()
    experiment = load_json(config_path)
    if experiment.get("schema_version") != "coarse_mean_frozen_audit_contract_v1":
        raise ValueError("unexpected frozen mean audit schema")
    protocol = experiment["protocol"]
    expected_protocol = {
        "split": "valid", "case_count": 48, "batch_size": 4, "workers": 0,
        "inference_dtype": "float32", "scoring_dtype": "float64",
        "checkpoint": "coarse_update_0512.pth", "matched_updates": list(UPDATES),
        "matched_subset_positions": [4, 16, 28, 40], "physical_tolerance": 1e-6,
        "historical_metric_atol": 1e-6, "historical_metric_rtol": 1e-5,
        "optimizer_steps": 0, "new_dates": False,
    }
    if protocol != expected_protocol:
        raise ValueError("frozen mean audit protocol differs from reviewed contract")
    repo_root = Path(__file__).resolve().parents[1]
    code_identity = _clean_code_identity(repo_root)
    run_dir = Path(experiment["source_run_dir"])
    if not run_dir.is_absolute():
        raise ValueError("source run directory must be absolute")
    _verify_hash_inventory(run_dir, experiment["source_artifacts_sha256"])
    _verify_hash_inventory(repo_root, experiment["source_files_sha256"])
    checkpoint_identity = _verify_checkpoint_identity(run_dir)
    metadata = load_json(run_dir / "metadata.json")
    manifest = load_json(run_dir / "coarse_cascade_manifest.json")
    sentinel = load_json(run_dir / "coarse_cascade_dataset_sentinel.json")
    if manifest.get("code_commit") != experiment["source_commit"]:
        raise ValueError("source commit binding differs")
    source_experiment = load_json(
        repo_root / "config/experiments/train_direct_dynamics_cascade_coarse_mean_v1.json"
    )
    if metadata.get("experiment_config") != source_experiment:
        raise ValueError("metadata experiment config differs from its hash-bound source")
    source_data = load_json(repo_root / "config/data/m2m_2f_two_stage_dynamics_protocol.json")
    effective_data = merge_config_overrides(source_data, source_experiment.get("data_overrides"))
    if metadata.get("data_config") != effective_data:
        raise ValueError("metadata data law differs from hash-bound source plus overrides")
    required_manifest = {
        "sampler": "CoarseConditionalMeanPredictor", "target": "Delta=C-P",
        "reconstruction": "m(c)=P(c)+Delta_theta(c)", "model_input_channels": 50,
        "model_output_channels": 6, "coarse_factor": 2,
        "stochastic_state_input": False, "ode_sampling": False,
        "clipping_or_postprocessing": False,
    }
    for key, value in required_manifest.items():
        if manifest.get(key) != value:
            raise ValueError(f"source manifest differs for {key}")
    subset = sentinel["pilot_subset"]
    if subset != metadata["dataset_provenance"]["pilot_subset"]:
        raise ValueError("sentinel and metadata subset provenance differ")
    indices = subset["validation_indices"]
    if len(indices) != 48 or _indices_sha256(indices) != experiment["validation_indices_sha256"]:
        raise ValueError("frozen validation indices differ")
    if subset["validation_indices_sha256"] != experiment["validation_indices_sha256"]:
        raise ValueError("sentinel validation index binding differs")

    output.mkdir(parents=True, exist_ok=False)
    lifecycle["output"] = output
    run_id = os.environ.get("COARSE_MEAN_AUDIT_RUN_ID", "").strip()
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", run_id) is None:
        raise ValueError("COARSE_MEAN_AUDIT_RUN_ID must be a safe non-empty identifier")
    tracker = ClearMLTracker(
        experiment["project_name"], f"{experiment['task_name']}-{run_id}",
        tags=experiment["clearml"]["tags"], env_path=experiment["clearml"]["env_path"],
    )
    lifecycle["tracker"] = tracker
    frozen_contract = {
        "schema_version": experiment["schema_version"],
        "run_id": run_id,
        "audit_config_sha256": _sha256(config_path),
        "source_commit": experiment["source_commit"],
        "audit_execution_commit": code_identity["git_commit"],
        "source_run_dir": str(run_dir),
        "source_artifacts_sha256": experiment["source_artifacts_sha256"],
        "source_files_sha256": experiment["source_files_sha256"],
        "protocol": protocol,
        "checkpoint_identity": checkpoint_identity,
        "raw_weights_not_ema": True,
        "training_permitted": False,
        "innovations_permitted": False,
        "fine_stage_permitted": False,
        "publication_ready": False,
    }
    _atomic_json(output / "audit_contract.json", frozen_contract)
    tracker.connect("frozen_mean_audit_contract", frozen_contract)
    _launch_status("source_verified", code_commit=code_identity["git_commit"],
                   output_dir=str(output), clearml_task_id=str(tracker.task.id))

    means = _repeat_field_stats(metadata["normalization_means"])
    stds = _repeat_field_stats(metadata["normalization_stds"])
    saved4, saved_payloads = _verify_saved4(run_dir, protocol, means, stds)
    _atomic_json(output / "saved4_recomputation.json", saved4)

    dataset = build_dataset(metadata["data_config"], split="valid")
    dataset_validation = validate_direct_dataset(dataset)
    calendar_inventory = _calendar_inventory(dataset, split="valid")
    if dataset.provenance() != metadata["dataset_provenance"]["valid"]:
        raise ValueError("rebuilt validation dataset provenance differs")
    items = [dataset[index] for index in indices]
    if _source_tensor_sha256(items[0]["valid_mask"][:1]) != metadata["dataset_provenance"][
        "static_valid_mask_sha256"
    ]:
        raise ValueError("rebuilt static ocean mask differs from frozen training provenance")
    batch = _fine_collate(items)
    case_ids = list(batch["meta"]["case_id"])
    if case_ids != subset["validation_case_ids"]:
        raise ValueError("rebuilt validation case IDs/order differ")
    valid = batch["valid_mask"][:, :1].float()
    truth_normalized, active, fraction = coarse_target(batch["truth"], valid)
    persistence_normalized, persistence_active, persistence_fraction = (
        coarse_persistence_from_condition(batch["structured_conditioning"], valid)
    )
    if not torch.equal(active, persistence_active) or not torch.equal(fraction, persistence_fraction):
        raise ValueError("truth and causal-persistence supports differ")
    background_normalized, _, _ = coarse_target(batch["background"], valid)
    if not torch.equal(background_normalized, persistence_normalized):
        raise ValueError("causal d0 persistence differs from dataset persistence baseline")
    truth_physical = channel_denormalize(truth_normalized, means, stds)
    persistence_physical = channel_denormalize(persistence_normalized, means, stds)
    positions = protocol["matched_subset_positions"]
    saved512 = saved_payloads[512]
    if [case_ids[position] for position in positions] != saved4["case_ids"]:
        raise ValueError("matched saved4 cases are not positions [4,16,28,40] in frozen48")
    for key, rebuilt in (
        ("structured_conditioning", batch["structured_conditioning"][positions]),
        ("fine_valid_mask", batch["valid_mask"][positions]),
        ("coarse_truth_physical", truth_physical[positions]),
        ("coarse_persistence_physical", persistence_physical[positions]),
        ("coarse_active_mask", active[positions]),
        ("coarse_ocean_fraction", fraction[positions]),
    ):
        torch.testing.assert_close(rebuilt, saved512[key], rtol=0.0, atol=0.0)

    fixed_inputs = {
        "case_ids": case_ids,
        "validation_indices": indices,
        "structured_conditioning": batch["structured_conditioning"].float(),
        "fine_valid_mask": batch["valid_mask"].float(),
        "coarse_active": active,
        "coarse_ocean_fraction": fraction,
        "truth_normalized": truth_normalized,
        "truth_physical": truth_physical,
        "persistence_normalized": persistence_normalized,
        "persistence_physical": persistence_physical,
        "normalization_means": means,
        "normalization_stds": stds,
    }
    _atomic_torch_save(fixed_inputs, output / "frozen48_inputs.pt")
    input_hashes = {
        key: _tensor_sha256(value) for key, value in fixed_inputs.items() if torch.is_tensor(value)
    }
    input_manifest = {
        "saved_before_inference": True,
        "file_sha256": _sha256(output / "frozen48_inputs.pt"),
        "tensor_content_sha256": input_hashes,
        "case_ids": case_ids,
        "dataset_validation": dataset_validation,
        "calendar_inventory": calendar_inventory,
    }
    _atomic_json(output / "frozen48_inputs_manifest.json", input_manifest)

    _launch_status("cpu_fp32_inference", code_commit=code_identity["git_commit"],
                   output_dir=str(output), clearml_task_id=str(tracker.task.id))
    training_config = TrainingConfig.from_dict(metadata["training_config"])
    if (tuple(training_config.image_size), training_config.in_channels,
            training_config.out_channels) != ((160, 128), 50, 6):
        raise ValueError("source architecture is not the reviewed 50-to-6 coarse model")
    model = build_unet(training_config).to(device="cpu", dtype=torch.float32).eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != 16_256_326:
        raise ValueError(f"unexpected source model size: {parameter_count}")
    state = torch.load(run_dir / protocol["checkpoint"], map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    predictor = CoarseConditionalMeanPredictor(model)
    predictions = []
    for start in range(0, 48, 4):
        stop = start + 4
        predictions.append(predictor.predict_conditioned(
            structured_conditioning=batch["structured_conditioning"][start:stop].float(),
            valid_mask=valid[start:stop], device=torch.device("cpu"),
        ))
    prediction_normalized = torch.cat(predictions)
    prediction_physical = channel_denormalize(prediction_normalized, means, stds)
    if not torch.isfinite(prediction_normalized[active.expand_as(prediction_normalized) > 0]).all():
        raise FloatingPointError("CPU FP32 inference produced NaN/Inf")
    raw_predictions = {
        "case_ids": case_ids,
        "checkpoint_sha256": experiment["source_artifacts_sha256"][protocol["checkpoint"]],
        "fixed_inputs_sha256": input_manifest["file_sha256"],
        "dtype": "torch.float32",
        "parameter_count": parameter_count,
        "timestep": 0.0,
        "prediction_normalized": prediction_normalized,
        "prediction_physical": prediction_physical,
    }
    _atomic_torch_save(raw_predictions, output / "frozen48_raw_predictions.pt")
    raw_manifest = {
        "saved_before_scoring_or_plotting": True,
        "file_sha256": _sha256(output / "frozen48_raw_predictions.pt"),
        "normalized_tensor_sha256": _tensor_sha256(prediction_normalized),
        "physical_tensor_sha256": _tensor_sha256(prediction_physical),
    }
    _atomic_json(output / "frozen48_raw_predictions_manifest.json", raw_manifest)

    metrics = score_point_forecast(
        prediction_normalized, prediction_physical, truth_normalized, truth_physical,
        persistence_normalized, persistence_physical, active, fraction,
    )
    fp32_comparison = _fp32_saved_comparison(
        prediction_normalized, prediction_physical, saved512, positions, fraction
    )
    tolerance = float(protocol["physical_tolerance"])
    all_skilled = all(
        isinstance(metrics["outputs"][key]["rmse_skill_over_persistence"], float)
        and metrics["outputs"][key]["rmse_skill_over_persistence"] > 0
        for key in OUTPUT_KEYS
    )
    physical_support_pass = all(
        metrics["outputs"][key]["raw_support"]["global_max_excess"] <= tolerance
        for key in OUTPUT_KEYS
    )
    figures = _save_panels(
        output, saved4["case_ids"], saved512["coarse_truth_physical"],
        saved512["coarse_persistence_physical"], saved512["coarse_mean_physical"],
        saved_payloads[2048]["coarse_mean_physical"], saved512["coarse_active_mask"],
    )
    decision = {
        "audit_integrity_pass": True,
        "retain_as_unconstrained_point_control": "retain" if all_skilled else "hold",
        "physical_support_pass": physical_support_pass,
        "physical_support_tolerance": tolerance,
        "innovations_permitted": False,
        "fine_stage_permitted": False,
        "training_permitted": False,
        "publication_ready": False,
        "rank_or_ensemble_calibration_computed": False,
        "reason": (
            "raw checkpoint512 is strictly better than persistence for all six frozen48 outputs"
            if all_skilled else "at least one frozen48 output does not beat persistence"
        ),
    }
    result = {
        "status": "complete_pending_astra_and_visual_review",
        "contract_sha256": _sha256(output / "audit_contract.json"),
        "fixed_inputs_manifest": input_manifest,
        "raw_predictions_manifest": raw_manifest,
        "saved4": saved4,
        "fp32_vs_saved_matched4": fp32_comparison,
        "metrics": metrics,
        "decision": decision,
        "figure_files": [str(path) for path in figures],
        "clearml_task_id": str(tracker.task.id),
    }
    _atomic_json(output / "frozen_mean_audit.json", result)
    for key in OUTPUT_KEYS:
        tracker.report_scalar("frozen48/rmse", key, metrics["outputs"][key]["aggregate_rmse"], 0)
        skill = metrics["outputs"][key]["rmse_skill_over_persistence"]
        if isinstance(skill, float):
            tracker.report_scalar("frozen48/skill", key, skill, 0)
    tracker.report_single_value("audit_integrity_pass", 1.0)
    tracker.report_single_value("all_six_beat_persistence", float(all_skilled))
    tracker.report_single_value("physical_support_pass", float(physical_support_pass))
    tracker.upload_artifact("frozen_mean_audit", output / "frozen_mean_audit.json")
    for path in figures:
        tracker.report_image("frozen_mean_audit", path.stem, path, 0)
    tracker.close()
    lifecycle["tracker"] = None
    _launch_status("complete_pending_astra_and_visual_review",
                   code_commit=code_identity["git_commit"], output_dir=str(output),
                   clearml_task_id=result["clearml_task_id"])
    return result


def run(config_path: Path, output: Path) -> dict[str, Any]:
    lifecycle: dict[str, Any] = {"output": None, "tracker": None}
    try:
        return _run_impl(config_path, output, lifecycle)
    except BaseException as error:
        failure = {
            "status": "failed", "error_type": type(error).__name__, "error": str(error),
            "audit_integrity_pass": False, "scientific_decision_permitted": False,
            "innovations_permitted": False, "fine_stage_permitted": False,
            "training_permitted": False, "publication_ready": False, "cleanup_errors": [],
        }
        created = lifecycle.get("output")
        if isinstance(created, Path) and created.is_dir():
            try: _atomic_json(created / "failure.json", failure)
            except Exception as cleanup: failure["cleanup_errors"].append(f"failure_artifact: {cleanup}")
        try: _launch_status(**failure)
        except Exception as cleanup: failure["cleanup_errors"].append(f"launch_status: {cleanup}")
        tracker = lifecycle.get("tracker")
        if tracker is not None:
            try: tracker.task.mark_failed(status_reason=type(error).__name__, status_message=str(error)[:1000])
            except Exception as cleanup: failure["cleanup_errors"].append(f"clearml_mark_failed: {cleanup}")
            try: tracker.close()
            except Exception as cleanup: failure["cleanup_errors"].append(f"clearml_close: {cleanup}")
        if isinstance(created, Path) and created.is_dir():
            try: _atomic_json(created / "failure.json", failure)
            except Exception: pass
        raise


def main() -> None:
    def terminate(signum, _frame):
        raise TimeoutError(f"frozen mean CPU audit received signal {signum}")
    signal.signal(signal.SIGTERM, terminate)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.config.resolve(), arguments.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
