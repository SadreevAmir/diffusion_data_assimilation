"""OOF empirical marginal quantile transport for the frozen cascade ensemble."""

from __future__ import annotations

import argparse
import json
import os
import signal
import traceback
from pathlib import Path
from typing import Any

import torch

from .clearml_tracking import ClearMLTracker
from .config import load_json
from .direct_dynamics_cascade import smooth_right_inverse
from .direct_dynamics_cascade_fine_training import _clean_code_identity
from .direct_dynamics_cascade_memberwise_affine import (
    CHANNELS,
    FIELDS,
    _atomic_strict_json,
    _atomic_torch_save,
    _finalize_failed_tracker,
    _fit_arrays,
    _gate,
    _require_finite_tree,
    _save_panels,
    _sha256,
    _validate_inputs,
    _validate_source_contract,
    apply_memberwise,
    compare,
    diagnostics,
    validate_folds,
)
from .transforms import channel_denormalize


def empirical_midrank_transport(
    values: torch.Tensor, source_sorted: torch.Tensor, target_sorted: torch.Tensor
) -> torch.Tensor:
    """Apply H^-1(G_mid(x)) with deterministic source-tie handling."""
    if source_sorted.ndim != 1 or target_sorted.ndim != 1:
        raise ValueError("empirical references must be one-dimensional")
    if source_sorted.numel() == 0 or target_sorted.numel() == 0:
        raise ValueError("empirical references must be nonempty")
    if not torch.isfinite(values).all() or not torch.isfinite(source_sorted).all() or not torch.isfinite(target_sorted).all():
        raise FloatingPointError("quantile transport received NaN/Inf")
    if torch.any(source_sorted[1:] < source_sorted[:-1]) or torch.any(target_sorted[1:] < target_sorted[:-1]):
        raise ValueError("empirical references must be sorted")
    flat = values.contiguous().view(-1)
    left = torch.searchsorted(source_sorted, flat, right=False)
    right = torch.searchsorted(source_sorted, flat, right=True)
    probability = (left.double() + right.double()) / (2.0 * source_sorted.numel())
    target_index = (torch.ceil(probability * target_sorted.numel()).long() - 1).clamp(0, target_sorted.numel() - 1)
    return target_sorted[target_index].reshape_as(values)


def fit_oof_quantile(
    raw: torch.Tensor,
    truth: torch.Tensor,
    valid: torch.Tensor,
    folds: list[list[int]],
) -> tuple[torch.Tensor, list[dict[str, Any]], dict[int, dict[str, tuple[float, float]]]]:
    candidate = torch.empty_like(raw)
    rows: list[dict[str, Any]] = []
    thresholds: dict[int, dict[str, tuple[float, float]]] = {}
    all_indices = set(range(raw.shape[0]))
    for fold_number, heldout in enumerate(folds):
        train = sorted(all_indices - set(heldout))
        fields = {}
        references = {}
        for offset, field in enumerate(FIELDS):
            member_values, targets = _fit_arrays(raw, truth, valid, train, offset)
            source_sorted = member_values.reshape(-1).sort().values
            target_sorted = targets.sort().values
            references[field] = (source_sorted, target_sorted)
            q01 = float(torch.quantile(target_sorted, 0.01))
            q99 = float(torch.quantile(target_sorted, 0.99))
            fields[field] = {
                "source_count": source_sorted.numel(), "target_count": target_sorted.numel(),
                "source_min": float(source_sorted[0]), "source_max": float(source_sorted[-1]),
                "target_min": float(target_sorted[0]), "target_max": float(target_sorted[-1]),
                "source_zero_mass": float((source_sorted == 0).double().mean()),
                "target_zero_mass": float((target_sorted == 0).double().mean()),
                "truth_q01": q01, "truth_q99": q99,
            }
        for case in heldout:
            thresholds[case] = {
                field: (fields[field]["truth_q01"], fields[field]["truth_q99"])
                for field in FIELDS
            }
        for offset, field in enumerate(FIELDS):
            source_sorted, target_sorted = references[field]
            candidate[heldout, :, offset::2] = empirical_midrank_transport(
                raw[heldout, :, offset::2], source_sorted, target_sorted
            )
        rows.append({"fold": fold_number, "fit_indices": train, "heldout_indices": heldout, "fields": fields})
    if not torch.isfinite(candidate).all():
        raise FloatingPointError("OOF quantile transport produced NaN/Inf")
    return candidate, rows, thresholds


def tie_fraction(ensemble: torch.Tensor, valid: torch.Tensor) -> dict[str, float]:
    result = {}
    mask = valid[:, None] > 0
    for channel, name in enumerate(CHANNELS):
        ordered = ensemble[:, :, channel : channel + 1].sort(1).values
        ties = ordered[:, 1:] == ordered[:, :-1]
        selected = mask.expand(-1, 7, -1, -1, -1)
        result[name] = float(ties[selected].double().mean())
    return result


def extreme_mean_mae(
    ensemble: torch.Tensor,
    truth: torch.Tensor,
    valid: torch.Tensor,
    thresholds: dict[int, dict[str, tuple[float, float]]],
) -> dict[str, dict[str, float]]:
    result = {}
    mean = ensemble.mean(1)
    for channel, name in enumerate(CHANNELS):
        field = FIELDS[channel % 2]
        low_rows, high_rows = [], []
        low_fraction, high_fraction = [], []
        for case in range(len(ensemble)):
            low, high = thresholds[case][field]
            ocean = valid[case, 0] > 0
            error = (mean[case, channel] - truth[case, channel]).abs()
            low_selected = ocean & (truth[case, channel] <= low)
            high_selected = ocean & (truth[case, channel] >= high)
            denominator = int(ocean.sum())
            low_rows.append(float(error[low_selected].mean()) if low_selected.any() else None)
            high_rows.append(float(error[high_selected].mean()) if high_selected.any() else None)
            low_fraction.append(float(low_selected.sum()) / denominator)
            high_fraction.append(float(high_selected.sum()) / denominator)
        low_nonempty = [value for value in low_rows if value is not None]
        high_nonempty = [value for value in high_rows if value is not None]
        result[name] = {
            "definition": "case-equal MAE on OOF fit-truth empirical q01/q99 threshold selectors; atom ties may select more than 1%",
            "low_quantile_threshold_mae": sum(low_nonempty) / len(low_nonempty) if low_nonempty else None,
            "high_quantile_threshold_mae": sum(high_nonempty) / len(high_nonempty) if high_nonempty else None,
            "low_nonempty_case_count": len(low_nonempty), "high_nonempty_case_count": len(high_nonempty),
            "low_per_case": low_rows, "high_per_case": high_rows,
            "low_selected_fraction_per_case": low_fraction, "high_selected_fraction_per_case": high_fraction,
        }
    return result


def run(config_path: Path, output: Path) -> dict[str, Any]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.device_count() != 0:
        raise RuntimeError("quantile transport gate must be CPU-only")
    if os.environ.get("CLEARML_REQUIRE_ONLINE") != "1":
        raise RuntimeError("quantile transport gate requires online ClearML")
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
        tracker.connect("quantile_transport_protocol", config)
        _atomic_strict_json(output / "status.json", {"status": "running", "clearml_task_id": str(tracker.task.id)})
        repository = Path(__file__).resolve().parents[1]
        source = Path(config["source"]["path"])
        if not source.is_file() or _sha256(source) != config["source"]["sha256"]:
            raise ValueError("frozen champion artifact differs")
        identity = _clean_code_identity(repository)
        payload = torch.load(source, map_location="cpu", weights_only=True)
        _validate_source_contract(payload, config, repository)
        valid = payload["valid"].float()
        mask_members = valid[:, None].expand(-1, 8, -1, -1, -1).flatten(0, 1)
        raw_normalized = payload["residual"].float() + smooth_right_inverse(
            payload["coarse"].float().flatten(0, 1), mask_members
        ).unflatten(0, (12, 8))
        field_means = torch.tensor(config["normalization"]["means"], dtype=torch.float32)
        field_stds = torch.tensor(config["normalization"]["stds"], dtype=torch.float32)
        channel_means, channel_stds = field_means.repeat(3), field_stds.repeat(3)
        raw = channel_denormalize(raw_normalized.flatten(0, 1), channel_means, channel_stds).unflatten(0, (12, 8))
        truth = channel_denormalize(payload["truth"].float(), channel_means, channel_stds)
        _validate_inputs(raw, truth, valid)
        folds = config["folds"]; validate_folds(folds, 12)
        support_parameters = {"sic_scale": 1.0, "sic_offset": 0.0, "sit_scale": 1.0, "sit_offset": 0.0}
        support = apply_memberwise(raw, field_stds, support_parameters)
        candidate, fold_fits, thresholds = fit_oof_quantile(raw, truth, valid, folds)
        provenance = {"source": config["source"], "normalization": config["normalization"], "case_ids": list(payload["case_ids"]), "folds": folds, "code_identity": identity}
        tensor_hash = _atomic_torch_save({"ensemble": candidate, "fold_fits": fold_fits, **provenance}, output / "oof_quantile_calibrated.pt")
        _atomic_strict_json(output / "fold_fits.json", {"status": "fit_complete", "fold_fits": fold_fits, **provenance})
        raw_diag, support_diag, candidate_diag = (diagnostics(value, truth, valid, channel_stds) for value in (raw, support, candidate))
        versus_raw = compare(candidate_diag, raw_diag, config["gate"], 0)
        versus_support = compare(candidate_diag, support_diag, config["gate"], 1)
        dummy_fits = [{"field_fits": {field: {"optimizer_success": True, "optimum_on_boundary": False} for field in FIELDS}}]
        gate = _gate(candidate_diag, raw_diag, support_diag, versus_raw, versus_support, dummy_fits, config["gate"])
        extremes = {label: extreme_mean_mae(value, truth, valid, thresholds) for label, value in (("raw", raw), ("support_only", support), ("candidate", candidate))}
        for name in CHANNELS:
            for tail in ("low_quantile_threshold_mae", "high_quantile_threshold_mae"):
                value, reference = extremes["candidate"][name][tail], extremes["support_only"][name][tail]
                if value is None or reference is None:
                    gate["failures"].append(f"{name} {tail} selector is empty")
                elif value > 1.01 * reference + 1e-8:
                    gate["failures"].append(f"{name} {tail} worsened versus support")
        gate["passed_pending_visual_review"] = not gate["failures"]
        extra = {"tie_fraction": {label: tie_fraction(value, valid) for label, value in (("raw", raw), ("support_only", support), ("candidate", candidate))}, "extreme_mean_mae": extremes}
        _require_finite_tree({"diagnostics": [raw_diag, support_diag, candidate_diag], "comparisons": [versus_raw, versus_support], "extra": extra})
        result = {"status": "scored_pending_visual_review", "training_performed": False, **provenance, "config_sha256": _sha256(config_path), "method": "OOF empirical H^-1(G_mid(x)); no bins, knots, smoothing, tail extrapolation, noise, or member reordering", "fold_fits": fold_fits, "controls": {"raw": raw_diag, "support_only": support_diag}, "candidate": candidate_diag, "comparison": {"versus_raw": versus_raw, "versus_support_only": versus_support}, "additional_diagnostics": extra, "gate": {**gate, "training_or_replacement_permitted": False}, "tensor_sha256": {"oof_quantile_calibrated.pt": tensor_hash}, "figures": [], "clearml_task_id": str(tracker.task.id)}
        _atomic_strict_json(output / "quantile_transport_gate.json", result)
        tracker.upload_artifact("quantile_transport_gate_scored", output / "quantile_transport_gate.json")
        figures = _save_panels(output, list(payload["case_ids"]), truth, raw, support, candidate, valid)
        for path in figures: tracker.report_image("quantile_transport_visual_qc", Path(path).stem, Path(path), 0)
        tracker.close(); tracker = None
        result["status"] = "complete_pending_astra_review"; result["figures"] = figures
        _atomic_strict_json(output / "quantile_transport_gate.json", result)
        _atomic_strict_json(output / "status.json", {"status": result["status"], "clearml_task_id": result["clearml_task_id"]})
        return result
    except BaseException as error:
        failure = {"status": "failed", "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()}
        _atomic_strict_json(output / "failure.json", failure)
        _atomic_strict_json(output / "status.json", {"status": "failed", "error_type": type(error).__name__, "error": str(error)})
        if tracker is not None: _finalize_failed_tracker(tracker, error, output / "failure.json")
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True, type=Path); parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args(); print(json.dumps(run(arguments.config.resolve(), arguments.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
