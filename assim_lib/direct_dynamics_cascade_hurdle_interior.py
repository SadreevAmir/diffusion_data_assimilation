"""OOF memberwise hurdle plus smooth positive-interior calibration."""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.optimize import minimize

from .clearml_tracking import ClearMLTracker
from .config import load_json
from .direct_dynamics_cascade import smooth_right_inverse
from .direct_dynamics_cascade_fine_training import _clean_code_identity
from .direct_dynamics_cascade_memberwise_affine import (
    FIELDS, _atomic_strict_json, _atomic_torch_save, _finalize_failed_tracker,
    _fit_arrays, _gate, _require_finite_tree, _save_panels, _sha256,
    _validate_inputs, _validate_source_contract, apply_memberwise, compare,
    diagnostics, validate_folds,
)
from .direct_dynamics_cascade_quantile_transport import extreme_mean_mae, tie_fraction
from .transforms import channel_denormalize


def apply_hurdle_interior(
    values: torch.Tensor, field_std: float, field: str,
    theta_standardized: float, shape: float, intercept: float,
) -> torch.Tensor:
    if field not in FIELDS or field_std <= 0 or shape <= 0:
        raise ValueError("invalid hurdle transform field/std/shape")
    if not all(math.isfinite(x) for x in (field_std, theta_standardized, shape, intercept)):
        raise ValueError("hurdle parameters must be finite")
    source = values.double()
    u = (source - field_std * theta_standardized) / field_std
    positive = u > 0
    output = torch.zeros_like(source)
    if positive.any():
        log_u = torch.log(u[positive])
        linear = shape * log_u + intercept
        transformed = torch.sigmoid(linear) if field == "sic" else field_std * torch.exp(linear)
        if not torch.isfinite(transformed).all():
            raise FloatingPointError("hurdle transform produced NaN/Inf")
        if field == "sic" and torch.any((transformed <= 0) | (transformed >= 1)):
            raise FloatingPointError("SIC positive-interior transform saturated numerically")
        output[positive] = transformed
    return output


def _objective(
    values: torch.Tensor, ordered_values: torch.Tensor, targets: torch.Tensor,
    field_std: float, field: str, vector: np.ndarray,
) -> float:
    try:
        transformed = apply_hurdle_interior(values, field_std, field, *map(float, vector))
        ordered = apply_hurdle_interior(ordered_values, field_std, field, *map(float, vector))
    except (ValueError, FloatingPointError):
        return 1e12
    count = values.shape[1]
    accuracy = (transformed - targets[:, None]).abs().mean()
    coefficients = 2 * torch.arange(count, dtype=torch.float64) - count + 1
    pair = (ordered * coefficients).sum(1).mean() / (count * (count - 1))
    score = float(((accuracy - pair) / field_std).item())
    return score if math.isfinite(score) else 1e12


def _fit_field(
    values: torch.Tensor, targets: torch.Tensor, field_std: float,
    field: str, spec: dict[str, Any],
) -> dict[str, Any]:
    values, targets = values.double(), targets.double()
    ordered = values.sort(1).values
    stride = int(spec["preview_pixel_stride"])
    preview_values, preview_ordered, preview_targets = values[::stride], ordered[::stride], targets[::stride]
    theta_bounds = tuple(spec["theta_bounds_standardized"])
    shape_bounds = tuple(spec["shape_bounds"])
    intercept_bounds = tuple(spec["intercept_bounds"])
    preview = []
    for theta in np.linspace(*theta_bounds, int(spec["preview_theta_count"])):
        for shape in spec["preview_shapes"]:
            for intercept in spec["preview_intercepts"]:
                vector = np.array([theta, shape, intercept], dtype=float)
                preview.append((_objective(preview_values, preview_ordered, preview_targets, field_std, field, vector), vector))
    starts = [row[1] for row in sorted(preview, key=lambda row: row[0])[:int(spec["optimizer_starts"])]]
    cache: dict[tuple[float, float, float], float] = {}

    def objective(vector: np.ndarray) -> float:
        clipped = np.array([
            np.clip(vector[0], *theta_bounds), np.clip(vector[1], *shape_bounds),
            np.clip(vector[2], *intercept_bounds),
        ])
        key = tuple(round(float(value), 8) for value in clipped)
        if key not in cache:
            cache[key] = _objective(values, ordered, targets, field_std, field, clipped)
        return cache[key]

    candidates = []
    for start in starts:
        fitted = minimize(
            objective, start, method="Powell",
            bounds=(theta_bounds, shape_bounds, intercept_bounds),
            options={"maxfev": int(spec["powell_maxfev"]), "xtol": float(spec["powell_xtol"]), "ftol": float(spec["powell_ftol"])},
        )
        candidates.append((objective(fitted.x), fitted.x, bool(fitted.success)))
    converged = [row for row in candidates if row[2]]
    score, vector, success = min(converged, key=lambda row: row[0]) if converged else min(candidates, key=lambda row: row[0])
    tolerance = 2 * max(float(spec["powell_xtol"]), 1e-6)
    bounds = (theta_bounds, shape_bounds, intercept_bounds)
    boundary = any(abs(float(value) - bound[0]) <= tolerance or abs(float(value) - bound[1]) <= tolerance for value, bound in zip(vector, bounds))
    return {
        "theta_standardized": float(vector[0]), "shape": float(vector[1]), "intercept": float(vector[2]),
        "objective": score, "optimizer_success": success, "optimum_on_boundary": boundary,
        "exact_evaluations": len(cache), "preview_evaluations": len(preview),
        "truth_q01": float(torch.quantile(targets, .01)), "truth_q99": float(torch.quantile(targets, .99)),
    }


def fit_oof(
    raw: torch.Tensor, truth: torch.Tensor, valid: torch.Tensor,
    field_stds: torch.Tensor, folds: list[list[int]], spec: dict[str, Any],
) -> tuple[torch.Tensor, list[dict[str, Any]], dict[int, dict[str, tuple[float, float]]]]:
    candidate = torch.empty(raw.shape, dtype=torch.float64)
    rows, thresholds = [], {}
    all_indices = set(range(raw.shape[0]))
    for fold_number, heldout in enumerate(folds):
        train = sorted(all_indices - set(heldout)); fits = {}
        for offset, field in enumerate(FIELDS):
            values, targets = _fit_arrays(raw, truth, valid, train, offset)
            fits[field] = _fit_field(values, targets, float(field_stds[offset]), field, spec)
        for case in heldout:
            thresholds[case] = {field: (fits[field]["truth_q01"], fits[field]["truth_q99"]) for field in FIELDS}
        for offset, field in enumerate(FIELDS):
            fit = fits[field]
            candidate[heldout, :, offset::2] = apply_hurdle_interior(
                raw[heldout, :, offset::2], float(field_stds[offset]), field,
                fit["theta_standardized"], fit["shape"], fit["intercept"],
            )
        rows.append({"fold": fold_number, "fit_indices": train, "heldout_indices": heldout, "field_fits": fits})
        _atomic_strict_json(Path(spec["progress_path"]), {"status": "fitting", "completed_folds": rows}) if spec.get("progress_path") else None
    if not torch.isfinite(candidate).all():
        raise FloatingPointError("OOF hurdle candidate contains NaN/Inf")
    return candidate, rows, thresholds


def run(config_path: Path, output: Path) -> dict[str, Any]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.device_count() != 0: raise RuntimeError("hurdle gate must be CPU-only")
    if os.environ.get("CLEARML_REQUIRE_ONLINE") != "1": raise RuntimeError("hurdle gate requires online ClearML")
    if output.exists() or output.is_symlink(): raise FileExistsError(f"refusing to replace {output}")
    config = load_json(config_path); output.mkdir(parents=True); tracker = None
    previous_handler = signal.getsignal(signal.SIGTERM); signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(TimeoutError("received SIGTERM")))
    try:
        torch.set_num_threads(6); torch.set_num_interop_threads(1)
        tracker = ClearMLTracker(config["project_name"], config["task_name"], tags=config["clearml"]["tags"], env_path=config["clearml"]["env_path"])
        tracker.connect("hurdle_interior_protocol", config); _atomic_strict_json(output / "status.json", {"status": "running", "clearml_task_id": str(tracker.task.id)})
        repository = Path(__file__).resolve().parents[1]; source = Path(config["source"]["path"])
        if not source.is_file() or _sha256(source) != config["source"]["sha256"]: raise ValueError("frozen champion artifact differs")
        identity = _clean_code_identity(repository); payload = torch.load(source, map_location="cpu", weights_only=True); _validate_source_contract(payload, config, repository)
        valid = payload["valid"].float(); mask_members = valid[:, None].expand(-1, 8, -1, -1, -1).flatten(0, 1)
        raw_normalized = payload["residual"].float() + smooth_right_inverse(payload["coarse"].float().flatten(0, 1), mask_members).unflatten(0, (12, 8))
        means = torch.tensor(config["normalization"]["means"]); stds = torch.tensor(config["normalization"]["stds"]); channel_means, channel_stds = means.repeat(3), stds.repeat(3)
        raw = channel_denormalize(raw_normalized.flatten(0, 1), channel_means, channel_stds).unflatten(0, (12, 8)); truth = channel_denormalize(payload["truth"].float(), channel_means, channel_stds)
        _validate_inputs(raw, truth, valid); folds = config["folds"]; validate_folds(folds, 12)
        support = apply_memberwise(raw, stds, {"sic_scale": 1., "sic_offset": 0., "sit_scale": 1., "sit_offset": 0.})
        fit_spec = {**config["fit"], "progress_path": str(output / "fit_progress.json")}
        candidate, fold_fits, thresholds = fit_oof(raw, truth, valid, stds, folds, fit_spec)
        provenance = {"source": config["source"], "normalization": config["normalization"], "case_ids": list(payload["case_ids"]), "folds": folds, "code_identity": identity}
        tensor_hash = _atomic_torch_save({"ensemble": candidate, "fold_fits": fold_fits, **provenance}, output / "oof_hurdle_calibrated.pt")
        raw_diag, support_diag, candidate_diag = (diagnostics(value, truth, valid, channel_stds) for value in (raw, support, candidate))
        versus_raw = compare(candidate_diag, raw_diag, config["gate"], 0); versus_support = compare(candidate_diag, support_diag, config["gate"], 1)
        gate = _gate(candidate_diag, raw_diag, support_diag, versus_raw, versus_support, fold_fits, config["gate"])
        extremes = {label: extreme_mean_mae(value, truth, valid, thresholds) for label, value in (("raw", raw), ("support_only", support), ("candidate", candidate))}
        for name, row in extremes["candidate"].items():
            for tail in ("low_quantile_threshold_mae", "high_quantile_threshold_mae"):
                value, reference = row[tail], extremes["support_only"][name][tail]
                if value is None or reference is None: gate["failures"].append(f"{name} {tail} selector empty")
                elif value > 1.01 * reference + 1e-8: gate["failures"].append(f"{name} {tail} worsened versus support")
        gate["passed_pending_visual_review"] = not gate["failures"]
        references = {}
        for label, reference in config["references"].items():
            path = Path(reference["path"])
            if _sha256(path) != reference["sha256"]: raise ValueError(f"{label} reference differs")
            loaded = json.loads(path.read_text()); references[label] = {"source_sha256": reference["sha256"], "primary": loaded["candidate"]["primary_standardized_fair_crps"], "mean_rank_tv": loaded["candidate"]["mean_rank_tv"], "gate_passed": loaded["gate"]["passed_pending_visual_review"]}
        extra = {"tie_fraction": {label: tie_fraction(value, valid) for label, value in (("raw", raw), ("support_only", support), ("candidate", candidate))}, "extreme_mean_mae": extremes}
        _require_finite_tree({"diagnostics": [raw_diag, support_diag, candidate_diag], "comparisons": [versus_raw, versus_support], "extra": extra})
        result = {"status": "scored_pending_visual_review", "training_performed": False, **provenance, "config_sha256": _sha256(config_path), "method": "single deterministic memberwise threshold plus smooth positive-interior transform; no resampling or rank objective", "fold_fits": fold_fits, "controls": {"raw": raw_diag, "support_only": support_diag}, "candidate": candidate_diag, "comparison": {"versus_raw": versus_raw, "versus_support_only": versus_support}, "prior_references": references, "additional_diagnostics": extra, "gate": {**gate, "training_or_replacement_permitted": False}, "tensor_sha256": {"oof_hurdle_calibrated.pt": tensor_hash}, "figures": [], "clearml_task_id": str(tracker.task.id)}
        _atomic_strict_json(output / "hurdle_interior_gate.json", result); tracker.upload_artifact("hurdle_interior_gate_scored", output / "hurdle_interior_gate.json")
        figures = _save_panels(output, list(payload["case_ids"]), truth, raw, support, candidate, valid)
        for path in figures: tracker.report_image("hurdle_interior_visual_qc", Path(path).stem, Path(path), 0)
        tracker.close(); tracker = None; result["status"] = "complete_pending_astra_review"; result["figures"] = figures
        _atomic_strict_json(output / "hurdle_interior_gate.json", result); _atomic_strict_json(output / "status.json", {"status": result["status"], "clearml_task_id": result["clearml_task_id"]}); return result
    except BaseException as error:
        failure = {"status": "failed", "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()}; _atomic_strict_json(output / "failure.json", failure); _atomic_strict_json(output / "status.json", {"status": "failed", "error_type": type(error).__name__, "error": str(error)})
        if tracker is not None: _finalize_failed_tracker(tracker, error, output / "failure.json")
        raise
    finally: signal.signal(signal.SIGTERM, previous_handler)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True, type=Path); parser.add_argument("--output", required=True, type=Path); args = parser.parse_args(); print(json.dumps(run(args.config.resolve(), args.output.resolve()), indent=2))


if __name__ == "__main__": main()
