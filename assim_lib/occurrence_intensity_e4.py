"""Executable publication-side oracle for the frozen E4 analog source."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch


REVIEWED_MODE = "occurrence_intensity_e4_engineering_sentinel"
FEATURE_NAMES = (
    "domain_mean", "domain_standard_deviation", "ice_extent",
    "established_ice_area", "meridional_centroid", "zonal_centroid",
)


def forecast_state_features(fields: torch.Tensor) -> torch.Tensor:
    """Return the six frozen forecast-only features for ``[N,H,W]`` fields."""
    if fields.ndim != 3 or not torch.is_floating_point(fields):
        raise ValueError("fields must be floating point with shape [N,H,W]")
    if not torch.all(torch.isfinite(fields)) or not torch.all((fields >= 0) & (fields <= 1)):
        raise ValueError("forecast fields must be finite and in [0,1]")
    n, height, width = fields.shape
    present = fields > 0
    established = fields > 0.15
    y = torch.linspace(0, 1, height, dtype=fields.dtype, device=fields.device).view(1, height, 1)
    x = torch.linspace(0, 1, width, dtype=fields.dtype, device=fields.device).view(1, 1, width)
    weights = fields
    mass = weights.sum(dim=(1, 2))
    safe_mass = torch.where(mass > 0, mass, torch.ones_like(mass))
    y_centroid = (weights * y).sum(dim=(1, 2)) / safe_mass
    x_centroid = (weights * x).sum(dim=(1, 2)) / safe_mass
    y_centroid = torch.where(mass > 0, y_centroid, torch.full_like(y_centroid, 0.5))
    x_centroid = torch.where(mass > 0, x_centroid, torch.full_like(x_centroid, 0.5))
    return torch.stack((
        fields.mean(dim=(1, 2)), fields.std(dim=(1, 2), unbiased=False),
        present.to(fields.dtype).mean(dim=(1, 2)),
        established.to(fields.dtype).mean(dim=(1, 2)), y_centroid, x_centroid,
    ), dim=1).reshape(n, len(FEATURE_NAMES))


def select_complete_analogs(
    training_features: torch.Tensor, query_features: torch.Tensor, *, neighbors: int = 10
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select analog indices using train-only scaling and chronological ties."""
    if training_features.ndim != 2 or query_features.ndim != 2:
        raise ValueError("features must have shape [N,6] and [Q,6]")
    if training_features.shape[1] != 6 or query_features.shape[1] != 6:
        raise ValueError("E4 requires exactly six forecast-only features")
    if training_features.shape[0] < neighbors or neighbors != 10:
        raise ValueError("E4 requires exactly ten training analogs")
    if not torch.all(torch.isfinite(training_features)) or not torch.all(torch.isfinite(query_features)):
        raise ValueError("features must be finite")
    mean = training_features.mean(dim=0)
    scale = training_features.std(dim=0, unbiased=False)
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    train_z = (training_features - mean) / scale
    query_z = (query_features - mean) / scale
    distances = torch.cdist(query_z, train_z)
    # stable=True makes the original chronological training index the tie-break.
    order = torch.argsort(distances, dim=1, stable=True)[:, :neighbors]
    return order, torch.gather(distances, 1, order)


def add_complete_residual_fields(
    base_members: torch.Tensor, training_residuals: torch.Tensor, analog_indices: torch.Tensor
) -> torch.Tensor:
    """Add one selected residual as a complete field, without clipping."""
    if base_members.ndim != 4 or training_residuals.ndim != 3:
        raise ValueError("base_members and residuals must be [Q,M,H,W] and [N,H,W]")
    if analog_indices.shape != base_members.shape[:2] or base_members.shape[-2:] != training_residuals.shape[-2:]:
        raise ValueError("analog assignment shape differs from complete fields")
    if not torch.all(torch.isfinite(base_members)) or not torch.all(torch.isfinite(training_residuals)):
        raise ValueError("base members and residual fields must be finite")
    selected = training_residuals[analog_indices]
    result = base_members + selected
    if not torch.all((result >= 0) & (result <= 1)):
        raise ValueError("E4 result outside physical support; clipping is prohibited")
    return result


def validate_config(config: dict[str, Any]) -> None:
    expected = {
        "mode": REVIEWED_MODE, "project_name": "generative-sea-ice-da",
        "task_name": "occurrence-intensity-e4-engineering-sentinel",
        "clearml": {"enabled": True}, "cases": 8,
        "selection_role": "engineering_only", "fit_scope": "training_fold_only",
        "features": list(FEATURE_NAMES), "neighbors": 10,
        "tie_break": "chronological_training_index", "whole_residual_fields": True,
        "hard_clipping": False,
    }
    if config != expected:
        raise ValueError("E4 config differs from the frozen engineering contract")


def controller_request(config_path: str | Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_bytes())
    validate_config(config)
    return {"reviewed_mode": REVIEWED_MODE, "resource_kind": "server_gpu",
            "admission_required": True, "launch_authorized": False, "cases": 8,
            "scientific_gate": False}


def run_engineering_sentinel(config_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    config_bytes = Path(config_path).read_bytes()
    config = json.loads(config_bytes)
    validate_config(config)
    generator = torch.Generator().manual_seed(4404)
    training = 0.3 + 0.4 * torch.rand((12, 5, 4), generator=generator)
    queries = training[:8].clone()
    features = forecast_state_features(training)
    query_features = forecast_state_features(queries)
    indices, distances = select_complete_analogs(features, query_features)
    residuals = 0.01 * (torch.rand((12, 5, 4), generator=generator) - 0.5)
    base = queries[:, None].repeat(1, 10, 1, 1)
    result = add_complete_residual_fields(base, residuals, indices)
    status = {"status": "completed", "selection_role": "engineering_only",
              "cases": 8, "scientific_gate": False}
    manifest = {
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "feature_names": list(FEATURE_NAMES), "training_records": 12,
        "analog_indices": indices.tolist(), "analog_distances": distances.tolist(),
        "training_only_standardization": True, "chronological_ties": True,
        "complete_field_shape": list(residuals.shape[1:]),
        "result_shape": list(result.shape), "clipping_calls": 0,
        "raw_arrays": "not_persisted",
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "run_status.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    (output / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {"run_status": status, "artifact_manifest": manifest}
