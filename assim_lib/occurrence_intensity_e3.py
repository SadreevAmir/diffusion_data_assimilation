"""Executable publication-side oracle for the frozen E3 anamorphosis."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

REVIEWED_MODE = "occurrence_intensity_e3_engineering_sentinel"


@dataclass(frozen=True)
class InteriorEmpiricalLaw:
    values: torch.Tensor
    probabilities: torch.Tensor


def fit_interior_law(training: torch.Tensor) -> InteriorEmpiricalLaw:
    """Fit a deterministic mid-rank empirical law to training interior only."""
    if not torch.is_floating_point(training) or not torch.all(torch.isfinite(training)):
        raise ValueError("training concentration must be finite floating point")
    if not torch.all((training >= 0) & (training <= 1)):
        raise ValueError("training concentration must be in [0,1]")
    interior = training[(training > 0) & (training < 1)].flatten()
    if interior.numel() < 2:
        raise ValueError("E3 requires at least two training interior values")
    values, counts = torch.unique(interior, sorted=True, return_counts=True)
    cumulative = torch.cumsum(counts, dim=0)
    starts = cumulative - counts
    probabilities = (starts + (counts.to(training.dtype) + 1.0) / 2.0) / (interior.numel() + 1.0)
    return InteriorEmpiricalLaw(values=values, probabilities=probabilities)


def _linear_map(x: torch.Tensor, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if source.numel() == 1:
        return torch.full_like(x, target[0])
    indices = torch.searchsorted(source, x.contiguous())
    right = indices.clamp(1, source.numel() - 1)
    left = right - 1
    weight = (x - source[left]) / (source[right] - source[left])
    mapped = target[left] + weight * (target[right] - target[left])
    return torch.where(x <= source[0], target[0], torch.where(x >= source[-1], target[-1], mapped))


def _validate_input(values: torch.Tensor, law: InteriorEmpiricalLaw) -> None:
    if not torch.is_floating_point(values) or not torch.all(torch.isfinite(values)):
        raise ValueError("concentration must be finite floating point")
    if not torch.all((values >= 0) & (values <= 1)):
        raise ValueError("concentration must be in [0,1]; clipping is prohibited")
    if law.values.ndim != 1 or law.probabilities.shape != law.values.shape:
        raise ValueError("invalid E3 empirical law")
    if law.values.numel() == 0 or not torch.all((law.values > 0) & (law.values < 1)):
        raise ValueError("law support must be strictly interior")
    if not torch.all(torch.diff(law.values) > 0) or not torch.all(torch.diff(law.probabilities) > 0):
        raise ValueError("law knots must be strictly increasing")


def transform_interior(concentration: torch.Tensor, law: InteriorEmpiricalLaw) -> torch.Tensor:
    """Map interior values to quantiles while preserving both atoms exactly."""
    _validate_input(concentration, law)
    result = concentration.clone()
    mask = (concentration > 0) & (concentration < 1)
    result[mask] = _linear_map(concentration[mask], law.values, law.probabilities)
    return result


def inverse_interior(coordinates: torch.Tensor, atom_template: torch.Tensor,
                     law: InteriorEmpiricalLaw) -> torch.Tensor:
    """Invert interior coordinates; atom decisions come from the frozen template."""
    _validate_input(atom_template, law)
    if coordinates.shape != atom_template.shape or not torch.all(torch.isfinite(coordinates)):
        raise ValueError("coordinates must be finite and match atom_template")
    result = atom_template.clone()
    mask = (atom_template > 0) & (atom_template < 1)
    if not torch.all((coordinates[mask] >= law.probabilities[0]) &
                     (coordinates[mask] <= law.probabilities[-1])):
        raise ValueError("interior coordinate lies outside the fitted inverse support")
    result[mask] = _linear_map(coordinates[mask], law.probabilities, law.values)
    return result


def validate_config(config: dict[str, Any]) -> None:
    expected = {
        "mode": REVIEWED_MODE, "project_name": "generative-sea-ice-da",
        "task_name": "occurrence-intensity-e3-engineering-sentinel",
        "clearml": {"enabled": True}, "cases": 8,
        "selection_role": "engineering_only", "source_stage": "accepted_e1_or_e2",
        "fit_scope": "training_fold_only", "cdf": "midrank_deterministic_ties",
        "boundary_atoms": [0.0, 1.0], "hard_clipping": False, "roundtrip_atol": 1e-7,
    }
    if config != expected:
        raise ValueError("E3 config differs from the frozen engineering contract")


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
    training = torch.tensor([0.0, 1e-6, 0.03, 0.03, 0.1, 0.4, 0.75, 0.999, 1.0])
    law = fit_interior_law(training)
    sample = torch.tensor([0.0, 1e-6, 0.03, 0.1, 0.4, 0.75, 0.999, 1.0])
    coordinates = transform_interior(sample, law)
    restored = inverse_interior(coordinates, sample, law)
    error = float(torch.max(torch.abs(restored - sample)))
    if error > config["roundtrip_atol"] or not torch.equal(restored[[0, -1]], sample[[0, -1]]):
        raise ValueError("E3 round-trip or atom preservation failed")
    status = {"status": "completed", "selection_role": "engineering_only",
              "cases": config["cases"], "scientific_gate": False}
    manifest = {"config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                "training_interior_count": int(((training > 0) & (training < 1)).sum()),
                "unique_knot_count": law.values.numel(), "roundtrip_max_abs_error": error,
                "zero_atoms_preserved": True, "one_atoms_preserved": True,
                "clipping_calls": 0, "raw_arrays": "not_persisted"}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "run_status.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    (output / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {"run_status": status, "artifact_manifest": manifest}
