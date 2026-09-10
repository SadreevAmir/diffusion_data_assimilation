"""Build immutable train-only diagonal statistics for coarse persistence residuals."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import torch
from torch.utils.data import Subset

from .config import load_json, merge_config_overrides, resolve_path
from .data import build_dataset
from .direct_dynamics_cascade_coarse import coarse_target
from .direct_dynamics_cascade_coarse_residual import coarse_persistence_from_condition
from .direct_dynamics_cascade_fine_training import (
    EXPECTED_SPLIT_LENGTHS,
    PREFETCH_FACTOR,
    _canonical_sha256,
    _clean_code_identity,
    _evenly_spaced_indices,
    _fine_collate,
    _indices_sha256,
    _sha256_file,
)
from .runtime import build_dataloader
from .trainer import _atomic_json


class CaseEqualResidualMoments:
    """Accumulate case-equal, ocean-fraction-weighted first and second moments."""

    def __init__(self, channels: int = 6):
        if channels <= 0:
            raise ValueError("residual moments require a positive channel count")
        self.channels = channels
        self.case_count = 0
        self.first = torch.zeros(channels, dtype=torch.float64)
        self.second = torch.zeros(channels, dtype=torch.float64)
        self.minimum = torch.full((channels,), math.inf, dtype=torch.float64)
        self.maximum = torch.full((channels,), -math.inf, dtype=torch.float64)

    def update(self, residual: torch.Tensor, ocean_fraction: torch.Tensor) -> None:
        if residual.ndim != 4 or residual.shape[1] != self.channels:
            raise ValueError("residual moment batch has the wrong shape")
        if ocean_fraction.shape != (residual.shape[0], 1, *residual.shape[-2:]):
            raise ValueError("residual moment weights have the wrong shape")
        value = residual.to(dtype=torch.float64, device="cpu")
        weight = ocean_fraction.to(dtype=torch.float64, device="cpu")
        if not torch.isfinite(weight).all() or torch.any((weight < 0) | (weight > 1)):
            raise ValueError("residual moment weights must be finite in [0,1]")
        support = weight.expand_as(value) > 0
        if not torch.isfinite(value[support]).all():
            raise FloatingPointError("residual moments received NaN/Inf on active ocean")
        denominator = weight.sum(dim=(2, 3))
        if torch.any(denominator <= 0):
            raise ValueError("every residual statistics case must contain ocean")
        safe = torch.where(support, value, torch.zeros_like(value))
        case_first = (safe * weight).sum(dim=(2, 3)) / denominator
        case_second = (safe.square() * weight).sum(dim=(2, 3)) / denominator
        self.first += case_first.sum(dim=0)
        self.second += case_second.sum(dim=0)
        for channel in range(self.channels):
            selected = value[:, channel][support[:, channel]]
            self.minimum[channel] = torch.minimum(self.minimum[channel], selected.min())
            self.maximum[channel] = torch.maximum(self.maximum[channel], selected.max())
        self.case_count += residual.shape[0]

    def finalize(self) -> dict[str, list[float] | int]:
        if self.case_count <= 1:
            raise ValueError("residual statistics require at least two cases")
        mean = self.first / self.case_count
        variance = self.second / self.case_count - mean.square()
        if not torch.isfinite(variance).all() or torch.any(variance <= 0):
            raise ValueError("residual statistics contain zero, negative, or non-finite variance")
        std = torch.sqrt(variance)
        return {
            "case_count": self.case_count,
            "means": mean.tolist(),
            "stds": std.tolist(),
            "variances": variance.tolist(),
            "minimums": self.minimum.tolist(),
            "maximums": self.maximum.tolist(),
        }


def build_statistics(config_path: Path, output_path: Path) -> dict:
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"refusing to replace residual statistics: {output_path}")
    experiment = load_json(config_path)
    config_dir = config_path.resolve().parent
    data_path = resolve_path(experiment["data_config"], config_dir)
    data_config = merge_config_overrides(load_json(data_path), experiment.get("data_overrides"))
    pilot = experiment.get("pilot", {})
    train_case_count = int(pilot.get("train_case_count", -1))
    if train_case_count != 4096:
        raise ValueError("residual statistics require the exact 4096-case pilot subset")
    repo_root = Path(__file__).resolve().parents[1]
    code_identity = _clean_code_identity(repo_root)
    dataset = build_dataset(data_config, split="train")
    if len(dataset) != EXPECTED_SPLIT_LENGTHS["train"]:
        raise ValueError("train archive length differs from its audited inventory")
    indices = _evenly_spaced_indices(len(dataset), train_case_count)
    subset = Subset(dataset, indices)
    loader = build_dataloader(
        subset,
        batch_size=8,
        num_workers=4,
        shuffle=False,
        collate_fn=_fine_collate,
        prefetch_factor=PREFETCH_FACTOR,
    )
    moments = CaseEqualResidualMoments()
    for raw in loader:
        valid = raw["valid_mask"][:, :1].float()
        clean, active, fraction = coarse_target(raw["truth"], valid)
        persistence, persistence_active, persistence_fraction = coarse_persistence_from_condition(
            raw["structured_conditioning"], valid
        )
        if not torch.equal(active, persistence_active) or not torch.equal(
            fraction, persistence_fraction
        ):
            raise ValueError("residual statistics target and persistence supports differ")
        residual = torch.where(active.expand_as(clean) > 0, clean - persistence, 0.0)
        moments.update(residual, fraction)
    summary = moments.finalize()
    if summary["case_count"] != train_case_count:
        raise RuntimeError("residual statistics did not consume the declared training subset")
    artifact = {
        "status": "complete",
        "split": "train",
        "estimator": "case-equal ocean-fraction-weighted population moments",
        "coordinate": "normalized coarse R=C-P for d3/d6/d9 SIC/SIT",
        "channel_order": ["d3_sic", "d3_sit", "d6_sic", "d6_sit", "d9_sic", "d9_sit"],
        "train_case_count": train_case_count,
        "train_indices_sha256": _indices_sha256(indices),
        "data_config_sha256": _canonical_sha256(data_config),
        "data_config_path": str(data_path.resolve()),
        "data_config_file_sha256": _sha256_file(data_path),
        "source_experiment_path": str(config_path.resolve()),
        "source_experiment_sha256": _sha256_file(config_path),
        "builder_sha256": _sha256_file(Path(__file__)),
        "code_identity": code_identity,
        **summary,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_path, artifact)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(build_statistics(arguments.config.resolve(), arguments.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
