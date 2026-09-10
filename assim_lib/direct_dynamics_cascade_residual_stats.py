"""Build immutable train-only diagonal statistics for coarse persistence residuals."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Subset

from .config import load_json, merge_config_overrides, resolve_path
from .data import build_dataset
from .direct_dynamics_cascade_coarse import coarse_target
from .direct_dynamics_cascade_coarse_residual import coarse_persistence_from_condition
from .direct_dynamics_cascade_fine_training import (
    EXPECTED_SPLIT_LENGTHS,
    PREFETCH_FACTOR,
    SHM_SAFETY_FRACTION,
    _batch_tensor_bytes,
    _calendar_inventory,
    _canonical_sha256,
    _clean_code_identity,
    _evenly_spaced_indices,
    _fine_collate,
    _indices_sha256,
    _sha256_file,
)
from .direct_dynamics_training import validate_direct_dataset
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


def _case_id(dataset, index: int) -> str:
    day_index, archive_slice = divmod(index, 24)
    target = dataset.calendar_pairs[day_index][1]
    return f"{target.date.isoformat()}_slice{archive_slice:02d}"


def _tensor_sha256(value: torch.Tensor) -> str:
    canonical = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(canonical.shape)).encode())
    digest.update(str(canonical.dtype).encode())
    digest.update(canonical.numpy().tobytes())
    return digest.hexdigest()


def residual_stats_ipc_preflight(
    subset,
    *,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    available_bytes: int | None = None,
) -> dict[str, int | float]:
    """Bound full-collate DataLoader IPC before any worker is constructed."""
    if batch_size <= 0 or num_workers < 0 or prefetch_factor <= 0:
        raise ValueError("residual statistics loader dimensions must be positive")
    single_sample_bytes = _batch_tensor_bytes(_fine_collate([subset[0]]))
    retained_batches = 1 + num_workers * prefetch_factor
    estimated = single_sample_bytes * batch_size * retained_batches
    if available_bytes is None:
        stats = os.statvfs("/dev/shm")
        available_bytes = int(stats.f_bavail * stats.f_frsize)
    budget = int(SHM_SAFETY_FRACTION * available_bytes)
    if estimated > budget:
        raise RuntimeError(
            "residual statistics DataLoader IPC estimate exceeds shared-memory safety budget: "
            f"estimated={estimated}, budget={budget}, available={available_bytes}"
        )
    return {
        "single_sample_bytes": single_sample_bytes,
        "retained_batches": retained_batches,
        "estimated_peak_ipc_bytes": estimated,
        "shared_memory_available_bytes": available_bytes,
        "safety_fraction": SHM_SAFETY_FRACTION,
        "prefetch_factor": prefetch_factor,
    }


def _make_stats_loader(
    subset,
    *,
    batch_size: int = 8,
    num_workers: int = 4,
    prefetch_factor: int = PREFETCH_FACTOR,
    available_bytes: int | None = None,
) -> tuple[Any, dict[str, int | float]]:
    ipc = residual_stats_ipc_preflight(
        subset,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        available_bytes=available_bytes,
    )
    loader = build_dataloader(
        subset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=_fine_collate,
        prefetch_factor=prefetch_factor,
    )
    return loader, ipc


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
    dataset_validation = validate_direct_dataset(dataset)
    calendar_inventory = _calendar_inventory(dataset, split="train")
    indices = _evenly_spaced_indices(len(dataset), train_case_count)
    selected_case_ids = [_case_id(dataset, index) for index in indices]
    subset = Subset(dataset, indices)
    reference_valid_mask = subset[0]["valid_mask"][:1].contiguous()
    if not torch.isfinite(reference_valid_mask).all():
        raise ValueError("static valid mask contains NaN/Inf")
    static_valid_mask_sha256 = _tensor_sha256(reference_valid_mask)
    loader, ipc = _make_stats_loader(subset)
    moments = CaseEqualResidualMoments()
    for raw in loader:
        valid = raw["valid_mask"][:, :1].float()
        expected_valid = reference_valid_mask.to(dtype=valid.dtype).expand_as(valid)
        if not torch.equal(valid, expected_valid):
            raise ValueError("selected training cases do not share one static valid mask")
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
        "selected_case_ids": selected_case_ids,
        "selected_case_ids_sha256": _canonical_sha256(selected_case_ids),
        "train_indices_sha256": _indices_sha256(indices),
        "ordered_inventory_sha256": calendar_inventory["ordered_inventory_sha256"],
        "static_valid_mask_sha256": static_valid_mask_sha256,
        "dataset_validation": dataset_validation,
        "calendar_inventory": calendar_inventory,
        "ipc_preflight": ipc,
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
    torch.set_num_threads(6)
    torch.set_num_interop_threads(1)
    print(json.dumps(build_statistics(arguments.config.resolve(), arguments.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
