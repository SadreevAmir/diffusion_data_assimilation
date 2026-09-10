"""Frozen train-only audit of the fine-cascade target and base measures.

This module does not train or load a model.  It compares the six-channel fine
target R=(I-UD)Y with the base sample eta=(I-UD)xi actually used by the flow.
Statistics are case-equal and computed on valid ocean.  Fourier statistics use
only fully-ocean patches, so land zeros cannot contaminate the spectrum.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import date
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Subset

from .config import load_json, merge_config_overrides, resolve_path
from .clearml_tracking import ClearMLTracker
from .data import build_dataset
from .direct_dynamics_cascade import masked_block_average, project_detail
from .direct_dynamics_cascade_fine_training import (
    EXPECTED_SPLIT_LENGTHS,
    _calendar_inventory,
    _canonical_sha256,
    _case_id,
    _clean_code_identity,
    _fine_collate,
    _sha256_file,
)
from .direct_dynamics_training import validate_direct_dataset
from .runtime import build_dataloader
from .trainer import _atomic_json

CHANNELS = ("d3_sic", "d3_sit", "d6_sic", "d6_sit", "d9_sic", "d9_sit")
SEASONAL_MONTH_DAYS = ((1, 15), (4, 15), (7, 15), (10, 15))
TRAIN_YEARS = tuple(range(2016, 2022))
SLICES_PER_DAY = 24


def frozen_seasonal_indices(dataset) -> tuple[list[int], list[str]]:
    """Select four declared days/year and every archive slice on those days."""
    day_by_date = {pair[1].date: index for index, pair in enumerate(dataset.calendar_pairs)}
    selected: list[int] = []
    ids: list[str] = []
    for year in TRAIN_YEARS:
        for month, day in SEASONAL_MONTH_DAYS:
            target_date = date(year, month, day)
            if target_date not in day_by_date:
                raise ValueError(f"frozen seasonal date is absent from train archive: {target_date}")
            day_index = day_by_date[target_date]
            for archive_slice in range(SLICES_PER_DAY):
                index = day_index * SLICES_PER_DAY + archive_slice
                selected.append(index)
                ids.append(_case_id(dataset, index))
    if len(selected) != 576 or len(set(selected)) != 576 or len(set(ids)) != 576:
        raise RuntimeError("frozen seasonal panel must contain exactly 576 unique cases")
    return selected, ids


def _case_equal_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if value.ndim != 4 or mask.shape != (value.shape[0], 1, *value.shape[-2:]):
        raise ValueError("value/mask shapes are incompatible")
    weights = mask.to(dtype=torch.float64).expand_as(value)
    data = value.to(dtype=torch.float64)
    support = weights > 0
    if not torch.isfinite(data[support]).all():
        raise FloatingPointError("audit statistic received NaN/Inf on valid ocean")
    denominator = weights.sum(dim=(2, 3))
    if torch.any(denominator <= 0):
        raise ValueError("every audit case must contain valid ocean")
    safe = torch.where(support, data, torch.zeros_like(data))
    return (safe * weights).sum(dim=(2, 3)) / denominator


def _region_masks(valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one-pixel fully-ocean interior and the complementary coast."""
    if valid.ndim != 4 or valid.shape[1] != 1:
        raise ValueError("valid mask must have shape [B,1,H,W]")
    binary = valid > 0
    neighbours = F.conv2d(binary.float(), torch.ones((1, 1, 3, 3)), padding=1)
    interior = binary & (neighbours == 9)
    coast = binary & ~interior
    return interior.float(), coast.float()


def _directional_increment_square(value: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dx = value[..., :, 1:] - value[..., :, :-1]
    dy = value[..., 1:, :] - value[..., :-1, :]
    mask_x = valid[..., :, 1:] * valid[..., :, :-1]
    mask_y = valid[..., 1:, :] * valid[..., :-1, :]
    return _case_equal_mean(dx.square(), mask_x), _case_equal_mean(dy.square(), mask_y)


def _phase_second_moment(value: torch.Tensor, valid: torch.Tensor, period: int) -> torch.Tensor:
    result = torch.empty((value.shape[0], value.shape[1], period, period), dtype=torch.float64)
    for iy in range(period):
        for ix in range(period):
            result[:, :, iy, ix] = _case_equal_mean(
                value[..., iy::period, ix::period].square(),
                valid[..., iy::period, ix::period],
            )
    return result


def _fully_ocean_patch_spectrum(
    value: torch.Tensor, valid: torch.Tensor, *, patch_size: int = 32
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return sums/counts of radial patch power for low/mid/high frequencies."""
    if value.shape[-2] % patch_size or value.shape[-1] % patch_size:
        raise ValueError("image dimensions must be divisible by patch size")
    patches = value.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    mask_patches = valid.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    fully_ocean = mask_patches.bool().all(dim=(-1, -2)).squeeze(1)
    window_1d = torch.hann_window(patch_size, periodic=False, dtype=torch.float64)
    window = window_1d[:, None] * window_1d[None, :]
    fy = torch.fft.fftfreq(patch_size, dtype=torch.float64)[:, None]
    fx = torch.fft.rfftfreq(patch_size, dtype=torch.float64)[None, :]
    radius = torch.sqrt(fy.square() + fx.square())
    bands = ((0.0, 0.125), (0.125, 0.25), (0.25, math.inf))
    sums = torch.zeros((value.shape[1], len(bands)), dtype=torch.float64)
    counts = torch.zeros(len(bands), dtype=torch.int64)
    packed = patches.permute(0, 2, 3, 1, 4, 5)[fully_ocean].to(dtype=torch.float64)
    if packed.numel() == 0:
        return sums, counts
    packed = (packed - packed.mean(dim=(-1, -2), keepdim=True)) * window
    power = torch.fft.rfft2(packed, norm="ortho").abs().square()
    for band_index, (lower, upper) in enumerate(bands):
        selector = (radius > lower) & (radius <= upper)
        sums[:, band_index] = power[..., selector].mean(dim=-1).sum(dim=0)
        counts[band_index] = packed.shape[0]
    return sums, counts


class _Accumulator:
    def __init__(self) -> None:
        self.case_count = 0
        self.second = {kind: torch.zeros(6, dtype=torch.float64) for kind in ("target", "base")}
        self.interior_second = {kind: torch.zeros(6, dtype=torch.float64) for kind in ("target", "base")}
        self.coast_second = {kind: torch.zeros(6, dtype=torch.float64) for kind in ("target", "base")}
        self.dx_second = {kind: torch.zeros(6, dtype=torch.float64) for kind in ("target", "base")}
        self.dy_second = {kind: torch.zeros(6, dtype=torch.float64) for kind in ("target", "base")}
        self.phase = {
            kind: {2: torch.zeros((6, 2, 2), dtype=torch.float64), 4: torch.zeros((6, 4, 4), dtype=torch.float64)}
            for kind in ("target", "base")
        }
        self.spectrum_sum = {kind: torch.zeros((6, 3), dtype=torch.float64) for kind in ("target", "base")}
        self.spectrum_count = {kind: torch.zeros(3, dtype=torch.int64) for kind in ("target", "base")}
        self.max_abs_coarse = {kind: 0.0 for kind in ("target", "base")}

    def update(self, target: torch.Tensor, base: torch.Tensor, valid: torch.Tensor) -> None:
        batch_size = target.shape[0]
        interior, coast = _region_masks(valid)
        for kind, value in (("target", target), ("base", base)):
            self.second[kind] += _case_equal_mean(value.square(), valid).sum(dim=0)
            self.interior_second[kind] += _case_equal_mean(value.square(), interior).sum(dim=0)
            self.coast_second[kind] += _case_equal_mean(value.square(), coast).sum(dim=0)
            dx, dy = _directional_increment_square(value, valid)
            self.dx_second[kind] += dx.sum(dim=0)
            self.dy_second[kind] += dy.sum(dim=0)
            for period in (2, 4):
                self.phase[kind][period] += _phase_second_moment(value, valid, period).sum(dim=0)
            spectrum_sum, spectrum_count = _fully_ocean_patch_spectrum(value, valid)
            self.spectrum_sum[kind] += spectrum_sum
            self.spectrum_count[kind] += spectrum_count
            coarse, fraction = masked_block_average(value.float(), valid.float())
            active = fraction > 0
            self.max_abs_coarse[kind] = max(
                self.max_abs_coarse[kind], float(coarse[active.expand_as(coarse)].abs().max())
            )
        self.case_count += batch_size

    def finalize(self) -> dict[str, Any]:
        if self.case_count != 576:
            raise RuntimeError(f"audit consumed {self.case_count} cases, expected 576")
        result: dict[str, Any] = {"case_count": self.case_count, "measures": {}}
        for kind in ("target", "base"):
            second = self.second[kind] / self.case_count
            result["measures"][kind] = {
                "rms": torch.sqrt(second).tolist(),
                "interior_rms": torch.sqrt(self.interior_second[kind] / self.case_count).tolist(),
                "coast_rms": torch.sqrt(self.coast_second[kind] / self.case_count).tolist(),
                "dx_rms": torch.sqrt(self.dx_second[kind] / self.case_count).tolist(),
                "dy_rms": torch.sqrt(self.dy_second[kind] / self.case_count).tolist(),
                "phase2_rms": torch.sqrt(self.phase[kind][2] / self.case_count).tolist(),
                "phase4_rms": torch.sqrt(self.phase[kind][4] / self.case_count).tolist(),
                "fully_ocean_patch_power_low_mid_high": (
                    self.spectrum_sum[kind]
                    / self.spectrum_count[kind].clamp_min(1).to(dtype=torch.float64)[None]
                ).tolist(),
                "fully_ocean_patch_counts_low_mid_high": self.spectrum_count[kind].tolist(),
                "max_abs_block_average": self.max_abs_coarse[kind],
            }
        target_second = self.second["target"] / self.case_count
        base_second = self.second["base"] / self.case_count
        scales = torch.sqrt(target_second / base_second)
        result["candidate_channel_scales"] = dict(zip(CHANNELS, scales.tolist(), strict=True))
        result["candidate_scale_formula"] = "sqrt(E_case<target^2>_ocean/E_case<base^2>_ocean)"
        return result


def run_audit(config_path: Path, output_path: Path) -> dict[str, Any]:
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"refusing to replace residual-law audit: {output_path}")
    experiment = load_json(config_path)
    clearml = experiment.get("clearml", {})
    if not clearml.get("enabled", False) or not clearml.get("env_path"):
        raise ValueError("residual-law audit requires explicit online ClearML")
    config_dir = config_path.resolve().parent
    data_path = resolve_path(experiment["data_config"], config_dir)
    data_config = merge_config_overrides(load_json(data_path), experiment.get("data_overrides"))
    repo_root = Path(__file__).resolve().parents[1]
    code_identity = _clean_code_identity(repo_root)
    dataset = build_dataset(data_config, split="train")
    if len(dataset) != EXPECTED_SPLIT_LENGTHS["train"]:
        raise ValueError("train archive length differs from audited inventory")
    dataset_validation = validate_direct_dataset(dataset)
    inventory = _calendar_inventory(dataset, split="train")
    indices, case_ids = frozen_seasonal_indices(dataset)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tracker = ClearMLTracker(
        experiment["project_name"],
        experiment["task_name"],
        tags=clearml.get("tags", []),
        env_path=clearml["env_path"],
    )
    protocol = {
        "split": "train_only",
        "case_count": len(indices),
        "selected_case_ids_sha256": _canonical_sha256(case_ids),
        "noise_seed": int(experiment.get("noise_seed", 82471)),
        "optimizer_steps": 0,
        "gpu_required": False,
    }
    tracker.connect("audit_protocol", protocol)
    subset = Subset(dataset, indices)
    loader = build_dataloader(
        subset,
        batch_size=int(experiment.get("batch_size", 4)),
        num_workers=int(experiment.get("num_workers", 4)),
        shuffle=False,
        collate_fn=_fine_collate,
        prefetch_factor=2,
    )
    generator = torch.Generator(device="cpu").manual_seed(int(experiment.get("noise_seed", 82471)))
    accumulator = _Accumulator()
    for raw in loader:
        truth = raw["truth"].float()
        valid = raw["valid_mask"][:, :1].float()
        target = project_detail(truth, valid)
        base = project_detail(torch.randn(truth.shape, generator=generator), valid)
        accumulator.update(target, base, valid)
    metrics = accumulator.finalize()
    result = {
        "status": "complete",
        "split": "train_only",
        "coordinate": "normalized fine detail; target=(I-UD)Y, base=(I-UD)xi",
        "channel_order": list(CHANNELS),
        "selection": "four frozen seasonal dates/year for 2016-2021; all 24 slices",
        "seasonal_month_days": [list(value) for value in SEASONAL_MONTH_DAYS],
        "selected_case_ids": case_ids,
        "selected_case_ids_sha256": _canonical_sha256(case_ids),
        "selected_indices_sha256": hashlib.sha256(",".join(map(str, indices)).encode()).hexdigest(),
        "noise_seed": int(experiment.get("noise_seed", 82471)),
        "dataset_validation": dataset_validation,
        "calendar_inventory": inventory,
        "data_config_sha256": _canonical_sha256(data_config),
        "data_config_file_sha256": _sha256_file(data_path),
        "experiment_sha256": _sha256_file(config_path),
        "builder_sha256": _sha256_file(Path(__file__)),
        "code_identity": code_identity,
        "clearml_task_id": str(tracker.task.id),
        **metrics,
    }
    _atomic_json(output_path, result)
    for channel, value in result["candidate_channel_scales"].items():
        tracker.report_single_value(f"candidate_scale/{channel}", value)
    tracker.upload_artifact("residual_law_audit", output_path)
    tracker.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(6)
    torch.set_num_interop_threads(1)
    print(json.dumps(run_audit(args.config.resolve(), args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
