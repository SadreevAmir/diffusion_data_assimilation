"""CPU-only audit of the frozen coarse target, increments, and base measure."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Subset

from .clearml_tracking import ClearMLTracker
from .config import load_json, merge_config_overrides, resolve_path
from .data import build_dataset
from .direct_dynamics_cascade_coarse import coarse_target
from .direct_dynamics_cascade_coarse_residual import coarse_persistence_from_condition
from .direct_dynamics_cascade_fine_training import (
    EXPECTED_SPLIT_LENGTHS,
    _canonical_sha256,
    _clean_code_identity,
    _fine_collate,
)
from .direct_dynamics_cascade_residual_law_audit import frozen_seasonal_indices
from .direct_dynamics_cascade_spatial_calibration_audit import (
    CHANNELS,
    _case_mean,
    _case_ocean_patch_power,
    _region_masks,
)
from .direct_dynamics_training import validate_direct_dataset
from .runtime import build_dataloader
from .trainer import _atomic_json


KINDS = ("absolute_target", "persistence_increment", "actual_absolute_base")
REGIONS = ("ocean", "interior", "coast")
TRAIN_DATE_BLOCKS = 24
SLICES_PER_DATE = 24


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _weighted_region_masks(active: torch.Tensor, fraction: torch.Tensor) -> dict[str, torch.Tensor]:
    regions = _region_masks(active)
    return {name: mask * fraction for name, mask in regions.items()}


def _case_channel_covariance(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Per-case 6x6 spatial covariance after removing each case/channel DC."""
    means = _case_mean(value, weight)
    centered = value.double() - means[..., None, None]
    support = weight.double()
    denominator = support.sum(dim=(-2, -1)).squeeze(1)
    weighted = centered * torch.sqrt(support)
    return torch.einsum("bihw,bjhw->bij", weighted, weighted) / denominator[:, None, None]


def _case_channel_second_cross(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Per-case raw second cross-moment, retaining DC variability."""
    support = weight.double()
    denominator = support.sum(dim=(-2, -1)).squeeze(1)
    weighted = value.double() * torch.sqrt(support)
    return torch.einsum("bihw,bjhw->bij", weighted, weighted) / denominator[:, None, None]


def _date_blocks(value: torch.Tensor) -> torch.Tensor:
    if value.shape[0] != TRAIN_DATE_BLOCKS * SLICES_PER_DATE:
        raise ValueError("train statistic does not contain 24 dates x 24 slices")
    return value.reshape(TRAIN_DATE_BLOCKS, SLICES_PER_DATE, *value.shape[1:]).mean(dim=1)


def _bootstrap_mean_ci(value: torch.Tensor, seed: int, *, draws: int = 10000) -> dict[str, Any]:
    if value.shape[0] != TRAIN_DATE_BLOCKS:
        raise ValueError("bootstrap unit must be the 24 frozen train dates")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(TRAIN_DATE_BLOCKS, (draws, TRAIN_DATE_BLOCKS), generator=generator)
    sampled = value[indices].mean(dim=1)
    return {
        "estimate": value.mean(dim=0).tolist(),
        "date_block_bootstrap_95_ci_low": torch.quantile(sampled, 0.025, dim=0).tolist(),
        "date_block_bootstrap_95_ci_high": torch.quantile(sampled, 0.975, dim=0).tolist(),
        "bootstrap_draws": draws,
    }


def _leave_one_year_out(value: torch.Tensor) -> list[dict[str, Any]]:
    if value.shape[0] != 24:
        raise ValueError("leave-one-year audit requires 24 date blocks")
    rows = []
    for year_index, year in enumerate(range(2016, 2022)):
        keep = torch.ones(24, dtype=torch.bool)
        keep[4 * year_index : 4 * (year_index + 1)] = False
        rows.append({"excluded_year": year, "mean": value[keep].mean(dim=0).tolist()})
    return rows


class _TrainAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self.dc = {kind: [] for kind in KINDS}
        self.second = {kind: {region: [] for region in REGIONS} for kind in KINDS}
        self.covariance = {kind: {region: [] for region in REGIONS} for kind in KINDS}
        self.raw_cross = {kind: {region: [] for region in REGIONS} for kind in KINDS}
        self.spectrum = {kind: [] for kind in KINDS}

    def update(
        self, target: torch.Tensor, persistence: torch.Tensor, base: torch.Tensor,
        active: torch.Tensor, fraction: torch.Tensor,
    ) -> None:
        values = {
            "absolute_target": target,
            "persistence_increment": target - persistence,
            "actual_absolute_base": base,
        }
        masks = _weighted_region_masks(active, fraction)
        for kind, value in values.items():
            self.dc[kind].append(_case_mean(value, masks["ocean"]))
            for region, mask in masks.items():
                self.second[kind][region].append(_case_mean(value.square(), mask))
                self.covariance[kind][region].append(_case_channel_covariance(value, mask))
                self.raw_cross[kind][region].append(_case_channel_second_cross(value, mask))
            self.spectrum[kind].append(_case_ocean_patch_power(value[:, None], active))
        self.count += target.shape[0]

    def finalize(self) -> dict[str, Any]:
        if self.count != 576:
            raise RuntimeError(f"coarse audit consumed {self.count} cases, expected 576")
        result: dict[str, Any] = {"case_count": self.count, "measures": {}}
        for kind in KINDS:
            dc = torch.cat(self.dc[kind])
            dc_blocks = _date_blocks(dc)
            row: dict[str, Any] = {
                "case_channel_dc": dc.tolist(),
                "mean_dc_by_channel": dc.mean(dim=0).tolist(),
                "std_case_dc_by_channel": dc.std(dim=0, unbiased=True).tolist(),
                "regions": {},
                "date_blocked": {
                    "unit": "calendar_date_with_all_24_slices",
                    "date_count": TRAIN_DATE_BLOCKS,
                    "dc": _bootstrap_mean_ci(dc_blocks, 84001),
                    "dc_leave_one_year_out": _leave_one_year_out(dc_blocks),
                    "regions": {},
                },
            }
            for region in REGIONS:
                second = torch.cat(self.second[kind][region])
                covariance = torch.cat(self.covariance[kind][region])
                raw_cross = torch.cat(self.raw_cross[kind][region])
                row["regions"][region] = {
                    "rms_by_channel": torch.sqrt(second.mean(dim=0)).tolist(),
                    "mean_spatial_covariance_6x6": covariance.mean(dim=0).tolist(),
                    "mean_raw_second_cross_moment_6x6": raw_cross.mean(dim=0).tolist(),
                }
                second_blocks = _date_blocks(second)
                covariance_blocks = _date_blocks(covariance)
                raw_cross_blocks = _date_blocks(raw_cross)
                row["date_blocked"]["regions"][region] = {
                    "second_moment": _bootstrap_mean_ci(second_blocks, 84101),
                    "spatial_covariance": _bootstrap_mean_ci(covariance_blocks, 84201),
                    "raw_second_cross_moment": _bootstrap_mean_ci(raw_cross_blocks, 84301),
                    "second_moment_leave_one_year_out": _leave_one_year_out(second_blocks),
                    "spatial_covariance_leave_one_year_out": _leave_one_year_out(covariance_blocks),
                }
            spectrum = torch.cat(self.spectrum[kind])
            if not torch.isfinite(spectrum).all():
                raise FloatingPointError("train panel lacks a fully-ocean coarse spectral patch")
            row["fully_ocean_patch_power_low_mid_high"] = spectrum.mean(dim=0).tolist()
            spectrum_blocks = _date_blocks(spectrum)
            row["date_blocked"]["fully_ocean_patch_power_low_mid_high"] = _bootstrap_mean_ci(
                spectrum_blocks, 84401
            )
            row["date_blocked"]["spectrum_leave_one_year_out"] = _leave_one_year_out(
                spectrum_blocks
            )
            result["measures"][kind] = row
        return result


def _validation_coarse_diagnostic(payload: dict[str, Any], dataset, indices: list[int]) -> dict[str, Any]:
    batch = _fine_collate([dataset[index] for index in indices])
    case_ids = tuple(batch["meta"]["case_id"])
    if tuple(payload["case_ids"]) != case_ids:
        raise ValueError("validation case identities differ from frozen coarse ensemble")
    valid = batch["valid_mask"][:, :1].float()
    truth, active, fraction = coarse_target(batch["truth"].float(), valid)
    persistence, persistence_active, persistence_fraction = coarse_persistence_from_condition(
        batch["structured_conditioning"].float(), valid
    )
    if not torch.equal(active, persistence_active) or not torch.equal(fraction, persistence_fraction):
        raise ValueError("validation target and persistence supports differ")
    members = payload["coarse"].float()
    if members.shape != (12, 8, 6, *truth.shape[-2:]):
        raise ValueError("frozen coarse ensemble shape differs")
    mean = members.mean(dim=1)
    anomaly = members - mean[:, None]
    variance = anomaly.square().sum(dim=1) / 7
    error = mean - truth
    persistence_error = persistence - truth
    masks = _weighted_region_masks(active, fraction)
    regions = {}
    for region, mask in masks.items():
        bias = _case_mean(error, mask)
        ensemble_variance = _case_mean(variance, mask)
        mean_error = _case_mean(error.square(), mask)
        persistence_mse = _case_mean(persistence_error.square(), mask)
        generator = torch.Generator().manual_seed(85001)
        bootstrap_indices = torch.randint(12, (10000, 12), generator=generator)
        sampled_bias = bias[bootstrap_indices].mean(dim=1)
        factor = 1.0 + 1.0 / 8.0
        sampled_ssr = torch.sqrt(
            factor * ensemble_variance[bootstrap_indices].mean(dim=(1, 2))
            / mean_error[bootstrap_indices].mean(dim=(1, 2)).clamp_min(torch.finfo(torch.float64).tiny)
        )
        regions[region] = {
            "case_channel_signed_bias": bias.tolist(),
            "case_channel_ensemble_variance": ensemble_variance.tolist(),
            "case_channel_squared_mean_error": mean_error.tolist(),
            "case_channel_persistence_squared_error": persistence_mse.tolist(),
            "date_block_bootstrap": {
                "mean_signed_bias_by_channel": bias.mean(dim=0).tolist(),
                "mean_signed_bias_95_ci_low": torch.quantile(sampled_bias, 0.025, dim=0).tolist(),
                "mean_signed_bias_95_ci_high": torch.quantile(sampled_bias, 0.975, dim=0).tolist(),
                "pooled_adjusted_ssr": float(torch.sqrt(factor * ensemble_variance.mean() / mean_error.mean())),
                "pooled_adjusted_ssr_95_ci_low": float(torch.quantile(sampled_ssr, 0.025)),
                "pooled_adjusted_ssr_95_ci_high": float(torch.quantile(sampled_ssr, 0.975)),
                "bootstrap_draws": 10000,
            },
        }
    spectra = {
        "ensemble_anomaly": _case_ocean_patch_power(anomaly, active).tolist(),
        "ensemble_mean_error": _case_ocean_patch_power(error[:, None], active).tolist(),
        "persistence_error": _case_ocean_patch_power(persistence_error[:, None], active).tolist(),
    }
    return {"regions": regions, "fully_ocean_patch_power_low_mid_high": spectra}


def run(config_path: Path, output: Path) -> dict[str, Any]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.device_count() != 0:
        raise RuntimeError("coarse reference audit must be CPU-only")
    if os.environ.get("CLEARML_REQUIRE_ONLINE") != "1":
        raise RuntimeError("coarse reference audit requires online ClearML")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to replace {output}")
    torch.set_num_threads(6)
    torch.set_num_interop_threads(1)
    experiment = load_json(config_path)
    config_dir = config_path.resolve().parent
    data_path = resolve_path(experiment["data_config"], config_dir)
    data_config = merge_config_overrides(load_json(data_path), experiment.get("data_overrides"))
    train = build_dataset(data_config, split="train")
    valid_dataset = build_dataset(data_config, split="valid")
    if len(train) != EXPECTED_SPLIT_LENGTHS["train"]:
        raise ValueError("train archive length differs from frozen protocol")
    validate_direct_dataset(train)
    validate_direct_dataset(valid_dataset)
    train_indices, train_ids = frozen_seasonal_indices(train)
    loader = build_dataloader(
        Subset(train, train_indices), batch_size=int(experiment.get("batch_size", 4)),
        num_workers=int(experiment.get("num_workers", 4)), shuffle=False,
        collate_fn=_fine_collate, prefetch_factor=2,
    )
    generator = torch.Generator().manual_seed(int(experiment["noise_seed"]))
    accumulator = _TrainAccumulator()
    for raw in loader:
        valid = raw["valid_mask"][:, :1].float()
        target, active, fraction = coarse_target(raw["truth"].float(), valid)
        persistence, persistence_active, persistence_fraction = coarse_persistence_from_condition(
            raw["structured_conditioning"].float(), valid
        )
        if not torch.equal(active, persistence_active) or not torch.equal(fraction, persistence_fraction):
            raise ValueError("train target and persistence supports differ")
        noise = torch.randn(target.shape, generator=generator)
        base = torch.where(active.expand_as(target) > 0, noise, torch.zeros_like(noise))
        accumulator.update(target, persistence, base, active, fraction)
    train_metrics = accumulator.finalize()

    source = experiment["validation_ensemble"]
    source_path = Path(source["path"])
    if not source_path.is_file() or _sha256(source_path) != source["sha256"]:
        raise ValueError("frozen validation ensemble artifact differs")
    payload = torch.load(source_path, map_location="cpu", weights_only=True)
    valid_indices = list(experiment["validation_indices"])
    validation = _validation_coarse_diagnostic(payload, valid_dataset, valid_indices)
    identity = _clean_code_identity(Path(__file__).resolve().parents[1])
    protocol = {
        "split": "train_for_reference_law_and_valid_for_frozen_diagnostic",
        "train_case_count": 576,
        "train_independent_block_count": 24,
        "slices_per_train_date_block": 24,
        "train_case_ids_sha256": _canonical_sha256(train_ids),
        "validation_case_count": 12,
        "noise_seed": int(experiment["noise_seed"]),
        "actual_base_measure": "masked iid N(0,1) in absolute normalized coarse state",
        "conditional_variance_claimed": False,
        "seasonal_coverage_caveat": "four fixed seasonal dates/year are not full annual coverage",
        "optimizer_steps": 0,
        "gpu_required": False,
        "code_identity": identity,
    }
    tracker = ClearMLTracker(
        experiment["project_name"], experiment["task_name"],
        tags=experiment["clearml"]["tags"], env_path=experiment["clearml"]["env_path"],
    )
    tracker.connect("audit_protocol", protocol)
    result = {
        "status": "complete",
        "publication_claim_permitted": False,
        "protocol": protocol,
        "train_reference": train_metrics,
        "frozen_validation_coarse": validation,
        "clearml_task_id": str(tracker.task.id),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(output, result)
    tracker.upload_artifact("coarse_reference_audit", output)
    tracker.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve(), args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
