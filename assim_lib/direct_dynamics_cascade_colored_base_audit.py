"""Train-only selection of a full-support colored Gaussian fine-flow base."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Subset

from .clearml_tracking import ClearMLTracker
from .config import load_json, merge_config_overrides, resolve_path
from .data import build_dataset
from .direct_dynamics_cascade import masked_block_average, project_detail
from .direct_dynamics_cascade_fine_colored import (
    COLORED_BASE_KIND,
    colored_projected_gaussian,
)
from .direct_dynamics_cascade_fine_training import (
    EXPECTED_SPLIT_LENGTHS,
    _calendar_inventory,
    _canonical_sha256,
    _clean_code_identity,
    _fine_collate,
    _sha256_file,
)
from .direct_dynamics_cascade_residual_law_audit import (
    CHANNELS,
    TRAIN_YEARS,
    _case_equal_mean,
    _directional_increment_square,
    _fully_ocean_patch_spectrum,
    frozen_seasonal_indices,
)
from .direct_dynamics_training import validate_direct_dataset
from .runtime import build_dataloader
from .trainer import _atomic_json


CANDIDATE_BLENDS = (0.0, 0.25, 0.5, 0.625, 0.75, 0.875)


class _MeasureAccumulator:
    def __init__(self) -> None:
        self.case_count = 0
        self.second = torch.zeros(6, dtype=torch.float64)
        self.dx_second = torch.zeros(6, dtype=torch.float64)
        self.dy_second = torch.zeros(6, dtype=torch.float64)
        self.spectrum_sum = torch.zeros((6, 3), dtype=torch.float64)
        self.spectrum_count = torch.zeros(3, dtype=torch.int64)
        self.max_abs_coarse = 0.0

    def update(self, value: torch.Tensor, valid: torch.Tensor) -> None:
        self.second += _case_equal_mean(value.square(), valid).sum(dim=0)
        dx, dy = _directional_increment_square(value, valid)
        self.dx_second += dx.sum(dim=0)
        self.dy_second += dy.sum(dim=0)
        spectrum_sum, spectrum_count = _fully_ocean_patch_spectrum(value, valid)
        self.spectrum_sum += spectrum_sum
        self.spectrum_count += spectrum_count
        coarse, fraction = masked_block_average(value.float(), valid.float())
        active = fraction > 0
        self.max_abs_coarse = max(
            self.max_abs_coarse,
            float(coarse[active.expand_as(coarse)].abs().max()),
        )
        self.case_count += int(value.shape[0])

    def merge(self, other: "_MeasureAccumulator") -> None:
        self.case_count += other.case_count
        self.second += other.second
        self.dx_second += other.dx_second
        self.dy_second += other.dy_second
        self.spectrum_sum += other.spectrum_sum
        self.spectrum_count += other.spectrum_count
        self.max_abs_coarse = max(self.max_abs_coarse, other.max_abs_coarse)

    def finalize(self) -> dict[str, Any]:
        if self.case_count <= 0 or torch.any(self.spectrum_count <= 0):
            raise RuntimeError("colored-base audit accumulator is empty")
        return {
            "case_count": self.case_count,
            "rms": torch.sqrt(self.second / self.case_count).tolist(),
            "dx_rms": torch.sqrt(self.dx_second / self.case_count).tolist(),
            "dy_rms": torch.sqrt(self.dy_second / self.case_count).tolist(),
            "fully_ocean_patch_power_low_mid_high": (
                self.spectrum_sum / self.spectrum_count.to(torch.float64)[None]
            ).tolist(),
            "fully_ocean_patch_counts_low_mid_high": self.spectrum_count.tolist(),
            "max_abs_block_average": self.max_abs_coarse,
        }


def _merged(values: dict[int, _MeasureAccumulator], excluded: int | None = None) -> dict[str, Any]:
    result = _MeasureAccumulator()
    for year, value in values.items():
        if year != excluded:
            result.merge(value)
    return result.finalize()


def _shape_score(target: dict[str, Any], base: dict[str, Any]) -> float:
    target_rms = torch.tensor(target["rms"], dtype=torch.float64)
    base_rms = torch.tensor(base["rms"], dtype=torch.float64)
    target_power = torch.tensor(
        target["fully_ocean_patch_power_low_mid_high"], dtype=torch.float64
    )
    base_power = torch.tensor(
        base["fully_ocean_patch_power_low_mid_high"], dtype=torch.float64
    )
    target_features = torch.cat(
        (
            torch.tensor(target["dx_rms"], dtype=torch.float64)[:, None] / target_rms[:, None],
            torch.tensor(target["dy_rms"], dtype=torch.float64)[:, None] / target_rms[:, None],
            target_power / target_power.sum(dim=1, keepdim=True),
        ),
        dim=1,
    )
    base_features = torch.cat(
        (
            torch.tensor(base["dx_rms"], dtype=torch.float64)[:, None] / base_rms[:, None],
            torch.tensor(base["dy_rms"], dtype=torch.float64)[:, None] / base_rms[:, None],
            base_power / base_power.sum(dim=1, keepdim=True),
        ),
        dim=1,
    )
    if torch.any(target_features <= 0) or torch.any(base_features <= 0):
        raise FloatingPointError("colored-base shape features must be strictly positive")
    return float(torch.log(base_features / target_features).square().mean())


def _scale_measure(value: dict[str, Any], scales: torch.Tensor) -> dict[str, Any]:
    power = torch.tensor(
        value["fully_ocean_patch_power_low_mid_high"], dtype=torch.float64
    )
    return {
        **value,
        "rms": (torch.tensor(value["rms"], dtype=torch.float64) * scales).tolist(),
        "dx_rms": (torch.tensor(value["dx_rms"], dtype=torch.float64) * scales).tolist(),
        "dy_rms": (torch.tensor(value["dy_rms"], dtype=torch.float64) * scales).tolist(),
        "fully_ocean_patch_power_low_mid_high": (power * scales[:, None].square()).tolist(),
        "max_abs_block_average": float(value["max_abs_block_average"] * scales.max()),
    }


def run_audit(config_path: Path, output_path: Path) -> dict[str, Any]:
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"refusing to replace colored-base audit: {output_path}")
    experiment = load_json(config_path)
    config_dir = config_path.resolve().parent
    data_path = resolve_path(experiment["data_config"], config_dir)
    data_config = merge_config_overrides(load_json(data_path), experiment.get("data_overrides"))
    code_identity = _clean_code_identity(Path(__file__).resolve().parents[1])
    dataset = build_dataset(data_config, split="train")
    if len(dataset) != EXPECTED_SPLIT_LENGTHS["train"]:
        raise ValueError("train archive length differs from audited inventory")
    indices, case_ids = frozen_seasonal_indices(dataset)
    loader = build_dataloader(
        Subset(dataset, indices),
        batch_size=int(experiment.get("batch_size", 4)),
        num_workers=int(experiment.get("num_workers", 4)),
        shuffle=False,
        collate_fn=_fine_collate,
        prefetch_factor=2,
    )
    target_by_year = {year: _MeasureAccumulator() for year in TRAIN_YEARS}
    base_by_blend_year = {
        blend: {year: _MeasureAccumulator() for year in TRAIN_YEARS}
        for blend in CANDIDATE_BLENDS
    }
    generator = torch.Generator(device="cpu").manual_seed(
        int(experiment.get("noise_seed", 97331))
    )
    offset = 0
    for raw in loader:
        truth = raw["truth"].float()
        valid = raw["valid_mask"][:, :1].float()
        years = [int(case_ids[offset + index][:4]) for index in range(truth.shape[0])]
        offset += truth.shape[0]
        if len(set(years)) != 1:
            raise RuntimeError("audit batch crossed a frozen train-year boundary")
        year = years[0]
        target = project_detail(truth, valid)
        target_by_year[year].update(target, valid)
        white = torch.randn(truth.shape, generator=generator)
        for blend in CANDIDATE_BLENDS:
            base = colored_projected_gaussian(
                white,
                valid,
                blend=blend,
                channel_scales=(1.0,) * 6,
            )
            base_by_blend_year[blend][year].update(base, valid)
    if offset != len(indices):
        raise RuntimeError("colored-base audit did not consume the frozen panel")

    target_all = _merged(target_by_year)
    candidates = []
    for blend in CANDIDATE_BLENDS:
        raw_base = _merged(base_by_blend_year[blend])
        scales = torch.tensor(target_all["rms"], dtype=torch.float64) / torch.tensor(
            raw_base["rms"], dtype=torch.float64
        )
        candidates.append(
            {
                "kind": COLORED_BASE_KIND,
                "blend": blend,
                "minimum_unprojected_spectral_gain": 1.0 - blend,
                "shape_score": _shape_score(target_all, raw_base),
                "channel_scales": scales.tolist(),
                "raw_base": raw_base,
                "scaled_base": _scale_measure(raw_base, scales),
            }
        )
    selected = min(candidates, key=lambda item: (item["shape_score"], item["blend"]))
    loyo = []
    for held_out in TRAIN_YEARS:
        target_fold = _merged(target_by_year, excluded=held_out)
        scores = {
            str(blend): _shape_score(
                target_fold, _merged(base_by_blend_year[blend], excluded=held_out)
            )
            for blend in CANDIDATE_BLENDS
        }
        choice = min(CANDIDATE_BLENDS, key=lambda blend: (scores[str(blend)], blend))
        loyo.append({"held_out_train_year": held_out, "selected_blend": choice, "scores": scores})
    stable = all(row["selected_blend"] == selected["blend"] for row in loyo)

    clearml = experiment.get("clearml", {})
    tracker = ClearMLTracker(
        experiment["project_name"],
        experiment["task_name"],
        tags=clearml.get("tags", []),
        env_path=clearml.get("env_path"),
    )
    result = {
        "status": "complete" if stable else "hold_unstable_train_year_selection",
        "schema_version": 1,
        "split": "train_only",
        "target_coordinate": "normalized exact detail R=(I-UD)Y",
        "base_coordinate": "S P ((1-alpha)I+alpha G)xi",
        "kernel_1d": [0.25, 0.5, 0.25],
        "candidate_blends": list(CANDIDATE_BLENDS),
        "selection_score": "mean squared log-ratio over dx/rms, dy/rms and normalized low/mid/high patch power",
        "channel_order": list(CHANNELS),
        "case_count": len(indices),
        "selected_case_ids_sha256": _canonical_sha256(case_ids),
        "selected_indices_sha256": hashlib.sha256(
            ",".join(map(str, indices)).encode()
        ).hexdigest(),
        "noise_seed": int(experiment.get("noise_seed", 97331)),
        "leave_one_train_year_out": loyo,
        "selection_stable_across_train_years": stable,
        "candidates": candidates,
        "selected": selected,
        "measures": {"target": target_all, "base": selected["scaled_base"]},
        "dataset_validation": validate_direct_dataset(dataset),
        "calendar_inventory": _calendar_inventory(dataset, split="train"),
        "data_config_sha256": _canonical_sha256(data_config),
        "data_config_file_sha256": _sha256_file(data_path),
        "experiment_sha256": _sha256_file(config_path),
        "builder_sha256": _sha256_file(Path(__file__)),
        "code_identity": code_identity,
        "optimizer_steps": 0,
        "gpu_required": False,
        "clearml_task_id": str(tracker.task.id),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_path, result)
    tracker.report_single_value("selected_blend", float(selected["blend"]))
    tracker.report_single_value("selected_shape_score", float(selected["shape_score"]))
    tracker.report_single_value("selection_stable_across_train_years", float(stable))
    tracker.upload_artifact("colored_base_train_only_audit", output_path)
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
