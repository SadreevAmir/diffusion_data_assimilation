"""Frozen spatial calibration audit for coarse-to-fine dynamics ensembles.

The audit replays already-trained checkpoints on the fixed validation panel.  It
does not fit, tune, or post-process a forecast.  Statistics are retained per
date so uncertainty is blocked by date rather than spuriously by pixel.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .clearml_tracking import ClearMLTracker
from .config import load_json
from .data import build_dataset
from .direct_dynamics_cascade import masked_block_average, project_detail, smooth_right_inverse
from .direct_dynamics_cascade_e2e_evaluation import _sha256, _verify_inventory
from .direct_dynamics_cascade_end_to_end import load_cascade_predictor
from .direct_dynamics_cascade_fine_training import _clean_code_identity, _fine_collate
from .direct_dynamics_training import validate_direct_dataset
from .trainer import _atomic_json


CHANNELS = ("d3_sic", "d3_sit", "d6_sic", "d6_sit", "d9_sic", "d9_sit")
COMPONENTS = ("coarse_lift", "fine_residual")
REGIONS = ("ocean", "interior", "coast")


def _case_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked case/channel mean for a [B,C,H,W] tensor."""
    if value.ndim != 4 or mask.shape != (value.shape[0], 1, *value.shape[-2:]):
        raise ValueError("case mean expects [B,C,H,W] values and [B,1,H,W] mask")
    value = value.double()
    weight = mask.double().expand_as(value)
    denominator = weight.sum(dim=(-2, -1))
    if torch.any(denominator <= 0):
        raise ValueError("every case/region must contain at least one pixel")
    return (torch.where(weight > 0, value, 0.0) * weight).sum(dim=(-2, -1)) / denominator


def _region_masks(valid: torch.Tensor) -> dict[str, torch.Tensor]:
    binary = valid[:, :1] > 0
    neighbours = F.conv2d(binary.float(), torch.ones((1, 1, 3, 3)), padding=1)
    interior = binary & (neighbours == 9)
    coast = binary & ~interior
    return {"ocean": binary.float(), "interior": interior.float(), "coast": coast.float()}


def _radial_selectors(size: int) -> tuple[torch.Tensor, ...]:
    fy = torch.fft.fftfreq(size, dtype=torch.float64)[:, None]
    fx = torch.fft.rfftfreq(size, dtype=torch.float64)[None, :]
    radius = torch.sqrt(fy.square() + fx.square())
    return (
        (radius > 0.0) & (radius <= 0.125),
        (radius > 0.125) & (radius <= 0.25),
        radius > 0.25,
    )


def _case_ocean_patch_power(
    value: torch.Tensor, valid: torch.Tensor, *, patch_size: int = 32
) -> torch.Tensor:
    """Per-case/channel low/mid/high power, averaging all leading samples."""
    if value.ndim != 5 or valid.shape != (value.shape[0], 1, *value.shape[-2:]):
        raise ValueError("patch power expects [B,K,C,H,W] and [B,1,H,W]")
    if value.shape[-2] % patch_size or value.shape[-1] % patch_size:
        raise ValueError("grid must be divisible by patch size")
    window_1d = torch.hann_window(patch_size, periodic=False, dtype=torch.float64)
    window = window_1d[:, None] * window_1d[None, :]
    selectors = _radial_selectors(patch_size)
    result = torch.full((value.shape[0], value.shape[2], 3), float("nan"), dtype=torch.float64)
    for case in range(value.shape[0]):
        patches = value[case].double().unfold(-2, patch_size, patch_size).unfold(
            -2, patch_size, patch_size
        )
        mask_patches = valid[case].bool().unfold(-2, patch_size, patch_size).unfold(
            -2, patch_size, patch_size
        )
        full = mask_patches.all(dim=(-1, -2)).squeeze(0)
        packed = patches.permute(2, 3, 0, 1, 4, 5)[full]
        if packed.numel() == 0:
            continue
        packed = (packed - packed.mean(dim=(-1, -2), keepdim=True)) * window
        power = torch.fft.rfft2(packed, norm="ortho").abs().square()
        for band, selector in enumerate(selectors):
            result[case, :, band] = power[..., selector].mean(dim=-1).mean(dim=(0, 1))
    return result


def _case_ocean_patch_cross_power(
    left: torch.Tensor, right: torch.Tensor, valid: torch.Tensor, *, patch_size: int = 32
) -> torch.Tensor:
    """Signed real cross-spectrum of two ensemble-anomaly components."""
    if left.shape != right.shape:
        raise ValueError("cross power requires matching tensors")
    if left.ndim != 5:
        raise ValueError("cross power expects [B,M,C,H,W]")
    window_1d = torch.hann_window(patch_size, periodic=False, dtype=torch.float64)
    window = window_1d[:, None] * window_1d[None, :]
    selectors = _radial_selectors(patch_size)
    result = torch.full((left.shape[0], left.shape[2], 3), float("nan"), dtype=torch.float64)
    for case in range(left.shape[0]):
        mask_patches = valid[case].bool().unfold(-2, patch_size, patch_size).unfold(
            -2, patch_size, patch_size
        )
        full = mask_patches.all(dim=(-1, -2)).squeeze(0)
        packed = []
        for value in (left[case], right[case]):
            patches = value.double().unfold(-2, patch_size, patch_size).unfold(
                -2, patch_size, patch_size
            )
            chosen = patches.permute(2, 3, 0, 1, 4, 5)[full]
            chosen = (chosen - chosen.mean(dim=(-1, -2), keepdim=True)) * window
            packed.append(torch.fft.rfft2(chosen, norm="ortho"))
        if packed[0].numel() == 0:
            continue
        cross = (packed[0] * packed[1].conj()).real
        for band, selector in enumerate(selectors):
            result[case, :, band] = cross[..., selector].mean(dim=-1).mean(dim=(0, 1))
    return result


def component_statistics(
    members: torch.Tensor, truth: torch.Tensor, valid: torch.Tensor
) -> dict[str, Any]:
    """Return date-blocked variance, mean-error, and ocean spectra."""
    if members.ndim != 5 or truth.shape != members[:, 0].shape:
        raise ValueError("component statistics require [B,M,C,H,W] and [B,C,H,W]")
    anomalies = members - members.mean(dim=1, keepdim=True)
    variance = anomalies.square().sum(dim=1) / (members.shape[1] - 1)
    mean_error = members.mean(dim=1) - truth
    result: dict[str, Any] = {"regions": {}}
    for name, mask in _region_masks(valid).items():
        result["regions"][name] = {
            "case_channel_ensemble_variance": _case_mean(variance, mask).tolist(),
            "case_channel_squared_mean_error": _case_mean(mean_error.square(), mask).tolist(),
        }
    result["fully_ocean_patch_power_low_mid_high"] = {
        "ensemble_anomaly": _case_ocean_patch_power(anomalies, valid).tolist(),
        "ensemble_mean_error": _case_ocean_patch_power(mean_error[:, None], valid).tolist(),
        "truth_component": _case_ocean_patch_power(truth[:, None], valid).tolist(),
    }
    return result


def cross_statistics(
    coarse_members: torch.Tensor, residual_members: torch.Tensor, valid: torch.Tensor
) -> dict[str, Any]:
    coarse_anomaly = coarse_members - coarse_members.mean(dim=1, keepdim=True)
    residual_anomaly = residual_members - residual_members.mean(dim=1, keepdim=True)
    covariance = (coarse_anomaly * residual_anomaly).sum(dim=1) / (coarse_members.shape[1] - 1)
    regions = {
        name: {"case_channel_cross_covariance": _case_mean(covariance, mask).tolist()}
        for name, mask in _region_masks(valid).items()
    }
    return {
        "regions": regions,
        "fully_ocean_patch_cross_power_low_mid_high": _case_ocean_patch_cross_power(
            coarse_anomaly, residual_anomaly, valid
        ).tolist(),
    }


def _candidate_spec_valid(candidate: dict[str, Any]) -> None:
    required = ("label", "fine_run_dir", "fine_code_commit", "fine_sha256", "checkpoint", "checkpoint_sha256")
    if any(key not in candidate for key in required):
        raise ValueError("candidate inventory is incomplete")
    if len(candidate["checkpoint_sha256"]) != 64:
        raise ValueError("candidate checkpoint requires SHA256")


@torch.no_grad()
def run(config_path: Path, output: Path) -> dict[str, Any]:
    if torch.cuda.device_count() != 1:
        raise RuntimeError("spatial calibration audit requires exactly one visible GPU")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to reuse output {output}")
    experiment = load_json(config_path)
    if experiment.get("case_indices") is None or len(experiment["case_indices"]) != 12:
        raise ValueError("audit requires twelve frozen validation cases")
    if int(experiment.get("members", 0)) != 8:
        raise ValueError("audit requires eight members")
    candidates = experiment.get("candidates", [])
    if [row.get("label") for row in candidates] != ["baseline2048", "matched2048", "champion4096"]:
        raise ValueError("audit requires the frozen ordered candidate panel")
    for candidate in candidates:
        _candidate_spec_valid(candidate)

    coarse = experiment["coarse"]
    coarse_root = Path(coarse["run_dir"])
    _verify_inventory(coarse_root, coarse["sha256"], "coarse")
    first_metadata = load_json(Path(candidates[0]["fine_run_dir"]) / "metadata.json")
    dataset = build_dataset(first_metadata["data_config"], split="valid")
    validate_direct_dataset(dataset)
    batch = _fine_collate([dataset[index] for index in experiment["case_indices"]])
    case_ids = tuple(batch["meta"]["case_id"])
    if list(case_ids) != experiment["case_ids"]:
        raise ValueError("frozen case identities differ from archive")
    valid = batch["valid_mask"][:, :1].float()
    truth = batch["truth"].float()
    truth_coarse, _ = masked_block_average(truth, valid)
    truth_lift = smooth_right_inverse(truth_coarse, valid)
    truth_residual = project_detail(truth, valid)

    output.mkdir(parents=True, exist_ok=False)
    code_identity = _clean_code_identity(Path(__file__).resolve().parents[1])
    tracker = ClearMLTracker(
        experiment["project_name"], experiment["task_name"],
        tags=experiment["clearml"]["tags"], env_path=experiment["clearml"]["env_path"],
    )
    protocol = {
        "schema_version": "cascade_spatial_calibration_audit_v1",
        "code_identity": code_identity,
        "case_ids": list(case_ids),
        "members": 8,
        "optimizer_steps": 0,
        "selection_or_tuning": False,
        "statistics_blocked_by": "date/case",
        "spectral_bands_cycles_per_pixel": [[0.0, 0.125], [0.125, 0.25], [0.25, "inf"]],
        "spectral_support": "fully_ocean_nonoverlapping_32x32_hann_patches",
    }
    tracker.connect("audit_protocol", protocol)
    result: dict[str, Any] = {"status": "complete", "protocol": protocol, "candidates": {}}
    device = torch.device("cuda:0")
    condition = batch["structured_conditioning"].float().to(device)
    valid_device = valid.to(device)
    reference_coarse = None
    for candidate in candidates:
        root = Path(candidate["fine_run_dir"])
        _verify_inventory(root, candidate["fine_sha256"], candidate["label"])
        checkpoint = root / candidate["checkpoint"]
        if not checkpoint.is_file() or _sha256(checkpoint) != candidate["checkpoint_sha256"]:
            raise ValueError(f"checkpoint mismatch for {candidate['label']}")
        metadata = load_json(root / "metadata.json")
        if metadata["data_config"] != first_metadata["data_config"]:
            raise ValueError("candidate data laws differ")
        predictor = load_cascade_predictor(
            coarse_run_dir=str(coarse_root), coarse_checkpoint_name=coarse["checkpoint"],
            coarse_model_config=load_json(coarse_root / "config.json"),
            coarse_checkpoint_sha256=coarse["sha256"][coarse["checkpoint"]],
            fine_run_dir=str(root), fine_checkpoint_name=candidate["checkpoint"],
            fine_model_config=load_json(root / "config.json"),
            fine_checkpoint_sha256=candidate["checkpoint_sha256"],
            expected_coarse_code_commit=coarse["code_commit"],
            expected_fine_code_commit=candidate["fine_code_commit"],
            replay_code_commit=code_identity["git_commit"],
            expected_forecast_contract_sha256=experiment["forecast_contract_sha256"], device=device,
        )
        ensemble = predictor.sample_ensemble(
            member_indices=tuple(range(8)), structured_conditioning=condition, valid_mask=valid_device,
            case_ids=case_ids, coarse_num_timesteps=17, fine_num_timesteps=17, device=device,
            method="rk4", rtol=1e-5, atol=1e-6, end_time=0.0,
        )
        coarse_members = ensemble["coarse"].cpu()
        if reference_coarse is None:
            reference_coarse = coarse_members
        elif not torch.equal(reference_coarse, coarse_members):
            raise RuntimeError("candidates did not reuse identical coarse members")
        mask_members = valid[:, None].expand(-1, 8, -1, -1, -1).flatten(0, 1)
        lifted = smooth_right_inverse(coarse_members.flatten(0, 1), mask_members).unflatten(0, (12, 8))
        residual = ensemble["residual"].cpu()
        reconstructed = lifted + residual
        if not torch.allclose(reconstructed, ensemble["forecast"].cpu(), atol=3e-6, rtol=0):
            raise RuntimeError("component reconstruction failed")
        row = {
            "checkpoint_sha256": candidate["checkpoint_sha256"],
            "components": {
                "coarse_lift": component_statistics(lifted, truth_lift, valid),
                "fine_residual": component_statistics(residual, truth_residual, valid),
            },
            "coarse_fine_cross": cross_statistics(lifted, residual, valid),
        }
        result["candidates"][candidate["label"]] = row
        del predictor, ensemble, coarse_members, lifted, residual, reconstructed
        torch.cuda.empty_cache()
    _atomic_json(output / "spatial_calibration_audit.json", result)
    tracker.upload_artifact("spatial_calibration_audit", output / "spatial_calibration_audit.json")
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
