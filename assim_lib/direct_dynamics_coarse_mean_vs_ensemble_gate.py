"""Frozen CPU comparison of a deterministic coarse mean and EMA9711 ensemble mean."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from .clearml_tracking import ClearMLTracker
from .config import load_json
from .direct_dynamics_cascade import masked_block_average
from .direct_dynamics_cascade_fine_training import _clean_code_identity
from .direct_dynamics_cascade_mean_frozen_audit import _selector_cases, _support_cases
from .direct_dynamics_training import DIRECT_LEADS
from .trainer import _atomic_json
from .transforms import channel_denormalize


CHANNELS = tuple(f"d{lead}_{field}" for lead in DIRECT_LEADS for field in ("sic", "sit"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(path: Path, expected: str) -> None:
    if not path.is_file() or _sha256(path) != expected:
        raise ValueError(f"frozen artifact differs: {path}")


def _regions(active: torch.Tensor, fraction: torch.Tensor) -> dict[str, torch.Tensor]:
    binary = active[:, :1] > 0
    neighbours = F.conv2d(binary.float(), torch.ones((1, 1, 3, 3)), padding=1)
    coast = binary & ~(binary & (neighbours == 9))
    return {"ocean": fraction.double(), "coast": coast.double() * fraction.double()}


def _case_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    expanded = weight.expand_as(value).double()
    denominator = expanded.sum((-2, -1))
    if torch.any(denominator <= 0):
        raise ValueError("empty scoring region")
    return (value.double() * expanded).sum((-2, -1)) / denominator


def _paired_interval(delta: torch.Tensor, seed: int, draws: int = 20000) -> dict[str, Any]:
    if delta.shape[0] != 12:
        raise ValueError("paired date interval requires the frozen 12 dates")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(12, (draws, 12), generator=generator)
    sampled = delta[indices].mean(dim=1)
    return {
        "estimate": delta.mean(dim=0).tolist(),
        "paired_date_bootstrap_95_ci_low": torch.quantile(sampled, 0.025, dim=0).tolist(),
        "paired_date_bootstrap_95_ci_high": torch.quantile(sampled, 0.975, dim=0).tolist(),
        "draws": draws,
    }


def _radial_low_selector(size: int) -> torch.Tensor:
    fy = torch.fft.fftfreq(size, dtype=torch.float64)[:, None]
    fx = torch.fft.rfftfreq(size, dtype=torch.float64)[None]
    radius = torch.sqrt(fy.square() + fx.square())
    return (radius > 0.0) & (radius <= 0.125)


def _case_low_power(value: torch.Tensor, valid: torch.Tensor, patch_size: int = 32) -> torch.Tensor:
    """Low-band power after the exact linear patch/window/FFT operator."""
    if value.ndim != 5 or value.shape[-2] % patch_size or value.shape[-1] % patch_size:
        raise ValueError("low-band input must be [B,K,C,H,W] on divisible grid")
    window_1d = torch.hann_window(patch_size, periodic=False, dtype=torch.float64)
    window = window_1d[:, None] * window_1d[None]
    selector = _radial_low_selector(patch_size)
    result = torch.full((value.shape[0], value.shape[2]), float("nan"), dtype=torch.float64)
    for case in range(value.shape[0]):
        patches = value[case].double().unfold(-2, patch_size, patch_size).unfold(
            -2, patch_size, patch_size
        )
        mask_patches = valid[case].bool().unfold(-2, patch_size, patch_size).unfold(
            -2, patch_size, patch_size
        )
        full = mask_patches.all(dim=(-1, -2)).squeeze(0)
        chosen = patches.permute(2, 3, 0, 1, 4, 5)[full]
        if chosen.numel() == 0:
            continue
        transformed = torch.fft.rfft2(
            (chosen - chosen.mean((-1, -2), keepdim=True)) * window, norm="ortho"
        )
        result[case] = transformed.abs().square()[..., selector].mean(-1).mean((0, 1))
    if not torch.isfinite(result).all():
        raise ValueError("frozen cases lack a fully-ocean low-band patch")
    return result


def _comparison(
    deterministic: torch.Tensor,
    members: torch.Tensor,
    truth: torch.Tensor,
    persistence: torch.Tensor,
    active: torch.Tensor,
    fraction: torch.Tensor,
) -> dict[str, Any]:
    member_count = members.shape[1]
    ensemble_mean = members.mean(1)
    anomalies = members - ensemble_mean[:, None]
    result: dict[str, Any] = {"regions": {}}
    for region, weight in _regions(active, fraction).items():
        deterministic_mse = _case_mean((deterministic - truth).square(), weight)
        ensemble_mse = _case_mean((ensemble_mean - truth).square(), weight)
        member_mean_anomaly_power = _case_mean(anomalies.square().mean(1), weight)
        mc_correction = member_mean_anomaly_power / (member_count - 1)
        adjusted = ensemble_mse - mc_correction
        persistence_mse = _case_mean((persistence - truth).square(), weight)
        delta = deterministic_mse - adjusted
        result["regions"][region] = {
            "deterministic_case_mse": deterministic_mse.tolist(),
            "ensemble_mean_case_mse": ensemble_mse.tolist(),
            "ensemble_predictive_mean_adjusted_case_mse": adjusted.tolist(),
            "ensemble_mean_mc_correction_case": mc_correction.tolist(),
            "persistence_case_mse": persistence_mse.tolist(),
            "deterministic_minus_adjusted_ensemble": _paired_interval(delta, 71001),
            "joint_channel_mean_deterministic_minus_adjusted_ensemble": _paired_interval(
                delta.mean(dim=1), 71011
            ),
        }
    deterministic_low = _case_low_power((deterministic - truth)[:, None], active)
    ensemble_low = _case_low_power((ensemble_mean - truth)[:, None], active)
    anomaly_low_mean = _case_low_power(anomalies, active)
    adjusted_low = ensemble_low - anomaly_low_mean / (member_count - 1)
    persistence_low = _case_low_power((persistence - truth)[:, None], active)
    result["native_coarse_low_band"] = {
        "deterministic_case_power": deterministic_low.tolist(),
        "ensemble_mean_case_power": ensemble_low.tolist(),
        "ensemble_predictive_mean_adjusted_case_power": adjusted_low.tolist(),
        "ensemble_mean_mc_correction_case": (anomaly_low_mean / (member_count - 1)).tolist(),
        "persistence_case_power": persistence_low.tolist(),
        "deterministic_minus_adjusted_ensemble": _paired_interval(
            deterministic_low - adjusted_low, 71002
        ),
        "joint_channel_mean_deterministic_minus_adjusted_ensemble": _paired_interval(
            (deterministic_low - adjusted_low).mean(dim=1), 71012
        ),
    }
    return result


def _save_panels(
    output: Path, case_ids: list[str], truth: torch.Tensor, persistence: torch.Tensor,
    deterministic: torch.Tensor, ensemble_mean: torch.Tensor, active: torch.Tensor,
) -> list[str]:
    visual = output / "visual_qc"; visual.mkdir()
    paths = []
    for case in (0, 4, 7, 11):
        figure, axes = plt.subplots(3, 4, figsize=(15, 10))
        for row, channel in enumerate((0, 4, 5)):
            values = (truth[case, channel], persistence[case, channel],
                      deterministic[case, channel], ensemble_mean[case, channel])
            field = "SIC" if channel % 2 == 0 else "SIT"
            low = min(0.0, *(float(x.min()) for x in values))
            high = max(1.0 if field == "SIC" else 0.0, *(float(x.max()) for x in values))
            for column, (label, value) in enumerate(zip(
                ("truth", "persistence", "deterministic512", "EMA9711 mean"), values, strict=True
            )):
                shown = value.clone(); shown[active[case, 0] == 0] = torch.nan
                image = axes[row, column].imshow(shown, origin="upper", cmap="viridis", vmin=low, vmax=high)
                axes[row, column].set_title(label); axes[row, column].set_ylabel(CHANNELS[channel])
                axes[row, column].set_xticks([]); axes[row, column].set_yticks([])
                figure.colorbar(image, ax=axes[row, column], fraction=0.046)
        figure.suptitle(f"Frozen mean comparison; {case_ids[case]}")
        figure.tight_layout(rect=(0, 0, 1, .97))
        path = visual / f"case{case:02d}_states.png"; figure.savefig(path, dpi=150); plt.close(figure)
        paths.append(str(path))

        channel = 5
        errors = (persistence[case, channel] - truth[case, channel],
                  deterministic[case, channel] - truth[case, channel],
                  ensemble_mean[case, channel] - truth[case, channel])
        limit = max(float(x.abs().max()) for x in errors)
        figure, axes = plt.subplots(1, 3, figsize=(12, 4))
        for axis, label, error in zip(axes, ("persistence", "deterministic512", "EMA9711 mean"), errors, strict=True):
            shown = error.clone(); shown[active[case, 0] == 0] = torch.nan
            image = axis.imshow(shown, origin="upper", cmap="coolwarm", vmin=-limit, vmax=limit)
            axis.set_title(label); axis.set_xticks([]); axis.set_yticks([])
            figure.colorbar(image, ax=axis, fraction=.046)
        figure.suptitle(f"d9 SIT signed error; shared scale; {case_ids[case]}")
        figure.tight_layout(rect=(0, 0, 1, .93))
        path = visual / f"case{case:02d}_d9_sit_signed_error.png"; figure.savefig(path, dpi=150); plt.close(figure)
        paths.append(str(path))
    return paths


def run(config_path: Path, output: Path) -> dict[str, Any]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.device_count() != 0:
        raise RuntimeError("mean-vs-ensemble gate must be CPU-only")
    if os.environ.get("CLEARML_REQUIRE_ONLINE") != "1":
        raise RuntimeError("mean-vs-ensemble gate requires online ClearML")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to replace {output}")
    config = load_json(config_path)
    code_identity = _clean_code_identity(Path(__file__).resolve().parents[1])
    for spec in config["artifacts"].values():
        _require(Path(spec["path"]), spec["sha256"])
    mean = torch.load(config["artifacts"]["mean_predictions"]["path"], map_location="cpu", weights_only=True)
    inputs = torch.load(config["artifacts"]["mean_inputs"]["path"], map_location="cpu", weights_only=True)
    ensemble = torch.load(config["artifacts"]["coarse_ensemble"]["path"], map_location="cpu", weights_only=True)
    case_ids = list(ensemble["case_ids"])
    positions = [inputs["case_ids"].index(case_id) for case_id in case_ids]
    if [mean["case_ids"][position] for position in positions] != case_ids:
        raise ValueError("mean and ensemble case identities differ")
    deterministic = mean["prediction_physical"][positions].float()
    truth = inputs["truth_physical"][positions].float()
    persistence = inputs["persistence_physical"][positions].float()
    active = inputs["coarse_active"][positions].float()
    fraction = inputs["coarse_ocean_fraction"][positions].float()
    means, stds = inputs["normalization_means"], inputs["normalization_stds"]
    coarse_members = channel_denormalize(ensemble["coarse"].float(), means, stds)
    downsampled_valid, recomputed_fraction = masked_block_average(
        ensemble["valid"].float(), ensemble["valid"].float(), 2
    )
    if not torch.equal((downsampled_valid > 0).float(), active) or not torch.equal(
        recomputed_fraction[:, :1], fraction
    ):
        raise ValueError("ensemble and deterministic coarse support differ")
    comparison = _comparison(
        deterministic, coarse_members, truth, persistence, active, fraction
    )
    ensemble_mean = coarse_members.mean(1)
    support = {
        "deterministic": {
            key: _support_cases(deterministic[:, i:i+1], active, "sic" if i % 2 == 0 else "sit")
            for i, key in enumerate(CHANNELS)
        },
        "ensemble_mean": {
            key: _support_cases(ensemble_mean[:, i:i+1], active, "sic" if i % 2 == 0 else "sit")
            for i, key in enumerate(CHANNELS)
        },
        "d9_sit_truth_sic_below_0p01": {
            "deterministic": _selector_cases(deterministic[:, 5:6], truth[:, 5:6], truth[:, 4:5] < .01, active),
            "ensemble_mean": _selector_cases(ensemble_mean[:, 5:6], truth[:, 5:6], truth[:, 4:5] < .01, active),
        },
    }
    ocean_ci = comparison["regions"]["ocean"][
        "joint_channel_mean_deterministic_minus_adjusted_ensemble"
    ]
    low_ci = comparison["native_coarse_low_band"][
        "joint_channel_mean_deterministic_minus_adjusted_ensemble"
    ]
    coast_ci = comparison["regions"]["coast"]["deterministic_minus_adjusted_ensemble"]
    ocean_joint_upper = float(ocean_ci["paired_date_bootstrap_95_ci_high"])
    low_joint_upper = float(low_ci["paired_date_bootstrap_95_ci_high"])
    d9_sit_coast_upper = float(coast_ci["paired_date_bootstrap_95_ci_high"][5])
    mean_candidate_permitted = ocean_joint_upper < 0 and low_joint_upper < 0 and d9_sit_coast_upper <= 0
    output.mkdir(parents=True)
    figures = _save_panels(output, case_ids, truth, persistence, deterministic, ensemble_mean, active)
    tracker = ClearMLTracker(
        config["project_name"], config["task_name"],
        tags=config["clearml"]["tags"], env_path=config["clearml"]["env_path"],
    )
    result = {
        "status": "complete_pending_astra_and_visual_review",
        "code_identity": code_identity,
        "config_sha256": _sha256(config_path),
        "case_ids": case_ids,
        "comparison": comparison,
        "raw_support_and_haze": support,
        "decision": {
            "mean_innovation_candidate_permitted": mean_candidate_permitted,
            "ocean_joint_upper_ci_mean": ocean_joint_upper,
            "low_band_joint_upper_ci_mean": low_joint_upper,
            "d9_sit_coast_upper_ci": d9_sit_coast_upper,
            "training_permitted": False,
        },
        "figures": figures,
        "clearml_task_id": str(tracker.task.id),
    }
    _atomic_json(output / "mean_vs_ensemble_gate.json", result)
    tracker.connect("frozen_mean_vs_ensemble_protocol", config)
    tracker.upload_artifact("mean_vs_ensemble_gate", output / "mean_vs_ensemble_gate.json")
    for path in figures: tracker.report_image("mean_vs_ensemble", Path(path).stem, Path(path), 0)
    tracker.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(6); torch.set_num_interop_threads(1)
    print(json.dumps(run(args.config.resolve(), args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
