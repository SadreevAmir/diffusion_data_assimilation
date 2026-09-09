"""Spatial diagnostics for saved censored-joint ensemble snapshots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch

from .direct_dynamics_training import DIRECT_LEADS
from .trainer import _atomic_json


OUTPUT_NAMES = tuple(
    name for lead in DIRECT_LEADS for name in (f"d{lead}_sic", f"d{lead}_sit")
)
SPATIAL_LAGS = (1, 2, 4, 8)


def _valid_pair_values(
    field: torch.Tensor, valid: torch.Tensor, lag: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return horizontal and vertical absolute increments over valid-ocean pairs."""

    if lag <= 0 or lag >= min(field.shape[-2:]):
        raise ValueError("spatial lag is outside the field")
    horizontal_mask = valid[..., lag:] & valid[..., :-lag]
    vertical_mask = valid[..., lag:, :] & valid[..., :-lag, :]
    horizontal = (field[..., lag:] - field[..., :-lag]).abs()
    vertical = (field[..., lag:, :] - field[..., :-lag, :]).abs()
    return horizontal[horizontal_mask], vertical[vertical_mask]


def mean_absolute_increment(
    field: torch.Tensor, valid: torch.Tensor, lag: int
) -> float:
    horizontal, vertical = _valid_pair_values(field, valid, lag)
    if horizontal.numel() + vertical.numel() == 0:
        raise ValueError("spatial increment has no valid-ocean pairs")
    return float(torch.cat((horizontal, vertical)).mean())


def snapshot_metrics(payload: dict[str, torch.Tensor]) -> dict[str, Any]:
    ensemble = payload["physical_ensemble"].float()
    truth = payload["truth"].float()
    valid = payload["valid_mask"][:, 0].bool()
    if ensemble.ndim != 5 or ensemble.shape[2] != len(OUTPUT_NAMES):
        raise ValueError("physical ensemble must have shape [case,member,6,H,W]")
    if truth.shape != ensemble.shape[:1] + ensemble.shape[2:]:
        raise ValueError("truth shape differs from physical ensemble")
    if valid.shape != truth.shape[:1] + truth.shape[-2:]:
        raise ValueError("valid mask shape differs from saved samples")
    active = valid[:, None, None].expand_as(ensemble)
    if not torch.all(torch.isfinite(ensemble[active])):
        raise FloatingPointError("ensemble contains NaN/Inf on active domain")

    result: dict[str, Any] = {"outputs": {}}
    for channel, name in enumerate(OUTPUT_NAMES):
        members = ensemble[:, :, channel]
        target = truth[:, channel]
        member_valid = valid[:, None].expand_as(members)
        spread = members.std(dim=1, unbiased=True)
        atom: dict[str, float] = {
            "zero_frequency": float((members[member_valid] == 0).float().mean())
        }
        if name.endswith("_sic"):
            atom["one_frequency"] = float((members[member_valid] == 1).float().mean())
        increments: dict[str, Any] = {}
        ensemble_mean = members.mean(dim=1)
        for lag in SPATIAL_LAGS:
            member_valid_lag = valid[:, None].expand_as(members)
            increments[str(lag)] = {
                "individual_members": mean_absolute_increment(
                    members, member_valid_lag, lag
                ),
                "ensemble_mean": mean_absolute_increment(ensemble_mean, valid, lag),
                "truth": mean_absolute_increment(target, valid, lag),
            }
        result["outputs"][name] = {
            "ensemble_sd_mean": float(spread[valid].mean()),
            "ensemble_sd_p95": float(torch.quantile(spread[valid], 0.95)),
            "atoms": atom,
            "mean_absolute_increments": increments,
        }
    return result


def _plot_case_maps(payload: dict[str, torch.Tensor], case: int, path: Path) -> None:
    ensemble = payload["physical_ensemble"].float()[case]
    valid = payload["valid_mask"].float()[case, 0]
    figure, axes = plt.subplots(3, len(OUTPUT_NAMES), figsize=(24, 11))
    for channel, name in enumerate(OUTPUT_NAMES):
        members = ensemble[:, channel]
        maps = [
            members.std(dim=0, unbiased=True),
            (members == 0).float().mean(dim=0),
            (members == 1).float().mean(dim=0)
            if name.endswith("_sic")
            else torch.full_like(valid, torch.nan),
        ]
        labels = ("ensemble SD", "P(exact zero)", "P(exact one)")
        for row, (value, label) in enumerate(zip(maps, labels, strict=True)):
            image = torch.where(valid > 0, value, torch.nan).cpu().numpy()
            axes[row, channel].imshow(
                image,
                origin="upper",
                vmin=0,
                vmax=None if row == 0 else 1,
                cmap="magma" if row == 0 else "viridis",
            )
            axes[row, channel].set_title(f"{name}: {label}")
            axes[row, channel].axis("off")
    figure.suptitle(f"Censored-joint spatial diagnostics; anchor {case}")
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def analyze_run(run_dir: Path) -> dict[str, Any]:
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    output_dir = run_dir / "spatial_diagnostics"
    output_dir.mkdir(exist_ok=False)
    snapshots: dict[str, Any] = {}
    for stage in sorted((run_dir / "diagnostics").glob("update_*")):
        sample_path = stage / "samples.pt"
        if not sample_path.is_file():
            raise FileNotFoundError(sample_path)
        payload = torch.load(sample_path, map_location="cpu", weights_only=True)
        metrics = snapshot_metrics(payload)
        snapshots[stage.name] = metrics
        _atomic_json(output_dir / f"{stage.name}_metrics.json", metrics)
        for case in range(payload["truth"].shape[0]):
            _plot_case_maps(
                payload, case, output_dir / f"{stage.name}_anchor_{case:02d}_maps.png"
            )
    result = {
        "status": "complete",
        "lags_pixels": SPATIAL_LAGS,
        "snapshots": snapshots,
    }
    _atomic_json(output_dir / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    arguments = parser.parse_args()
    print(json.dumps(analyze_run(arguments.run_dir), indent=2))


if __name__ == "__main__":
    main()
