#!/usr/bin/env python3
"""CPU-only comparison of frozen historical cascade component tensors."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch

from assim_lib.direct_dynamics_cascade import masked_block_average, project_detail, smooth_right_inverse
from assim_lib.direct_dynamics_cascade_spatial_calibration_audit import (
    CHANNELS,
    component_statistics,
    cross_statistics,
)


EXPECTED = {
    "baseline2048": "57df51edfe918ef581b7b7decd0f21725aa780dbc733cd225404041d80069a25",
    "matched2048": "cbccfc3402a64d99a66e113de85d460f243317d12505c49736c777dead45c37b",
    "champion4096": "af55cd7d5cbda26ea57bef98c8ecebf6158f7a17b4698d4b3fb608e226cb28ff",
}
FINITE_ENSEMBLE_FACTOR = 1.0 + 1.0 / 8.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _adjusted_ssr(variance: torch.Tensor, mse: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(FINITE_ENSEMBLE_FACTOR * variance / mse.clamp_min(torch.finfo(torch.float64).tiny))


def _summarize_component(stats: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {"regions": {}, "fully_ocean_spectrum": {}}
    for region, values in stats["regions"].items():
        variance = torch.tensor(values["case_channel_ensemble_variance"], dtype=torch.float64)
        mse = torch.tensor(values["case_channel_squared_mean_error"], dtype=torch.float64)
        summary["regions"][region] = {
            "mean_ensemble_variance_by_channel": variance.mean(dim=0).tolist(),
            "mean_squared_mean_error_by_channel": mse.mean(dim=0).tolist(),
            "finite_ensemble_adjusted_ssr_by_channel": _adjusted_ssr(
                variance.mean(dim=0), mse.mean(dim=0)
            ).tolist(),
            "pooled_finite_ensemble_adjusted_ssr": float(_adjusted_ssr(variance.mean(), mse.mean())),
        }
    spectrum = stats["fully_ocean_patch_power_low_mid_high"]
    anomaly = torch.tensor(spectrum["ensemble_anomaly"], dtype=torch.float64)
    error = torch.tensor(spectrum["ensemble_mean_error"], dtype=torch.float64)
    truth = torch.tensor(spectrum["truth_component"], dtype=torch.float64)
    finite = torch.isfinite(anomaly) & torch.isfinite(error) & torch.isfinite(truth)
    if not finite.all():
        raise FloatingPointError("every frozen case must contain fully-ocean spectral patches")
    summary["fully_ocean_spectrum"] = {
        "mean_anomaly_power_by_channel_band": anomaly.mean(dim=0).tolist(),
        "mean_error_power_by_channel_band": error.mean(dim=0).tolist(),
        "mean_truth_power_by_channel_band": truth.mean(dim=0).tolist(),
        "finite_ensemble_adjusted_spectral_ssr_by_channel_band": _adjusted_ssr(
            anomaly.mean(dim=0), error.mean(dim=0)
        ).tolist(),
    }
    return summary


def _bootstrap_ssr_difference(
    left: dict[str, Any], right: dict[str, Any], region: str, *, draws: int = 10000
) -> dict[str, float]:
    left_var = torch.tensor(left["regions"][region]["case_channel_ensemble_variance"], dtype=torch.float64)
    left_mse = torch.tensor(left["regions"][region]["case_channel_squared_mean_error"], dtype=torch.float64)
    right_var = torch.tensor(right["regions"][region]["case_channel_ensemble_variance"], dtype=torch.float64)
    right_mse = torch.tensor(right["regions"][region]["case_channel_squared_mean_error"], dtype=torch.float64)
    generator = torch.Generator().manual_seed(71031)
    indices = torch.randint(left_var.shape[0], (draws, left_var.shape[0]), generator=generator)
    def sampled(var: torch.Tensor, mse: torch.Tensor) -> torch.Tensor:
        return _adjusted_ssr(var[indices].mean(dim=(1, 2)), mse[indices].mean(dim=(1, 2)))
    estimate = float(_adjusted_ssr(left_var.mean(), left_mse.mean()) - _adjusted_ssr(right_var.mean(), right_mse.mean()))
    delta = sampled(left_var, left_mse) - sampled(right_var, right_mse)
    return {
        "left_minus_right": estimate,
        "date_block_bootstrap_95_ci_low": float(torch.quantile(delta, 0.025)),
        "date_block_bootstrap_95_ci_high": float(torch.quantile(delta, 0.975)),
        "bootstrap_draws": draws,
    }


def _bootstrap_spectral_ssr_difference(
    left: dict[str, Any], right: dict[str, Any], band: int, *, draws: int = 10000
) -> dict[str, float]:
    def arrays(stats: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        spectrum = stats["fully_ocean_patch_power_low_mid_high"]
        anomaly = torch.tensor(spectrum["ensemble_anomaly"], dtype=torch.float64)[..., band]
        error = torch.tensor(spectrum["ensemble_mean_error"], dtype=torch.float64)[..., band]
        if not torch.isfinite(anomaly).all() or not torch.isfinite(error).all():
            raise FloatingPointError("spectral bootstrap received nonfinite cases")
        return anomaly, error
    left_var, left_mse = arrays(left)
    right_var, right_mse = arrays(right)
    generator = torch.Generator().manual_seed(71031 + band)
    indices = torch.randint(left_var.shape[0], (draws, left_var.shape[0]), generator=generator)
    def sampled(var: torch.Tensor, mse: torch.Tensor) -> torch.Tensor:
        return _adjusted_ssr(var[indices].mean(dim=(1, 2)), mse[indices].mean(dim=(1, 2)))
    estimate = float(_adjusted_ssr(left_var.mean(), left_mse.mean()) - _adjusted_ssr(right_var.mean(), right_mse.mean()))
    delta = sampled(left_var, left_mse) - sampled(right_var, right_mse)
    return {
        "left_minus_right": estimate,
        "date_block_bootstrap_95_ci_low": float(torch.quantile(delta, 0.025)),
        "date_block_bootstrap_95_ci_high": float(torch.quantile(delta, 0.975)),
        "bootstrap_draws": draws,
    }


def run(component_dir: Path, output: Path) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to replace {output}")
    torch.set_num_threads(6)
    torch.set_num_interop_threads(1)
    payloads = {}
    for label, expected in EXPECTED.items():
        path = component_dir / f"{label}.pt"
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"frozen component artifact mismatch: {label}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["candidate"] != label or payload["fine_checkpoint_sha256"] == "":
            raise ValueError(f"component identity mismatch: {label}")
        payloads[label] = payload
    reference = payloads["baseline2048"]
    for label, payload in payloads.items():
        for key in ("case_ids", "member_indices", "coarse_seeds", "solver"):
            if payload[key] != reference[key]:
                raise ValueError(f"paired replay identity differs for {label}: {key}")
        for key in ("coarse", "truth", "valid"):
            if not torch.equal(payload[key], reference[key]):
                raise ValueError(f"paired tensor differs for {label}: {key}")
    truth = reference["truth"].float()
    valid = reference["valid"].float()
    truth_coarse, _ = masked_block_average(truth, valid)
    truth_lift = smooth_right_inverse(truth_coarse, valid)
    truth_residual = project_detail(truth, valid)
    mask_members = valid[:, None].expand(-1, 8, -1, -1, -1).flatten(0, 1)

    result: dict[str, Any] = {
        "status": "complete",
        "schema_version": "cascade_spatial_calibration_audit_v1",
        "channels": CHANNELS,
        "case_ids": reference["case_ids"],
        "members": 8,
        "finite_ensemble_variance_factor": FINITE_ENSEMBLE_FACTOR,
        "uncertainty_unit": "date/case",
        "component_artifact_sha256": EXPECTED,
        "candidates": {},
    }
    raw_stats = {}
    for label, payload in payloads.items():
        coarse = payload["coarse"].float()
        lifted = smooth_right_inverse(coarse.flatten(0, 1), mask_members).unflatten(0, (12, 8))
        residual = payload["residual"].float()
        recovered, recovered_fraction = masked_block_average(
            residual.flatten(0, 1), mask_members
        )
        active = recovered_fraction > 0
        max_residual_block_mean = float(recovered[active.expand_as(recovered)].abs().max())
        if max_residual_block_mean > 3e-6:
            raise RuntimeError(f"fine residual violates coarse constraint: {label}")
        component = {
            "coarse_lift": component_statistics(lifted, truth_lift, valid),
            "fine_residual": component_statistics(residual, truth_residual, valid),
        }
        raw_stats[label] = component
        result["candidates"][label] = {
            "replay_commit": payload["replay_commit"],
            "fine_checkpoint_sha256": payload["fine_checkpoint_sha256"],
            "max_abs_residual_block_mean": max_residual_block_mean,
            "components": {name: _summarize_component(values) for name, values in component.items()},
            "coarse_fine_cross": cross_statistics(lifted, residual, valid),
            "date_blocked_raw": component,
        }
    result["paired_date_block_bootstrap"] = {}
    for left, right in (("matched2048", "baseline2048"), ("matched2048", "champion4096")):
        key = f"{left}_minus_{right}"
        result["paired_date_block_bootstrap"][key] = {
            component: {
                "spatial": {
                    region: _bootstrap_ssr_difference(
                        raw_stats[left][component], raw_stats[right][component], region
                    )
                    for region in ("ocean", "interior", "coast")
                },
                "fully_ocean_spectral": {
                    band: _bootstrap_spectral_ssr_difference(
                        raw_stats[left][component], raw_stats[right][component], index
                    )
                    for index, band in enumerate(("low", "mid", "high"))
                },
            }
            for component in ("coarse_lift", "fine_residual")
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(output, result)
    print(json.dumps({"status": "complete", "output": str(output), "sha256": _sha256(output)}, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--component-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.component_dir.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
