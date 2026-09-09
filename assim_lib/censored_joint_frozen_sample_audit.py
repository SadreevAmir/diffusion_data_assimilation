"""Frozen CPU audit of one-pass censored dynamics ensemble snapshots.

The audit is deliberately read-only with respect to scientific artifacts.  It
checks exact paired identities, independently replays the pointwise decoder,
and measures whether grid-periodic texture is already present before censoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch

from .direct_dynamics_training import DIRECT_LEADS


OUTPUT_NAMES = tuple(
    name for lead in DIRECT_LEADS for name in (f"d{lead}_sic", f"d{lead}_sit")
)
PHASE_PERIODS = (2, 4, 8)
FROZEN_MEANS = (0.19301218262209952, 0.18969311571248842)
FROZEN_STDS = (0.36890927421384667, 0.4359687842884708)
FROZEN_MANIFEST = {
    "baseline_sha256": "05fe7bef4136be45c99986ad3a70a516c64d733d4946de3d4934af9056c09014",
    "candidate_sha256": "9b6d93061d47cdd4ed7e99214fd4782c37dd768e712faa39ce1c5dd3e50a6886",
    "historical_sha256": "79760af54d54fd9e190d74b9ecda18c1d85ea0852522f3e8343a15bde4caf814",
    "baseline_checkpoint_sha256": "d766e1105140873a6ed73c12846e4da060349e4b8d6f0e9d9e19801b3cd7f262",
    "candidate_checkpoint_sha256": "166013d57dbed12ab6581095cfb82b2b15510ea38f6c9156fa0e66afd12c1d86",
    "noise_sha256": "8a8c2a202c5fd3021b63c7507fca29d8644aaa5f62239033466c23eef5d09159",
    "group": "validation",
    "update": 1024,
    "means": FROZEN_MEANS,
    "stds": FROZEN_STDS,
    "baseline_cases": 12,
    "historical_cases": 2,
    "members": 8,
    "image_size": (320, 256),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    temporary.replace(path)


def _expanded_stats(
    means: tuple[float, float], stds: tuple[float, float], value: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    repeated_means = torch.tensor(means * 3, dtype=value.dtype).view(
        *((1,) * (value.ndim - 3)), 6, 1, 1
    )
    repeated_stds = torch.tensor(stds * 3, dtype=value.dtype).view(
        *((1,) * (value.ndim - 3)), 6, 1, 1
    )
    return repeated_means, repeated_stds


def decode_latent(
    latent: torch.Tensor,
    valid: torch.Tensor,
    *,
    means: tuple[float, float],
    stds: tuple[float, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return uncensored and censored physical fields without using runner code."""

    if latent.ndim != 5 or latent.shape[2] != 6:
        raise ValueError("latent must have shape [case,member,6,height,width]")
    if valid.shape != latent.shape[:1] + (1,) + latent.shape[-2:]:
        raise ValueError("valid mask shape differs from latent")
    if len(means) != 2 or len(stds) != 2:
        raise ValueError("decoder requires two field means and standard deviations")
    if not all(torch.isfinite(torch.tensor((*means, *stds)))) or any(
        value <= 0 for value in stds
    ):
        raise ValueError("decoder statistics must be finite with positive stds")
    active = valid[:, None].expand_as(latent) > 0
    safe_latent = torch.where(active, latent.float(), torch.zeros_like(latent.float()))
    if not torch.all(torch.isfinite(safe_latent[active])):
        raise FloatingPointError("active latent contains NaN/Inf")
    mean, std = _expanded_stats(means, stds, safe_latent)
    uncensored = safe_latent * std + mean
    censored = uncensored.clone()
    censored[:, :, 0::2].clamp_(0.0, 1.0)
    censored[:, :, 1::2].clamp_(min=0.0)
    uncensored = torch.where(active, uncensored, torch.zeros_like(uncensored))
    censored = torch.where(active, censored, torch.zeros_like(censored))
    return uncensored, censored


def _validate_payload(
    payload: dict[str, Any],
    *,
    label: str,
    require_latent: bool,
    expected_cases: int,
    expected_members: int = 8,
    image_size: tuple[int, int] = (320, 256),
) -> None:
    required = ["physical_ensemble", "truth", "persistence", "valid_mask"]
    if require_latent:
        required.append("uncensored_latent_normalized")
    missing = [key for key in required if key not in payload]
    if missing:
        raise KeyError(f"{label} missing required keys: {missing}")
    ensemble = payload["physical_ensemble"]
    truth = payload["truth"]
    persistence = payload["persistence"]
    valid = payload["valid_mask"]
    expected_ensemble = (expected_cases, expected_members, 6, *image_size)
    if tuple(ensemble.shape) != expected_ensemble:
        raise ValueError(f"{label} ensemble shape differs: {tuple(ensemble.shape)}")
    if tuple(truth.shape) != (expected_cases, 6, *image_size):
        raise ValueError(f"{label} truth shape differs")
    if tuple(persistence.shape) != tuple(truth.shape):
        raise ValueError(f"{label} persistence shape differs")
    if tuple(valid.shape) != (expected_cases, 1, *image_size):
        raise ValueError(f"{label} valid-mask shape differs")
    if not torch.all((valid == 0) | (valid == 1)) or not torch.any(valid == 1):
        raise ValueError(f"{label} valid mask must be nonempty and exactly binary")
    active_ensemble = valid[:, None].expand_as(ensemble) > 0
    active_field = valid.expand_as(truth) > 0
    for name, value, mask in (
        ("ensemble", ensemble, active_ensemble),
        ("truth", truth, active_field),
        ("persistence", persistence, active_field),
    ):
        if not torch.all(torch.isfinite(value[mask])):
            raise FloatingPointError(f"{label} active {name} contains NaN/Inf")
    if require_latent:
        latent = payload["uncensored_latent_normalized"]
        if tuple(latent.shape) != expected_ensemble:
            raise ValueError(f"{label} latent shape differs")
        if not torch.all(torch.isfinite(latent[active_ensemble])):
            raise FloatingPointError(f"{label} active latent contains NaN/Inf")


def _validate_snapshot_identity(
    payload: dict[str, Any],
    *,
    label: str,
    checkpoint_sha256: str,
    manifest: dict[str, Any],
    expected_cases: int,
) -> None:
    metadata = payload.get("metadata")
    noise = payload.get("noise_sha256")
    checkpoint = payload.get("checkpoint")
    if not isinstance(metadata, list) or len(metadata) != expected_cases:
        raise ValueError(f"{label} metadata identity is missing or incomplete")
    if not isinstance(noise, str) or len(noise) != 64 or noise != manifest["noise_sha256"]:
        raise ValueError(f"{label} noise identity differs")
    if payload.get("group") != manifest["group"] or payload.get("update") != manifest["update"]:
        raise ValueError(f"{label} group/update identity differs")
    if not isinstance(checkpoint, dict) or checkpoint.get("sha256") != checkpoint_sha256:
        raise ValueError(f"{label} checkpoint identity differs")


def _edge_values(
    field: torch.Tensor, valid: torch.Tensor, dy: int, dx: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if dy < 0:
        field = torch.flip(field, dims=(-2,))
        valid = torch.flip(valid, dims=(-2,))
        dy = -dy
    if dx < 0:
        field = torch.flip(field, dims=(-1,))
        valid = torch.flip(valid, dims=(-1,))
        dx = -dx
    h = field.shape[-2] - dy
    w = field.shape[-1] - dx
    if h <= 0 or w <= 0:
        raise ValueError("shift is outside field")
    left = field[..., :h, :w]
    right = field[..., dy : dy + h, dx : dx + w]
    pair_mask = valid[..., :h, :w] & valid[..., dy : dy + h, dx : dx + w]
    return right - left, pair_mask


def _mean_abs_increment(field: torch.Tensor, valid: torch.Tensor) -> float:
    values = []
    for dy, dx in ((1, 0), (0, 1)):
        delta, mask = _edge_values(field, valid, dy, dx)
        values.append(delta.abs()[mask.expand_as(delta)])
    joined = torch.cat(values)
    if joined.numel() == 0:
        raise ValueError("no valid adjacent ocean pairs")
    result = float(joined.mean())
    if not torch.isfinite(torch.tensor(result)):
        raise FloatingPointError("spatial increment is NaN/Inf")
    return result


def _safe_ratio(numerator: float, denominator: float) -> dict[str, Any]:
    if denominator == 0.0:
        return {"value": None, "null_reason": "zero_denominator"}
    value = numerator / denominator
    if not torch.isfinite(torch.tensor(value)):
        raise FloatingPointError("spatial ratio is NaN/Inf")
    return {"value": value, "null_reason": None}


def decoder_invariants(
    payload: dict[str, Any],
    *,
    means: tuple[float, float],
    stds: tuple[float, float],
    tolerance: float = 2e-6,
) -> dict[str, Any]:
    latent = payload["uncensored_latent_normalized"].float()
    saved = payload["physical_ensemble"].float()
    valid = payload["valid_mask"].bool()
    uncensored, decoded = decode_latent(latent, valid, means=means, stds=stds)
    active = valid[:, None].expand_as(saved)
    reconstruction_error = float((decoded[active] - saved[active]).abs().max())

    persistence = payload["persistence"].float()
    mean, std = _expanded_stats(means, stds, persistence)
    normalized_persistence = (persistence - mean) / std
    _, zero_residual_decoded = decode_latent(
        normalized_persistence[:, None], valid, means=means, stds=stds
    )
    persistence_error = float(
        (zero_residual_decoded[:, 0][valid.expand_as(persistence)]
         - persistence[valid.expand_as(persistence)]).abs().max()
    )

    violation_count = 0
    maximum_excess = 0.0
    for dy, dx in ((1, 0), (0, 1), (1, 1), (1, -1)):
        before, mask = _edge_values(uncensored, active, dy, dx)
        after, _ = _edge_values(decoded, active, dy, dx)
        excess = after.abs() - before.abs()
        selected = excess[mask]
        violation_count += int((selected > tolerance).sum())
        maximum_excess = max(maximum_excess, float(selected.max()))
    return {
        "independent_decoder_max_abs_error": reconstruction_error,
        "zero_residual_persistence_max_abs_error": persistence_error,
        "one_lipschitz_violation_count": violation_count,
        "one_lipschitz_max_excess": maximum_excess,
        "tolerance": tolerance,
        "passed": reconstruction_error <= tolerance
        and persistence_error <= tolerance
        and violation_count == 0,
    }


def paired_identity(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    tensor_keys = ("truth", "persistence", "valid_mask")
    missing = [
        key
        for key in (*tensor_keys, "metadata", "noise_sha256")
        if key not in baseline
        or key not in candidate
        or baseline[key] is None
        or candidate[key] is None
    ]
    exact = {
        key: key not in missing and torch.equal(baseline[key], candidate[key])
        for key in tensor_keys
    }
    metadata_equal = (
        "metadata" not in missing
        and baseline["metadata"] == candidate["metadata"]
    )
    noise_equal = (
        "noise_sha256" not in missing
        and baseline["noise_sha256"] == candidate["noise_sha256"]
    )
    return {
        "tensor_exact": exact,
        "metadata_exact": metadata_equal,
        "noise_sha256_exact": noise_equal,
        "baseline_noise_sha256": baseline.get("noise_sha256"),
        "candidate_noise_sha256": candidate.get("noise_sha256"),
        "missing_required_keys": missing,
        "passed": not missing and all(exact.values()) and metadata_equal and noise_equal,
    }


def _phase_map(
    field: torch.Tensor, valid: torch.Tensor, period: int
) -> dict[str, Any]:
    """Aggregate member-anomaly adjacent increments by source-pixel grid phase."""

    height, width = field.shape[-2:]
    rows = torch.arange(height).view(height, 1).expand(height, width)
    cols = torch.arange(width).view(1, width).expand(height, width)
    values = torch.zeros((period, period), dtype=torch.float64)
    counts = torch.zeros((period, period), dtype=torch.int64)
    for dy, dx in ((1, 0), (0, 1)):
        delta, mask = _edge_values(field, valid, dy, dx)
        h, w = delta.shape[-2:]
        phase_row = rows[:h, :w] % period
        phase_col = cols[:h, :w] % period
        for row in range(period):
            for col in range(period):
                phase = (phase_row == row) & (phase_col == col)
                selected_mask = mask & phase
                selected = delta.abs()[selected_mask.expand_as(delta)]
                if selected.numel():
                    values[row, col] += selected.double().sum()
                    counts[row, col] += selected.numel()
    means = values / counts.clamp(min=1)
    populated = means[counts > 0]
    if populated.numel() == 0 or int(counts.sum()) == 0:
        raise ValueError("phase map has no valid pairs")
    overall = float(values.sum() / counts.sum())
    contrast = _safe_ratio(float(populated.max() - populated.min()), overall)
    return {
        "mean_absolute_increment": means.tolist(),
        "counts": counts.tolist(),
        "relative_phase_contrast": contrast,
    }


def _stage_summary(field: torch.Tensor, valid: torch.Tensor) -> dict[str, Any]:
    aggregate = _mean_abs_increment(field, valid)
    per_case = []
    for case in range(field.shape[0]):
        case_field = field[case : case + 1]
        case_valid = valid[case : case + 1]
        per_case.append(
            {
                "case": case,
                "lag1_increment": _mean_abs_increment(case_field, case_valid),
                "phase_contrast": {
                    str(period): _phase_map(case_field, case_valid, period)[
                        "relative_phase_contrast"
                    ]
                    for period in PHASE_PERIODS
                },
            }
        )
    return {
        "lag1_increment": aggregate,
        "phase": {
            str(period): _phase_map(field, valid, period)
            for period in PHASE_PERIODS
        },
        "per_case": per_case,
    }


def spatial_summary(
    payload: dict[str, Any],
    *,
    means: tuple[float, float] | None = None,
    stds: tuple[float, float] | None = None,
) -> dict[str, Any]:
    ensemble = payload["physical_ensemble"].float()
    truth = payload["truth"].float()
    persistence = payload["persistence"].float()
    valid = payload["valid_mask"].bool()
    member_mean = ensemble.mean(dim=1, keepdim=True)
    anomalies = ensemble - member_mean
    uncensored = None
    residual = None
    if means is not None or stds is not None:
        if means is None or stds is None:
            raise ValueError("means and stds must be supplied together")
        uncensored, _ = decode_latent(
            payload["uncensored_latent_normalized"].float(),
            valid,
            means=means,
            stds=stds,
        )
        residual = uncensored - persistence[:, None]
    result: dict[str, Any] = {}
    for channel, name in enumerate(OUTPUT_NAMES):
        field_valid = valid[:, None]
        truth_summary = _stage_summary(truth[:, channel : channel + 1], valid)
        persistence_summary = _stage_summary(
            persistence[:, channel : channel + 1], valid
        )
        member_summary = _stage_summary(
            ensemble[:, :, channel : channel + 1], field_valid
        )
        mean_summary = _stage_summary(
            member_mean[:, :, channel : channel + 1], field_valid
        )
        anomaly_summary = _stage_summary(
            anomalies[:, :, channel : channel + 1], field_valid
        )
        entry: dict[str, Any] = {
            "truth": truth_summary,
            "persistence": persistence_summary,
            "censored_individual": member_summary,
            "censored_ensemble_mean": mean_summary,
            "censored_member_anomaly": anomaly_summary,
            "censored_member_to_truth_ratio": _safe_ratio(
                member_summary["lag1_increment"], truth_summary["lag1_increment"]
            ),
        }
        if uncensored is not None and residual is not None:
            uncensored_field = uncensored[:, :, channel : channel + 1]
            residual_field = residual[:, :, channel : channel + 1]
            uncensored_mean = uncensored_field.mean(dim=1, keepdim=True)
            residual_mean = residual_field.mean(dim=1, keepdim=True)
            uncensored_anomaly = uncensored_field - uncensored_mean
            residual_anomaly = residual_field - residual_mean
            uncensored_summary = _stage_summary(uncensored_field, field_valid)
            residual_summary = _stage_summary(residual_field, field_valid)
            entry.update(
                {
                    "uncensored_individual": uncensored_summary,
                    "uncensored_ensemble_mean": _stage_summary(
                        uncensored_mean, field_valid
                    ),
                    "uncensored_member_anomaly": _stage_summary(
                        uncensored_anomaly, field_valid
                    ),
                    "uncensored_member_to_truth_ratio": _safe_ratio(
                        uncensored_summary["lag1_increment"],
                        truth_summary["lag1_increment"],
                    ),
                    "residual_individual": residual_summary,
                    "residual_ensemble_mean": _stage_summary(residual_mean, field_valid),
                    "residual_member_anomaly": _stage_summary(
                        residual_anomaly, field_valid
                    ),
                }
            )
        result[name] = entry
    return result


def _plot_pipeline(
    payload: dict[str, Any],
    output: Path,
    *,
    label: str,
    means: tuple[float, float],
    stds: tuple[float, float],
    physical_limits: dict[str, tuple[float, float]],
    residual_limits: dict[str, tuple[float, float]],
) -> None:
    case = 5
    member = 0
    latent = payload["uncensored_latent_normalized"].float()
    valid = payload["valid_mask"].bool()
    uncensored, censored = decode_latent(latent, valid, means=means, stds=stds)
    persistence = payload["persistence"].float()
    truth = payload["truth"].float()
    residual = uncensored - persistence[:, None]
    for offset, field in ((0, "sic"), (1, "sit")):
        figure, axes = plt.subplots(3, 5, figsize=(20, 11), constrained_layout=True)
        for row, lead in enumerate(DIRECT_LEADS):
            channel = 2 * row + offset
            panels = (
                (truth[case, channel], "truth"),
                (persistence[case, channel], "persistence d0"),
                (residual[case, member, channel], "predicted residual"),
                (uncensored[case, member, channel], "uncensored physical"),
                (censored[case, member, channel], "censored sample"),
            )
            for axis, (image, title) in zip(axes[row], panels, strict=True):
                masked = image.clone()
                masked[~valid[case, 0]] = torch.nan
                limits = residual_limits[field] if title == "predicted residual" else physical_limits[field]
                axis.imshow(
                    masked.numpy(),
                    origin="upper",
                    interpolation="nearest",
                    cmap="coolwarm" if title == "predicted residual" else None,
                    vmin=limits[0],
                    vmax=limits[1],
                )
                axis.set_title(
                    f"d+{lead} {title}\nrange [{limits[0]:.3g}, {limits[1]:.3g}]"
                )
                axis.set_xticks([])
                axis.set_yticks([])
        figure.suptitle(
            f"Frozen decoder audit: {label}, case {case}, member {member}, {field.upper()}"
        )
        figure.savefig(output / f"{label}_case05_member0_{field}_pipeline.png", dpi=170)
        plt.close(figure)


def _shared_plot_limits(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    means: tuple[float, float],
    stds: tuple[float, float],
) -> tuple[dict[str, tuple[float, float]], dict[str, tuple[float, float]]]:
    physical_limits: dict[str, tuple[float, float]] = {}
    residual_limits: dict[str, tuple[float, float]] = {}
    for offset, field in ((0, "sic"), (1, "sit")):
        physical_values = []
        residual_values = []
        for payload in (baseline, candidate):
            valid = payload["valid_mask"].bool()
            uncensored, _ = decode_latent(
                payload["uncensored_latent_normalized"].float(),
                valid,
                means=means,
                stds=stds,
            )
            persistence = payload["persistence"].float()
            truth = payload["truth"].float()
            channels = slice(offset, 6, 2)
            active_member = valid[:, None].expand_as(uncensored[:, :, channels])
            active_field = valid.expand_as(truth[:, channels])
            physical_values.extend(
                [
                    uncensored[:, :, channels][active_member],
                    truth[:, channels][active_field],
                    persistence[:, channels][active_field],
                ]
            )
            residual_values.append(
                (uncensored[:, :, channels] - persistence[:, None, channels])[active_member]
            )
        joined = torch.cat(physical_values)
        residual = torch.cat(residual_values)
        physical_limits[field] = (float(joined.min()), float(joined.max()))
        radius = float(residual.abs().max())
        residual_limits[field] = (-radius, radius)
    return physical_limits, residual_limits


def audit(
    baseline_path: Path,
    candidate_path: Path,
    historical_path: Path,
    output_dir: Path,
    *,
    expected_manifest: dict[str, Any] = FROZEN_MANIFEST,
    create_plots: bool = True,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse audit output: {output_dir}")
    output_dir.mkdir(parents=True)
    paths = {
        "baseline": baseline_path,
        "candidate": candidate_path,
        "historical": historical_path,
    }
    inputs = {
        label: {"path": str(path), "sha256": _sha256(path)}
        for label, path in paths.items()
    }
    source_identity = {
        label: inputs[label]["sha256"] == expected_manifest[f"{label}_sha256"]
        for label in paths
    }
    if not all(source_identity.values()):
        failure = {
            "status": "failed",
            "stage": "source_identity",
            "inputs": inputs,
            "source_identity": source_identity,
        }
        _atomic_json(output_dir / "failure.json", failure)
        return failure
    means = tuple(float(value) for value in expected_manifest["means"])
    stds = tuple(float(value) for value in expected_manifest["stds"])
    try:
        baseline = torch.load(baseline_path, map_location="cpu", weights_only=True)
        candidate = torch.load(candidate_path, map_location="cpu", weights_only=True)
        historical_raw = torch.load(historical_path, map_location="cpu", weights_only=True)
        historical = {
            "physical_ensemble": historical_raw["ensemble"],
            "truth": historical_raw["truth"],
            "persistence": historical_raw["persistence"],
            "valid_mask": historical_raw["valid_mask"],
        }
        image_size = tuple(int(value) for value in expected_manifest["image_size"])
        expected_members = int(expected_manifest["members"])
        baseline_cases = int(expected_manifest["baseline_cases"])
        historical_cases = int(expected_manifest["historical_cases"])
        _validate_payload(
            baseline,
            label="baseline",
            require_latent=True,
            expected_cases=baseline_cases,
            expected_members=expected_members,
            image_size=image_size,
        )
        _validate_payload(
            candidate,
            label="candidate",
            require_latent=True,
            expected_cases=baseline_cases,
            expected_members=expected_members,
            image_size=image_size,
        )
        _validate_payload(
            historical,
            label="historical_ema6",
            require_latent=False,
            expected_cases=historical_cases,
            expected_members=expected_members,
            image_size=image_size,
        )
        _validate_snapshot_identity(
            baseline,
            label="baseline",
            checkpoint_sha256=expected_manifest["baseline_checkpoint_sha256"],
            manifest=expected_manifest,
            expected_cases=baseline_cases,
        )
        _validate_snapshot_identity(
            candidate,
            label="candidate",
            checkpoint_sha256=expected_manifest["candidate_checkpoint_sha256"],
            manifest=expected_manifest,
            expected_cases=baseline_cases,
        )
        if not isinstance(historical_raw.get("case_identities"), list) or len(
            historical_raw["case_identities"]
        ) != historical_cases:
            raise ValueError("historical EMA6 case identities are missing")
        identity = paired_identity(baseline, candidate)
        baseline_decoder = decoder_invariants(baseline, means=means, stds=stds)
        candidate_decoder = decoder_invariants(candidate, means=means, stds=stds)
    except BaseException as error:
        failure = {
            "status": "failed",
            "stage": "payload_or_decoder_validation",
            "error_type": type(error).__name__,
            "message": str(error),
            "inputs": inputs,
            "source_identity": source_identity,
        }
        _atomic_json(output_dir / "failure.json", failure)
        return failure
    mandatory_passed = identity["passed"] and baseline_decoder["passed"] and candidate_decoder["passed"]
    if not mandatory_passed:
        failure = {
            "status": "failed",
            "stage": "paired_identity_or_decoder",
            "inputs": inputs,
            "source_identity": source_identity,
            "paired_identity": identity,
            "decoder": {"baseline": baseline_decoder, "candidate": candidate_decoder},
        }
        _atomic_json(output_dir / "failure.json", failure)
        return failure
    try:
        result = {
            "status": "complete",
            "cpu_only": True,
            "inputs": inputs,
            "source_identity": source_identity,
            "normalization": {"means": means, "stds": stds},
            "historical_case_identities": historical_raw["case_identities"],
            "paired_identity": identity,
            "decoder": {
                "baseline": baseline_decoder,
                "candidate": candidate_decoder,
            },
            "spatial": {
                "baseline": spatial_summary(baseline, means=means, stds=stds),
                "candidate": spatial_summary(candidate, means=means, stds=stds),
                "historical_ema6": spatial_summary(historical),
            },
            "limitations": [
                "Historical EMA6 cases are a visual/texture control, not paired dates with the heldout snapshots.",
                "This saved-sample stage is not a neural-network shift-equivariance or activation-localization test.",
            ],
        }
        if create_plots:
            physical_limits, residual_limits = _shared_plot_limits(
                baseline, candidate, means=means, stds=stds
            )
            result["plot_limits"] = {
                "physical": physical_limits,
                "residual": residual_limits,
                "origin": "upper",
                "interpolation": "nearest",
            }
            for label, payload in (("baseline", baseline), ("candidate", candidate)):
                _plot_pipeline(
                    payload,
                    output_dir,
                    label=label,
                    means=means,
                    stds=stds,
                    physical_limits=physical_limits,
                    residual_limits=residual_limits,
                )
        _atomic_json(output_dir / "audit.json", result)
        return result
    except BaseException as error:
        failure = {
            "status": "failed",
            "stage": "spatial_or_render",
            "error_type": type(error).__name__,
            "message": str(error),
            "inputs": inputs,
            "source_identity": source_identity,
            "paired_identity": identity,
            "decoder": {"baseline": baseline_decoder, "candidate": candidate_decoder},
        }
        _atomic_json(output_dir / "failure.json", failure)
        return failure


def main() -> None:
    torch.set_num_threads(6)
    torch.set_num_interop_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--historical", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = audit(
        args.baseline,
        args.candidate,
        args.historical,
        args.output_dir,
    )
    print(json.dumps({"status": result["status"], "output": str(args.output_dir)}))
    if result["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
