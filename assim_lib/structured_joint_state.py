"""Distribution-preserving coordinates for joint SIC/SIT flow matching.

The M2M archive has a joint open-water atom (SIC=SIT=0) and a censored
upper SIC value.  A smooth ODE started from Gaussian noise cannot create those
atoms exactly in ordinary Euclidean SIC/SIT coordinates.  This module instead
uses continuously dequantized occurrence/censoring coordinates and continuous
coordinates only on the branches where they are physically defined.

Inactive intensity coordinates receive independent standard-normal draws and
are discarded by the physical decoder.  This makes those auxiliary coordinates
continuous without claiming that their spatial statistics hide the discrete
class.  Interior values in the float16 archive are treated as a continuous-
resolution approximation; only the structural open-water and SIC-cap atoms are
represented explicitly.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping

import torch

SCHEMA_VERSION = "structured_joint_state_stats_v3"
LATENT_CHANNELS = 4


class StructuredDecodeSaturationError(ValueError):
    """Finite latent could not be represented on its declared physical branch."""

    def __init__(self, message: str, diagnostics: Mapping[str, object]):
        super().__init__(message)
        self.diagnostics = dict(diagnostics)
INACTIVE_FILLER_LAW = "independent_standard_normal"
INTERIOR_VALUE_LAW = "continuous_resolution_approximation"


def canonical_mapping_sha256(payload: Mapping) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finite_scalar(stats: Mapping, name: str) -> float:
    value = float(stats[name])
    if not math.isfinite(value):
        raise ValueError(f"structured-state statistic {name!r} must be finite")
    return value


def validate_structured_state_stats(stats: Mapping) -> None:
    """Validate the immutable train-only statistics contract."""
    if stats.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"expected schema_version={SCHEMA_VERSION!r}")
    if stats.get("source_split") != "train":
        raise ValueError("structured-state statistics must come from the train split")
    if not isinstance(stats.get("data_config_sha256"), str) or len(stats["data_config_sha256"]) != 64:
        raise ValueError("structured-state statistics require data_config_sha256")
    if not isinstance(stats.get("pair_manifest_sha256"), str) or len(stats["pair_manifest_sha256"]) != 64:
        raise ValueError("structured-state statistics require pair_manifest_sha256")

    cap = _finite_scalar(stats, "sic_cap")
    if not 0.0 < cap <= 1.0:
        raise ValueError("sic_cap must lie in (0, 1]")
    for prefix in ("occurrence_logit", "cap_logit", "sic_interior_logit", "sit_positive_log"):
        _finite_scalar(stats, f"{prefix}_mean")
        scale = _finite_scalar(stats, f"{prefix}_std")
        if scale <= 0.0:
            raise ValueError(f"{prefix}_std must be positive")

    probability = _finite_scalar(stats, "cap_probability_given_ice")
    if not 0.0 < probability < 1.0:
        raise ValueError("cap_probability_given_ice must lie strictly inside (0, 1)")
    if stats.get("inactive_filler_law") != INACTIVE_FILLER_LAW:
        raise ValueError(f"inactive_filler_law must be {INACTIVE_FILLER_LAW!r}")
    if stats.get("interior_value_law") != INTERIOR_VALUE_LAW:
        raise ValueError(f"interior_value_law must be {INTERIOR_VALUE_LAW!r}")
    normalization = stats.get("conditioning_normalization")
    if not isinstance(normalization, Mapping):
        raise ValueError("structured-state statistics require conditioning_normalization")
    if normalization.get("source_split") != "train":
        raise ValueError("conditioning normalization must come from the train split")
    if normalization.get("fields") != ["siconc", "sithic"]:
        raise ValueError("conditioning normalization must bind SIC/SIT field order")
    if normalization.get("open_water_nan_as_physical_zero") is not True:
        raise ValueError("conditioning normalization must include open water as physical zero")
    means = normalization.get("means")
    stds = normalization.get("stds")
    if not isinstance(means, list) or not isinstance(stds, list) or len(means) != 2 or len(stds) != 2:
        raise ValueError("conditioning normalization requires two means and two stds")
    for name, values in (("means", means), ("stds", stds)):
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError(f"conditioning normalization {name} must be finite")
    if not all(float(value) > 0.0 for value in stds):
        raise ValueError("conditioning normalization stds must be positive")


def validate_conditioning_normalization(data_config: Mapping, stats: Mapping) -> None:
    """Require model conditioning normalization to equal the train-only artifact."""
    validate_structured_state_stats(stats)
    normalization = stats["conditioning_normalization"]
    expected = {
        "fields": normalization["fields"],
        "means": normalization["means"],
        "stds": normalization["stds"],
    }
    actual = {
        "fields": list(data_config.get("fields", [])),
        "means": list(data_config.get("means", [])),
        "stds": list(data_config.get("stds", [])),
    }
    if actual != expected:
        raise ValueError(
            f"data conditioning normalization does not match train-only statistics: "
            f"actual={actual}, expected={expected}"
        )


def _rand_like(reference: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
    return torch.rand(
        reference.shape,
        dtype=reference.dtype,
        device=reference.device,
        generator=generator,
    )


def _open_unit_rand_like(
    reference: torch.Tensor,
    generator: torch.Generator | None,
) -> torch.Tensor:
    uniform = _rand_like(reference, generator)
    epsilon = torch.finfo(reference.dtype).eps
    return uniform.clamp(min=epsilon, max=1.0 - epsilon)


def _standardize(value: torch.Tensor, stats: Mapping, prefix: str) -> torch.Tensor:
    mean = _finite_scalar(stats, f"{prefix}_mean")
    std = _finite_scalar(stats, f"{prefix}_std")
    return (value - mean) / std


def _unstandardize(value: torch.Tensor, stats: Mapping, prefix: str) -> torch.Tensor:
    mean = _finite_scalar(stats, f"{prefix}_mean")
    std = _finite_scalar(stats, f"{prefix}_std")
    return value * std + mean


def _standard_normal_like(
    reference: torch.Tensor,
    generator: torch.Generator | None,
) -> torch.Tensor:
    return torch.randn(
        reference.shape,
        dtype=reference.dtype,
        device=reference.device,
        generator=generator,
    )


def encode_structured_joint_state(
    physical_state: torch.Tensor,
    valid_mask: torch.Tensor,
    stats: Mapping,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Encode physical ``[SIC,SIT]`` into four continuous flow coordinates."""
    validate_structured_state_stats(stats)
    if physical_state.ndim != 4 or physical_state.shape[1] != 2:
        raise ValueError("physical_state must have shape [B,2,H,W]")
    if valid_mask.ndim != 4 or valid_mask.shape[0] != physical_state.shape[0]:
        raise ValueError("valid_mask must be co-registered with physical_state")
    if valid_mask.shape[-2:] != physical_state.shape[-2:] or valid_mask.shape[1] not in (1, 2):
        raise ValueError("valid_mask must have one or two spatially co-registered channels")
    valid = valid_mask[:, :1] > 0
    sic = physical_state[:, :1]
    sit = physical_state[:, 1:2]
    if not torch.all(torch.isfinite(sic[valid])) or not torch.all(torch.isfinite(sit[valid])):
        raise ValueError("SIC and SIT must be finite on the valid domain")

    cap = _finite_scalar(stats, "sic_cap")
    tolerance = max(torch.finfo(sic.dtype).eps * max(cap, 1.0) * 2.0, 1e-7)
    if not torch.all((sic[valid] >= 0.0) & (sic[valid] <= cap + tolerance)):
        raise ValueError("SIC is outside the train-supported interval [0, sic_cap]")
    if not torch.all(sit[valid] >= 0.0):
        raise ValueError("SIT must be non-negative")
    occurrence_sic = sic > 0.0
    occurrence_sit = sit > 0.0
    if not torch.equal(occurrence_sic[valid], occurrence_sit[valid]):
        raise ValueError("joint support requires SIC>0 if and only if SIT>0")

    occurrence = occurrence_sic
    capped = occurrence & (torch.abs(sic - cap) <= tolerance)
    interior = occurrence & ~capped
    if torch.any(interior & ((sic <= 0.0) | (sic >= cap))):
        raise ValueError("interior SIC must lie strictly inside (0, sic_cap)")

    u_occurrence = _open_unit_rand_like(sic, generator)
    occurrence_logit = torch.logit(0.5 * (occurrence.to(sic.dtype) + u_occurrence))
    z_occurrence = _standardize(occurrence_logit, stats, "occurrence_logit")

    cap_probability = _finite_scalar(stats, "cap_probability_given_ice")
    filler_cap = _rand_like(sic, generator) < cap_probability
    cap_class = torch.where(occurrence, capped, filler_cap)
    u_cap = _open_unit_rand_like(sic, generator)
    cap_logit = torch.logit(0.5 * (cap_class.to(sic.dtype) + u_cap))
    z_cap = _standardize(cap_logit, stats, "cap_logit")

    safe_ratio = torch.where(interior, sic / cap, torch.full_like(sic, 0.5))
    active_sic = _standardize(torch.logit(safe_ratio), stats, "sic_interior_logit")
    # These coordinates are absent on the current physical branch and are
    # discarded by the decoder.  Independent continuous noise avoids adding
    # the artificial atoms created by a flat empirical quantile table.  It is
    # not claimed to conceal class information in spatial/joint statistics.
    filler_sic = _standard_normal_like(sic, generator)
    z_sic = torch.where(interior, active_sic, filler_sic)

    safe_sit = torch.where(occurrence, sit, torch.ones_like(sit))
    active_sit = _standardize(torch.log(safe_sit), stats, "sit_positive_log")
    filler_sit = _standard_normal_like(sit, generator)
    z_sit = torch.where(occurrence, active_sit, filler_sit)
    latent = torch.cat((z_occurrence, z_cap, z_sic, z_sit), dim=1)
    if not torch.all(torch.isfinite(latent[valid.expand_as(latent)])):
        raise ValueError("structured joint encoding produced a non-finite latent")
    return latent


def decode_structured_joint_state(latent: torch.Tensor, stats: Mapping) -> torch.Tensor:
    """Decode four flow coordinates to exact joint SIC/SIT support without clipping."""
    validate_structured_state_stats(stats)
    if latent.ndim != 4 or latent.shape[1] != LATENT_CHANNELS:
        raise ValueError(f"latent must have shape [B,{LATENT_CHANNELS},H,W]")
    if not torch.all(torch.isfinite(latent)):
        raise ValueError("latent must be finite")

    occurrence_logit = _unstandardize(latent[:, 0:1], stats, "occurrence_logit")
    cap_logit = _unstandardize(latent[:, 1:2], stats, "cap_logit")
    occurrence = occurrence_logit >= 0.0
    capped = occurrence & (cap_logit >= 0.0)
    cap = _finite_scalar(stats, "sic_cap")
    interior_logit = _unstandardize(latent[:, 2:3], stats, "sic_interior_logit")
    positive_sit_log = _unstandardize(latent[:, 3:4], stats, "sit_positive_log")
    interior_sic = cap * torch.sigmoid(interior_logit)
    positive_sit = torch.exp(positive_sit_log)
    interior = occurrence & ~capped
    sic_saturated = interior & ((interior_sic <= 0.0) | (interior_sic >= cap))
    sit_saturated = occurrence & ((positive_sit <= 0.0) | ~torch.isfinite(positive_sit))
    sic_count = int(sic_saturated.sum().item())
    sit_count = int(sit_saturated.sum().item())
    if sic_count or sit_count:
        diagnostics = {
            "sic_saturation_count": sic_count,
            "sit_saturation_count": sit_count,
            "latent_dtype": str(latent.dtype),
            "latent_min": float(latent.min().item()),
            "latent_max": float(latent.max().item()),
            "interior_logit_min": float(interior_logit.min().item()),
            "interior_logit_max": float(interior_logit.max().item()),
            "positive_sit_log_min": float(positive_sit_log.min().item()),
            "positive_sit_log_max": float(positive_sit_log.max().item()),
        }
        if sic_count and not sit_count:
            message = "decoded active interior SIC saturated outside the open interval"
        elif sit_count and not sic_count:
            message = "decoded active SIT saturated or became non-finite"
        else:
            message = "decoded active SIC/SIT saturated outside the declared support"
        raise StructuredDecodeSaturationError(message, diagnostics)
    sic = torch.where(occurrence, torch.where(capped, torch.full_like(interior_sic, cap), interior_sic), 0.0)
    sit = torch.where(occurrence, positive_sit, 0.0)
    physical = torch.cat((sic, sit), dim=1)
    if not torch.all(torch.isfinite(physical)):
        raise ValueError("structured joint decoding produced a non-finite physical state")
    return physical


def encode_structured_joint_trajectory(
    physical_trajectory: torch.Tensor,
    valid_mask: torch.Tensor,
    stats: Mapping,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Encode flattened ``[SIC_d,SIT_d,...,SIC_d+K,SIT_d+K]`` jointly."""
    if physical_trajectory.ndim != 4 or physical_trajectory.shape[1] % 2 != 0:
        raise ValueError("physical trajectory must have shape [B,2*T,H,W]")
    latents = []
    for start in range(0, physical_trajectory.shape[1], 2):
        latents.append(
            encode_structured_joint_state(
                physical_trajectory[:, start : start + 2],
                valid_mask,
                stats,
                generator=generator,
            )
        )
    return torch.cat(latents, dim=1)


def decode_structured_joint_trajectory(latent: torch.Tensor, stats: Mapping) -> torch.Tensor:
    """Decode flattened four-coordinate trajectory into paired physical fields."""
    if latent.ndim != 4 or latent.shape[1] % LATENT_CHANNELS != 0:
        raise ValueError("latent trajectory must have shape [B,4*T,H,W]")
    if not torch.all(torch.isfinite(latent)):
        raise ValueError("latent trajectory must be finite before per-lead decoding")
    physical = []
    for lead, start in enumerate(range(0, latent.shape[1], LATENT_CHANNELS)):
        try:
            physical.append(
                decode_structured_joint_state(latent[:, start : start + 4], stats)
            )
        except StructuredDecodeSaturationError as error:
            raise StructuredDecodeSaturationError(
                f"lead {lead}: {error}",
                {**error.diagnostics, "lead_index": lead},
            ) from error
    return torch.cat(physical, dim=1)


def dequantized_logit_moments(probability: float) -> tuple[float, float]:
    """Exact moments of logit((Bernoulli(p)+U)/2), U~Uniform(0,1)."""
    probability = float(probability)
    if not 0.0 < probability < 1.0:
        raise ValueError("probability must lie strictly inside (0, 1)")
    mean = (2.0 * probability - 1.0) * (2.0 * math.log(2.0))
    variance = math.pi**2 / 3.0 - mean * mean
    if variance <= 0.0:
        raise ValueError("dequantized-logit variance must be positive")
    return mean, math.sqrt(variance)
