"""Mathematically consistent coordinates for zero-inflated bounded SIC.

The generated state lives in unconstrained logit coordinates.  This avoids the
old mismatch where a flow was trained on values in ``[0, 1]`` and an unrelated
sigmoid was applied only after ODE integration.
"""

from __future__ import annotations

import torch


def open_unit_uniform_like(
    reference: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
    margin: float = 1e-5,
) -> torch.Tensor:
    """Draw a reproducible auxiliary variable strictly inside ``(0, 1)``."""
    if not torch.is_floating_point(reference):
        raise ValueError("reference must be floating point")
    if not 0.0 < float(margin) < 0.5:
        raise ValueError("margin must lie strictly between zero and one half")
    draw = torch.rand(
        reference.shape,
        dtype=reference.dtype,
        device=reference.device,
        generator=generator,
    )
    return float(margin) + (1.0 - 2.0 * float(margin)) * draw


def _require_open_unit(value: torch.Tensor, name: str) -> None:
    if not torch.is_floating_point(value) or not torch.all(torch.isfinite(value)):
        raise ValueError(f"{name} must be finite floating point")
    if not torch.all((value > 0.0) & (value < 1.0)):
        raise ValueError(f"{name} must lie strictly inside (0, 1)")


def encode_bounded_sic(
    concentration: torch.Tensor,
    occurrence_uniform: torch.Tensor,
    zero_intensity_uniform: torch.Tensor,
) -> torch.Tensor:
    """Encode physical SIC as occurrence-logit and conditional-intensity-logit.

    ``SIC=0`` is an exact atom.  Its intensity coordinate is filled with an
    independent auxiliary uniform so the continuous flow target remains
    nonsingular.  Positive SIC must be strictly below one; an archive with an
    exact upper atom requires a separate reviewed head and is rejected here.
    """
    if concentration.shape != occurrence_uniform.shape or concentration.shape != zero_intensity_uniform.shape:
        raise ValueError("concentration and auxiliary variables must have identical shapes")
    if not torch.is_floating_point(concentration) or not torch.all(torch.isfinite(concentration)):
        raise ValueError("concentration must be finite floating point")
    if not torch.all((concentration >= 0.0) & (concentration < 1.0)):
        raise ValueError("two-coordinate SIC pilot requires physical SIC in [0, 1)")
    _require_open_unit(occurrence_uniform, "occurrence_uniform")
    _require_open_unit(zero_intensity_uniform, "zero_intensity_uniform")

    occurrence = concentration > 0.0
    dequantized_occurrence = 0.5 * (
        occurrence.to(dtype=concentration.dtype) + occurrence_uniform
    )
    conditional_intensity = torch.where(
        occurrence,
        concentration,
        zero_intensity_uniform,
    )
    _require_open_unit(dequantized_occurrence, "dequantized_occurrence")
    _require_open_unit(conditional_intensity, "conditional_intensity")
    return torch.cat(
        (
            torch.logit(dequantized_occurrence),
            torch.logit(conditional_intensity),
        ),
        dim=1,
    )


def decode_bounded_sic(latent: torch.Tensor) -> torch.Tensor:
    """Decode an unconstrained two-channel flow state without clipping."""
    if latent.ndim < 2 or latent.shape[1] != 2:
        raise ValueError("bounded SIC latent must have exactly two channels")
    if not torch.is_floating_point(latent) or not torch.all(torch.isfinite(latent)):
        raise ValueError("bounded SIC latent must be finite floating point")
    occurrence = latent[:, :1] >= 0.0
    intensity = torch.sigmoid(latent[:, 1:2])
    return torch.where(occurrence, intensity, torch.zeros_like(intensity))


def bounded_sic_velocity_loss(
    prediction: torch.Tensor,
    target_velocity: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    occurrence_weight: float = 0.5,
    intensity_weight: float = 0.5,
) -> torch.Tensor:
    """Return a channel-balanced masked velocity MSE for the two SIC heads."""
    if prediction.shape != target_velocity.shape or prediction.ndim != 4 or prediction.shape[1] != 2:
        raise ValueError("prediction and target_velocity must have shape [B,2,H,W]")
    if valid_mask.ndim != 4 or valid_mask.shape[0] != prediction.shape[0] or valid_mask.shape[-2:] != prediction.shape[-2:]:
        raise ValueError("valid_mask must be co-registered with the velocity fields")
    if valid_mask.shape[1] not in (1, 2):
        raise ValueError("valid_mask must have one or two channels")
    weights = prediction.new_tensor([occurrence_weight, intensity_weight])
    if not torch.all(weights > 0.0) or not torch.isclose(weights.sum(), weights.new_tensor(1.0)):
        raise ValueError("positive channel weights must sum to one")
    mask = valid_mask[:, :1].to(dtype=prediction.dtype)
    squared = (prediction - target_velocity).square() * mask
    per_channel = squared.sum(dim=(0, 2, 3)) / mask.sum().clamp_min(1.0)
    return torch.sum(weights * per_channel)
