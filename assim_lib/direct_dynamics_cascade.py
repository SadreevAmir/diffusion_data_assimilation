"""Exact two-resolution representation for joint SIC/SIT dynamics.

The coarse field and the fine residual form a deterministic bijection on the
valid-ocean grid.  The fine component is projected into the nullspace of the
masked block-average operator, so it cannot silently change the coarse member.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class CascadeState:
    coarse: torch.Tensor
    coarse_ocean_fraction: torch.Tensor
    residual: torch.Tensor


def _validate(value: torch.Tensor, mask: torch.Tensor, factor: int) -> torch.Tensor:
    if value.ndim != 4 or mask.ndim != 4:
        raise ValueError("value and mask must have shape [batch,channel,y,x]")
    if mask.shape[0] != value.shape[0] or mask.shape[-2:] != value.shape[-2:]:
        raise ValueError("value and mask batch/spatial shapes differ")
    if mask.shape[1] not in (1, value.shape[1]):
        raise ValueError("mask must have one channel or match the value channels")
    if not isinstance(factor, int) or factor < 2:
        raise ValueError("cascade factor must be an integer >= 2")
    if value.shape[-2] % factor or value.shape[-1] % factor:
        raise ValueError("spatial dimensions must be divisible by the cascade factor")
    if not value.is_floating_point():
        raise ValueError("cascade values must use a floating-point dtype")
    if not torch.isfinite(mask).all() or torch.any((mask != 0) & (mask != 1)):
        raise ValueError("fine mask must be finite and binary")
    expanded = mask.expand_as(value).to(dtype=value.dtype)
    if not torch.any(expanded > 0):
        raise ValueError("fine mask contains no valid ocean")
    if not torch.isfinite(value[expanded > 0]).all():
        raise FloatingPointError("value contains NaN/Inf on valid ocean")
    return expanded


def masked_block_average(
    value: torch.Tensor, mask: torch.Tensor, factor: int = 2
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the fixed valid-ocean block-average operator D."""
    expanded = _validate(value, mask, factor)
    safe = torch.where(expanded > 0, value, torch.zeros_like(value))
    numerator = F.avg_pool2d(safe * expanded, kernel_size=factor, stride=factor)
    ocean_fraction = F.avg_pool2d(expanded, kernel_size=factor, stride=factor)
    coarse = numerator / ocean_fraction.clamp_min(torch.finfo(value.dtype).eps)
    coarse = torch.where(ocean_fraction > 0, coarse, torch.zeros_like(coarse))
    if not torch.isfinite(coarse[ocean_fraction > 0]).all():
        raise FloatingPointError("block averaging produced NaN/Inf on active coarse ocean")
    return coarse, ocean_fraction


def masked_cell_replication(coarse: torch.Tensor, fine_mask: torch.Tensor, factor: int = 2) -> torch.Tensor:
    """Right-inverse Q of D using constant values inside each coarse cell."""
    if coarse.ndim != 4 or fine_mask.ndim != 4:
        raise ValueError("coarse and fine_mask must be four-dimensional")
    if not coarse.is_floating_point():
        raise ValueError("coarse values must use a floating-point dtype")
    expected = (coarse.shape[-2] * factor, coarse.shape[-1] * factor)
    if fine_mask.shape[0] != coarse.shape[0] or fine_mask.shape[-2:] != expected:
        raise ValueError("fine mask is incompatible with the coarse tensor")
    if fine_mask.shape[1] not in (1, coarse.shape[1]):
        raise ValueError("fine mask channel count is incompatible with coarse")
    expanded_mask = fine_mask.expand(-1, coarse.shape[1], -1, -1)
    if not torch.isfinite(expanded_mask).all() or torch.any((expanded_mask != 0) & (expanded_mask != 1)):
        raise ValueError("fine mask must be finite and binary")
    support = F.avg_pool2d(expanded_mask.to(dtype=coarse.dtype), kernel_size=factor, stride=factor) > 0
    if not torch.isfinite(coarse[support]).all():
        raise FloatingPointError("coarse value contains NaN/Inf on active ocean cells")
    safe_coarse = torch.where(support, coarse, torch.zeros_like(coarse))
    repeated = safe_coarse.repeat_interleave(factor, -2).repeat_interleave(factor, -1)
    result = torch.where(expanded_mask > 0, repeated, torch.zeros_like(repeated))
    if not torch.isfinite(result[expanded_mask > 0]).all():
        raise FloatingPointError("cell replication produced NaN/Inf on valid ocean")
    return result


def smooth_right_inverse(coarse: torch.Tensor, fine_mask: torch.Tensor, factor: int = 2) -> torch.Tensor:
    """Lift coarse fields smoothly while enforcing D U = I exactly.

    Bilinear interpolation is corrected by a masked cell replication of its
    block-average error.  Empty land-only cells remain zero.
    """
    if coarse.ndim != 4 or fine_mask.ndim != 4 or not coarse.is_floating_point():
        raise ValueError("coarse and fine mask must be four-dimensional floating-point tensors")
    expected = (coarse.shape[-2] * factor, coarse.shape[-1] * factor)
    if fine_mask.shape[0] != coarse.shape[0] or fine_mask.shape[-2:] != expected:
        raise ValueError("fine mask is incompatible with the coarse tensor")
    if fine_mask.shape[1] not in (1, coarse.shape[1]):
        raise ValueError("fine mask channel count is incompatible with coarse")
    expanded_mask = fine_mask.expand(-1, coarse.shape[1], -1, -1)
    if not torch.isfinite(expanded_mask).all() or torch.any((expanded_mask != 0) & (expanded_mask != 1)):
        raise ValueError("fine mask must be finite and binary")
    support = F.avg_pool2d(expanded_mask.to(dtype=coarse.dtype), kernel_size=factor, stride=factor) > 0
    if not torch.isfinite(coarse[support]).all():
        raise FloatingPointError("coarse value contains NaN/Inf on active ocean cells")
    safe_coarse = torch.where(support, coarse, torch.zeros_like(coarse))
    interpolated_support = F.interpolate(
        support.to(dtype=coarse.dtype), size=expected, mode="bilinear", align_corners=False
    )
    smooth = F.interpolate(
        safe_coarse, size=expected, mode="bilinear", align_corners=False
    ) / interpolated_support.clamp_min(torch.finfo(coarse.dtype).eps)
    smooth = torch.where(interpolated_support > 0, smooth, torch.zeros_like(smooth))
    smooth = torch.where(expanded_mask > 0, smooth, torch.zeros_like(smooth))
    recovered, ocean_fraction = masked_block_average(smooth, fine_mask, factor)
    correction = torch.where(ocean_fraction > 0, safe_coarse - recovered, torch.zeros_like(coarse))
    result = smooth + masked_cell_replication(correction, fine_mask, factor)
    if not torch.isfinite(result[expanded_mask > 0]).all():
        raise FloatingPointError("smooth right inverse produced NaN/Inf on valid ocean")
    return result


def project_detail(value: torch.Tensor, mask: torch.Tensor, factor: int = 2) -> torch.Tensor:
    """Project a high-resolution field into ker(D)."""
    coarse, _ = masked_block_average(value, mask, factor)
    projected = value - smooth_right_inverse(coarse, mask, factor)
    projected = torch.where(mask.expand_as(value) > 0, projected, torch.zeros_like(projected))
    if not torch.isfinite(projected[mask.expand_as(value) > 0]).all():
        raise FloatingPointError("detail projection produced NaN/Inf on valid ocean")
    return projected


def decompose(value: torch.Tensor, mask: torch.Tensor, factor: int = 2) -> CascadeState:
    """Return C=D Y and R=(I-U D)Y for all channels jointly."""
    coarse, ocean_fraction = masked_block_average(value, mask, factor)
    lift = smooth_right_inverse(coarse, mask, factor)
    residual = torch.where(mask.expand_as(value) > 0, value - lift, torch.zeros_like(value))
    if not torch.isfinite(residual[mask.expand_as(value) > 0]).all():
        raise FloatingPointError("cascade residual contains NaN/Inf on valid ocean")
    return CascadeState(coarse, ocean_fraction, residual)


def reconstruct(state: CascadeState, mask: torch.Tensor, factor: int = 2) -> torch.Tensor:
    """Reconstruct the full field from a matched coarse member and residual."""
    lift = smooth_right_inverse(state.coarse, mask, factor)
    if state.residual.shape != lift.shape:
        raise ValueError("coarse member and residual shapes are incompatible")
    residual_coarse, ocean_fraction = masked_block_average(state.residual, mask, factor)
    if ocean_fraction.shape != state.coarse_ocean_fraction.shape or not torch.allclose(
        ocean_fraction, state.coarse_ocean_fraction.to(ocean_fraction), atol=0, rtol=0
    ):
        raise ValueError("stored and reconstructed coarse-ocean support differ")
    active = ocean_fraction > 0
    scale = max(float(state.residual[mask.expand_as(state.residual) > 0].abs().max()), 1.0)
    tolerance = 64 * torch.finfo(state.residual.dtype).eps * scale
    if torch.any(residual_coarse[active].abs() > tolerance):
        raise ValueError("residual is not in the nullspace of the coarse operator")
    value = lift + state.residual
    value = torch.where(mask.expand_as(value) > 0, value, torch.zeros_like(value))
    if not torch.isfinite(value[mask.expand_as(value) > 0]).all():
        raise FloatingPointError("cascade reconstruction produced NaN/Inf on valid ocean")
    return value


def residual_flow_pair(
    clean: torch.Tensor,
    noise: torch.Tensor,
    mask: torch.Tensor,
    time: torch.Tensor,
    factor: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct the fine conditional-flow state and velocity in ker(D)."""
    if time.ndim != 1 or time.shape[0] != clean.shape[0]:
        raise ValueError("time must have one scalar per batch element")
    if not torch.isfinite(time).all() or torch.any((time < 0) | (time > 1)):
        raise ValueError("flow time must be finite and lie in [0,1]")
    if noise.shape != clean.shape:
        raise ValueError("clean and noise shapes differ")
    clean_detail = project_detail(clean, mask, factor)
    noise_detail = project_detail(noise, mask, factor)
    time_view = time.to(dtype=clean.dtype, device=clean.device).reshape(-1, 1, 1, 1)
    state = (1.0 - time_view) * clean_detail + time_view * noise_detail
    velocity = noise_detail - clean_detail
    valid = mask.expand_as(clean) > 0
    if not torch.isfinite(state[valid]).all() or not torch.isfinite(velocity[valid]).all():
        raise FloatingPointError("residual flow pair produced NaN/Inf on valid ocean")
    return state, velocity, clean_detail, noise_detail
