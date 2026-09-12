"""Uniform physical SIC decoder preserving masked 2x2 coarse means."""

from __future__ import annotations

import torch

from .direct_dynamics_cascade import masked_block_average
from .direct_dynamics_sit_support_decoder import _from_blocks, _to_blocks


def project_masked_blocks_to_unit_interval_mean(
    fine: torch.Tensor,
    coarse: torch.Tensor,
    fine_mask: torch.Tensor,
    factor: int = 2,
) -> torch.Tensor:
    """Euclidean projection onto ``0 <= z <= 1`` and ``mean(z)=clip(C,0,1)``.

    Every valid ocean value is treated by the same law. Land remains zero.
    Computation is in float64 so the capped-simplex constraint survives the
    counterfactual scoring audit without a hidden fitting parameter.
    """
    if fine.ndim != 4 or coarse.ndim != 4 or fine_mask.ndim != 4:
        raise ValueError("fine, coarse and fine_mask must be four-dimensional")
    if not isinstance(factor, int) or factor < 2:
        raise ValueError("factor must be an integer >= 2")
    if fine.shape[-2] % factor or fine.shape[-1] % factor:
        raise ValueError("fine spatial dimensions must be divisible by factor")
    expected_coarse = (
        fine.shape[0], fine.shape[1], fine.shape[-2] // factor, fine.shape[-1] // factor
    )
    if coarse.shape != expected_coarse:
        raise ValueError("coarse shape is incompatible with fine")
    if fine_mask.shape[0] != fine.shape[0] or fine_mask.shape[-2:] != fine.shape[-2:]:
        raise ValueError("fine mask batch/spatial shape is incompatible with fine")
    if fine_mask.shape[1] not in (1, fine.shape[1]):
        raise ValueError("fine mask must have one channel or match fine")
    if not fine.is_floating_point() or not coarse.is_floating_point():
        raise ValueError("fine and coarse must use floating-point dtypes")

    values = fine.double()
    coarse_values = coarse.to(device=fine.device, dtype=torch.float64)
    mask = fine_mask.to(device=fine.device).expand_as(fine)
    if not torch.isfinite(mask).all() or torch.any((mask != 0) & (mask != 1)):
        raise ValueError("fine mask must be finite and binary")
    support = mask > 0
    if not torch.any(support):
        raise ValueError("fine mask contains no valid ocean")
    if not torch.isfinite(values[support]).all():
        raise FloatingPointError("fine SIC contains NaN/Inf on valid ocean")

    blocks = _to_blocks(values, factor)
    block_mask = _to_blocks(mask, factor) > 0
    count = block_mask.sum(dim=-1)
    active = count > 0
    if not torch.isfinite(coarse_values[active]).all():
        raise FloatingPointError("coarse SIC contains NaN/Inf on active ocean")
    target_mean = coarse_values.clamp(0.0, 1.0)
    target_sum = count.double() * target_mean

    positive_infinity = torch.full_like(blocks, torch.inf)
    negative_infinity = torch.full_like(blocks, -torch.inf)
    block_maximum = torch.where(block_mask, blocks, negative_infinity).max(dim=-1).values
    block_maximum = torch.where(active, block_maximum, torch.zeros_like(block_maximum))
    centered = torch.where(
        block_mask, blocks - block_maximum.unsqueeze(-1), torch.zeros_like(blocks)
    )
    if not torch.isfinite(centered[block_mask]).all():
        raise FloatingPointError("centered SIC values overflowed")
    centered_minimum = torch.where(
        block_mask, centered, positive_infinity
    ).min(dim=-1).values
    lower_theta = torch.where(active, centered_minimum - 1.0, torch.zeros_like(centered_minimum))
    upper_theta = torch.zeros_like(lower_theta)

    # For z(theta)=clip(y-theta,0,1), sum(z) decreases monotonically.
    # 80 bisections are deterministic and far beyond float64 resolution here.
    for _ in range(80):
        theta = 0.5 * (lower_theta + upper_theta)
        projected = torch.where(
            block_mask,
            (centered - theta.unsqueeze(-1)).clamp(0.0, 1.0),
            torch.zeros_like(blocks),
        )
        sum_is_too_large = projected.sum(dim=-1) > target_sum
        lower_theta = torch.where(sum_is_too_large, theta, lower_theta)
        upper_theta = torch.where(sum_is_too_large, upper_theta, theta)

    theta = 0.5 * (lower_theta + upper_theta)
    projected = torch.where(
        block_mask,
        (centered - theta.unsqueeze(-1)).clamp(0.0, 1.0),
        torch.zeros_like(blocks),
    )
    residual = target_sum - projected.sum(dim=-1)
    budget_scale = torch.maximum(torch.ones_like(target_sum), target_sum.abs())
    pre_correction_tolerance = 256 * torch.finfo(torch.float64).eps * budget_scale
    if torch.any(residual.abs()[active] > pre_correction_tolerance[active]):
        raise RuntimeError("capped-simplex bisection did not converge")
    # Remove only the proven roundoff residue without changing inactive coordinates.
    for index in range(factor * factor):
        current = projected[..., index]
        usable = block_mask[..., index]
        delta = torch.minimum(residual.clamp_min(0), 1.0 - current)
        delta += torch.maximum(residual.clamp_max(0), -current)
        delta = torch.where(usable, delta, torch.zeros_like(delta))
        projected[..., index] = current + delta
        residual = residual - delta

    result = _from_blocks(projected, factor)
    result = torch.where(support, result, torch.zeros_like(result))
    if not torch.isfinite(result[support]).all():
        raise FloatingPointError("bounded SIC decoder produced NaN/Inf")
    tolerance = 128 * torch.finfo(torch.float64).eps
    if float(result[support].min()) < -tolerance or float(result[support].max()) > 1 + tolerance:
        raise RuntimeError("bounded SIC decoder violated [0,1]")
    recovered, fraction = masked_block_average(result, mask, factor)
    if float((recovered - target_mean)[fraction > 0].abs().max()) > tolerance:
        raise RuntimeError("bounded SIC decoder failed to preserve clipped coarse means")
    return result


def sic_support_decoder_checks(
    fine: torch.Tensor,
    coarse: torch.Tensor,
    fine_mask: torch.Tensor,
    factor: int = 2,
) -> dict[str, float]:
    decoded = project_masked_blocks_to_unit_interval_mean(
        fine, coarse, fine_mask, factor
    )
    recovered, fraction = masked_block_average(decoded, fine_mask, factor)
    active = fraction > 0
    target = coarse.double().clamp(0.0, 1.0)
    valid = fine_mask.expand_as(decoded) > 0
    return {
        "minimum_valid_sic": float(decoded[valid].min()),
        "maximum_valid_sic": float(decoded[valid].max()),
        "maximum_clipped_coarse_error": float((recovered - target)[active].abs().max()),
    }
