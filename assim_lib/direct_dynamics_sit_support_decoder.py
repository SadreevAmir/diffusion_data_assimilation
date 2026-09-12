"""Support-consistent decoder for fine SIT and its physical coarse mean."""

from __future__ import annotations

import torch

from .direct_dynamics_cascade import masked_block_average


def _to_blocks(value: torch.Tensor, factor: int) -> torch.Tensor:
    batch, channels, height, width = value.shape
    return (
        value.reshape(batch, channels, height // factor, factor, width // factor, factor)
        .permute(0, 1, 2, 4, 3, 5)
        .reshape(batch, channels, height // factor, width // factor, factor * factor)
    )


def _from_blocks(value: torch.Tensor, factor: int) -> torch.Tensor:
    batch, channels, coarse_height, coarse_width, _ = value.shape
    return (
        value.reshape(batch, channels, coarse_height, coarse_width, factor, factor)
        .permute(0, 1, 2, 4, 3, 5)
        .reshape(batch, channels, coarse_height * factor, coarse_width * factor)
    )


def project_masked_blocks_to_nonnegative_mean(
    fine: torch.Tensor,
    coarse: torch.Tensor,
    fine_mask: torch.Tensor,
    factor: int = 2,
) -> torch.Tensor:
    """Project each valid-ocean block onto a simplex with mean ``max(coarse, 0)``.

    The operation is the exact Euclidean projection over valid pixels in each
    block. Land pixels remain zero. It is differentiable almost everywhere;
    no straight-through estimator is used.
    """
    if fine.ndim != 4 or coarse.ndim != 4 or fine_mask.ndim != 4:
        raise ValueError("fine, coarse and fine_mask must be four-dimensional")
    if not isinstance(factor, int) or factor < 2:
        raise ValueError("factor must be an integer >= 2")
    if fine.shape[-2] % factor or fine.shape[-1] % factor:
        raise ValueError("fine spatial dimensions must be divisible by factor")
    expected_coarse = (
        fine.shape[0],
        fine.shape[1],
        fine.shape[-2] // factor,
        fine.shape[-1] // factor,
    )
    if coarse.shape != expected_coarse:
        raise ValueError("coarse shape is incompatible with fine")
    if (
        fine_mask.shape[0] != fine.shape[0]
        or fine_mask.shape[-2:] != fine.shape[-2:]
    ):
        raise ValueError("fine mask batch/spatial shape is incompatible with fine")
    if fine_mask.shape[1] not in (1, fine.shape[1]):
        raise ValueError("fine mask must have one channel or match fine")
    if not fine.is_floating_point() or not coarse.is_floating_point():
        raise ValueError("fine and coarse must use floating-point dtypes")

    work_dtype = (
        torch.float32 if fine.dtype in (torch.float16, torch.bfloat16) else fine.dtype
    )
    fine_work = fine.to(dtype=work_dtype)
    coarse_work = coarse.to(device=fine.device, dtype=work_dtype)
    mask = fine_mask.to(device=fine.device).expand_as(fine)
    if not torch.isfinite(mask).all() or torch.any((mask != 0) & (mask != 1)):
        raise ValueError("fine mask must be finite and binary")
    support = mask > 0
    if not torch.any(support):
        raise ValueError("fine mask contains no valid ocean")
    if not torch.isfinite(fine[support]).all():
        raise FloatingPointError("fine contains NaN/Inf on valid ocean")
    coarse_fraction = _to_blocks(mask.to(dtype=work_dtype), factor).mean(dim=-1)
    active = coarse_fraction > 0
    if not torch.isfinite(coarse_work[active]).all():
        raise FloatingPointError("coarse contains NaN/Inf on active ocean")

    blocks = _to_blocks(fine_work, factor)
    block_mask = _to_blocks(mask, factor) > 0
    count = block_mask.sum(dim=-1)
    target_sum = count.to(dtype=work_dtype) * coarse_work.clamp_min(0)
    if not torch.isfinite(target_sum[active]).all():
        raise FloatingPointError("simplex target sum overflowed")

    negative_infinity = torch.full_like(blocks, -torch.inf)
    block_maximum = torch.where(block_mask, blocks, negative_infinity).max(dim=-1).values
    block_maximum = torch.where(active, block_maximum, torch.zeros_like(block_maximum))
    centered = torch.where(
        block_mask, blocks - block_maximum.unsqueeze(-1), torch.zeros_like(blocks)
    )
    if not torch.isfinite(centered[block_mask]).all():
        raise FloatingPointError("simplex centering overflowed")
    sorted_values = torch.sort(
        torch.where(block_mask, centered, negative_infinity), dim=-1, descending=True
    ).values
    positions = torch.arange(
        1, factor * factor + 1, device=fine.device, dtype=work_dtype
    ).reshape((1,) * (blocks.ndim - 1) + (-1,))
    valid_positions = positions <= count.unsqueeze(-1)
    cumulative = torch.where(valid_positions, sorted_values, 0).cumsum(dim=-1)
    if not torch.isfinite(cumulative[valid_positions]).all():
        raise FloatingPointError("simplex cumulative sum overflowed")
    theta_candidates = (cumulative - target_sum.unsqueeze(-1)) / positions
    if not torch.isfinite(theta_candidates[valid_positions]).all():
        raise FloatingPointError("simplex threshold overflowed")
    selected = valid_positions & (sorted_values > theta_candidates)
    rho_raw = selected.sum(dim=-1)
    positive_budget = target_sum > 0
    if torch.any(positive_budget & (rho_raw == 0)):
        raise RuntimeError("positive simplex budget has no admissible rho")
    rho = rho_raw.clamp_min(1)
    theta = torch.gather(theta_candidates, -1, rho.unsqueeze(-1) - 1)
    projected_blocks = torch.where(
        block_mask,
        (centered - theta).clamp_min(0),
        torch.zeros_like(blocks),
    )
    projected_blocks = torch.where(
        positive_budget.unsqueeze(-1),
        projected_blocks,
        torch.zeros_like(projected_blocks),
    )
    achieved_sum = projected_blocks.sum(dim=-1)
    budget_scale = torch.maximum(
        torch.ones_like(target_sum),
        torch.maximum(target_sum.abs(), projected_blocks.abs().sum(dim=-1)),
    )
    budget_tolerance = 64 * torch.finfo(work_dtype).eps * budget_scale
    if torch.any((achieved_sum - target_sum).abs()[active] > budget_tolerance[active]):
        raise RuntimeError("simplex projection failed to preserve its target sum")
    projected = _from_blocks(projected_blocks, factor)
    projected = torch.where(support, projected, torch.zeros_like(projected))
    if not torch.isfinite(projected[support]).all():
        raise FloatingPointError("support decoder produced NaN/Inf")
    # Half/bfloat16 callers receive the promoted FP32 result. Casting back can
    # overflow a valid simplex solution or destroy its exact block budget.
    return projected


def support_decoder_checks(
    fine: torch.Tensor,
    coarse: torch.Tensor,
    fine_mask: torch.Tensor,
    factor: int = 2,
    fp32_ulps: int = 128,
) -> dict[str, float]:
    """Run invariant checks and return their maximum physical errors."""
    decoded = project_masked_blocks_to_nonnegative_mean(fine, coarse, fine_mask, factor)
    recovered, fraction = masked_block_average(decoded, fine_mask, factor)
    active = fraction > 0
    target = coarse.clamp_min(0)
    scale = max(1.0, float(target[active].abs().max()), float(decoded.abs().max()))
    tolerance = fp32_ulps * torch.finfo(torch.float32).eps * scale
    minimum = float(decoded[fine_mask.expand_as(decoded) > 0].min())
    coarse_error = float((recovered - target)[active].abs().max())
    if minimum < -tolerance or coarse_error > tolerance:
        raise RuntimeError("support decoder invariant exceeds FP32-source tolerance")
    return {
        "fp32_source_tolerance": tolerance,
        "minimum_valid_value": minimum,
        "maximum_coarse_error": coarse_error,
    }
