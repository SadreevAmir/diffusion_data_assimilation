"""Proper trajectory-geometry score for native SIC/SIT ensembles.

The module deliberately contains no sampler or optimizer code.  It maps every
member and the verifying trajectory through the same condition-dependent,
target-independent observable map and evaluates an unbiased finite-ensemble
energy score.  Adding this score to a strictly proper native score therefore
does not turn rank-histogram fitting into the training objective.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F


LEADS = 3
FIELDS = 2
CHANNELS = LEADS * FIELDS
SIC_THRESHOLD = 0.15
EDGE_RADII = (2, 6, 12)
GROUP_WEIGHTS = {
    "d160_trajectory": 0.25,
    "d80_increments": 0.25,
    "regional_area_product": 0.25,
    "initial_edge_profiles": 0.25,
}


def _validate_inputs(
    members: torch.Tensor,
    truth: torch.Tensor,
    valid: torch.Tensor,
    initial_sic: torch.Tensor,
    cell_area: torch.Tensor,
) -> None:
    if members.ndim != 5 or members.shape[1] < 2 or members.shape[2] != CHANNELS:
        raise ValueError("members must have shape [B,M,6,H,W] with M>=2")
    if truth.shape != members.shape[:1] + members.shape[2:]:
        raise ValueError("truth must have shape [B,6,H,W]")
    expected_scalar = truth.shape[:1] + (1,) + truth.shape[-2:]
    if valid.shape != expected_scalar or initial_sic.shape != expected_scalar:
        raise ValueError("valid and initial_sic must have shape [B,1,H,W]")
    if cell_area.shape not in (expected_scalar, (1, 1, *truth.shape[-2:])):
        raise ValueError("cell_area must be per-case or broadcastable [1,1,H,W]")
    if not torch.all((valid == 0) | (valid == 1)):
        raise ValueError("valid must be exactly binary")
    if not torch.all(torch.isfinite(cell_area)) or torch.any(cell_area < 0):
        raise ValueError("cell_area must be finite and non-negative")
    positive_area = (cell_area * valid).sum(dim=(-2, -1))
    if torch.any(positive_area <= 0):
        raise ValueError("every case needs positive valid-ocean area")
    member_mask = valid[:, None].expand_as(members) > 0
    truth_mask = valid.expand_as(truth) > 0
    if not torch.all(torch.isfinite(members[member_mask])):
        raise FloatingPointError("members contain NaN/Inf on valid ocean")
    if not torch.all(torch.isfinite(truth[truth_mask])):
        raise FloatingPointError("truth contains NaN/Inf on valid ocean")
    if not torch.all(torch.isfinite(initial_sic[valid > 0])):
        raise FloatingPointError("initial SIC contains NaN/Inf on valid ocean")


def _safe_physical(
    members: torch.Tensor, truth: torch.Tensor, valid: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    member_mask = valid[:, None].expand_as(members) > 0
    truth_mask = valid.expand_as(truth) > 0
    return (
        torch.where(member_mask, members, torch.zeros_like(members)),
        torch.where(truth_mask, truth, torch.zeros_like(truth)),
    )


def _area_weighted_block_mean(
    values: torch.Tensor,
    valid: torch.Tensor,
    cell_area: torch.Tensor,
    factor: int,
) -> torch.Tensor:
    if factor <= 0 or values.shape[-2] % factor or values.shape[-1] % factor:
        raise ValueError("block factor must divide both spatial dimensions")
    if values.ndim not in (4, 5):
        raise ValueError("values must be [B,C,H,W] or [B,M,C,H,W]")
    area = cell_area.expand(values.shape[0], -1, -1, -1) * valid
    if values.ndim == 5:
        area = area[:, None]
        flat_values = values.flatten(0, 1)
        flat_area = area.expand(-1, values.shape[1], -1, -1, -1).flatten(0, 1)
        numerator = F.avg_pool2d(flat_values * flat_area, factor, factor)
        denominator = F.avg_pool2d(flat_area, factor, factor)
        result = torch.where(
            denominator > 0,
            numerator / denominator.clamp_min(torch.finfo(values.dtype).tiny),
            torch.zeros_like(numerator),
        )
        return result.unflatten(0, (values.shape[0], values.shape[1]))
    numerator = F.avg_pool2d(values * area, factor, factor)
    denominator = F.avg_pool2d(area, factor, factor)
    return torch.where(
        denominator > 0,
        numerator / denominator.clamp_min(torch.finfo(values.dtype).tiny),
        torch.zeros_like(numerator),
    )


def _trajectory_increments(values: torch.Tensor) -> torch.Tensor:
    if values.shape[-3] != CHANNELS:
        raise ValueError("trajectory must contain interleaved SIC/SIT at three leads")
    return torch.cat(
        (
            values[..., 2:4, :, :] - values[..., 0:2, :, :],
            values[..., 4:6, :, :] - values[..., 2:4, :, :],
        ),
        dim=-3,
    )


def _fixed_region_means(
    values: torch.Tensor,
    valid: torch.Tensor,
    cell_area: torch.Tensor,
    *,
    regions_y: int = 2,
    regions_x: int = 2,
) -> torch.Tensor:
    """Return area-weighted regional means without target-dependent regions."""

    height, width = values.shape[-2:]
    if height % regions_y or width % regions_x:
        raise ValueError("fixed region grid must divide the image")
    area = cell_area.expand(values.shape[0], -1, -1, -1) * valid
    if values.ndim == 5:
        area = area[:, None]
    rows = []
    step_y, step_x = height // regions_y, width // regions_x
    for row in range(regions_y):
        for column in range(regions_x):
            ys = slice(row * step_y, (row + 1) * step_y)
            xs = slice(column * step_x, (column + 1) * step_x)
            region_area = area[..., ys, xs]
            denominator = region_area.sum(dim=(-2, -1)).clamp_min(
                torch.finfo(values.dtype).tiny
            )
            numerator = (values[..., ys, xs] * region_area).sum(dim=(-2, -1))
            rows.append(numerator / denominator)
    return torch.stack(rows, dim=-1)


def _edge_band_masks(
    initial_sic: torch.Tensor,
    valid: torch.Tensor,
    radii: Sequence[int] = EDGE_RADII,
) -> torch.Tensor:
    """Build nested inner/outer bands from the issued initial ice edge."""

    if tuple(sorted(set(int(value) for value in radii))) != tuple(radii):
        raise ValueError("edge radii must be strictly increasing positive integers")
    ice = (initial_sic >= SIC_THRESHOLD).to(initial_sic.dtype) * valid
    water = valid.to(initial_sic.dtype)
    open_water = (water - ice).clamp(0.0, 1.0)
    previous_inner = torch.zeros_like(ice)
    previous_outer = torch.zeros_like(ice)
    bands = []
    for radius in radii:
        if radius <= 0:
            raise ValueError("edge radius must be positive")
        kernel = 2 * int(radius) + 1
        near_ice = F.max_pool2d(ice, kernel, stride=1, padding=radius)
        near_open_water = F.max_pool2d(
            open_water, kernel, stride=1, padding=radius
        )
        # Land is neither ice nor open water.  In particular, a coast cannot
        # manufacture an ice/open-water boundary in an all-ice water domain.
        inner = ice * near_open_water
        outer = open_water * near_ice
        bands.extend(((inner - previous_inner).clamp(0.0, 1.0),
                      (outer - previous_outer).clamp(0.0, 1.0)))
        previous_inner, previous_outer = inner, outer
    return torch.stack(bands, dim=1).squeeze(2)


def _edge_profiles(
    area_product: torch.Tensor,
    bands: torch.Tensor,
    cell_area: torch.Tensor,
) -> torch.Tensor:
    """Weighted SIC/product profiles in condition-defined edge bands."""

    if area_product.ndim not in (4, 5) or area_product.shape[-3] != CHANNELS:
        raise ValueError("area_product must contain six interleaved channels")
    batch = area_product.shape[0]
    base_area = cell_area.expand(batch, -1, -1, -1)
    band_weight = bands[:, None] * base_area[:, :, None]
    denominator = band_weight.sum(dim=(-2, -1)).clamp_min(
        torch.finfo(area_product.dtype).tiny
    )
    if area_product.ndim == 5:
        weighted = area_product[:, :, :, None] * band_weight[:, None]
        return weighted.sum(dim=(-2, -1)) / denominator[:, None]
    weighted = area_product[:, :, None] * band_weight
    return weighted.sum(dim=(-2, -1)) / denominator


def _normalize_group(value: torch.Tensor, scale: torch.Tensor, weight: float) -> torch.Tensor:
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError("observable group weight must be finite and positive")
    while scale.ndim < value.ndim:
        scale = scale.unsqueeze(0)
    normalized = value / scale
    dimension = normalized.shape[-1]
    if dimension <= 0:
        raise ValueError("observable group is empty")
    return normalized * math.sqrt(weight / dimension)


def geometry_observable_vectors(
    members: torch.Tensor,
    truth: torch.Tensor,
    valid: torch.Tensor,
    initial_sic: torch.Tensor,
    cell_area: torch.Tensor,
    *,
    sic_scale: float,
    sit_scale: float,
    product_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, tuple[int, int]]]:
    """Map native trajectories to one balanced joint geometry vector."""

    _validate_inputs(members, truth, valid, initial_sic, cell_area)
    for name, value in {
        "sic_scale": sic_scale,
        "sit_scale": sit_scale,
        "product_scale": product_scale,
    }.items():
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    safe_members, safe_truth = _safe_physical(members, truth, valid)
    field_scale = members.new_tensor((sic_scale, sit_scale) * LEADS)
    increment_scale = members.new_tensor((sic_scale, sit_scale) * (LEADS - 1))

    d160_members = _area_weighted_block_mean(
        safe_members, valid, cell_area, factor=2
    ).flatten(start_dim=2)
    d160_truth = _area_weighted_block_mean(
        safe_truth, valid, cell_area, factor=2
    ).flatten(start_dim=1)
    d160_scale = field_scale.repeat_interleave(d160_members.shape[-1] // CHANNELS)

    d80_member_increment = _trajectory_increments(
        _area_weighted_block_mean(safe_members, valid, cell_area, factor=4)
    ).flatten(start_dim=2)
    d80_truth_increment = _trajectory_increments(
        _area_weighted_block_mean(safe_truth, valid, cell_area, factor=4)
    ).flatten(start_dim=1)
    d80_scale = increment_scale.repeat_interleave(
        d80_member_increment.shape[-1] // (CHANNELS - FIELDS)
    )

    def area_product(value: torch.Tensor) -> torch.Tensor:
        sic = value[..., 0::2, :, :]
        sit = value[..., 1::2, :, :]
        return torch.stack(
            (sic[..., 0, :, :], sic[..., 0, :, :] * sit[..., 0, :, :],
             sic[..., 1, :, :], sic[..., 1, :, :] * sit[..., 1, :, :],
             sic[..., 2, :, :], sic[..., 2, :, :] * sit[..., 2, :, :]),
            dim=-3,
        )

    member_area_product = area_product(safe_members)
    truth_area_product = area_product(safe_truth)
    regional_members = _fixed_region_means(
        member_area_product, valid, cell_area
    ).flatten(start_dim=2)
    regional_truth = _fixed_region_means(
        truth_area_product, valid, cell_area
    ).flatten(start_dim=1)
    budget_scale = members.new_tensor((sic_scale, product_scale) * LEADS).repeat_interleave(4)

    bands = _edge_band_masks(initial_sic, valid)
    edge_members = _edge_profiles(member_area_product, bands, cell_area).flatten(start_dim=2)
    edge_truth = _edge_profiles(truth_area_product, bands, cell_area).flatten(start_dim=1)
    edge_scale = members.new_tensor((sic_scale, product_scale) * LEADS).repeat_interleave(
        bands.shape[1]
    )

    groups = (
        ("d160_trajectory", d160_members, d160_truth, d160_scale),
        ("d80_increments", d80_member_increment, d80_truth_increment, d80_scale),
        ("regional_area_product", regional_members, regional_truth, budget_scale),
        ("initial_edge_profiles", edge_members, edge_truth, edge_scale),
    )
    member_vectors, truth_vectors = [], []
    slices: dict[str, tuple[int, int]] = {}
    offset = 0
    for name, member_group, truth_group, scale in groups:
        weight = GROUP_WEIGHTS[name]
        member_group = _normalize_group(member_group, scale, weight)
        truth_group = _normalize_group(truth_group, scale, weight)
        member_vectors.append(member_group)
        truth_vectors.append(truth_group)
        slices[name] = (offset, offset + member_group.shape[-1])
        offset += member_group.shape[-1]
    member_vector = torch.cat(member_vectors, dim=-1)
    truth_vector = torch.cat(truth_vectors, dim=-1)
    if not torch.all(torch.isfinite(member_vector)) or not torch.all(torch.isfinite(truth_vector)):
        raise FloatingPointError("geometry observable vector is non-finite")
    return member_vector, truth_vector, slices


def unbiased_energy_score_vectors(
    members: torch.Tensor, truth: torch.Tensor
) -> torch.Tensor:
    """Case-equal U-statistic ES using every unordered member pair."""

    if members.ndim != 3 or truth.shape != members.shape[:1] + members.shape[2:]:
        raise ValueError("vectors must have shapes [B,M,D] and [B,D]")
    count = members.shape[1]
    if count < 2:
        raise ValueError("unbiased energy score requires M>=2")
    if not torch.all(torch.isfinite(members)) or not torch.all(torch.isfinite(truth)):
        raise FloatingPointError("energy-score vectors contain NaN/Inf")
    observation = torch.linalg.vector_norm(members - truth[:, None], dim=-1).mean(dim=1)
    pair_sum = members.new_zeros(members.shape[0])
    for left in range(count):
        for right in range(left + 1, count):
            pair_sum = pair_sum + torch.linalg.vector_norm(
                members[:, left] - members[:, right], dim=-1
            )
    result = (observation - pair_sum / (count * (count - 1))).mean()
    if not torch.isfinite(result):
        raise FloatingPointError("geometry energy score is non-finite")
    return result


def geometry_energy_score(
    members: torch.Tensor,
    truth: torch.Tensor,
    valid: torch.Tensor,
    initial_sic: torch.Tensor,
    cell_area: torch.Tensor,
    *,
    sic_scale: float,
    sit_scale: float,
    product_scale: float,
) -> tuple[torch.Tensor, dict[str, tuple[int, int]]]:
    member_vector, truth_vector, slices = geometry_observable_vectors(
        members,
        truth,
        valid,
        initial_sic,
        cell_area,
        sic_scale=sic_scale,
        sit_scale=sit_scale,
        product_scale=product_scale,
    )
    return unbiased_energy_score_vectors(member_vector, truth_vector), slices
