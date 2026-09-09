"""Proper global-plus-local energy score for censored joint sea-ice fields."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch


PATCH_SIZES = (8, 16)
PATCHES_PER_CONDITION = 8
GLOBAL_WEIGHT = 0.5
PATCH_SCALE_WEIGHTS = (0.25, 0.25)


def _validate_shapes(
    members: torch.Tensor,
    truth: torch.Tensor,
    valid: torch.Tensor,
    stds: Sequence[float],
) -> None:
    if members.ndim != 5:
        raise ValueError("members must have shape [condition, member, channel, y, x]")
    if truth.shape != (members.shape[0], *members.shape[2:]):
        raise ValueError("truth shape does not match members")
    if valid.shape != (members.shape[0], 1, *members.shape[-2:]):
        raise ValueError("valid mask shape does not match members")
    if members.shape[1] < 2:
        raise ValueError("unbiased energy score requires at least two members")
    if len(stds) != members.shape[2] or any(
        not math.isfinite(value) or value <= 0 for value in stds
    ):
        raise ValueError("one finite positive train-derived std is required per channel")
    if not torch.all((valid == 0) | (valid == 1)):
        raise ValueError("valid mask must be exactly binary")


def _active_inputs_are_finite(
    members: torch.Tensor,
    truth: torch.Tensor,
    valid: torch.Tensor,
) -> bool:
    active = valid.bool()
    active_members = active[:, None].expand_as(members)
    active_truth = active.expand_as(truth)
    return bool(
        torch.all(torch.isfinite(members[active_members]))
        and torch.all(torch.isfinite(truth[active_truth]))
    )


def _energy_score_core(
    members: torch.Tensor,
    truth: torch.Tensor,
    valid: torch.Tensor,
    stds: Sequence[float],
) -> torch.Tensor:
    """Energy-score arithmetic after a single outer validation/finite gate."""

    batch, member_count, channels = members.shape[:3]
    active = valid.bool()
    active_members = active[:, None].expand_as(members)
    active_truth = active.expand_as(truth)
    safe_members = torch.where(active_members, members, torch.zeros_like(members))
    safe_truth = torch.where(active_truth, truth, torch.zeros_like(truth))
    scale = members.new_tensor(stds).view(1, 1, channels, 1, 1)
    mask = active.to(dtype=members.dtype).unsqueeze(1)
    valid_counts = valid.sum(dim=(-2, -1)).reshape(batch)
    normalizer = torch.sqrt(valid_counts.to(members.dtype) * channels)

    observation_difference = ((safe_members - safe_truth[:, None]) / scale) * mask
    observation_distance = torch.linalg.vector_norm(
        observation_difference.flatten(start_dim=2), dim=2
    ) / normalizer[:, None]

    pair_difference = (
        (safe_members[:, :, None] - safe_members[:, None, :])
        / scale.unsqueeze(2)
    ) * mask.unsqueeze(2)
    pair_distance = torch.linalg.vector_norm(
        pair_difference.flatten(start_dim=3), dim=3
    ) / normalizer[:, None, None]
    case_score = observation_distance.mean(dim=1) - pair_distance.sum(
        dim=(1, 2)
    ) / (2.0 * member_count * (member_count - 1))
    return case_score.mean()


def _energy_score_with_mask(
    members: torch.Tensor,
    truth: torch.Tensor,
    valid: torch.Tensor,
    stds: Sequence[float],
) -> torch.Tensor:
    """Case-equal unbiased ES after physical censoring and std normalization."""

    _validate_shapes(members, truth, valid, stds)
    if torch.any(valid.sum(dim=(-2, -1)) <= 0):
        raise ValueError("every scored region must contain a valid pixel")
    active = valid.bool()
    active_members = active[:, None].expand_as(members)
    active_truth = active.expand_as(truth)
    if not torch.all(torch.isfinite(members[active_members])):
        raise FloatingPointError("active members contain NaN/Inf")
    if not torch.all(torch.isfinite(truth[active_truth])):
        raise FloatingPointError("active truth contains NaN/Inf")
    result = _energy_score_core(members, truth, valid, stds)
    if not torch.isfinite(result):
        raise FloatingPointError("energy-score result is NaN/Inf")
    return result


def sample_valid_centres(
    valid: torch.Tensor,
    count: int,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample valid-ocean centres with replacement, independently per condition."""

    if valid.ndim != 4 or valid.shape[1] != 1:
        raise ValueError("valid must have shape [condition, 1, y, x]")
    if count <= 0:
        raise ValueError("centre count must be positive")
    if not torch.all((valid == 0) | (valid == 1)):
        raise ValueError("valid mask must be exactly binary")
    centres = []
    for case in range(valid.shape[0]):
        coordinates = torch.nonzero(valid[case, 0] > 0, as_tuple=False)
        if coordinates.numel() == 0:
            raise ValueError("cannot sample a centre from an empty valid mask")
        choices = torch.randint(
            coordinates.shape[0],
            (count,),
            device=valid.device,
            generator=generator,
        )
        centres.append(coordinates[choices])
    return torch.stack(centres)


def patch_energy_score(
    members: torch.Tensor,
    truth: torch.Tensor,
    valid: torch.Tensor,
    stds: Sequence[float],
    centres: torch.Tensor,
    patch_size: int,
) -> torch.Tensor:
    """Mean proper ES over already-generated full-field patches."""

    _validate_shapes(members, truth, valid, stds)
    if not _active_inputs_are_finite(members, truth, valid):
        raise FloatingPointError("active patch-score inputs contain NaN/Inf")
    if centres.ndim != 3 or centres.shape[0] != members.shape[0] or centres.shape[2] != 2:
        raise ValueError("centres must have shape [condition, patch, (y, x)]")
    if centres.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise ValueError("patch centres must be integer coordinates")
    if centres.device != members.device:
        raise ValueError("patch centres and scored fields must share a device")
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    height, width = members.shape[-2:]
    in_bounds = (
        (centres[..., 0] >= 0)
        & (centres[..., 0] < height)
        & (centres[..., 1] >= 0)
        & (centres[..., 1] < width)
    )
    if not torch.all(in_bounds):
        raise ValueError("patch centre lies outside the image")
    case_indices = torch.arange(members.shape[0], device=centres.device)[:, None]
    centre_valid = valid[
        case_indices, 0, centres[..., 0].long(), centres[..., 1].long()
    ]
    if not torch.all(centre_valid == 1):
        raise ValueError("every patch centre must be a valid-ocean pixel")
    scores = []
    half = patch_size // 2
    for case in range(members.shape[0]):
        for centre_y, centre_x in centres[case].tolist():
            top = centre_y - half
            left = centre_x - half
            bottom = top + patch_size
            right = left + patch_size
            y0, y1 = max(0, top), min(height, bottom)
            x0, x1 = max(0, left), min(width, right)
            scores.append(
                _energy_score_core(
                    members[case : case + 1, :, :, y0:y1, x0:x1],
                    truth[case : case + 1, :, y0:y1, x0:x1],
                    valid[case : case + 1, :, y0:y1, x0:x1],
                    stds,
                )
            )
    result = torch.stack(scores).mean()
    if not torch.isfinite(result):
        raise FloatingPointError("patch energy-score result is NaN/Inf")
    return result


def multiscale_joint_energy_score(
    members: torch.Tensor,
    truth: torch.Tensor,
    valid: torch.Tensor,
    *,
    stds: Sequence[float],
    centres: torch.Tensor,
    expected_centres: int | None = PATCHES_PER_CONDITION,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Frozen 1/2 global + 1/4 8px patch + 1/4 16px patch objective."""

    if len(PATCH_SIZES) != len(PATCH_SCALE_WEIGHTS):
        raise RuntimeError("patch scales and weights differ")
    if not math.isclose(GLOBAL_WEIGHT + sum(PATCH_SCALE_WEIGHTS), 1.0):
        raise RuntimeError("multiscale objective weights must sum to one")
    if expected_centres is not None and centres.shape[1] != expected_centres:
        raise ValueError(
            f"expected {expected_centres} patch centres per condition"
        )
    components = {
        "global": _energy_score_with_mask(members, truth, valid, stds)
    }
    for patch_size in PATCH_SIZES:
        components[f"patch_{patch_size}"] = patch_energy_score(
            members, truth, valid, stds, centres, patch_size
        )
    total = GLOBAL_WEIGHT * components["global"]
    for weight, patch_size in zip(PATCH_SCALE_WEIGHTS, PATCH_SIZES, strict=True):
        total = total + weight * components[f"patch_{patch_size}"]
    if not torch.isfinite(total):
        raise FloatingPointError("multiscale energy-score result is NaN/Inf")
    return total, components
