"""Proper threshold-weighted score for SIC/SIT boundary-sensitive refinement.

The transformation is strictly increasing: its derivative is one everywhere
and two inside the frozen physical boundary bands.  Applying ordinary CRPS in
this coordinate is therefore a transformed kernel score, not a post-processing
rule and not a hard-indicator surrogate.
"""

from __future__ import annotations

from typing import Final

import torch

from .direct_dynamics_cascade_coarse_proper_refinement import (
    standardized_fair_crps,
    standardized_joint_energy,
)


BOUNDARY_WEIGHT_MULTIPLIER: Final[float] = 2.0
SIC_BOUNDARY_BANDS: Final[tuple[tuple[float, float], ...]] = (
    (0.0, 0.02),
    (0.98, 1.0),
)
SIT_BOUNDARY_BANDS_METRES: Final[tuple[tuple[float, float], ...]] = ((0.0, 0.02),)


def _validate_channel_stats(
    values: torch.Tensor,
    channel_means: torch.Tensor,
    channel_stds: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if values.ndim < 3 or values.shape[-3] != 6:
        raise ValueError("threshold transform requires six SIC/SIT trajectory channels")
    means = torch.as_tensor(channel_means, device=values.device, dtype=values.dtype)
    stds = torch.as_tensor(channel_stds, device=values.device, dtype=values.dtype)
    if means.numel() != 6 or stds.numel() != 6:
        raise ValueError("threshold transform requires six channel means and stds")
    if not torch.isfinite(means).all() or not torch.isfinite(stds).all() or torch.any(stds <= 0):
        raise ValueError("threshold transform requires finite means and positive stds")
    shape = (1,) * (values.ndim - 3) + (6, 1, 1)
    return means.reshape(shape), stds.reshape(shape)


def boundary_emphasis_transform(
    values: torch.Tensor,
    channel_means: torch.Tensor,
    channel_stds: torch.Tensor,
) -> torch.Tensor:
    """Map normalized values through the frozen positive threshold weight.

    For normalized coordinate ``x`` and a physical band ``[a,b]``, the added
    term is the integral of an extra unit weight across that band.  Thus the
    full derivative is 2 inside a band and 1 outside, including outside the
    nominal physical support.
    """
    means, stds = _validate_channel_stats(values, channel_means, channel_stds)
    transformed = values
    extra_weight = BOUNDARY_WEIGHT_MULTIPLIER - 1.0
    for channel in range(6):
        bands = SIC_BOUNDARY_BANDS if channel % 2 == 0 else SIT_BOUNDARY_BANDS_METRES
        selected = values[..., channel : channel + 1, :, :]
        mean = means[..., channel : channel + 1, :, :]
        std = stds[..., channel : channel + 1, :, :]
        added = torch.zeros_like(selected)
        for lower_physical, upper_physical in bands:
            lower = (lower_physical - mean) / std
            upper = (upper_physical - mean) / std
            added = added + torch.clamp(selected - lower, min=0.0) - torch.clamp(
                selected - upper, min=0.0
            )
        update = selected + extra_weight * added
        transformed = torch.cat(
            (
                transformed[..., :channel, :, :],
                update,
                transformed[..., channel + 1 :, :, :],
            ),
            dim=-3,
        )
    return transformed


def threshold_weighted_fair_crps(
    members: torch.Tensor,
    truth: torch.Tensor,
    ocean_fraction: torch.Tensor,
    channel_means: torch.Tensor,
    channel_stds: torch.Tensor,
) -> torch.Tensor:
    return standardized_fair_crps(
        boundary_emphasis_transform(members, channel_means, channel_stds),
        boundary_emphasis_transform(truth, channel_means, channel_stds),
        ocean_fraction,
    )


def threshold_weighted_proper_objective(
    members: torch.Tensor,
    truth: torch.Tensor,
    ocean_fraction: torch.Tensor,
    channel_means: torch.Tensor,
    channel_stds: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    crps = threshold_weighted_fair_crps(
        members, truth, ocean_fraction, channel_means, channel_stds
    )
    energy = standardized_joint_energy(members, truth, ocean_fraction)
    return 0.75 * crps + 0.25 * energy, crps, energy
