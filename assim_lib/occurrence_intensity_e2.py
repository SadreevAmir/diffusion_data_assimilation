"""Fail-closed primitives for the predeclared E2 temporal mechanism.

This module is a publication-side correctness oracle, not an experiment runner.
It makes the time index of every observation operator explicit so a future
trusted implementation cannot silently evaluate all observations at the
current generated state.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch


ObservationOperator = Callable[[torch.Tensor], torch.Tensor]


def apply_operators_at_own_time(
    state_trajectory: torch.Tensor,
    observations: torch.Tensor,
    masks: torch.Tensor,
    operators: Sequence[ObservationOperator],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply operator ``k`` only to generated state ``k`` under a finite mask.

    The ordered trajectory is ``[b_t, b_{t-1}, b_{t-2}]`` with shape
    ``[B,3,C,H,W]``. Observations and masks have shape ``[B,3,H,W]``.
    Missing observations may be non-finite only outside their masks. Returned
    predictions and innovations are finite and exactly zero outside each mask.
    """
    if state_trajectory.ndim != 5 or state_trajectory.shape[1] != 3:
        raise ValueError("state_trajectory must have shape [B,3,C,H,W]")
    expected = (state_trajectory.shape[0], 3, *state_trajectory.shape[-2:])
    if observations.shape != expected or masks.shape != expected:
        raise ValueError("observations and masks must have shape [B,3,H,W]")
    if len(operators) != 3:
        raise ValueError("E2 requires exactly three time-specific operators")
    if not torch.all(torch.isfinite(state_trajectory)):
        raise ValueError("state_trajectory must be finite")
    if not torch.all(torch.isfinite(masks)) or not torch.all((masks == 0) | (masks == 1)):
        raise ValueError("masks must be finite and binary")

    observed = masks.bool()
    if not torch.all(torch.isfinite(observations[observed])):
        raise ValueError("observations must be finite on observed pixels")

    predictions = []
    innovations = []
    for lag, operator in enumerate(operators):
        prediction = operator(state_trajectory[:, lag])
        if prediction.shape != observations[:, lag].shape:
            raise ValueError(f"operator {lag} returned an invalid shape")
        if not torch.all(torch.isfinite(prediction)):
            raise ValueError(f"operator {lag} returned non-finite values")
        zero = torch.zeros_like(prediction)
        masked_prediction = torch.where(observed[:, lag], prediction, zero)
        masked_observation = torch.where(observed[:, lag], observations[:, lag], zero)
        predictions.append(masked_prediction)
        innovations.append(masked_observation - masked_prediction)
    return torch.stack(predictions, dim=1), torch.stack(innovations, dim=1)
