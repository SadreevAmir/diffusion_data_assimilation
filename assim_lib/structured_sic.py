"""Pure primitives for provenance-aware, bounded sea-ice modelling.

The functions in this module deliberately contain no dataset I/O or experiment
launcher.  They are small enough to audit independently before a trusted
training mode is admitted.
"""

from __future__ import annotations

import torch


def make_lagged_observation_channels(
    background_trajectory: torch.Tensor,
    values: torch.Tensor,
    masks: torch.Tensor,
    ages_days: torch.Tensor,
    provenance: torch.Tensor,
) -> torch.Tensor:
    """Return separate innovation/value/mask/age/real/synthetic channels per lag.

    ``background_trajectory``, values and masks are ``[B,L,H,W]``.  Each
    innovation at observation day ``t-k`` is formed against the matching
    background frame ``b_{t-k}``, never against the current frame broadcast
    over all lags.
    ``ages_days`` and ``provenance`` are ``[B,L]``. Provenance is exactly zero
    for real observations and one for synthetic observations. Missing pixels
    are zero in every value-bearing or metadata channel.
    """
    if values.shape != masks.shape or values.ndim != 4:
        raise ValueError("values and masks must have identical [B,L,H,W] shape")
    if background_trajectory.shape != values.shape:
        raise ValueError("background_trajectory must have the same [B,L,H,W] shape as values")
    expected_metadata = values.shape[:2]
    if ages_days.shape != expected_metadata or provenance.shape != expected_metadata:
        raise ValueError("ages_days and provenance must have shape [B,L]")
    if not torch.all(torch.isfinite(background_trajectory)) or not torch.all(torch.isfinite(values)):
        raise ValueError("background_trajectory and values must be finite")
    if not torch.all(torch.isfinite(masks)) or not torch.all(torch.isfinite(ages_days)):
        raise ValueError("masks and ages_days must be finite")
    if not torch.all(torch.isfinite(provenance)):
        raise ValueError("provenance must be finite")
    if not torch.all((masks == 0) | (masks == 1)):
        raise ValueError("masks must be binary")
    if not torch.all(ages_days >= 0):
        raise ValueError("ages_days cannot be negative")
    if not torch.all((provenance == 0) | (provenance == 1)):
        raise ValueError("provenance must be binary: real=0, synthetic=1")

    mask = masks.to(dtype=values.dtype)
    age = ages_days.to(dtype=values.dtype)[..., None, None] * mask
    synthetic = provenance.to(dtype=values.dtype)[..., None, None] * mask
    real = (1.0 - provenance.to(dtype=values.dtype))[..., None, None] * mask
    innovation = (values - background_trajectory) * mask
    observed_value = values * mask
    # Lag-major layout keeps all six meanings adjacent for each observation age.
    return torch.stack((innovation, observed_value, mask, age, real, synthetic), dim=2).flatten(1, 2)


def encode_zero_inflated_sic(
    concentration: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode bounded SIC as an exact occurrence atom and positive intensity.

    Occurrence is the physical atom ``1{SIC > 0}``.  The second coordinate is
    the concentration itself on the positive branch and zero at the atom.  This
    primitive is injective on ``[0,1]`` and uses neither epsilon clipping nor a
    finite-logit surrogate for exact boundary values.
    """
    if not torch.all(torch.isfinite(concentration)):
        raise ValueError("concentration must be finite")
    if not torch.all((concentration >= 0) & (concentration <= 1)):
        raise ValueError("concentration must lie in [0,1]")

    occurrence_atom = (concentration > 0).to(dtype=concentration.dtype)
    positive_intensity = torch.where(
        occurrence_atom.bool(), concentration, torch.zeros_like(concentration)
    )
    return occurrence_atom, positive_intensity


def decode_zero_inflated_sic(
    occurrence_atom: torch.Tensor,
    positive_intensity: torch.Tensor,
) -> torch.Tensor:
    """Decode the exact atom/intensity representation without clipping."""
    if occurrence_atom.shape != positive_intensity.shape:
        raise ValueError("occurrence_atom and positive_intensity must have identical shapes")
    if not torch.all(torch.isfinite(occurrence_atom)) or not torch.all(torch.isfinite(positive_intensity)):
        raise ValueError("encoded values must be finite")
    if not torch.all((occurrence_atom == 0) | (occurrence_atom == 1)):
        raise ValueError("occurrence_atom must be binary")
    if not torch.all((positive_intensity >= 0) & (positive_intensity <= 1)):
        raise ValueError("positive_intensity must lie in [0,1]")
    if not torch.all((occurrence_atom == 1) | (positive_intensity == 0)):
        raise ValueError("absent atoms require zero intensity")
    if not torch.all((occurrence_atom == 0) | (positive_intensity > 0)):
        raise ValueError("present atoms require strictly positive intensity")
    return torch.where(occurrence_atom.bool(), positive_intensity, torch.zeros_like(positive_intensity))
