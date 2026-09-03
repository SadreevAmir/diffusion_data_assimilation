"""Pure primitives for provenance-aware, bounded sea-ice modelling.

The functions in this module deliberately contain no dataset I/O or experiment
launcher.  They are small enough to audit independently before a trusted
training mode is admitted.
"""

from __future__ import annotations

import torch


def make_lagged_observation_channels(
    background: torch.Tensor,
    values: torch.Tensor,
    masks: torch.Tensor,
    ages_days: torch.Tensor,
    provenance: torch.Tensor,
) -> torch.Tensor:
    """Return separate innovation/value/mask/age/real/synthetic channels per lag.

    ``background`` is ``[B,1,H,W]``; values and masks are ``[B,L,H,W]``.
    ``ages_days`` and ``provenance`` are ``[B,L]``. Provenance is exactly zero
    for real observations and one for synthetic observations. Missing pixels
    are zero in every value-bearing or metadata channel.
    """
    if background.ndim != 4 or background.shape[1] != 1:
        raise ValueError("background must have shape [B,1,H,W]")
    if values.shape != masks.shape or values.ndim != 4:
        raise ValueError("values and masks must have identical [B,L,H,W] shape")
    if values.shape[0] != background.shape[0] or values.shape[2:] != background.shape[2:]:
        raise ValueError("observation and background batch/spatial shapes differ")
    expected_metadata = values.shape[:2]
    if ages_days.shape != expected_metadata or provenance.shape != expected_metadata:
        raise ValueError("ages_days and provenance must have shape [B,L]")
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
    innovation = (values - background) * mask
    observed_value = values * mask
    # Lag-major layout keeps all six meanings adjacent for each observation age.
    return torch.stack((innovation, observed_value, mask, age, real, synthetic), dim=2).flatten(1, 2)


def encode_zero_inflated_sic(
    concentration: torch.Tensor,
    *,
    occurrence_threshold: float = 0.15,
    epsilon: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode bounded SIC as occurrence and conditional-intensity logits.

    The transform is deterministic and does not clip a generated concentration:
    bounds are imposed by the representation.  The fixed epsilon is numerical,
    not a tunable scientific parameter.
    """
    if not 0.0 < occurrence_threshold < 1.0:
        raise ValueError("occurrence_threshold must lie strictly within (0,1)")
    if not 0.0 < epsilon < 0.5:
        raise ValueError("epsilon must lie strictly within (0,0.5)")
    if not torch.all(torch.isfinite(concentration)):
        raise ValueError("concentration must be finite")
    if not torch.all((concentration >= 0) & (concentration <= 1)):
        raise ValueError("concentration must lie in [0,1]")

    occurrence = concentration > occurrence_threshold
    occurrence_probability = torch.where(
        occurrence,
        torch.full_like(concentration, 1.0 - epsilon),
        torch.full_like(concentration, epsilon),
    )
    scaled = ((concentration - occurrence_threshold) / (1.0 - occurrence_threshold)).clamp(
        epsilon, 1.0 - epsilon
    )
    occurrence_logit = torch.logit(occurrence_probability)
    intensity_logit = torch.logit(scaled)
    return occurrence_logit, intensity_logit


def decode_zero_inflated_sic(
    occurrence_logit: torch.Tensor,
    intensity_logit: torch.Tensor,
    *,
    occurrence_threshold: float = 0.15,
) -> torch.Tensor:
    """Decode one member into exact zero or bounded positive concentration."""
    if occurrence_logit.shape != intensity_logit.shape:
        raise ValueError("occurrence and intensity logits must have identical shapes")
    if not torch.all(torch.isfinite(occurrence_logit)) or not torch.all(torch.isfinite(intensity_logit)):
        raise ValueError("logits must be finite")
    present = occurrence_logit > 0
    positive = occurrence_threshold + (1.0 - occurrence_threshold) * torch.sigmoid(intensity_logit)
    return torch.where(present, positive, torch.zeros_like(positive))
