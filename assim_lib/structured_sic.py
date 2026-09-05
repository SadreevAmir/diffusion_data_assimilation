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
    geometry_provenance: torch.Tensor,
    value_provenance: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return separate innovation/value/mask/age/real/synthetic channels per lag.

    ``background_trajectory``, values and masks are ``[B,L,H,W]``.  Each
    innovation at observation day ``t-k`` is formed against the matching
    background frame ``b_{t-k}``, never against the current frame broadcast
    over all lags.
    ``ages_days`` and ``geometry_provenance`` are ``[B,L]``.  Value provenance
    is ``[B,L,H,W]`` and may differ from footprint provenance pixel by pixel.
    Zero means real and one means synthetic.  Missing values may be NaN; only
    observed pixels must be finite and all emitted channels are finite.
    """
    if values.shape != masks.shape or values.ndim != 4:
        raise ValueError("values and masks must have identical [B,L,H,W] shape")
    if background_trajectory.shape != values.shape:
        raise ValueError("background_trajectory must have the same [B,L,H,W] shape as values")
    if values.shape[1] != 3:
        raise ValueError("the corrected contract requires exactly three lags")
    expected_metadata = values.shape[:2]
    if ages_days.shape != expected_metadata or geometry_provenance.shape != expected_metadata:
        raise ValueError("ages_days and geometry_provenance must have shape [B,L]")
    if value_provenance is None:
        value_provenance = geometry_provenance[..., None, None].expand_as(values)
    if value_provenance.shape != values.shape:
        raise ValueError("value_provenance must have shape [B,L,H,W]")
    if not torch.all(torch.isfinite(masks)) or not torch.all(torch.isfinite(ages_days)):
        raise ValueError("masks and ages_days must be finite")
    if not torch.all(torch.isfinite(geometry_provenance)):
        raise ValueError("geometry_provenance must be finite")
    if not torch.all((masks == 0) | (masks == 1)):
        raise ValueError("masks must be binary")
    if not torch.all(ages_days >= 0):
        raise ValueError("ages_days cannot be negative")
    expected_ages = torch.arange(3, device=ages_days.device, dtype=ages_days.dtype).expand_as(ages_days)
    if not torch.equal(ages_days, expected_ages):
        raise ValueError("ages_days must be exactly [0,1,2] for every batch item")
    if not torch.all((geometry_provenance == 0) | (geometry_provenance == 1)):
        raise ValueError("geometry_provenance must be binary: real=0, synthetic=1")

    observed = masks.bool()
    for name, tensor in (("background_trajectory", background_trajectory), ("values", values),
                         ("value_provenance", value_provenance)):
        if not torch.all(torch.isfinite(tensor[observed])):
            raise ValueError(f"{name} must be finite on observed pixels")
    if not torch.all((value_provenance[observed] == 0) | (value_provenance[observed] == 1)):
        raise ValueError("value_provenance must be binary on observed pixels")

    mask = masks.to(dtype=values.dtype)
    age = ages_days.to(dtype=values.dtype)[..., None, None] * mask
    zero = torch.zeros_like(values)
    clean_values = torch.where(observed, values, zero)
    clean_background = torch.where(observed, background_trajectory, zero)
    value_synthetic = torch.where(observed, value_provenance.to(values.dtype), zero)
    geometry_synthetic = geometry_provenance.to(values.dtype)[..., None, None] * mask
    geometry_real = mask - geometry_synthetic
    value_real = mask - value_synthetic
    innovation = torch.where(observed, clean_values - clean_background, zero)
    # Eight lag-major channels keep geometry and value lineage distinguishable.
    return torch.stack(
        (innovation, clean_values, mask, age, geometry_real, geometry_synthetic,
         value_real, value_synthetic), dim=2
    ).flatten(1, 2)


def continuous_dequantize_occurrence(
    occurrence_atom: torch.Tensor, uniform: torch.Tensor
) -> torch.Tensor:
    """Map a Bernoulli atom to disjoint continuous half-intervals for CFM.

    Given ``U ~ Uniform(0,1)``, ``Z=(A+U)/2`` has density two on ``[0,.5)``
    for exact zero and on ``[.5,1)`` for positive SIC.  Thresholding at .5
    therefore recovers the physical atom almost surely.
    """
    if occurrence_atom.shape != uniform.shape:
        raise ValueError("occurrence_atom and uniform must have identical shapes")
    if not torch.all(torch.isfinite(uniform)) or not torch.all((uniform >= 0) & (uniform < 1)):
        raise ValueError("uniform must be finite and lie in [0,1)")
    if not torch.all((occurrence_atom == 0) | (occurrence_atom == 1)):
        raise ValueError("occurrence_atom must be binary")
    return 0.5 * (occurrence_atom.to(uniform.dtype) + uniform)


def make_cfm_targets(
    concentration: torch.Tensor,
    occurrence_uniform: torch.Tensor,
    zero_intensity_uniform: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return complete occurrence/intensity CFM targets.

    Positive intensity equals SIC.  At the zero atom it is an independent
    ``Uniform(0,1)`` auxiliary variable, so the joint target has no singular
    constant intensity coordinate.  Decoding ignores this auxiliary coordinate
    whenever the dequantized occurrence lies below .5.
    """
    occurrence, intensity = encode_zero_inflated_sic(concentration)
    if zero_intensity_uniform.shape != concentration.shape:
        raise ValueError("zero_intensity_uniform must match concentration")
    if not torch.all(torch.isfinite(zero_intensity_uniform)) or not torch.all(
        (zero_intensity_uniform >= 0) & (zero_intensity_uniform <= 1)
    ):
        raise ValueError("zero_intensity_uniform must be finite and lie in [0,1]")
    occurrence_target = continuous_dequantize_occurrence(occurrence, occurrence_uniform)
    intensity_target = torch.where(occurrence.bool(), intensity, zero_intensity_uniform)
    return occurrence_target, intensity_target


def decode_cfm_targets(
    occurrence_target: torch.Tensor, intensity_target: torch.Tensor
) -> torch.Tensor:
    """Decode generated continuous coordinates without output clipping."""
    if occurrence_target.shape != intensity_target.shape:
        raise ValueError("CFM coordinates must have identical shapes")
    if not torch.all(torch.isfinite(occurrence_target)) or not torch.all(torch.isfinite(intensity_target)):
        raise ValueError("CFM coordinates must be finite")
    if not torch.all((occurrence_target >= 0) & (occurrence_target <= 1)):
        raise ValueError("occurrence_target must lie in [0,1]")
    if not torch.all((intensity_target >= 0) & (intensity_target <= 1)):
        raise ValueError("intensity_target must lie in [0,1]")
    return torch.where(occurrence_target >= 0.5, intensity_target, torch.zeros_like(intensity_target))


def exact_one_policy(training_concentration: torch.Tensor) -> str:
    """Freeze the upper-boundary policy from training inventory only."""
    if not torch.all(torch.isfinite(training_concentration)):
        raise ValueError("training concentration must be finite")
    if not torch.all((training_concentration >= 0) & (training_concentration <= 1)):
        raise ValueError("training concentration must lie in [0,1]")
    return "explicit_exact_one_atom" if torch.any(training_concentration == 1) else "no_exact_one_atom"


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
