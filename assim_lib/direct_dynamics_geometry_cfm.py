"""Law-preserving geometry preconditioning for direct-dynamics CFM.

The geometry term is a conditioned quadratic norm of the velocity error.  Every
operator below is linear in the error and may depend only on information already
present in the conditioning state (valid ocean, d0 SIC/SIT and fixed regions).
Consequently a positive geometry weight changes finite-capacity optimisation but
not the population conditional-mean velocity targeted by flow matching.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F


LEADS = 3
FIELDS = 2
CHANNELS = LEADS * FIELDS


@dataclass(frozen=True)
class GeometryCFMSpec:
    multiscale_factors: tuple[int, ...] = (4, 16)
    field_stds: tuple[float, float] = (1.0, 1.0)
    product_scale: float = 1.0
    multiscale_group_weight: float = 1.0
    lead_group_weight: float = 1.0
    product_group_weight: float = 1.0
    edge_group_weight: float = 1.0
    region_group_weight: float = 1.0

    def validate(self) -> None:
        if not self.multiscale_factors or any(int(value) <= 0 for value in self.multiscale_factors):
            raise ValueError("multiscale factors must be positive")
        finite_positive = (*self.field_stds, self.product_scale)
        if any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in finite_positive):
            raise ValueError("field and product scales must be finite and positive")
        group_weights = (
            self.multiscale_group_weight,
            self.lead_group_weight,
            self.product_group_weight,
            self.edge_group_weight,
            self.region_group_weight,
        )
        if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in group_weights):
            raise ValueError("geometry group weights must be finite and non-negative")


def _validate_inputs(
    error: torch.Tensor,
    valid_mask: torch.Tensor,
    initial_sic: torch.Tensor,
    initial_sit: torch.Tensor,
) -> None:
    if error.ndim != 4 or error.shape[1] != CHANNELS:
        raise ValueError(f"error must have shape [B,{CHANNELS},H,W]")
    expected = (error.shape[0], 1, *error.shape[-2:])
    for name, value in (
        ("valid_mask", valid_mask),
        ("initial_sic", initial_sic),
        ("initial_sit", initial_sit),
    ):
        if tuple(value.shape) != expected:
            raise ValueError(f"{name} must have shape {expected}, got {tuple(value.shape)}")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} contains non-finite values")
    if not bool(torch.isfinite(error).all()):
        raise ValueError("error contains non-finite values")
    if bool(torch.any(valid_mask < 0)) or bool(torch.any(valid_mask > 1)):
        raise ValueError("valid_mask must lie in [0,1]")
    if not bool(torch.all((valid_mask == 0) | (valid_mask == 1))):
        raise ValueError("valid_mask must be binary")
    if bool(torch.any(valid_mask.flatten(1).sum(dim=1) <= 0)):
        raise ValueError("every case must contain at least one valid ocean cell")


def _expanded_mask(error: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    return valid_mask.detach().to(dtype=error.dtype, device=error.device).expand_as(error)


def _validate_region_masks(
    error: torch.Tensor,
    valid_mask: torch.Tensor,
    region_masks: torch.Tensor | None,
) -> None:
    if region_masks is None:
        return
    expected = (error.shape[0], region_masks.shape[1], *error.shape[-2:])
    if region_masks.ndim != 4 or tuple(region_masks.shape) != expected:
        raise ValueError("region_masks must have shape [B,R,H,W]")
    if region_masks.shape[1] <= 0:
        raise ValueError("region_masks must contain at least one region")
    if not bool(torch.isfinite(region_masks).all()) or bool(torch.any(region_masks < 0)):
        raise ValueError("region_masks must be finite and non-negative")
    active = region_masks.detach() * valid_mask.detach()
    if bool(torch.any(active.sum(dim=(-2, -1)) <= 0)):
        raise ValueError("every region must overlap valid ocean in every case")


def masked_mean_pool(error: torch.Tensor, valid_mask: torch.Tensor, factor: int) -> torch.Tensor:
    """Blockwise valid-ocean mean, linear in ``error`` for a fixed mask."""
    if factor <= 0 or error.shape[-2] % factor or error.shape[-1] % factor:
        raise ValueError("pooling factor must divide both spatial dimensions")
    mask = _expanded_mask(error, valid_mask)
    numerator = F.avg_pool2d(error * mask, factor, stride=factor)
    denominator = F.avg_pool2d(mask, factor, stride=factor)
    return torch.where(denominator > 0, numerator / denominator.clamp_min(torch.finfo(error.dtype).tiny), 0)


def masked_mean_pool_adjoint(
    probe: torch.Tensor,
    valid_mask: torch.Tensor,
    factor: int,
    input_channels: int,
) -> torch.Tensor:
    """Euclidean adjoint of :func:`masked_mean_pool`."""
    height, width = valid_mask.shape[-2:]
    expected = (valid_mask.shape[0], input_channels, height // factor, width // factor)
    if tuple(probe.shape) != expected:
        raise ValueError(f"pool probe must have shape {expected}")
    mask = valid_mask.to(dtype=probe.dtype, device=probe.device).expand(-1, input_channels, -1, -1)
    counts = F.avg_pool2d(mask, factor, stride=factor) * float(factor * factor)
    scaled = torch.where(counts > 0, probe / counts.clamp_min(torch.finfo(probe.dtype).tiny), 0)
    lifted = scaled.repeat_interleave(factor, dim=-2).repeat_interleave(factor, dim=-1)
    return lifted * mask


def lead_difference(error: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    trajectory = error.reshape(error.shape[0], LEADS, FIELDS, *error.shape[-2:])
    mask = valid_mask.detach().to(dtype=error.dtype, device=error.device).unsqueeze(1)
    return (trajectory[:, 1:] - trajectory[:, :-1]) * mask


def lead_difference_adjoint(probe: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    expected = (valid_mask.shape[0], LEADS - 1, FIELDS, *valid_mask.shape[-2:])
    if tuple(probe.shape) != expected:
        raise ValueError(f"lead probe must have shape {expected}")
    result = probe.new_zeros((probe.shape[0], LEADS, FIELDS, *probe.shape[-2:]))
    result[:, 0] -= probe[:, 0]
    result[:, 1] += probe[:, 0] - probe[:, 1]
    result[:, 2] += probe[:, 1]
    result *= valid_mask.detach().to(dtype=probe.dtype, device=probe.device).unsqueeze(1)
    return result.reshape(probe.shape[0], CHANNELS, *probe.shape[-2:])


def linearized_volume_error(
    error: torch.Tensor,
    valid_mask: torch.Tensor,
    initial_sic: torch.Tensor,
    initial_sit: torch.Tensor,
    field_stds: tuple[float, float],
    product_scale: float,
) -> torch.Tensor:
    """d0 linearisation ``H0 e_A + A0 e_H``; no future-truth weights."""
    trajectory = error.reshape(error.shape[0], LEADS, FIELDS, *error.shape[-2:])
    sic0 = initial_sic.detach().to(dtype=error.dtype, device=error.device).unsqueeze(1)
    sit0 = initial_sit.detach().to(dtype=error.dtype, device=error.device).unsqueeze(1)
    mask = valid_mask.detach().to(dtype=error.dtype, device=error.device).unsqueeze(1)
    result = sit0 * float(field_stds[0]) * trajectory[:, :, 0:1]
    result = result + sic0 * float(field_stds[1]) * trajectory[:, :, 1:2]
    return (result.squeeze(2) / float(product_scale)) * mask.squeeze(2)


def linearized_volume_adjoint(
    probe: torch.Tensor,
    valid_mask: torch.Tensor,
    initial_sic: torch.Tensor,
    initial_sit: torch.Tensor,
    field_stds: tuple[float, float],
    product_scale: float,
) -> torch.Tensor:
    expected = (valid_mask.shape[0], LEADS, *valid_mask.shape[-2:])
    if tuple(probe.shape) != expected:
        raise ValueError(f"product probe must have shape {expected}")
    mask = valid_mask.detach().to(dtype=probe.dtype, device=probe.device).unsqueeze(1)
    sic0 = initial_sic.detach().to(dtype=probe.dtype, device=probe.device).unsqueeze(1)
    sit0 = initial_sit.detach().to(dtype=probe.dtype, device=probe.device).unsqueeze(1)
    active = probe.unsqueeze(2) * mask / float(product_scale)
    result = probe.new_empty((probe.shape[0], LEADS, FIELDS, *probe.shape[-2:]))
    result[:, :, 0:1] = active * sit0 * float(field_stds[0])
    result[:, :, 1:2] = active * sic0 * float(field_stds[1])
    return result.reshape(probe.shape[0], CHANNELS, *probe.shape[-2:])


def initial_edge_weight(initial_sic: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """A bounded d0-only ice-edge tube weight, detached from model gradients."""
    sic = initial_sic.detach()
    valid = valid_mask.detach().to(dtype=sic.dtype, device=sic.device)
    dy_pair = valid[..., 1:, :] * valid[..., :-1, :]
    dx_pair = valid[..., :, 1:] * valid[..., :, :-1]
    dy = F.pad((sic[..., 1:, :] - sic[..., :-1, :]).abs() * dy_pair, (0, 0, 0, 1))
    dx = F.pad((sic[..., :, 1:] - sic[..., :, :-1]).abs() * dx_pair, (0, 1, 0, 0))
    raw = (dx + dy) * valid
    numerator = F.avg_pool2d(raw, 5, stride=1, padding=2)
    denominator = F.avg_pool2d(valid, 5, stride=1, padding=2)
    smoothed = torch.where(
        denominator > 0,
        numerator / denominator.clamp_min(torch.finfo(sic.dtype).tiny),
        0,
    ) * valid
    maximum = smoothed.flatten(1).amax(dim=1).view(-1, 1, 1, 1).clamp_min(1e-12)
    return (smoothed / maximum).clamp(0.0, 1.0) * valid


def region_mean(error: torch.Tensor, valid_mask: torch.Tensor, region_masks: torch.Tensor) -> torch.Tensor:
    weights = region_masks.detach().to(dtype=error.dtype, device=error.device).unsqueeze(1)
    weights = weights * valid_mask.detach().to(dtype=error.dtype, device=error.device).unsqueeze(2)
    denominator = weights.sum(dim=(-2, -1)).clamp_min(torch.finfo(error.dtype).tiny)
    numerator = (error.unsqueeze(2) * weights).sum(dim=(-2, -1))
    return numerator / denominator


def region_mean_adjoint(
    probe: torch.Tensor,
    valid_mask: torch.Tensor,
    region_masks: torch.Tensor,
    input_channels: int,
) -> torch.Tensor:
    expected = (valid_mask.shape[0], input_channels, region_masks.shape[1])
    if tuple(probe.shape) != expected:
        raise ValueError(f"region probe must have shape {expected}")
    weights = region_masks.detach().to(dtype=probe.dtype, device=probe.device).unsqueeze(1)
    weights = weights * valid_mask.detach().to(dtype=probe.dtype, device=probe.device).unsqueeze(2)
    denominator = weights.sum(dim=(-2, -1)).clamp_min(torch.finfo(probe.dtype).tiny)
    return (probe[..., None, None] * weights / denominator[..., None, None]).sum(dim=2)


def apply_conditioned_geometry_operator(
    error: torch.Tensor,
    valid_mask: torch.Tensor,
    initial_sic: torch.Tensor,
    initial_sit: torch.Tensor,
    *,
    region_masks: torch.Tensor | None = None,
    spec: GeometryCFMSpec = GeometryCFMSpec(),
) -> dict[str, torch.Tensor]:
    spec.validate()
    _validate_inputs(error, valid_mask, initial_sic, initial_sit)
    _validate_region_masks(error, valid_mask, region_masks)
    mask = _expanded_mask(error, valid_mask)
    edge = initial_edge_weight(initial_sic, valid_mask).to(dtype=error.dtype, device=error.device)
    result = {
        f"multiscale_{factor}": masked_mean_pool(error, valid_mask, factor)
        for factor in spec.multiscale_factors
    }
    result["lead"] = lead_difference(error, valid_mask)
    result["product"] = linearized_volume_error(
        error, valid_mask, initial_sic, initial_sit, spec.field_stds, spec.product_scale
    )
    result["edge"] = error * mask * edge.sqrt().expand_as(error)
    if region_masks is not None:
        result["region"] = region_mean(error, valid_mask, region_masks)
    return result


def adjoint_conditioned_geometry_operator(
    probes: dict[str, torch.Tensor],
    valid_mask: torch.Tensor,
    initial_sic: torch.Tensor,
    initial_sit: torch.Tensor,
    *,
    input_channels: int = CHANNELS,
    region_masks: torch.Tensor | None = None,
    spec: GeometryCFMSpec = GeometryCFMSpec(),
) -> torch.Tensor:
    spec.validate()
    shape = (valid_mask.shape[0], input_channels, *valid_mask.shape[-2:])
    reference = next(iter(probes.values()), valid_mask)
    result = torch.zeros(shape, dtype=reference.dtype, device=reference.device)
    for factor in spec.multiscale_factors:
        key = f"multiscale_{factor}"
        if key in probes:
            result += masked_mean_pool_adjoint(probes[key], valid_mask, factor, input_channels)
    if "lead" in probes:
        result += lead_difference_adjoint(probes["lead"], valid_mask)
    if "product" in probes:
        result += linearized_volume_adjoint(
            probes["product"], valid_mask, initial_sic, initial_sit, spec.field_stds, spec.product_scale
        )
    if "edge" in probes:
        edge = initial_edge_weight(initial_sic, valid_mask).to(dtype=result.dtype, device=result.device)
        mask = valid_mask.detach().to(dtype=result.dtype, device=result.device).expand_as(result)
        result += probes["edge"] * mask * edge.sqrt().expand_as(result)
    if "region" in probes:
        if region_masks is None:
            raise ValueError("region probe supplied without region_masks")
        result += region_mean_adjoint(probes["region"], valid_mask, region_masks, input_channels)
    unknown = set(probes) - {
        *(f"multiscale_{factor}" for factor in spec.multiscale_factors),
        "lead",
        "product",
        "edge",
        "region",
    }
    if unknown:
        raise ValueError(f"unknown geometry probes: {sorted(unknown)}")
    return result


def _mean_square(value: torch.Tensor) -> torch.Tensor:
    return value.square().sum() / float(value.numel())


def geometry_quadratic(
    error: torch.Tensor,
    valid_mask: torch.Tensor,
    initial_sic: torch.Tensor,
    initial_sit: torch.Tensor,
    *,
    region_masks: torch.Tensor | None = None,
    spec: GeometryCFMSpec = GeometryCFMSpec(),
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    features = apply_conditioned_geometry_operator(
        error,
        valid_mask,
        initial_sic,
        initial_sit,
        region_masks=region_masks,
        spec=spec,
    )
    terms: dict[str, torch.Tensor] = {}
    multiscale = [features[f"multiscale_{factor}"] for factor in spec.multiscale_factors]
    terms["multiscale"] = sum((_mean_square(value) for value in multiscale), error.new_tensor(0.0))
    terms["lead"] = _mean_square(features["lead"])
    terms["product"] = _mean_square(features["product"])
    terms["edge"] = _mean_square(features["edge"])
    terms["region"] = (
        _mean_square(features["region"]) if "region" in features else error.new_tensor(0.0)
    )
    weighted = (
        float(spec.multiscale_group_weight) * terms["multiscale"]
        + float(spec.lead_group_weight) * terms["lead"]
        + float(spec.product_group_weight) * terms["product"]
        + float(spec.edge_group_weight) * terms["edge"]
        + float(spec.region_group_weight) * terms["region"]
    )
    return weighted, terms


def geometry_weighted_cfm_loss(
    prediction: torch.Tensor,
    target_velocity: torch.Tensor,
    valid_mask: torch.Tensor,
    initial_sic: torch.Tensor,
    initial_sit: torch.Tensor,
    *,
    geometry_weight: float,
    region_masks: torch.Tensor | None = None,
    spec: GeometryCFMSpec = GeometryCFMSpec(),
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if not math.isfinite(float(geometry_weight)) or geometry_weight < 0:
        raise ValueError("geometry_weight must be finite and non-negative")
    if tuple(prediction.shape) != tuple(target_velocity.shape):
        raise ValueError("prediction and target_velocity must have exactly matching shapes")
    if prediction.dtype != target_velocity.dtype or prediction.device != target_velocity.device:
        raise ValueError("prediction and target_velocity must share dtype and device")
    _validate_inputs(prediction - target_velocity, valid_mask, initial_sic, initial_sit)
    _validate_region_masks(prediction, valid_mask, region_masks)
    mask = _expanded_mask(prediction, valid_mask)
    native = ((prediction - target_velocity).square() * mask).sum() / mask.sum().clamp_min(1.0)
    # Preserve the production baseline bit-for-bit and gradient-for-gradient.
    if geometry_weight == 0:
        return native, {"native": native, "geometry": native.new_tensor(0.0), "total": native}
    geometry, terms = geometry_quadratic(
        prediction - target_velocity,
        valid_mask,
        initial_sic,
        initial_sit,
        region_masks=region_masks,
        spec=spec,
    )
    total = native + float(geometry_weight) * geometry
    return total, {"native": native, "geometry": geometry, "total": total, **terms}


def _admission_tensors(dtype: torch.dtype = torch.float64):
    generator = torch.Generator().manual_seed(918273)
    error = torch.randn((2, CHANNELS, 16, 16), generator=generator, dtype=dtype)
    valid = (torch.rand((2, 1, 16, 16), generator=generator, dtype=dtype) > 0.2).to(dtype)
    sic0 = torch.rand((2, 1, 16, 16), generator=generator, dtype=dtype)
    sit0 = 3.0 * torch.rand((2, 1, 16, 16), generator=generator, dtype=dtype)
    regions = torch.zeros((2, 3, 16, 16), dtype=dtype)
    regions[:, 0, :8] = 1
    regions[:, 1, 8:] = 1
    regions[:, 2, :, 4:12] = 1
    return error, valid, sic0, sit0, regions, generator


def run_cpu_admission() -> dict[str, object]:
    """Execute the numerical gate required before any one-GPU A/B."""
    error, valid, sic0, sit0, regions, generator = _admission_tensors()
    spec = GeometryCFMSpec(multiscale_factors=(4, 8), field_stds=(0.31, 0.77), product_scale=0.52)

    pred_a = error.clone().requires_grad_(True)
    pred_b = error.clone().requires_grad_(True)
    target = torch.randn(error.shape, generator=generator, dtype=error.dtype)
    mask = valid.expand_as(error)
    baseline = ((pred_a - target).square() * mask).sum() / mask.sum().clamp_min(1.0)
    candidate, _ = geometry_weighted_cfm_loss(
        pred_b, target, valid, sic0, sit0, geometry_weight=0.0, region_masks=regions, spec=spec
    )
    baseline.backward()
    candidate.backward()
    lambda_zero_loss_error = float((baseline.detach() - candidate.detach()).abs())
    lambda_zero_gradient_error = float((pred_a.grad - pred_b.grad).abs().max())

    features = apply_conditioned_geometry_operator(
        error, valid, sic0, sit0, region_masks=regions, spec=spec
    )
    probes = {
        name: torch.randn(value.shape, generator=generator, dtype=value.dtype) for name, value in features.items()
    }
    lhs = sum(((features[name] * probes[name]).sum() for name in features), error.new_tensor(0.0))
    adjoint = adjoint_conditioned_geometry_operator(
        probes, valid, sic0, sit0, region_masks=regions, spec=spec
    )
    rhs = (error * adjoint).sum()
    adjoint_error = float((lhs - rhs).abs() / torch.maximum(lhs.abs(), rhs.abs()).clamp_min(1.0))

    variable = error.clone().requires_grad_(True)
    direction = torch.randn(error.shape, generator=generator, dtype=error.dtype)
    geometry, _ = geometry_quadratic(
        variable, valid, sic0, sit0, region_masks=regions, spec=spec
    )
    gradient = torch.autograd.grad(geometry, variable)[0]
    analytic_directional = (gradient * direction).sum()
    epsilon = 1e-6
    plus = geometry_quadratic(
        variable.detach() + epsilon * direction, valid, sic0, sit0, region_masks=regions, spec=spec
    )[0]
    minus = geometry_quadratic(
        variable.detach() - epsilon * direction, valid, sic0, sit0, region_masks=regions, spec=spec
    )[0]
    finite_directional = (plus - minus) / (2.0 * epsilon)
    finite_difference_error = float(
        (analytic_directional - finite_directional).abs()
        / torch.maximum(analytic_directional.abs(), finite_directional.abs()).clamp_min(1.0)
    )

    # For one fixed condition, every SPD conditioned quadratic has the sample
    # conditional mean as its exact empirical minimiser.
    oracle_generator = torch.Generator().manual_seed(3381)
    oracle_targets = torch.randn((7, CHANNELS, 8, 8), generator=oracle_generator, dtype=torch.float64)
    oracle_valid = torch.ones((7, 1, 8, 8), dtype=torch.float64)
    shared_sic = torch.rand((1, 1, 8, 8), generator=oracle_generator, dtype=torch.float64).repeat(7, 1, 1, 1)
    shared_sit = torch.rand((1, 1, 8, 8), generator=oracle_generator, dtype=torch.float64).repeat(7, 1, 1, 1)
    shared_regions = torch.ones((7, 1, 8, 8), dtype=torch.float64)
    oracle_mean = oracle_targets.mean(dim=0, keepdim=True).clone().requires_grad_(True)
    oracle_prediction = oracle_mean.expand_as(oracle_targets)
    oracle_loss, _ = geometry_weighted_cfm_loss(
        oracle_prediction,
        oracle_targets,
        oracle_valid,
        shared_sic,
        shared_sit,
        geometry_weight=0.37,
        region_masks=shared_regions,
        spec=GeometryCFMSpec(multiscale_factors=(2, 4)),
    )
    oracle_gradient = torch.autograd.grad(oracle_loss, oracle_mean)[0]
    oracle_gradient_max_abs = float(oracle_gradient.abs().max())
    oracle_direction = torch.randn(oracle_mean.shape, generator=oracle_generator, dtype=torch.float64)
    step = 1e-4
    shifted_loss, _ = geometry_weighted_cfm_loss(
        (oracle_mean.detach() + step * oracle_direction).expand_as(oracle_targets),
        oracle_targets,
        oracle_valid,
        shared_sic,
        shared_sit,
        geometry_weight=0.37,
        region_masks=shared_regions,
        spec=GeometryCFMSpec(multiscale_factors=(2, 4)),
    )
    oracle_curvature = float(2.0 * (shifted_loss - oracle_loss.detach()) / (step * step))

    thresholds = {
        "lambda_zero_loss_max_abs": 0.0,
        "lambda_zero_gradient_max_abs": 0.0,
        "adjoint_relative_error": 1e-12,
        "finite_difference_relative_error": 1e-7,
        "oracle_gradient_max_abs": 1e-12,
        "oracle_directional_curvature_min": 0.0,
    }
    measurements = {
        "lambda_zero_loss_max_abs": lambda_zero_loss_error,
        "lambda_zero_gradient_max_abs": lambda_zero_gradient_error,
        "adjoint_relative_error": adjoint_error,
        "finite_difference_relative_error": finite_difference_error,
        "oracle_gradient_max_abs": oracle_gradient_max_abs,
        "oracle_directional_curvature": oracle_curvature,
    }
    passed = (
        lambda_zero_loss_error == 0.0
        and lambda_zero_gradient_error == 0.0
        and adjoint_error <= thresholds["adjoint_relative_error"]
        and finite_difference_error <= thresholds["finite_difference_relative_error"]
        and oracle_gradient_max_abs <= thresholds["oracle_gradient_max_abs"]
        and oracle_curvature > thresholds["oracle_directional_curvature_min"]
    )
    return {
        "schema_version": "direct_dynamics_geometry_weighted_cfm_cpu_admission_v1",
        "status": "passed" if passed else "failed",
        "scope": "CPU-only implementation and law-preservation admission; not a calibration result",
        "future_truth_weights": False,
        "condition_sources": ["valid_ocean", "d0_sic", "d0_sit", "fixed_regions"],
        "measurements": measurements,
        "thresholds": thresholds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_cpu_admission()
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
