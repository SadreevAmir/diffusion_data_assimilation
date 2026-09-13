import torch

from assim_lib.direct_dynamics_geometry_cfm import (
    GeometryCFMSpec,
    adjoint_conditioned_geometry_operator,
    apply_conditioned_geometry_operator,
    geometry_quadratic,
    geometry_weighted_cfm_loss,
    run_cpu_admission,
)


def _inputs():
    generator = torch.Generator().manual_seed(1234)
    error = torch.randn((2, 6, 16, 16), generator=generator, dtype=torch.float64)
    valid = (torch.rand((2, 1, 16, 16), generator=generator, dtype=torch.float64) > 0.15).double()
    sic0 = torch.rand((2, 1, 16, 16), generator=generator, dtype=torch.float64)
    sit0 = 2.5 * torch.rand((2, 1, 16, 16), generator=generator, dtype=torch.float64)
    regions = torch.zeros((2, 2, 16, 16), dtype=torch.float64)
    regions[:, 0, :, :8] = 1
    regions[:, 1, :, 8:] = 1
    return error, valid, sic0, sit0, regions, generator


def test_lambda_zero_is_exact_native_masked_cfm_loss_and_gradient():
    prediction, valid, sic0, sit0, regions, generator = _inputs()
    target = torch.randn(prediction.shape, generator=generator, dtype=prediction.dtype)
    baseline_prediction = prediction.clone().requires_grad_(True)
    weighted_prediction = prediction.clone().requires_grad_(True)
    mask = valid.expand_as(prediction)
    baseline = ((baseline_prediction - target).square() * mask).sum() / mask.sum().clamp_min(1.0)
    candidate, diagnostics = geometry_weighted_cfm_loss(
        weighted_prediction,
        target,
        valid,
        sic0,
        sit0,
        geometry_weight=0.0,
        region_masks=regions,
    )
    baseline.backward()
    candidate.backward()
    assert torch.equal(baseline, candidate)
    assert torch.equal(baseline_prediction.grad, weighted_prediction.grad)
    assert diagnostics["geometry"].item() == 0.0


def test_full_conditioned_operator_has_exact_numerical_adjoint():
    error, valid, sic0, sit0, regions, generator = _inputs()
    spec = GeometryCFMSpec(multiscale_factors=(4, 8), field_stds=(0.2, 0.7), product_scale=0.4)
    features = apply_conditioned_geometry_operator(
        error, valid, sic0, sit0, region_masks=regions, spec=spec
    )
    probes = {
        key: torch.randn(value.shape, generator=generator, dtype=value.dtype)
        for key, value in features.items()
    }
    lhs = sum((features[key] * probes[key]).sum() for key in features)
    rhs = (
        error
        * adjoint_conditioned_geometry_operator(
            probes, valid, sic0, sit0, region_masks=regions, spec=spec
        )
    ).sum()
    torch.testing.assert_close(lhs, rhs, rtol=1e-12, atol=1e-12)


def test_geometry_quadratic_gradient_matches_central_difference():
    error, valid, sic0, sit0, regions, generator = _inputs()
    variable = error.clone().requires_grad_(True)
    direction = torch.randn(error.shape, generator=generator, dtype=error.dtype)
    loss = geometry_quadratic(variable, valid, sic0, sit0, region_masks=regions)[0]
    gradient = torch.autograd.grad(loss, variable)[0]
    analytic = (gradient * direction).sum()
    epsilon = 1e-6
    plus = geometry_quadratic(
        variable.detach() + epsilon * direction, valid, sic0, sit0, region_masks=regions
    )[0]
    minus = geometry_quadratic(
        variable.detach() - epsilon * direction, valid, sic0, sit0, region_masks=regions
    )[0]
    numerical = (plus - minus) / (2 * epsilon)
    torch.testing.assert_close(analytic, numerical, rtol=1e-7, atol=1e-8)


def test_conditioning_is_detached_from_geometry_metric():
    error, valid, sic0, sit0, regions, _ = _inputs()
    sic0.requires_grad_(True)
    sit0.requires_grad_(True)
    regions.requires_grad_(True)
    loss = geometry_quadratic(error.requires_grad_(True), valid, sic0, sit0, region_masks=regions)[0]
    loss.backward()
    assert sic0.grad is None
    assert sit0.grad is None
    assert regions.grad is None


def test_cpu_admission_passes_and_is_explicitly_not_scientific_evidence():
    result = run_cpu_admission()
    assert result["status"] == "passed"
    assert "not a calibration result" in result["scope"]
    assert result["future_truth_weights"] is False
