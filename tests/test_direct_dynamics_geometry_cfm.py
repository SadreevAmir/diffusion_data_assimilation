import torch

from assim_lib.direct_dynamics_geometry_cfm import (
    GeometryCFMSpec,
    adjoint_conditioned_geometry_operator,
    apply_conditioned_geometry_operator,
    geometry_quadratic,
    geometry_weighted_cfm_loss,
    run_cpu_admission,
)
from assim_lib.trainer import UNetTrainer


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
    baseline = UNetTrainer._masked_mse(baseline_prediction, target, mask)
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
    valid.requires_grad_(True)
    sic0.requires_grad_(True)
    sit0.requires_grad_(True)
    regions.requires_grad_(True)
    loss = geometry_quadratic(error.requires_grad_(True), valid, sic0, sit0, region_masks=regions)[0]
    loss.backward()
    assert valid.grad is None
    assert sic0.grad is None
    assert sit0.grad is None
    assert regions.grad is None


def test_coast_is_not_mistaken_for_an_ice_edge():
    from assim_lib.direct_dynamics_geometry_cfm import initial_edge_weight

    valid = torch.ones((1, 1, 16, 16), dtype=torch.float64)
    valid[..., :, :4] = 0
    constant_ocean_a = torch.full_like(valid, 0.65)
    constant_ocean_a[..., :, :4] = 0
    constant_ocean_b = constant_ocean_a.clone()
    constant_ocean_b[..., :, :4] = 91.0
    edge_a = initial_edge_weight(constant_ocean_a, valid)
    edge_b = initial_edge_weight(constant_ocean_b, valid)
    assert torch.count_nonzero(edge_a) == 0
    assert torch.equal(edge_a, edge_b)

    true_edge = constant_ocean_a.clone()
    true_edge[..., :, 10:] = 0.0
    detected = initial_edge_weight(true_edge, valid)
    assert detected[..., :, 8:12].max() > 0


def test_loss_fails_closed_on_malformed_target_and_empty_mask():
    prediction, valid, sic0, sit0, regions, _ = _inputs()
    try:
        geometry_weighted_cfm_loss(
            prediction,
            prediction[:, :1],
            valid,
            sic0,
            sit0,
            geometry_weight=0.0,
            region_masks=regions,
        )
    except ValueError as error:
        assert "matching shapes" in str(error)
    else:
        raise AssertionError("broadcastable malformed target was accepted")

    empty = valid.clone()
    empty[0] = 0
    try:
        geometry_weighted_cfm_loss(
            prediction,
            prediction,
            empty,
            sic0,
            sit0,
            geometry_weight=0.0,
            region_masks=regions,
        )
    except ValueError as error:
        assert "valid ocean" in str(error)
    else:
        raise AssertionError("empty valid-ocean case was accepted")


def test_loss_fails_closed_on_invalid_regions():
    prediction, valid, sic0, sit0, regions, _ = _inputs()
    regions[0, 0] = 0
    try:
        geometry_weighted_cfm_loss(
            prediction,
            prediction,
            valid,
            sic0,
            sit0,
            geometry_weight=0.1,
            region_masks=regions,
        )
    except ValueError as error:
        assert "overlap valid ocean" in str(error)
    else:
        raise AssertionError("empty conditioned region was accepted")


def test_cpu_admission_passes_and_is_explicitly_not_scientific_evidence():
    result = run_cpu_admission()
    assert result["status"] == "passed"
    assert "not a calibration result" in result["scope"]
    assert result["future_truth_weights"] is False
