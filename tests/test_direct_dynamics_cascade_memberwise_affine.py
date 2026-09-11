import tempfile
from pathlib import Path

import torch

from assim_lib.direct_dynamics_cascade_memberwise_affine import (
    _positive_ratio,
    _require_finite_tree,
    _validate_inputs,
    _finalize_failed_tracker,
    _haze_failures,
    _exact_atom_metrics,
    apply_memberwise,
    fair_crps_cases,
    validate_folds,
)


def test_memberwise_transform_does_not_depend_on_other_members():
    member = torch.tensor([-0.2, 0.5, 1.2, -0.1, 0.4, 2.0]).reshape(1, 1, 6, 1, 1)
    companion = torch.zeros_like(member)
    parameters = {"sic_scale": 1.2, "sic_offset": -0.1, "sit_scale": 0.8, "sit_offset": 0.2}
    alone = apply_memberwise(member, torch.tensor([1.0, 2.0]), parameters)
    together = apply_memberwise(torch.cat((member, companion), dim=1), torch.tensor([1.0, 2.0]), parameters)
    assert torch.equal(alone[:, 0], together[:, 0])
    assert torch.all((alone[:, :, 0::2] >= 0) & (alone[:, :, 0::2] <= 1))
    assert torch.all(alone[:, :, 1::2] >= 0)


def test_support_control_is_exact_censoring():
    values = torch.tensor([-0.1, -0.2, 1.1, 0.3, 0.5, -0.4]).reshape(1, 1, 6, 1, 1)
    result = apply_memberwise(
        values,
        torch.tensor([1.0, 1.0]),
        {"sic_scale": 1.0, "sic_offset": 0.0, "sit_scale": 1.0, "sit_offset": 0.0},
    )
    torch.testing.assert_close(
        result.flatten(), torch.tensor([0.0, 0.0, 1.0, 0.3, 0.5, 0.0])
    )


def test_fair_crps_two_member_closed_form():
    members = torch.tensor([0.0, 2.0]).reshape(1, 2, 1, 1, 1)
    truth = torch.tensor([[[[1.0]]]])
    valid = torch.ones(1, 1, 1, 1)
    assert fair_crps_cases(members, truth, valid).item() == 0.0


def test_folds_are_disjoint_and_cover_twelve_cases():
    validate_folds([[0, 3, 6, 9], [1, 4, 7, 10], [2, 5, 8, 11]], 12)


def test_nonfinite_diagnostics_fail_closed():
    try:
        _require_finite_tree({"metric": [1.0, float("nan")]})
    except FloatingPointError:
        pass
    else:
        raise AssertionError("NaN must fail closed")
    try:
        _positive_ratio(1.0, 0.0, "test")
    except FloatingPointError:
        pass
    else:
        raise AssertionError("zero denominator must fail closed")


def test_empty_mask_fails_closed():
    raw = torch.zeros(12, 8, 6, 320, 256)
    truth = torch.zeros(12, 6, 320, 256)
    valid = torch.zeros(12, 1, 320, 256)
    try:
        _validate_inputs(raw, truth, valid)
    except ValueError:
        return
    raise AssertionError("empty masks must fail closed")


def test_exact_zero_haze_can_reject_when_low_ice_aggregate_would_pass():
    spec = {
        "haze_mean_tolerance_m": 0.001, "haze_p95_tolerance_m": 0.001,
        "haze_fraction_tolerance": 0.005, "exact_zero_fraction_tolerance": 0.005,
    }
    base = {"exact_zero_fraction": .8, "mean_positive_sit": .01, "p95_positive_sit": .02, "fraction_gt_0p01": .1}
    candidate = {**base, "p95_positive_sit": .03}
    failures = _haze_failures(candidate, base, spec, "d3_sit exact-zero", True)
    assert any("p95_positive_sit increased" in failure for failure in failures)


def test_negative_sit_is_not_counted_as_an_exact_zero_atom():
    ensemble = torch.zeros(1, 3, 6, 1, 1)
    ensemble[0, :, 1, 0, 0] = torch.tensor([-0.02, 0.0, 0.02])
    metrics = _exact_atom_metrics(ensemble, torch.zeros(1, 6, 1, 1), torch.ones(1, 1, 1, 1))
    torch.testing.assert_close(
        torch.tensor(metrics["d3_sit"]["open_water_exact_zero"]["exact_zero_fraction"]),
        torch.tensor(1.0 / 3.0),
    )


def test_tracker_close_runs_even_when_failure_upload_raises():
    calls = []
    class Task:
        def mark_failed(self, **kwargs): calls.append("mark_failed")
    class Tracker:
        task = Task()
        def upload_artifact(self, *args): calls.append("upload"); raise RuntimeError("network")
        def close(self): calls.append("close")
    with tempfile.TemporaryDirectory() as directory:
        _finalize_failed_tracker(Tracker(), RuntimeError("primary"), Path(directory) / "failure.json")
    assert calls == ["mark_failed", "upload", "close"]
