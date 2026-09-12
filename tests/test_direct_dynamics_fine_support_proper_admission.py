import json
import hashlib

import pytest
import torch
from torchdiffeq import odeint

import assim_lib.direct_dynamics_fine_support_proper_admission as admission
from assim_lib.direct_dynamics_cascade_coarse_fine_boundary_audit import _load_normalization
from assim_lib.direct_dynamics_fine_support_proper_admission import (
    _analytic_finite_member_check,
    apply_training_sic_decoder,
    differentiable_fixed_budget_projection,
)
from assim_lib.direct_dynamics_sic_support_decoder import (
    project_masked_blocks_to_unit_interval_mean,
)
from assim_lib.direct_dynamics_fine_support_proper_integration import (
    CompactFineNetwork,
    compact_fine_integration_check,
    fine_rk4_interval,
    fine_velocity,
)
from assim_lib.runtime import make_normalized_xy_grid


def test_differentiable_forward_is_bitwise_reviewed_projection():
    fine = torch.tensor([[[[-0.3, 0.2], [0.8, 1.4]]]], dtype=torch.float64)
    coarse = torch.tensor([[[[0.5]]]], dtype=torch.float64)
    mask = torch.ones_like(fine)
    expected = project_masked_blocks_to_unit_interval_mean(fine, coarse, mask)
    actual = differentiable_fixed_budget_projection(fine, coarse, mask)
    assert torch.equal(actual, expected)


def test_projection_backward_matches_gradcheck_away_from_active_set_kinks():
    fine = torch.tensor([[[[0.1, 0.3], [0.6, 0.9]]]], dtype=torch.float64, requires_grad=True)
    coarse = torch.tensor([[[[0.475]]]], dtype=torch.float64)
    mask = torch.ones_like(fine)
    assert torch.autograd.gradcheck(
        lambda value: differentiable_fixed_budget_projection(value, coarse, mask),
        (fine,), eps=1e-6, atol=2e-6, rtol=2e-5,
    )


def test_roundoff_repair_does_not_make_saturated_endpoint_differentiable():
    fine = torch.tensor([[[[-1.0, 0.2], [0.7, 2.0]]]], dtype=torch.float64, requires_grad=True)
    coarse = torch.tensor([[[[0.5]]]], dtype=torch.float64)
    mask = torch.ones_like(fine)
    decoded = differentiable_fixed_budget_projection(fine, coarse, mask)
    decoded[0, 0, 0, 0].backward()
    assert torch.equal(fine.grad, torch.zeros_like(fine.grad))


def test_projection_backward_matches_gradcheck_with_saturation_and_coast():
    fine = torch.tensor([[[[-1.0, 0.2], [0.7, 2.0]]]], dtype=torch.float64, requires_grad=True)
    coarse = torch.tensor([[[[0.3]]]], dtype=torch.float64)
    mask = torch.tensor([[[[1.0, 1.0], [1.0, 0.0]]]], dtype=torch.float64)
    assert torch.autograd.gradcheck(
        lambda value: differentiable_fixed_budget_projection(value, coarse, mask),
        (fine,), eps=1e-6, atol=2e-6, rtol=2e-5,
    )


def test_zero_budget_gradient_is_zero_and_mixed_budget_gradient_is_live():
    fine = torch.tensor(
        [[[[0.3, -0.2, 0.1, 0.4], [0.8, 1.2, 0.6, 0.9]]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    coarse = torch.tensor([[[[0.0, 0.5]]]], dtype=torch.float64)
    mask = torch.ones_like(fine)
    decoded = differentiable_fixed_budget_projection(fine, coarse, mask)
    decoded.square().sum().backward()
    assert torch.equal(fine.grad[..., :2], torch.zeros_like(fine.grad[..., :2]))
    assert float(fine.grad[..., 2:].norm()) > 0


def test_finite_member_score_matches_analytic_unbiased_value():
    result = _analytic_finite_member_check()
    assert result["expected"] == pytest.approx(1 / 6, abs=1e-14, rel=0)
    assert result["fair_crps"] == pytest.approx(1 / 6, abs=1e-14, rel=0)
    assert result["joint_energy"] == pytest.approx(1 / 6, abs=1e-14, rel=0)


def test_normalization_metadata_sha_tamper_is_rejected(tmp_path):
    metadata = tmp_path / "normalization.json"
    metadata.write_text(json.dumps({"data_config": {"fields": ["siconc", "sithic"], "means": [0.0, 0.0], "stds": [1.0, 1.0]}}))
    original_sha = hashlib.sha256(metadata.read_bytes()).hexdigest()
    metadata.write_text(metadata.read_text() + "\n")
    with pytest.raises(ValueError, match="normalization source metadata SHA mismatch"):
        _load_normalization({"normalization_source": {"path": str(metadata), "sha256": original_sha}})


def test_training_decoder_rejects_trainable_coarse_budget():
    fine = torch.zeros(1, 2, 6, 2, 2, dtype=torch.float64, requires_grad=True)
    coarse = torch.zeros(1, 2, 6, 1, 1, dtype=torch.float64, requires_grad=True)
    mask = torch.ones(1, 1, 2, 2, dtype=torch.float64)
    try:
        apply_training_sic_decoder(fine, coarse, mask)
    except ValueError as error:
        assert "fixed coarse budget" in str(error)
    else:
        raise AssertionError("trainable coarse budget was accepted")


def test_failure_after_reservation_is_durable(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"schema_version": "fine_support_proper_cpu_admission_v1"}))
    output = tmp_path / "status.json"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch, "set_num_threads", lambda _: None)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda _: None)
    monkeypatch.setattr(torch, "get_num_threads", lambda: 6)
    monkeypatch.setattr(torch, "get_num_interop_threads", lambda: 1)
    monkeypatch.setattr(
        admission,
        "_load_bound_provenance",
        lambda *_: (_ for _ in ()).throw(RuntimeError("synthetic failure")),
    )
    with pytest.raises(RuntimeError, match="synthetic failure"):
        admission.run(config_path, output)
    status = json.loads(output.read_text())
    assert status["status"] == "failed"
    assert status["error_type"] == "RuntimeError"
    assert status["error"] == "synthetic failure"


def test_compact_fine_production_path_has_terminal_parameter_gradient():
    result = compact_fine_integration_check()
    assert result["status"] == "pass"
    assert result["rk4_intervals"] == 32
    assert result["frozen_prefix_intervals"] == 31
    assert result["trainable_terminal_intervals"] == 1
    assert result["candidate_control_max_abs"] <= 1e-6
    assert result["production_replay_max_abs"] <= 2e-5
    assert result["terminal_parameter_gradient_norm"] > 0
    assert result["frozen_prefix_has_no_graph"]
    assert result["frozen_parameter_gradients_absent"]


def test_every_reverse_time_rk4_state_is_bitwise_torchdiffeq():
    torch.manual_seed(31)
    model = CompactFineNetwork().eval()
    state = torch.randn(2, 6, 8, 8)
    condition = torch.randn(2, 21, 8, 8)
    active = torch.ones(2, 1, 8, 8)
    grid = make_normalized_xy_grid(8, 8, device=torch.device("cpu"), dtype=torch.float32)
    timepoints = torch.linspace(1.0, 0.0, 33, dtype=torch.float32)

    def production_velocity(time, value):
        return fine_velocity(model, value, condition, active, grid, time)

    production = odeint(
        production_velocity,
        state,
        timepoints,
        method="rk4",
        options={"step_size": 1.0 / 32.0},
    )
    custom = [state]
    for index in range(32):
        custom.append(
            fine_rk4_interval(
                model,
                custom[-1],
                condition,
                active,
                grid,
                timepoints[index],
                timepoints[index + 1],
            )
        )
    custom = torch.stack(custom)
    assert torch.equal(custom, production)
