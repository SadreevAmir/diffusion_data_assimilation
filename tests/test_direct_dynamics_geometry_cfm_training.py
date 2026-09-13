import json
from pathlib import Path

import torch

from assim_lib.direct_dynamics_geometry_cfm_training import (
    _check_protocol,
    _load_frozen_metric,
    compute_matched_loss,
    make_matched_schedule,
    training_contract_sha256,
)
from assim_lib.trainer import UNetTrainer


ROOT = Path(__file__).resolve().parents[1]


def _protocol():
    return json.loads(
        (ROOT / "config/experiments/train_direct_dynamics_geometry_full_cfm_ab_v1.json").read_text()
    )["protocol"]


def test_frozen_metric_is_loaded_from_bound_config_not_defaults():
    path = ROOT / "config/admission/direct_dynamics_geometry_weighted_cfm_v1.json"
    spec, weight, _ = _load_frozen_metric(
        path, "6b44cbbff783084264c290b89d429b01270d4179cfa164cf1ef6f09b099c33c8"
    )
    assert spec.multiscale_factors == (2, 4)
    assert spec.multiscale_group_weight == 0.125
    assert spec.region_group_weight == 0.0
    assert weight == 0.1


def test_schedule_is_exactly_reproducible_without_repeated_examples():
    protocol = _protocol()
    _check_protocol(protocol)
    left = make_matched_schedule(10000, protocol)
    right = make_matched_schedule(10000, protocol)
    assert left == right
    assert len(left["indices"]) == protocol["updates_per_arm"] * protocol["batch_size"]
    assert len(set(left["indices"])) == len(left["indices"])
    times = torch.tensor(left["timesteps"])
    assert torch.all((times > 0) & (times < 1))


def test_real_matched_loss_path_is_fp32_and_control_is_native_exact():
    generator = torch.Generator().manual_seed(912)
    prediction = torch.randn((2, 6, 16, 16), generator=generator, dtype=torch.bfloat16).requires_grad_(True)
    target = torch.randn((2, 6, 16, 16), generator=generator, dtype=torch.float32)
    valid = torch.ones((2, 1, 16, 16), dtype=torch.float32)
    batch = {
        "valid": valid,
        "initial_sic": torch.rand((2, 1, 16, 16), generator=generator),
        "initial_sit": torch.rand((2, 1, 16, 16), generator=generator),
    }
    metric_path = ROOT / "config/admission/direct_dynamics_geometry_weighted_cfm_v1.json"
    spec, weight, _ = _load_frozen_metric(
        metric_path, "6b44cbbff783084264c290b89d429b01270d4179cfa164cf1ef6f09b099c33c8"
    )
    control, control_parts = compute_matched_loss(
        prediction, target, batch, geometry_weight=0.0, spec=spec
    )
    native = UNetTrainer._masked_mse(prediction.float(), target, valid)
    assert torch.equal(control, native)
    assert control.dtype == torch.float32
    treatment, treatment_parts = compute_matched_loss(
        prediction, target, batch, geometry_weight=weight, spec=spec
    )
    assert treatment.dtype == torch.float32
    assert treatment > control
    assert treatment_parts["geometry"] > 0
    treatment.backward()
    assert prediction.grad is not None and torch.isfinite(prediction.grad).all()
    assert control_parts["geometry"].item() == 0.0


def test_protocol_rejects_compile_or_any_unreviewed_extra():
    protocol = _protocol()
    protocol["torch_compile"] = True
    try:
        _check_protocol(protocol)
    except ValueError as error:
        assert "torch_compile" in str(error)
    else:
        raise AssertionError("unreviewed compile setting was accepted")


def test_preflight_binding_does_not_change_scientific_contract_hash():
    path = ROOT / "config/experiments/train_direct_dynamics_geometry_full_cfm_ab_v1.json"
    experiment = json.loads(path.read_text())
    original = training_contract_sha256(experiment)
    experiment["required_preflight"] = {"path": "/new/preflight.json", "sha256": "f" * 64}
    assert training_contract_sha256(experiment) == original
