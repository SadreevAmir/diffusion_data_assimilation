import contextlib
import json
from pathlib import Path
from unittest import mock

import torch

from assim_lib.direct_dynamics_geometry_cfm_training import (
    _check_protocol,
    _gradient_norm,
    _assert_exact_matched_prediction,
    _load_frozen_metric,
    _merge_status,
    _one_forward,
    _persist_preflight_evidence,
    _record_terminal_failure,
    compute_matched_loss,
    make_matched_schedule,
    training_contract_sha256,
)
from assim_lib.direct_dynamics_geometry_cfm import GeometryCFMSpec
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


def test_gradient_norm_rejects_zero_parameter_gradient():
    model = torch.nn.Linear(2, 1)
    for parameter in model.parameters():
        parameter.grad = torch.zeros_like(parameter)
    try:
        _gradient_norm(model)
    except FloatingPointError as error:
        assert "strictly positive" in str(error)
    else:
        raise AssertionError("zero parameter gradient was accepted")


def test_failure_after_evidence_preserves_evidence_and_closes_tracker(tmp_path):
    calls = []

    class Task:
        def mark_failed(self, **_kwargs):
            calls.append("mark_failed")

    class Tracker:
        task = Task()

        def close(self):
            calls.append("close")

    status = tmp_path / "status.json"
    _merge_status(
        status,
        status="fixed_inputs_and_predictions_saved_before_backward",
        fixed_inputs_and_predictions_sha256="a" * 64,
    )
    _record_terminal_failure(status, Tracker(), RuntimeError("injected"))
    payload = json.loads(status.read_text())
    assert payload["status"] == "failed_terminal_non_resumable"
    assert payload["fixed_inputs_and_predictions_sha256"] == "a" * 64
    assert calls == ["mark_failed", "close"]


def test_mismatched_predictions_are_durable_before_parity_failure(tmp_path):
    path = tmp_path / "fixed_inputs_and_predictions.pt"
    common = {"truth": torch.zeros((1, 1, 2, 2))}
    control = torch.zeros((1, 1, 2, 2))
    treatment = torch.ones((1, 1, 2, 2))
    _persist_preflight_evidence(
        path,
        common,
        {"control_evidence_forward": control, "treatment_evidence_forward": treatment},
    )
    try:
        _assert_exact_matched_prediction(control, treatment)
    except RuntimeError as error:
        assert "predictions differ" in str(error)
    else:
        raise AssertionError("intentional parity mismatch was accepted")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert torch.equal(payload["predictions"]["control_evidence_forward"], control)
    assert torch.equal(payload["predictions"]["treatment_evidence_forward"], treatment)


def test_failure_status_write_cannot_skip_cleanup_or_mask_primary(tmp_path):
    calls = []

    class Task:
        def mark_failed(self, **_kwargs):
            calls.append("mark_failed")

    class Tracker:
        task = Task()

        def close(self):
            calls.append("close")

    primary = RuntimeError("primary failure")
    with mock.patch(
        "assim_lib.direct_dynamics_geometry_cfm_training._merge_status",
        side_effect=OSError("status disk unavailable"),
    ):
        result = _record_terminal_failure(tmp_path / "status.json", Tracker(), primary)
    assert calls == ["mark_failed", "close"]
    assert result["error"] == "primary failure"
    assert result["error_type"] == "RuntimeError"
    assert any("failure_status_write" in item for item in result["cleanup_errors"])


def test_production_forward_persists_prediction_before_scoring_failure(tmp_path):
    class Model(torch.nn.Module):
        def forward(self, model_input, _time, return_dict=False):
            assert return_dict is False
            return (torch.full_like(model_input[:, :6], 2.5),)

    batch = {
        "truth": torch.zeros((1, 6, 2, 2)),
        "valid": torch.ones((1, 1, 2, 2)),
        "condition": torch.zeros((1, 15, 2, 2)),
        "initial_sic": torch.zeros((1, 1, 2, 2)),
        "initial_sit": torch.zeros((1, 1, 2, 2)),
    }
    evidence_path = tmp_path / "scoring_failure_evidence.pt"

    def persist(prediction):
        _persist_preflight_evidence(
            evidence_path,
            {"truth": batch["truth"]},
            {"prediction_before_scoring": prediction},
        )

    with (
        mock.patch("torch.autocast", return_value=contextlib.nullcontext()),
        mock.patch(
            "assim_lib.direct_dynamics_geometry_cfm_training.compute_matched_loss",
            side_effect=RuntimeError("injected scoring failure"),
        ),
    ):
        try:
            _one_forward(
                Model(),
                batch,
                torch.tensor([0.5]),
                7,
                8,
                torch.zeros((1, 2, 2, 2)),
                0.0,
                GeometryCFMSpec(),
                prediction_callback=persist,
            )
        except RuntimeError as error:
            assert str(error) == "injected scoring failure"
        else:
            raise AssertionError("injected scoring failure was swallowed")
    payload = torch.load(evidence_path, map_location="cpu", weights_only=True)
    saved = payload["predictions"]["prediction_before_scoring"]
    assert torch.equal(saved, torch.full((1, 6, 2, 2), 2.5))
    assert payload["prediction_sha256"]["prediction_before_scoring"]


def test_launcher_has_exact_clean_one_gpu_admission_and_timeout():
    source = (ROOT / "scripts/run_direct_dynamics_geometry_full_cfm_ab.sh").read_text()
    assert "GEOMETRY_FULL_CFM_EXPECTED_COMMIT" in source
    assert "git status --porcelain --untracked-files=all" in source
    assert "scripts/require_single_gpu_uuid.sh" in source
    assert "GPU-40ff1cbb-07cc-992e-23cb-8fe181972b76" in source
    assert "memory.used,utilization.gpu" in source
    assert "for sample in {1..11}" in source
    assert 'export CUDA_VISIBLE_DEVICES="$GPU_UUID"' in source
    assert "CLEARML_REQUIRE_ONLINE=1" in source
    assert "timeout --foreground" in source
    assert "RESULT_ROOT=\"/home/autoresearch_results/direct_dynamics_geometry_cfm_v1\"" in source
