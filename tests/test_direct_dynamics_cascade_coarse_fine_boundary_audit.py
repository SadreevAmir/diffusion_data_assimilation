import json
import tempfile
from pathlib import Path

import tempfile
from pathlib import Path

import torch

from assim_lib.direct_dynamics_cascade import decompose
from assim_lib.direct_dynamics_cascade_coarse_fine_boundary_audit import (
    _decomposition,
    _finish_success,
    _heterogeneous_block_counterexample,
    _record_failure,
    _strict_atomic_json,
    _terminate,
    _strict_atomic_json,
)


def _payload(forecast: torch.Tensor, mask: torch.Tensor) -> dict:
    batch, members = forecast.shape[:2]
    flat = forecast.flatten(0, 1)
    flat_mask = mask[:, None].expand(-1, members, -1, -1, -1).flatten(0, 1)
    state = decompose(flat, flat_mask)
    return {
        "forecast_normalized": forecast,
        "coarse_normalized": state.coarse.unflatten(0, (batch, members)),
        "residual_normalized": state.residual.unflatten(0, (batch, members)),
        "truth_normalized": torch.zeros_like(forecast[:, 0]),
        "valid_mask": mask,
    }


def test_heterogeneous_block_proves_coarse_boundary_event_is_not_fine_event():
    result = _heterogeneous_block_counterexample()
    assert result["fine_fraction_sit_le_0p01"] == 0.75
    assert abs(result["coarse_mean_sit_metres"] - 0.02) < 1e-14
    assert result["coarse_event_sit_le_0p01"] is False


def test_reconstruction_nullspace_and_mse_decomposition_keep_cross_term():
    generator = torch.Generator().manual_seed(9921)
    mask = torch.ones((2, 1, 8, 10), dtype=torch.float64)
    mask[:, :, :2, :2] = 0
    raw = _payload(torch.randn((2, 3, 6, 8, 10), generator=generator, dtype=torch.float64), mask)
    candidate = _payload(
        raw["forecast_normalized"]
        + 0.1 * torch.randn((2, 3, 6, 8, 10), generator=generator, dtype=torch.float64),
        mask,
    )
    result = _decomposition(raw, candidate)
    assert result["reconstruction_max_abs"] < 1e-14
    assert result["residual_coarse_max_abs"] < 1e-14
    assert result["mse_change_identity_max_abs"] < 1e-14
    assert "cross_coarse_residual" in result["per_case_output_terms"]


def test_corrupted_residual_decomposition_fails_closed():
    mask = torch.ones((1, 1, 4, 4), dtype=torch.float64)
    raw = _payload(torch.zeros((1, 2, 6, 4, 4), dtype=torch.float64), mask)
    candidate = _payload(torch.ones((1, 2, 6, 4, 4), dtype=torch.float64), mask)
    candidate["residual_normalized"] = candidate["residual_normalized"].clone()
    candidate["residual_normalized"][:, :, :, 0, 0] = 0.5
    try:
        _decomposition(raw, candidate)
    except ValueError as error:
        assert "delta" in str(error) or "nullspace" in str(error)
    else:
        raise AssertionError("corrupted decomposition was accepted")


def test_strict_json_rejects_nonfinite_result():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "result.json"
        try:
            _strict_atomic_json(path, {"metric": float("nan")})
        except FloatingPointError:
            pass
        else:
            raise AssertionError("nonfinite JSON was accepted")
        assert not path.exists()


def test_tracker_close_failure_becomes_failed_without_losing_metrics():
    class FailingTracker:
        def close(self):
            raise RuntimeError("injected close failure")

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        status, metrics = root / "status.json", root / "metrics.json"
        reservation = {"status": "running", "config_sha256": "a" * 64}
        _strict_atomic_json(status, reservation)
        _strict_atomic_json(metrics, {"score": 0.125})
        try:
            _finish_success(FailingTracker(), status, metrics, reservation)
        except RuntimeError as error:
            _record_failure(status, reservation, error)
        else:
            raise AssertionError("tracker close failure was suppressed")
        assert json.loads(status.read_text())["status"] == "failed"
        assert json.loads(metrics.read_text()) == {"score": 0.125}


def test_timeout_signal_raises_into_common_failure_boundary():
    try:
        _terminate(15, None)
    except TimeoutError as error:
        assert "signal 15" in str(error)
    else:
        raise AssertionError("termination signal was ignored")


def test_corrupted_residual_decomposition_fails_closed():
    mask = torch.ones((1, 1, 4, 4), dtype=torch.float64)
    raw = _payload(torch.zeros((1, 2, 6, 4, 4), dtype=torch.float64), mask)
    candidate = _payload(torch.ones((1, 2, 6, 4, 4), dtype=torch.float64), mask)
    candidate["residual_normalized"] = candidate["residual_normalized"].clone()
    candidate["residual_normalized"][:, :, :, 0, 0] = 0.5
    try:
        _decomposition(raw, candidate)
    except ValueError as error:
        assert "delta" in str(error) or "nullspace" in str(error)
    else:
        raise AssertionError("corrupted decomposition was accepted")


def test_strict_json_rejects_nonfinite_result():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "result.json"
        try:
            _strict_atomic_json(path, {"metric": float("nan")})
        except FloatingPointError:
            pass
        else:
            raise AssertionError("nonfinite JSON was accepted")
        assert not path.exists()
