import json
import tempfile
from pathlib import Path

import torch

from assim_lib.direct_dynamics_coarse_budget_actual_gpu_preflight import (
    EXPECTED_PROTOCOL,
    _finalize_failure,
    _require_exact_coarse_replay,
    _validate_config,
)


ROOT = Path(__file__).resolve().parents[1]


def test_actual_preflight_contract_is_zero_update_and_uses_proved_sources():
    path = (
        ROOT
        / "config/experiments/preflight_direct_dynamics_coarse_budget_actual_gpu_v1.json"
    )
    config = json.loads(path.read_text(encoding="utf-8"))
    _validate_config(config)
    assert config["protocol"] == EXPECTED_PROTOCOL
    assert config["optimizer_steps"] == 0
    assert config["test_2023"] == "closed"
    assert config["source"]["coarse"]["checkpoint"] == "ema_coarse_update_9711.pth"
    assert config["source"]["fine"]["checkpoint"] == "mechanics_update_2048.pth"


def test_actual_preflight_runner_has_no_optimizer_or_test_dataset_path():
    source = (
        ROOT
        / "assim_lib/direct_dynamics_coarse_budget_actual_gpu_preflight.py"
    ).read_text(encoding="utf-8")
    assert "torch.optim" not in source
    assert "build_dataset(data_config, split=\"train\")" in source
    assert "split=\"test\"" not in source
    assert '"fine_resampled_for_candidate": False' in source
    assert "torch.no_grad(), torch.autocast(" in source
    assert "production_samples.pth" in source
    assert "candidate_control_coarse.pth" in source


def test_actual_preflight_rejects_even_small_nonexact_coarse_replay():
    production = torch.ones((1, 4, 6, 2, 2), dtype=torch.float32)
    control = production.clone()
    control[0, 0, 0, 0, 0] = torch.nextafter(
        control[0, 0, 0, 0, 0], torch.tensor(2.0)
    )
    try:
        _require_exact_coarse_replay(control, production)
    except RuntimeError as error:
        assert "not exact production EMA9711" in str(error)
    else:
        raise AssertionError("nonexact production coarse replay was accepted")


def test_failure_lifecycle_closes_tracker_after_mark_failed_error():
    class FailingTask:
        def mark_failed(self, **_kwargs):
            raise RuntimeError("injected mark failure")

    class Tracker:
        def __init__(self):
            self.task = FailingTask()
            self.closed = False

        def close(self):
            self.closed = True

    tracker = Tracker()
    with tempfile.TemporaryDirectory() as directory:
        status = Path(directory) / "status.json"
        _finalize_failure(
            status,
            {"optimizer_steps": 0},
            ValueError("primary failure"),
            tracker,
            {"production_samples.pth": "abc"},
        )
        payload = json.loads(status.read_text(encoding="utf-8"))
    assert tracker.closed
    assert payload["error"] == "primary failure"
    assert payload["evidence_sha256"]["production_samples.pth"] == "abc"
    assert any("clearml_mark_failed" in item for item in payload["cleanup_errors"])


def test_gpu_launcher_reuses_shared_admission_envelope():
    script = (
        ROOT / "scripts/run_direct_dynamics_coarse_budget_actual_gpu_preflight.sh"
    ).read_text(encoding="utf-8")
    for required in (
        ".gpu_job.lock",
        "require_single_gpu_uuid.sh",
        "sleep 30",
        "CLEARML_REQUIRE_ONLINE=1",
        "timeout --foreground --signal=TERM --kill-after=60s 1740s",
    ):
        assert required in script
