import inspect
import json
from pathlib import Path

import pytest
import torch

import assim_lib.direct_dynamics_fine_support_proper_gpu_preflight as preflight
from assim_lib.direct_dynamics_cascade import masked_block_average
from assim_lib.direct_dynamics_cascade_coarse_proper_refinement import _atomic_torch_save
from assim_lib.direct_dynamics_cascade_coarse_fine_boundary_audit import _strict_atomic_json


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config/experiments/preflight_direct_dynamics_fine_support_proper_gpu_v1.json"


def test_actual_preflight_contract_is_zero_update_and_test_closed():
    config = json.loads(CONFIG_PATH.read_text())
    assert config["schema_version"] == "fine_support_proper_actual_gpu_preflight_v1"
    assert config["protocol"] == preflight.EXPECTED_PROTOCOL
    assert config["protocol"]["optimizer_steps"] == 0
    assert config["protocol"]["test_2023"] == "closed"
    assert config["protocol"]["members"] == 4
    assert config["protocol"]["rk4_intervals"] == 32
    assert config["protocol"]["frozen_prefix_intervals"] == 31
    assert config["protocol"]["trainable_terminal_intervals"] == 1
    assert config["source"]["checkpoint"] == "mechanics_update_2048.pth"
    assert all(
        len(value) == 64
        for value in config["source"]["implementation_sha256"].values()
    )


def test_actual_preflight_runner_has_no_optimizer_or_training_step():
    source = inspect.getsource(preflight)
    assert "torch.optim" not in source
    assert "optimizer.step" not in source
    assert '"optimizer_created": False' in source
    assert '"optimizer_steps": 0' in source


def test_launcher_uses_shared_lock_and_five_minute_busy_observation():
    source = (ROOT / "scripts/run_direct_dynamics_fine_support_proper_gpu_preflight.sh").read_text()
    assert ".gpu_job.lock" in source
    assert "require_single_gpu_uuid.sh" in source
    assert "for sample in {1..11}" in source
    assert "sleep 30" in source
    assert '"$GPU_UTILIZATION" -ge 5' in source
    assert "timeout --foreground --signal=TERM --kill-after=60s 1740s" in source


def test_real_stats_canonical_teacher_preserves_zero_atoms_with_coast():
    means = torch.tensor([0.19301218262209952, 0.18969311571248842] * 3)
    stds = torch.tensor([0.36890927421384667, 0.4359687842884708] * 3)
    mask = torch.ones(1, 1, 4, 4, dtype=torch.float32)
    mask[..., 0, 0] = 0.0
    physical = torch.rand(1, 6, 4, 4, generator=torch.Generator().manual_seed(7))
    physical[:, 1::2] *= 2.0
    physical[:, 0::2, :2, :2] = 0.0
    normalized = (physical - means.reshape(1, 6, 1, 1)) / stds.reshape(1, 6, 1, 1)
    coarse_normalized, _ = masked_block_average(normalized, mask)
    truth, teacher, consistency = preflight.canonical_truth_and_teacher_coarse(
        normalized, coarse_normalized, mask, means, stds
    )
    encoded_zero = -means[0] / stds[0]
    selected = (normalized[:, 0:1] == encoded_zero) & (mask > 0)
    assert torch.any(selected)
    assert torch.equal(truth[:, 0:1][selected], torch.zeros_like(truth[:, 0:1][selected]))
    expected, fraction = masked_block_average(truth, mask.double())
    assert torch.equal(teacher, expected)
    assert consistency <= 8 * torch.finfo(torch.float32).eps
    assert torch.any((fraction > 0) & (fraction < 1))


def test_failure_status_retains_both_early_evidence_hashes(tmp_path):
    fixed = tmp_path / "fixed.pth"
    samples = tmp_path / "samples.pth"
    fixed_sha = _atomic_torch_save({"fixed": torch.ones(1)}, fixed)
    samples_sha = _atomic_torch_save({"samples": torch.zeros(1)}, samples)
    status = tmp_path / "status.json"
    _strict_atomic_json(
        status,
        preflight._failure_payload(
            {"optimizer_steps": 0},
            RuntimeError("injected parity failure"),
            fixed_inputs_sha256=fixed_sha,
            step0_samples_sha256=samples_sha,
        ),
    )
    payload = json.loads(status.read_text())
    assert fixed.is_file() and samples.is_file()
    assert payload["status"] == "failed"
    assert payload["fixed_inputs_sha256"] == fixed_sha
    assert payload["step0_samples_sha256"] == samples_sha


def test_failure_after_reservation_is_durable(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": "fine_support_proper_actual_gpu_preflight_v1",
                "protocol": preflight.EXPECTED_PROTOCOL,
            }
        )
    )
    output = tmp_path / "result"
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        preflight,
        "_clean_code_identity",
        lambda *_: (_ for _ in ()).throw(RuntimeError("synthetic failure")),
    )
    with pytest.raises(RuntimeError, match="synthetic failure"):
        preflight.run(config_path, output)
    status = json.loads((output / "status.json").read_text())
    assert status["status"] == "failed"
    assert status["error_type"] == "RuntimeError"
    assert status["error"] == "synthetic failure"
    assert status["optimizer_steps"] == 0
