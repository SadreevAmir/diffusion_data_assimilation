import json
import tempfile
from pathlib import Path

import torch

from assim_lib.direct_dynamics_mixed_support_mask_gpu_preflight import _finalize_failure
from assim_lib.direct_dynamics_mixed_support_mask_gpu_preflight import _atomic_torch_save
from assim_lib.direct_dynamics_mixed_support_mask_runner import sample_masks


def test_preflight_is_zero_update_and_train_only():
    source = Path("assim_lib/direct_dynamics_mixed_support_mask_gpu_preflight.py").read_text()
    assert '"optimizer_created": False' in source
    assert '"optimizer_steps": 0' in source
    assert 'build_dataset(dataset_config, "train")' in source
    assert '"test_2023_accessed": False' in source
    assert "torch.optim" not in source
    assert "fixed_inputs.pt" in source
    assert "prediction.pt" in source
    assert "sample.pt" in source
    assert "terminal_latent.pt" in source
    assert "parameters changed during zero-update preflight" in source
    assert "CLEARML_REQUIRE_ONLINE" in source
    assert "signal.SIGTERM" in source and "signal.SIGINT" in source


def test_wrapper_enforces_single_gpu_policy_and_exact_commit():
    source = Path("scripts/run_direct_dynamics_mixed_support_mask_gpu_preflight.sh").read_text()
    assert "IDEA_F1_EXPECTED_COMMIT" in source
    assert "git status --porcelain --untracked-files=all" in source
    assert "require_single_gpu_uuid.sh" in source
    assert "GPU-40ff1cbb-07cc-992e-23cb-8fe181972b76" in source
    assert "for sample in {1..11}" in source and "sleep 30" in source
    assert '"$GPU_UTILIZATION" -lt 5' in source
    assert "flock -n 9" in source
    assert "570s" in source and "--kill-after=30s" in source
    assert "CLEARML_REQUIRE_ONLINE=1" in source


def test_failure_injection_boundary_preserves_primary_error_and_closes_tracker():
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
            RuntimeError("injected preflight failure after fixed_evidence"),
            tracker,
            {"fixed_inputs.pt": "abc"},
        )
        payload = json.loads(status.read_text())
    assert tracker.closed
    assert payload["error"] == "injected preflight failure after fixed_evidence"
    assert payload["evidence_sha256"]["fixed_inputs.pt"] == "abc"
    assert any("clearml_mark_failed" in item for item in payload["cleanup_errors"])


def test_nonfinite_sampling_latent_is_saved_before_decoder_preserves_primary_error():
    class NaNVelocity(torch.nn.Module):
        def forward(self, state, *_args):
            return torch.full_like(state, float("nan"))

    condition = torch.zeros((1, 15, 4, 4))
    d0 = torch.zeros((1, 1, 4, 4))
    valid = torch.ones_like(d0)
    noise = torch.zeros((1, 3, 4, 4))
    with tempfile.TemporaryDirectory() as directory:
        terminal = Path(directory) / "terminal.pt"

        def save(value):
            _atomic_torch_save({"terminal_latent": value}, terminal)

        try:
            sample_masks(
                NaNVelocity(), condition=condition, d0_occurrence=d0,
                valid=valid, noise=noise, steps=1, before_decode=save,
            )
        except ValueError as error:
            primary = str(error)
        else:
            raise AssertionError("nonfinite terminal latent reached the decoder unnoticed")
        saved = torch.load(terminal, weights_only=True)["terminal_latent"]
    assert not torch.isfinite(saved).all()
    assert primary == "binary latent must be finite"
