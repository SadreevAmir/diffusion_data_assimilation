import inspect
import json
from pathlib import Path

import pytest
import torch

import assim_lib.direct_dynamics_fine_support_proper_training as training
from assim_lib.direct_dynamics_cascade_coarse_proper_refinement import _atomic_torch_save


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/experiments/train_direct_dynamics_fine_support_proper_64_v1.json"


def test_training_contract_is_exact_bounded_protocol():
    config = json.loads(CONFIG.read_text())
    assert config["schema_version"] == "fine_support_proper_training_v1"
    assert config["protocol"] == training.EXPECTED_PROTOCOL
    assert config["protocol"]["updates"] == 64
    assert config["protocol"]["members"] == 4
    assert config["protocol"]["learning_rate"] == 1e-5
    assert config["protocol"]["test_2023"] == "closed"
    assert not config["protocol"]["activation_checkpointing"]
    assert not config["protocol"]["additional_ema"]


def test_each_update_is_checkpointed_before_progress_reporting():
    source = inspect.getsource(training.run)
    loop = source[source.index("for update in range") :]
    attempted = loop.index("attempted_optimizer_steps = update")
    write_ahead = loop.index("_write_step_pending(")
    step = loop.index("optimizer.step()")
    executed = loop.index("executed_optimizer_steps = update")
    checkpoint = loop.index("next_resume_sha = _atomic_torch_save")
    committed = loop.index("completed_updates = update")
    progress = loop.index("_strict_atomic_json(\n                progress_path")
    report = loop.index("tracker.report_scalar")
    assert attempted < write_ahead < step < executed < checkpoint < committed < progress < report
    assert "gradient_clipping" not in loop
    assert "additional_ema" not in loop


def test_resume_checkpoint_roundtrip_binds_update_and_optimizer(tmp_path):
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.0)
    loss = model(torch.ones(1, 3)).square().sum()
    loss.backward()
    optimizer.step()
    identity = {"git_commit": "a" * 40, "git_worktree": "clean"}
    history = [
        {
            "update": 1,
            "train_index": 7,
            "noise_seed": training.EXPECTED_PROTOCOL["seed"] + 1,
            "objective": float(loss.detach()),
        }
    ]
    started = 1000.0
    deadline = started + 60 * training.EXPECTED_PROTOCOL["max_gpu_minutes"]
    path = tmp_path / "resume.pth"
    sha = _atomic_torch_save(
        training._resume_payload(
            model,
            optimizer,
            completed_updates=1,
            history=history,
            config_sha256="b" * 64,
            code_identity=identity,
            source_checkpoint_sha256="c" * 64,
            training_started_unix=started,
            deadline_unix=deadline,
        ),
        path,
    )
    restored_model = torch.nn.Linear(3, 2)
    restored_optimizer = torch.optim.AdamW(
        restored_model.parameters(), lr=1e-5, weight_decay=0.0
    )
    completed, restored_history, restored_sha = training._restore_resume(
        path,
        restored_model,
        restored_optimizer,
        config_sha256="b" * 64,
        code_identity=identity,
        source_checkpoint_sha256="c" * 64,
        expected_sha256=sha,
        expected_completed_updates=1,
        expected_indices=[7],
        training_started_unix=started,
        deadline_unix=deadline,
        device=torch.device("cpu"),
    )
    assert completed == 1
    assert restored_history == history
    assert restored_sha == sha
    for expected, actual in zip(model.parameters(), restored_model.parameters()):
        assert torch.equal(expected, actual)
    assert restored_optimizer.state_dict()["state"]


def test_resume_rejects_identity_mismatch(tmp_path):
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    path = tmp_path / "resume.pth"
    started = 1000.0
    deadline = started + 60 * training.EXPECTED_PROTOCOL["max_gpu_minutes"]
    sha = _atomic_torch_save(
        training._resume_payload(
            model,
            optimizer,
            completed_updates=1,
            history=[
                {
                    "update": 1,
                    "train_index": 7,
                    "noise_seed": training.EXPECTED_PROTOCOL["seed"] + 1,
                }
            ],
            config_sha256="b" * 64,
            code_identity={"git_commit": "a" * 40, "git_worktree": "clean"},
            source_checkpoint_sha256="c" * 64,
            training_started_unix=started,
            deadline_unix=deadline,
        ),
        path,
    )
    with pytest.raises(ValueError, match="identity differs"):
        training._restore_resume(
            path,
            model,
            optimizer,
            config_sha256="d" * 64,
            code_identity={"git_commit": "a" * 40, "git_worktree": "clean"},
            source_checkpoint_sha256="c" * 64,
            expected_sha256=sha,
            expected_completed_updates=1,
            expected_indices=[7],
            training_started_unix=started,
            deadline_unix=deadline,
            device=torch.device("cpu"),
        )


def test_write_ahead_makes_a_crashed_pending_step_non_resumable(tmp_path, monkeypatch):
    output = tmp_path / "run"
    output.mkdir()
    (output / "resume_latest.pth").write_bytes(b"forensic evidence only")
    started = 1000.0
    reservation = {
        "resume": False,
        "training_started_unix": started,
        "deadline_unix": started
        + 60 * training.EXPECTED_PROTOCOL["max_gpu_minutes"],
    }
    training._write_step_pending(
        output / "status.json",
        reservation,
        update=4,
        committed_updates=3,
        latest_resume_sha256="a" * 64,
        code_identity={"git_commit": "a" * 40, "git_worktree": "clean"},
        clearml_task_id="task",
    )
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    model(torch.ones(1, 1)).square().sum().backward()
    optimizer.step()
    # Simulate SIGKILL here: no checkpoint/status commit follows the real step.
    persisted = json.loads((output / "status.json").read_text())
    assert persisted["status"] == "step_pending"
    assert persisted["attempted_optimizer_steps"] == 4
    assert persisted["executed_optimizer_steps"] == 3
    assert persisted["completed_updates"] == 3
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": "fine_support_proper_training_v1",
                "protocol": training.EXPECTED_PROTOCOL,
            }
        )
    )
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(training.time, "time", lambda: started + 1)
    with pytest.raises(ValueError, match="ambiguous uncommitted optimizer step"):
        training.run(config_path, output, resume=True)


def test_expired_resume_is_rejected_before_cuda_access(tmp_path, monkeypatch):
    output = tmp_path / "run"
    output.mkdir()
    (output / "resume_latest.pth").write_bytes(b"bound later")
    started = 1000.0
    (output / "status.json").write_text(
        json.dumps(
            {
                "status": "training",
                "completed_updates": 3,
                "optimizer_steps": 3,
                "attempted_optimizer_steps": 3,
                "executed_optimizer_steps": 3,
                "latest_resume_sha256": "a" * 64,
                "training_started_unix": started,
                "deadline_unix": started
                + 60 * training.EXPECTED_PROTOCOL["max_gpu_minutes"],
            }
        )
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": "fine_support_proper_training_v1",
                "protocol": training.EXPECTED_PROTOCOL,
            }
        )
    )
    monkeypatch.setattr(
        torch.cuda,
        "device_count",
        lambda: (_ for _ in ()).throw(AssertionError("CUDA must not be touched")),
    )
    monkeypatch.setattr(
        training.time,
        "time",
        lambda: started + 60 * training.EXPECTED_PROTOCOL["max_gpu_minutes"] + 1,
    )
    with pytest.raises(TimeoutError, match="already exhausted"):
        training.run(config_path, output, resume=True)


def test_failure_before_first_update_is_durable(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": "fine_support_proper_training_v1",
                "protocol": training.EXPECTED_PROTOCOL,
            }
        )
    )
    output = tmp_path / "run"
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        training,
        "_clean_code_identity",
        lambda *_: (_ for _ in ()).throw(RuntimeError("synthetic failure")),
    )
    with pytest.raises(RuntimeError, match="synthetic failure"):
        training.run(config_path, output)
    status = json.loads((output / "status.json").read_text())
    assert status["status"] == "failed"
    assert status["completed_updates"] == 0
    assert status["optimizer_steps"] == 0
    assert status["attempted_optimizer_steps"] == 0
    assert status["executed_optimizer_steps"] == 0


def test_failure_status_error_does_not_replace_primary_exception(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": "fine_support_proper_training_v1",
                "protocol": training.EXPECTED_PROTOCOL,
            }
        )
    )
    output = tmp_path / "run"
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        training,
        "_clean_code_identity",
        lambda *_: (_ for _ in ()).throw(RuntimeError("primary failure")),
    )
    original = training._strict_atomic_json
    calls = 0

    def fail_second_write(path, payload):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("secondary status failure")
        return original(path, payload)

    monkeypatch.setattr(training, "_strict_atomic_json", fail_second_write)
    with pytest.raises(RuntimeError, match="primary failure"):
        training.run(config_path, output)


def test_launcher_has_fresh_resume_and_shared_gpu_safety():
    source = (
        ROOT / "scripts/run_direct_dynamics_fine_support_proper_training.sh"
    ).read_text()
    assert "FINE_SUPPORT_TRAIN_ATTEMPT_ID" in source
    assert '"$MODE" != "fresh"' in source
    assert 'resume_arg=(--resume)' in source
    assert ".gpu_job.lock" in source
    assert "require_single_gpu_uuid.sh" in source
    assert "for sample in {1..11}" in source and "sleep 30" in source
    assert '"$GPU_UTILIZATION" -ge 5' in source
    assert "remaining_budget_seconds" in source
    assert 'term_seconds=1740' in source and 'kill_after_seconds=60' in source
    assert 'remaining - kill_after_seconds' in source


def test_failure_lifecycle_marks_clearml_failed_without_masking_primary():
    source = inspect.getsource(training.run)
    failure = source[source.index("except BaseException as error:") :]
    status = failure.index("_strict_atomic_json(status_path, failure)")
    marked = failure.index("tracker.task.mark_failed")
    reraised = failure.index("        raise\n")
    assert status < marked < reraised
    assert "except BaseException:\n                pass" in failure
