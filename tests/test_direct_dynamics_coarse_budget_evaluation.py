from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest
import torch

import assim_lib.direct_dynamics_coarse_budget_evaluation as evaluation
from assim_lib.direct_dynamics_coarse_budget_evaluation import (
    CHECKPOINT_KEYS,
    _finalize_failure,
    _finalize_success,
    _standardize_physical,
    _training_anchored_candidate,
    _validate_config,
    load_checkpoint_cpu_gate,
    run,
)
from assim_lib.direct_dynamics_coarse_budget_training import EXPECTED_PROTOCOL


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source() -> dict:
    return {
        "forecast_contract_sha256": "f" * 64,
        "coarse": {
            "checkpoint": "coarse.pth",
            "checkpoint_sha256": "a" * 64,
            "files_sha256": {"coarse.pth": "a" * 64},
        },
        "fine": {
            "checkpoint": "fine.pth",
            "checkpoint_sha256": "b" * 64,
            "files_sha256": {"fine.pth": "b" * 64},
        },
    }


def _record(source: dict) -> dict:
    return {
        "completed_updates": 64,
        "protocol": copy.deepcopy(EXPECTED_PROTOCOL),
        "train_indices": list(range(64)),
        "code_identity": {"git_commit": "c" * 40, "git_worktree": "clean"},
        "actual_admission": {"metrics_sha256": "d" * 64},
        "effective_forecast_contract_sha256": source["forecast_contract_sha256"],
        "source_files_sha256": {
            "coarse": source["coarse"]["files_sha256"],
            "fine": source["fine"]["files_sha256"],
        },
    }


def _payload(record: dict, source: dict) -> dict:
    payload = {
        "model": {"weight": torch.ones(2, 3), "bias": torch.zeros(2)},
        "optimizer": {
            "state": {
                0: {
                    "step": torch.tensor(64.0),
                    "exp_avg": torch.zeros(2, 3),
                    "exp_avg_sq": torch.ones(2, 3),
                },
                1: {
                    "step": torch.tensor(64.0),
                    "exp_avg": torch.zeros(2),
                    "exp_avg_sq": torch.ones(2),
                },
            },
            "param_groups": [
                {
                    "params": [0, 1],
                    "lr": EXPECTED_PROTOCOL["learning_rate"],
                    "weight_decay": EXPECTED_PROTOCOL["weight_decay"],
                    "amsgrad": False,
                }
            ],
        },
        "source": copy.deepcopy(source),
    }
    for key in (
        "completed_updates",
        "protocol",
        "train_indices",
        "code_identity",
        "actual_admission",
    ):
        payload[key] = copy.deepcopy(record[key])
    assert set(payload) == CHECKPOINT_KEYS
    return payload


def _write_checkpoint(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "terminal.pth"
    torch.save(payload, path)
    return path


def test_checkpoint_cpu_gate_accepts_exact_finite_adamw64(tmp_path: Path) -> None:
    source = _source()
    record = _record(source)
    path = _write_checkpoint(tmp_path, _payload(record, source))
    loaded, report = load_checkpoint_cpu_gate(path, _sha256(path), record, source)
    assert set(loaded) == CHECKPOINT_KEYS
    assert report["optimizer_step_unique"] == [64]
    assert report["model_tensor_count"] == 2
    assert report["all_model_and_optimizer_tensors_finite"] is True


@pytest.mark.parametrize("failure", ["step", "model_nan", "source", "extra_key"])
def test_checkpoint_cpu_gate_fails_closed(tmp_path: Path, failure: str) -> None:
    source = _source()
    record = _record(source)
    payload = _payload(record, source)
    if failure == "step":
        payload["optimizer"]["state"][0]["step"] = torch.tensor(63.0)
    elif failure == "model_nan":
        payload["model"]["weight"][0, 0] = torch.nan
    elif failure == "source":
        payload["source"]["forecast_contract_sha256"] = "0" * 64
    else:
        payload["unreviewed"] = True
    path = _write_checkpoint(tmp_path, payload)
    with pytest.raises((ValueError, FloatingPointError)):
        load_checkpoint_cpu_gate(path, _sha256(path), record, source)


def test_validation_contract_rejects_wrong_panel_or_solver() -> None:
    config = {
        "schema_version": "coarse_budget_paired_validation_v1",
        "split": "valid",
        "test_2023": "closed",
        "cases": 12,
        "members": 8,
        "coarse_rk4_timepoints": 17,
        "fine_rk4_timepoints": 33,
        "case_indices": list(range(12)),
        "case_ids": [f"case-{index}" for index in range(12)],
        "decision_gate": {
            "primary": "case_equal_six_channel_train_standardized_fair_crps",
            "bootstrap_draws": 100000,
        },
        "labels": {
            "control": "control",
            "candidate": "candidate",
            "raw_secondary": "raw",
        },
    }
    _validate_config(config)
    config["fine_rk4_timepoints"] = 17
    with pytest.raises(ValueError, match="RK4"):
        _validate_config(config)


def test_training_anchor_preserves_exact_fp32_arithmetic() -> None:
    production = torch.tensor([1.0, 0.0], dtype=torch.float32)
    control = torch.tensor([1.0, torch.nextafter(torch.tensor(0.0), torch.tensor(1.0))])
    candidate = torch.tensor([-1.0e-8, -torch.nextafter(torch.tensor(0.0), torch.tensor(1.0))])
    expected = production.detach() + (candidate - control.detach())
    actual = _training_anchored_candidate(production, control, candidate)
    assert torch.equal(actual, expected)
    assert not torch.equal(actual, candidate)


def test_physical_standardization_stays_fp64_without_reencoding() -> None:
    physical = torch.tensor([[[[[1.00000003]], [[2.0]], [[3.0]], [[4.0]], [[5.0]], [[6.0]]]]])
    stds = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=torch.float64)
    standardized = _standardize_physical(physical.double(), stds)
    assert standardized.dtype == torch.float64
    assert torch.equal(standardized, physical.double() / stds.reshape(1, 1, 6, 1, 1))


def test_failure_finalization_marks_tracker_and_preserves_evidence(tmp_path: Path) -> None:
    class Task:
        def __init__(self) -> None:
            self.failed = False

        def mark_failed(self, **_kwargs) -> None:
            self.failed = True

    class Tracker:
        def __init__(self) -> None:
            self.task = Task()
            self.closed = False

        def close(self) -> None:
            self.closed = True

    tracker = Tracker()
    status = tmp_path / "status.json"
    manifest = {"artifacts": {"fixed_inputs": {"sha256": "a" * 64}}}
    (tmp_path / "evidence_manifest.json").write_text(
        __import__("json").dumps(manifest)
    )
    lifecycle = {
        "output": tmp_path,
        "status_path": status,
        "reservation": {"status": "reserved", "optimizer_steps": 0},
        "tracker": tracker,
        "evidence_manifest": manifest,
    }
    failure = _finalize_failure(lifecycle, RuntimeError("injected"))
    assert failure["status"] == "failed"
    assert failure["evidence_artifacts"] == manifest["artifacts"]
    assert tracker.task.failed is True
    assert tracker.closed is True
    assert '"status": "failed"' in status.read_text()


def test_failure_manifest_hash_error_does_not_mask_primary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = {"artifacts": {}}
    manifest_path = tmp_path / "evidence_manifest.json"
    manifest_path.write_text("{}")

    def fail_hash(path: Path) -> str:
        if path == manifest_path:
            raise OSError("injected manifest read failure")
        return "b" * 64

    monkeypatch.setattr(evaluation, "_sha256", fail_hash)
    lifecycle = {
        "output": tmp_path,
        "status_path": tmp_path / "status.json",
        "reservation": {"status": "reserved"},
        "tracker": None,
        "evidence_manifest": manifest,
    }
    failure = _finalize_failure(lifecycle, RuntimeError("primary"))
    assert failure["error"] == "primary"
    assert any("evidence_manifest_read" in row for row in failure["cleanup_errors"])


def test_existing_output_is_never_modified_on_reservation_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "owner.txt"
    marker.write_text("pre-existing")
    monkeypatch.setenv("CLEARML_REQUIRE_ONLINE", "1")
    with pytest.raises(FileExistsError):
        run(tmp_path / "unused.json", output)
    assert marker.read_text() == "pre-existing"
    assert sorted(path.name for path in output.iterdir()) == ["owner.txt"]


def test_success_status_does_not_mutate_uploaded_numerical_bytes(tmp_path: Path) -> None:
    result_path = tmp_path / "result.json"
    result_path.write_bytes(b'{"metric":0.125}\n')
    original = result_path.read_bytes()
    digest = _sha256(result_path)
    manifest = tmp_path / "evidence_manifest.json"
    manifest.write_text("{}")
    status = _finalize_success(
        tmp_path / "status.json",
        {"status": "reserved"},
        result_path,
        digest,
        manifest,
        "clearml-task",
    )
    assert result_path.read_bytes() == original
    assert status["result_sha256"] == digest
