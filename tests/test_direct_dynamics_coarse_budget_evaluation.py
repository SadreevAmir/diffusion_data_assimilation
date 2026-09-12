from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest
import torch

from assim_lib.direct_dynamics_coarse_budget_evaluation import (
    CHECKPOINT_KEYS,
    _validate_config,
    load_checkpoint_cpu_gate,
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
    }
    _validate_config(config)
    config["fine_rk4_timepoints"] = 17
    with pytest.raises(ValueError, match="RK4"):
        _validate_config(config)
