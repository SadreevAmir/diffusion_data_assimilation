from __future__ import annotations

import copy
import json
from pathlib import Path

import torch

from assim_lib.direct_dynamics_geometry_training import (
    _atomic_torch_save,
    _check_protocol,
    _durable_commit_update,
    _durable_optimizer_step,
    _failure_status,
    _sha256,
    _validate_prefix_cache_binding,
)
from assim_lib.trainer import _atomic_json


def _protocol() -> dict:
    path = Path("config/experiments/train_direct_dynamics_geometry_objective_ab_v1.json")
    return json.loads(path.read_text())["protocol"]


def test_reviewed_training_protocol_is_accepted_exactly():
    _check_protocol(_protocol())


def test_training_protocol_rejects_zero_geometry_weight_and_extra_keys():
    changed = copy.deepcopy(_protocol())
    changed["geometry_weight"] = 0.0
    try:
        _check_protocol(changed)
    except ValueError as error:
        assert "geometry_weight" in str(error)
    else:
        raise AssertionError("zero geometry weight was accepted")

    changed = copy.deepcopy(_protocol())
    changed["unreviewed"] = True
    try:
        _check_protocol(changed)
    except ValueError as error:
        assert "unreviewed" in str(error)
    else:
        raise AssertionError("extra protocol key was accepted")


def test_prefix_cache_rejects_changed_input_and_self_consistent_replacement(tmp_path):
    prefix_path = tmp_path / "prefix.pth"
    _atomic_torch_save({"prefix": torch.ones(2)}, prefix_path)
    input_sha = {"condition": "condition-a", "truth": "truth-a"}
    code = {"suffix_sha256": "code-a"}
    payload = {
        "train_index": 17,
        "noise_seed": 99,
        "input_sha256": input_sha,
        "source_checkpoint_sha256": "source-a",
        "code_identity": code,
        "prefix_sha256": "prefix-a",
    }
    manifest = {
        "update": 3,
        "train_index": 17,
        "noise_seed": 99,
        "input_sha256": input_sha,
        "prefix_sha256": "prefix-a",
        "file_sha256": _sha256(prefix_path),
    }
    _validate_prefix_cache_binding(
        payload=payload,
        manifest_entry=manifest,
        prefix_path=prefix_path,
        update=3,
        train_index=17,
        noise_seed=99,
        input_sha256=input_sha,
        source_checkpoint_sha256="source-a",
        code_identity=code,
    )
    changed_input = {**input_sha, "condition": "condition-b"}
    try:
        _validate_prefix_cache_binding(
            payload=payload,
            manifest_entry=manifest,
            prefix_path=prefix_path,
            update=3,
            train_index=17,
            noise_seed=99,
            input_sha256=changed_input,
            source_checkpoint_sha256="source-a",
            code_identity=code,
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("changed treatment condition was accepted")

    _atomic_torch_save({"prefix": torch.zeros(2), "prefix_sha256": "replacement"}, prefix_path)
    replaced_payload = {**payload, "prefix_sha256": "replacement"}
    replaced_manifest = {**manifest, "prefix_sha256": "replacement"}
    try:
        _validate_prefix_cache_binding(
            payload=replaced_payload,
            manifest_entry=replaced_manifest,
            prefix_path=prefix_path,
            update=3,
            train_index=17,
            noise_seed=99,
            input_sha256=input_sha,
            source_checkpoint_sha256="source-a",
            code_identity=code,
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("cache replacement without external file binding was accepted")


def test_real_adamw_step_and_reporting_failure_preserve_counters_and_checkpoint(tmp_path):
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=1e-2, weight_decay=0.0)
    parameter.grad = torch.tensor(0.5)
    status_path = tmp_path / "status.json"
    history_path = tmp_path / "control_history.json"
    lifecycle = {
        "phase": "arm_initializing",
        "arm": "control",
        "attempted_update": 0,
        "executed_update": 0,
        "committed_update": 0,
    }
    latest = {}
    context = {"completed_arms": [], "latest_checkpoints": latest, "clearml_task_id": "tiny"}
    before = parameter.detach().clone()
    _durable_optimizer_step(
        optimizer,
        status_path,
        lifecycle,
        context,
        update=256,
        committed_updates=255,
        train_index=123,
        noise_seed=456,
    )
    assert not torch.equal(before, parameter.detach())
    executed = json.loads(status_path.read_text())
    assert executed["phase"] == "step_executed_uncommitted"
    assert executed["attempted_update"] == executed["executed_update"] == 256
    assert executed["committed_update"] == 255

    checkpoint_path = tmp_path / "control_latest_checkpoint.pth"
    _atomic_torch_save({"model": parameter.detach(), "completed_updates": 256}, checkpoint_path)
    latest["control"] = {
        "path": str(checkpoint_path),
        "sha256": _sha256(checkpoint_path),
        "completed_updates": 256,
    }
    history = [{"update": value} for value in range(1, 257)]
    _durable_commit_update(
        status_path=status_path,
        history_path=history_path,
        history=history,
        lifecycle=lifecycle,
        status_context=context,
        update=256,
    )
    reporting_error = RuntimeError("simulated ClearML reporting failure")
    failure = _failure_status(lifecycle, {"control": history}, latest, {}, reporting_error)
    _atomic_json(status_path, failure)
    terminal = json.loads(status_path.read_text())
    assert terminal["phase"] == "update_committed"
    assert terminal["attempted_update"] == terminal["executed_update"] == 256
    assert terminal["committed_update"] == 256
    assert terminal["completed_updates"] == {"control": 256}
    assert terminal["latest_checkpoints"]["control"] == latest["control"]
    assert _sha256(checkpoint_path) == latest["control"]["sha256"]
