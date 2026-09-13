import json
from pathlib import Path

from assim_lib.direct_dynamics_geometry_cfm_training import _sha256
from assim_lib.direct_dynamics_geometry_cfm_transactional_training import (
    _close_and_mark_success,
    _commit_update_evidence,
    _record_mutation_boundary,
    _validate_completion,
    _write_ahead_update,
)


def test_update_lifecycle_is_write_ahead_then_mutated_then_durable(tmp_path):
    status = tmp_path / "status.json"
    history = tmp_path / "control" / "history.json"
    history.parent.mkdir()
    committed = {"control": 0, "treatment": 0}
    attempt = {
        "case_ids": ["case"],
        "timesteps_sha256": "t",
        "noise_seed": 1,
        "dropout_seed": 2,
    }
    _write_ahead_update(status, "control", 1, committed, attempt)
    payload = json.loads(status.read_text())
    assert payload["optimizer_mutation_may_have_occurred"] is False
    assert payload["committed_updates_by_arm"]["control"] == 0

    _record_mutation_boundary(status, "control", 1, committed)
    payload = json.loads(status.read_text())
    assert payload["optimizer_mutation_may_have_occurred"] is True
    assert payload["committed_updates_by_arm"]["control"] == 0

    record = {
        "update": 1,
        "case_ids": ["case"],
        "attempt": attempt,
        "input_sha256": {"truth": "a", "noise": "b", "timesteps": "c"},
    }
    _commit_update_evidence(
        status, history, "control", [record], committed, {"control": {}, "treatment": {}}
    )
    payload = json.loads(status.read_text())
    assert payload["optimizer_mutation_may_have_occurred"] is False
    assert payload["committed_updates_by_arm"]["control"] == 1
    assert json.loads(history.read_text()) == [record]


def _make_checkpoint(root: Path, arm: str, update: int):
    checkpoint = root / arm / f"update_{update:04d}"
    checkpoint.mkdir(parents=True)
    names = {
        "model": "model.pth",
        "ema_model": "ema_model.pth",
        "optimizer_recovery": "optimizer_recovery.pth",
    }
    result = {}
    for key, name in names.items():
        path = checkpoint / name
        path.write_bytes(f"{arm}-{update}-{key}".encode())
        result[key] = _sha256(path)
    return result


def test_completion_requires_exact_sequences_pairing_and_checkpoint_shas(tmp_path):
    attempt = {
        "case_ids": ["case"],
        "timesteps_sha256": "t",
        "noise_seed": 1,
        "dropout_seed": 2,
    }
    record = {
        "update": 1,
        "case_ids": ["case"],
        "attempt": attempt,
        "input_sha256": {"truth": "a", "noise": "b", "timesteps": "c"},
    }
    histories = {arm: [dict(record)] for arm in ("control", "treatment")}
    checkpoints = {
        arm: {"1": _make_checkpoint(tmp_path, arm, 1)}
        for arm in ("control", "treatment")
    }
    protocol = {"updates_per_arm": 1, "checkpoint_updates": [1]}
    _validate_completion(
        tmp_path, histories, checkpoints, protocol, {"control": 1, "treatment": 1}
    )
    histories["treatment"][0]["attempt"] = dict(attempt, noise_seed=99)
    try:
        _validate_completion(
            tmp_path, histories, checkpoints, protocol, {"control": 1, "treatment": 1}
        )
    except RuntimeError as error:
        assert "attempt" in str(error)
    else:
        raise AssertionError("divergent arm schedule was accepted")


def test_clearml_close_failure_never_writes_terminal_success(tmp_path):
    class Tracker:
        def close(self):
            raise RuntimeError("close failed")

    result_path = tmp_path / "result.json"
    status_path = tmp_path / "status.json"
    try:
        _close_and_mark_success(
            result_path, status_path, Tracker(), {"schema_version": "test"}
        )
    except RuntimeError as error:
        assert str(error) == "close failed"
    else:
        raise AssertionError("ClearML close failure was swallowed")
    assert json.loads(result_path.read_text())["status"] == "completed_pending_clearml_close"
    assert json.loads(status_path.read_text())["status"] == "completed_pending_clearml_close"


def test_clearml_reporting_is_after_durable_update_commit_in_production_loop():
    source = Path(
        "assim_lib/direct_dynamics_geometry_cfm_transactional_training.py"
    ).read_text()
    commit_position = source.index("_commit_update_evidence(\n", source.index("def run_training"))
    report_position = source.index("tracker.report_scalar(\n", commit_position)
    assert commit_position < report_position


def test_wrapper_uses_transactional_module_only_for_training():
    source = Path("scripts/run_direct_dynamics_geometry_full_cfm_ab.sh").read_text()
    assert "assim_lib.direct_dynamics_geometry_cfm_transactional_training" in source
    assert "--mode preflight" in source
