import tempfile
from pathlib import Path

import torch

from assim_lib.config import load_json
from assim_lib.direct_dynamics_mixed_support_mask_training import (
    MODE,
    _atomic_torch_save,
    _transactional_arm_update,
    _validate_completion,
)


def test_frozen_training_protocol():
    config = load_json("config/experiments/train_direct_dynamics_mixed_support_mask_ab_v1.json")
    assert config["schema_version"] == MODE
    protocol = config["protocol"]
    assert protocol["updates_per_arm"] == 512 and protocol["batch_size"] == 8
    assert protocol["train_years"] == [2016, 2017, 2018, 2019, 2020]
    assert protocol["heldout_year"] == 2021 and protocol["test_2023"] == "closed"
    assert protocol["evaluation_checkpoint"] == "ema_update_0512"
    source = Path("assim_lib/direct_dynamics_mixed_support_mask_training.py").read_text()
    assert "build_matched_models" in source and "masked_cfm_loss" in source
    assert "torch.optim.AdamW" in source
    assert "torch.compile" not in source
    assert '"heldout_values_accessed": False' in source
    assert '"test_2023_accessed": False' in source


def test_training_wrapper_enforces_shared_single_gpu_policy():
    source = Path("scripts/run_direct_dynamics_mixed_support_mask_ab_training.sh").read_text()
    for required in (
        "git status --porcelain --untracked-files=all", ".gpu_job.lock",
        "require_single_gpu_uuid.sh", "for sample in {1..11}", "sleep 30",
        '"$GPU_UTILIZATION" -lt 5', "CLEARML_REQUIRE_ONLINE=1", "7140s",
    ):
        assert required in source


def test_failure_between_optimizer_and_ema_is_not_committed():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    ema = {name: value.detach().clone() for name, value in model.state_dict().items()}
    before_ema = {name: value.clone() for name, value in ema.items()}
    stages = []
    loss = model(torch.ones((1, 2))).square().mean()
    try:
        _transactional_arm_update(
            label="candidate", update=1, model=model, optimizer=optimizer, ema=ema,
            loss=loss, ema_decay=0.99,
            record_stage=lambda label, update, stage: stages.append((label, update, stage)),
            fail_after="optimizer_executed",
        )
    except RuntimeError as error:
        assert "between optimizer and EMA" in str(error)
    else:
        raise AssertionError("injected transactional failure was ignored")
    assert stages == [("candidate", 1, "write_ahead"), ("candidate", 1, "optimizer_executed")]
    assert all(torch.equal(ema[name], before_ema[name]) for name in ema)


def test_reporting_failure_after_commit_preserves_latest_checkpoint():
    with tempfile.TemporaryDirectory() as directory:
        latest = Path(directory) / "latest.pt"
        digest = _atomic_torch_save({"update": 512}, latest)
        try:
            raise RuntimeError("injected reporting failure")
        except RuntimeError as error:
            assert str(error) == "injected reporting failure"
        assert latest.is_file() and len(digest) == 64
        assert torch.load(latest, weights_only=True)["update"] == 512


def test_history_mismatch_blocks_completion():
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    ema = {"candidate": {name: value.clone() for name, value in model.state_dict().items()},
           "control": {name: value.clone() for name, value in model.state_dict().items()}}
    try:
        _validate_completion(
            history=[{"update": 2}], updates=2, checkpoints={}, output_dir=Path("."),
            models={"candidate": model, "control": model}, ema=ema,
            optimizers={"candidate": optimizer, "control": optimizer},
        )
    except RuntimeError as error:
        assert "incomplete or out of order" in str(error)
    else:
        raise AssertionError("mismatched history passed completion")
