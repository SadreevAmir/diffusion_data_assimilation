import json
from pathlib import Path

from assim_lib.direct_dynamics_coarse_budget_actual_gpu_preflight import (
    EXPECTED_PROTOCOL,
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
