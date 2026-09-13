from pathlib import Path

from assim_lib.config import load_json
from assim_lib.direct_dynamics_mixed_support_mask_training import MODE, _validate_contract


def test_frozen_training_protocol():
    config = load_json("config/experiments/train_direct_dynamics_mixed_support_mask_ab_v1.json")
    assert config["schema_version"] == MODE
    protocol = config["protocol"]
    assert protocol["updates_per_arm"] == 512 and protocol["batch_size"] == 8
    assert protocol["train_years"] == [2016, 2017, 2018, 2019, 2020]
    assert protocol["heldout_year"] == 2021 and protocol["test_2023"] == "closed"
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
