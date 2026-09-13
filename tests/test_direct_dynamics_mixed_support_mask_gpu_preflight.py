from pathlib import Path


def test_preflight_is_zero_update_and_train_only():
    source = Path("assim_lib/direct_dynamics_mixed_support_mask_gpu_preflight.py").read_text()
    assert '"optimizer_created": False' in source
    assert '"optimizer_steps": 0' in source
    assert 'build_dataset(dataset_config, "train")' in source
    assert '"test_2023_accessed": False' in source
    assert "torch.optim" not in source


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
