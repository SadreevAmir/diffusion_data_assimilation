import copy
import inspect
import json
from pathlib import Path

import torch

import assim_lib.direct_dynamics_fine_support_proper_evaluation as evaluation
from assim_lib.direct_dynamics_cascade_fine import teacher_coarse_condition
from assim_lib.direct_dynamics_cascade_fine_colored import (
    ColoredVariancePreconditionedFineCascadeSampler,
)
from assim_lib.direct_dynamics_fine_support_proper_integration import CompactFineNetwork


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/experiments/evaluate_direct_dynamics_fine_support_proper_v1.json"


def test_config_freezes_reviewed_development_gate():
    config = json.loads(CONFIG.read_text())
    evaluation._validate_config(config)
    assert config["split"] == "valid"
    assert config["panel"]["role"] == "development_reuse"
    assert config["cases"] == 12 and config["members"] == 8
    assert config["coarse_rk4_timepoints"] == 17
    assert config["fine_rk4_timepoints"] == 33
    assert config["decision_gate"]["bootstrap_draws"] == 100000
    assert config["decision_gate"]["bootstrap_seed"] == 20260912
    assert config["optimizer_steps"] == 0
    assert config["test_2023"] == "closed"


def test_unrefined_terminal_fine_wrapper_replays_source_on_cpu():
    torch.manual_seed(17)
    height = width = 8
    valid = torch.ones(1, 1, height, width)
    structured = torch.randn(1, 15, height, width)
    structured[:, 2:3] = valid
    truth = torch.randn(1, 6, height, width)
    condition, _, _ = teacher_coarse_condition(truth, structured, valid)
    network = CompactFineNetwork()
    source = ColoredVariancePreconditionedFineCascadeSampler(
        network,
        target_residual_rms=(0.4,) * 6,
        projected_base_rms=(0.7,) * 6,
        blend=0.75,
        channel_scales=(0.5,) * 6,
    )
    terminal = copy.deepcopy(source.sampler.model)
    hybrid = evaluation.TerminalRefinedFineSampler(source, terminal)
    white = torch.randn(1, 6, height, width)
    zeros = torch.zeros_like(white)
    kwargs = dict(
        background=zeros,
        background_mask=torch.ones_like(zeros),
        obs_values=zeros[:, :2],
        obs_mask=zeros[:, :2],
        water_mask=valid,
        valid_mask=valid,
        state_mask=valid.expand(-1, 6, -1, -1),
        size=(height, width),
        num_timesteps=33,
        device=torch.device("cpu"),
        method="rk4",
        rtol=1e-5,
        atol=1e-6,
        start_mode="noise",
        initial_noise=white,
        sample_target="state",
        model_conditioning=condition,
        state_channels=6,
        end_time=0.0,
    )
    control = source.sample_conditioned(**kwargs)
    candidate = hybrid.sample_conditioned(**kwargs)
    assert torch.equal(control, candidate)


def test_scored_sic_law_has_exact_support_and_generated_coarse_budget():
    generator = torch.Generator().manual_seed(91)
    means = torch.tensor([0.2, 0.3] * 3)
    stds = torch.tensor([0.4, 0.5] * 3)
    physical = torch.randn(1, 4, 6, 4, 4, generator=generator)
    coarse_physical = torch.rand(1, 4, 6, 2, 2, generator=generator)
    forecast = (
        physical - means.reshape(1, 1, 6, 1, 1)
    ) / stds.reshape(1, 1, 6, 1, 1)
    coarse = (
        coarse_physical - means.reshape(1, 1, 6, 1, 1)
    ) / stds.reshape(1, 1, 6, 1, 1)
    valid = torch.ones(1, 1, 4, 4)
    normalized, decoded, error = evaluation._decode_sic_law(
        forecast, coarse, valid, means, stds
    )
    assert normalized.shape == forecast.shape
    assert decoded.shape == forecast.shape
    assert error <= 3e-14
    assert torch.all(decoded[:, :, 0::2] >= 0)
    assert torch.all(decoded[:, :, 0::2] <= 1)


def test_evaluator_persists_forecasts_before_scoring_and_numbers_before_plots():
    source = inspect.getsource(evaluation.run)
    raw_evidence = source.index("raw_evidence_sha = _persist_and_register")
    paired_checks = source.index("randomness = {")
    scored_evidence = source.index("\n        evidence_sha = _persist_and_register")
    primary_score = source.index("primary = score_ensemble")
    numerical = source.index("_atomic_json(numerical_path, result)")
    rank_plot = source.index("_save_rank_histograms")
    assert raw_evidence < paired_checks < scored_evidence < primary_score < numerical < rank_plot
    assert '"raw_secondary_metrics"' in source
    assert '"raw_forecast_physical"' in source
    assert '"scored_forecast_physical"' in source
    assert "paired branches differ in" in source
    assert "unrefined terminal fine hybrid does not replay production" in source


def test_analysis_failures_leave_durable_bound_evidence(tmp_path):
    for stage in ("parity", "decoder", "scorer"):
        directory = tmp_path / stage
        directory.mkdir()
        contract_path = directory / "contract.json"
        contract = {"evidence_sha256": {}}
        evaluation._persist_and_register(
            directory / f"{stage}.pt",
            {"stage": stage, "value": torch.ones(1)},
            contract_path,
            contract,
        )
        try:
            raise RuntimeError(f"injected {stage} failure")
        except RuntimeError:
            pass
        assert (directory / f"{stage}.pt").is_file()
        persisted = json.loads(contract_path.read_text())
        assert f"{stage}.pt" in persisted["evidence_sha256"]


def test_launcher_is_one_gpu_zero_optimizer_bounded():
    source = (
        ROOT / "scripts/run_direct_dynamics_fine_support_proper_evaluation.sh"
    ).read_text()
    assert ".gpu_job.lock" in source
    assert "require_single_gpu_uuid.sh" in source
    assert "for sample in {1..11}" in source and "sleep 30" in source
    assert '"$GPU_UTILIZATION" -ge 5' in source
    assert 'initial_memory="$GPU_MEMORY"' in source
    assert '"$initial_memory" -le 1024 && "$GPU_MEMORY" -gt 1024' in source
    assert source.count("for sample in {1..11}") == 2
    assert "CLEARML_REQUIRE_ONLINE=1" in source
    assert "--kill-after=60s 1740s" in source
    assert "train" not in source.lower().replace("fine_support", "")
