import contextlib
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import torch
from torchdiffeq import odeint

from assim_lib import direct_dynamics_cascade_proper_refinement_evaluation as evaluation
from assim_lib.direct_dynamics_cascade_coarse_proper_refinement import (
    frozen_prefix,
    hybrid_terminal_sample,
    mask_aware_block_average_score_inputs,
    multiscale_proper_objective,
    _persist_training_update_then_report,
    proper_objective,
    rk4_interval,
    standardized_fair_crps,
    standardized_joint_energy,
)
from assim_lib.direct_dynamics_cascade import masked_block_average, smooth_right_inverse


def _confirmation_experiment():
    return json.loads(
        Path(
            "config/experiments/evaluate_direct_dynamics_cascade_proper_refinement_confirmation_v1.json"
        ).read_text()
    )


def _sit_support_confirmation_experiment():
    return json.loads(
        Path(
            "config/experiments/evaluate_direct_dynamics_cascade_sit_support_confirmation_v1.json"
        ).read_text()
    )


def _sit_support_final_test_experiment():
    return json.loads(
        Path(
            "config/experiments/evaluate_direct_dynamics_cascade_sit_support_final_test_2023_v1.json"
        ).read_text()
    )


def test_frozen_confirmation_panel_is_disjoint_and_valid():
    experiment = _confirmation_experiment()
    evaluation._validate(experiment)
    broken = deepcopy(experiment)
    broken["case_ids"][0] = "2022-01-05_slice12"
    with unittest.TestCase().assertRaisesRegex(ValueError, "overlap"):
        evaluation._validate(broken)


def test_sit_support_confirmation_manifest_is_new_disjoint_and_frozen():
    experiment = _sit_support_confirmation_experiment()
    evaluation._validate(experiment)
    inventory = experiment["panel"]["prior_date_level_inventory_search"]
    assert inventory["observed_2022_date_count"] == 223
    assert inventory["selected_date_matches"] == 0
    assert experiment["support_decoder"]["fit_on_confirmation"] is False

    broken = deepcopy(experiment)
    broken["support_decoder"]["fit_on_confirmation"] = True
    with unittest.TestCase().assertRaisesRegex(ValueError, "must not fit"):
        evaluation._validate(broken)


def test_sit_support_confirmation_decoder_integration_accepts_tuple_stats():
    coarse = torch.tensor(
        [[[[[0.4]], [[-0.2]], [[0.6]], [[0.3]], [[0.2]], [[-0.1]]]]],
        dtype=torch.float32,
    )
    valid = torch.ones(1, 1, 2, 2)
    fine = smooth_right_inverse(coarse.flatten(0, 1), valid, 2).unflatten(0, (1, 1))
    fine[:, :, 1::2, 0, 0] -= 0.4
    fine[:, :, 1::2, 1, 1] += 0.4
    normalized, physical, error = evaluation._apply_frozen_support_decoder(
        fine, coarse, valid, (0.0,) * 6, (1.0,) * 6
    )
    recovered, fraction = masked_block_average(physical.flatten(0, 1), valid)
    target = coarse.flatten(0, 1).clone()
    target[:, 1::2].clamp_min_(0)
    assert error < 3e-6
    assert float(physical[:, :, 1::2].min()) == 0.0
    assert torch.allclose(recovered[fraction.expand_as(recovered) > 0], target.flatten())
    assert torch.equal(normalized, physical)


def test_sit_support_final_test_is_sealed_and_calendar_frozen():
    experiment = _sit_support_final_test_experiment()
    evaluation._validate(experiment)
    assert experiment["split"] == "test"
    assert experiment["panel"]["authorization_state"] == "sealed_not_authorized"
    assert experiment["panel"]["data_values_inspected_before_freeze"] is False
    assert experiment["case_ids"][0] == "2023-01-03_slice01"
    assert experiment["case_ids"][-1] == "2023-07-09_slice06"

    broken = deepcopy(experiment)
    broken["panel"]["requires_explicit_user_authorization"] = False
    with unittest.TestCase().assertRaisesRegex(ValueError, "sealing contract"):
        evaluation._validate(broken)

    bypass = deepcopy(experiment)
    bypass["panel"]["role"] = "frozen_confirmation"
    with unittest.TestCase().assertRaisesRegex(ValueError, "test split requires"):
        evaluation._validate(bypass)


def test_final_test_single_use_blocks_before_second_dataset_access(monkeypatch):
    experiment = _sit_support_final_test_experiment()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        marker = root / ".sit_support_final_test_2023_v1.consumed.json"
        experiment["panel"]["single_use_marker"] = str(marker)
        config_path = root / "config.json"
        config_path.write_text(json.dumps(experiment))
        calls = []
        monkeypatch.setattr(
            evaluation,
            "build_dataset",
            lambda config, split: calls.append((config, split)) or "dataset",
        )
        monkeypatch.setattr(evaluation, "FINAL_TEST_SINGLE_USE_MARKER", marker)
        assert evaluation._build_dataset_after_final_test_seal(
            {}, "test", experiment, config_path, root / "first", {"git_commit": "a" * 40}
        ) == "dataset"
        with unittest.TestCase().assertRaises(FileExistsError):
            evaluation._build_dataset_after_final_test_seal(
                {}, "test", experiment, config_path, root / "second", {"git_commit": "a" * 40}
            )
        assert len(calls) == 1


def test_confirmation_primary_is_case_equal_and_bootstrapped():
    raw = {"case_equal_mean": 1.0, "case_values": [1.0] * 12}
    candidate = {"case_equal_mean": 0.9, "case_values": [0.9] * 12}
    result = evaluation._paired_primary_confirmation(
        raw, candidate, _confirmation_experiment()["decision_gate"]
    )
    assert result["primary_passed"]
    assert result["paired_date_bootstrap_95_ci_high"] < 0
    assert abs(result["relative_change_of_aggregate"] + 0.1) < 1e-12


def test_paired_evaluation_uses_reviewed_precision(monkeypatch):
    active = {"value": False}

    class Autocast:
        def __enter__(self):
            active["value"] = True

        def __exit__(self, *_):
            active["value"] = False

    class Predictor:
        def sample_ensemble(self, **kwargs):
            assert active["value"]
            assert kwargs == {"token": 7}
            return {"ok": True}

    monkeypatch.setattr(torch, "autocast", lambda **_: Autocast())
    assert evaluation._sample_with_reviewed_precision(Predictor(), token=7) == {"ok": True}


def test_raw_evidence_precedes_scoring_failure():
    class ScoreFailure(RuntimeError):
        pass

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "candidate_raw.pt"
        registered = []

        def fail_score():
            raise ScoreFailure("injected scorer failure")

        try:
            evaluation._persist_evidence_then_score(
                path,
                {"forecast": torch.ones(1), "raw_noise": torch.zeros(1)},
                registered.append,
                fail_score,
            )
        except ScoreFailure as error:
            assert str(error) == "injected scorer failure"
        else:
            raise AssertionError("scoring failure was not preserved")
        assert path.is_file()
        assert registered == [evaluation._sha256(path)]


def test_evaluation_failure_is_durable_and_closes_tracker(monkeypatch):
    class PrimaryFailure(RuntimeError):
        pass

    class Tracker:
        closed = 0

        def close(self):
            self.closed += 1
            raise RuntimeError("secondary close failure")

    def fail_run(*_):
        raise PrimaryFailure("primary evaluation failure")

    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory)
        tracker = Tracker()
        monkeypatch.setattr(evaluation, "_ACTIVE_TRACKER", tracker)
        monkeypatch.setattr(evaluation, "_run_impl", fail_run)
        try:
            evaluation.run(Path("config.json"), output)
        except PrimaryFailure as error:
            assert str(error) == "primary evaluation failure"
        else:
            raise AssertionError("primary failure was not preserved")
        assert json.loads((output / "failure.json").read_text())["error_type"] == "PrimaryFailure"
        assert tracker.closed == 1


def test_final_checkpoint_precedes_reporting_failure():
    class ReportingFailure(RuntimeError):
        pass

    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory)

        def fail_reporting():
            raise ReportingFailure("network failed on update 64")

        try:
            _persist_training_update_then_report(
                output,
                {"completed_updates": 64, "history": [{"update": 64}]},
                fail_reporting,
                final_checkpoint={"model": {"weight": torch.ones(1)}, "completed_updates": 64},
            )
        except ReportingFailure as error:
            assert str(error) == "network failed on update 64"
        else:
            raise AssertionError("reporting failure was not preserved")
        assert (output / "terminal_model_update_64.pth").is_file()
        assert json.loads((output / "progress.json").read_text())["completed_updates"] == 64


def test_fair_crps_two_member_closed_form():
    members = torch.zeros(1, 2, 6, 1, 1)
    members[:, 1] = 2.0
    truth = torch.ones(1, 6, 1, 1)
    fraction = torch.ones(1, 1, 1, 1)
    assert standardized_fair_crps(members, truth, fraction).item() == 0.0


def test_joint_energy_two_member_closed_form():
    members = torch.zeros(1, 2, 6, 1, 1)
    members[:, 1] = 2.0
    truth = torch.ones(1, 6, 1, 1)
    fraction = torch.ones(1, 1, 1, 1)
    assert abs(standardized_joint_energy(members, truth, fraction).item()) < 1e-5
    objective, crps, energy = proper_objective(members, truth, fraction)
    assert torch.isfinite(objective + crps + energy)


def test_mask_aware_four_by_four_average_does_not_mix_land_zero():
    members = torch.full((1, 2, 6, 4, 4), 7.0)
    truth = torch.full((1, 6, 4, 4), 5.0)
    fraction = torch.ones(1, 1, 4, 4)
    fraction[..., :2, :2] = 0.0
    members[..., :2, :2] = 0.0
    truth[..., :2, :2] = 0.0
    coarse_members, coarse_truth, coarse_fraction = mask_aware_block_average_score_inputs(
        members, truth, fraction
    )
    assert torch.equal(coarse_members, torch.full_like(coarse_members, 7.0))
    assert torch.equal(coarse_truth, torch.full_like(coarse_truth, 5.0))
    assert torch.equal(coarse_fraction, torch.full_like(coarse_fraction, 0.75))


def test_multiscale_objective_preserves_backward():
    members = torch.randn(1, 4, 6, 8, 8, requires_grad=True)
    truth = torch.randn(1, 6, 8, 8)
    fraction = torch.ones(1, 1, 8, 8)
    objective, marginal, energy = multiscale_proper_objective(members, truth, fraction)
    assert torch.isfinite(objective + marginal + energy)
    objective.backward()
    assert members.grad is not None and torch.isfinite(members.grad).all()


def test_scores_ignore_inactive_nan_and_preserve_backward():
    members = torch.zeros(1, 2, 6, 1, 2, requires_grad=True)
    truth = torch.zeros(1, 6, 1, 2)
    fraction = torch.tensor([[[[1.0, 0.0]]]])
    with torch.no_grad():
        members[..., 1] = torch.nan
        truth[..., 1] = torch.nan
        members[:, 1, :, :, 0] = 1.0
    objective, crps, energy = proper_objective(members, truth, fraction)
    assert torch.isfinite(objective + crps + energy)
    objective.backward()
    assert members.grad is not None
    assert torch.isfinite(members.grad[..., 0]).all()


def test_joint_energy_exact_zero_has_finite_zero_gradient():
    members = torch.zeros(1, 4, 6, 2, 2, requires_grad=True)
    truth = torch.zeros(1, 6, 2, 2)
    fraction = torch.ones(1, 1, 2, 2)
    score = standardized_joint_energy(members, truth, fraction)
    score.backward()
    assert score.item() == 0.0
    assert torch.isfinite(members.grad).all()
    assert torch.count_nonzero(members.grad) == 0


def test_fractional_ocean_weights_change_score_case_equally():
    members = torch.zeros(1, 2, 6, 1, 2)
    members[:, :, :, :, 1] = 2.0
    truth = torch.zeros(1, 6, 1, 2)
    equal = standardized_fair_crps(members, truth, torch.ones(1, 1, 1, 2))
    coastal = standardized_fair_crps(
        members, truth, torch.tensor([[[[1.0, 0.25]]]])
    )
    assert coastal < equal


class _SquareVelocity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.calls = 0

    def forward(self, model_input, timesteps, return_dict=False):
        self.calls += 1
        state = model_input[:, :6]
        time = timesteps.reshape(-1, 1, 1, 1) / 1000.0
        return (self.scale * (state.square() + 0.1 * time),)


def test_rk4_reverse_interval_matches_torchdiffeq_three_eighths_rule(monkeypatch):
    monkeypatch.setattr(torch, "autocast", lambda **_: contextlib.nullcontext())
    state = torch.ones(1, 6, 2, 2)
    condition = torch.zeros(1, 48, 2, 2)
    active = torch.ones(1, 1, 2, 2)
    grid = torch.zeros(1, 2, 2, 2)
    step = -1.0 / 16.0
    def velocity(value, time):
        return value**2 + 0.1 * time
    k1 = velocity(1.0, 1 / 16)
    k2 = velocity(1.0 + step * k1 / 3.0, 1 / 16 + step / 3.0)
    k3 = velocity(1.0 + step * (k2 - k1 / 3.0), 1 / 16 + 2 * step / 3.0)
    k4 = velocity(1.0 + step * (k1 - k2 + k3), 0.0)
    expected = 1.0 + step * (k1 + 3.0 * k2 + 3.0 * k3 + k4) / 8.0
    result = rk4_interval(_SquareVelocity(), state, condition, active, grid, 1 / 16, 0.0)
    assert torch.allclose(result, torch.full_like(result, expected), atol=1e-7, rtol=0)


def test_hybrid_replays_torchdiffeq_and_gradient_is_suffix_only(monkeypatch):
    monkeypatch.setattr(torch, "autocast", lambda **_: contextlib.nullcontext())
    frozen = _SquareVelocity()
    frozen.scale.requires_grad_(False)
    candidate = _SquareVelocity()
    noise = torch.tensor(
        [[[[1.0, 9.0], [0.5, 9.0]]]]
    ).expand(1, 6, 2, 2).clone()
    condition = torch.zeros(1, 48, 2, 2)
    active = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    grid = torch.zeros(1, 2, 2, 2)
    prefix = frozen_prefix(frozen, noise, condition, active, grid)
    assert not prefix.requires_grad
    assert frozen.calls == 60
    result = hybrid_terminal_sample(candidate, prefix, condition, active, grid)
    assert candidate.calls == 4
    result.sum().backward()
    assert candidate.scale.grad is not None
    assert torch.isfinite(candidate.scale.grad) and candidate.scale.grad != 0
    assert frozen.scale.grad is None

    masked_noise = torch.where(active.expand_as(noise) > 0, noise, torch.zeros_like(noise))
    reference_model = _SquareVelocity()
    reference_model.scale.requires_grad_(False)
    def f(time, state):
        velocity = reference_model.scale * (state.square() + 0.1 * time)
        return torch.where(active.expand_as(velocity) > 0, velocity, torch.zeros_like(velocity))
    timeline = torch.linspace(1.0, 0.0, 17)
    reference = odeint(
        f,
        masked_noise,
        timeline,
        method="rk4",
        options={"step_size": 1.0 / 16.0},
    )[-1]
    assert torch.allclose(result.detach(), reference, atol=2e-6, rtol=0)


def test_launcher_pins_gpu_and_closes_lifecycle():
    source = open(
        "scripts/run_direct_dynamics_cascade_coarse_proper_refinement.sh",
        encoding="utf-8",
    ).read()
    assert 'export CUDA_VISIBLE_DEVICES="$GPU_UUID"' in source
    assert "exit.json" in source
    assert "gpu became busy" in source.lower()
    assert "1740s" in source and "--kill-after=60s" in source
    evaluation_source = open(
        "scripts/run_direct_dynamics_cascade_e2e_evaluation.sh",
        encoding="utf-8",
    ).read()
    assert "require_single_gpu_uuid.sh" in evaluation_source
    assert 'export CUDA_VISIBLE_DEVICES="$GPU_UUID"' in evaluation_source
    assert "evaluate_direct_dynamics_cascade_proper_refinement_confirmation_v1.json" in evaluation_source
    assert "GPU became busy" in evaluation_source
    assert "2640s" in evaluation_source and "--kill-after=60s" in evaluation_source
