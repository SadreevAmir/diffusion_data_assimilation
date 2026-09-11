import json
import subprocess
import tempfile
from copy import deepcopy
from pathlib import Path

import torch

from assim_lib import direct_dynamics_cascade_proper_refinement_evaluation as evaluation
from assim_lib.direct_dynamics_threshold_weighted_score import (
    boundary_emphasis_transform,
    threshold_weighted_fair_crps,
    threshold_weighted_proper_objective,
)
from assim_lib.direct_dynamics_cascade_coarse_proper_refinement import (
    _configured_proper_objective,
    _persist_step0_evidence_then_assess,
    _validate_reviewed_protocol,
    standardized_fair_crps,
)


MEANS = torch.tensor([0.5, 1.0] * 3)
STDS = torch.tensor([0.25, 0.5] * 3)


def _normalize(physical: torch.Tensor) -> torch.Tensor:
    return (physical - MEANS.reshape(6, 1, 1)) / STDS.reshape(6, 1, 1)


def _experiment() -> dict:
    return json.loads(
        Path(
            "config/experiments/train_direct_dynamics_cascade_threshold_weighted_refinement_v1.json"
        ).read_text()
    )


def _evaluation_experiment() -> dict:
    return json.loads(
        Path(
            "config/experiments/evaluate_direct_dynamics_cascade_threshold_weighted_refinement_v1.json"
        ).read_text()
    )


def test_threshold_weighted_protocol_is_exact_and_frozen():
    experiment = _experiment()
    _validate_reviewed_protocol(experiment["protocol"])
    changed = deepcopy(experiment["protocol"])
    changed["marginal_score"]["boundary_weight_multiplier"] = 3.0
    try:
        _validate_reviewed_protocol(changed)
    except ValueError:
        pass
    else:
        raise AssertionError("unreviewed threshold weight was accepted")


def test_threshold_evaluation_is_explicitly_development_only_and_valid():
    experiment = _evaluation_experiment()
    assert experiment["panel"]["role"] == "development_reuse"
    assert experiment["decision_gate"]["claim_policy"] == (
        "development_only_not_independent_confirmation"
    )
    evaluation._validate(experiment)
    launcher = Path("scripts/run_direct_dynamics_cascade_e2e_evaluation.sh").read_text()
    assert "evaluate_direct_dynamics_cascade_threshold_weighted_refinement_v1.json" in launcher


def test_threshold_evaluation_launcher_executes_terminal_module_dispatch():
    environment = {
        "PATH": "/usr/bin:/bin",
        "CASCADE_E2E_RUN_ID": "resolve_threshold_module",
        "CASCADE_E2E_CONFIG": (
            "config/experiments/"
            "evaluate_direct_dynamics_cascade_threshold_weighted_refinement_v1.json"
        ),
        "CASCADE_E2E_RESOLVE_ONLY": "1",
    }
    result = subprocess.run(
        ["bash", "scripts/run_direct_dynamics_cascade_e2e_evaluation.sh"],
        cwd=Path.cwd(),
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == (
        "assim_lib.direct_dynamics_cascade_proper_refinement_evaluation"
    )


def test_launcher_rejects_unreviewed_config_before_gpu_admission():
    environment = {
        "PATH": "/usr/bin:/bin",
        "PROPER_REFINEMENT_RUN_ID": "reject_unreviewed",
        "PROPER_REFINEMENT_CONFIG": "config/experiments/not_reviewed.json",
        "PROPER_REFINEMENT_OUTPUT_GROUP": "proper_refinement",
    }
    result = subprocess.run(
        ["bash", "scripts/run_direct_dynamics_cascade_coarse_proper_refinement.sh"],
        cwd=Path.cwd(),
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "unreviewed proper-refinement config/output pair" in result.stderr
    assert "nvidia-smi" not in result.stderr


def test_step0_evidence_survives_assessment_failure_with_original_exception():
    class InjectedScoreFailure(RuntimeError):
        pass

    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory)

        def fail():
            raise InjectedScoreFailure("injected threshold scorer failure")

        try:
            _persist_step0_evidence_then_assess(
                output,
                {"candidate": torch.ones(1), "marginal_score_contract": {"kind": "test"}},
                fail,
            )
        except InjectedScoreFailure as error:
            assert str(error) == "injected threshold scorer failure"
        else:
            raise AssertionError("assessment failure was not preserved")
        assert (output / "step0_samples.pth").is_file()
        status = json.loads((output / "status.json").read_text())
        assert status["status"] == "failed"
        assert status["error_type"] == "InjectedScoreFailure"
        assert len(status["step0_samples_sha256"]) == 64


def test_configured_objective_reaches_threshold_weighted_production_score():
    experiment = _experiment()
    physical_members = torch.tensor([-0.01, 0.01]).reshape(1, 2, 1, 1, 1).expand(
        -1, -1, 6, -1, -1
    )
    physical_truth = torch.full((1, 6, 1, 1), 0.03)
    data_config = {
        "fields": ["siconc", "sithic"],
        "means": MEANS[:2].tolist(),
        "stds": STDS[:2].tolist(),
    }
    members = (physical_members - MEANS.reshape(1, 1, 6, 1, 1)) / STDS.reshape(
        1, 1, 6, 1, 1
    )
    truth = (physical_truth - MEANS.reshape(1, 6, 1, 1)) / STDS.reshape(1, 6, 1, 1)
    objective, marginal, energy = _configured_proper_objective(
        members,
        truth,
        torch.ones((1, 1, 1, 1)),
        experiment["protocol"],
        data_config,
    )
    assert torch.allclose(marginal, torch.tensor(0.09), atol=1e-7)
    assert torch.allclose(objective, 0.75 * marginal + 0.25 * energy)


def test_transform_is_identity_plus_exact_frozen_boundary_arc_length():
    physical = torch.tensor([-0.01, 0.0, 0.01, 0.02, 0.50, 0.98, 0.99, 1.0, 1.01])
    values = _normalize(physical.reshape(1, 1, -1).expand(6, -1, -1))
    transformed = boundary_emphasis_transform(values, MEANS, STDS)
    delta = transformed - values
    # SIC has two 0.02-wide physical bands; SIT has only the low band.
    assert torch.all(delta[:, 0, 0] == 0)
    assert torch.allclose(delta[0, 0, -1], torch.tensor(0.16), atol=1e-6)
    assert torch.allclose(delta[1, 0, -1], torch.tensor(0.04), atol=1e-6)
    assert torch.all(torch.diff(transformed, dim=-1) > 0)


def test_threshold_weighted_score_has_finite_nonzero_gradient():
    generator = torch.Generator().manual_seed(701)
    members = torch.randn((2, 4, 6, 3, 2), generator=generator, requires_grad=True)
    truth = torch.randn((2, 6, 3, 2), generator=generator)
    fraction = torch.ones((2, 1, 3, 2))
    objective, score, energy = threshold_weighted_proper_objective(
        members, truth, fraction, MEANS, STDS
    )
    objective.backward()
    assert torch.isfinite(objective + score + energy)
    assert members.grad is not None and torch.isfinite(members.grad).all()
    assert torch.count_nonzero(members.grad) > 0


def test_unbiased_two_member_estimator_keeps_closed_form():
    physical_members = torch.zeros((1, 2, 6, 1, 1))
    physical_members[:, 1] = 0.02
    physical_truth = torch.full((1, 6, 1, 1), 0.01)
    members = (physical_members - MEANS.reshape(1, 1, 6, 1, 1)) / STDS.reshape(
        1, 1, 6, 1, 1
    )
    truth = (physical_truth - MEANS.reshape(1, 6, 1, 1)) / STDS.reshape(1, 6, 1, 1)
    score = threshold_weighted_fair_crps(
        members, truth, torch.ones((1, 1, 1, 1)), MEANS, STDS
    )
    assert abs(float(score)) < 1e-7


def test_production_weighted_score_has_independent_nonzero_reference():
    physical_members = torch.tensor([-0.01, 0.01]).reshape(1, 2, 1, 1, 1).expand(
        -1, -1, 6, -1, -1
    )
    physical_truth = torch.full((1, 6, 1, 1), 0.03)
    members = (physical_members - MEANS.reshape(1, 1, 6, 1, 1)) / STDS.reshape(
        1, 1, 6, 1, 1
    )
    truth = (physical_truth - MEANS.reshape(1, 6, 1, 1)) / STDS.reshape(1, 6, 1, 1)
    fraction = torch.ones((1, 1, 1, 1))
    weighted = threshold_weighted_fair_crps(members, truth, fraction, MEANS, STDS)
    ordinary = standardized_fair_crps(members, truth, fraction)
    assert torch.allclose(weighted, torch.tensor(0.09), atol=1e-7)
    assert torch.allclose(ordinary, torch.tensor(0.06), atol=1e-7)


def test_mixed_law_oracle_prefers_the_true_distribution():
    # Exact expected transformed-kernel score on a law with boundary atoms and
    # an interior component.  No Monte Carlo noise is involved.
    support_physical = torch.tensor([0.0, 0.01, 0.5, 0.99, 1.0])
    support = _normalize(support_physical.reshape(1, 1, -1).expand(6, -1, -1))[0, 0]
    embedded = torch.zeros((1, 6, 1, 5))
    embedded[:, 0, 0] = support
    transformed = boundary_emphasis_transform(embedded, MEANS, STDS)[0, 0, 0]
    truth_probability = torch.tensor([0.30, 0.10, 0.20, 0.10, 0.30], dtype=torch.float64)

    def expected_score(forecast_probability: torch.Tensor) -> torch.Tensor:
        distance = (transformed[:, None] - transformed[None, :]).abs().double()
        accuracy = (forecast_probability[:, None] * truth_probability[None, :] * distance).sum()
        dispersion = 0.5 * (
            forecast_probability[:, None] * forecast_probability[None, :] * distance
        ).sum()
        return accuracy - dispersion

    oracle = expected_score(truth_probability)
    alternatives = (
        torch.tensor([0.20, 0.20, 0.20, 0.20, 0.20], dtype=torch.float64),
        torch.tensor([0.40, 0.05, 0.20, 0.05, 0.30], dtype=torch.float64),
        torch.tensor([0.20, 0.10, 0.40, 0.10, 0.20], dtype=torch.float64),
    )
    assert all(expected_score(candidate) > oracle for candidate in alternatives)


def test_production_estimator_matches_exact_mixed_law_population_score():
    support_physical = torch.tensor([0.0, 0.01, 0.5, 0.99, 1.0])
    probability = torch.tensor([0.30, 0.10, 0.20, 0.10, 0.30], dtype=torch.float64)
    channel_support = (
        support_physical.reshape(1, -1) - MEANS.reshape(-1, 1)
    ) / STDS.reshape(-1, 1)
    embedded = channel_support.reshape(1, 6, 1, 5)
    transformed = boundary_emphasis_transform(embedded, MEANS, STDS)[0, :, 0].double()
    distance = (transformed[:, :, None] - transformed[:, None, :]).abs()
    population = (
        probability[None, :, None] * probability[None, None, :] * distance
    ).sum(dim=(1, 2))
    expected_population_score = 0.5 * population.mean()

    production_expectation = torch.zeros((), dtype=torch.float64)
    fraction = torch.ones((1, 1, 1, 1))
    for left in range(5):
        for right in range(5):
            members = torch.stack(
                (channel_support[:, left], channel_support[:, right]), dim=0
            ).reshape(1, 2, 6, 1, 1)
            for observation in range(5):
                truth = channel_support[:, observation].reshape(1, 6, 1, 1)
                score = threshold_weighted_fair_crps(
                    members, truth, fraction, MEANS, STDS
                ).double()
                production_expectation = production_expectation + (
                    probability[left]
                    * probability[right]
                    * probability[observation]
                    * score
                )
    assert torch.allclose(production_expectation, expected_population_score, atol=1e-7)
