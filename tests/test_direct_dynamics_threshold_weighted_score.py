import torch

from assim_lib.direct_dynamics_threshold_weighted_score import (
    boundary_emphasis_transform,
    threshold_weighted_fair_crps,
    threshold_weighted_proper_objective,
)


MEANS = torch.tensor([0.5, 1.0] * 3)
STDS = torch.tensor([0.25, 0.5] * 3)


def _normalize(physical: torch.Tensor) -> torch.Tensor:
    return (physical - MEANS.reshape(6, 1, 1)) / STDS.reshape(6, 1, 1)


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
