import torch

from assim_lib.direct_dynamics_coarse_correlated_reference import (
    ActiveCoarseLayout,
    fit_reference,
    identity_reference,
)


def test_active_layout_round_trip_and_padding_invariance():
    active = torch.zeros(1, 1, 4, 5)
    active[..., 1:3, 1:4] = 1
    fraction = active * 0.75
    layout = ActiveCoarseLayout.build(active, fraction)
    value = torch.randn(3, 6, 4, 5)
    restored = layout.restore(layout.vectorize(value))
    support = active.expand_as(value).bool()
    assert torch.equal(restored[support], value[support])
    assert torch.count_nonzero(restored[~support]) == 0


def test_identity_branch_is_exact_and_consumes_no_extra_rng():
    reference = identity_reference(19)
    first = torch.Generator().manual_seed(91)
    expected_generator = torch.Generator().manual_seed(91)
    latent = torch.Generator().manual_seed(77)
    latent_state = latent.get_state().clone()
    actual = reference.sample_flat(7, generator=first, latent_generator=latent)
    expected = torch.randn((7, 19), generator=expected_generator)
    assert torch.equal(actual, expected)
    assert torch.equal(latent.get_state(), latent_state)


def test_fitted_reference_has_unit_diagonal_and_empirical_covariance():
    generator = torch.Generator().manual_seed(4)
    x = torch.randn(160, 40, generator=generator)
    x[:, 1:] += 0.65 * x[:, :1]
    reference = fit_reference(
        x, torch.linspace(0.2, 1.0, 40), rank=8, alpha=0.25, seed=12
    )
    diagonal, low_rank = reference.factors()
    covariance = torch.diag(diagonal) + low_rank @ low_rank.T
    assert torch.allclose(torch.diag(covariance), torch.ones(40, dtype=torch.float64), atol=1e-6)
    assert torch.linalg.eigvalsh(covariance).min() > 0
    samples = reference.sample_flat(
        60000,
        generator=torch.Generator().manual_seed(20),
        latent_generator=torch.Generator().manual_seed(21),
        dtype=torch.float64,
    )
    empirical = samples.T @ samples / samples.shape[0]
    assert torch.allclose(torch.diag(empirical), torch.ones(40, dtype=torch.float64), atol=0.025)
    assert torch.mean(torch.abs(empirical - covariance)) < 0.012


def test_woodbury_score_matches_dense_score():
    x = torch.randn(80, 32, generator=torch.Generator().manual_seed(10))
    reference = fit_reference(x, torch.ones(32), rank=6, alpha=0.5, seed=31)
    probe = torch.randn(5, 32, generator=torch.Generator().manual_seed(11), dtype=torch.float64)
    diagonal, low_rank = reference.factors()
    covariance = torch.diag(diagonal) + low_rank @ low_rank.T
    dense = (
        torch.einsum("bi,ij,bj->b", probe, torch.linalg.inv(covariance), probe)
        + torch.linalg.slogdet(covariance).logabsdet
    ) / (2 * 32)
    assert torch.allclose(reference.gaussian_score(probe), dense, atol=1e-10)
