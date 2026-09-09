import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from assim_lib.censored_joint_energy_toy import (
    ConditionalAffineGenerator,
    _make_figure,
    censor_physical,
    fit_censored_joint_toy,
    target_parameters,
    unbiased_energy_score,
)


class CensoredJointEnergyToyTests(unittest.TestCase):
    def test_censoring_creates_exact_physical_atoms(self) -> None:
        latent = torch.tensor(
            [[-1.0, -2.0, 2.0, 3.0], [0.2, 0.1, 0.8, -0.1]],
            dtype=torch.float64,
        )
        physical = censor_physical(latent)
        self.assertTrue(torch.equal(physical[0], torch.tensor([0.0, 0.0, 1.0, 3.0])))
        self.assertEqual(float(physical[1, 3]), 0.0)

    def test_energy_score_uses_unbiased_off_diagonal_pairing(self) -> None:
        members = torch.tensor([[[0.0], [2.0]]], dtype=torch.float64)
        truth = torch.tensor([[0.0]], dtype=torch.float64)
        score = unbiased_energy_score(members, truth, torch.ones(1, dtype=torch.float64))
        # First term is 1; ordered off-diagonal distances sum to 4 and the
        # denominator is 2*M*(M-1)=4, hence the score is zero.
        self.assertEqual(float(score), 0.0)

    def test_target_covariance_is_joint_not_diagonal(self) -> None:
        _, factors = target_parameters()
        covariance = factors @ factors.transpose(-1, -2)
        self.assertGreater(float(covariance[0].tril(-1).abs().max()), 0.0)
        self.assertGreater(float(covariance[1].tril(-1).abs().max()), 0.0)

    def test_gate_tolerances_are_predeclared(self) -> None:
        from assim_lib.censored_joint_energy_toy import GATE_TOLERANCES

        self.assertEqual(
            set(GATE_TOLERANCES),
            {
                "learned_normalized_mean_linf",
                "learned_normalized_covariance_linf",
                "learned_boundary_mass_linf",
                "learned_joint_sit_zero_abs",
                "learned_energy_score_gap_vs_target",
                "independent_marginal_boundary_mass_linf",
                "open_thin_independent_joint_sit_zero_abs_min",
                "icy_independent_normalized_covariance_linf_min",
                "icy_independent_energy_score_gap_min",
            },
        )

    def test_generator_initialization_is_stochastic_and_full_covariance(self) -> None:
        model = ConditionalAffineGenerator(seed=1701)
        factors = model.factors()
        self.assertTrue(torch.all(torch.diagonal(factors, dim1=-2, dim2=-1) > 0))
        self.assertGreater(float(torch.tril(factors, diagonal=-1).abs().sum()), 0.0)

    def test_short_fit_has_live_gradients_and_no_forbidden_mse(self) -> None:
        model, training = fit_censored_joint_toy(updates=4, seed=1701)
        probe = training["dead_censoring_probe"]
        self.assertGreater(probe["mean_gradient_norm"], 0.0)
        self.assertGreater(probe["factor_gradient_norm"], 0.0)
        self.assertFalse(training["per_member_mse"])

        with TemporaryDirectory() as directory:
            figure = _make_figure({}, model, seed=1701)
            path = Path(directory) / "probe.png"
            figure.savefig(path, dpi=40)
            import matplotlib.pyplot as plt

            plt.close(figure)
            self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
