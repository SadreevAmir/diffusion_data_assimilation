import unittest
import math

import torch

from assim_lib.joint_atom_law_oracle import _posterior_probabilities
from assim_lib.joint_atom_law_oracle import joint_atom_law_oracle_report


class JointAtomLawOracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = joint_atom_law_oracle_report(sample_count=8192, seed=1701)

    def test_joint_posterior_draw_preserves_atoms_and_law(self) -> None:
        for epsilon in ("0.01", "0.02"):
            draw = self.report["epsilons"][epsilon]["posterior_draw"]
            self.assertEqual(draw["exact_joint_atom_mass"], 1.0)
            self.assertLess(draw["frequency_linf_error_vs_target_law"], 0.025)
            self.assertLess(draw["covariance_linf_error"], 0.06)

    def test_posterior_mean_erases_zero_atoms(self) -> None:
        for epsilon in ("0.01", "0.02"):
            mean = self.report["epsilons"][epsilon]["posterior_mean"]
            self.assertEqual(mean["zero_mass_sit_a"], 0.0)
            self.assertEqual(mean["zero_mass_sit_b"], 0.0)
            self.assertGreater(mean["mean_nearest_atom_distance"], 0.0)

    def test_exact_zero_delta_is_preserved(self) -> None:
        for epsilon in ("0.01", "0.02"):
            zero = self.report["pure_zero_delta"][epsilon]
            self.assertEqual(zero["posterior_mean_exact_zero_mass"], 1.0)
            self.assertEqual(zero["posterior_draw_exact_zero_mass"], 1.0)
            self.assertEqual(zero["posterior_probability"], 1.0)

    def test_gaussian_bayes_posterior_matches_independent_formula(self) -> None:
        atoms = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
        weights = torch.tensor([0.25, 0.75], dtype=torch.float64)
        state = torch.tensor([[0.20]], dtype=torch.float64)
        epsilon = 0.50
        actual = _posterior_probabilities(state, atoms, weights, epsilon)[0]
        likelihoods = [
            math.exp(-0.5 * ((0.20 - (1.0 - epsilon) * atom) / epsilon) ** 2)
            for atom in (0.0, 1.0)
        ]
        unnormalized = [0.25 * likelihoods[0], 0.75 * likelihoods[1]]
        normalizer = sum(unnormalized)
        expected = torch.tensor(
            [value / normalizer for value in unnormalized], dtype=torch.float64
        )
        self.assertTrue(torch.allclose(actual, expected, atol=1e-14, rtol=1e-14))

    def test_repeated_ambiguous_state_draws_match_conditional_posterior(self) -> None:
        probe = self.report["fixed_ambiguous_state_probe"]
        self.assertLess(probe["draw_frequency_linf_error_vs_posterior"], 0.025)
        self.assertGreater(probe["posterior_linf_difference_from_prior"], 0.1)

    def test_gate_remains_cpu_only(self) -> None:
        gate = self.report["gate"]
        self.assertTrue(gate["posterior_draw_preserves_atoms"])
        self.assertTrue(gate["posterior_mean_erases_zero_atoms"])
        self.assertTrue(gate["proper_scores_favor_joint_posterior_law"])
        self.assertTrue(gate["pure_zero_delta_preserved"])
        self.assertTrue(gate["fixed_state_conditional_draw_matches_posterior"])
        self.assertFalse(gate["permits_gpu_training"])


if __name__ == "__main__":
    unittest.main()
