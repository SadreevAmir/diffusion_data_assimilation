import unittest

import torch

from assim_lib.direct_dynamics_affine_calibration import (
    _extra_gate,
    _structural_comparison,
    apply_persistence_affine,
    boundary_event_metrics,
    coverage_metrics,
    fit_persistence_affine,
    joint_energy_score,
    primary_score_comparison,
    standardized_fair_crps,
)


class PersistenceAffineCalibrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.means = torch.zeros(6)
        self.stds = torch.ones(6)
        self.persistence = torch.zeros(2, 6, 2, 2)
        self.mask = torch.ones(2, 1, 2, 2)

    def test_identity_is_exact(self) -> None:
        ensemble = torch.randn(2, 8, 6, 2, 2)
        actual = apply_persistence_affine(
            ensemble,
            self.persistence,
            self.means,
            self.stds,
            scale=1.0,
            sic_offset=0.0,
            sit_offset=0.0,
        )
        self.assertTrue(torch.equal(actual, ensemble))

    def test_map_preserves_member_order_and_common_trajectory_scale(self) -> None:
        ensemble = torch.arange(2 * 8 * 6 * 2 * 2, dtype=torch.float32).reshape(2, 8, 6, 2, 2)
        actual = apply_persistence_affine(
            ensemble,
            self.persistence,
            self.means,
            self.stds,
            scale=1.2,
            sic_offset=0.02,
            sit_offset=-0.03,
        )
        self.assertTrue(torch.all(actual[:, 1:] > actual[:, :-1]))
        before = ensemble[:, :, 4] - ensemble[:, :, 0]
        after = actual[:, :, 4] - actual[:, :, 0]
        self.assertTrue(torch.allclose(after, 1.2 * before))

    def test_fit_recovers_known_affine_correction(self) -> None:
        generator = torch.Generator().manual_seed(7)
        ensemble = torch.randn(2, 8, 6, 2, 2, generator=generator) * 0.2
        truth = self.persistence + 1.25 * (ensemble[:, 0] - self.persistence)
        truth[:, 0::2] += 0.04
        truth[:, 1::2] -= 0.02
        fitted = fit_persistence_affine(ensemble, truth, self.persistence, self.mask, self.means, self.stds)
        self.assertGreaterEqual(fitted["scale"], 0.8)
        self.assertLessEqual(fitted["scale"], 1.5)
        self.assertGreaterEqual(fitted["sic_offset"], -0.1)
        self.assertLessEqual(fitted["sic_offset"], 0.1)
        self.assertLessEqual(fitted["objective"], fitted["objective_identity"] + 1e-6)
        self.assertLessEqual(fitted["objective_scale1_profiled_offsets"], fitted["objective_identity"] + 1e-6)

    def test_reported_objective_matches_independent_pairwise_reference(self) -> None:
        generator = torch.Generator().manual_seed(19)
        means = torch.tensor([0.2, 0.6, 0.2, 0.6, 0.2, 0.6])
        stds = torch.tensor([0.3, 0.8, 0.3, 0.8, 0.3, 0.8])
        persistence = torch.randn(3, 6, 2, 2, generator=generator) * stds[None, :, None, None]
        persistence += means[None, :, None, None]
        ensemble = (
            persistence[:, None]
            + torch.randn(3, 8, 6, 2, 2, generator=generator) * stds[None, None, :, None, None] * 0.25
        )
        truth = persistence + torch.randn(3, 6, 2, 2, generator=generator) * stds[None, :, None, None] * 0.3
        mask = torch.ones(3, 1, 2, 2)
        fitted = fit_persistence_affine(ensemble, truth, persistence, mask, means, stds)
        candidate = apply_persistence_affine(
            ensemble,
            persistence,
            means,
            stds,
            scale=fitted["scale"],
            sic_offset=fitted["sic_offset"],
            sit_offset=fitted["sit_offset"],
        )

        standardized_members = candidate / stds[None, None, :, None, None]
        standardized_truth = truth / stds[None, :, None, None]
        output_scores = []
        for channel in range(6):
            member_values = standardized_members[:, :, channel].permute(0, 2, 3, 1).reshape(-1, 8)
            target_values = standardized_truth[:, channel].reshape(-1, 1)
            first = (member_values - target_values).abs().mean()
            pairwise = (member_values[:, :, None] - member_values[:, None, :]).abs().sum(
                dim=(1, 2)
            ).mean() / (2 * 8 * 7)
            output_scores.append(first - pairwise)
        direct_objective = torch.stack(output_scores).mean().item()
        self.assertAlmostEqual(fitted["objective"], direct_objective, places=6)
        self.assertLessEqual(fitted["objective"], fitted["objective_identity"] + 1e-6)

    def test_coverage_uses_attainable_eight_member_references(self) -> None:
        ensemble = torch.zeros(1, 8, 6, 1, 1)
        truth = torch.zeros(1, 6, 1, 1)
        metrics = coverage_metrics(ensemble, truth, torch.ones(1, 1, 1, 1))
        self.assertAlmostEqual(metrics["d3_sic"]["outer_1_8"]["coverage"], 7.0 / 9.0)
        self.assertEqual(metrics["d3_sic"]["outer_1_8"]["iid_reference"], 7.0 / 9.0)
        self.assertEqual(metrics["d3_sic"]["inner_2_7"]["iid_reference"], 5.0 / 9.0)

    def test_coverage_fractionalizes_partial_ties(self) -> None:
        members = torch.tensor([0, 0, 0, 1, 2, 3, 4, 5], dtype=torch.float32)
        ensemble = members.reshape(1, 8, 1, 1, 1).expand(1, 8, 6, 1, 1)
        truth = torch.zeros(1, 6, 1, 1)
        metrics = coverage_metrics(ensemble, truth, torch.ones(1, 1, 1, 1))
        self.assertEqual(metrics["d3_sic"]["outer_1_8"]["coverage"], 0.75)
        self.assertEqual(metrics["d3_sic"]["inner_2_7"]["coverage"], 0.5)

    def test_joint_energy_accepts_runtime_mask_shape(self) -> None:
        ensemble = torch.zeros(2, 8, 6, 2, 2)
        truth = torch.zeros(2, 6, 2, 2)
        self.assertEqual(joint_energy_score(ensemble, truth, self.mask, self.stds), 0.0)

    def test_boundary_events_and_haze_are_explicit(self) -> None:
        ensemble = torch.zeros(2, 8, 6, 2, 2)
        truth = torch.zeros(2, 6, 2, 2)
        metrics = boundary_event_metrics(ensemble, truth, self.mask)
        self.assertEqual(metrics["d3_sic"]["sic_le_0p01"]["brier"], 0.0)
        haze = metrics["d3_sit"]["open_water_haze"]
        self.assertEqual(haze["mean_positive_sit"], 0.0)
        self.assertEqual(haze["p95_positive_sit"], 0.0)
        self.assertEqual(haze["fraction_gt_0p01"], 0.0)
        self.assertEqual(
            metrics["d3_sic"]["sic_le_0p01"]["reliability"]["8/8"]["case_equal_observed_frequency"],
            1.0,
        )

    def test_extra_gate_rejects_open_water_haze_growth_above_one_centimetre(self) -> None:
        truth = torch.zeros(2, 6, 2, 2)
        baseline_ensemble = torch.full((2, 8, 6, 2, 2), 0.005)
        candidate_ensemble = torch.full((2, 8, 6, 2, 2), 0.02)

        def diagnostics(ensemble: torch.Tensor) -> dict:
            per_case = []
            for _ in range(2):
                per_case.append(
                    {
                        "leads": {
                            lead: {field: {"member_roughness": 1.0} for field in ("sic", "sit")}
                            for lead in ("d3", "d6", "d9")
                        },
                        "temporal_change": {"d3_to_d6": 1.0, "d6_to_d9": 1.0},
                    }
                )
            return {
                "coverage": coverage_metrics(ensemble, truth, self.mask),
                "joint_energy_score": 1.0,
                "boundary_events": boundary_event_metrics(ensemble, truth, self.mask),
                "scores": {"per_case": per_case},
            }

        gate = _extra_gate(diagnostics(candidate_ensemble), diagnostics(baseline_ensemble))
        self.assertFalse(gate["passed"])
        self.assertTrue(any("open_water_haze" in failure for failure in gate["failures"]))

    def test_temporal_comparison_uses_case_equal_rmse(self) -> None:
        def diagnostics(values: tuple[float, float]) -> dict:
            cases = []
            for value in values:
                cases.append(
                    {
                        "leads": {
                            lead: {field: {"member_roughness": 1.0} for field in ("sic", "sit")}
                            for lead in ("d3", "d6", "d9")
                        },
                        "temporal_change": {"d3_to_d6": value},
                    }
                )
            return {"scores": {"per_case": cases}}

        comparison = _structural_comparison(diagnostics((0.0, 2.0)), diagnostics((1.0, 1.0)))
        self.assertAlmostEqual(comparison["temporal_rmse_ratios"]["d3_to_d6"], 2**0.5)

    def test_primary_score_identity_ratio_is_one(self) -> None:
        generator = torch.Generator().manual_seed(23)
        ensemble = torch.randn(2, 8, 6, 2, 2, generator=generator)
        truth = torch.randn(2, 6, 2, 2, generator=generator)
        score = standardized_fair_crps(ensemble, truth, self.mask, self.stds)
        comparison = primary_score_comparison(score, score)
        self.assertAlmostEqual(comparison["ratio"], 1.0)
        self.assertAlmostEqual(comparison["paired_date_bootstrap_upper95"], 1.0)


if __name__ == "__main__":
    unittest.main()
