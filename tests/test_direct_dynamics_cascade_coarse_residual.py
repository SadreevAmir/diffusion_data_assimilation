import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch import nn

from assim_lib.direct_dynamics_cascade_coarse_residual import (
    CoarsePersistenceResidualSampler,
    CoarsePersistenceResidualTrainer,
    CoarseStandardizedPersistenceResidualSampler,
    coarse_persistence_from_condition,
    coarse_residual_flow_pair,
    coarse_standardized_residual_flow_pair,
    publish_residual_mechanics_gate,
    residual_mechanics_gate,
    residual_model_input_for_test,
    standardize_coarse_residual,
    unstandardize_coarse_residual,
)
from assim_lib.direct_dynamics_cascade_residual_stats import CaseEqualResidualMoments


def _condition(d0: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    condition = torch.zeros((d0.shape[0], 15, *d0.shape[-2:]), dtype=d0.dtype)
    condition[:, :2] = d0
    condition[:, 2:3] = valid
    return condition


class CoarsePersistenceResidualTests(unittest.TestCase):
    def test_residual_round_trip_and_flow_sign_are_exact_on_ocean(self) -> None:
        valid = torch.ones((1, 1, 4, 4))
        d0 = torch.stack((torch.full((4, 4), 0.4), torch.full((4, 4), 0.8)))[None]
        condition = _condition(d0, valid)
        persistence = d0.repeat(1, 3, 1, 1)
        truth = persistence + 0.25
        noise = torch.full((1, 6, 2, 2), -0.5)
        state, velocity, residual, coarse_persistence, fraction = coarse_residual_flow_pair(
            truth, noise, condition, valid, torch.tensor([0.25])
        )
        clean = coarse_persistence + residual
        self.assertTrue(torch.allclose(clean, torch.nn.functional.avg_pool2d(truth, 2), atol=1e-7))
        self.assertTrue(torch.allclose(residual, torch.full_like(residual, 0.25), atol=1e-7))
        self.assertTrue(torch.allclose(velocity, noise - residual, atol=1e-7))
        self.assertTrue(torch.allclose(state, 0.75 * residual + 0.25 * noise))
        self.assertTrue(torch.equal(fraction, torch.ones_like(fraction)))

    def test_land_is_zero_and_never_receives_residual_noise(self) -> None:
        valid = torch.ones((1, 1, 4, 4))
        valid[..., :2, :2] = 0
        d0 = torch.ones((1, 2, 4, 4))
        d0[..., :2, :2] = 0
        condition = _condition(d0, valid)
        truth = d0.repeat(1, 3, 1, 1)
        noise = torch.full((1, 6, 2, 2), 7.0)
        state, velocity, residual, persistence, _ = coarse_residual_flow_pair(
            truth, noise, condition, valid, torch.tensor([0.5])
        )
        for value in (state, velocity, residual, persistence):
            self.assertTrue(torch.all(value[..., 0, 0] == 0))

    def test_sampler_oracle_reconstructs_persistence_plus_residual(self) -> None:
        class ConstantVelocity(nn.Module):
            def __init__(self, velocity):
                super().__init__()
                self.register_buffer("velocity", velocity)

            def forward(self, value, timestep, return_dict=False):
                del timestep, return_dict
                return (self.velocity.expand(value.shape[0], -1, -1, -1),)

        valid = torch.ones((1, 1, 4, 4))
        d0 = torch.stack((torch.full((4, 4), 0.2), torch.full((4, 4), 0.7)))[None]
        condition = _condition(d0, valid)
        noise = torch.full((1, 6, 2, 2), 0.9)
        desired_residual = torch.full_like(noise, -0.1)
        sampler = CoarsePersistenceResidualSampler(ConstantVelocity(noise - desired_residual))
        reconstructed = sampler.sample_conditioned(
            structured_conditioning=condition,
            valid_mask=valid,
            initial_noise=noise,
            num_timesteps=17,
            device=torch.device("cpu"),
            method="rk4",
            end_time=0.0,
        )
        persistence, _, _ = coarse_persistence_from_condition(condition, valid)
        self.assertTrue(torch.allclose(sampler.last_residual, desired_residual, atol=1e-6))
        self.assertTrue(torch.allclose(reconstructed, persistence + desired_residual, atol=1e-6))

    def test_model_input_keeps_exact_lossless_56_channel_contract(self) -> None:
        valid = torch.ones((2, 1, 8, 8))
        condition = _condition(torch.zeros((2, 2, 8, 8)), valid)
        state = torch.zeros((2, 6, 4, 4))
        model_input = residual_model_input_for_test(state, condition, valid)
        self.assertEqual(model_input.shape, (2, 56, 4, 4))

    def test_low_ice_metric_cannot_hide_symmetric_member_noise(self) -> None:
        ensemble = torch.zeros((1, 2, 6, 2, 2))
        ensemble[:, 0, 1::2] = 0.1
        ensemble[:, 1, 1::2] = -0.1
        truth = torch.zeros((1, 6, 2, 2))
        metrics = CoarsePersistenceResidualTrainer._low_ice_metrics(
            ensemble, truth, torch.ones((1, 1, 2, 2))
        )
        self.assertAlmostEqual(metrics["d3_low_ice_raw_sit_rms"], 0.1, places=6)
        self.assertAlmostEqual(metrics["d3_low_ice_raw_sit_mae"], 0.1, places=6)

    def test_gate_requires_uniform_improvement_skill_and_tail_control(self) -> None:
        baseline = {}
        final = {}
        for lead in (3, 6, 9):
            for field in ("sic", "sit"):
                key = f"d{lead}_{field}_raw_mean_rmse"
                baseline[key] = 1.0
                final[key] = 0.8
                final[f"d{lead}_{field}_persistence_rmse"] = 0.9
        for field in ("sic", "sit"):
            for suffix in (
                "raw_support_excess_mean_all",
                "raw_support_excess_p95_all",
                "raw_support_excess_max",
                "raw_support_violation_fraction",
            ):
                key = f"{field}_{suffix}"
                baseline[key] = 0.01
                final[key] = 0.009
        passed = residual_mechanics_gate(final, baseline)
        self.assertEqual(passed["status"], "numeric_pass_pending_visual_review")
        final["d3_sic_raw_mean_rmse"] = 1.03
        failed = residual_mechanics_gate(final, baseline)
        self.assertEqual(failed["decision"], "reject_residual_candidate")

        final["d3_sic_raw_mean_rmse"] = 0.8
        final["sic_raw_support_excess_max"] = 0.011
        tail_failed = residual_mechanics_gate(final, baseline)
        self.assertEqual(tail_failed["decision"], "reject_residual_candidate")
        self.assertFalse(
            tail_failed["criteria"]["raw_support_mean_p95_max_and_frequency_not_worse"]
        )

    def test_terminal_publication_uses_the_same_residual_gate_everywhere(self) -> None:
        class Tracker:
            def __init__(self):
                self.calls = []

            def report_single_value(self, key, value):
                self.calls.append((key, value))

        gate = {
            "status": "numeric_pass_pending_visual_review",
            "decision": "hold_for_visual_review",
        }
        history = [{"epoch": 1, "val_loss": 0.5}]
        tracker = Tracker()
        with TemporaryDirectory() as temporary:
            publish_residual_mechanics_gate(
                output_dir=temporary,
                val_history=history,
                clearml=tracker,
                gate=gate,
            )
            import json

            gate_artifact = json.loads(
                (Path(temporary) / "coarse_mechanics_gate.json").read_text(encoding="utf-8")
            )
            metrics_artifact = json.loads(
                (Path(temporary) / "metrics.json").read_text(encoding="utf-8")
            )
        self.assertEqual(gate_artifact, gate)
        self.assertEqual(metrics_artifact[-1]["coarse_mechanics_gate"], gate)
        self.assertEqual(
            tracker.calls,
            [("coarse_residual_mechanics_gate_passed", 1.0)],
        )

    def test_standardized_residual_round_trip_and_flow_sign(self) -> None:
        valid = torch.ones((1, 1, 4, 4))
        d0 = torch.stack((torch.full((4, 4), 0.4), torch.full((4, 4), 0.8)))[None]
        condition = _condition(d0, valid)
        truth = d0.repeat(1, 3, 1, 1) + 0.25
        statistics = {
            "means": [0.05, -0.10, 0.05, -0.10, 0.05, -0.10],
            "stds": [0.10, 0.20, 0.10, 0.20, 0.10, 0.20],
        }
        noise = torch.full((1, 6, 2, 2), -0.5)
        state, velocity, residual, persistence, fraction = (
            coarse_standardized_residual_flow_pair(
                truth,
                noise,
                condition,
                valid,
                torch.tensor([0.25]),
                statistics,
            )
        )
        active = (fraction > 0).float()
        whitened = standardize_coarse_residual(residual, active, statistics)
        self.assertTrue(torch.allclose(state, 0.75 * whitened + 0.25 * noise))
        self.assertTrue(torch.allclose(velocity, noise - whitened))
        recovered = unstandardize_coarse_residual(whitened, active, statistics)
        self.assertTrue(torch.allclose(persistence + recovered, torch.nn.functional.avg_pool2d(truth, 2)))

    def test_standardized_residual_is_invariant_to_channel_units(self) -> None:
        generator = torch.Generator().manual_seed(7)
        residual = torch.randn((2, 6, 3, 2), generator=generator)
        active = torch.ones((2, 1, 3, 2))
        means = torch.tensor([0.2, -0.4, 0.1, 0.6, -0.2, 0.3])
        stds = torch.tensor([0.3, 0.7, 0.2, 1.1, 0.9, 0.4])
        scales = torch.tensor([2.0, 1000.0, 0.5, 10.0, 4.0, 0.25]).reshape(1, 6, 1, 1)
        base = standardize_coarse_residual(
            residual, active, {"means": means.tolist(), "stds": stds.tolist()}
        )
        changed = standardize_coarse_residual(
            residual * scales,
            active,
            {"means": (means * scales.flatten()).tolist(), "stds": (stds * scales.flatten()).tolist()},
        )
        self.assertTrue(torch.allclose(base, changed, atol=1e-6))

    def test_affine_inverse_preserves_correlated_gaussian_moments(self) -> None:
        generator = torch.Generator().manual_seed(19)
        mixing = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.4, 0.8, 0.0, 0.0, 0.0, 0.0],
                [0.2, -0.1, 0.9, 0.0, 0.0, 0.0],
                [0.0, 0.3, 0.1, 0.7, 0.0, 0.0],
                [0.1, 0.0, 0.2, -0.2, 0.8, 0.0],
                [-0.1, 0.2, 0.0, 0.1, 0.3, 0.7],
            ]
        )
        standardized = (torch.randn((50000, 6), generator=generator) @ mixing.T).reshape(
            50000, 6, 1, 1
        )
        means = torch.tensor([0.2, -0.4, 0.1, 0.6, -0.2, 0.3])
        stds = torch.tensor([0.3, 0.7, 0.2, 1.1, 0.9, 0.4])
        physical = unstandardize_coarse_residual(
            standardized,
            torch.ones((50000, 1, 1, 1)),
            {"means": means.tolist(), "stds": stds.tolist()},
        ).reshape(50000, 6)
        empirical_mean = physical.mean(dim=0)
        empirical_covariance = torch.cov(physical.T)
        expected_covariance = torch.diag(stds) @ (mixing @ mixing.T) @ torch.diag(stds)
        self.assertTrue(torch.allclose(empirical_mean, means, atol=0.015, rtol=0.0))
        self.assertTrue(torch.allclose(empirical_covariance, expected_covariance, atol=0.015, rtol=0.04))

    def test_standardized_sampler_oracle_inverts_nonzero_affine_map(self) -> None:
        class ConstantVelocity(nn.Module):
            def __init__(self, velocity):
                super().__init__()
                self.register_buffer("velocity", velocity)

            def forward(self, value, timestep, return_dict=False):
                del timestep, return_dict
                return (self.velocity.expand(value.shape[0], -1, -1, -1),)

        valid = torch.ones((1, 1, 4, 4))
        d0 = torch.stack((torch.full((4, 4), 0.2), torch.full((4, 4), 0.7)))[None]
        condition = _condition(d0, valid)
        noise = torch.full((1, 6, 2, 2), 0.9)
        desired_whitened = torch.full_like(noise, -0.1)
        statistics = {
            "means": [0.05, -0.10, 0.05, -0.10, 0.05, -0.10],
            "stds": [0.10, 0.20, 0.10, 0.20, 0.10, 0.20],
        }
        sampler = CoarseStandardizedPersistenceResidualSampler(
            ConstantVelocity(noise - desired_whitened), statistics
        )
        reconstructed = sampler.sample_conditioned(
            structured_conditioning=condition,
            valid_mask=valid,
            initial_noise=noise,
            num_timesteps=17,
            device=torch.device("cpu"),
            method="rk4",
            end_time=0.0,
        )
        persistence, active, _ = coarse_persistence_from_condition(condition, valid)
        desired_residual = unstandardize_coarse_residual(
            desired_whitened, active, statistics
        )
        self.assertTrue(torch.allclose(sampler.last_residual, desired_residual, atol=1e-6))
        self.assertTrue(torch.allclose(reconstructed, persistence + desired_residual, atol=1e-6))

    def test_case_equal_residual_moments_do_not_overweight_large_ocean_case(self) -> None:
        accumulator = CaseEqualResidualMoments(channels=1)
        residual = torch.tensor([[[[0.0, 2.0]]], [[[10.0, 99.0]]]])
        fraction = torch.tensor([[[[1.0, 1.0]]], [[[1.0, 0.0]]]])
        accumulator.update(residual, fraction)
        result = accumulator.finalize()
        self.assertAlmostEqual(result["means"][0], 5.5)
        self.assertAlmostEqual(result["variances"][0], 20.75)


if __name__ == "__main__":
    unittest.main()
