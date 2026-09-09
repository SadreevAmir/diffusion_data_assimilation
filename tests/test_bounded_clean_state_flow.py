import unittest

import torch
from torch import nn

from assim_lib.bounded_clean_state_flow import (
    BoundedCleanStateSampler,
    bounded_clean_state_loss,
    clean_prediction_velocity,
    decode_bounded_sic_sit,
    endpoint_variance_deficit_report,
    finite_mixture_oracle_report,
)


class _ConstantBoundedModel(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        logits = torch.empty(channels)
        logits[0::2] = torch.logit(torch.tensor(0.8))
        logits[1::2] = torch.log(torch.expm1(torch.tensor(0.6)))
        self.register_buffer("logits", logits)

    def forward(self, value, timestep):
        del timestep
        return (self.logits[None, :, None, None].expand(value.shape[0], -1, *value.shape[-2:]),)


class BoundedCleanStateFlowTests(unittest.TestCase):
    def test_decoder_support_and_clean_loss_gradient(self) -> None:
        logits = torch.tensor(
            [[[[100.0]], [[-100.0]], [[-100.0]], [[100.0]]]],
            requires_grad=True,
        )
        physical = decode_bounded_sic_sit(logits)
        self.assertTrue(torch.all((physical[:, 0::2] >= 0.0) & (physical[:, 0::2] <= 1.0)))
        self.assertTrue(torch.all(physical[:, 1::2] >= 0.0))
        truth = torch.zeros_like(logits)
        loss = bounded_clean_state_loss(
            logits,
            truth,
            torch.ones((1, 1, 1, 1)),
            means=(0.2, 0.3, 0.2, 0.3),
            stds=(0.4, 0.5, 0.4, 0.5),
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.all(torch.isfinite(logits.grad)))

    def test_inactive_nonfinite_values_are_neutralized_before_backward(self) -> None:
        logits = torch.zeros((1, 2, 1, 2), requires_grad=True)
        with torch.no_grad():
            logits[..., 1] = float("nan")
        truth = torch.zeros_like(logits)
        truth[..., 1] = float("inf")
        loss = bounded_clean_state_loss(
            logits,
            truth,
            torch.tensor([[[[1.0, 0.0]]]]),
            means=(0.2, 0.3),
            stds=(0.4, 0.5),
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.all(torch.isfinite(logits.grad)))
        self.assertTrue(torch.all(logits.grad[..., 1] == 0))

    def test_active_nonfinite_logits_fail_closed(self) -> None:
        logits = torch.tensor([[[[float("inf")]], [[0.0]]]])
        with self.assertRaises(FloatingPointError):
            bounded_clean_state_loss(
                logits,
                torch.zeros_like(logits),
                torch.ones((1, 1, 1, 1)),
                means=(0.2, 0.3),
                stds=(0.4, 0.5),
            )

    def test_finite_logits_that_overflow_mse_fail_closed(self) -> None:
        logits = torch.tensor([[[[0.0]], [[1e20]]]], dtype=torch.float32)
        with self.assertRaises(FloatingPointError):
            bounded_clean_state_loss(
                logits,
                torch.zeros_like(logits),
                torch.ones((1, 1, 1, 1)),
                means=(0.2, 0.3),
                stds=(0.4, 0.5),
            )

    def test_velocity_has_linear_path_sign_and_masks_land(self) -> None:
        clean = torch.tensor([[[[2.0, 2.0]], [[-1.0, -1.0]]]])
        noise = torch.tensor([[[[4.0, 4.0]], [[3.0, 3.0]]]])
        time = torch.tensor([0.25])
        state = (1.0 - time[:, None, None, None]) * clean + time[:, None, None, None] * noise
        mask = torch.tensor([[[[1.0, 0.0]]]])
        velocity = clean_prediction_velocity(
            state, clean, time, minimum_time=0.01, valid_mask=mask
        )
        self.assertTrue(torch.equal(velocity[..., 0], (noise - clean)[..., 0]))
        self.assertTrue(torch.all(velocity[..., 1] == 0))

    def test_sampler_returns_bounded_endpoint_and_zero_land(self) -> None:
        channels = 6
        initial_noise = torch.randn((2, channels, 2, 3))
        conditioning = torch.zeros((2, 15, 2, 3))
        valid = torch.ones((2, 1, 2, 3))
        valid[..., 0, :] = 0
        sampler = BoundedCleanStateSampler(
            (model := _ConstantBoundedModel(channels)),
            means=(0.2, 0.3) * 3,
            stds=(0.4, 0.5) * 3,
            epsilon=0.02,
        )
        result = sampler.sample(initial_noise, conditioning, valid, num_steps=8)
        self.assertTrue(torch.all(result.physical[:, 0::2] >= 0.0))
        self.assertTrue(torch.all(result.physical[:, 0::2] <= 1.0))
        self.assertTrue(torch.all(result.physical[:, 1::2] >= 0.0))
        self.assertTrue(torch.all(result.physical[..., 0, :] == 0))
        self.assertTrue(torch.all(result.normalized[..., 0, :] == 0))
        self.assertTrue(torch.all(result.terminal_ode_state[..., 0, :] == 0))
        clean = torch.empty_like(initial_noise)
        clean[:, 0::2] = (0.8 - 0.2) / 0.4
        clean[:, 1::2] = (0.6 - 0.3) / 0.5
        expected_terminal = clean + 0.02 * (initial_noise - clean)
        active = valid.expand_as(initial_noise) > 0
        self.assertTrue(
            torch.allclose(result.terminal_ode_state[active], expected_terminal[active], atol=2e-5)
        )
        self.assertTrue(model.training)

    def test_rk4_endpoint_nodes_are_safe_across_dtypes(self) -> None:
        for dtype in (torch.float32, torch.float64):
            for epsilon, steps in ((0.01, 32), (0.02, 7), (0.001, 13)):
                model = _ConstantBoundedModel(2).to(dtype=dtype)
                sampler = BoundedCleanStateSampler(
                    model,
                    means=(0.2, 0.3),
                    stds=(0.4, 0.5),
                    epsilon=epsilon,
                )
                result = sampler.sample(
                    torch.randn((2, 2, 1, 2), dtype=dtype),
                    torch.zeros((2, 0, 1, 2), dtype=dtype),
                    torch.ones((2, 1, 1, 2), dtype=dtype),
                    num_steps=steps,
                )
                self.assertTrue(torch.all(torch.isfinite(result.terminal_ode_state)))

    def test_masks_are_binary_and_nonempty(self) -> None:
        logits = torch.zeros((1, 2, 1, 1))
        for mask in (torch.zeros((1, 1, 1, 1)), torch.full((1, 1, 1, 1), 0.5)):
            with self.assertRaises(ValueError):
                bounded_clean_state_loss(
                    logits,
                    torch.zeros_like(logits),
                    mask,
                    means=(0.2, 0.3),
                    stds=(0.4, 0.5),
                )

    def test_finite_mixture_oracle_recovers_joint_law(self) -> None:
        report = finite_mixture_oracle_report(sample_count=1024, num_steps=32)
        self.assertGreaterEqual(report["support_min_sic"], 0.0)
        self.assertLessEqual(report["support_max_sic"], 1.0)
        self.assertGreaterEqual(report["support_min_sit"], 0.0)
        self.assertLess(report["mean_linf_error"], 0.035)
        self.assertLess(report["covariance_linf_error"], 0.08)
        self.assertLess(report["component_frequency_linf_error"], 0.035)
        self.assertLess(report["mean_nearest_atom_distance"], 0.01)
        self.assertLess(report["doubled_steps_mean_abs_difference"], 2e-5)
        self.assertLess(report["halved_epsilon_mean_abs_difference"], 0.01)
        self.assertLess(report["production_vs_reference_mean_abs_difference"], 1e-8)
        self.assertLess(report["reverse_probe_distance_ratio"], 1.0)
        self.assertGreater(report["wrong_sign_component_frequency_linf_error"], 0.2)
        self.assertTrue(all(ratio > 0.95 for ratio in report["endpoint_to_terminal_std_ratio"]))

    def test_finite_epsilon_endpoint_deficit_is_measured(self) -> None:
        report = endpoint_variance_deficit_report(sample_count=4096, epsilon=0.05)
        self.assertTrue(all(0.0 < value < 1.0 for value in report["variance_ratio"]))
        self.assertTrue(all(value > 0.0 for value in report["variance_deficit"]))


if __name__ == "__main__":
    unittest.main()
