from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import torch
from diffusers.optimization import get_cosine_schedule_with_warmup

from assim_lib.config import TrainingConfig
from assim_lib.flow_parameterization import (
    GAUSSIAN_PATH_PRECONDITIONED,
    RAW_VELOCITY,
    gaussian_path_scale,
    reconstruct_velocity,
    velocity_model_state,
)


class GaussianPathPreconditioningTests(unittest.TestCase):
    def test_paired_panel_is_frozen_and_protocol_hash_bound(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        panel_path = repo / "paper/STRUCTURED_PAIRED_PILOT_PANEL.json"
        protocol = json.loads(
            (repo / "paper/STRUCTURED_GAUSSIAN_PRECONDITIONED_PILOT_PROTOCOL.json")
            .read_text(encoding="utf-8")
        )
        panel = json.loads(panel_path.read_text(encoding="utf-8"))
        digest = hashlib.sha256(panel_path.read_bytes()).hexdigest()
        self.assertEqual(
            protocol["evaluation"]["paired_panel_manifest_sha256"], digest
        )
        self.assertEqual(
            [row["dataset_index"] for row in panel["cases"]],
            [0, 45, 90, 135, 180, 225, 270, 315],
        )
        self.assertEqual(
            [row["season"] for row in panel["cases"]],
            ["DJF", "DJF", "MAM", "MAM", "JJA", "JJA", "SON", "SON"],
        )
        self.assertEqual(len(set(panel["member_seeds"])), 8)

    def test_pilot_lr_matches_reference_through_step_2128(self) -> None:
        reference_parameter = torch.nn.Parameter(torch.zeros(()))
        pilot_parameter = torch.nn.Parameter(torch.zeros(()))
        reference_optimizer = torch.optim.AdamW([reference_parameter], lr=1e-4)
        pilot_optimizer = torch.optim.AdamW([pilot_parameter], lr=1e-4)
        schedulers = [
            get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=200,
                num_training_steps=4123,
            )
            for optimizer in (reference_optimizer, pilot_optimizer)
        ]
        for _ in range(2128):
            for optimizer, scheduler in zip(
                (reference_optimizer, pilot_optimizer), schedulers, strict=True
            ):
                optimizer.step()
                scheduler.step()
            self.assertEqual(
                reference_optimizer.param_groups[0]["lr"],
                pilot_optimizer.param_groups[0]["lr"],
            )

    def test_scale_has_exact_endpoint_and_midpoint_values(self) -> None:
        state = torch.ones(3, 2, 4, 5, dtype=torch.float64)
        _, scale = gaussian_path_scale(
            state, torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        )
        expected = torch.tensor(
            [1.0, 2.0**-0.5, 1.0], dtype=torch.float64
        ).view(3, 1, 1, 1)
        torch.testing.assert_close(scale, expected, rtol=1e-14, atol=1e-14)

    def test_reparameterization_can_represent_any_full_velocity(self) -> None:
        generator = torch.Generator().manual_seed(1701)
        state = torch.randn(5, 4, 3, 2, generator=generator, dtype=torch.float64)
        target = torch.randn(5, 4, 3, 2, generator=generator, dtype=torch.float64)
        timesteps = torch.tensor([0.0, 0.13, 0.5, 0.87, 1.0], dtype=torch.float64)
        time, scale = gaussian_path_scale(state, timesteps)
        analytic = (2.0 * time - 1.0) * state / scale.square()
        residual_output = scale * (target - analytic)
        actual = reconstruct_velocity(
            residual_output,
            state,
            timesteps,
            GAUSSIAN_PATH_PRECONDITIONED,
        )
        torch.testing.assert_close(actual, target, rtol=1e-13, atol=1e-13)

    def test_gaussian_conditional_velocity_needs_zero_neural_residual(self) -> None:
        state = torch.tensor([[[[2.0]]], [[[3.0]]]], dtype=torch.float64)
        timesteps = torch.tensor([0.25, 0.75], dtype=torch.float64)
        actual = reconstruct_velocity(
            torch.zeros_like(state),
            state,
            timesteps,
            GAUSSIAN_PATH_PRECONDITIONED,
        )
        time, scale = gaussian_path_scale(state, timesteps)
        expected = (2.0 * time - 1.0) * state / scale.square()
        torch.testing.assert_close(actual, expected)

    def test_raw_parameterization_is_exact_identity(self) -> None:
        state = torch.randn(2, 3, 4, 5)
        output = torch.randn_like(state)
        timesteps = torch.tensor([0.2, 0.8])
        self.assertIs(velocity_model_state(state, timesteps, RAW_VELOCITY), state)
        self.assertIs(
            reconstruct_velocity(output, state, timesteps, RAW_VELOCITY), output
        )

    def test_invalid_time_and_shape_fail_closed(self) -> None:
        state = torch.ones(2, 1, 1, 1)
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            velocity_model_state(
                state,
                torch.tensor([-0.1, 0.5]),
                GAUSSIAN_PATH_PRECONDITIONED,
            )
        with self.assertRaisesRegex(ValueError, "one entry per state"):
            velocity_model_state(
                state,
                torch.tensor([0.1, 0.2, 0.3]),
                GAUSSIAN_PATH_PRECONDITIONED,
            )

    def test_non_structured_config_rejects_preconditioning(self) -> None:
        with self.assertRaisesRegex(ValueError, "only valid"):
            TrainingConfig(
                training_objective="flow",
                structured_velocity_parameterization=(
                    GAUSSIAN_PATH_PRECONDITIONED
                ),
            ).validate()


if __name__ == "__main__":
    unittest.main()
