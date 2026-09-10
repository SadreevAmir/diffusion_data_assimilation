import unittest
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch
from diffusers.training_utils import EMAModel
from torch import nn

from assim_lib.config import TrainingConfig
from assim_lib.direct_dynamics_cascade import masked_block_average, project_detail
from assim_lib.direct_dynamics_cascade_fine import (
    FINE_CONDITION_CHANNELS,
    FINE_INPUT_CHANNELS,
    FineCascadeDynamicsTrainer,
    FineCascadeSampler,
    ProjectedDetailModel,
    fine_model_input_for_test,
    generated_coarse_condition,
    load_fine_cascade_sampler,
    teacher_coarse_condition,
    validate_fine_condition,
)
from assim_lib.model_io import build_unet


class ZeroModel(nn.Module):
    def forward(self, model_input, timestep, return_dict=False):
        del timestep, return_dict
        return (torch.zeros_like(model_input[:, :6]),)


class ConstantVelocityModel(nn.Module):
    def __init__(self, velocity: torch.Tensor):
        super().__init__()
        self.register_buffer("velocity", velocity)

    def forward(self, model_input, timestep, return_dict=False):
        del timestep, return_dict
        return (self.velocity.expand(model_input.shape[0], -1, -1, -1),)


class TrainableVelocityModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(FINE_INPUT_CHANNELS, 6, 1, bias=True)

    def forward(self, model_input, timestep, return_dict=False):
        del timestep, return_dict
        return (self.projection(model_input),)


class FailureHarness(FineCascadeDynamicsTrainer):
    def __init__(self):
        pass

    @contextmanager
    def _sampling_model(self):
        yield self.model

    def _normalized_to_physical(self, value):
        return value.float()


class FineCascadeIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.generator = torch.Generator().manual_seed(909)
        self.truth = torch.randn(2, 6, 8, 10, generator=self.generator)
        self.condition = torch.randn(2, 15, 8, 10, generator=self.generator)
        self.mask = torch.ones(2, 1, 8, 10)
        self.mask[:, :, :2, :2] = 0
        self.condition[:, 2:3] = self.mask

    def test_teacher_condition_preserves_causal_channels_and_exact_decomposition(self) -> None:
        condition, coarse, residual = teacher_coarse_condition(self.truth, self.condition, self.mask)
        self.assertEqual(condition.shape[1], FINE_CONDITION_CHANNELS)
        self.assertTrue(torch.equal(condition[:, :15], self.condition))
        residual_coarse, fraction = masked_block_average(residual, self.mask)
        self.assertLess(float(residual_coarse[fraction > 0].abs().max()), 3e-6)
        recovered_coarse, _ = masked_block_average(condition[:, -6:] + residual, self.mask)
        self.assertTrue(torch.allclose(recovered_coarse, coarse, atol=3e-6, rtol=0))

    def test_condition_rejects_mask_mismatch_and_detail_contaminated_lift(self) -> None:
        wrong_mask = self.mask.clone()
        wrong_mask[:, :, -1, -1] = 0
        with self.assertRaisesRegex(ValueError, "embedded dynamics mask"):
            teacher_coarse_condition(self.truth, self.condition, wrong_mask)

        condition, _, _ = teacher_coarse_condition(self.truth, self.condition, self.mask)
        contaminated = condition.clone()
        detail = project_detail(torch.randn(self.truth.shape, generator=self.generator), self.mask)
        contaminated[:, -6:] += 0.1 * detail
        with self.assertRaisesRegex(ValueError, "detail-contaminated"):
            validate_fine_condition(contaminated)

    def test_generated_coarse_constructor_is_canonical_and_member_bound(self) -> None:
        coarse, _ = masked_block_average(self.truth, self.mask)
        condition = generated_coarse_condition(self.condition, coarse, self.mask)
        mask, lift = validate_fine_condition(condition, self.mask)
        recovered, _ = masked_block_average(lift, mask)
        self.assertTrue(torch.allclose(recovered, coarse, atol=3e-6, rtol=0))

    def test_model_input_layout_and_velocity_projection(self) -> None:
        state = torch.randn(2, 6, 8, 10, generator=self.generator)
        model_input = fine_model_input_for_test(state, self.condition, self.truth, self.mask)
        self.assertEqual(model_input.shape[1], FINE_INPUT_CHANNELS)
        projected = ProjectedDetailModel(ZeroModel())(model_input, torch.full((2,), 500.0))[0]
        coarse, fraction = masked_block_average(projected, self.mask)
        self.assertLess(float(coarse[fraction > 0].abs().max()), 1e-7)

    def test_training_projection_stays_fp32_and_actual_loss_backpropagates(self) -> None:
        state = torch.randn(2, 6, 8, 10, generator=self.generator)
        model_input = fine_model_input_for_test(state, self.condition, self.truth, self.mask)
        bf16_projected = ProjectedDetailModel(ZeroModel())(model_input.bfloat16(), torch.full((2,), 500.0))[0]
        self.assertEqual(bf16_projected.dtype, torch.float32)
        raw_model = TrainableVelocityModel()
        projected = ProjectedDetailModel(raw_model)(model_input, torch.full((2,), 500.0))[0]
        coarse, fraction = masked_block_average(projected, self.mask)
        self.assertLess(float(coarse[fraction > 0].abs().max().detach()), 3e-6)
        target = project_detail(torch.randn(self.truth.shape, generator=self.generator), self.mask)
        trainer = FineCascadeDynamicsTrainer.__new__(FineCascadeDynamicsTrainer)
        loss = trainer._flow_matching_loss(projected, target, {"valid_mask": self.mask})
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(raw_model.projection.weight.grad.abs().sum()), 0.0)

    def test_rk4_sampler_recovers_oracle_detail_without_changing_coarse_member(self) -> None:
        condition, coarse, clean_detail = teacher_coarse_condition(
            self.truth[:1], self.condition[:1], self.mask[:1]
        )
        initial_noise = project_detail(torch.randn(1, 6, 8, 10, generator=self.generator), self.mask[:1])
        velocity = initial_noise - clean_detail
        sampler = FineCascadeSampler(ConstantVelocityModel(velocity))
        zeros = torch.zeros(1, 6, 8, 10, dtype=torch.bfloat16)
        result = sampler.sample_conditioned(
            background=zeros,
            background_mask=torch.ones_like(zeros),
            obs_values=zeros[:, :2],
            obs_mask=zeros[:, :2],
            water_mask=torch.ones(1, 1, 8, 10, dtype=torch.bfloat16),
            size=(8, 10),
            num_timesteps=5,
            device=torch.device("cpu"),
            method="rk4",
            start_mode="noise",
            initial_noise=initial_noise,
            sample_target="state",
            model_conditioning=condition,
            state_channels=6,
            end_time=0.0,
        )
        valid = self.mask[:1].expand_as(result) > 0
        self.assertEqual(result.dtype, torch.float32)
        self.assertTrue(torch.allclose(result[valid], self.truth[:1][valid], atol=3e-5, rtol=0))
        result_coarse, _ = masked_block_average(result, self.mask[:1])
        self.assertTrue(torch.allclose(result_coarse, coarse, atol=3e-6, rtol=0))

        mismatched = self.mask[:1].clone()
        mismatched[:, :, -1, -1] = 0
        with self.assertRaisesRegex(ValueError, "valid_mask differs"):
            sampler.sample_conditioned(
                background=zeros,
                background_mask=torch.ones_like(zeros),
                obs_values=zeros[:, :2],
                obs_mask=zeros[:, :2],
                water_mask=torch.ones(1, 1, 8, 10),
                valid_mask=mismatched,
                size=(8, 10),
                num_timesteps=2,
                device=torch.device("cpu"),
                method="rk4",
                start_mode="noise",
                initial_noise=initial_noise,
                sample_target="state",
                model_conditioning=condition,
                state_channels=6,
                end_time=0.0,
            )
        mismatched_state = self.mask[:1].expand(-1, 6, -1, -1).clone()
        mismatched_state[:, :, -1, -1] = 0
        with self.assertRaisesRegex(ValueError, "state_mask differs"):
            sampler.sample_conditioned(
                background=zeros,
                background_mask=torch.ones_like(zeros),
                obs_values=zeros[:, :2],
                obs_mask=zeros[:, :2],
                water_mask=torch.ones(1, 1, 8, 10),
                state_mask=mismatched_state,
                size=(8, 10),
                num_timesteps=2,
                device=torch.device("cpu"),
                method="rk4",
                start_mode="noise",
                initial_noise=initial_noise,
                sample_target="state",
                model_conditioning=condition,
                state_channels=6,
                end_time=0.0,
            )

    def test_constructor_rejects_unsafe_pilot_configuration_before_runtime_setup(self) -> None:
        base = dict(
            in_channels=FINE_INPUT_CHANNELS,
            out_channels=6,
            clearml_enabled=False,
            metric_every_n_epochs=0,
            sample_every_n_epochs=0,
        )
        cases = (
            (TrainingConfig(**base, activation_checkpointing=True), "activation checkpointing"),
            (TrainingConfig(**base, gradient_accumulation_steps=2), "gradient_accumulation"),
            (TrainingConfig(**{**base, "metric_every_n_epochs": 1}), "oracle diagnostics"),
            (TrainingConfig(**{**base, "in_channels": 23}), "29-to-6"),
        )
        for config, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                FineCascadeDynamicsTrainer(config=config, model=ZeroModel())

    def test_oracle_payload_is_atomic_and_complete_before_plot_failure(self) -> None:
        condition, _, _ = teacher_coarse_condition(self.truth[:1], self.condition[:1], self.mask[:1])
        batch = {
            "truth": self.truth[:1],
            "background": torch.zeros_like(self.truth[:1]),
            "obs_values": torch.zeros(1, 2, 8, 10),
            "obs_mask": torch.zeros(1, 2, 8, 10),
            "water_mask": self.mask[:1],
            "valid_mask": self.mask[:1],
            "structured_conditioning": condition,
            "structured_physical_truth": self.truth[:1],
            "structured_physical_background": torch.zeros_like(self.truth[:1]),
        }
        harness = FailureHarness()
        harness.model = ZeroModel()
        harness.accelerator = SimpleNamespace(device=torch.device("cpu"))
        harness.config = SimpleNamespace(
            image_size=(8, 10),
            num_sample_timesteps=2,
            sample_method="rk4",
            sample_rtol=1e-3,
            sample_atol=1e-4,
            sample_use_ema=False,
        )
        harness.clearml = None
        harness._direct_validation_batch = batch
        harness._fine_validation_case_ids = ("case-zero",)
        harness._latest_fine_sampler = None
        with TemporaryDirectory() as directory:
            harness.output_dir = directory
            for name in ("mechanics_update_0064.pth", "ema_mechanics_update_0064.pth"):
                torch.save({"ok": True}, Path(directory) / name)
            payload_path = Path(directory) / "direct_diagnostics" / "oracle_coarse_update_0064_samples.pt"
            with (
                patch(
                    "assim_lib.direct_dynamics_training.DirectDynamicsTrainer._raw_support_metrics",
                    side_effect=RuntimeError("injected scorer failure"),
                ),
                self.assertRaisesRegex(RuntimeError, "injected scorer failure"),
            ):
                harness._direct_diagnostic(63, "update_0064")
            pending = torch.load(payload_path, map_location="cpu", weights_only=False)
            self.assertEqual(pending["diagnostic_status"], "raw_samples_saved_metrics_pending")
            self.assertEqual(pending["diagnostic_role"], "oracle_true_coarse_mechanics_only")
            self.assertIn("raw_initial_noise_normalized", pending)
            self.assertIn("checkpoint_sha256", pending)

            with (
                patch(
                    "assim_lib.structured_trajectory_evaluation.make_structured_trajectory_figure",
                    side_effect=RuntimeError("injected plot failure"),
                ),
                self.assertRaisesRegex(RuntimeError, "injected plot failure"),
            ):
                harness._direct_diagnostic(63, "update_0064")
            payload = torch.load(payload_path, map_location="cpu", weights_only=False)
            self.assertEqual(payload["diagnostic_status"], "complete_before_plot")
            self.assertEqual(payload["diagnostic_role"], "oracle_true_coarse_mechanics_only")
            self.assertFalse(payload["forecast_claim_permitted"])
            self.assertEqual(payload["validation_case_ids"], ["case-zero"])
            self.assertIn("raw_initial_noise_normalized", payload)
            self.assertIn("checkpoint_sha256", payload)
            self.assertFalse(list(Path(directory).rglob("*.incomplete")))

    def test_raw_and_ema_save_reload_keep_fine_sampler_semantics(self) -> None:
        config = TrainingConfig(
            image_size=(8, 8),
            in_channels=FINE_INPUT_CHANNELS,
            out_channels=6,
            block_out_channels=(8,),
            down_block_types=("DownBlock2D",),
            up_block_types=("UpBlock2D",),
            norm_num_groups=4,
            clearml_enabled=False,
        )
        raw_model = build_unet(config).eval()
        condition, _, _ = teacher_coarse_condition(
            self.truth[:1, :, :8, :8],
            self.condition[:1, :, :8, :8],
            self.mask[:1, :, :8, :8],
        )
        initial_noise = torch.randn(1, 6, 8, 8, generator=self.generator)
        kwargs = {
            "background": torch.zeros(1, 6, 8, 8),
            "background_mask": torch.ones(1, 6, 8, 8),
            "obs_values": torch.zeros(1, 2, 8, 8),
            "obs_mask": torch.zeros(1, 2, 8, 8),
            "water_mask": torch.ones(1, 1, 8, 8),
            "size": (8, 8),
            "num_timesteps": 2,
            "device": torch.device("cpu"),
            "method": "rk4",
            "start_mode": "noise",
            "initial_noise": initial_noise,
            "sample_target": "state",
            "model_conditioning": condition,
            "state_channels": 6,
            "end_time": 0.0,
        }
        expected = FineCascadeSampler(raw_model).sample_conditioned(**kwargs)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "fine_cascade_manifest.json").write_text(
                '{"schema_version":1,"sampler":"FineCascadeSampler",'
                '"condition_channels":21,"model_input_channels":29,'
                '"model_output_channels":6,"coarse_factor":2}',
                encoding="utf-8",
            )
            torch.save(raw_model.state_dict(), root / "raw.pth")
            ema = EMAModel(raw_model.parameters())
            torch.save(ema.state_dict(), root / "ema.pth")
            model_config = dict(config.__dict__)
            for checkpoint in ("raw.pth", "ema.pth"):
                loaded = load_fine_cascade_sampler(
                    directory, checkpoint, model_config, device=torch.device("cpu")
                )
                actual = loaded.sample_conditioned(**kwargs)
                self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=0))


if __name__ == "__main__":
    unittest.main()
