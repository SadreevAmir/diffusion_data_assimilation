import hashlib
import json
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
from assim_lib.direct_dynamics_cascade import masked_block_average, project_detail, smooth_right_inverse
from assim_lib.direct_dynamics_cascade_coarse import (
    COARSE_CONDITION_CHANNELS,
    COARSE_INPUT_CHANNELS,
    CoarseCascadeDynamicsTrainer,
    CoarseCascadeSampler,
    coarse_flow_pair,
    coarse_model_input_for_test,
    coarse_target,
    load_coarse_cascade_sampler,
    lossless_coarse_condition,
    ocean_fraction_weighted_mse,
    recover_full_condition_for_test,
    write_coarse_manifest,
)
from assim_lib.direct_dynamics_cascade_fine import FineCascadeSampler, generated_coarse_condition
from assim_lib.model_io import build_unet
from assim_lib.runtime import make_normalized_xy_grid


class ConstantVelocityModel(nn.Module):
    def __init__(self, velocity: torch.Tensor):
        super().__init__()
        self.register_buffer("velocity", velocity)

    def forward(self, model_input, timestep, return_dict=False):
        del timestep, return_dict
        return (self.velocity.expand(model_input.shape[0], -1, -1, -1),)


class TrainableModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(COARSE_INPUT_CHANNELS, 6, 1)

    def forward(self, model_input, timestep, return_dict=False):
        del timestep, return_dict
        return (self.projection(model_input),)


class DiagnosticHarness(CoarseCascadeDynamicsTrainer):
    def __init__(self):
        pass

    @contextmanager
    def _sampling_model(self):
        yield self.model

    def _normalized_to_physical(self, value):
        return value.float()

    def _sampling_weight_label(self):
        return "raw"


class CoarseCascadeTests(unittest.TestCase):
    CODE_COMMIT = "a" * 40

    def setUp(self):
        generator = torch.Generator().manual_seed(3901)
        self.condition = torch.randn(2, 15, 8, 10, generator=generator)
        self.mask = torch.ones(2, 1, 8, 10)
        self.mask[:, :, :2, :2] = 0
        self.mask[:, :, -1, -1] = 0
        self.condition[:, 2:3] = self.mask
        self.condition[:, :2] *= self.mask
        self.condition[:, 7:11] = torch.randint(0, 2, (2, 4, 8, 10), generator=generator).float() * self.mask
        self.condition[:, 3:7] *= self.condition[:, 7:11]
        for channel in range(11, 15):
            self.condition[:, channel] = self.condition[:, channel, :1, :1].expand(-1, 8, 10)
        self.truth = torch.randn(2, 6, 8, 10, generator=generator)
        self.noise = torch.randn(2, 6, 4, 5, generator=generator)

    def test_condition_encoding_is_exactly_invertible_and_preserves_coast(self):
        encoded, active, fraction = lossless_coarse_condition(self.condition, self.mask)
        self.assertEqual(tuple(encoded.shape), (2, COARSE_CONDITION_CHANNELS, 4, 5))
        self.assertTrue(torch.equal(recover_full_condition_for_test(encoded), self.condition))
        self.assertTrue(torch.equal(active, (fraction > 0).float()))
        self.assertGreater(float(fraction[0, 0, -1, -1]), 0.0)
        self.assertLess(float(fraction[0, 0, -1, -1]), 1.0)
        # PyTorch packs four subpixel phases per source channel.
        expected_mask_phases = torch.nn.functional.pixel_unshuffle(self.mask, 2)
        self.assertTrue(torch.equal(encoded[:, 8:12], expected_mask_phases))

    def test_condition_rejects_mask_mismatch_nonconstant_calendar_and_nonfinite(self):
        wrong = self.mask.clone()
        wrong[:, :, 3, 3] = 0
        with self.assertRaisesRegex(ValueError, "embedded dynamics mask"):
            lossless_coarse_condition(self.condition, wrong)
        nonconstant = self.condition.clone()
        nonconstant[:, 11, -1, -1] += 1
        with self.assertRaisesRegex(ValueError, "spatial constants"):
            lossless_coarse_condition(nonconstant, self.mask)
        nonfinite = self.condition.clone()
        nonfinite[:, 5, 2, 2] = float("nan")
        with self.assertRaisesRegex(FloatingPointError, "NaN/Inf"):
            lossless_coarse_condition(nonfinite, self.mask)
        bad_forcing_mask = self.condition.clone()
        bad_forcing_mask[:, 7, 2, 2] = 0.5
        with self.assertRaisesRegex(ValueError, "forcing masks must be binary"):
            lossless_coarse_condition(bad_forcing_mask, self.mask)
        unsupported_forcing = self.condition.clone()
        unsupported_forcing[:, 3, 2, 2] = 1.0
        unsupported_forcing[:, 7, 2, 2] = 0.0
        with self.assertRaisesRegex(ValueError, "wherever their mask is absent"):
            lossless_coarse_condition(unsupported_forcing, self.mask)

    def test_target_is_masked_average_and_flow_is_zero_off_support(self):
        state, velocity, clean, fraction = coarse_flow_pair(
            self.truth, self.noise, self.mask, torch.tensor([0.25, 0.75])
        )
        direct, active, expected_fraction = coarse_target(self.truth, self.mask)
        self.assertTrue(torch.equal(clean, direct))
        self.assertTrue(torch.equal(fraction, expected_fraction))
        invalid = active.expand_as(clean) == 0
        self.assertTrue(torch.equal(state[invalid], torch.zeros_like(state[invalid])))
        self.assertTrue(torch.equal(velocity[invalid], torch.zeros_like(velocity[invalid])))
        inactive_nan = self.noise.clone()
        inactive_nan[invalid] = float("nan")
        state2, velocity2, _, _ = coarse_flow_pair(
            self.truth, inactive_nan, self.mask, torch.tensor([0.25, 0.75])
        )
        self.assertTrue(torch.isfinite(state2).all())
        self.assertTrue(torch.isfinite(velocity2).all())
        active_nan = self.noise.clone()
        active_nan[active.expand_as(active_nan) > 0] = float("nan")
        with self.assertRaisesRegex(FloatingPointError, "coarse noise"):
            coarse_flow_pair(self.truth, active_nan, self.mask, torch.tensor([0.25, 0.75]))

    def test_ocean_fraction_loss_matches_case_equal_manual_reference_and_backpropagates(self):
        base = torch.tensor([[[[1.0, 2.0]]], [[[3.0, 5.0]]]], requires_grad=True)
        prediction = base.expand(-1, 6, -1, -1)
        target = torch.zeros_like(prediction)
        fraction = torch.tensor([[[[1.0, 0.5]]], [[[0.25, 1.0]]]])
        actual = ocean_fraction_weighted_mse(prediction, target, fraction)
        case0 = (1.0**2 * 1.0 + 2.0**2 * 0.5) / 1.5
        case1 = (3.0**2 * 0.25 + 5.0**2 * 1.0) / 1.25
        expected = torch.tensor((case0 + case1) / 2)
        self.assertTrue(torch.allclose(actual, expected))
        actual.backward()
        self.assertTrue(torch.isfinite(base.grad).all())
        inactive = fraction.expand_as(prediction) == 0
        with_nan = prediction.detach().clone()
        with_nan[inactive] = float("nan")
        self.assertTrue(torch.isfinite(ocean_fraction_weighted_mse(with_nan, target, fraction)))
        overflow = torch.full_like(prediction, torch.finfo(torch.float32).max)
        with self.assertRaisesRegex(FloatingPointError, "overflowed"):
            ocean_fraction_weighted_mse(overflow, target, torch.ones_like(fraction))
        reduction_overflow = torch.full((1, 6, 8, 8), 1e18)
        with self.assertRaisesRegex(FloatingPointError, "reduction overflowed"):
            ocean_fraction_weighted_mse(
                reduction_overflow,
                torch.zeros_like(reduction_overflow),
                torch.ones(1, 1, 8, 8),
            )

    def test_fraction_weighted_rmse_handles_partial_ocean_and_rejects_empty_case(self):
        quarter = torch.tensor([[[[0.25]]]])
        error_one = torch.ones(1, 1, 1, 1)
        rmse = CoarseCascadeDynamicsTrainer._fraction_weighted_rmse(
            error_one, torch.zeros_like(error_one), quarter
        )
        self.assertAlmostEqual(rmse, 1.0)
        with self.assertRaisesRegex(ValueError, "must contain valid ocean"):
            CoarseCascadeDynamicsTrainer._fraction_weighted_rmse(
                error_one, torch.zeros_like(error_one), torch.zeros_like(quarter)
            )

    def test_model_input_and_trainer_contract(self):
        state = torch.randn(2, 6, 4, 5)
        with patch(
            "assim_lib.direct_dynamics_cascade_coarse.make_normalized_xy_grid",
            wraps=make_normalized_xy_grid,
        ) as grid_builder:
            model_input = coarse_model_input_for_test(state, self.condition, self.mask)
        self.assertEqual(grid_builder.call_args.kwargs["device"], state.device)
        self.assertEqual(tuple(model_input.shape), (2, COARSE_INPUT_CHANNELS, 4, 5))
        trainable = TrainableModel()
        prediction = trainable(model_input, torch.tensor([100.0, 900.0]))[0]
        target = torch.randn_like(prediction)
        _, _, fraction = coarse_target(self.truth, self.mask)
        loss = ocean_fraction_weighted_mse(prediction, target, fraction)
        loss.backward()
        self.assertGreater(float(trainable.projection.weight.grad.abs().sum()), 0.0)
        base = dict(
            image_size=(160, 128),
            in_channels=COARSE_INPUT_CHANNELS,
            out_channels=6,
            training_objective="flow",
            timestep_sampler="stratified_uniform",
            activation_checkpointing=False,
            gradient_accumulation_steps=1,
            metric_every_n_epochs=0,
            sample_every_n_epochs=0,
            clearml_enabled=False,
        )
        CoarseCascadeDynamicsTrainer.__new__(CoarseCascadeDynamicsTrainer)
        with self.assertRaisesRegex(ValueError, "56-to-6"):
            CoarseCascadeDynamicsTrainer(
                config=TrainingConfig(**{**base, "in_channels": 55}), model=nn.Identity()
            )
        unsafe = (
            ({"activation_checkpointing": True}, "activation checkpointing"),
            ({"gradient_accumulation_steps": 2}, "gradient_accumulation"),
            ({"timestep_sampler": "uniform"}, "stratified_uniform"),
            ({"metric_every_n_epochs": 1}, "generic diagnostics"),
            ({"sample_every_n_epochs": 1}, "generic diagnostics"),
        )
        for overrides, message in unsafe:
            with self.subTest(overrides=overrides), self.assertRaisesRegex(ValueError, message):
                CoarseCascadeDynamicsTrainer(
                    config=TrainingConfig(**{**base, **overrides}), model=nn.Identity()
                )

    def test_rk4_recovers_oracle_coarse_state(self):
        clean, active, _ = coarse_target(self.truth[:1], self.mask[:1])
        noise = self.noise[:1]
        velocity = noise - clean
        sampler = CoarseCascadeSampler(ConstantVelocityModel(velocity))
        result = sampler.sample_conditioned(
            structured_conditioning=self.condition[:1],
            valid_mask=self.mask[:1],
            initial_noise=noise,
            num_timesteps=5,
            device=torch.device("cpu"),
        )
        support = active.expand_as(clean) > 0
        self.assertTrue(torch.allclose(result[support], clean[support], atol=3e-5, rtol=0))
        self.assertTrue(torch.equal(result[~support], torch.zeros_like(result[~support])))
        bf16_result = sampler.sample_conditioned(
            structured_conditioning=self.condition[:1].bfloat16(),
            valid_mask=self.mask[:1].bfloat16(),
            initial_noise=noise.bfloat16(),
            num_timesteps=5,
            device=torch.device("cpu"),
        )
        self.assertEqual(bf16_result.dtype, torch.float32)
        self.assertTrue(torch.allclose(bf16_result[support], clean[support], atol=5e-3, rtol=0))

    def test_case_member_seed_is_reorder_and_chunk_invariant(self):
        ids = ("2022-01-03_slice00", "2022-07-03_slice12", "2022-11-03_slice23")
        reference = {
            (case_id, member): CoarseCascadeDynamicsTrainer._member_seed(case_id, member)
            for case_id in ids
            for member in range(3)
        }
        reordered = {
            (case_id, member): CoarseCascadeDynamicsTrainer._member_seed(case_id, member)
            for case_id in reversed(ids)
            for member in range(3)
        }
        self.assertEqual(reference, reordered)
        self.assertEqual(len(set(reference.values())), len(reference))

    def test_memberwise_coarse_to_fine_composition_cannot_silently_swap_coarse(self):
        causal = self.condition[:1]
        valid = self.mask[:1]
        generator = torch.Generator().manual_seed(813)
        coarse_members = torch.randn(2, 6, 4, 5, generator=generator)
        _, coarse_fraction = masked_block_average(valid.expand(2, -1, -1, -1), valid.expand(2, -1, -1, -1))
        coarse_members = torch.where(coarse_fraction.expand_as(coarse_members) > 0, coarse_members, 0.0)
        detail_members = project_detail(
            torch.randn(2, 6, 8, 10, generator=generator), valid.expand(2, -1, -1, -1)
        )
        fine_noise = project_detail(
            torch.randn(2, 6, 8, 10, generator=generator), valid.expand(2, -1, -1, -1)
        )
        outputs = []
        for member in range(2):
            coarse = coarse_members[member : member + 1]
            detail = detail_members[member : member + 1]
            condition = generated_coarse_condition(causal, coarse, valid)
            sampler = FineCascadeSampler(ConstantVelocityModel(fine_noise[member : member + 1] - detail))
            output = sampler.sample_conditioned(
                background=torch.zeros(1, 6, 8, 10),
                obs_values=torch.zeros(1, 2, 8, 10),
                obs_mask=torch.zeros(1, 2, 8, 10),
                water_mask=valid,
                size=(8, 10),
                num_timesteps=5,
                device=torch.device("cpu"),
                method="rk4",
                start_mode="noise",
                initial_noise=fine_noise[member : member + 1],
                sample_target="state",
                model_conditioning=condition,
                state_channels=6,
                end_time=0.0,
            )
            recovered, _ = masked_block_average(output, valid)
            self.assertTrue(torch.allclose(recovered, coarse, atol=3e-6, rtol=0))
            expected = smooth_right_inverse(coarse, valid) + detail
            support = valid.expand_as(expected) > 0
            self.assertTrue(torch.allclose(output[support], expected[support], atol=3e-5, rtol=0))
            outputs.append(output)
        wrong_coarse, _ = masked_block_average(outputs[0], valid)
        self.assertFalse(torch.allclose(wrong_coarse, coarse_members[1:2], atol=3e-6, rtol=0))

    def test_raw_and_ema_reload_use_tagged_sampler(self):
        config = TrainingConfig(
            image_size=(160, 128),
            in_channels=COARSE_INPUT_CHANNELS,
            out_channels=6,
            block_out_channels=(8,),
            down_block_types=("DownBlock2D",),
            up_block_types=("UpBlock2D",),
            norm_num_groups=4,
            add_attention=False,
            clearml_enabled=False,
        )
        model = build_unet(config).eval()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_coarse_manifest(root, config, {"git_commit": self.CODE_COMMIT})
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.fill_(0.25)
            torch.save(model.state_dict(), root / "raw.pth")
            ema = EMAModel(model.parameters())
            for shadow in ema.shadow_params:
                shadow.zero_()
            torch.save(ema.state_dict(), root / "ema.pth")
            loaded_parameters = {}
            for name in ("raw.pth", "ema.pth"):
                checkpoint = root / name
                expected_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                loaded = load_coarse_cascade_sampler(
                    directory,
                    name,
                    dict(config.__dict__),
                    expected_sha,
                    self.CODE_COMMIT,
                    device=torch.device("cpu"),
                )
                self.assertIsInstance(loaded, CoarseCascadeSampler)
                loaded_parameters[name] = next(loaded.sampler.model.parameters()).detach().clone()
            self.assertTrue(torch.all(loaded_parameters["raw.pth"] == 0.25))
            self.assertTrue(torch.all(loaded_parameters["ema.pth"] == 0.0))
            changed = dict(config.__dict__)
            changed["norm_num_groups"] = 2
            raw_sha = hashlib.sha256((root / "raw.pth").read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "architecture differs"):
                load_coarse_cascade_sampler(
                    directory,
                    "raw.pth",
                    changed,
                    raw_sha,
                    self.CODE_COMMIT,
                    device=torch.device("cpu"),
                )
            with self.assertRaisesRegex(ValueError, "SHA256 differs"):
                load_coarse_cascade_sampler(
                    directory,
                    "raw.pth",
                    dict(config.__dict__),
                    "0" * 64,
                    self.CODE_COMMIT,
                    device=torch.device("cpu"),
                )

    def test_diagnostic_payload_survives_scorer_and_plot_failures(self):
        batch = {
            "truth": self.truth[:1],
            "background": torch.zeros_like(self.truth[:1]),
            "valid_mask": self.mask[:1],
            "structured_conditioning": self.condition[:1],
        }
        harness = DiagnosticHarness()
        harness.model = ConstantVelocityModel(torch.zeros(1, 6, 4, 5))
        harness.accelerator = SimpleNamespace(device=torch.device("cpu"))
        harness.config = SimpleNamespace(
            image_size=(4, 5),
            num_sample_timesteps=2,
            sample_method="rk4",
            sample_rtol=1e-5,
            sample_atol=1e-6,
        )
        harness.clearml = None
        harness._coarse_diagnostic_batch = batch
        harness._coarse_diagnostic_case_ids = ("2022-07-03_slice12",)
        with TemporaryDirectory() as directory:
            harness.output_dir = directory
            for name in ("coarse_update_0064.pth", "ema_coarse_update_0064.pth"):
                torch.save({"durable": True}, Path(directory) / name)
            payload_path = Path(directory) / "coarse_diagnostics" / "update_0064_samples.pt"
            with (
                patch.object(harness, "_raw_support_metrics", side_effect=RuntimeError("score fail")),
                self.assertRaisesRegex(RuntimeError, "score fail"),
            ):
                harness._coarse_diagnostic(63, "update_0064", "coarse_update_0064.pth")
            pending = torch.load(payload_path, map_location="cpu", weights_only=False)
            self.assertEqual(pending["diagnostic_status"], "raw_samples_saved_metrics_pending")
            self.assertFalse(pending["future_truth_used_for_condition"])
            self.assertIn("raw_coarse_noise_normalized", pending)
            with (
                patch(
                    "assim_lib.structured_trajectory_evaluation.make_structured_trajectory_figure",
                    side_effect=RuntimeError("plot fail"),
                ),
                self.assertRaisesRegex(RuntimeError, "plot fail"),
            ):
                harness._coarse_diagnostic(63, "update_0064", "coarse_update_0064.pth")
            complete = torch.load(payload_path, map_location="cpu", weights_only=False)
            self.assertEqual(complete["diagnostic_status"], "complete_before_plot")
            self.assertTrue(
                all(torch.isfinite(torch.tensor(value)) for value in complete["metrics"].values())
            )

    def test_terminal_gate_is_invariant_to_sic_sit_units(self):
        outputs = ("d3_sic", "d3_sit", "d6_sic", "d6_sit", "d9_sic", "d9_sit")

        def record(step: int, raw: float, *, sit_scale: float) -> dict[str, float]:
            metrics = {
                "step": float(step),
                "joint_raw_support_violation_fraction": 0.1,
            }
            for name in outputs:
                scale = sit_scale if name.endswith("sit") else 1.0
                metrics[f"{name}_raw_mean_rmse"] = raw * scale
                metrics[f"{name}_persistence_rmse"] = 1.5 * scale
                metrics[f"{name}_member_roughness"] = 2.0 * scale
                metrics[f"{name}_truth_roughness"] = 1.0 * scale
            return metrics

        summaries = []
        for sit_scale in (1.0, 1000.0):
            harness = DiagnosticHarness()
            harness.accelerator = SimpleNamespace(is_main_process=True)
            harness.clearml = None
            harness.val_history = [{}]
            harness._coarse_diagnostic_history = [
                record(64, 2.0, sit_scale=sit_scale),
                record(256, 1.2, sit_scale=sit_scale),
                record(512, 1.0, sit_scale=sit_scale),
            ]
            with TemporaryDirectory() as directory:
                harness.output_dir = directory
                harness._after_training_epoch(0, 512)
                gate = json.loads(
                    (Path(directory) / "coarse_mechanics_gate.json").read_text(encoding="utf-8")
                )
            summaries.append(gate["summary"])
        self.assertEqual(summaries[0], summaries[1])
        self.assertAlmostEqual(summaries[0]["mean_dimensionless_rmse_improvement_from_update64"], 0.5)
        self.assertAlmostEqual(summaries[0]["mean_dimensionless_rmse_skill_over_persistence"], 1.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
