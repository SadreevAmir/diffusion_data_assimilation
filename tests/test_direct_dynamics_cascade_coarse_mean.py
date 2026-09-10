import json
import unittest
from contextlib import contextmanager, nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch

from assim_lib.config import TrainingConfig
from assim_lib.direct_dynamics_cascade_coarse import (
    recover_full_condition_for_test,
)
from assim_lib.direct_dynamics_cascade_coarse_mean import (
    COARSE_MEAN_INPUT_CHANNELS,
    CoarseConditionalMeanPredictor,
    CoarseConditionalMeanTrainer,
    coarse_mean_model_input,
    coarse_mean_target,
)
from assim_lib.direct_dynamics_cascade_coarse_training import _gpu_admission_smoke


def _causal_batch(batch_size: int = 2, height: int = 8, width: int = 8):
    valid = torch.ones(batch_size, 1, height, width)
    condition = torch.zeros(batch_size, 15, height, width)
    y = torch.linspace(0.0, 1.0, height).reshape(1, height, 1)
    x = torch.linspace(0.0, 1.0, width).reshape(1, 1, width)
    condition[:, 0] = 0.2 + 0.4 * y + 0.1 * x
    condition[:, 1] = 0.1 + 0.2 * y + 0.3 * x
    condition[:, 2:3] = valid
    condition[:, 3:7] = torch.arange(1, 5).reshape(1, 4, 1, 1)
    condition[:, 7:11] = valid
    condition[:, 11:15] = torch.tensor([0.25, -0.5, 0.75, -1.0]).reshape(
        1, 4, 1, 1
    )
    truth = condition[:, :2].repeat(1, 3, 1, 1)
    truth = truth + torch.arange(1, 7).reshape(1, 6, 1, 1) * 0.01
    return truth, condition, valid


class _ConstantDeltaModel(torch.nn.Module):
    def __init__(self, delta: torch.Tensor):
        super().__init__()
        self.register_buffer("delta", delta)
        self.calls = []

    def forward(self, sample, timestep, return_dict=False):
        self.calls.append((sample.detach().clone(), timestep.detach().clone(), return_dict))
        return (self.delta.expand(sample.shape[0], -1, -1, -1),)


class _SmokeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.output = torch.nn.Conv2d(COARSE_MEAN_INPUT_CHANNELS, 6, 1)

    def forward(self, sample, timestep, return_dict=False):
        del timestep, return_dict
        return (self.output(sample),)


class _MeanDiagnosticHarness(CoarseConditionalMeanTrainer):
    def __init__(self):
        pass

    @contextmanager
    def _sampling_model(self):
        yield self.model

    def _normalized_to_physical(self, value):
        return value.float()


class CoarseConditionalMeanTests(unittest.TestCase):
    def test_model_input_is_lossless_causal_condition_without_state(self):
        truth, condition, valid = _causal_batch()
        model_input, active, fraction = coarse_mean_model_input(condition, valid)
        self.assertEqual(
            tuple(model_input.shape),
            (truth.shape[0], COARSE_MEAN_INPUT_CHANNELS, 4, 4),
        )
        recovered = recover_full_condition_for_test(model_input[:, 2:])
        self.assertTrue(torch.equal(recovered, condition))
        self.assertTrue(torch.equal(active, torch.ones_like(active)))
        self.assertTrue(torch.equal(fraction, torch.ones_like(fraction)))

        changed_future = truth + 1000.0
        unchanged_input, _, _ = coarse_mean_model_input(condition, valid)
        self.assertFalse(torch.equal(changed_future, truth))
        self.assertTrue(torch.equal(unchanged_input, model_input))

    def test_target_is_exact_persistence_increment(self):
        truth, condition, valid = _causal_batch()
        delta, persistence, clean, fraction = coarse_mean_target(
            truth, condition, valid
        )
        self.assertTrue(torch.equal(persistence + delta, clean))
        expected = torch.arange(1, 7).reshape(1, 6, 1, 1) * 0.01
        self.assertTrue(torch.allclose(delta, expected.expand_as(delta), atol=1e-7))
        self.assertTrue(torch.equal(fraction, torch.ones_like(fraction)))

    def test_predictor_is_one_forward_at_zero_and_reconstructs_map(self):
        _, condition, valid = _causal_batch()
        delta = torch.arange(1, 7).reshape(1, 6, 1, 1).expand(1, 6, 4, 4) * 0.01
        model = _ConstantDeltaModel(delta)
        prediction = CoarseConditionalMeanPredictor(model).predict_conditioned(
            structured_conditioning=condition,
            valid_mask=valid,
        )
        _, persistence, _, _ = coarse_mean_target(
            condition[:, :2].repeat(1, 3, 1, 1), condition, valid
        )
        self.assertTrue(torch.allclose(prediction, persistence + delta))
        self.assertEqual(len(model.calls), 1)
        sample, timestep, return_dict = model.calls[0]
        self.assertEqual(sample.shape[1], COARSE_MEAN_INPUT_CHANNELS)
        self.assertTrue(torch.equal(timestep, torch.zeros_like(timestep)))
        self.assertFalse(return_dict)

    def test_predictor_rejects_overflow_in_persistence_reconstruction(self):
        _, condition, valid = _causal_batch(batch_size=1)
        condition[:, :2] = torch.finfo(torch.float32).max / 4
        delta = torch.full(
            (1, 6, 4, 4), torch.finfo(torch.float32).max, dtype=torch.float32
        )
        with self.assertRaisesRegex(FloatingPointError, "reconstruction"):
            CoarseConditionalMeanPredictor(
                _ConstantDeltaModel(delta)
            ).predict_conditioned(
                structured_conditioning=condition,
                valid_mask=valid,
            )

    def test_predictor_rejects_wrong_delta_shape(self):
        _, condition, valid = _causal_batch(batch_size=1)
        wrong = torch.zeros(1, 5, 4, 4)
        with self.assertRaisesRegex(ValueError, "shape differs"):
            CoarseConditionalMeanPredictor(
                _ConstantDeltaModel(wrong)
            ).predict_conditioned(
                structured_conditioning=condition,
                valid_mask=valid,
            )

    def test_residual_and_absolute_map_mse_have_identical_gradients(self):
        _, condition, valid = _causal_batch(batch_size=1)
        truth = condition[:, :2].repeat(1, 3, 1, 1) + 0.0625
        delta_target, persistence, clean, fraction = coarse_mean_target(
            truth, condition, valid
        )
        delta_prediction = torch.randn_like(delta_target, requires_grad=True)
        absolute_prediction = persistence + delta_prediction
        residual_loss = (((delta_prediction - delta_target).square()) * fraction).mean()
        absolute_loss = (((absolute_prediction - clean).square()) * fraction).mean()
        residual_gradient = torch.autograd.grad(
            residual_loss, delta_prediction, retain_graph=True
        )[0]
        absolute_gradient = torch.autograd.grad(absolute_loss, delta_prediction)[0]
        self.assertTrue(torch.allclose(residual_loss, absolute_loss, rtol=0, atol=1e-6))
        self.assertTrue(
            torch.allclose(residual_gradient, absolute_gradient, rtol=0, atol=1e-7)
        )

    def test_training_pair_contains_no_random_state_and_rejects_flow_time(self):
        truth, condition, valid = _causal_batch()
        trainer = CoarseConditionalMeanTrainer.__new__(CoarseConditionalMeanTrainer)
        batch = {
            "structured_conditioning": condition,
            "valid_mask": valid,
            "background": condition[:, :2].repeat(1, 3, 1, 1),
        }
        state, target = trainer._make_training_pair(
            truth, batch, torch.zeros(truth.shape[0])
        )
        expected, _, _, _ = coarse_mean_target(truth, condition, valid)
        self.assertTrue(torch.equal(state, torch.zeros_like(state)))
        self.assertTrue(torch.equal(target, expected))
        with self.assertRaisesRegex(ValueError, "zero timestep"):
            trainer._make_training_pair(
                truth, batch, torch.full((truth.shape[0],), 0.5)
            )

    def test_gpu_smoke_dispatch_executes_deterministic_mean_branch(self):
        truth, condition, valid = _causal_batch()
        raw = {
            "truth": truth,
            "background": condition[:, :2].repeat(1, 3, 1, 1),
            "valid_mask": valid,
            "structured_conditioning": condition,
        }
        config = SimpleNamespace(
            image_size=(4, 4),
            in_channels=COARSE_MEAN_INPUT_CHANNELS,
            activation_checkpointing=False,
            gradient_accumulation_steps=1,
        )
        cpu = torch.device("cpu")
        with (
            patch(
                "assim_lib.direct_dynamics_cascade_coarse_training.torch.device",
                return_value=cpu,
            ),
            patch(
                "assim_lib.direct_dynamics_cascade_coarse_training.torch.autocast",
                return_value=nullcontext(),
            ),
            patch(
                "assim_lib.direct_dynamics_cascade_coarse_training.torch.cuda.reset_peak_memory_stats"
            ),
            patch(
                "assim_lib.direct_dynamics_cascade_coarse_training.torch.cuda.max_memory_allocated",
                return_value=0,
            ),
            patch(
                "assim_lib.direct_dynamics_cascade_coarse_training.torch.cuda.empty_cache"
            ),
        ):
            result = _gpu_admission_smoke(
                _SmokeModel(), raw, config, deterministic_mean=True
            )
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["state_coordinate"], "deterministic_conditional_mean")
        self.assertEqual(result["model_input_shape"], [2, 50, 4, 4])

    def test_committed_config_is_strict_deterministic_mean_contract(self):
        path = Path("config/methods/direct_dynamics_cascade_coarse_mean_2f.json")
        raw = json.loads(path.read_text(encoding="utf-8"))
        config = TrainingConfig.from_dict(raw)
        self.assertEqual((config.in_channels, config.out_channels), (50, 6))
        self.assertEqual(config.training_objective, "deterministic_mean")
        self.assertEqual(config.timestep_sampler, "constant_zero")
        self.assertEqual(config.num_sample_timesteps, 1)
        self.assertFalse(config.activation_checkpointing)
        self.assertFalse(config.sample_use_ema)
        self.assertEqual(config.sample_every_n_epochs, 0)
        self.assertEqual(config.metric_every_n_epochs, 0)

        raw["timestep_sampler"] = "stratified_uniform"
        with self.assertRaisesRegex(ValueError, "constant_zero"):
            TrainingConfig.from_dict(raw)

    def test_terminal_gate_holds_for_frozen_review_without_flow_threshold(self):
        outputs = ("d3_sic", "d3_sit", "d6_sic", "d6_sit", "d9_sic", "d9_sit")

        def record(step: int, rmse: float) -> dict[str, float]:
            metrics = {
                "step": float(step),
                "joint_raw_support_violation_fraction": 0.02,
            }
            for output in outputs:
                metrics[f"{output}_raw_mean_rmse"] = rmse
                metrics[f"{output}_persistence_rmse"] = 1.0
                metrics[f"{output}_member_roughness"] = 1.1
                metrics[f"{output}_truth_roughness"] = 1.0
            return metrics

        trainer = CoarseConditionalMeanTrainer.__new__(CoarseConditionalMeanTrainer)
        trainer.accelerator = SimpleNamespace(is_main_process=True)
        trainer.clearml = None
        trainer._planned_updates = 2048
        trainer._coarse_diagnostic_steps = {511, 1023, 1535, 2047}
        # Only 1% improvement from the first diagnostic: the inherited flow
        # gate would reject this solely for being below its arbitrary 20% rule.
        trainer._coarse_diagnostic_history = [
            record(512, 0.90),
            record(1024, 0.895),
            record(1536, 0.892),
            record(2048, 0.891),
        ]
        trainer.val_history = [{}]
        with TemporaryDirectory() as directory:
            trainer.output_dir = directory
            trainer._after_training_epoch(3, 2048)
            gate = json.loads(
                (Path(directory) / "coarse_mean_gate.json").read_text(
                    encoding="utf-8"
                )
            )
            runner_gate = json.loads(
                (Path(directory) / "coarse_mechanics_gate.json").read_text(
                    encoding="utf-8"
                )
            )
        self.assertEqual(gate, runner_gate)
        self.assertEqual(gate["decision"], "hold_for_frozen_mean_review")
        self.assertFalse(gate["automatic_acceptance_or_rejection"])
        self.assertFalse(gate["innovations_stage_permitted"])
        self.assertAlmostEqual(
            gate["summary"]["mean_dimensionless_rmse_skill_over_persistence"],
            0.109,
        )

    def test_diagnostic_payload_survives_scorer_and_plot_failures(self):
        truth, condition, valid = _causal_batch(batch_size=1)
        harness = _MeanDiagnosticHarness()
        harness.model = _ConstantDeltaModel(torch.zeros(1, 6, 4, 4))
        harness.accelerator = SimpleNamespace(device=torch.device("cpu"))
        harness.clearml = None
        harness._coarse_diagnostic_batch = {
            "truth": truth,
            "valid_mask": valid,
            "structured_conditioning": condition,
        }
        harness._coarse_diagnostic_case_ids = ("2022-07-03_slice12",)
        with TemporaryDirectory() as directory:
            harness.output_dir = directory
            for name in ("coarse_update_0512.pth", "ema_coarse_update_0512.pth"):
                torch.save({"durable": True}, Path(directory) / name)
            payload_path = Path(directory) / "coarse_mean_diagnostics" / "update_0512_mean.pt"
            with (
                patch.object(
                    harness,
                    "_raw_support_metrics",
                    side_effect=RuntimeError("score fail"),
                ),
                self.assertRaisesRegex(RuntimeError, "score fail"),
            ):
                harness._coarse_diagnostic(
                    511, "update_0512", "coarse_update_0512.pth"
                )
            pending = torch.load(payload_path, map_location="cpu", weights_only=False)
            self.assertEqual(
                pending["diagnostic_status"], "raw_samples_saved_metrics_pending"
            )
            self.assertFalse(pending["future_truth_used_for_condition"])
            self.assertIn("coarse_mean_normalized", pending)

            with (
                patch(
                    "assim_lib.structured_trajectory_evaluation.make_structured_trajectory_figure",
                    side_effect=RuntimeError("plot fail"),
                ),
                self.assertRaisesRegex(RuntimeError, "plot fail"),
            ):
                harness._coarse_diagnostic(
                    511, "update_0512", "coarse_update_0512.pth"
                )
            scored = torch.load(payload_path, map_location="cpu", weights_only=False)
            self.assertEqual(scored["diagnostic_status"], "complete_before_plot")
            self.assertIn("d9_sit_raw_mean_rmse", scored["metrics"])
            self.assertIn("joint_raw_support_violation_fraction", scored["metrics"])


if __name__ == "__main__":
    unittest.main()
