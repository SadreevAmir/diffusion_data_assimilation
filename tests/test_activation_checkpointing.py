from __future__ import annotations

import copy
import unittest
from unittest import mock

import torch
from diffusers.training_utils import EMAModel

from assim_lib.config import TrainingConfig
from assim_lib.model_io import build_unet
from assim_lib.trainer import configure_activation_checkpointing


def _tiny_config(*, checkpointing: bool) -> TrainingConfig:
    return TrainingConfig(
        image_size=(8, 8),
        in_channels=4,
        out_channels=2,
        block_out_channels=(32, 32),
        down_block_types=("DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D"),
        layers_per_block=1,
        norm_num_groups=8,
        dropout=0.0,
        activation_checkpointing=checkpointing,
    )


class ActivationCheckpointingTests(unittest.TestCase):
    def test_disabled_path_is_an_exact_noop(self) -> None:
        model = build_unet(_tiny_config(checkpointing=False))
        self.assertEqual(configure_activation_checkpointing(model, False), ())
        self.assertFalse(model.is_gradient_checkpointing)

    def test_native_non_reentrant_path_preserves_output_gradient_and_rng(self) -> None:
        torch.manual_seed(7123)
        reference = build_unet(_tiny_config(checkpointing=False))
        candidate = copy.deepcopy(reference)
        modules = configure_activation_checkpointing(candidate, True)
        self.assertTrue(modules)
        self.assertTrue(candidate.is_gradient_checkpointing)

        value = torch.randn(2, 4, 8, 8)
        timestep = torch.tensor([125.0, 875.0])

        def run(model):
            model.train()
            model.zero_grad(set_to_none=True)
            torch.manual_seed(9127)
            before = torch.random.get_rng_state().clone()
            output = model(value, timestep, return_dict=False)[0]
            output.square().mean().backward()
            after = torch.random.get_rng_state().clone()
            gradients = {
                name: None if parameter.grad is None else parameter.grad.detach().clone()
                for name, parameter in model.named_parameters()
            }
            return output.detach(), gradients, before, after

        ref_output, ref_gradients, ref_before, ref_after = run(reference)
        candidate_output, candidate_gradients, candidate_before, candidate_after = run(candidate)
        self.assertTrue(torch.equal(ref_output, candidate_output))
        self.assertTrue(torch.equal(ref_before, candidate_before))
        self.assertTrue(torch.equal(ref_after, candidate_after))
        self.assertEqual(ref_gradients.keys(), candidate_gradients.keys())
        for name in ref_gradients:
            reference_gradient = ref_gradients[name]
            candidate_gradient = candidate_gradients[name]
            self.assertEqual(reference_gradient is None, candidate_gradient is None, name)
            if reference_gradient is not None:
                self.assertTrue(torch.isfinite(reference_gradient).all(), name)
                self.assertTrue(torch.isfinite(candidate_gradient).all(), name)
                torch.testing.assert_close(
                    candidate_gradient,
                    reference_gradient,
                    atol=1e-6,
                    rtol=1e-5,
                    msg=name,
                )

    def test_version_and_support_mismatch_fail_closed(self) -> None:
        model = build_unet(_tiny_config(checkpointing=True))
        with mock.patch("assim_lib.trainer.diffusers_version", "0.35.2"):
            with self.assertRaisesRegex(RuntimeError, "diffusers==0.36.0"):
                configure_activation_checkpointing(model, True)

        unsupported = torch.nn.Linear(2, 2)
        with self.assertRaisesRegex(RuntimeError, "does not support"):
            configure_activation_checkpointing(unsupported, True)

    def test_one_adam_scheduler_ema_update_is_equivalent(self) -> None:
        torch.manual_seed(5151)
        initial = build_unet(_tiny_config(checkpointing=False)).state_dict()
        value = torch.randn(2, 4, 8, 8)
        timestep = torch.tensor([250.0, 750.0])

        def run(checkpointing: bool):
            model = build_unet(_tiny_config(checkpointing=checkpointing))
            model.load_state_dict(initial)
            configure_activation_checkpointing(model, checkpointing)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lambda step: 1.0 / (step + 1)
            )
            ema = EMAModel(model.parameters(), decay=0.999)
            optimizer.zero_grad(set_to_none=True)
            output = model(value, timestep, return_dict=False)[0]
            output.square().mean().backward()
            optimizer.step()
            scheduler.step()
            ema.step(model.parameters())
            return {
                "parameters": {
                    name: parameter.detach().clone()
                    for name, parameter in model.named_parameters()
                },
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "ema": [value.detach().clone() for value in ema.shadow_params],
            }

        reference = run(False)
        candidate = run(True)
        self.assertEqual(reference["parameters"].keys(), candidate["parameters"].keys())
        for name, value in reference["parameters"].items():
            torch.testing.assert_close(candidate["parameters"][name], value, atol=0, rtol=0)
        self.assertEqual(reference["optimizer"]["param_groups"], candidate["optimizer"]["param_groups"])
        for reference_state, candidate_state in zip(
            reference["optimizer"]["state"].values(),
            candidate["optimizer"]["state"].values(),
            strict=True,
        ):
            self.assertEqual(reference_state.keys(), candidate_state.keys())
            for key, value in reference_state.items():
                if torch.is_tensor(value):
                    torch.testing.assert_close(candidate_state[key], value, atol=0, rtol=0)
                else:
                    self.assertEqual(candidate_state[key], value)
        self.assertEqual(reference["scheduler"], candidate["scheduler"])
        self.assertEqual(len(reference["ema"]), len(candidate["ema"]))
        for reference_shadow, candidate_shadow in zip(
            reference["ema"], candidate["ema"], strict=True
        ):
            torch.testing.assert_close(candidate_shadow, reference_shadow, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
