from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from assim_lib import locked_mc_dropout_ensemble as locked
from assim_lib.config import TrainingConfig
from assim_lib.model_io import build_unet


class LockedMcDropoutEnsembleTests(unittest.TestCase):
    def test_frozen_contract_file_matches_compiled_contract(self) -> None:
        repo = Path(locked.__file__).resolve().parents[1]
        path = repo / locked.CONTRACT_CONFIG
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), locked.frozen_contract())
        self.assertEqual(locked.MODE, "validation_locked_mc_dropout_sampling")
        self.assertEqual(locked.CANDIDATE_METHOD, "locked_mc_dropout_p010_final_ema_ensemble")
        self.assertEqual(locked.DEPENDENT_GATE_MODE, "validation_locked_mc_dropout_gate")

    def test_seed_schedules_are_exact_unique_and_disjoint(self) -> None:
        noise = [
            locked.noise_seed(case_index, member_index)
            for case_index in range(locked.CASE_COUNT)
            for member_index in range(locked.MEMBER_COUNT)
        ]
        masks = [
            locked.mask_seed(case_index, member_index, layer_index)
            for case_index in range(locked.CASE_COUNT)
            for member_index in range(locked.MEMBER_COUNT)
            for layer_index in range(len(locked.ACTIVE_DROPOUT_LAYERS))
        ]
        self.assertEqual(noise, list(range(1234, 1634)))
        self.assertEqual(len(masks), 5600)
        self.assertEqual(len(masks), len(set(masks)))
        self.assertTrue(set(noise).isdisjoint(masks))

    def test_locked_mask_repeats_for_ode_calls_and_changes_by_member(self) -> None:
        controller = locked.LockedDropoutController()
        wrapper = locked.LockedDropout(controller, locked.ACTIVE_DROPOUT_LAYERS[0], 0)
        controller.wrappers = [wrapper]
        controller.model_class = "test"
        value = torch.ones((1, 3, 5, 7), dtype=torch.float32)

        controller.activate(0, (0,))
        first = wrapper(value)
        second = wrapper(value)
        self.assertTrue(torch.equal(first, second))
        first_record = controller._records[0][0]["layers"][0]
        self.assertEqual(first_record["mask_seed"], 271_828_000)
        self.assertEqual(first_record["mask_shape"], [3, 5, 7])

        controller.activate(0, (1,))
        third = wrapper(value)
        self.assertFalse(torch.equal(first, third))
        self.assertEqual(controller._records[0][1]["layers"][0]["mask_seed"], 271_828_014)

    def test_locked_mask_fails_if_nonbatch_activation_shape_changes(self) -> None:
        controller = locked.LockedDropoutController()
        wrapper = locked.LockedDropout(controller, locked.ACTIVE_DROPOUT_LAYERS[0], 0)
        controller.wrappers = [wrapper]
        controller.model_class = "test"
        controller.activate(0, (0,))
        wrapper(torch.ones((1, 2, 3, 4)))
        with self.assertRaisesRegex(ValueError, "shape changed"):
            wrapper(torch.ones((1, 2, 3, 5)))

    def test_exact_unet_inventory_is_admitted_and_restored(self) -> None:
        repo = Path(locked.__file__).resolve().parents[1]
        config = TrainingConfig.from_dict(
            json.loads(
                (repo / "config/methods/concat_conditioning_diffusion_balanced_2f.json").read_text(
                    encoding="utf-8"
                )
            )
        )
        model = build_unet(config)
        model.eval()
        before = {name: module for name, module in model.named_modules() if type(module) is nn.Dropout}
        self.assertEqual(set(before), set(locked.ACTIVE_DROPOUT_LAYERS) | set(locked.ZERO_DROPOUT_LAYERS))
        controller = locked.LockedDropoutController()
        controller.install(model)
        after = dict(model.named_modules())
        self.assertTrue(
            all(isinstance(after[name], locked.LockedDropout) for name in locked.ACTIVE_DROPOUT_LAYERS)
        )
        self.assertTrue(all(after[name] is before[name] for name in locked.ZERO_DROPOUT_LAYERS))
        controller.restore()
        restored = dict(model.named_modules())
        self.assertTrue(all(restored[name] is before[name] for name in before))

    def test_inventory_fails_closed_for_extra_dropout_and_batchnorm(self) -> None:
        fake_class = type("UNet2DModel", (nn.Sequential,), {})
        model = fake_class(nn.Dropout(0.1), nn.BatchNorm1d(2))
        model.config = SimpleNamespace(
            sample_size=(320, 256),
            in_channels=13,
            out_channels=2,
            layers_per_block=1,
            block_out_channels=(96, 192, 384, 384),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
            norm_num_groups=32,
            dropout=0.1,
        )
        model.eval()
        with self.assertRaisesRegex(ValueError, "BatchNorm"):
            locked.LockedDropoutController().install(model)

    def test_manifest_validator_rejects_seed_drift(self) -> None:
        cases = []
        for case_index in range(locked.CASE_COUNT):
            members = []
            for member_index in range(locked.MEMBER_COUNT):
                layers = []
                for layer_index, layer_name in enumerate(locked.ACTIVE_DROPOUT_LAYERS):
                    layers.append(
                        {
                            "layer_index": layer_index,
                            "layer_name": layer_name,
                            "mask_seed": locked.mask_seed(case_index, member_index, layer_index),
                            "mask_shape": [1],
                            "mask_sha256": "a" * 64,
                            "keep_fraction": 0.9,
                        }
                    )
                members.append({"member_index": member_index, "layers": layers})
            cases.append({"case_index": case_index, "members": members})
        payload = {
            "schema_version": 1,
            "candidate_method": locked.CANDIDATE_METHOD,
            "model_class": "UNet2DModel",
            "torch_version": torch.__version__,
            "active_layers": list(locked.ACTIVE_DROPOUT_LAYERS),
            "excluded_zero_probability_layers": list(locked.ZERO_DROPOUT_LAYERS),
            "cases": cases,
        }
        self.assertEqual(locked._validate_mask_manifest(payload), cases)
        cases[2]["members"][3]["layers"][4]["mask_seed"] += 1
        with self.assertRaisesRegex(ValueError, "provenance differs"):
            locked._validate_mask_manifest(payload)

    def test_standalone_help_and_script_fail_closed(self) -> None:
        repo = Path(locked.__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "-m", "assim_lib.locked_mc_dropout_ensemble", "--help"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--source-experiment", result.stdout)
        script = (repo / "scripts/run_locked_mc_dropout_ensemble.sh").read_text(encoding="utf-8")
        self.assertIn('MODE" != "validation_locked_mc_dropout_sampling', script)
        self.assertNotIn("curl", script)
        self.assertNotIn("wget", script)


if __name__ == "__main__":
    unittest.main()
