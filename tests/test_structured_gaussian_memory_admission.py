from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from assim_lib import structured_gaussian_memory_admission as admission
from assim_lib.config import TrainingConfig


class StructuredGaussianMemoryAdmissionTests(unittest.TestCase):
    def test_cpu_parity_is_exact_for_output_update_ema_and_rng(self) -> None:
        result = admission.cpu_parity_record()
        self.assertEqual(result["status"], "passed")
        self.assertTrue(result["forward_bitwise_equal"])
        self.assertTrue(result["rng_state_equal"])
        self.assertTrue(result["post_adam_parameters_bitwise_equal"])
        self.assertTrue(result["post_ema_parameters_bitwise_equal"])
        self.assertGreater(len(result["checkpointed_modules"]), 0)

    def test_exact_checkpointed_pilot_contract_is_accepted(self) -> None:
        repo = Path(admission.__file__).resolve().parents[1]
        method = json.loads((repo / admission.EXPECTED_METHOD).read_text())
        stats_path = (repo / admission.EXPECTED_METHOD).parent / method[
            "structured_state_stats_path"
        ]
        method["structured_state_stats"] = json.loads(stats_path.resolve().read_text())
        config = TrainingConfig.from_dict(method)
        admission._validate_pilot_config(config)
        with self.assertRaisesRegex(ValueError, "contract differs"):
            admission._validate_pilot_config(replace_config(config, train_batch_size=8))

    def test_runner_is_fixed_online_and_never_opens_test(self) -> None:
        repo = Path(admission.__file__).resolve().parents[1]
        script = (repo / "scripts/run_structured_gaussian_memory_admission.sh").read_text()
        source = Path(admission.__file__).read_text()
        self.assertIn(admission.EXPECTED_EXPERIMENT, script)
        self.assertIn(admission.EXPECTED_PROTOCOL, source)
        self.assertNotIn("CLEARML_OFFLINE_MODE", script)
        self.assertNotIn('split="test"', source)
        self.assertIn('"test_2023_used": False', source)
        self.assertIn("EXPECTED_TRAIN_BATCHES = 2", source)
        self.assertIn("EXPECTED_VALIDATION_BATCHES = 1", source)

    def test_run_admission_refuses_wrong_path_and_existing_output(self) -> None:
        repo = Path(admission.__file__).resolve().parents[1]
        experiment = repo / admission.EXPECTED_EXPERIMENT
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result"
            with self.assertRaisesRegex(ValueError, "exact frozen"):
                admission.run_admission(repo / "wrong.json", output)
            output.mkdir()
            with self.assertRaisesRegex(ValueError, "already exists"):
                admission.run_admission(experiment, output)

    def test_trainer_adapter_requires_checkpointing_and_exact_batches(self) -> None:
        config = TrainingConfig(activation_checkpointing=False)
        with mock.patch.object(admission.UNetTrainer, "__init__", return_value=None):
            with self.assertRaisesRegex(ValueError, "checkpointing=true"):
                admission.MemoryAdmissionTrainer(
                    config,
                    torch.nn.Linear(2, 2),
                    object(),
                    object(),
                    object(),
                    object(),
                    object(),
                )


def replace_config(config: TrainingConfig, **changes) -> TrainingConfig:
    values = {**config.__dict__, **changes}
    return TrainingConfig.from_dict(values)


if __name__ == "__main__":
    unittest.main()
