from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from assim_lib.two_stage_native_learning_pilot import (
    CONTRACT_SHA256S,
    MIN_SHM_BYTES,
    _build_experiment,
)


class TwoStageNativeLearningPilotTests(unittest.TestCase):
    def test_all_pinned_contract_digests_are_complete_sha256_values(self) -> None:
        self.assertEqual(len(CONTRACT_SHA256S), 6)
        for name, digest in CONTRACT_SHA256S.items():
            with self.subTest(name=name):
                self.assertEqual(len(digest), 64)
                self.assertTrue(all(character in "0123456789abcdef" for character in digest))

    def test_assimilation_and_dynamics_contracts_are_distinct_and_safe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "repo"
            output = base / "result"
            contracts = base / "contracts"
            for role in ("assimilation", "dynamics"):
                config = _build_experiment(role, root, output, contracts)
                training = config["training"]
                self.assertEqual(training["num_workers_train"], 4)
                self.assertEqual(training["num_workers_val"], 2)
                self.assertFalse(training["activation_checkpointing"])
                self.assertEqual(training["sample_every_n_epochs"], 1)
                self.assertEqual(training["metric_every_n_epochs"], 2)
                self.assertEqual(training["metric_num_cases"], 4)
                self.assertEqual(training["metric_num_ensemble"], 5)
                self.assertIn(role, config["task_name"])
                self.assertIn(role, config["data_config"])
                self.assertIn(role, training["structured_state_stats_path"])
            self.assertEqual(MIN_SHM_BYTES, 32 * 1024**3)

    def test_scientific_validation_is_tied_to_final_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            for role in ("assimilation", "dynamics"):
                config = _build_experiment(role, base / "repo", base / "out", base / "contracts")
                training = config["training"]
                epochs = 4
                sample_epochs = [
                    epoch
                    for epoch in range(epochs)
                    if (epoch + 1) % training["sample_every_n_epochs"] == 0
                ]
                metric_epochs = [
                    epoch
                    for epoch in range(epochs)
                    if (epoch + 1) % training["metric_every_n_epochs"] == 0
                ]
                self.assertEqual(sample_epochs, [0, 1, 2, 3])
                self.assertEqual(metric_epochs, [1, 3])

    def test_unknown_role_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown learning role"):
            _build_experiment("joint", Path("/repo"), Path("/result"), Path("/contracts"))


if __name__ == "__main__":
    unittest.main()
