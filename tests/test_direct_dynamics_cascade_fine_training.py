import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from assim_lib.config import TrainingConfig
from assim_lib.direct_dynamics_cascade_fine_training import (
    FINE_BATCH_KEYS,
    _clean_code_identity,
    _evenly_spaced_indices,
    _fine_collate,
    _ipc_preflight,
)


def _sample(case_id: str) -> dict:
    height, width = 4, 4
    channels = {
        "truth": 6,
        "background": 6,
        "obs_values": 2,
        "obs_mask": 2,
        "valid_mask": 1,
        "water_mask": 5,
        "structured_conditioning": 15,
        "structured_physical_truth": 6,
        "structured_physical_background": 6,
    }
    sample = {key: torch.zeros(count, height, width, dtype=torch.float32) for key, count in channels.items()}
    sample["meta"] = {"case_id": case_id}
    sample["unneeded_large_tensor"] = torch.zeros(100, height, width)
    return sample


class FineCascadeRunnerTests(unittest.TestCase):
    def test_clean_identity_strips_git_environment_poisoning(self) -> None:
        results = [
            SimpleNamespace(stdout="a" * 40 + "\n"),
            SimpleNamespace(stdout=""),
        ]
        with (
            patch.dict(
                os.environ,
                {
                    "GIT_DIR": "/tmp/attacker",
                    "GIT_WORK_TREE": "/tmp/attacker-tree",
                    "GIT_INDEX_FILE": "/tmp/attacker-index",
                    "GIT_COMMON_DIR": "/tmp/attacker-common",
                },
            ),
            patch("subprocess.run", side_effect=results) as run,
        ):
            identity = _clean_code_identity(Path("/trusted/repo"))
        self.assertEqual(identity["git_commit"], "a" * 40)
        for call in run.call_args_list:
            self.assertFalse(any(key.startswith("GIT_") for key in call.kwargs["env"]))
            self.assertEqual(call.kwargs["cwd"], Path("/trusted/repo"))
        self.assertIn("--untracked-files=all", run.call_args_list[1].args[0])

    def test_even_subset_is_unique_and_spans_exact_archive(self) -> None:
        indices = _evenly_spaced_indices(51_792, 4_096)
        self.assertEqual(len(indices), len(set(indices)))
        self.assertEqual((indices[0], indices[-1]), (0, 51_791))

    def test_compact_parent_collate_keeps_only_contract_and_case_ids(self) -> None:
        batch = _fine_collate([_sample("a"), _sample("b")])
        self.assertEqual(set(batch), {*FINE_BATCH_KEYS, "meta"})
        self.assertEqual(batch["meta"]["case_id"], ["a", "b"])
        self.assertEqual(batch["truth"].shape, (2, 6, 4, 4))

    def test_ipc_gate_accounts_for_simultaneous_persistent_queues(self) -> None:
        config = TrainingConfig(
            train_batch_size=8,
            eval_batch_size=4,
            num_workers_train=4,
            num_workers_val=2,
        )
        generous = SimpleNamespace(f_bavail=10_000_000, f_frsize=4096)
        with patch("os.statvfs", return_value=generous):
            result = _ipc_preflight([_sample("train")], [_sample("valid")], config)
        self.assertEqual(result["train_retained_batches"], 17)
        self.assertEqual(result["validation_retained_batches"], 9)
        self.assertEqual(result["prefetch_factor"], 4)

        tiny = SimpleNamespace(f_bavail=1, f_frsize=4096)
        with (
            patch("os.statvfs", return_value=tiny),
            self.assertRaisesRegex(RuntimeError, "shared-memory safety budget"),
        ):
            _ipc_preflight([_sample("train")], [_sample("valid")], config)


if __name__ == "__main__":
    unittest.main()
