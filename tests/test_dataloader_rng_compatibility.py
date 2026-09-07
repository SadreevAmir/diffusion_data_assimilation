from __future__ import annotations

import unittest

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader, TensorDataset

from assim_lib.runtime import preserve_persistent_worker_rng


class DataLoaderRngCompatibilityTests(unittest.TestCase):
    @staticmethod
    def _trajectory(num_workers: int, *, adapt_zero_worker: bool):
        torch.manual_seed(1701)
        # Model construction consumes CPU RNG before the first loader iterator.
        torch.rand(41)
        dataset = TensorDataset(torch.arange(24, dtype=torch.int64))
        train = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
        )
        validation = DataLoader(
            dataset,
            batch_size=6,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
        )
        accelerator = Accelerator(cpu=True)
        train, validation = accelerator.prepare(train, validation)
        if adapt_zero_worker:
            train = preserve_persistent_worker_rng(train)
            validation = preserve_persistent_worker_rng(validation)

        trace = []
        try:
            for _ in range(3):
                train_order = torch.cat([batch[0] for batch in train]).tolist()
                validation_order = torch.cat([batch[0] for batch in validation]).tolist()
                validation_first_again = next(iter(validation))[0].tolist()
                cpu_rng_probe = torch.randint(0, 2**31, (4,)).tolist()
                trace.append(
                    (
                        train_order,
                        validation_order,
                        validation_first_again,
                        cpu_rng_probe,
                    )
                )
        finally:
            for prepared in (train, validation):
                base = getattr(prepared, "base_dataloader", prepared)
                base = getattr(base, "loader", base)
                iterator = getattr(base, "_iterator", None)
                if iterator is not None:
                    iterator._shutdown_workers()
        return trace

    def test_zero_worker_adapter_matches_persistent_worker_train_val_rng(self) -> None:
        persistent = self._trajectory(1, adapt_zero_worker=False)
        zero_worker = self._trajectory(0, adapt_zero_worker=True)
        self.assertEqual(zero_worker, persistent)


if __name__ == "__main__":
    unittest.main()
