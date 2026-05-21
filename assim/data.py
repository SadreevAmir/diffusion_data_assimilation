from datetime import datetime, timedelta

import numpy as np


class SmokeDataset:
    name = "SmokeDataset"

    def __init__(self, config):
        self.config = config
        self.n_cases = int(config.get("n_cases", 4))
        self.shape = tuple(int(v) for v in config.get("shape", [2, 16, 16]))
        self.noise_std = float(config.get("noise_std", 0.2))
        self.seed = int(config.get("seed", 42))
        self.start_time = datetime.fromisoformat(config.get("start_time", "2000-01-01"))

    def __len__(self):
        return self.n_cases

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

    def __getitem__(self, idx):
        rng = np.random.default_rng(self.seed + idx)
        truth = rng.normal(0.0, 1.0, size=self.shape).astype(np.float32)
        background = truth + rng.normal(0.0, self.noise_std, size=self.shape).astype(np.float32)
        tensors = {
            "background": background,
            "observations": None,
        }
        targets = {
            "truth": truth,
        }
        meta = {
            "case_id": f"case_{idx:04d}",
            "time": (self.start_time + timedelta(days=idx)).date().isoformat(),
        }
        return tensors, targets, meta


datasets = {
    SmokeDataset.name: SmokeDataset,
}
