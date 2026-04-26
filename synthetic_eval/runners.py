from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


class EnsembleRunner:
    def run(self, x_true: np.ndarray, mask: np.ndarray, observed: np.ndarray, ensemble_size: int, seed: int) -> np.ndarray:
        raise NotImplementedError


@dataclass
class DummyRunner(EnsembleRunner):
    noise_std: float = 0.05

    def run(self, x_true: np.ndarray, mask: np.ndarray, observed: np.ndarray, ensemble_size: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, self.noise_std, size=(ensemble_size,) + x_true.shape).astype(np.float32)
        ensemble = x_true[None, ...].astype(np.float32) + noise
        ensemble[:, 0] = np.clip(ensemble[:, 0], 0.0, 1.0)
        if ensemble.shape[1] > 1:
            ensemble[:, 1] = np.clip(ensemble[:, 1], 0.0, None)
        return ensemble


@dataclass
class PersistenceRunner(EnsembleRunner):
    perturbation_std: float = 0.02

    def run(self, x_true: np.ndarray, mask: np.ndarray, observed: np.ndarray, ensemble_size: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        fill = np.zeros_like(x_true, dtype=np.float32)
        for channel in range(x_true.shape[0]):
            obs = observed[channel][mask.astype(bool)]
            fill[channel] = float(obs.mean()) if obs.size else 0.0
        base = np.where(mask[None, :, :] > 0, observed, fill)
        return base[None, ...] + rng.normal(0.0, self.perturbation_std, size=(ensemble_size,) + x_true.shape).astype(np.float32)


class ConcatRunner(EnsembleRunner):
    def __init__(
        self,
        run_dir: str,
        checkpoint_name: str,
        channel_mean,
        channel_std,
        num_timesteps: int = 50,
        method: str = "euler",
        device: str | None = None,
    ):
        import torch

        from fid.export_concat_samples import load_concat_sampler
        from utils import get_device

        self.channel_mean = channel_mean
        self.channel_std = channel_std
        self.num_timesteps = num_timesteps
        self.method = method
        self.device = device or get_device()
        if not os.path.isdir(run_dir):
            raise FileNotFoundError(run_dir)
        self.sampler = load_concat_sampler(run_dir, checkpoint_name, self.device)

    def run(self, x_true: np.ndarray, mask: np.ndarray, observed: np.ndarray, ensemble_size: int, seed: int) -> np.ndarray:
        import torch

        from fid.export_concat_samples import IMAGE_SIZE
        from utils import channel_denormalize, channel_normalize

        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        x_norm = channel_normalize(torch.from_numpy(x_true).float(), self.channel_mean, self.channel_std)
        mask_t = torch.from_numpy(mask).float().view(1, 1, *mask.shape).to(self.device)
        observed_norm = x_norm.unsqueeze(0).to(self.device) * mask_t
        mask_batch = mask_t.expand(ensemble_size, -1, -1, -1)
        observed_batch = observed_norm.expand(ensemble_size, -1, -1, -1)
        samples = self.sampler.sample_conditioned(
            mask=mask_batch,
            observed=observed_batch,
            size=IMAGE_SIZE,
            num_timesteps=self.num_timesteps,
            device=self.device,
            method=self.method,
        )
        return channel_denormalize(samples.detach().cpu(), self.channel_mean, self.channel_std).numpy().astype(np.float32)
