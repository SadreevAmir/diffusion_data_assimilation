from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import DataLoader


class PersistentWorkerRngCompatibilityAdapter:
    """Preserve persistent-worker CPU RNG consumption without tensor IPC."""

    def __init__(self, loader):
        if int(getattr(loader, "num_workers", -1)) != 0:
            raise ValueError("RNG compatibility adapter requires num_workers=0")
        self.loader = loader
        self.iterator_count = 0

    def __iter__(self):
        if self.iterator_count == 0:
            self.iterator_count += 1
            return self.loader.__iter__()
        state = torch.random.get_rng_state()
        try:
            iterator = self.loader.__iter__()
        finally:
            torch.random.set_rng_state(state)
        self.iterator_count += 1
        return iterator

    def __len__(self):
        return len(self.loader)

    def __getattr__(self, name):
        return getattr(self.loader, name)


def preserve_persistent_worker_rng(loader):
    """Wrap a loader, including the base loader created by Accelerate."""

    base = getattr(loader, "base_dataloader", None)
    if base is not None:
        loader.base_dataloader = PersistentWorkerRngCompatibilityAdapter(base)
        return loader
    return PersistentWorkerRngCompatibilityAdapter(loader)


def dataloader_batch_count(dataset_size: int, batch_size: int, *, drop_last: bool) -> int:
    """Return the exact number of batches produced by the configured loader."""
    if dataset_size < 0 or batch_size <= 0:
        raise ValueError("dataset_size must be non-negative and batch_size must be positive")
    if drop_last:
        return dataset_size // batch_size
    return (dataset_size + batch_size - 1) // batch_size


def optimizer_step_budget(
    dataloader_batches: int,
    num_epochs: int,
    gradient_accumulation_steps: int,
) -> tuple[int, int]:
    """Return optimizer steps per epoch and over the complete training run."""
    if dataloader_batches <= 0 or num_epochs <= 0 or gradient_accumulation_steps <= 0:
        raise ValueError("step-budget inputs must be positive")
    steps_per_epoch = (
        dataloader_batches + gradient_accumulation_steps - 1
    ) // gradient_accumulation_steps
    return steps_per_epoch, steps_per_epoch * num_epochs


def validate_optimizer_step_budget(config, dataloader_batches: int) -> tuple[int, int]:
    """Fail closed against the actual DataLoader length used by the runtime."""
    steps_per_epoch, planned_steps = optimizer_step_budget(
        dataloader_batches,
        int(config.num_epochs),
        int(config.gradient_accumulation_steps),
    )
    if int(config.lr_warmup_steps) >= planned_steps:
        raise ValueError("learning-rate warmup consumes the entire planned training run")
    minimum = int(config.minimum_optimizer_steps)
    if planned_steps < minimum:
        raise ValueError(
            f"actual runtime optimizer-step budget {planned_steps} is below the "
            f"predeclared minimum {minimum}"
        )
    return steps_per_epoch, planned_steps


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_normalized_xy_grid(
    height: int,
    width: int,
    batch_size: int = 1,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    ys = torch.linspace(0.0, 1.0, height, device=device, dtype=dtype)
    xs = torch.linspace(0.0, 1.0, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([xx, yy], dim=0)
    grid = (grid - 0.5) / np.sqrt(1.0 / 12.0)
    return grid.unsqueeze(0).expand(batch_size, -1, -1, -1)


def add_noise(images: torch.Tensor, timesteps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    time = timesteps.view(-1, *([1] * (images.dim() - 1)))
    noise = torch.randn_like(images)
    noisy_images = (1.0 - time) * images + time * noise
    return noisy_images, noise - images


def build_dataloader(dataset, batch_size: int, num_workers: int, *, shuffle: bool) -> DataLoader:
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
        "drop_last": shuffle,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = 4
    return DataLoader(dataset, **kwargs)
