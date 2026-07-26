from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import DataLoader


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
