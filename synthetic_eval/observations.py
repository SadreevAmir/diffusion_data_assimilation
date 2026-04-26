from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from utils import generate_satellite_track_mask


MaskType = Literal["random", "block", "swath"]


@dataclass(frozen=True)
class ObservationConfig:
    mask_type: MaskType = "random"
    density: float = 0.05
    noise_std: float = 0.0
    seed: int = 0
    block_size: int = 16
    n_tracks_range: tuple[int, int] = (1, 4)


def make_observation_mask(
    image_size: tuple[int, int],
    config: ObservationConfig,
    valid_mask: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng(config.seed)
    h, w = image_size
    valid = np.ones((h, w), dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    if valid.shape != (h, w):
        raise ValueError(f"valid_mask shape {valid.shape} does not match image_size {(h, w)}")

    if config.mask_type == "random":
        mask = (rng.random((h, w)) < config.density) & valid
        return mask.astype(np.float32)

    if config.mask_type == "block":
        return _make_block_mask((h, w), config.density, config.block_size, valid, rng)

    if config.mask_type == "swath":
        state = np.random.get_state()
        np.random.seed(int(rng.integers(0, 2**31 - 1)))
        try:
            mask = generate_satellite_track_mask(
                image_size=(h, w),
                batch_size=1,
                valid_mask=valid.astype(np.float32),
                n_tracks_range=config.n_tracks_range,
            )[0]
        finally:
            np.random.set_state(state)
        return mask.astype(np.float32)

    raise ValueError(f"Unknown mask_type: {config.mask_type}")


def _make_block_mask(
    image_size: tuple[int, int],
    density: float,
    block_size: int,
    valid: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    h, w = image_size
    target = int(round(float(density) * int(valid.sum())))
    mask = np.zeros((h, w), dtype=bool)
    block_size = max(1, int(block_size))
    max_attempts = max(32, 4 * target // (block_size * block_size) + 32)

    for _ in range(max_attempts):
        if int((mask & valid).sum()) >= target:
            break
        y0 = int(rng.integers(0, max(1, h - block_size + 1)))
        x0 = int(rng.integers(0, max(1, w - block_size + 1)))
        mask[y0 : y0 + block_size, x0 : x0 + block_size] = True

    candidates = np.flatnonzero((mask & valid).reshape(-1))
    if candidates.size > target > 0:
        keep = rng.choice(candidates, size=target, replace=False)
        out = np.zeros(h * w, dtype=bool)
        out[keep] = True
        return out.reshape(h, w).astype(np.float32)
    return (mask & valid).astype(np.float32)


def make_sparse_observation(
    x_true: np.ndarray,
    mask: np.ndarray,
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    x = np.asarray(x_true, dtype=np.float32)
    mask_2d = np.asarray(mask, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError(f"Expected x_true shape (C, H, W), got {x.shape}")
    if mask_2d.shape != x.shape[-2:]:
        raise ValueError(f"Mask shape {mask_2d.shape} does not match field shape {x.shape[-2:]}")
    noise = rng.normal(0.0, noise_std, size=x.shape).astype(np.float32) if noise_std > 0 else 0.0
    return (x + noise) * mask_2d[None, :, :]

