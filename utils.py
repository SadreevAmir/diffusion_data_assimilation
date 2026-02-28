import os
from typing import Optional, Callable, List, Literal
import numpy as np
import torch
from torch.utils.data import Dataset


def get_device() -> str:
    """Returns the best available device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


class NpyImageDataset(Dataset):
    def __init__(self,
                 folder: str,
                 file_list: Optional[List[str]] = None,
                 transform: Optional[Callable] = None,
                 preload: bool = False,
                 mmap_mode: Literal['r', 'r+', 'w+', 'c', None] = None):

        self.folder = folder
        if file_list is None:
            self.files = sorted([f for f in os.listdir(folder) if f.endswith('.npy')])
        else:
            self.files = file_list
        if len(self.files) == 0:
            raise ValueError(f"No .npy files found in {folder}")
        self.transform = transform
        self.preload = preload
        self.mmap_mode = mmap_mode

        self._data = None
        if self.preload:
            self._data = [self._load_npy(os.path.join(self.folder, f)) for f in self.files]

    def __len__(self):
        return len(self.files)

    def _load_npy(self, path: str) -> torch.Tensor:
        arr = np.load(path, mmap_mode=self.mmap_mode) if self.mmap_mode else np.load(path)
        arr = np.asarray(arr)
        if arr.ndim == 3:
            if arr.shape[0] == 2:
                arr_cfirst = arr
            elif arr.shape[-1] == 2:
                arr_cfirst = np.moveaxis(arr, -1, 0)
            else:
                arr_cfirst = arr
        elif arr.ndim == 2:
            arr_cfirst = arr[None, ...]
        else:
            raise ValueError(f"Unsupported array shape {arr.shape} in file {path}")

        if arr_cfirst.dtype != np.float32:
            arr_cfirst = arr_cfirst.astype(np.float32)

        tensor = torch.from_numpy(arr_cfirst)
        return tensor

    def __getitem__(self, idx):
        if self.preload:
            assert self._data is not None
            tensor = self._data[idx]
        else:
            path = os.path.join(self.folder, self.files[idx])
            tensor = self._load_npy(path)

        if self.transform is not None:
            tensor = self.transform(tensor)

        return tensor



def channel_normalize(x: torch.Tensor, channel_mean, channel_std) -> torch.Tensor:
    mean = torch.as_tensor(channel_mean, device=x.device, dtype=x.dtype).view(-1, 1, 1)
    std  = torch.as_tensor(channel_std,  device=x.device, dtype=x.dtype).view(-1, 1, 1)
    return (x - mean) / std

def channel_denormalize(images, channel_mean, channel_std):
    mean = torch.as_tensor(channel_mean, device=images.device, dtype=images.dtype).view(-1, 1, 1)
    std  = torch.as_tensor(channel_std,  device=images.device, dtype=images.dtype).view(-1, 1, 1)
    images = images * std + mean
    images[:, 0] = torch.clip(images[:, 0], min=0, max=1)
    images[:, 1] = torch.clip(images[:, 1], min=0)
    return images
 


def make_normalized_xy_grid(
    H: int,
    W: int,
    batch_size: int = 1,
    device='cpu',
    dtype: torch.dtype = torch.float32,
):
    ys = torch.linspace(0.0, 1.0, H, device=device, dtype=dtype)
    xs = torch.linspace(0.0, 1.0, W, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")

    grid = torch.stack([xx, yy], dim=0)
    mean = 0.5
    std = np.sqrt(1.0 / 12.0)
    grid = (grid - mean) / std

    grid = grid.unsqueeze(0).expand(batch_size, -1, -1, -1)

    return grid


def add_noise(images: torch.Tensor, timesteps):
  timesteps = timesteps.view(-1, *([1]*(images.dim() - 1)))
  eps = torch.randn_like(images)
  noisy_images = (1 - timesteps) * images + timesteps * eps

  return noisy_images, eps - images


def generate_satellite_track_mask(
    image_size: tuple,
    batch_size: int = 1,
    valid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Generate a batch of binary masks with 0–5 random straight satellite tracks each.

    Fully vectorized: no Python loops. All track parameters are generated at once,
    then scattered into the mask array in two passes (row-dominant / col-dominant).

    Args:
        image_size: (H, W)
        batch_size: number of masks to generate.
        valid_mask: float32 array of shape (H, W), 1 where observations
                    are allowed. Defaults to all-ones (no restriction).

    Returns:
        masks: float32 array of shape (batch_size, H, W).
    """
    H, W = image_size
    if valid_mask is None:
        valid_mask = np.ones((H, W), dtype=np.float32)

    n_max = 5
    masks = np.zeros((batch_size, H, W), dtype=np.float32)

    # Draw all parameters at once: (bs, n_max)
    n_tracks = np.random.randint(0, n_max + 1, size=batch_size)
    active = (np.arange(n_max)[None, :] < n_tracks[:, None]).ravel()  # (bs*n_max,)

    y0 = np.random.uniform(0, H, size=batch_size * n_max)
    x0 = np.random.uniform(0, W, size=batch_size * n_max)
    angles = np.random.uniform(0, np.pi, size=batch_size * n_max)
    dy = np.sin(angles)
    dx = np.cos(angles)
    use_rows = np.abs(dy) >= np.abs(dx)

    # Batch index for each (sample, track) pair
    batch_idx = np.repeat(np.arange(batch_size), n_max)  # (bs*n_max,)

    rows_arr = np.arange(H)
    cols_arr = np.arange(W)

    # --- Row-dominant tracks: iterate over rows, compute col per row ---
    sel = np.where(active & use_rows)[0]
    if len(sel):
        b = batch_idx[sel]                          # (n,)
        cols_t = np.round(
            x0[sel, None] + (rows_arr - y0[sel, None]) * (dx[sel] / dy[sel])[:, None]
        ).astype(int)                               # (n, H)
        b_exp = np.broadcast_to(b[:, None], cols_t.shape)
        r_exp = np.broadcast_to(rows_arr,   cols_t.shape)
        ok = (cols_t >= 0) & (cols_t < W)
        masks[b_exp[ok], r_exp[ok], cols_t[ok]] = 1.0

    # --- Col-dominant tracks: iterate over cols, compute row per col ---
    sel = np.where(active & ~use_rows)[0]
    if len(sel):
        b = batch_idx[sel]                          # (n,)
        rows_t = np.round(
            y0[sel, None] + (cols_arr - x0[sel, None]) * (dy[sel] / dx[sel])[:, None]
        ).astype(int)                               # (n, W)
        b_exp = np.broadcast_to(b[:, None],  rows_t.shape)
        c_exp = np.broadcast_to(cols_arr,    rows_t.shape)
        ok = (rows_t >= 0) & (rows_t < H)
        masks[b_exp[ok], rows_t[ok], c_exp[ok]] = 1.0

    return masks * valid_mask[None, :]

