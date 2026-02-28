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
    H: int = 320,
    W: int = 256,
    batch_size = 1,
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


def generate_satellite_track_mask(image_size: tuple) -> np.ndarray:
    """
    Generate a binary mask with 2–5 random straight satellite tracks.

    For each track: picks a random point and a random direction,
    then draws a line across the full image.

    Args:
        image_size: (H, W)

    Returns:
        mask: float32 array of shape (H, W), 0 everywhere except tracks (1).
    """
    H, W = image_size
    mask = np.zeros((H, W), dtype=np.float32)

    n_tracks = np.random.randint(2, 6)  # 2, 3, 4, or 5 tracks

    for _ in range(n_tracks):
        y0 = np.random.uniform(0, H)
        x0 = np.random.uniform(0, W)
        angle = np.random.uniform(0, np.pi)
        dy = np.sin(angle)
        dx = np.cos(angle)

        if abs(dy) >= abs(dx):
            rows = np.arange(H)
            t = (rows - y0) / dy
            cols = np.round(x0 + t * dx).astype(int)
            valid = (cols >= 0) & (cols < W)
            mask[rows[valid], cols[valid]] = 1.0
        else:
            cols = np.arange(W)
            t = (cols - x0) / dx
            rows = np.round(y0 + t * dy).astype(int)
            valid = (rows >= 0) & (rows < H)
            mask[rows[valid], cols[valid]] = 1.0

    return mask

