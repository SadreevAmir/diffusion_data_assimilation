import os
from typing import Optional, Callable, List
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
                 mmap_mode: Optional[str] = None):

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
            tensor = self._data[idx]
        else:
            path = os.path.join(self.folder, self.files[idx])
            tensor = self._load_npy(path)

        if self.transform is not None:
            tensor = self.transform(tensor)

        return tensor



def channel_normalize(x: torch.Tensor, channel_mean: np.array, channel_std: np.array ) -> torch.Tensor:
    mean = torch.as_tensor(channel_mean, device=x.device, dtype=x.dtype).view(-1, 1, 1)
    std  = torch.as_tensor(channel_std,  device=x.device, dtype=x.dtype).view(-1, 1, 1)
    return (x - mean) / std

def channel_denormalize(images, channel_mean, channel_std):
    mean = torch.as_tensor(channel_mean, device=images.device, dtype=images.dtype).view(-1, 1, 1)
    std  = torch.as_tensor(channel_std,  device=images.device, dtype=images.dtype).view(-1, 1, 1)
    images = images*std + mean
    images[0] = torch.clip(images[0], min=0, max=1)
    images[1] = torch.clip(images[1], min=0)
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

