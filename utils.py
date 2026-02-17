import os
from typing import Optional, Callable, List
import numpy as np
import torch
from torch.utils.data import Dataset



"""
dataset.py

Датасет для последовательностей кадров с интервалом сутки.

Структура файлов:
  ocean+atmosphere_24_{date}_processed{hour:03d}.npy
  shape каждого файла: (2, 320, 256)

Логика:
  - Группируем файлы по часу (005, 006, ..., 023)
  - Внутри каждой группы сортируем по дате
  - Один сэмпл = T последовательных дней одного и того же часа
  
Пример при T=4, hour=005:
  2018-07-06_005 → 2018-07-07_005 → 2018-07-08_005 → 2018-07-09_005
"""

import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict


class SequentialIceDataset(Dataset):
    """
    Параметры
    ---------
    folder : str
        Путь к папке с .npy файлами.
    num_frames : int
        Число кадров в одной последовательности (T).
        При T=1 поведение идентично старому NpyImageDataset.
    transform : callable, optional
        Применяется к каждому кадру независимо.
        Например: lambda x: channel_normalize(x, mean, std)
    hours : list[int], optional
        Какие часы использовать. None = все доступные.
        Пример: [5, 12, 18] — только утро/день/вечер.
    stride : int
        Шаг между последовательностями внутри одной группы часов.
        stride=1: максимально много сэмплов (скользящее окно)
        stride=T: без перекрытия
    mmap_mode : str or None
        Режим memory-mapped чтения numpy. 'r' = читаем без загрузки в RAM.
    """

    def __init__(
        self,
        folder: str,
        num_frames: int = 4,
        transform=None,
        hours: list = None,
        stride: int = 1,
        mmap_mode: str = 'r',
    ):
        self.folder = folder
        self.num_frames = num_frames
        self.transform = transform
        self.mmap_mode = mmap_mode

        # ── Парсим имена файлов ──────────────────────────────────────────────
        # Паттерн: ocean+atmosphere_24_{date}_processed{hour:03d}.npy
        pattern = re.compile(
            r'ocean\+atmosphere_24_(\d{4}-\d{2}-\d{2})_processed(\d{3})\.npy'
        )

        # Группируем по часу: {hour: [(date_str, filepath), ...]}
        groups = defaultdict(list)

        for fname in sorted(os.listdir(folder)):
            m = pattern.match(fname)
            if m:
                date_str, hour_str = m.group(1), m.group(2)
                hour = int(hour_str)
                if hours is None or hour in hours:
                    groups[hour].append((date_str, os.path.join(folder, fname)))

        # Сортируем каждую группу по дате
        for hour in groups:
            groups[hour].sort(key=lambda x: x[0])

        # ── Строим индекс сэмплов ────────────────────────────────────────────
        # Каждый сэмпл — это список из T путей к файлам
        # (T последовательных дат одного часа)
        self.samples = []

        for hour, entries in groups.items():
            dates = [e[0] for e in entries]
            paths = [e[1] for e in entries]
            n = len(paths)

            # Скользящее окно с заданным stride
            for start in range(0, n - num_frames + 1, stride):
                seq_paths = paths[start : start + num_frames]
                seq_dates = dates[start : start + num_frames]

                # Проверяем что даты действительно идут подряд (нет пропусков)
                if self._is_consecutive(seq_dates):
                    self.samples.append(seq_paths)

        if len(self.samples) == 0:
            raise ValueError(
                f"Не найдено ни одной последовательности длиной {num_frames} "
                f"в папке {folder}. Проверь num_frames и наличие файлов."
            )

        print(f"SequentialIceDataset: {len(self.samples)} последовательностей "
              f"по {num_frames} кадров из {folder}")

    def _is_consecutive(self, dates: list) -> bool:
        """Проверяет что список дат идёт подряд без пропусков."""
        from datetime import datetime, timedelta
        fmt = "%Y-%m-%d"
        for i in range(1, len(dates)):
            d1 = datetime.strptime(dates[i-1], fmt)
            d2 = datetime.strptime(dates[i], fmt)
            if d2 - d1 != timedelta(days=1):
                return False
        return True

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        paths = self.samples[idx]

        frames = []
        for path in paths:
            # mmap_mode='r' — файл не загружается целиком в RAM
            arr = np.load(path, mmap_mode=self.mmap_mode)  # (2, 320, 256)
            frame = torch.from_numpy(arr.copy()).float()    # (2, 320, 256)

            if self.transform is not None:
                frame = self.transform(frame)

            frames.append(frame)

        # Стакаем по временному измерению
        # frames: список из T тензоров (2, 320, 256)
        # → (2, T, 320, 256)
        video = torch.stack(frames, dim=1)   # (2, T, H, W)

        return video


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
    device='cuda',
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

