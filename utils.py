import os
from typing import Optional, Callable, List, Literal
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import matplotlib.colors as mcolors


def get_device() -> str:
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


class MixedSatelliteTrackDataset(Dataset):
    def __init__(
        self,
        folder: str,
        image_size: tuple,
        valid_mask: Optional[np.ndarray] = None,
        npy_fraction: float = 0.5,
        generate_fraction: float = 0.3,
        empty_fraction: float = 0.2,
        n_tracks_range: tuple = (0, 5),
        file_list: Optional[List[str]] = None,
        mmap_mode: Literal['r', 'r+', 'w+', 'c', None] = None,
        length: Optional[int] = None,
    ):
        assert abs(npy_fraction + generate_fraction + empty_fraction - 1.0) < 1e-6, \
            "npy_fraction + generate_fraction + empty_fraction must equal 1.0"

        self.folder = folder
        self.image_size = image_size
        self.valid_mask = valid_mask
        self.npy_fraction = npy_fraction
        self.generate_fraction = generate_fraction
        self.n_tracks_range = n_tracks_range
        self.mmap_mode = mmap_mode

        if file_list is not None:
            self.files = file_list
        elif npy_fraction > 0:
            self.files = sorted([f for f in os.listdir(folder) if f.endswith('.npy')])
            if not self.files:
                raise ValueError(f"No .npy files found in {folder} but npy_fraction={npy_fraction}")
        else:
            self.files = []

        self._len = length if length is not None else max(len(self.files), 1)

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        r = np.random.random()

        if r < self.npy_fraction and self.files:
            path = os.path.join(self.folder, self.files[idx % len(self.files)])
            arr = np.load(path, mmap_mode=self.mmap_mode) if self.mmap_mode else np.load(path)
            arr = np.asarray(arr, dtype=np.float32)
            if arr.ndim == 2:
                arr = arr[None, :]
            return torch.from_numpy(arr)

        if r < self.npy_fraction + self.generate_fraction:
            mask = generate_satellite_track_mask(self.image_size, 1, self.valid_mask, self.n_tracks_range)
            return torch.from_numpy(mask)

        H, W = self.image_size
        return torch.zeros(1, H, W, dtype=torch.float32)


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
    n_tracks_range: tuple = (0, 5),
) -> np.ndarray:
    H, W = image_size
    if valid_mask is None:
        valid_mask = np.ones((H, W), dtype=np.float32)

    n_min, n_max = n_tracks_range
    masks = np.zeros((batch_size, H, W), dtype=np.float32)

    n_tracks = np.random.randint(n_min, n_max + 1, size=batch_size)
    active = (np.arange(n_max)[None, :] < n_tracks[:, None]).ravel()

    y0 = np.random.uniform(0, H, size=batch_size * n_max)
    x0 = np.random.uniform(0, W, size=batch_size * n_max)
    angles = np.random.uniform(0, np.pi, size=batch_size * n_max)
    dy = np.sin(angles)
    dx = np.cos(angles)
    use_rows = np.abs(dy) >= np.abs(dx)

    batch_idx = np.repeat(np.arange(batch_size), n_max)

    rows_arr = np.arange(H)
    cols_arr = np.arange(W)

    sel = np.where(active & use_rows)[0]
    if len(sel):
        b = batch_idx[sel]
        cols_t = np.round(
            x0[sel, None] + (rows_arr - y0[sel, None]) * (dx[sel] / dy[sel])[:, None]
        ).astype(int)
        b_exp = np.broadcast_to(b[:, None], cols_t.shape)
        r_exp = np.broadcast_to(rows_arr,   cols_t.shape)
        ok = (cols_t >= 0) & (cols_t < W)
        masks[b_exp[ok], r_exp[ok], cols_t[ok]] = 1.0

    sel = np.where(active & ~use_rows)[0]
    if len(sel):
        b = batch_idx[sel]
        rows_t = np.round(
            y0[sel, None] + (cols_arr - x0[sel, None]) * (dy[sel] / dx[sel])[:, None]
        ).astype(int)
        b_exp = np.broadcast_to(b[:, None],  rows_t.shape)
        c_exp = np.broadcast_to(cols_arr,    rows_t.shape)
        ok = (rows_t >= 0) & (rows_t < H)
        masks[b_exp[ok], rows_t[ok], c_exp[ok]] = 1.0

    return masks * valid_mask[None, :]


def make_plot(sea_ice_samples, channel_mean, channel_std, num_samples, title=''):
    sea_ice_samples = channel_denormalize(sea_ice_samples, channel_mean=channel_mean, channel_std=channel_std)
    n = num_samples
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    imgs = [sea_ice_samples[i][0].detach().cpu() for i in range(n)]
    vmin = min(img.min().item() for img in imgs)
    vmax = max(img.max().item() for img in imgs)

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(rows, cols, figsize=(cols*2 + 1, rows*2))
    axes = axes.flatten()
    fig.suptitle(title, fontsize=16, y=0.98)

    for idx, ax in enumerate(axes):
        if idx < n:
            ax.imshow(imgs[idx].numpy(), norm=norm, cmap='viridis')
        ax.axis('off')

    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes((0.88, 0.15, 0.02, 0.7))
    fig.colorbar(sm, cax=cbar_ax)
    plt.show()


def make_difference_plot(sea_ice_samples_1, sea_ice_samples_2, channel_mean, channel_std, num_samples, land_mask, title=''):
    sea_ice_samples_1 = channel_denormalize(sea_ice_samples_1.detach().cpu(), channel_mean=channel_mean, channel_std=channel_std)
    sea_ice_samples_2 = channel_denormalize(sea_ice_samples_2.detach().cpu(), channel_mean=channel_mean, channel_std=channel_std)
    n = num_samples
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    mask = land_mask.detach().cpu() if isinstance(land_mask, torch.Tensor) else torch.tensor(land_mask)
    mask_np = mask.numpy() if mask.ndim == 2 else mask[0].numpy()
    n_water = np.sum(mask_np)

    diff = (sea_ice_samples_1 - sea_ice_samples_2) * mask
    imgs = [diff[i][0] for i in range(n)]
    vmin = min(img.min().item() for img in imgs)
    vmax = max(img.max().item() for img in imgs)

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(rows, cols, figsize=(cols*2 + 1, rows*2 + 0.1))
    axes = axes.flatten()
    fig.suptitle("Conditioned samples delta", fontsize=16, y=0.98)

    for idx, ax in enumerate(axes):
        ax.axis('off')
        if idx >= n:
            continue
        img = imgs[idx].numpy()
        ax.imshow(img, norm=norm, cmap='viridis')
        mean_square = np.sum(img**2) / n_water
        ax.text(0.5, -0.1, f'MSE: {mean_square:.3f}',
                transform=ax.transAxes, ha='center', va='top', fontsize=8)

    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])
    fig.subplots_adjust(right=0.85, bottom=0.1)
    cbar_ax = fig.add_axes((0.88, 0.15, 0.02, 0.7))
    fig.colorbar(sm, cax=cbar_ax)
    plt.show()


def make_scalar_plot_grid(images, num_samples, title='', cmap='magma', case_labels=None):
    n = num_samples
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    imgs = [images[i].detach().cpu() for i in range(n)]
    vmin = min(img.min().item() for img in imgs)
    vmax = max(img.max().item() for img in imgs)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2 + 1, rows * 2))
    axes = np.atleast_1d(axes).flatten()
    fig.suptitle(title, fontsize=16, y=0.98)

    for idx, ax in enumerate(axes):
        if idx < n:
            ax.imshow(imgs[idx].numpy(), norm=norm, cmap=cmap)
            if case_labels is not None:
                ax.set_title(str(case_labels[idx]), fontsize=10)
        ax.axis('off')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes((0.88, 0.15, 0.02, 0.7))
    fig.colorbar(sm, cax=cbar_ax)
    plt.show()


def make_ensemble_case_plot(ensemble, channel_mean, channel_std, case_indices=None, title_prefix='Case'):
    ensemble = channel_denormalize(
        ensemble.detach().cpu(),
        channel_mean=channel_mean,
        channel_std=channel_std,
    )

    ensemble_size, n_cases = ensemble.shape[:2]
    imgs = ensemble[:, :, 0]
    vmin = imgs.min().item()
    vmax = imgs.max().item()
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    for case_idx in range(n_cases):
        fig, axes = plt.subplots(1, ensemble_size, figsize=(ensemble_size * 2, 2.6))
        axes = np.atleast_1d(axes).flatten()

        for member_idx, ax in enumerate(axes):
            ax.imshow(imgs[member_idx, case_idx].numpy(), norm=norm, cmap='viridis')
            ax.set_title(f'member {member_idx}', fontsize=9)
            ax.axis('off')

        case_label = case_idx if case_indices is None else int(case_indices[case_idx])
        fig.suptitle(f'{title_prefix} {case_label} - Ensemble samples', fontsize=14, y=0.98)
        fig.subplots_adjust(right=0.92)
        cbar_ax = fig.add_axes((0.94, 0.18, 0.015, 0.64))
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
        sm.set_array([])
        fig.colorbar(sm, cax=cbar_ax)
        plt.show()


def _rank_histogram_probabilities(ranks: np.ndarray, ensemble_size: int) -> np.ndarray:
    counts = np.bincount(ranks.astype(np.int64), minlength=ensemble_size + 1)
    total = counts.sum()
    if total == 0:
        return np.zeros(ensemble_size + 1, dtype=np.float64)
    return counts / total


def _draw_rank_histogram(ax, probabilities: np.ndarray, title: str):
    x = np.arange(len(probabilities))
    ax.bar(x, probabilities, color='black', width=0.85)
    ax.set_xticks(x)
    ax.set_xlabel('Rank of truth')
    ax.set_ylabel('Probability')
    ax.set_title(title)


def plot_rank_histogram(
    ensemble,
    truth,
    channel=0,
    stride=16,
    valid_mask=None,
    title='Rank histogram',
    case_indices=None,
    per_case=False,
    per_case_cols=4,
):
    ensemble = ensemble.detach().cpu()
    truth = truth.detach().cpu()

    if ensemble.ndim != 5:
        raise ValueError(f"Expected ensemble shape (M, N, C, H, W), got {tuple(ensemble.shape)}")
    if truth.ndim != 4:
        raise ValueError(f"Expected truth shape (N, C, H, W), got {tuple(truth.shape)}")
    if ensemble.shape[1] != truth.shape[0]:
        raise ValueError(
            f"Expected matching case counts, got ensemble {ensemble.shape[1]} and truth {truth.shape[0]}"
        )
    if not 0 <= channel < ensemble.shape[2]:
        raise ValueError(f"Channel {channel} is out of bounds for {ensemble.shape[2]} channels")

    members = ensemble[:, :, channel, ::stride, ::stride]
    target = truth[:, channel, ::stride, ::stride]

    if valid_mask is not None:
        mask = torch.as_tensor(valid_mask, dtype=torch.bool)
        if mask.ndim == 3:
            mask = mask[0]
        mask = mask[::stride, ::stride]
        members = members[:, :, mask]
        target = target[:, mask]
    else:
        members = members.reshape(members.shape[0], members.shape[1], -1)
        target = target.reshape(target.shape[0], -1)

    members = members.permute(1, 2, 0)
    target = target.reshape(target.shape[0], -1)
    if members.shape[1] == 0:
        raise ValueError('No grid points left for rank histogram after applying stride and mask')

    ranks_by_case = (members < target.unsqueeze(-1)).sum(dim=-1).numpy()
    ensemble_size = members.shape[-1]
    case_probabilities = np.stack(
        [_rank_histogram_probabilities(case_ranks, ensemble_size) for case_ranks in ranks_by_case],
        axis=0,
    )
    all_ranks = ranks_by_case.reshape(-1)
    all_probabilities = _rank_histogram_probabilities(all_ranks, ensemble_size)

    y_max = max(all_probabilities.max(), case_probabilities.max(initial=0.0))
    y_max = max(0.05, 1.1 * y_max)

    fig, ax = plt.subplots(figsize=(8, 4))
    _draw_rank_histogram(ax, all_probabilities, title)
    ax.set_ylim(0, y_max)
    plt.show()

    if per_case:
        n_cases = ranks_by_case.shape[0]
        cols = min(per_case_cols, n_cases)
        rows = int(np.ceil(n_cases / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), sharex=True, sharey=True)
        axes = np.atleast_1d(axes).flatten()

        for case_idx, ax in enumerate(axes):
            if case_idx < n_cases:
                case_label = case_idx if case_indices is None else int(case_indices[case_idx])
                _draw_rank_histogram(ax, case_probabilities[case_idx], f'case {case_label}')
                ax.set_ylim(0, y_max)
            else:
                ax.axis('off')

        fig.suptitle(f'{title} by case', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.show()

    return {
        'all_ranks': all_ranks,
        'ranks_by_case': ranks_by_case,
        'all_probabilities': all_probabilities,
        'case_probabilities': case_probabilities,
    }
