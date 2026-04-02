import glob
import json
import os
import platform
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


def default_data_root() -> str:
    if platform.system() == "Darwin":
        return "/Users/amir/sciml/sea_ice_data"
    return "/mnt/sciml/a.sadreev/sea_ice_data"


def default_stats_json_path(data_root: str | None = None) -> str:
    data_root = data_root or default_data_root()
    return os.path.join(data_root, "train", "stats.json")


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_channel_stats(stats_json_path: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
    with open(stats_json_path) as f:
        stats = json.load(f)
    return tuple(stats["mean"]), tuple(stats["std"])


def _safe_torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _to_channel_first(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr[None, ...].astype(np.float32, copy=False)
    if arr.ndim != 3:
        raise ValueError(f"Unsupported array shape {arr.shape}")

    if arr.shape[0] in (1, 2, 3):
        out = arr
    elif arr.shape[-1] in (1, 2, 3):
        out = np.moveaxis(arr, -1, 0)
    else:
        raise ValueError(f"Unsupported array shape {arr.shape}")
    return out.astype(np.float32, copy=False)


def _matrix_sqrt_psd(mat: np.ndarray) -> np.ndarray:
    mat = 0.5 * (mat + mat.T)
    eigvals, eigvecs = np.linalg.eigh(mat)
    eigvals = np.clip(eigvals, a_min=0.0, a_max=None)
    return (eigvecs * np.sqrt(eigvals)) @ eigvecs.T


def _import_matplotlib_cm():
    try:
        from matplotlib import cm
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for RGB rendering in fid/inception_fid.py. "
            "Install it in the environment where you run FID."
        ) from exc
    return cm


def _import_torchvision_inception():
    try:
        from torchvision.models import Inception_V3_Weights, inception_v3
    except ImportError as exc:
        raise ImportError(
            "torchvision is required for Inception-FID. "
            "Install torchvision in the environment where you run FID."
        ) from exc
    return inception_v3, Inception_V3_Weights


@dataclass
class FIDStats:
    mu: np.ndarray
    sigma: np.ndarray
    count: int
    feature_dim: int


@dataclass
class RenderConfig:
    mode: str = "channel0"
    channel0_cmap: str = "viridis"
    channel1_cmap: str = "magma"
    channel0_vmin: float = 0.0
    channel0_vmax: float = 1.0
    channel1_vmin: float = 0.0
    channel1_vmax: float = 1.0
    blend_alpha: float = 0.5
    input_normalized: bool = False
    channel_mean: tuple[float, ...] | None = None
    channel_std: tuple[float, ...] | None = None


class NpyRGBRenderDataset(Dataset):
    def __init__(
        self,
        folder: str,
        render_config: RenderConfig,
        max_items: int | None = None,
        mmap_mode: str | None = "r",
    ):
        self.folder = folder
        self.render_config = render_config
        self.mmap_mode = mmap_mode
        self.files = sorted([f for f in os.listdir(folder) if f.endswith(".npy")])
        if max_items is not None:
            self.files = self.files[:max_items]
        if not self.files:
            raise ValueError(f"No .npy files found in {folder}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = os.path.join(self.folder, self.files[idx])
        arr = np.load(path, mmap_mode=self.mmap_mode) if self.mmap_mode else np.load(path)
        field = _to_channel_first(np.asarray(arr, dtype=np.float32))
        rgb = render_field_to_rgb(field, self.render_config)
        return torch.from_numpy(rgb)


class InceptionFeatureExtractor(nn.Module):
    def __init__(
        self,
        device: str | torch.device | None = None,
        resize_to: int = 299,
        weights_path: str | None = None,
    ):
        super().__init__()
        inception_v3, Inception_V3_Weights = _import_torchvision_inception()

        self.device_name = str(device or default_device())
        self.resize_to = int(resize_to)
        if self.resize_to <= 0:
            raise ValueError(f"resize_to must be positive, got {self.resize_to}")

        if weights_path:
            model = inception_v3(weights=None, aux_logits=False, transform_input=False)
            model.load_state_dict(_safe_torch_load(weights_path))
        else:
            model = inception_v3(
                weights=Inception_V3_Weights.DEFAULT,
                aux_logits=False,
                transform_input=False,
            )

        model.fc = nn.Identity()
        model.eval().to(self.device_name)
        self.model = model
        self.register_buffer(
            "image_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device_name, dtype=torch.float32)
        if x.shape[-2:] != (self.resize_to, self.resize_to):
            x = F.interpolate(
                x,
                size=(self.resize_to, self.resize_to),
                mode="bilinear",
                align_corners=False,
            )
        x = (x - self.image_mean) / self.image_std
        features = self.model(x)
        return features.flatten(start_dim=1)


def _denormalize_field(field: np.ndarray, render_config: RenderConfig) -> np.ndarray:
    if not render_config.input_normalized:
        return field
    if render_config.channel_mean is None or render_config.channel_std is None:
        raise ValueError("channel_mean/channel_std are required when input_normalized=True")

    mean = np.asarray(render_config.channel_mean, dtype=np.float32)[:, None, None]
    std = np.asarray(render_config.channel_std, dtype=np.float32)[:, None, None]
    field = field * std + mean
    field = field.copy()
    if field.shape[0] >= 1:
        field[0] = np.clip(field[0], 0.0, 1.0)
    if field.shape[0] >= 2:
        field[1] = np.clip(field[1], 0.0, None)
    return field


def _render_channel(channel: np.ndarray, cmap_name: str, vmin: float, vmax: float) -> np.ndarray:
    if vmax <= vmin:
        raise ValueError(f"Expected vmax > vmin, got {vmax} <= {vmin}")

    cm = _import_matplotlib_cm()
    cmap = cm.get_cmap(cmap_name)
    normed = np.clip((channel - vmin) / (vmax - vmin), 0.0, 1.0)
    rgba = cmap(normed)
    return rgba[..., :3].astype(np.float32, copy=False)


def render_field_to_rgb(field: np.ndarray, render_config: RenderConfig) -> np.ndarray:
    field = _denormalize_field(field, render_config)

    if render_config.mode == "channel0":
        rgb = _render_channel(
            field[0],
            render_config.channel0_cmap,
            render_config.channel0_vmin,
            render_config.channel0_vmax,
        )
    elif render_config.mode == "channel1":
        if field.shape[0] < 2:
            raise ValueError("mode='channel1' requires at least 2 channels")
        rgb = _render_channel(
            field[1],
            render_config.channel1_cmap,
            render_config.channel1_vmin,
            render_config.channel1_vmax,
        )
    elif render_config.mode == "blend":
        if field.shape[0] < 2:
            raise ValueError("mode='blend' requires at least 2 channels")
        rgb0 = _render_channel(
            field[0],
            render_config.channel0_cmap,
            render_config.channel0_vmin,
            render_config.channel0_vmax,
        )
        rgb1 = _render_channel(
            field[1],
            render_config.channel1_cmap,
            render_config.channel1_vmin,
            render_config.channel1_vmax,
        )
        alpha = float(np.clip(render_config.blend_alpha, 0.0, 1.0))
        rgb = alpha * rgb0 + (1.0 - alpha) * rgb1
        rgb = np.clip(rgb, 0.0, 1.0)
    else:
        raise ValueError(f"Unsupported render mode {render_config.mode!r}")

    return np.moveaxis(rgb, -1, 0).astype(np.float32, copy=False)


def compute_feature_stats(features: np.ndarray) -> FIDStats:
    if features.ndim != 2:
        raise ValueError(f"Expected 2D feature array, got shape {features.shape}")
    if features.shape[0] == 0:
        raise ValueError("Cannot compute FID stats for an empty feature set")

    mu = features.mean(axis=0)
    if features.shape[0] == 1:
        sigma = np.zeros((features.shape[1], features.shape[1]), dtype=np.float64)
    else:
        sigma = np.cov(features, rowvar=False)
    return FIDStats(
        mu=mu.astype(np.float64, copy=False),
        sigma=sigma.astype(np.float64, copy=False),
        count=int(features.shape[0]),
        feature_dim=int(features.shape[1]),
    )


def compute_feature_stats_for_folder(
    folder: str,
    extractor: InceptionFeatureExtractor,
    render_config: RenderConfig,
    batch_size: int = 16,
    num_workers: int = 0,
    max_items: int | None = None,
    desc: str | None = None,
) -> FIDStats:
    dataset = NpyRGBRenderDataset(
        folder=folder,
        render_config=render_config,
        max_items=max_items,
        mmap_mode="r",
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available() and extractor.device_name.startswith("cuda"),
    )

    feature_chunks: list[np.ndarray] = []
    progress = tqdm(loader, desc=desc or f"features:{Path(folder).name}")
    for batch in progress:
        features = extractor(batch).detach().cpu().numpy().astype(np.float64, copy=False)
        feature_chunks.append(features)

    if not feature_chunks:
        raise ValueError(f"No features extracted from folder {folder}")

    return compute_feature_stats(np.concatenate(feature_chunks, axis=0))


def compute_frechet_distance(
    real_stats: FIDStats,
    fake_stats: FIDStats,
    eps: float = 1e-6,
) -> float:
    mu1 = real_stats.mu.astype(np.float64, copy=False)
    mu2 = fake_stats.mu.astype(np.float64, copy=False)
    sigma1 = real_stats.sigma.astype(np.float64, copy=False)
    sigma2 = fake_stats.sigma.astype(np.float64, copy=False)

    eye = np.eye(sigma1.shape[0], dtype=np.float64)
    sigma1 = sigma1 + eps * eye
    sigma2 = sigma2 + eps * eye

    diff = mu1 - mu2
    sqrt_sigma1 = _matrix_sqrt_psd(sigma1)
    middle = sqrt_sigma1 @ sigma2 @ sqrt_sigma1
    middle = 0.5 * (middle + middle.T)
    covmean = _matrix_sqrt_psd(middle)

    score = diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean)
    return float(max(score, 0.0))


def save_stats(path: str, stats: FIDStats, meta: dict | None = None) -> None:
    meta = meta or {}
    np.savez_compressed(
        path,
        mu=stats.mu,
        sigma=stats.sigma,
        count=np.array(stats.count, dtype=np.int64),
        feature_dim=np.array(stats.feature_dim, dtype=np.int64),
        meta_json=np.array(json.dumps(meta), dtype=np.str_),
    )


def load_saved_stats(path: str) -> tuple[FIDStats, dict]:
    data = np.load(path, allow_pickle=False)
    meta = json.loads(str(data["meta_json"])) if "meta_json" in data else {}
    stats = FIDStats(
        mu=data["mu"].astype(np.float64, copy=False),
        sigma=data["sigma"].astype(np.float64, copy=False),
        count=int(data["count"]),
        feature_dim=int(data["feature_dim"]),
    )
    return stats, meta


def default_inception_weights_path(repo_root: str | None = None) -> str | None:
    repo_root = repo_root or os.getcwd()
    candidates = sorted(
        glob.glob(os.path.join(repo_root, "**", "inception_v3*.pth"), recursive=True)
    )
    return candidates[-1] if candidates else None
