import glob
import json
import os
import platform
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
from diffusers.models.unets.unet_2d import UNet2DModel
from diffusers.training_utils import EMAModel

from concat.sampler import Sampler as ConcatSampler
from utils import (
    NpyImageDataset,
    channel_denormalize,
    channel_normalize,
    generate_satellite_track_mask,
    get_device,
)


IMAGE_SIZE = (320, 256)


def infer_repo_and_data_dirs() -> tuple[str, str]:
    repo_dir = os.getcwd()
    if platform.system() == "Darwin":
        data_dir = "/Users/amir/sciml/sea_ice_data"
    else:
        data_dir = "/mnt/sciml/a.sadreev/sea_ice_data"
    return repo_dir, data_dir


def load_stats(data_root: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
    with open(os.path.join(data_root, "train", "stats.json")) as f:
        stats = json.load(f)
    return tuple(stats["mean"]), tuple(stats["std"])


def load_valid_mask(data_root: str) -> np.ndarray:
    return np.load(os.path.join(data_root, "mask_padding.npy")).astype(np.float32)


def make_val_dataset(data_root: str, channel_mean, channel_std) -> NpyImageDataset:
    transform = partial(channel_normalize, channel_mean=channel_mean, channel_std=channel_std)
    return NpyImageDataset(
        folder=os.path.join(data_root, "valid"),
        transform=transform,
        preload=False,
        mmap_mode="r",
    )


def load_validation_case(
    data_root: str,
    channel_mean,
    channel_std,
    case_index: int,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    dataset = make_val_dataset(data_root, channel_mean, channel_std)
    if case_index < 0:
        case_index = len(dataset) + case_index
    clean = dataset[case_index].unsqueeze(0)
    if device is not None:
        clean = clean.to(device)
    return clean


def build_conditioning(
    clean: torch.Tensor,
    valid_mask: np.ndarray,
    image_size: tuple[int, int],
    n_tracks_range: tuple[int, int],
    device: str | torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = device or clean.device
    mask = torch.from_numpy(
        generate_satellite_track_mask(
            image_size=image_size,
            batch_size=1,
            valid_mask=valid_mask,
            n_tracks_range=n_tracks_range,
        )
    ).unsqueeze(1).to(device=device, dtype=clean.dtype)
    observed = clean.to(device) * mask
    return mask, observed


def _safe_torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def find_latest_run(base_dir: str, checkpoint_name: str) -> str:
    candidates = sorted(glob.glob(os.path.join(base_dir, "**", checkpoint_name), recursive=True))
    if not candidates:
        raise FileNotFoundError(f"No {checkpoint_name} found under {base_dir}")
    return str(Path(candidates[-1]).parent)


def load_concat_model(
    run_dir: str,
    image_size: tuple[int, int] = IMAGE_SIZE,
    checkpoint_name: str = "ema_best_model.pth",
    device: str | torch.device | None = None,
):
    device = device or get_device()
    model = UNet2DModel(
        sample_size=image_size,
        in_channels=7,
        out_channels=2,
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 512, 512),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )

    checkpoint_path = os.path.join(run_dir, checkpoint_name)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ema = EMAModel(model.parameters(), decay=0.999)
    ema.load_state_dict(_safe_torch_load(checkpoint_path))
    ema.copy_to(model.parameters())
    model.eval().to(device)
    return model, ConcatSampler(model)


def denormalize_batch(
    images: torch.Tensor,
    channel_mean,
    channel_std,
) -> torch.Tensor:
    return channel_denormalize(images.clone(), channel_mean, channel_std)


def compute_rmse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    diff = pred - target
    if mask is not None:
        diff = diff * mask
        denom = mask.sum().clamp(min=1.0)
        mse = diff.pow(2).sum() / denom
    else:
        mse = diff.pow(2).mean()
    return float(torch.sqrt(mse).item())


def compute_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    diff = (pred - target).abs()
    if mask is not None:
        diff = diff * mask
        denom = mask.sum().clamp(min=1.0)
        return float((diff.sum() / denom).item())
    return float(diff.mean().item())


def summarize_ensemble(
    ensemble: torch.Tensor,
    truth: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, float]:
    truth = truth.squeeze(0)
    mask_2ch = mask.squeeze(0).expand_as(truth)
    ensemble_mean = ensemble.mean(dim=0)
    ensemble_std = ensemble.std(dim=0, unbiased=False)
    member_rmse = torch.tensor([compute_rmse(member, truth) for member in ensemble], dtype=torch.float32)

    return {
        "ensemble_size": float(ensemble.shape[0]),
        "rmse_mean_all": compute_rmse(ensemble_mean, truth),
        "rmse_mean_masked": compute_rmse(ensemble_mean, truth, mask_2ch),
        "mae_mean_all": compute_mae(ensemble_mean, truth),
        "mae_mean_masked": compute_mae(ensemble_mean, truth, mask_2ch),
        "member_rmse_mean": float(member_rmse.mean().item()),
        "member_rmse_std": float(member_rmse.std(unbiased=False).item()),
        "ensemble_spread_mean": float(ensemble_std.mean().item()),
        "ensemble_spread_p95": float(torch.quantile(ensemble_std.flatten(), 0.95).item()),
    }


def summarize_single_sample(
    sample: torch.Tensor,
    truth: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, float]:
    truth = truth.squeeze(0)
    sample = sample.squeeze(0)
    mask_2ch = mask.squeeze(0).expand_as(truth)
    return {
        "rmse_all": compute_rmse(sample, truth),
        "rmse_masked": compute_rmse(sample, truth, mask_2ch),
        "mae_all": compute_mae(sample, truth),
        "mae_masked": compute_mae(sample, truth, mask_2ch),
    }


def sample_concat_ensemble(
    sampler: ConcatSampler,
    mask: torch.Tensor,
    observed: torch.Tensor,
    size: tuple[int, int],
    num_timesteps: int,
    ensemble_size: int,
    device: str | torch.device | None = None,
    method: str = "euler",
    rtol: float = 1e-3,
    atol: float = 1e-4,
    base_seed: int | None = None,
) -> torch.Tensor:
    device = device or observed.device
    members = []
    for idx in range(ensemble_size):
        if base_seed is not None:
            seed = base_seed + idx
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        sample = sampler.sample_conditioned(
            mask=mask,
            observed=observed,
            size=size,
            num_timesteps=num_timesteps,
            device=device,
            method=method,
            rtol=rtol,
            atol=atol,
        )
        members.append(sample[0].detach().cpu())

    return torch.stack(members, dim=0)


def run_concat_sampling_setup(
    sampler: ConcatSampler,
    mask: torch.Tensor,
    observed: torch.Tensor,
    truth: torch.Tensor,
    size: tuple[int, int],
    config: dict,
    device: str | torch.device | None = None,
) -> dict:
    seed = int(config.get("seed", 0))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    started = time.perf_counter()
    sample = sampler.sample_conditioned(
        mask=mask,
        observed=observed,
        size=size,
        num_timesteps=int(config["num_timesteps"]),
        device=device,
        method=str(config["method"]),
        rtol=float(config.get("rtol", 1e-3)),
        atol=float(config.get("atol", 1e-4)),
    ).detach().cpu()
    elapsed = time.perf_counter() - started

    metrics = summarize_single_sample(sample, truth.cpu(), mask.cpu())
    metrics["elapsed_sec"] = elapsed
    return {
        "name": str(config["name"]),
        "config": dict(config),
        "sample": sample,
        "metrics": metrics,
    }
