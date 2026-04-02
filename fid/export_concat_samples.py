import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import torch
from diffusers.models.unets.unet_2d import UNet2DModel
from diffusers.training_utils import EMAModel
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from concat.sampler import Sampler as ConcatSampler
from fid.inception_fid import default_data_root, default_stats_json_path, load_channel_stats
from utils import (
    NpyImageDataset,
    channel_denormalize,
    channel_normalize,
    generate_satellite_track_mask,
    get_device,
)


IMAGE_SIZE = (320, 256)


class NamedNpyDataset(Dataset):
    def __init__(self, folder: str, channel_mean, channel_std):
        self.base = NpyImageDataset(
            folder=folder,
            transform=lambda x: channel_normalize(x, channel_mean=channel_mean, channel_std=channel_std),
            preload=False,
            mmap_mode="r",
        )

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        return self.base[idx], self.base.files[idx]


def _safe_torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _default_concat_run_dir(repo_root: str, checkpoint_name: str) -> str:
    candidates = sorted(glob.glob(os.path.join(repo_root, "checkpoints", "**", checkpoint_name), recursive=True))
    if not candidates:
        raise FileNotFoundError(f"No {checkpoint_name} found under checkpoints/**")
    return str(Path(candidates[-1]).parent)


def load_concat_sampler(run_dir: str, checkpoint_name: str, device: str | torch.device):
    model = UNet2DModel(
        sample_size=IMAGE_SIZE,
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
    ema = EMAModel(model.parameters(), decay=0.999)
    ema.load_state_dict(_safe_torch_load(checkpoint_path))
    ema.copy_to(model.parameters())
    model.eval().to(device)
    return ConcatSampler(model)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a folder of .npy samples with the concat-conditioned model for later Inception-FID computation."
    )
    parser.add_argument("--output-dir", required=True, help="Where generated .npy files will be written")
    parser.add_argument("--repo-root", default=os.getcwd(), help="Repository root")
    parser.add_argument("--data-root", default=default_data_root(), help="Base sea-ice data root")
    parser.add_argument("--source-dir", help="Folder with source cases, defaults to <data-root>/valid")
    parser.add_argument("--stats-json", help="Path to stats.json for normalization")
    parser.add_argument("--mask-path", help="Path to mask_padding.npy")
    parser.add_argument("--concat-run-dir", help="Path to concat checkpoint directory")
    parser.add_argument("--checkpoint-name", default="ema_best_model.pth")
    parser.add_argument("--device", default=None, help="cuda, mps or cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of exported samples")
    parser.add_argument("--num-timesteps", type=int, default=50)
    parser.add_argument("--method", default="euler", help="ODE solver used by concat sampler")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--n-tracks-min", type=int, default=1)
    parser.add_argument("--n-tracks-max", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    source_dir = args.source_dir or os.path.join(args.data_root, "valid")
    stats_json = args.stats_json or default_stats_json_path(args.data_root)
    mask_path = args.mask_path or os.path.join(args.data_root, "mask_padding.npy")
    device = args.device or get_device()
    concat_run_dir = args.concat_run_dir or _default_concat_run_dir(args.repo_root, args.checkpoint_name)

    channel_mean, channel_std = load_channel_stats(stats_json)
    valid_mask = np.load(mask_path).astype(np.float32)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset = NamedNpyDataset(source_dir, channel_mean=channel_mean, channel_std=channel_std)
    sampler = load_concat_sampler(concat_run_dir, args.checkpoint_name, device)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    os.makedirs(args.output_dir, exist_ok=True)
    exported = 0
    progress = tqdm(loader, desc="export concat samples")
    for clean_batch, names in progress:
        if args.limit is not None and exported >= args.limit:
            break

        if args.limit is not None:
            keep = min(clean_batch.shape[0], args.limit - exported)
            clean_batch = clean_batch[:keep]
            names = list(names[:keep])
            if keep <= 0:
                break
        else:
            names = list(names)

        clean_batch = clean_batch.to(device)
        bs = clean_batch.shape[0]
        mask_np = generate_satellite_track_mask(
            image_size=IMAGE_SIZE,
            batch_size=bs,
            valid_mask=valid_mask,
            n_tracks_range=(args.n_tracks_min, args.n_tracks_max),
        )
        mask = torch.from_numpy(mask_np).unsqueeze(1).to(device=device, dtype=clean_batch.dtype)
        observed = clean_batch * mask

        samples = sampler.sample_conditioned(
            mask=mask,
            observed=observed,
            size=IMAGE_SIZE,
            num_timesteps=args.num_timesteps,
            device=device,
            method=args.method,
        )
        samples = channel_denormalize(samples.detach().cpu(), channel_mean, channel_std)

        for sample, name in zip(samples, names):
            path = os.path.join(args.output_dir, name)
            np.save(path, sample.numpy().astype(np.float32))
            exported += 1

    payload = {
        "output_dir": os.path.abspath(args.output_dir),
        "source_dir": os.path.abspath(source_dir),
        "concat_run_dir": os.path.abspath(concat_run_dir),
        "checkpoint_name": args.checkpoint_name,
        "num_exported": exported,
        "num_timesteps": args.num_timesteps,
        "method": args.method,
        "seed": args.seed,
        "n_tracks_range": [args.n_tracks_min, args.n_tracks_max],
        "stats_json": os.path.abspath(stats_json),
        "mask_path": os.path.abspath(mask_path),
    }
    with open(os.path.join(args.output_dir, "export_meta.json"), "w") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
