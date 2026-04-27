from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from functools import partial
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("MPLBACKEND", "Agg")

from utils import NpyImageDataset, channel_normalize

from .observations import ObservationConfig, make_observation_mask


DATE_RE = re.compile(r"(20\d{2}-\d{2}-\d{2})")


def default_data_root() -> str:
    if platform.system() == "Darwin":
        return "/Users/amir/sciml/sea_ice_data"
    return "/mnt/sciml/a.sadreev/sea_ice_data"


def load_channel_stats(stats_json_path: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
    with open(stats_json_path) as f:
        stats = json.load(f)
    return tuple(stats["mean"]), tuple(stats["std"])


@dataclass(frozen=True)
class ReferenceCase:
    index: int
    file: str
    date: str


def parse_file_date(name: str) -> date | None:
    match = DATE_RE.search(name)
    if match is None:
        return None
    return datetime.strptime(match.group(1), "%Y-%m-%d").date()


def first_file_per_day(files: list[str]) -> list[ReferenceCase]:
    by_day: dict[date, int] = {}
    for idx, name in enumerate(files):
        parsed = parse_file_date(name)
        if parsed is None:
            continue
        by_day.setdefault(parsed, idx)
    return [
        ReferenceCase(index=idx, file=files[idx], date=day.isoformat())
        for day, idx in sorted(by_day.items())
    ]


def select_reference_cases(files: list[str], args: argparse.Namespace) -> list[ReferenceCase]:
    daily = first_file_per_day(files)
    if daily:
        if args.selection == "december":
            rows = [case for case in daily if datetime.strptime(case.date, "%Y-%m-%d").date().month == 12]
            return rows[: args.n_days] if args.n_days is not None else rows

        if args.selection == "consecutive":
            start = datetime.strptime(args.start_date, "%Y-%m-%d").date() if args.start_date else datetime.strptime(daily[0].date, "%Y-%m-%d").date()
            want = {start + timedelta(days=offset) for offset in range(args.n_days)}
            by_date = {datetime.strptime(case.date, "%Y-%m-%d").date(): case for case in daily}
            return [by_date[day] for day in sorted(want) if day in by_date]

        if args.selection == "first-days":
            return daily[: args.n_days]

    count = args.n_days if args.n_days is not None else len(files)
    stride = max(1, int(args.daily_stride))
    indices = np.arange(0, len(files), stride, dtype=int)[:count]
    return [
        ReferenceCase(index=int(idx), file=files[int(idx)], date=f"index_{int(idx)}")
        for idx in indices
    ]


def masked_condition_mse(candidate_norm: torch.Tensor, observed_norm: torch.Tensor, mask_2d: np.ndarray, channel_weights: np.ndarray) -> float:
    mask = torch.as_tensor(mask_2d, dtype=candidate_norm.dtype).view(1, *mask_2d.shape)
    weights = torch.as_tensor(channel_weights, dtype=candidate_norm.dtype).view(-1, 1, 1)
    diff2 = torch.square((candidate_norm * mask - observed_norm) * weights)
    denom = (mask.sum() * weights.numel()).clamp(min=1.0)
    return float(diff2.sum().item() / denom.item())


def physical_mse(candidate_norm: torch.Tensor, reference_norm: torch.Tensor, channel_mean, channel_std, valid_mask: np.ndarray) -> dict[str, float]:
    mean = torch.as_tensor(channel_mean, dtype=candidate_norm.dtype).view(-1, 1, 1)
    std = torch.as_tensor(channel_std, dtype=candidate_norm.dtype).view(-1, 1, 1)
    candidate = (candidate_norm * std + mean).numpy()
    reference = (reference_norm * std + mean).numpy()
    mask = valid_mask.astype(bool)
    out = {"mse_all_channels": float(np.mean((candidate[:, mask] - reference[:, mask]) ** 2))}
    for ch in range(candidate.shape[0]):
        out[f"mse_ch{ch}"] = float(np.mean((candidate[ch, mask] - reference[ch, mask]) ** 2))
    return out


def write_csv(path: str | Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch kNN baseline for sparse sea-ice observations.")
    parser.add_argument("--data-root", default=default_data_root())
    parser.add_argument("--reference-dir", help="Defaults to <data-root>/valid")
    parser.add_argument("--neighbor-dir", help="Defaults to <data-root>/train")
    parser.add_argument("--stats-json", help="Defaults to <data-root>/train/stats.json")
    parser.add_argument("--mask-path", help="Defaults to <data-root>/mask_padding.npy")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--selection", choices=("december", "consecutive", "first-days"), default="december")
    parser.add_argument("--start-date", default=None, help="YYYY-MM-DD for --selection consecutive")
    parser.add_argument("--n-days", type=int, default=30, help="Use 31 for all December days if needed")
    parser.add_argument("--daily-stride", type=int, default=24, help="Fallback stride when dates are not parseable")
    parser.add_argument("--k-neighbors", type=int, default=1)
    parser.add_argument("--neighbor-limit", type=int, default=None, help="Smoke-test limit on train candidates")
    parser.add_argument("--include-same-file", action="store_true", help="Allow a neighbor with the same file name as the reference")
    parser.add_argument("--mask-type", choices=("random", "block", "swath"), default="swath")
    parser.add_argument("--density", type=float, default=0.05)
    parser.add_argument("--noise-level", type=float, default=0.0, help="Recorded for compatibility; kNN distance uses noiseless observed_norm")
    parser.add_argument("--seed", type=int, default=20260407)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--n-tracks-min", type=int, default=7)
    parser.add_argument("--n-tracks-max", type=int, default=7)
    parser.add_argument("--channel-weights", default="1.0,1.0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference_dir = args.reference_dir or os.path.join(args.data_root, "valid")
    neighbor_dir = args.neighbor_dir or os.path.join(args.data_root, "train")
    stats_json = args.stats_json or os.path.join(args.data_root, "train", "stats.json")
    mask_path = args.mask_path or os.path.join(args.data_root, "mask_padding.npy")
    channel_mean, channel_std = load_channel_stats(stats_json)
    valid_mask = np.load(mask_path).astype(np.float32)
    channel_weights = np.array([float(x) for x in args.channel_weights.split(",") if x.strip()], dtype=np.float32)

    transform = partial(channel_normalize, channel_mean=channel_mean, channel_std=channel_std)
    reference_dataset = NpyImageDataset(reference_dir, transform=transform, preload=False, mmap_mode="r")
    neighbor_dataset = NpyImageDataset(neighbor_dir, transform=transform, preload=False, mmap_mode="r")
    references = select_reference_cases(reference_dataset.files, args)
    if not references:
        raise ValueError("No reference cases selected")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_rows = [asdict(case) for case in references]
    write_csv(output_dir / "selected_references.csv", reference_rows)

    neighbor_count = len(neighbor_dataset) if args.neighbor_limit is None else min(len(neighbor_dataset), args.neighbor_limit)
    all_rows: list[dict[str, object]] = []
    image_size = tuple(valid_mask.shape)

    for ref_pos, ref_case in enumerate(tqdm(references, desc="reference days")):
        reference_norm = reference_dataset[ref_case.index]
        obs_config = ObservationConfig(
            mask_type=args.mask_type,
            density=args.density,
            noise_std=args.noise_level,
            seed=args.seed + ref_pos,
            block_size=args.block_size,
            n_tracks_range=(args.n_tracks_min, args.n_tracks_max),
        )
        rng = np.random.default_rng(args.seed + ref_pos)
        mask = make_observation_mask(image_size, obs_config, valid_mask, rng)
        observed_norm = reference_norm * torch.from_numpy(mask).to(dtype=reference_norm.dtype).view(1, *mask.shape)

        distances: list[tuple[float, int, str]] = []
        for idx in tqdm(range(neighbor_count), desc=f"kNN {ref_case.date}", leave=False):
            if not args.include_same_file and neighbor_dataset.files[idx] == ref_case.file:
                continue
            candidate = neighbor_dataset[idx]
            distance = masked_condition_mse(candidate, observed_norm, mask, channel_weights)
            distances.append((distance, idx, neighbor_dataset.files[idx]))

        distances.sort(key=lambda item: item[0])
        for rank, (distance, idx, name) in enumerate(distances[: args.k_neighbors]):
            candidate = neighbor_dataset[idx]
            row: dict[str, object] = {
                "reference_rank": ref_pos,
                "reference_index": ref_case.index,
                "reference_file": ref_case.file,
                "reference_date": ref_case.date,
                "neighbor_rank": rank,
                "neighbor_index": idx,
                "neighbor_file": name,
                "condition_mse_norm": distance,
                "mask_type": args.mask_type,
                "density": args.density,
                "observed_fraction": float((mask * valid_mask).sum() / max(1.0, valid_mask.sum())),
            }
            row.update(physical_mse(candidate, reference_norm, channel_mean, channel_std, valid_mask))
            all_rows.append(row)

    write_csv(output_dir / "knn_neighbors.csv", all_rows)
    summary = {
        "output_dir": str(output_dir.resolve()),
        "reference_dir": str(Path(reference_dir).resolve()),
        "neighbor_dir": str(Path(neighbor_dir).resolve()),
        "stats_json": str(Path(stats_json).resolve()),
        "mask_path": str(Path(mask_path).resolve()),
        "selection": args.selection,
        "start_date": args.start_date,
        "n_days_requested": args.n_days,
        "n_references": len(references),
        "k_neighbors": args.k_neighbors,
        "neighbor_count": neighbor_count,
        "include_same_file": args.include_same_file,
        "mask_type": args.mask_type,
        "density": args.density,
        "seed": args.seed,
        "channel_weights": channel_weights.tolist(),
        "outputs": ["selected_references.csv", "knn_neighbors.csv", "summary.json"],
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
