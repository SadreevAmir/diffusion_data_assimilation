from __future__ import annotations

import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
from tqdm.auto import tqdm

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lorenz_baseline.common import Lorenz63Config, Standardizer, rk4_step_numpy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Lorenz-63 train/val/test splits.")
    parser.add_argument("--output-dir", default="lorenz_baseline/data", help="Output directory")
    parser.add_argument("--train-size", type=int, default=200_000)
    parser.add_argument("--val-size", type=int, default=20_000)
    parser.add_argument("--test-size", type=int, default=20_000)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--burn-in", type=int, default=1000)
    parser.add_argument("--thin", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--chunk-size", type=int, default=25_000)
    return parser.parse_args()


def sample_initial_states(num_states: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-20.0, 20.0, size=(num_states, 1))
    y = rng.uniform(-30.0, 30.0, size=(num_states, 1))
    z = rng.uniform(0.0, 50.0, size=(num_states, 1))
    return np.concatenate([x, y, z], axis=1).astype(np.float64)


def generate_chunk(target_count: int, seed: int, config: Lorenz63Config) -> np.ndarray:
    if target_count <= 0:
        return np.empty((0, 3), dtype=np.float32)

    traj_batch = min(max(512, target_count // 8), 4096, target_count)
    state = sample_initial_states(traj_batch, seed=seed)
    rng = np.random.default_rng(seed + 1)

    for _ in range(config.burn_in_steps):
        state = rk4_step_numpy(
            state,
            dt=config.dt,
            sigma=config.sigma,
            rho=config.rho,
            beta=config.beta,
        )

    chunks: list[np.ndarray] = []
    remaining = target_count
    while remaining > 0:
        for _ in range(config.thin):
            state = rk4_step_numpy(
                state,
                dt=config.dt,
                sigma=config.sigma,
                rho=config.rho,
                beta=config.beta,
            )
        take = min(remaining, state.shape[0])
        indices = rng.permutation(state.shape[0])[:take]
        chunks.append(state[indices].astype(np.float32).copy())
        remaining -= take

    return np.concatenate(chunks, axis=0)[:target_count]


def split_into_chunks(total_size: int, chunk_size: int) -> list[int]:
    num_chunks = math.ceil(total_size / chunk_size)
    return [
        min(chunk_size, total_size - idx * chunk_size)
        for idx in range(num_chunks)
        if total_size - idx * chunk_size > 0
    ]


def generate_split(
    split_name: str,
    total_size: int,
    chunk_size: int,
    workers: int,
    seed: int,
    config: Lorenz63Config,
) -> np.ndarray:
    chunk_sizes = split_into_chunks(total_size, chunk_size)
    states: list[np.ndarray] = []
    progress = tqdm(total=total_size, desc=f"generate {split_name}")

    try:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(generate_chunk, size, seed + idx * 9973, config): size
                for idx, size in enumerate(chunk_sizes)
            }
            for future in as_completed(futures):
                chunk = future.result()
                states.append(chunk)
                progress.update(len(chunk))
    except (PermissionError, OSError) as exc:
        print(
            f"process pool unavailable for {split_name} ({exc}); "
            "falling back to sequential chunk generation"
        )
        for idx, size in enumerate(chunk_sizes):
            chunk = generate_chunk(size, seed + idx * 9973, config)
            states.append(chunk)
            progress.update(len(chunk))
    finally:
        progress.close()

    merged = np.concatenate(states, axis=0).astype(np.float32)
    rng = np.random.default_rng(seed)
    rng.shuffle(merged, axis=0)
    return merged[:total_size]


def save_split(output_dir: Path, split: str, states: np.ndarray) -> None:
    np.savez_compressed(output_dir / f"{split}.npz", states=states.astype(np.float32))


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = Lorenz63Config(
        dt=args.dt,
        burn_in_steps=args.burn_in,
        thin=args.thin,
    )

    split_sizes = {
        "train": args.train_size,
        "val": args.val_size,
        "test": args.test_size,
    }

    generated: dict[str, np.ndarray] = {}
    for offset, (split, size) in enumerate(split_sizes.items()):
        generated[split] = generate_split(
            split_name=split,
            total_size=size,
            chunk_size=args.chunk_size,
            workers=args.workers,
            seed=args.seed + offset * 100_000,
            config=config,
        )
        save_split(output_dir, split, generated[split])

    standardizer = Standardizer.from_states(generated["train"])
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "system": "Lorenz-63",
        "parameters": {
            "sigma": config.sigma,
            "rho": config.rho,
            "beta": config.beta,
        },
        "integration": {
            "dt": config.dt,
            "burn_in_steps": config.burn_in_steps,
            "thin": config.thin,
        },
        "split_sizes": split_sizes,
        "seed": args.seed,
        "workers": args.workers,
        "chunk_size": args.chunk_size,
        "normalization": {
            "mean": standardizer.mean.tolist(),
            "std": standardizer.std.tolist(),
        },
    }

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
