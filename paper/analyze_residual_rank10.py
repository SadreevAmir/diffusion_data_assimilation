"""Localize the one-sided rank-10 failure of a saved SIC ensemble.

This analysis is diagnostic only: it never changes or recalibrates members.
It reads server-side ``.npz`` case files produced by the residual-CFM sampler
and emits a compact JSON summary plus one visual review panel.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REGIMES = (
    ("exact_zero", lambda x: x == 0.0),
    ("trace_0_0.15", lambda x: (x > 0.0) & (x <= 0.15)),
    ("marginal_0.15_0.8", lambda x: (x > 0.15) & (x < 0.8)),
    ("high_0.8_0.95", lambda x: (x >= 0.8) & (x < 0.95)),
    ("near_one_0.95_1", lambda x: (x >= 0.95) & (x < 1.0)),
    ("exact_one", lambda x: x == 1.0),
)


def _row(mask: np.ndarray, rank10: np.ndarray, truth: np.ndarray,
         member_max: np.ndarray) -> dict[str, float | int | None]:
    count = int(mask.sum())
    hits = int((mask & rank10).sum())
    if count == 0:
        return {
            "pixels": 0,
            "rank10_pixels": 0,
            "rank10_rate": None,
            "mean_truth": None,
            "mean_member_max": None,
            "mean_positive_gap_when_rank10": None,
        }
    selected = mask & rank10
    return {
        "pixels": count,
        "rank10_pixels": hits,
        "rank10_rate": hits / count,
        "mean_truth": float(truth[mask].mean()),
        "mean_member_max": float(member_max[mask].mean()),
        "mean_positive_gap_when_rank10": (
            float((truth[selected] - member_max[selected]).mean()) if hits else None
        ),
    }


def analyze(input_dir: Path, output_dir: Path) -> dict:
    paths = sorted(input_dir.glob("*.npz"))
    if len(paths) != 40:
        raise ValueError(f"expected exactly 40 cases, found {len(paths)}")

    cases = []
    pooled_truth = []
    pooled_max = []
    pooled_rank10 = []
    pooled_track = []
    rank10_count = None
    valid_count = None

    for path in paths:
        with np.load(path, allow_pickle=False) as payload:
            ensemble = np.asarray(payload["analysis_ensemble"][:, 0], dtype=np.float64)
            truth = np.asarray(payload["truth"][0], dtype=np.float64)
            valid = np.asarray(payload["valid_mask"][0], dtype=bool)
            track = np.asarray(payload["obs_mask"][0], dtype=bool) & valid
        if ensemble.shape != (10, *truth.shape) or valid.shape != truth.shape:
            raise ValueError(f"unexpected case shape in {path.name}")
        if not np.isfinite(ensemble[:, valid]).all() or not np.isfinite(truth[valid]).all():
            raise ValueError(f"non-finite valid value in {path.name}")
        member_max = ensemble.max(axis=0)
        member_min = ensemble.min(axis=0)
        rank10 = valid & (truth > member_max)
        rank0 = valid & (truth < member_min)
        cases.append({
            "case": path.stem,
            "valid_pixels": int(valid.sum()),
            "rank10_rate": float(rank10.sum() / valid.sum()),
            "rank0_rate": float(rank0.sum() / valid.sum()),
            "rank10_track_rate": (
                float(rank10[track].mean()) if np.any(track) else None
            ),
            "rank10_off_track_rate": float(rank10[valid & ~track].mean()),
            "mean_bias": float((ensemble.mean(axis=0)[valid] - truth[valid]).mean()),
        })
        pooled_truth.append(truth[valid])
        pooled_max.append(member_max[valid])
        pooled_rank10.append(rank10[valid])
        pooled_track.append(track[valid])
        if rank10_count is None:
            rank10_count = np.zeros_like(truth, dtype=np.int64)
            valid_count = np.zeros_like(truth, dtype=np.int64)
        rank10_count += rank10
        valid_count += valid

    truth = np.concatenate(pooled_truth)
    member_max = np.concatenate(pooled_max)
    rank10 = np.concatenate(pooled_rank10)
    track = np.concatenate(pooled_track)
    valid = np.ones_like(rank10, dtype=bool)
    regimes = {
        name: _row(selector(truth), rank10, truth, member_max)
        for name, selector in REGIMES
    }
    for row in regimes.values():
        row["share_of_all_rank10"] = (
            row["rank10_pixels"] / int(rank10.sum()) if np.any(rank10) else None
        )
    summary = {
        "cases": len(paths),
        "ensemble_size": 10,
        "valid_pixels": int(valid.sum()),
        "rank10_pixels": int(rank10.sum()),
        "rank10_rate": float(rank10.mean()),
        "rank0_rate": float(np.mean([
            case["rank0_rate"] for case in cases
        ])),
        "conditional_non_rank10_bin_expectation": float((1.0 - rank10.mean()) / 10.0),
        "track": _row(track, rank10, truth, member_max),
        "off_track": _row(~track, rank10, truth, member_max),
        "regimes": regimes,
        "per_case": cases,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rank10_localization.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    assert rank10_count is not None and valid_count is not None
    frequency = np.divide(
        rank10_count,
        valid_count,
        out=np.full_like(rank10_count, np.nan, dtype=np.float64),
        where=valid_count > 0,
    )
    figure, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    image = axes[0, 0].imshow(frequency, vmin=0.0, vmax=1.0, cmap="magma")
    axes[0, 0].set_title("Frequency of truth > all 10 members")
    axes[0, 0].axis("off")
    figure.colorbar(image, ax=axes[0, 0], fraction=0.046)

    bins = np.linspace(0.0, 1.0, 41)
    axes[0, 1].hist(truth, bins=bins, density=True, alpha=0.45, label="all valid")
    axes[0, 1].hist(truth[rank10], bins=bins, density=True, alpha=0.65, label="rank 10")
    axes[0, 1].set_xlabel("Truth SIC")
    axes[0, 1].set_ylabel("Density")
    axes[0, 1].legend()

    case_rates = [row["rank10_rate"] for row in cases]
    axes[1, 0].plot(case_rates, marker="o", markersize=3)
    axes[1, 0].axhline(float(rank10.mean()), color="black", linestyle="--")
    axes[1, 0].set_xlabel("Validation case order")
    axes[1, 0].set_ylabel("Rank-10 rate")
    axes[1, 0].set_ylim(0.0, 1.0)

    names = [name for name, _ in REGIMES]
    rates = [regimes[name]["rank10_rate"] or 0.0 for name in names]
    axes[1, 1].bar(np.arange(len(names)), rates)
    axes[1, 1].set_xticks(np.arange(len(names)), names, rotation=35, ha="right")
    axes[1, 1].set_ylabel("P(rank 10 | truth regime)")
    axes[1, 1].set_ylim(0.0, 1.0)
    figure.suptitle("Residual-CFM one-sided rank failure localization")
    figure.savefig(output_dir / "rank10_localization.png", dpi=180)
    plt.close(figure)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.input_dir, args.output_dir), sort_keys=True))


if __name__ == "__main__":
    main()
