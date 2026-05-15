from __future__ import annotations

import argparse
import csv
import json
import os
import platform
from datetime import datetime

import numpy as np
from tqdm.auto import tqdm

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("MPLBACKEND", "Agg")

from utils import NpyImageDataset

from .metrics import (
    binned_spread_skill,
    energy_score,
    ensemble_crps,
    ice_summaries,
    interval_coverage,
    rank_histogram,
    spread_skill,
    weighted_mean,
)
from .observations import ObservationConfig, make_observation_mask, make_sparse_observation
from .runners import ConcatRunner, DummyRunner, EnsembleRunner, PersistenceRunner


CHANNEL_NAMES = ("concentration", "thickness")
LEVELS = (0.5, 0.8, 0.9, 0.95)


def default_data_root() -> str:
    if platform.system() == "Darwin":
        return "/Users/amir/sciml/sea_ice_data"
    return "/mnt/sciml/a.sadreev/sea_ice_data"


def default_stats_json_path(data_root: str | None = None) -> str:
    data_root = data_root or default_data_root()
    return os.path.join(data_root, "train", "stats.json")


def load_channel_stats(stats_json_path: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
    with open(stats_json_path) as f:
        stats = json.load(f)
    return tuple(stats["mean"]), tuple(stats["std"])


def parse_float_list(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthetic ensemble benchmark for sea-ice data assimilation.")
    parser.add_argument("--input-dir", help="Folder with full x_true .npy fields. Defaults to <data-root>/valid")
    parser.add_argument("--data-root", default=default_data_root(), help="Sea-ice data root")
    parser.add_argument("--output-dir", required=True, help="Reproducible output directory")
    parser.add_argument("--stats-json", help="stats.json path; used by concat runner")
    parser.add_argument("--mask-path", help="mask_padding.npy path")
    parser.add_argument("--variables", default="concentration,thickness", help="Comma-separated variable names")
    parser.add_argument("--date-range", default="", help="Recorded in metadata; file-based datasets currently use sorted .npy order")
    parser.add_argument("--ensemble-size", type=int, default=8)
    parser.add_argument("--n-cases", type=int, default=None, help="Number of ground-truth cases to evaluate")
    parser.add_argument(
        "--case-selection",
        choices=("first", "linspace", "daily"),
        default="first",
        help="How to select ground-truth cases when --n-cases or --max-timesteps is set",
    )
    parser.add_argument("--daily-stride", type=int, default=24, help="File stride for --case-selection daily on hourly data")
    parser.add_argument("--densities", default="0.01,0.05,0.10,0.25", help="Comma-separated observation densities")
    parser.add_argument("--mask-types", default="random,block", help="Comma-separated mask types: random,block,swath")
    parser.add_argument("--swath-repeats", type=int, default=1, help="Independent swath masks per case for swath conditioning")
    parser.add_argument("--n-tracks-min", type=int, default=1, help="Minimum generated tracks for swath masks")
    parser.add_argument("--n-tracks-max", type=int, default=4, help="Maximum generated tracks for swath masks")
    parser.add_argument("--noise-levels", default="0.0", help="Comma-separated observation noise std values")
    parser.add_argument(
        "--eval-region",
        choices=("all", "observed", "unobserved"),
        default="all",
        help="Spatial region used for metrics/rank histograms",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-timesteps", type=int, default=None, help="Smoke-test limit")
    parser.add_argument("--runner", choices=("dummy", "persistence", "concat"), default="dummy")
    parser.add_argument("--dummy-noise-std", type=float, default=0.05)
    parser.add_argument("--persistence-perturbation-std", type=float, default=0.02)
    parser.add_argument("--concat-run-dir", help="Concat checkpoint directory")
    parser.add_argument("--checkpoint-name", default="ema_best_model.pth")
    parser.add_argument("--num-timesteps", type=int, default=50, help="Concat sampler ODE steps")
    parser.add_argument("--method", default="euler", help="Concat sampler ODE method")
    parser.add_argument("--device", default=None)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--rank-stride", type=int, default=4, help="Subsample grid for rank histograms")
    parser.add_argument("--energy-score-limit", type=int, default=20000, help="Max flattened points for energy score; <=0 disables")
    parser.add_argument("--save-tensors", action="store_true", help="Save per-case ensemble/truth/mask/observed tensors")
    parser.add_argument(
        "--save-tensor-limit",
        type=int,
        default=None,
        help="Maximum number of local cases for tensor saving; default saves all selected cases",
    )
    parser.add_argument(
        "--save-tensor-format",
        choices=("npz", "npz_compressed"),
        default="npz",
        help="Tensor archive format. npz is faster; npz_compressed uses less disk and more CPU.",
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def select_case_indices(dataset_len: int, args: argparse.Namespace) -> np.ndarray:
    requested = args.n_cases if args.n_cases is not None else args.max_timesteps
    if requested is None:
        return np.arange(dataset_len, dtype=int)
    requested = min(int(requested), dataset_len)
    if requested <= 0:
        return np.empty(0, dtype=int)

    if args.case_selection == "first":
        return np.arange(requested, dtype=int)
    if args.case_selection == "linspace":
        return np.linspace(0, dataset_len - 1, requested, dtype=int)
    if args.case_selection == "daily":
        stride = max(1, int(args.daily_stride))
        indices = np.arange(0, dataset_len, stride, dtype=int)
        if indices.size >= requested:
            return indices[:requested]
        return np.linspace(0, dataset_len - 1, requested, dtype=int)
    raise ValueError(f"Unknown case_selection: {args.case_selection}")


def load_runner(args: argparse.Namespace, channel_mean, channel_std) -> EnsembleRunner:
    if args.runner == "dummy":
        return DummyRunner(noise_std=args.dummy_noise_std)
    if args.runner == "persistence":
        return PersistenceRunner(perturbation_std=args.persistence_perturbation_std)
    from fid.export_concat_samples import _default_concat_run_dir

    concat_run_dir = args.concat_run_dir or _default_concat_run_dir(os.getcwd(), args.checkpoint_name)
    return ConcatRunner(
        run_dir=concat_run_dir,
        checkpoint_name=args.checkpoint_name,
        channel_mean=channel_mean,
        channel_std=channel_std,
        num_timesteps=args.num_timesteps,
        method=args.method,
        device=args.device,
    )


def season_from_name(name: str, case_index: int) -> str:
    digits = "".join(ch if ch.isdigit() else " " for ch in name).split()
    for token in digits:
        if len(token) >= 6:
            month = int(token[4:6])
            return ("DJF", "MAM", "JJA", "SON")[(month % 12) // 3]
    return f"case_block_{case_index // 24:04d}"


def ice_regime(truth: np.ndarray, valid_mask: np.ndarray | None) -> str:
    if truth.shape[0] == 0:
        return "unknown"
    mask = np.ones(truth.shape[-2:], dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    mean_conc = float(np.mean(truth[0][mask])) if np.any(mask) else 0.0
    if mean_conc < 0.15:
        return "open_water"
    if mean_conc < 0.80:
        return "marginal_ice_zone"
    return "compact_ice"


def write_csv(path: str, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def tag_from_condition(mask_type: str, density: float, noise: float, condition_index: int = 0, n_conditions: int = 1) -> str:
    repeat = f"_r{condition_index:02d}" if n_conditions > 1 else ""
    return f"{mask_type}{repeat}_d{density:g}_n{noise:g}".replace(".", "p")


def build_condition_specs(
    mask_types: list[str], densities: list[float], noise_levels: list[float], args: argparse.Namespace
) -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    for mask_type in mask_types:
        repeats = max(1, int(args.swath_repeats)) if mask_type == "swath" else 1
        for density in densities:
            for noise in noise_levels:
                for condition_index in range(repeats):
                    specs.append(
                        {
                            "mask_type": mask_type,
                            "density": density,
                            "noise": noise,
                            "condition_index": condition_index,
                            "n_conditions": repeats,
                        }
                    )
    return specs


def evaluation_mask(valid: np.ndarray, observed_mask: np.ndarray, eval_region: str) -> np.ndarray:
    valid_bool = np.asarray(valid, dtype=np.float32) > 0
    observed_bool = np.asarray(observed_mask, dtype=np.float32) > 0
    if eval_region == "all":
        return valid_bool.astype(np.float32)
    if eval_region == "observed":
        return (valid_bool & observed_bool).astype(np.float32)
    if eval_region == "unobserved":
        return (valid_bool & ~observed_bool).astype(np.float32)
    raise ValueError(f"Unknown eval_region: {eval_region}")


def should_save_tensors(args: argparse.Namespace, local_case_index: int) -> bool:
    if not args.save_tensors:
        return False
    if args.save_tensor_limit is None:
        return True
    return local_case_index < int(args.save_tensor_limit)


def save_case_tensors(
    samples_dir: str,
    args: argparse.Namespace,
    local_case_index: int,
    case_index: int,
    case_name: str,
    condition_tag: str,
    combo_seed: int,
    x_true: np.ndarray,
    mask: np.ndarray,
    observed: np.ndarray,
    ensemble: np.ndarray,
) -> str:
    condition_dir = os.path.join(samples_dir, condition_tag)
    os.makedirs(condition_dir, exist_ok=True)
    safe_name = os.path.basename(case_name).replace(os.sep, "_")
    out_path = os.path.join(condition_dir, f"case{local_case_index:04d}_idx{case_index}_{safe_name}.npz")
    payload = {
        "ensemble": ensemble.astype(np.float32, copy=False),
        "truth": x_true.astype(np.float32, copy=False),
        "mask": mask.astype(np.float32, copy=False),
        "observed": observed.astype(np.float32, copy=False),
        "local_case_index": np.array(local_case_index, dtype=np.int64),
        "case_index": np.array(case_index, dtype=np.int64),
        "case_name": np.array(case_name),
        "condition_tag": np.array(condition_tag),
        "combo_seed": np.array(combo_seed, dtype=np.int64),
    }
    if args.save_tensor_format == "npz_compressed":
        np.savez_compressed(out_path, **payload)
    else:
        np.savez(out_path, **payload)
    return out_path


def main() -> None:
    args = parse_args()
    if not args.no_plots:
        from .plotting import plot_coverage, plot_density_curve, plot_example_panel, plot_rank_histogram, plot_spread_skill

    input_dir = args.input_dir or os.path.join(args.data_root, "valid")
    stats_json = args.stats_json or default_stats_json_path(args.data_root)
    mask_path = args.mask_path or os.path.join(args.data_root, "mask_padding.npy")
    channel_mean, channel_std = load_channel_stats(stats_json)
    valid_mask = np.load(mask_path).astype(np.float32) if os.path.exists(mask_path) else None
    variables = parse_str_list(args.variables)
    densities = parse_float_list(args.densities)
    mask_types = parse_str_list(args.mask_types)
    noise_levels = parse_float_list(args.noise_levels)

    os.makedirs(args.output_dir, exist_ok=True)
    plots_dir = os.path.join(args.output_dir, "plots")
    arrays_dir = os.path.join(args.output_dir, "arrays")
    samples_dir = os.path.join(args.output_dir, "samples")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(arrays_dir, exist_ok=True)
    if args.save_tensors:
        os.makedirs(samples_dir, exist_ok=True)

    dataset = NpyImageDataset(input_dir, preload=False, mmap_mode="r")
    runner = load_runner(args, channel_mean, channel_std)
    case_indices = select_case_indices(len(dataset), args)
    n_cases = int(case_indices.size)

    meta = {
        "created_at": datetime.now().isoformat(),
        "input_dir": os.path.abspath(input_dir),
        "output_dir": os.path.abspath(args.output_dir),
        "stats_json": os.path.abspath(stats_json),
        "mask_path": os.path.abspath(mask_path) if os.path.exists(mask_path) else None,
        "variables": variables,
        "date_range": args.date_range,
        "ensemble_size": args.ensemble_size,
        "densities": densities,
        "mask_types": mask_types,
        "swath_repeats": args.swath_repeats,
        "n_tracks_range": [args.n_tracks_min, args.n_tracks_max],
        "noise_levels": noise_levels,
        "eval_region": args.eval_region,
        "seed": args.seed,
        "n_cases": args.n_cases,
        "case_selection": args.case_selection,
        "daily_stride": args.daily_stride,
        "selected_case_indices": case_indices.tolist(),
        "max_timesteps": args.max_timesteps,
        "runner": args.runner,
        "runner_checkpoint_metadata": getattr(runner, "checkpoint_metadata", None),
        "rank_stride": args.rank_stride,
        "save_tensors": args.save_tensors,
        "save_tensor_limit": args.save_tensor_limit,
        "save_tensor_format": args.save_tensor_format,
        "samples_dir": os.path.abspath(samples_dir) if args.save_tensors else None,
        "note": "Synthetic benchmark evaluates posterior ensemble directly against full x_true.",
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    case_rows: list[dict[str, object]] = []
    spread_bins: list[dict[str, object]] = []
    rank_counts: dict[tuple[str, str, float, float], np.ndarray] = {}
    coverage_hits: dict[tuple[str, str, float, float, str], list[float]] = {}
    map_sums: dict[tuple[str, str, float, float, str], np.ndarray] = {}
    map_counts: dict[tuple[str, str, float, float, str], float] = {}

    condition_specs = build_condition_specs(mask_types, densities, noise_levels, args)
    for local_case_index, case_index in enumerate(tqdm(case_indices, desc="synthetic eval cases")):
        x_true = np.asarray(dataset[int(case_index)], dtype=np.float32)
        name = dataset.files[int(case_index)]
        h, w = x_true.shape[-2:]
        valid = valid_mask if valid_mask is not None else np.ones((h, w), dtype=np.float32)
        season = season_from_name(name, case_index)
        regime = ice_regime(x_true, valid)

        for combo_index, spec in enumerate(condition_specs):
            mask_type = str(spec["mask_type"])
            density = float(spec["density"])
            noise_std = float(spec["noise"])
            condition_index = int(spec["condition_index"])
            n_conditions = int(spec["n_conditions"])
            combo_seed = args.seed + 100000 * local_case_index + 1000 * combo_index
            rng = np.random.default_rng(combo_seed)
            obs_config = ObservationConfig(
                mask_type=mask_type,
                density=density,
                noise_std=noise_std,
                seed=combo_seed,
                block_size=args.block_size,
                n_tracks_range=(args.n_tracks_min, args.n_tracks_max),
            )
            mask = make_observation_mask((h, w), obs_config, valid, rng)
            eval_mask = evaluation_mask(valid, mask, args.eval_region)
            observed = make_sparse_observation(x_true, mask, noise_std, rng)
            ensemble = runner.run(x_true, mask, observed, args.ensemble_size, combo_seed)
            condition_tag = tag_from_condition(mask_type, density, noise_std, condition_index, n_conditions)
            saved_tensor_path = ""
            if should_save_tensors(args, local_case_index):
                saved_tensor_path = save_case_tensors(
                    samples_dir=samples_dir,
                    args=args,
                    local_case_index=int(local_case_index),
                    case_index=int(case_index),
                    case_name=name,
                    condition_tag=condition_tag,
                    combo_seed=int(combo_seed),
                    x_true=x_true,
                    mask=mask,
                    observed=observed,
                    ensemble=ensemble,
                )
            mean = ensemble.mean(axis=0)
            crps = ensemble_crps(ensemble, x_true)
            rmse_by_var = np.array(
                [np.sqrt(weighted_mean((mean[ch] - x_true[ch]) ** 2, eval_mask)) for ch in range(x_true.shape[0])]
            )
            crps_by_var = [weighted_mean(crps[ch], eval_mask) for ch in range(x_true.shape[0])]

            if args.energy_score_limit > 0:
                flat_valid = np.flatnonzero(np.broadcast_to(eval_mask[None, :, :] > 0, x_true.shape).reshape(-1))
                if flat_valid.size > args.energy_score_limit:
                    flat_valid = rng.choice(flat_valid, size=args.energy_score_limit, replace=False)
                ens_flat = ensemble.reshape(args.ensemble_size, -1)[:, flat_valid]
                truth_flat = x_true.reshape(-1)[flat_valid]
                e_score = energy_score(ens_flat, truth_flat)
            else:
                e_score = float("nan")

            summary = ice_summaries(mean, x_true, eval_mask)
            for ch, var_name in enumerate(variables[: x_true.shape[0]]):
                var_weights = eval_mask
                ss = spread_skill(ensemble[:, ch], x_true[ch], var_weights)
                cover = interval_coverage(ensemble[:, ch], x_true[ch], LEVELS)
                row = {
                    "case_index": int(case_index),
                    "local_case_index": int(local_case_index),
                    "case_name": name,
                    "season": season,
                    "ice_regime": regime,
                    "variable": var_name,
                    "mask_type": mask_type,
                    "density": density,
                    "noise_level": noise_std,
                    "condition_index": condition_index,
                    "condition_tag": condition_tag,
                    "eval_region": args.eval_region,
                    "observed_fraction": float((mask * valid).sum() / max(1.0, valid.sum())),
                    "evaluated_fraction": float(eval_mask.sum() / max(1.0, valid.sum())),
                    "rmse": float(rmse_by_var[ch]),
                    "crps": float(crps_by_var[ch]),
                    "energy_score": e_score,
                    "tensor_path": saved_tensor_path,
                    **ss,
                    **summary,
                }
                for level_key, hits in cover.items():
                    value = weighted_mean(hits.astype(np.float64), var_weights)
                    row[f"coverage_{level_key}"] = value
                    coverage_hits.setdefault((var_name, mask_type, density, noise_std, level_key), []).append(value)
                case_rows.append(row)

                rank_mask = (eval_mask[:: args.rank_stride, :: args.rank_stride] > 0)
                rank_key = (var_name, mask_type, density, noise_std)
                counts = rank_histogram(
                    ensemble[:, ch, :: args.rank_stride, :: args.rank_stride],
                    x_true[ch, :: args.rank_stride, :: args.rank_stride],
                    seed=combo_seed + ch,
                    weights_mask=rank_mask,
                )
                rank_counts[rank_key] = rank_counts.get(rank_key, np.zeros_like(counts)) + counts

                for bin_row in binned_spread_skill(ensemble[:, ch], x_true[ch], weights_mask=eval_mask > 0):
                    spread_bins.append({
                        **bin_row,
                        "variable": var_name,
                        "mask_type": mask_type,
                        "condition_tag": condition_tag,
                        "density": density,
                        "noise_level": noise_std,
                        "eval_region": args.eval_region,
                    })

                for map_name, arr in (("mean_error", mean[ch] - x_true[ch]), ("crps", crps[ch])):
                    map_key = (var_name, mask_type, density, noise_std, map_name)
                    map_sums[map_key] = map_sums.get(map_key, np.zeros_like(arr, dtype=np.float64)) + arr
                    map_counts[map_key] = map_counts.get(map_key, 0.0) + 1.0

            if not args.no_plots and case_index == 0:
                plot_example_panel(x_true, mask, ensemble, os.path.join(plots_dir, f"example_{condition_tag}.png"))

    write_csv(os.path.join(args.output_dir, "per_case_metrics.csv"), case_rows)
    write_csv(os.path.join(args.output_dir, "spread_skill_bins.csv"), spread_bins)

    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in case_rows:
        key = (row["variable"], row["mask_type"], row["density"], row["noise_level"], row["eval_region"])
        grouped.setdefault(key, []).append(row)
    aggregate_rows = []
    for (var_name, mask_type, density, noise, eval_region), rows in grouped.items():
        out: dict[str, object] = {
            "variable": var_name,
            "mask_type": mask_type,
            "density": density,
            "noise_level": noise,
            "eval_region": eval_region,
            "n_condition_cases": len(rows),
            "n_unique_cases": len({int(r["case_index"]) for r in rows}),
        }
        for metric in ("rmse", "crps", "energy_score", "spread", "skill_rmse", "spread_skill_ratio"):
            out[metric] = float(np.nanmean([float(r[metric]) for r in rows]))
        for level in LEVELS:
            key = f"coverage_{level:g}"
            out[key] = float(np.nanmean([float(r[key]) for r in rows]))
        aggregate_rows.append(out)
    write_csv(os.path.join(args.output_dir, "aggregate_metrics.csv"), aggregate_rows)
    with open(os.path.join(args.output_dir, "aggregate_metrics.json"), "w") as f:
        json.dump(aggregate_rows, f, indent=2)

    np.savez_compressed(
        os.path.join(arrays_dir, "rank_histograms.npz"),
        **{f"{k[0]}__{k[1]}".replace(".", "p"): v for k, v in rank_counts.items()},
    )
    for key, total in map_sums.items():
        var_name, mask_type, density, noise, map_name = key
        tag = f"{var_name}__{mask_type}__d{density:g}__n{noise:g}__{map_name}".replace(".", "p")
        np.save(os.path.join(arrays_dir, f"{tag}.npy"), (total / map_counts[key]).astype(np.float32))

    if not args.no_plots:
        for key, counts in rank_counts.items():
            var_name, mask_type, density, noise = key
            tag = f"{var_name}_{mask_type}_d{density:g}_n{noise:g}".replace(".", "p")
            plot_rank_histogram(counts, os.path.join(plots_dir, f"rank_hist_{tag}.png"), f"Rank histogram: {tag}")
            empirical = [float(np.mean(coverage_hits[(var_name, mask_type, density, noise, f'{level:g}')])) for level in LEVELS]
            plot_coverage(list(LEVELS), empirical, os.path.join(plots_dir, f"coverage_{tag}.png"), f"Coverage: {tag}")
            rows = [
                r
                for r in spread_bins
                if r["variable"] == var_name
                and r["mask_type"] == mask_type
                and r["density"] == density
                and r["noise_level"] == noise
            ]
            plot_spread_skill(rows, os.path.join(plots_dir, f"spread_skill_{tag}.png"), f"Spread-skill: {tag}")
        plot_density_curve(case_rows, os.path.join(plots_dir, "rmse_vs_density.png"), "rmse")
        plot_density_curve(case_rows, os.path.join(plots_dir, "crps_vs_density.png"), "crps")

    print(json.dumps({"output_dir": os.path.abspath(args.output_dir), "n_cases": n_cases, "n_rows": len(case_rows)}, indent=2))


if __name__ == "__main__":
    main()
