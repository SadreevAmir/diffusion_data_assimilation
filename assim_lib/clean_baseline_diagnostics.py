"""Fixed visual and distributional checks for the clean retraining pilot.

The module only reads physical ensemble arrays written by ``assim_lib.evaluate``.
It never changes members.  Mixed-distribution ranks are deterministic and
tie-aware: D members equal to truth contribute uniformly to D+1 admissible
ranks instead of being assigned a randomized rank.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


SIC_BANDS = (
    ("exact_zero", lambda values: values == 0.0),
    ("trace_0_0.15", lambda values: (values > 0.0) & (values <= 0.15)),
    ("marginal_0.15_0.8", lambda values: (values > 0.15) & (values < 0.8)),
    ("high_0.8_0.95", lambda values: (values >= 0.8) & (values < 0.95)),
    ("near_one_0.95_1", lambda values: (values >= 0.95) & (values < 1.0)),
    ("exact_one", lambda values: values == 1.0),
)
SEASONS = {
    1: "DJF",
    2: "DJF",
    3: "MAM",
    4: "MAM",
    5: "MAM",
    6: "JJA",
    7: "JJA",
    8: "JJA",
    9: "SON",
    10: "SON",
    11: "SON",
    12: "DJF",
}


def tie_aware_rank_counts(
    ensemble: np.ndarray,
    truth: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """Return fractional rank counts for an ensemble with mixed atoms."""
    members = np.asarray(ensemble)
    target = np.asarray(truth)
    mask = np.asarray(valid, dtype=bool)
    if members.ndim != target.ndim + 1 or members.shape[1:] != target.shape:
        raise ValueError("ensemble must have shape [M,...] co-registered with truth")
    if mask.shape != target.shape:
        raise ValueError("valid mask must be co-registered with truth")
    if members.shape[0] < 2:
        raise ValueError("tie-aware ensemble diagnostics require at least two members")
    if not np.all(np.isfinite(members[:, mask])) or not np.all(np.isfinite(target[mask])):
        raise ValueError("non-finite values are forbidden on the valid domain")

    below = np.sum(members < target[None, ...], axis=0)
    tied = np.sum(members == target[None, ...], axis=0)
    counts = np.zeros(members.shape[0] + 1, dtype=np.float64)
    share = 1.0 / (tied + 1.0)
    for rank in range(members.shape[0] + 1):
        admissible = mask & (rank >= below) & (rank <= below + tied)
        counts[rank] = float(np.sum(share[admissible]))
    if not math.isclose(float(counts.sum()), float(mask.sum()), rel_tol=0.0, abs_tol=1e-6):
        raise RuntimeError("fractional rank mass does not equal the valid-pixel count")
    return counts


def fair_crps_sum(ensemble: np.ndarray, truth: np.ndarray, valid: np.ndarray) -> float:
    """Return the valid-pixel sum of the finite-ensemble unbiased CRPS."""
    members = np.asarray(ensemble, dtype=np.float64)[:, np.asarray(valid, dtype=bool)]
    target = np.asarray(truth, dtype=np.float64)[np.asarray(valid, dtype=bool)]
    member_count = members.shape[0]
    if member_count < 2:
        raise ValueError("fair CRPS requires at least two members")
    score = np.mean(np.abs(members - target[None, :]), axis=0)
    pair_sum = np.zeros_like(score)
    for left in range(member_count):
        for right in range(left + 1, member_count):
            pair_sum += np.abs(members[left] - members[right])
    score -= pair_sum / (member_count * (member_count - 1))
    return float(np.sum(score))


def _ordinary_crps_sum(ensemble: np.ndarray, truth: np.ndarray, valid: np.ndarray) -> float:
    members = np.asarray(ensemble, dtype=np.float64)[:, np.asarray(valid, dtype=bool)]
    target = np.asarray(truth, dtype=np.float64)[np.asarray(valid, dtype=bool)]
    member_count = members.shape[0]
    score = np.mean(np.abs(members - target[None, :]), axis=0)
    pair_sum = np.zeros_like(score)
    for left in range(member_count):
        for right in range(left + 1, member_count):
            pair_sum += np.abs(members[left] - members[right])
    score -= pair_sum / (member_count * member_count)
    return float(np.sum(score))


def _endpoint_totals(ensemble: np.ndarray, truth: np.ndarray, valid: np.ndarray, value: float) -> dict:
    selector = np.asarray(valid, dtype=bool)
    observed = (np.asarray(truth)[selector] == value).astype(np.float64)
    probability = np.mean(np.asarray(ensemble)[:, selector] == value, axis=0)
    return {
        "count": int(selector.sum()),
        "truth_sum": float(observed.sum()),
        "forecast_sum": float(probability.sum()),
        "brier_sum": float(np.square(probability - observed).sum()),
    }


def _new_rank_group() -> dict[str, Any]:
    return {"pixels": 0, "rank_mass": None, "strict_upper": 0}


def _add_rank_group(
    group: dict[str, Any],
    ensemble: np.ndarray,
    truth: np.ndarray,
    selector: np.ndarray,
) -> None:
    count = int(selector.sum())
    if count == 0:
        return
    counts = tie_aware_rank_counts(ensemble, truth, selector)
    if group["rank_mass"] is None:
        group["rank_mass"] = np.zeros_like(counts)
    group["pixels"] += count
    group["rank_mass"] += counts
    group["strict_upper"] += int(np.sum(selector & (truth > np.max(ensemble, axis=0))))


def _finish_rank_group(group: dict[str, Any]) -> dict[str, Any]:
    count = int(group["pixels"])
    if count == 0:
        return {"pixels": 0, "tie_aware_rank10_rate": None, "strict_truth_above_all_rate": None}
    return {
        "pixels": count,
        "tie_aware_rank10_rate": float(group["rank_mass"][-1] / count),
        "strict_truth_above_all_rate": float(group["strict_upper"] / count),
    }


def _case_date(payload: Any, fallback: str) -> date:
    scalar = payload.item() if isinstance(payload, np.ndarray) and payload.ndim == 0 else payload
    text = str(scalar) if scalar is not None else fallback
    match = re.search(r"\d{4}-\d{2}-\d{2}", text)
    if match is not None:
        text = match.group(0)
    try:
        return date.fromisoformat(text)
    except ValueError as error:
        raise ValueError(f"cannot recover ISO target date from {text!r}") from error


def evaluate_sample_directory(sample_dir: Path) -> dict[str, Any]:
    paths = sorted(sample_dir.glob("*.npz"))
    if not paths:
        raise ValueError(f"no NPZ sample arrays found in {sample_dir}")
    rank_counts = None
    point_count = 0
    error_sum = 0.0
    squared_error_sum = 0.0
    fair_sum = 0.0
    ordinary_sum = 0.0
    endpoints = {"zero": defaultdict(float), "one": defaultdict(float)}
    bands = {name: _new_rank_group() for name, _ in SIC_BANDS}
    seasons = {name: _new_rank_group() for name in ("DJF", "MAM", "JJA", "SON")}
    cases = []
    member_count = None

    for path in paths:
        with np.load(path, allow_pickle=False) as payload:
            fields = [str(field) for field in payload["fields"].tolist()]
            if "siconc" not in fields:
                raise ValueError(f"siconc is absent from {path.name}")
            channel = fields.index("siconc")
            ensemble = np.asarray(payload["analysis_ensemble"][:, channel], dtype=np.float64)
            truth = np.asarray(payload["truth"][channel], dtype=np.float64)
            background = np.asarray(payload["background"][channel], dtype=np.float64)
            valid = np.asarray(payload["valid_mask"][channel], dtype=bool)
            target_date = _case_date(
                payload["target_date"] if "target_date" in payload else None,
                path.name,
            )
        if ensemble.ndim != 3 or ensemble.shape[1:] != truth.shape:
            raise ValueError(f"unexpected ensemble shape in {path.name}")
        if background.shape != truth.shape or valid.shape != truth.shape:
            raise ValueError(f"truth/background/mask shapes differ in {path.name}")
        if member_count is None:
            member_count = ensemble.shape[0]
        if ensemble.shape[0] != member_count:
            raise ValueError("member count changes between cases")
        if not np.all(np.isfinite(ensemble[:, valid])) or not np.all(np.isfinite(truth[valid])):
            raise ValueError(f"non-finite valid-domain values in {path.name}")
        if np.any((ensemble[:, valid] < 0.0) | (ensemble[:, valid] > 1.0)):
            raise ValueError(f"physical SIC member lies outside [0,1] in {path.name}")
        if np.any((truth[valid] < 0.0) | (truth[valid] > 1.0)):
            raise ValueError(f"physical SIC truth lies outside [0,1] in {path.name}")

        counts = tie_aware_rank_counts(ensemble, truth, valid)
        rank_counts = counts if rank_counts is None else rank_counts + counts
        mean = np.mean(ensemble, axis=0)
        difference = mean[valid] - truth[valid]
        count = int(valid.sum())
        point_count += count
        error_sum += float(difference.sum())
        squared_error_sum += float(np.square(difference).sum())
        fair_sum += fair_crps_sum(ensemble, truth, valid)
        ordinary_sum += _ordinary_crps_sum(ensemble, truth, valid)
        for endpoint_name, endpoint_value in (("zero", 0.0), ("one", 1.0)):
            totals = _endpoint_totals(ensemble, truth, valid, endpoint_value)
            for key, value in totals.items():
                endpoints[endpoint_name][key] += value
        for band_name, predicate in SIC_BANDS:
            _add_rank_group(bands[band_name], ensemble, truth, valid & predicate(truth))
        _add_rank_group(seasons[SEASONS[target_date.month]], ensemble, truth, valid)
        cases.append(
            {
                "path": path.name,
                "target_date": target_date.isoformat(),
                "pixels": count,
                "fair_crps": fair_crps_sum(ensemble, truth, valid) / count,
                "mean_rmse": float(np.sqrt(np.mean(np.square(difference)))),
                "mean_bias": float(np.mean(difference)),
                "strict_truth_above_all_rate": float(np.mean(truth[valid] > np.max(ensemble, axis=0)[valid])),
            }
        )

    assert rank_counts is not None and member_count is not None
    frequencies = rank_counts / point_count
    uniform = 1.0 / (member_count + 1)
    endpoint_metrics = {}
    for name, totals in endpoints.items():
        count = totals["count"]
        endpoint_metrics[name] = {
            "truth_frequency": totals["truth_sum"] / count,
            "mean_forecast_probability": totals["forecast_sum"] / count,
            "probability_bias": (totals["forecast_sum"] - totals["truth_sum"]) / count,
            "brier_score": totals["brier_sum"] / count,
        }
    return {
        "schema_version": "clean-baseline-diagnostics-v1",
        "cases": len(paths),
        "ensemble_size": member_count,
        "valid_pixels": point_count,
        "fair_crps": fair_sum / point_count,
        "ordinary_crps": ordinary_sum / point_count,
        "ensemble_mean_rmse": math.sqrt(squared_error_sum / point_count),
        "ensemble_mean_bias": error_sum / point_count,
        "rank_histogram": {
            "definition": "fractional_tie_aware_D_plus_1_admissible_ranks",
            "counts": rank_counts.tolist(),
            "frequencies": frequencies.tolist(),
            "total_variation_from_uniform": float(0.5 * np.abs(frequencies - uniform).sum()),
            "normalized_mean_rank": float(
                np.dot(np.arange(member_count + 1), frequencies) / member_count
            ),
        },
        "endpoint_mass_calibration": endpoint_metrics,
        "rank10_by_truth_band": {name: _finish_rank_group(group) for name, group in bands.items()},
        "rank10_by_season": {name: _finish_rank_group(group) for name, group in seasons.items()},
        "per_case": cases,
    }


def _sample_inventory(sample_dir: Path) -> dict[str, Path]:
    inventory = {}
    for path in sorted(sample_dir.glob("*.npz")):
        with np.load(path, allow_pickle=False) as payload:
            target = _case_date(
                payload["target_date"] if "target_date" in payload else None,
                path.name,
            ).isoformat()
        if target in inventory:
            raise ValueError(f"duplicate target date {target} in {sample_dir}")
        inventory[target] = path
    return inventory


def make_comparison_panels(
    candidate_dir: Path,
    reference_dir: Path,
    output_dir: Path,
    *,
    case_orders: tuple[int, ...],
    member_ids: tuple[int, ...],
) -> list[str]:
    candidate = _sample_inventory(candidate_dir)
    reference = _sample_inventory(reference_dir)
    dates = sorted(candidate)
    if set(candidate) != set(reference):
        raise ValueError("candidate and accepted-reference dates differ")
    if any(order < 0 or order >= len(dates) for order in case_orders):
        raise ValueError("fixed visual case order is outside the sample inventory")
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []

    for order in case_orders:
        target_date = dates[order]
        rows = []
        variants = (
            ("new pilot", candidate[target_date]),
            ("accepted reference", reference[target_date]),
        )
        for label, path in variants:
            with np.load(path, allow_pickle=False) as payload:
                fields = [str(field) for field in payload["fields"].tolist()]
                channel = fields.index("siconc")
                ensemble = np.asarray(payload["analysis_ensemble"][:, channel], dtype=np.float64)
                truth = np.asarray(payload["truth"][channel], dtype=np.float64)
                background = np.asarray(payload["background"][channel], dtype=np.float64)
                valid = np.asarray(payload["valid_mask"][channel], dtype=bool)
            if any(member < 0 or member >= ensemble.shape[0] for member in member_ids):
                raise ValueError("fixed visual member id is outside the ensemble")
            rows.append((label, ensemble, truth, background, valid))
        if not np.array_equal(rows[0][4], rows[1][4]):
            raise ValueError(f"candidate/reference valid masks differ on {target_date}")
        if not np.array_equal(rows[0][2], rows[1][2]):
            raise ValueError(f"candidate/reference truth arrays differ on {target_date}")
        if not np.array_equal(rows[0][3], rows[1][3]):
            raise ValueError(f"candidate/reference background arrays differ on {target_date}")

        titles = ["truth", "background", *(f"member {member}" for member in member_ids), "mean", "std"]
        figure, axes = plt.subplots(
            2,
            len(titles),
            figsize=(2.7 * len(titles), 6.6),
            squeeze=False,
            constrained_layout=True,
        )
        figure.suptitle(
            f"Fixed member visual gate — {target_date}\n"
            "native array orientation (row 0 at top), identical SIC scale [0,1]",
            fontsize=15,
        )
        for row_index, (label, ensemble, truth, background, valid) in enumerate(rows):
            values = [truth, background]
            values.extend(ensemble[member] for member in member_ids)
            values.extend((ensemble.mean(axis=0), ensemble.std(axis=0)))
            for column, (title, field) in enumerate(zip(titles, values, strict=True)):
                axis = axes[row_index, column]
                display = np.where(valid, field, np.nan)
                image = axis.imshow(
                    np.ma.masked_invalid(display),
                    origin="upper",
                    interpolation="nearest",
                    cmap="Blues_r" if title != "std" else "magma",
                    vmin=0.0,
                    vmax=1.0,
                )
                axis.set_title(title)
                axis.axis("off")
                if column == 0:
                    axis.set_ylabel(label)
                if title in ("mean", "std"):
                    figure.colorbar(image, ax=axis, fraction=0.045, pad=0.02)
        path = output_dir / f"fixed_members_{order:02d}_{target_date}.png"
        figure.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        outputs.append(str(path))
    return outputs


def make_rank_comparison_figure(
    candidate: dict[str, Any],
    reference: dict[str, Any],
    output_path: Path,
) -> str:
    """Plot the primary tie-aware ranks and endpoint-mass calibration."""
    candidate_rank = np.asarray(candidate["rank_histogram"]["frequencies"], dtype=np.float64)
    reference_rank = np.asarray(reference["rank_histogram"]["frequencies"], dtype=np.float64)
    if candidate_rank.shape != reference_rank.shape:
        raise ValueError("candidate and reference rank histograms differ in width")
    positions = np.arange(candidate_rank.size)
    width = 0.38
    figure, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
    axes[0].bar(positions - width / 2, reference_rank, width, label="accepted reference")
    axes[0].bar(positions + width / 2, candidate_rank, width, label="new pilot")
    axes[0].axhline(1.0 / candidate_rank.size, color="black", linestyle="--", label="uniform")
    axes[0].set_xticks(positions)
    axes[0].set_xlabel("Tie-aware fractional rank")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("SIC rank histogram (ties split over D+1 ranks)")
    axes[0].legend()

    labels = ["Y=0 truth", "Y=0 forecast", "Y=1 truth", "Y=1 forecast"]
    reference_endpoint = reference["endpoint_mass_calibration"]
    candidate_endpoint = candidate["endpoint_mass_calibration"]
    reference_values = [
        reference_endpoint["zero"]["truth_frequency"],
        reference_endpoint["zero"]["mean_forecast_probability"],
        reference_endpoint["one"]["truth_frequency"],
        reference_endpoint["one"]["mean_forecast_probability"],
    ]
    candidate_values = [
        candidate_endpoint["zero"]["truth_frequency"],
        candidate_endpoint["zero"]["mean_forecast_probability"],
        candidate_endpoint["one"]["truth_frequency"],
        candidate_endpoint["one"]["mean_forecast_probability"],
    ]
    endpoint_positions = np.arange(len(labels))
    axes[1].bar(endpoint_positions - width / 2, reference_values, width, label="accepted reference")
    axes[1].bar(endpoint_positions + width / 2, candidate_values, width, label="new pilot")
    axes[1].set_xticks(endpoint_positions, labels, rotation=20, ha="right")
    axes[1].set_ylabel("Frequency / mean forecast probability")
    axes[1].set_title("Endpoint mass calibration")
    axes[1].legend()
    figure.suptitle("Fixed clean-pilot probabilistic diagnostics", fontsize=15)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return str(output_path)


def analyze(
    candidate_dir: Path,
    reference_dir: Path,
    output_dir: Path,
    *,
    case_orders: tuple[int, ...] = (0, 2, 4, 6),
    member_ids: tuple[int, ...] = (0, 3, 6, 9),
) -> dict[str, Any]:
    candidate_metrics = evaluate_sample_directory(candidate_dir)
    reference_metrics = evaluate_sample_directory(reference_dir)
    if candidate_metrics["cases"] != reference_metrics["cases"]:
        raise ValueError("candidate and accepted-reference case counts differ")
    output_dir.mkdir(parents=True, exist_ok=True)
    panels = make_comparison_panels(
        candidate_dir,
        reference_dir,
        output_dir,
        case_orders=case_orders,
        member_ids=member_ids,
    )
    rank_figure = make_rank_comparison_figure(
        candidate_metrics,
        reference_metrics,
        output_dir / "tie_aware_rank_and_endpoint_mass.png",
    )
    comparison_keys = ("fair_crps", "ordinary_crps", "ensemble_mean_rmse", "ensemble_mean_bias")
    summary = {
        "schema_version": "clean-baseline-comparison-v1",
        "candidate": candidate_metrics,
        "accepted_reference": reference_metrics,
        "candidate_minus_reference": {
            key: candidate_metrics[key] - reference_metrics[key] for key in comparison_keys
        },
        "visual_gate": {
            "status": "REQUIRES_HUMAN_REVIEW",
            "continuation_rule": (
                "Do not extend training or start calibration until every fixed panel has been "
                "visually checked for smooth individual SIC fields, correct orientation, no "
                "narrow track imprint, and no noise/texture regression versus the accepted reference."
            ),
            "case_orders": list(case_orders),
            "member_ids": list(member_ids),
            "orientation": "native_array_row_0_at_top_no_flip_no_transpose",
            "siconc_scale": [0.0, 1.0],
            "panels": panels,
            "rank_and_endpoint_figure": rank_figure,
        },
    }
    (output_dir / "diagnostics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", required=True, type=Path)
    parser.add_argument("--reference-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--case-orders", default="0,2,4,6")
    parser.add_argument("--member-ids", default="0,3,6,9")
    args = parser.parse_args()
    case_orders = tuple(int(value) for value in args.case_orders.split(",") if value)
    member_ids = tuple(int(value) for value in args.member_ids.split(",") if value)
    print(
        json.dumps(
            analyze(
                args.candidate_dir,
                args.reference_dir,
                args.output_dir,
                case_orders=case_orders,
                member_ids=member_ids,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
