#!/usr/bin/env python3
"""Build paired uncertainty artifacts from compact raw/corrected case tables.

The inputs contain case-level metrics only.  They are joined by an explicit ISO
date key; row order is never used and raw ensemble values are not required.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import random
from pathlib import Path


METRICS = (
    "analysis_fair_crps",
    "analysis_crps",
    "analysis_spread_skill_ratio",
    "analysis_coverage_90",
)
EXPECTED_CASES = 40
RECONCILIATION_TOLERANCE = 1e-10
EXPECTED_CASE_MEANS = {
    "analysis_fair_crps": (0.05849058504420133, 0.055689673560032536),
    "analysis_crps": (0.06210828098743627, 0.06110134210979743),
    "analysis_spread_skill_ratio": (0.7240662109968337, 1.061483512716757),
    "analysis_coverage_90": (0.5050835783457039, 0.8787749038797623),
}


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def bootstrap_mean_ci(
    deltas: list[float], *, samples: int, seed: int
) -> tuple[float, float]:
    rng = random.Random(seed)
    size = len(deltas)
    means = [
        sum(deltas[rng.randrange(size)] for _ in range(size)) / size
        for _ in range(samples)
    ]
    return percentile(means, 0.025), percentile(means, 0.975)


def circular_block_bootstrap_mean_ci(
    deltas: list[float], *, samples: int, seed: int, block_length: int
) -> tuple[float, float]:
    """Bootstrap the mean using circular contiguous blocks of ordered cases."""
    if not 1 < block_length <= len(deltas):
        raise ValueError("block_length must be between 2 and the number of cases")
    rng = random.Random(seed)
    size = len(deltas)
    means = []
    for _ in range(samples):
        resample = []
        while len(resample) < size:
            start = rng.randrange(size)
            resample.extend(
                deltas[(start + offset) % size] for offset in range(block_length)
            )
        means.append(sum(resample[:size]) / size)
    return percentile(means, 0.025), percentile(means, 0.975)


def read_table(
    path: Path, *, date_column: str, metric_prefix: str
) -> dict[dt.date, dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("input table has no cases")
    if len(rows) != EXPECTED_CASES:
        raise ValueError(
            f"expected {EXPECTED_CASES} cases from the frozen protocol, found {len(rows)}"
        )
    required = {f"{metric_prefix}{metric}" for metric in METRICS}
    required.add(date_column)
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(f"input table is missing columns: {', '.join(missing)}")
    if "case_index" in rows[0]:
        case_ids = [row["case_index"] for row in rows]
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("case_index values must be unique")
    dated_rows: dict[dt.date, dict[str, float]] = {}
    for index, row in enumerate(rows):
        try:
            date = dt.date.fromisoformat(row[date_column])
        except ValueError as error:
            raise ValueError(
                f"case {index} has a non-ISO date in {date_column}: {row[date_column]!r}"
            ) from error
        if date in dated_rows:
            raise ValueError(f"{date_column} values must be unique")
        case = {
            metric: float(row[f"{metric_prefix}{metric}"]) for metric in METRICS
        }
        nonfinite = sorted(name for name, value in case.items() if not math.isfinite(value))
        if nonfinite:
            raise ValueError(
                f"case {index} has non-finite values in: {', '.join(nonfinite)}"
            )
        dated_rows[date] = case
    return dated_rows


def read_cases(
    corrected_path: Path, raw_path: Path, *, date_column: str
) -> list[dict[str, float]]:
    corrected = read_table(corrected_path, date_column=date_column, metric_prefix="")
    raw = read_table(raw_path, date_column=date_column, metric_prefix="")
    if corrected.keys() != raw.keys():
        missing_raw = sorted(date.isoformat() for date in corrected.keys() - raw.keys())
        missing_corrected = sorted(date.isoformat() for date in raw.keys() - corrected.keys())
        raise ValueError(
            "raw and corrected date sets differ; "
            f"missing from raw: {missing_raw}; missing from corrected: {missing_corrected}"
        )
    return [
        {
            **corrected[date],
            **{f"raw_{metric}": raw[date][metric] for metric in METRICS},
        }
        for date in sorted(corrected)
    ]


def summarize(
    cases: list[dict[str, float]],
    *,
    samples: int,
    seed: int,
    block_length: int,
    date_column: str,
    input_sha256: str,
) -> dict[str, object]:
    metrics: dict[str, object] = {}
    for offset, metric in enumerate(METRICS):
        raw = [case[f"raw_{metric}"] for case in cases]
        corrected = [case[metric] for case in cases]
        deltas = [after - before for before, after in zip(raw, corrected)]
        low, high = bootstrap_mean_ci(deltas, samples=samples, seed=seed + offset)
        block_low, block_high = circular_block_bootstrap_mean_ci(
            deltas,
            samples=samples,
            seed=seed + len(METRICS) + offset,
            block_length=block_length,
        )
        metrics[metric] = {
            "raw_case_mean": sum(raw) / len(raw),
            "corrected_case_mean": sum(corrected) / len(corrected),
            "paired_mean_delta": sum(deltas) / len(deltas),
            "paired_mean_delta_percentile_bootstrap_95_ci": [low, high],
            "paired_mean_delta_circular_block_bootstrap_95_ci": [
                block_low,
                block_high,
            ],
            "cases_improved": sum(delta < 0 for delta in deltas),
            "cases_total": len(deltas),
        }
    return {
        "schema_version": 2,
        "input_sha256": input_sha256,
        "analysis_unit": "case",
        "temporal_order_column": date_column,
        "interval_methods": [
            "paired nonparametric percentile bootstrap over cases",
            "paired circular contiguous-block percentile bootstrap over date-ordered cases",
        ],
        "block_length_cases": block_length,
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "metrics": metrics,
    }


def reconcile_case_means(summary: dict[str, object]) -> None:
    """Fail closed unless case means reproduce the trusted aggregate summary."""
    metrics = summary["metrics"]
    assert isinstance(metrics, dict)
    failures = []
    for metric, (expected_raw, expected_corrected) in EXPECTED_CASE_MEANS.items():
        values = metrics[metric]
        assert isinstance(values, dict)
        for label, expected in (
            ("raw_case_mean", expected_raw),
            ("corrected_case_mean", expected_corrected),
        ):
            observed = values[label]
            assert isinstance(observed, float)
            error = abs(observed - expected)
            if error > RECONCILIATION_TOLERANCE:
                failures.append(
                    f"{metric}.{label}: observed {observed:.17g}, "
                    f"expected {expected:.17g}, absolute error {error:.3g}"
                )
    if failures:
        raise ValueError(
            "case means do not reconcile with the trusted aggregate summary "
            f"within {RECONCILIATION_TOLERANCE:g}: " + "; ".join(failures)
        )


def make_svg(deltas: list[float], output: Path) -> None:
    width, height = 760, 360
    left, right, top, bottom = 72, 24, 34, 58
    plot_width, plot_height = width - left - right, height - top - bottom
    bound = max(abs(min(deltas)), abs(max(deltas)), 1e-12) * 1.08
    x = lambda value: left + (value + bound) / (2 * bound) * plot_width
    ordered = sorted(deltas)
    circles = []
    for index, value in enumerate(ordered):
        y = top + plot_height - (index + 0.5) / len(ordered) * plot_height
        color = "#2166ac" if value < 0 else "#b2182b"
        circles.append(f'<circle cx="{x(value):.2f}" cy="{y:.2f}" r="4" fill="{color}"/>')
    mean = sum(deltas) / len(deltas)
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{left}" y="20" font-family="sans-serif" font-size="15">Paired case-level fair-CRPS change (corrected − raw)</text>
<line x1="{x(0):.2f}" y1="{top}" x2="{x(0):.2f}" y2="{top + plot_height}" stroke="#555" stroke-dasharray="4 4"/>
<line x1="{x(mean):.2f}" y1="{top}" x2="{x(mean):.2f}" y2="{top + plot_height}" stroke="#111" stroke-width="2"/>
{''.join(circles)}
<line x1="{left}" y1="{top + plot_height}" x2="{width - right}" y2="{top + plot_height}" stroke="#222"/>
<text x="{left}" y="{height - 20}" font-family="sans-serif" font-size="12">lower is better; vertical solid line is the paired case mean</text>
<text x="{width - right}" y="{height - 20}" text-anchor="end" font-family="sans-serif" font-size="12">n={len(deltas)}</text>
</svg>'''
    output.write_text(svg, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("corrected_csv", type=Path)
    parser.add_argument("--raw-csv", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20220815)
    parser.add_argument("--date-column", required=True)
    parser.add_argument("--block-length", type=int, required=True)
    args = parser.parse_args()
    if args.bootstrap_samples < 1_000:
        parser.error("--bootstrap-samples must be at least 1000")
    if args.summary.resolve() == args.figure.resolve():
        parser.error("--summary and --figure must be different paths")
    if not 1 < args.block_length <= EXPECTED_CASES:
        parser.error(f"--block-length must be between 2 and {EXPECTED_CASES}")
    cases = read_cases(
        args.corrected_csv, args.raw_csv, date_column=args.date_column
    )
    input_sha256 = hashlib.sha256(
        args.corrected_csv.read_bytes() + b"\0" + args.raw_csv.read_bytes()
    ).hexdigest()
    summary = summarize(
        cases,
        samples=args.bootstrap_samples,
        seed=args.seed,
        block_length=args.block_length,
        date_column=args.date_column,
        input_sha256=input_sha256,
    )
    reconcile_case_means(summary)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    fair_deltas = [
        case["analysis_fair_crps"] - case["raw_analysis_fair_crps"]
        for case in cases
    ]
    make_svg(fair_deltas, args.figure)


if __name__ == "__main__":
    main()
