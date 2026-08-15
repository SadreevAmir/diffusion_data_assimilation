#!/usr/bin/env python3
"""Build the paired uncertainty summary and compact SVG for the frozen method.

The input is the compact ``per_case_metrics.csv`` emitted by the established
cross-fitted spread analysis.  No raw ensemble values are read or required.
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


def read_cases(path: Path, *, date_column: str) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("input table has no cases")
    if len(rows) != EXPECTED_CASES:
        raise ValueError(
            f"expected {EXPECTED_CASES} cases from the frozen protocol, found {len(rows)}"
        )
    required = {name for metric in METRICS for name in (metric, f"raw_{metric}")}
    required.add(date_column)
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(f"input table is missing columns: {', '.join(missing)}")
    if "case_index" in rows[0]:
        case_ids = [row["case_index"] for row in rows]
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("case_index values must be unique")
    dated_rows = []
    for index, row in enumerate(rows):
        try:
            date = dt.date.fromisoformat(row[date_column])
        except ValueError as error:
            raise ValueError(
                f"case {index} has a non-ISO date in {date_column}: {row[date_column]!r}"
            ) from error
        dated_rows.append((date, row))
    dates = [date for date, _ in dated_rows]
    if len(set(dates)) != len(dates):
        raise ValueError(f"{date_column} values must be unique")
    dated_rows.sort(key=lambda item: item[0])
    cases = [
        {name: float(row[name]) for name in required if name != date_column}
        for _, row in dated_rows
    ]
    for index, case in enumerate(cases):
        nonfinite = sorted(name for name, value in case.items() if not math.isfinite(value))
        if nonfinite:
            raise ValueError(
                f"case {index} has non-finite values in: {', '.join(nonfinite)}"
            )
    return cases


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
    parser.add_argument("input_csv", type=Path)
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
    cases = read_cases(args.input_csv, date_column=args.date_column)
    input_sha256 = hashlib.sha256(args.input_csv.read_bytes()).hexdigest()
    summary = summarize(
        cases,
        samples=args.bootstrap_samples,
        seed=args.seed,
        block_length=args.block_length,
        date_column=args.date_column,
        input_sha256=input_sha256,
    )
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
