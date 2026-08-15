#!/usr/bin/env python3
"""Build the paired uncertainty summary and compact SVG for the frozen method.

The input is the compact ``per_case_metrics.csv`` emitted by the established
cross-fitted spread analysis.  No raw ensemble values are read or required.
"""

from __future__ import annotations

import argparse
import csv
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


def read_cases(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("input table has no cases")
    required = {name for metric in METRICS for name in (metric, f"raw_{metric}")}
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(f"input table is missing columns: {', '.join(missing)}")
    return [{name: float(row[name]) for name in required} for row in rows]


def summarize(
    cases: list[dict[str, float]], *, samples: int, seed: int
) -> dict[str, object]:
    metrics: dict[str, object] = {}
    for offset, metric in enumerate(METRICS):
        raw = [case[f"raw_{metric}"] for case in cases]
        corrected = [case[metric] for case in cases]
        deltas = [after - before for before, after in zip(raw, corrected)]
        low, high = bootstrap_mean_ci(deltas, samples=samples, seed=seed + offset)
        metrics[metric] = {
            "raw_case_mean": sum(raw) / len(raw),
            "corrected_case_mean": sum(corrected) / len(corrected),
            "paired_mean_delta": sum(deltas) / len(deltas),
            "paired_mean_delta_percentile_bootstrap_95_ci": [low, high],
            "cases_improved": sum(delta < 0 for delta in deltas),
            "cases_total": len(deltas),
        }
    return {
        "analysis_unit": "case",
        "interval_method": "paired nonparametric percentile bootstrap over cases",
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
    args = parser.parse_args()
    if args.bootstrap_samples < 1_000:
        parser.error("--bootstrap-samples must be at least 1000")
    cases = read_cases(args.input_csv)
    summary = summarize(cases, samples=args.bootstrap_samples, seed=args.seed)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    fair_deltas = [
        case["analysis_fair_crps"] - case["raw_analysis_fair_crps"]
        for case in cases
    ]
    make_svg(fair_deltas, args.figure)


if __name__ == "__main__":
    main()
