#!/usr/bin/env python3
"""Create an aggregate calibration figure from a trusted summary JSON."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path

if __package__:
    from .atomic_publish import publish_text_artifacts
    from .validate_server_only_manifest import validate_manifest
else:
    from atomic_publish import publish_text_artifacts
    from validate_server_only_manifest import validate_manifest


METRICS = (
    ("Fair CRPS", "analysis_fair_crps", "raw_analysis_fair_crps", "lower"),
    ("Ordinary CRPS", "analysis_crps", "raw_analysis_crps", "lower"),
    ("Spread-skill ratio", "analysis_spread_skill_ratio", "raw_analysis_spread_skill_ratio", "one"),
    ("50% interval diagnostic", "analysis_coverage_50", "raw_analysis_coverage_50", "target"),
    ("80% interval diagnostic", "analysis_coverage_80", "raw_analysis_coverage_80", "target"),
    ("90% interval diagnostic", "analysis_coverage_90", "raw_analysis_coverage_90", "target"),
    ("95% interval diagnostic", "analysis_coverage_95", "raw_analysis_coverage_95", "target"),
)
EXPECTED_EXPERIMENT = "joint_crossfit_spread_calibration_valid"


def load_summary(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], dict):
        raise ValueError("aggregate JSON must be a one-row list")
    row = payload[0]
    required = {key for _, corrected, raw, _ in METRICS for key in (corrected, raw)} | {"num_cases"}
    missing = sorted(required - row.keys())
    if missing:
        raise ValueError(f"missing required keys: {', '.join(missing)}")
    if int(row["num_cases"]) != 40:
        raise ValueError("num_cases must equal 40")
    result: dict[str, float] = {}
    for key in required:
        value = float(row[key])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid value for {key}")
        result[key] = value
    for _, corrected, raw, _ in METRICS:
        if result[corrected] == 0 and result[raw] == 0:
            raise ValueError(f"undefined zero scale for {corrected} and {raw}")
    return result


def make_svg(row: dict[str, float]) -> str:
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="590" viewBox="0 0 900 590">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#18212b}.title{font-size:24px;font-weight:700}.sub{font-size:14px;fill:#4d5966}.label{font-size:14px}.value{font-size:12px}.axis{stroke:#aab3bd}.raw{fill:#8b9bad}.corrected{fill:#177e89}.target{stroke:#c44536;stroke-width:2;stroke-dasharray:5 4}</style>',
        '<text class="title" x="36" y="38">Cross-fitted global spread correction</text>',
        '<text class="sub" x="36" y="62">Aggregate validation diagnostics; 40 dates, 10 ensemble members</text>',
    ]
    y = 94
    for label, corrected_key, raw_key, kind in METRICS:
        raw, corrected = row[raw_key], row[corrected_key]
        scale_max = max(raw, corrected) * 1.18 if kind == "lower" else 1.12
        target = None if kind == "lower" else (1.0 if kind == "one" else float(label.split("%", 1)[0]) / 100.0)
        x0, span = 270, 560
        raw_w, corrected_w = span * raw / scale_max, span * corrected / scale_max
        parts.extend((
            f'<text class="label" x="36" y="{y + 25}">{html.escape(label)}</text>',
            f'<line class="axis" x1="{x0}" y1="{y + 18}" x2="{x0 + span}" y2="{y + 18}"/>',
            f'<rect class="raw" x="{x0}" y="{y}" width="{raw_w:.2f}" height="14"/>',
            f'<rect class="corrected" x="{x0}" y="{y + 22}" width="{corrected_w:.2f}" height="14"/>',
            f'<text class="value" x="{min(x0 + raw_w + 6, 842):.2f}" y="{y + 12}">{raw:.4f}</text>',
            f'<text class="value" x="{min(x0 + corrected_w + 6, 842):.2f}" y="{y + 34}">{corrected:.4f}</text>',
        ))
        if target is not None:
            tx = x0 + span * target / scale_max
            parts.append(f'<line class="target" x1="{tx:.2f}" y1="{y - 3}" x2="{tx:.2f}" y2="{y + 39}"/>')
        y += 65
    parts.extend((
        '<rect class="raw" x="270" y="556" width="18" height="12"/><text class="sub" x="296" y="567">Raw ensemble</text>',
        '<rect class="corrected" x="420" y="556" width="18" height="12"/><text class="sub" x="446" y="567">Cross-fitted correction</text>',
        '<line class="target" x1="628" y1="562" x2="650" y2="562"/><text class="sub" x="658" y="567">Reference target</text>',
        '</svg>',
    ))
    return "\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("aggregate_json", type=Path)
    parser.add_argument("output_svg", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    validate_manifest(args.aggregate_json, args.manifest, EXPECTED_EXPERIMENT)
    output = make_svg(load_summary(args.aggregate_json))
    publish_text_artifacts(((args.output_svg, output),))


if __name__ == "__main__":
    main()
