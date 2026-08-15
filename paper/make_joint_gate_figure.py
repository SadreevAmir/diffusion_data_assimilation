#!/usr/bin/env python3
"""Create a claim-led joint-gate figure from the trusted aggregate JSON."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path


METHOD = "crossfit_frozen_scale_mean_preserving_capped_simplex"
METRICS = (
    ("Fair CRPS", "analysis_fair_crps", 0.97),
    ("Established-ice Brier", "established_ice_brier_score", 1.01),
    ("Mean IIEE", "analysis_mean_iiee", 1.02),
    ("Edge disagreement", "edge_disagreement_fraction", 1.02),
    ("Local variogram, lag 1", "local_variogram_score_lag_1_p05", 1.02),
)


def load_summary(path: Path) -> tuple[dict[str, float], dict[str, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("method_aggregates")
    gate = payload.get("gate")
    if not isinstance(rows, list) or not isinstance(gate, dict):
        raise ValueError("expected method_aggregates and gate objects")
    if gate.get("candidate_method") != METHOD or gate.get("overall_eligible") is not False:
        raise ValueError("unexpected candidate or gate decision")
    selected = {row.get("method"): row for row in rows if row.get("region") == "full"}
    if set(("raw", METHOD)) - selected.keys():
        raise ValueError("missing raw or projected-spread full-region row")
    raw, candidate = selected["raw"], selected[METHOD]
    if int(raw.get("num_cases", -1)) != 40 or int(candidate.get("num_cases", -1)) != 40:
        raise ValueError("num_cases must equal 40")
    for _, key, _ in METRICS:
        for row in (raw, candidate):
            value = float(row[key])
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid value for {key}")
    return raw, candidate


def make_svg(raw: dict[str, float], candidate: dict[str, float]) -> str:
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="470" viewBox="0 0 900 470">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#18212b}.title{font-size:23px;font-weight:700}.sub{font-size:14px;fill:#4d5966}.label{font-size:14px}.value{font-size:12px}.axis{stroke:#aab3bd}.raw{fill:#8b9bad}.candidate{fill:#177e89}.limit{stroke:#c44536;stroke-width:2;stroke-dasharray:5 4}</style>',
        '<text class="title" x="36" y="38">Mean-preserving spread: joint-gate diagnostics</text>',
        '<text class="sub" x="36" y="62">Candidate/raw ratios; lower is better; 40 dates, 10 members</text>',
    ]
    y = 94
    for label, key, limit in METRICS:
        ratio = candidate[key] / raw[key]
        x0, span, scale_max = 310, 520, 1.25
        raw_w, candidate_w = span / scale_max, span * ratio / scale_max
        lx = x0 + span * limit / scale_max
        parts.extend((
            f'<text class="label" x="36" y="{y + 25}">{html.escape(label)}</text>',
            f'<line class="axis" x1="{x0}" y1="{y + 18}" x2="{x0 + span}" y2="{y + 18}"/>',
            f'<rect class="raw" x="{x0}" y="{y}" width="{raw_w:.2f}" height="14"/>',
            f'<rect class="candidate" x="{x0}" y="{y + 22}" width="{candidate_w:.2f}" height="14"/>',
            f'<line class="limit" x1="{lx:.2f}" y1="{y - 3}" x2="{lx:.2f}" y2="{y + 39}"/>',
            f'<text class="value" x="{min(x0 + candidate_w + 6, 850):.2f}" y="{y + 34}">{ratio:.3f}x</text>',
        ))
        y += 68
    parts.extend((
        '<rect class="raw" x="310" y="438" width="18" height="12"/><text class="sub" x="336" y="449">Raw (1.0x)</text>',
        '<rect class="candidate" x="440" y="438" width="18" height="12"/><text class="sub" x="466" y="449">Projected spread</text>',
        '<line class="limit" x1="625" y1="444" x2="647" y2="444"/><text class="sub" x="655" y="449">Gate limit</text>',
        '</svg>',
    ))
    return "\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("aggregate_json", type=Path)
    parser.add_argument("output_svg", type=Path)
    args = parser.parse_args()
    raw, candidate = load_summary(args.aggregate_json)
    args.output_svg.parent.mkdir(parents=True, exist_ok=True)
    args.output_svg.write_text(make_svg(raw, candidate), encoding="utf-8")


if __name__ == "__main__":
    main()
