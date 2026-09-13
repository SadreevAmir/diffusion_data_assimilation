"""Train-only CPU statistic for the A*H proxy used by geometry-score scaling."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np


ROOT = Path("/mnt/sciml/data_assimilation")
PREDS = ROOT / "da_arctic_2015-2024_v0.1/preds"
WATER = ~np.load(ROOT / "land_mask.npy")
LEADS = (0, 3, 6, 9)


def path(day: date) -> Path:
    assert date(2016, 1, 1) <= day <= date(2021, 12, 31)
    return PREDS / f"ocean+atmosphere_24_{day.isoformat()}.npy"


anchors = []
for year in range(2016, 2022):
    for month in range(1, 13):
        for day_of_month in (8, 22):
            anchor = date(year, month, day_of_month)
            if all(path(anchor + timedelta(days=lead)).is_file() for lead in LEADS):
                anchors.append(anchor)

count = 0
total = 0.0
total_square = 0.0
for anchor in anchors:
    for lead in LEADS:
        raw = np.load(path(anchor + timedelta(days=lead)), mmap_mode="r")[:2, 23]
        sic = np.asarray(raw[0], dtype=np.float64)
        sit = np.asarray(raw[1], dtype=np.float64)
        paired_open = np.isnan(sic) & np.isnan(sit)
        if np.any(np.isnan(sic) != np.isnan(sit)):
            raise ValueError("SIC/SIT missingness mismatch")
        if np.any(np.isinf(sic)) or np.any(np.isinf(sit)):
            raise ValueError("SIC/SIT contains infinity")
        product = np.where(paired_open, 0.0, sic * sit)[WATER]
        if not np.all(np.isfinite(product)):
            raise FloatingPointError("A*H contains NaN/Inf on water")
        count += product.size
        total += float(product.sum(dtype=np.float64))
        total_square += float(np.square(product).sum(dtype=np.float64))

mean = total / count
variance = max(total_square / count - mean * mean, 0.0)
print(
    json.dumps(
        {
            "schema_version": "train_geometry_product_scale_v1",
            "split": "train-2016-2021",
            "anchor_rule": "8th_and_22nd_each_month",
            "archive_slice": 23,
            "lead_days": list(LEADS),
            "anchors": len(anchors),
            "values": count,
            "area_times_sit_mean": mean,
            "area_times_sit_population_std": variance**0.5,
            "test_2023_read": False,
        },
        indent=2,
    )
)
