from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ForecastRecord:
    date: date
    path: Path


def parse_date(value: str) -> date | None:
    dashed = re.search(r"\d{4}-\d{2}-\d{2}", value)
    if dashed is not None:
        return datetime.strptime(dashed.group(0), "%Y-%m-%d").date()
    compact = re.search(r"\d{8}", value)
    if compact is not None:
        return datetime.strptime(compact.group(0), "%Y%m%d").date()
    return None


def as_date(value: str | date) -> date:
    return value if isinstance(value, date) else datetime.fromisoformat(value).date()


def records_in_range(
    records: list[ForecastRecord],
    start: date,
    end: date,
) -> list[ForecastRecord]:
    return [record for record in records if start <= record.date <= end]


def forecast_records(preds_dir: str | Path, lead_time_hours: int | None) -> list[ForecastRecord]:
    directory = Path(preds_dir)
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    files = sorted(directory.glob("*.npy"))
    if lead_time_hours is not None:
        prefix = f"ocean+atmosphere_{lead_time_hours}"
        files = [path for path in files if path.name.startswith(prefix)]
    records = [
        ForecastRecord(parsed, path) for path in files if (parsed := parse_date(path.name)) is not None
    ]
    if not records:
        raise FileNotFoundError(f"No dated .npy forecast files found in {directory}")
    return records


@lru_cache(maxsize=128)
def open_npy_mmap(path: str | Path) -> np.ndarray:
    return np.load(str(path), mmap_mode="r")


def field_at_hour(
    path: str | Path,
    hour_index: int,
    indices: list[int] | np.ndarray | None = None,
    *,
    copy: bool = True,
) -> np.ndarray:
    source = Path(path)
    field = open_npy_mmap(source)
    if field.ndim == 4:
        if not -field.shape[1] <= hour_index < field.shape[1]:
            raise IndexError(
                f"hour_index={hour_index} is outside the {field.shape[1]} time slices in {source}"
            )
        selected = field[:, hour_index] if indices is None else field[indices, hour_index]
    elif field.ndim == 3:
        selected = field if indices is None else field[indices]
    else:
        raise ValueError(f"Expected forecast array [V,T,H,W] or [V,H,W], got {field.shape} in {source}")
    return np.array(selected, copy=copy)
