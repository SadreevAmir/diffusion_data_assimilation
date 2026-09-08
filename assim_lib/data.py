from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .forecast import (
    as_date,
    field_at_hour,
    forecast_records,
    parse_date,
    records_in_range,
)
from .transforms import load_valid_mask, make_observation_tensors, pad_to_size, prepare_model_field

THREEDVAR_MAIN_200D_SPLITS = {
    "train": {
        "back_start_day": "2015-01-01",
        "back_end_day": "2020-12-31",
        "obs_start_day": "2016-01-01",
        "obs_end_day": "2021-12-31",
    },
    "valid": {
        "back_start_day": "2021-01-01",
        "back_end_day": "2021-12-31",
        "obs_start_day": "2022-01-01",
        "obs_end_day": "2022-12-31",
    },
    "test": {
        "back_start_day": "2022-01-01",
        "back_end_day": "2022-07-19",
        "obs_start_day": "2023-01-01",
        "obs_end_day": "2023-07-19",
    },
}

STRUCTURED_CONDITIONING_LAYOUT = "structured_sic_lagged_v1"
STRUCTURED_CONDITIONING_CHANNELS = 17
STRUCTURED_TRAJECTORY_CONDITIONING_LAYOUT = "structured_sic_sit_trajectory_v1"
STRUCTURED_ASSIMILATION_CONDITIONING_LAYOUT = "structured_sic_sit_assimilation_v1"
STRUCTURED_DYNAMICS_CONDITIONING_LAYOUT = "structured_sic_sit_dynamics_v1"
STRUCTURED_SIC_SIT_LAYOUTS = {
    STRUCTURED_TRAJECTORY_CONDITIONING_LAYOUT,
    STRUCTURED_ASSIMILATION_CONDITIONING_LAYOUT,
    STRUCTURED_DYNAMICS_CONDITIONING_LAYOUT,
}


def previous_calendar_date(value: date) -> date | None:
    """Return the same month/day in the preceding year.

    February 29 has no exact calendar counterpart in a non-leap preceding
    year and is deliberately omitted instead of being silently shifted.
    """
    try:
        return value.replace(year=value.year - 1)
    except ValueError:
        return None


def calendar_feature_values(target_date: date, hour: int, names: Iterable[str]) -> tuple[float, ...]:
    """Build cyclic calendar features without a post-February leap-year phase shift."""
    values: list[float] = []
    for name in names:
        if name == "day_of_year":
            if target_date.month == 2 and target_date.day == 29:
                day_index = 58.5
            else:
                reference = date(2001, target_date.month, target_date.day)
                day_index = float((reference - date(2001, 1, 1)).days)
            angle = 2.0 * np.pi * day_index / 365.0
        elif name in {"hour", "time_index"}:
            angle = 2.0 * np.pi * (int(hour) % 24) / 24.0
        else:
            raise ValueError(f"Unknown calendar feature: {name!r}")
        values.extend((float(np.sin(angle)), float(np.cos(angle))))
    return tuple(values)


def validate_temporal_split_protocol(config: Mapping) -> None:
    """Enforce the temporal split used by the 3D-Var model-to-model suite."""
    protocol = config.get("split_protocol")
    if protocol is None:
        return
    if protocol != "3dvar_main_200d":
        raise ValueError(f"Unknown split_protocol={protocol!r}")

    mismatches = {}
    for split, expected in THREEDVAR_MAIN_200D_SPLITS.items():
        actual = config.get(split)
        if not isinstance(actual, Mapping):
            mismatches[split] = (actual, expected)
            continue
        for key, expected_value in expected.items():
            if actual.get(key) != expected_value:
                mismatches[f"{split}.{key}"] = (actual.get(key), expected_value)
    if mismatches:
        raise ValueError(f"Data split does not match the 3D-Var train/validation/test protocol: {mismatches}")


def _sral_transform(arr: np.ndarray, index: int) -> np.ndarray:
    if arr.ndim == 2:
        return arr.astype(np.float32, copy=False)
    if arr.ndim != 3:
        raise ValueError(f"Expected SRAL array [C,H,W] or [H,W], got {arr.shape}")

    if index == 1:
        out = np.round(arr[1].astype(np.float32, copy=True), decimals=0)
        out[out == 2.0] = 1.0
        out[out == 3.0] = np.nan
        return out

    if index == 8:
        ice_class = np.round(arr[1].astype(np.float32, copy=True), decimals=0)
        ice_class[ice_class == 2.0] = 1.0
        ice_class[ice_class == 3.0] = 1.0
        out = arr[7].astype(np.float32, copy=True)
        out[np.isnan(ice_class)] = np.nan
        return out

    if index in (10, 11):
        ice_class = np.round(arr[1].astype(np.float32, copy=True), decimals=0)
        out = arr[7].astype(np.float32, copy=True)
        out[np.isnan(ice_class)] = np.nan
        out[ice_class == 1.0] = 1.0
        out[ice_class == 2.0] = 1.0
        return out

    if index == 13:
        return 0.5 * _sral_transform(arr, 8) + 0.5 * _sral_transform(arr, 10)

    if index == 14:
        ice_class = np.round(arr[1].astype(np.float32, copy=True), decimals=0)
        amsr = arr[7].astype(np.float32, copy=True)
        out = np.full(amsr.shape, np.nan, dtype=np.float32)
        finite_amsr = np.isfinite(amsr)
        open_agree = (ice_class == 0.0) & finite_amsr & (amsr <= 0.15)
        ice_agree = ((ice_class == 1.0) | (ice_class == 2.0)) & finite_amsr & (amsr >= 0.15)
        out[open_agree | ice_agree] = amsr[open_agree | ice_agree]
        return out

    if index == 15:
        ice_class = np.round(arr[1].astype(np.float32, copy=True), decimals=0)
        amsr = arr[7].astype(np.float32, copy=True)
        out = np.full(amsr.shape, np.nan, dtype=np.float32)
        finite_amsr = np.isfinite(amsr)
        open_water = (ice_class == 0.0) & finite_amsr
        ice = ((ice_class == 1.0) | (ice_class == 2.0)) & finite_amsr
        out[open_water] = 0.5 * amsr[open_water]
        out[ice] = 0.5 * amsr[ice] + 0.5
        return out

    raise ValueError(f"Unsupported sral_transform_index={index}")


def _combine_sral_files(paths: list[Path]) -> np.ndarray | None:
    combined = None
    expected_shape = None
    for path in paths:
        arr = np.load(path)
        if expected_shape is None:
            expected_shape = arr.shape
            combined = np.full(arr.shape, np.nan, dtype=np.float32)
        elif arr.shape != expected_shape:
            raise ValueError(
                f"SRAL files for one date have inconsistent shapes: "
                f"expected {expected_shape}, got {arr.shape} in {path}"
            )
        finite = np.isfinite(arr)
        fill = np.isnan(combined) & finite
        combined[fill] = arr[fill]
    return combined


def _generate_track_mask(
    image_size: tuple[int, int], valid_mask: torch.Tensor, n_tracks_range, rng
) -> torch.Tensor:
    height, width = image_size
    n_min, n_max = int(n_tracks_range[0]), int(n_tracks_range[1])
    n_tracks = int(rng.integers(n_min, n_max + 1)) if n_max >= n_min else n_min
    mask = np.zeros((height, width), dtype=np.float32)

    for _ in range(n_tracks):
        y0 = float(rng.uniform(0, height))
        x0 = float(rng.uniform(0, width))
        angle = float(rng.uniform(0, np.pi))
        dy = np.sin(angle)
        dx = np.cos(angle)
        if abs(dy) >= abs(dx):
            rows = np.arange(height)
            cols = np.round(x0 + (rows - y0) * dx / max(abs(dy), 1e-6) * np.sign(dy)).astype(int)
            ok = (cols >= 0) & (cols < width)
            mask[rows[ok], cols[ok]] = 1.0
        else:
            cols = np.arange(width)
            rows = np.round(y0 + (cols - x0) * dy / max(abs(dx), 1e-6) * np.sign(dx)).astype(int)
            ok = (rows >= 0) & (rows < height)
            mask[rows[ok], cols[ok]] = 1.0

    valid_2d = (
        valid_mask[0].detach().cpu().numpy() if valid_mask.ndim == 3 else valid_mask.detach().cpu().numpy()
    )
    return torch.from_numpy(mask * (valid_2d > 0)).to(dtype=torch.float32)


def _random_mask(image_size: tuple[int, int], valid_mask: torch.Tensor, density: float, rng) -> torch.Tensor:
    mask = (rng.random(image_size) < float(density)).astype(np.float32)
    valid_2d = (
        valid_mask[0].detach().cpu().numpy() if valid_mask.ndim == 3 else valid_mask.detach().cpu().numpy()
    )
    return torch.from_numpy(mask * (valid_2d > 0)).to(dtype=torch.float32)


def _make_mask(image_size, valid_mask, config, rng, mask_config=None) -> torch.Tensor:
    mask_config = config.get("observation_mask", {}) if mask_config is None else mask_config
    kind = mask_config.get("kind", "generated_track")
    if kind == "generated_track":
        return _generate_track_mask(image_size, valid_mask, mask_config.get("n_tracks_range", [1, 4]), rng)
    if kind == "random":
        return _random_mask(image_size, valid_mask, float(mask_config.get("density", 0.05)), rng)
    if kind == "uniform_points":
        valid_2d = (
            valid_mask[0].detach().cpu().numpy() > 0
            if valid_mask.ndim == 3
            else valid_mask.detach().cpu().numpy() > 0
        )
        coords = np.argwhere(valid_2d)
        mask = np.zeros(image_size, dtype=np.float32)
        if coords.size == 0:
            return torch.from_numpy(mask)
        density = float(mask_config.get("density", 0.01))
        max_points = mask_config.get("max_points")
        count = max(1, int(round(coords.shape[0] * density)))
        if max_points is not None:
            count = min(count, int(max_points))
        count = min(count, coords.shape[0])
        chosen = coords[rng.choice(coords.shape[0], size=count, replace=False)]
        mask[chosen[:, 0], chosen[:, 1]] = 1.0
        return torch.from_numpy(mask)
    raise ValueError(f"Unknown observation_mask kind: {kind}")


class M2MForecastDataset(Dataset):
    name = "M2MForecastDataset"

    def __init__(self, config, split: str = "train", *, forecast_only: bool = False):
        validate_temporal_split_protocol(config)
        if split not in config or not isinstance(config[split], Mapping):
            raise KeyError(f"Missing dataset split configuration: {split!r}")
        self.config = {**config, **config.get(split, {})}
        self.split = split
        self.indices = [int(v) for v in self.config["indices"]]
        self.means = self.config["means"]
        self.stds = self.config["stds"]
        self.padding_values = self.config["padding_values"]
        self.image_size = tuple(int(v) for v in self.config.get("image_size", [320, 256]))
        self.hour_index = int(self.config.get("target_slice_index", self.config.get("target_hour_index", 23)))
        self.hour_mode = str(self.config.get("hour_mode", "fixed"))
        if self.hour_mode not in ("fixed", "all"):
            raise ValueError(f"Unknown hour_mode={self.hour_mode!r}; expected 'fixed' or 'all'")
        self.hours_per_day = 24 if self.hour_mode == "all" else 1
        self.calendar_features = tuple(self.config.get("calendar_features", ()))
        unexpected_calendar_features = set(self.calendar_features) - {
            "day_of_year",
            "hour",
            "time_index",
        }
        if unexpected_calendar_features:
            raise ValueError(f"Unknown calendar_features: {sorted(unexpected_calendar_features)}")
        self.observed_channels = [int(v) for v in self.config.get("observed_channels", [0])]
        self.dynamic_forcing_indices = tuple(
            int(value) for value in self.config.get("dynamic_forcing_indices", ())
        )
        self.assimilation_range = int(self.config.get("assimilation_range", 1))
        self.future_horizon_days = int(self.config.get("future_horizon_days", 0))
        if self.future_horizon_days < 0:
            raise ValueError("future_horizon_days must be non-negative")
        default_leads = list(range(self.future_horizon_days + 1))
        self.trajectory_lead_days = tuple(
            int(value) for value in self.config.get("trajectory_lead_days", default_leads)
        )
        if (
            not self.trajectory_lead_days
            or any(value < 0 for value in self.trajectory_lead_days)
            or tuple(sorted(set(self.trajectory_lead_days))) != self.trajectory_lead_days
        ):
            raise ValueError(
                "trajectory_lead_days must be a non-empty increasing list of unique non-negative days"
            )
        self.obs_shift = max(0, self.assimilation_range - 1)
        split_seed_offset = {"train": 0, "valid": 10000, "test": 20000}.get(split, 30000)
        self.seed = int(self.config.get("seed", 1234)) + split_seed_offset
        self.resample_observation_masks_each_epoch = bool(
            self.config.get("resample_observation_masks_each_epoch", False)
        )
        self._epoch = torch.zeros((), dtype=torch.int64).share_memory_()
        self.background_strategy = str(self.config.get("background_strategy", "indexed"))
        if self.background_strategy not in ("indexed", "year_ago_jitter", "calendar_year_ago", "none"):
            raise ValueError(
                f"Unknown background_strategy={self.background_strategy!r}; "
                "expected 'indexed', 'year_ago_jitter', 'calendar_year_ago', or 'none'"
            )
        self.background_year_offset_days = int(self.config.get("background_year_offset_days", 365))
        self.background_jitter_days = int(self.config.get("background_jitter_days", 7))

        preds_dir = Path(self.config["dataset_dir"]) / "preds"
        lead = self.config.get("lead_time_hours", None)
        records = forecast_records(preds_dir, None if lead is None else int(lead))
        self.background_start = as_date(self.config["back_start_day"])
        self.background_end = as_date(self.config["back_end_day"])
        self.target_start = as_date(self.config["obs_start_day"])
        self.target_end = as_date(self.config["obs_end_day"])
        self.back_data = records_in_range(
            records,
            self.background_start,
            self.background_end,
        )

        obs_start = self.target_start - timedelta(days=self.obs_shift)
        obs_end = self.target_end + timedelta(days=1)
        self.obs_data = records_in_range(records, obs_start, obs_end)
        if (self.background_strategy != "none" and len(self.back_data) == 0) or len(self.obs_data) == 0:
            raise ValueError("No forecast records selected for M2M date ranges")
        indexed_range_too_short = (
            self.background_strategy == "indexed"
            and len(self.obs_data) < len(self.back_data) + self.obs_shift
        )
        if indexed_range_too_short:
            raise ValueError(
                "Observation/target date range is too short for background range and assimilation_range"
            )

        self.sral_records = self._load_sral_records()
        self.records_by_date = {record.date: record for record in records}
        self._sorted_record_dates = sorted(self.records_by_date.keys())
        structured_dynamics = (
            self.config.get("conditioning_layout") == STRUCTURED_DYNAMICS_CONDITIONING_LAYOUT
        )
        self.calendar_pairs = (
            self._build_calendar_pairs()
            if (self.background_strategy == "calendar_year_ago" or structured_dynamics) and not forecast_only
            else []
        )
        self.forecast_only = bool(forecast_only)
        self.base_valid_mask = self._load_base_valid_mask()
        self._validate_observation_mask_config()
        self._validate_structured_conditioning_config()
        self._validate_structured_sral_coverage()

    def _build_calendar_pairs(self):
        dynamics_layout = self.config.get("conditioning_layout") == STRUCTURED_DYNAMICS_CONDITIONING_LAYOUT
        pairs = []
        for target_record in self.obs_data:
            if not self.target_start <= target_record.date <= self.target_end:
                continue
            if target_record.date + timedelta(days=max(self.trajectory_lead_days)) > self.target_end:
                continue
            targets_available = all(
                self.records_by_date.get(target_record.date + timedelta(days=lead)) is not None
                for lead in self.trajectory_lead_days
            )
            if dynamics_layout:
                if targets_available:
                    # The first member is an anchor placeholder, not a background.
                    # No previous-year field participates in dynamics selection or input.
                    pairs.append((target_record, target_record))
                continue
            background_date = previous_calendar_date(target_record.date)
            if background_date is None or not self.background_start <= background_date <= self.background_end:
                continue
            background_record = self.records_by_date.get(background_date)
            trajectory_available = all(
                self.records_by_date.get(target_record.date + timedelta(days=lead)) is not None
                and previous_calendar_date(target_record.date + timedelta(days=lead)) is not None
                and self.records_by_date.get(
                    previous_calendar_date(target_record.date + timedelta(days=lead))
                )
                is not None
                for lead in self.trajectory_lead_days
            )
            if background_record is not None and trajectory_available:
                pairs.append((background_record, target_record))
        if not pairs:
            raise ValueError("No exact previous-calendar-year background/target pairs are available")
        return pairs

    def set_epoch(self, epoch: int) -> None:
        if int(epoch) < 0:
            raise ValueError(f"epoch must be non-negative, got {epoch}")
        self._epoch.fill_(int(epoch))

    def _rng_for_index(self, idx: int, *, stream: int = 0) -> np.random.Generator:
        epoch = (
            int(self._epoch.item())
            if self.split == "train" and self.resample_observation_masks_each_epoch
            else 0
        )
        return np.random.default_rng(np.random.SeedSequence([self.seed, int(idx), epoch, int(stream)]))

    def _validate_observation_mask_config(self) -> None:
        mask_config = self.config.get("observation_mask", {})
        if mask_config.get("kind", "generated_track") != "sral_tracks":
            return

        synthetic_probability = float(mask_config.get("synthetic_probability", 0.0))
        empty_probability = float(mask_config.get("empty_probability", 0.0))
        for name, value in (
            ("synthetic_probability", synthetic_probability),
            ("empty_probability", empty_probability),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"observation_mask.{name} must be within [0, 1], got {value}")
        if synthetic_probability + empty_probability > 1.0:
            raise ValueError("observation_mask synthetic_probability + empty_probability must be <= 1")

        synthetic_config = mask_config.get("synthetic", {})
        if synthetic_probability > 0.0 and not isinstance(synthetic_config, dict):
            raise ValueError("observation_mask.synthetic must be an object when synthetic_probability > 0")
        synthetic_kind = (
            synthetic_config.get("kind", "generated_track") if isinstance(synthetic_config, dict) else None
        )
        if synthetic_kind == "sral_tracks":
            raise ValueError("observation_mask.synthetic.kind cannot be 'sral_tracks'")

    def _validate_structured_conditioning_config(self) -> None:
        layout = self.config.get("conditioning_layout")
        if layout not in {
            STRUCTURED_CONDITIONING_LAYOUT,
            *STRUCTURED_SIC_SIT_LAYOUTS,
        }:
            return
        mask_config = self.config.get("observation_mask", {})
        uses_tracks = layout != STRUCTURED_DYNAMICS_CONDITIONING_LAYOUT
        expected = {
            "indices": [0, 1],
            "observed_channels": ([0, 1] if layout in STRUCTURED_SIC_SIT_LAYOUTS else [0]),
            "assimilation_range": 3 if uses_tracks else 1,
            "background_strategy": (
                "none" if layout == STRUCTURED_DYNAMICS_CONDITIONING_LAYOUT else "calendar_year_ago"
            ),
            "calendar_features": (
                ("day_of_year", "time_index")
                if layout in STRUCTURED_SIC_SIT_LAYOUTS
                else ("day_of_year", "hour")
            ),
            "mask_kind": "sral_tracks" if uses_tracks else "none",
            "sral_transform_index": 1 if uses_tracks else 11,
            "synthetic_probability": 0.0,
            "empty_probability": 0.0,
            "require_finite_model_values": (
                True if layout == STRUCTURED_DYNAMICS_CONDITIONING_LAYOUT else False
            ),
        }
        actual = {
            "indices": self.indices,
            "observed_channels": self.observed_channels,
            "assimilation_range": self.assimilation_range,
            "background_strategy": self.background_strategy,
            "calendar_features": self.calendar_features,
            "mask_kind": mask_config.get("kind", "generated_track"),
            "sral_transform_index": int(
                mask_config.get("sral_transform_index", self.config.get("sral_transform_index", 11))
            ),
            "synthetic_probability": float(mask_config.get("synthetic_probability", 0.0)),
            "empty_probability": float(mask_config.get("empty_probability", 0.0)),
            "require_finite_model_values": bool(mask_config.get("require_finite_model_values", True)),
        }
        if layout in STRUCTURED_SIC_SIT_LAYOUTS:
            expected.update(
                {
                    "source_array_dtype": "float16",
                    "model_nan_semantics": "joint_sic_sit_nan_means_open_water_zero",
                    "target_slice_index": 23,
                    "utc_time_coordinate_verified": False,
                    "time_claim_policy": "archive_date_only_no_utc_or_operational_lead_claim",
                    "forbidden_time_labels": [
                        "operational_lead",
                        "23:00_UTC",
                        "24h_issue_time",
                    ],
                    "dynamic_forcing_indices": [],
                }
            )
            actual.update(
                {
                    "source_array_dtype": self.config.get("source_array_dtype"),
                    "model_nan_semantics": self.config.get("model_nan_semantics"),
                    "target_slice_index": self.config.get("target_slice_index"),
                    "utc_time_coordinate_verified": self.config.get("utc_time_coordinate_verified"),
                    "time_claim_policy": self.config.get("time_claim_policy"),
                    "forbidden_time_labels": self.config.get("forbidden_time_labels"),
                    "dynamic_forcing_indices": self.config.get("dynamic_forcing_indices"),
                }
            )
        if layout == STRUCTURED_TRAJECTORY_CONDITIONING_LAYOUT:
            expected.update(
                {
                    "trajectory_lead_days": (0, 1, 2, 3),
                    "trajectory_semantics": "consecutive_daily_archive_snapshots",
                }
            )
            actual.update(
                {
                    "trajectory_lead_days": self.trajectory_lead_days,
                    "trajectory_semantics": self.config.get("trajectory_semantics"),
                }
            )
        elif layout == STRUCTURED_ASSIMILATION_CONDITIONING_LAYOUT:
            expected.update(
                {
                    "trajectory_lead_days": (0,),
                    "trajectory_semantics": "analysis_snapshot_only",
                }
            )
            actual.update(
                {
                    "trajectory_lead_days": self.trajectory_lead_days,
                    "trajectory_semantics": self.config.get("trajectory_semantics"),
                }
            )
        elif layout == STRUCTURED_DYNAMICS_CONDITIONING_LAYOUT:
            expected["dynamic_forcing_indices"] = list(self.dynamic_forcing_indices)
            expected.update(
                {
                    "trajectory_lead_days": (3, 6, 9),
                    "trajectory_semantics": "state_only_forecast_snapshots_d_plus_3_6_9",
                }
            )
            actual.update(
                {
                    "trajectory_lead_days": self.trajectory_lead_days,
                    "trajectory_semantics": self.config.get("trajectory_semantics"),
                }
            )
            if tuple(sorted(set(self.dynamic_forcing_indices))) != self.dynamic_forcing_indices or not set(
                self.dynamic_forcing_indices
            ).issubset({6, 7, 13, 14}):
                raise ValueError(
                    "dynamics forcing indices must be an increasing unique subset of [6,7,13,14]"
                )
        mismatches = {
            key: (actual[key], expected_value)
            for key, expected_value in expected.items()
            if actual[key] != expected_value
        }
        if mismatches:
            raise ValueError(f"{layout} has an incompatible data protocol: {mismatches}")
        if self.future_horizon_days != max(self.trajectory_lead_days):
            raise ValueError("future_horizon_days must equal max(trajectory_lead_days)")

    def _load_base_valid_mask(self) -> torch.Tensor:
        channels = len(self.indices)
        if "mask_path" not in self.config or not self.config["mask_path"]:
            return torch.ones((channels, *self.image_size), dtype=torch.float32)

        raw = np.load(self.config["mask_path"])
        raw = np.asarray(raw, dtype=np.float32)
        if self.config.get("mask_true_is_invalid", True):
            raw = 1.0 - raw
        return load_valid_mask(raw, channels, self.image_size)

    def _load_sral_records(self) -> dict[date, list[Path]]:
        requires_sral = (
            self.config.get("observation_mask", {}).get("kind", "generated_track") == "sral_tracks"
        )
        sral_dir = self.config.get("sral_dir")
        if not sral_dir:
            if requires_sral:
                raise ValueError("sral_tracks conditioning requires an explicit sral_dir")
            return {}
        sral_path = Path(sral_dir)
        if not sral_path.is_dir():
            if requires_sral:
                raise ValueError(f"SRAL geometry directory is missing: {sral_path}")
            return {}
        out: dict[date, list[Path]] = {}
        files = sorted(sral_path.glob("*.npy"))
        if requires_sral and not files:
            raise ValueError(f"SRAL geometry directory contains no .npy files: {sral_path}")
        expected_shape = None
        transform_index = int(
            self.config.get("observation_mask", {}).get(
                "sral_transform_index", self.config.get("sral_transform_index", 11)
            )
        )
        for path in files:
            parsed = parse_date(path.name)
            if parsed is None:
                raise ValueError(f"SRAL filename has no unambiguous date: {path}")
            array = np.load(path, mmap_mode="r")
            if array.ndim != 3:
                raise ValueError(f"SRAL source must be [C,H,W], got {array.shape} in {path}")
            if transform_index >= array.shape[0]:
                raise ValueError(
                    f"SRAL transform index {transform_index} exceeds {array.shape[0]} channels in {path}"
                )
            if array.shape[-2] > self.image_size[0] or array.shape[-1] > self.image_size[1]:
                raise ValueError(
                    f"SRAL geometry {array.shape[-2:]} exceeds image size {self.image_size} in {path}"
                )
            if expected_shape is None:
                expected_shape = array.shape
            elif array.shape != expected_shape:
                raise ValueError(
                    f"SRAL archive shape drift: expected {expected_shape}, got {array.shape} in {path}"
                )
            out.setdefault(parsed, []).append(path)
        return out

    def _validate_structured_sral_coverage(self) -> None:
        self.structured_sral_availability = {}
        if self.config.get("conditioning_layout") not in {
            STRUCTURED_TRAJECTORY_CONDITIONING_LAYOUT,
            STRUCTURED_ASSIMILATION_CONDITIONING_LAYOUT,
        }:
            return
        for lag in range(3):
            candidate_dates = set()
            for _, target_record in self.calendar_pairs:
                value_date = target_record.date - timedelta(days=lag)
                needs_lag_background = (
                    self.config.get("conditioning_layout") == STRUCTURED_TRAJECTORY_CONDITIONING_LAYOUT
                )
                background_date = previous_calendar_date(value_date)
                if value_date in self.records_by_date and (
                    not needs_lag_background
                    or (background_date is not None and background_date in self.records_by_date)
                ):
                    candidate_dates.add(value_date)
            available_dates = candidate_dates & set(self.sral_records)
            if candidate_dates and not available_dates:
                raise ValueError(
                    f"SRAL observation law is empty for split={self.split} lag={lag}: "
                    f"{len(candidate_dates)} candidate dates and zero available files"
                )
            missing_dates = candidate_dates - available_dates
            self.structured_sral_availability[f"lag{lag}"] = {
                "candidate_date_count": len(candidate_dates),
                "available_file_date_count": len(available_dates),
                "missing_date_count": len(missing_dates),
                "missing_fraction": (len(missing_dates) / len(candidate_dates) if candidate_dates else None),
            }

    @classmethod
    def build_structured_forecast_item(cls, config: Mapping, target_date: str | date) -> dict:
        """Build truth-free d..d+3 inputs from background trajectory and past/current tracks."""
        anchor = as_date(target_date)
        background_dates = [
            previous_calendar_date(anchor + timedelta(days=lead))
            for lead in range(int(config.get("future_horizon_days", 0)) + 1)
        ]
        if any(value is None for value in background_dates):
            raise ValueError("forecast horizon has no exact previous-calendar-year background date")
        inference_config = dict(config)
        inference_config["inference"] = {
            "back_start_day": min(background_dates).isoformat(),
            "back_end_day": max(background_dates).isoformat(),
            "obs_start_day": anchor.isoformat(),
            "obs_end_day": anchor.isoformat(),
        }
        dataset = cls(inference_config, split="inference", forecast_only=True)
        return dataset._structured_forecast_item(anchor)

    def _structured_forecast_item(self, target_date: date) -> dict:
        if self.config.get("conditioning_layout") != STRUCTURED_TRAJECTORY_CONDITIONING_LAYOUT:
            raise ValueError("truth-free forecast builder requires structured trajectory conditioning")
        hour = self.hour_index
        background_records = []
        for lead in range(self.future_horizon_days + 1):
            background_date = previous_calendar_date(target_date + timedelta(days=lead))
            record = self.records_by_date.get(background_date) if background_date else None
            if record is None:
                raise FileNotFoundError(f"missing exact previous-calendar-year background for d+{lead}")
            background_records.append(record)
        raw_backgrounds = [field_at_hour(record.path, hour, self.indices) for record in background_records]
        for record, raw in zip(background_records, raw_backgrounds, strict=True):
            self._validate_structured_raw_field(raw, record.path)
        normalized_backgrounds = [
            prepare_model_field(
                raw,
                None,
                self.means,
                self.stds,
                self.padding_values,
                self.image_size,
            )[0]
            for raw in raw_backgrounds
        ]
        background = torch.cat(normalized_backgrounds, dim=0)
        valid_mask = self.base_valid_mask.clone()
        condition, obs_values, obs_mask, flow_mask, lag0_physical, diagnostics = (
            self._structured_trajectory_conditioning(target_date, hour, background, valid_mask)
        )
        physical_backgrounds = []
        padding = torch.as_tensor(self.padding_values, dtype=torch.float32).view(-1, 1, 1)
        for raw in raw_backgrounds:
            physical = torch.as_tensor(raw, dtype=torch.float32)
            physical = torch.where(torch.isfinite(physical), physical, padding)
            physical_backgrounds.append(
                pad_to_size(physical, self.image_size, fill_value=self.padding_values)
            )
        lag0_mask = (1.0 - flow_mask[:1]) * valid_mask[:1]
        feature_values = calendar_feature_values(target_date, hour, self.calendar_features)
        return {
            "background": background,
            "obs_values": obs_values,
            "obs_mask": obs_mask,
            "valid_mask": valid_mask,
            "water_mask": torch.cat(
                [
                    valid_mask[:1],
                    *[
                        torch.full((1, *self.image_size), value, dtype=torch.float32)
                        for value in feature_values
                    ],
                ],
                dim=0,
            ),
            "structured_conditioning": condition,
            "structured_flow_mask": flow_mask,
            "structured_lag0_mask": lag0_mask,
            "structured_lag0_physical_values": lag0_physical,
            "structured_physical_background": torch.cat(physical_backgrounds, dim=0),
            "meta": {
                "case_id": f"{target_date.isoformat()}_slice{hour:02d}_forecast",
                "target_date": target_date.isoformat(),
                "archive_slice_index": hour,
                "trajectory_semantics": self.config["trajectory_semantics"],
                "background_trajectory_paths": [str(record.path) for record in background_records],
                "target_trajectory_paths": [],
                "truth_free_forecast_builder": True,
                **diagnostics,
            },
        }

    def _num_days(self) -> int:
        if self.background_strategy == "calendar_year_ago" or (
            self.config.get("conditioning_layout") == STRUCTURED_DYNAMICS_CONDITIONING_LAYOUT
        ):
            return len(self.calendar_pairs)
        if self.background_strategy == "year_ago_jitter":
            return max(len(self.obs_data) - self.obs_shift, 0)
        return len(self.back_data)

    def __len__(self):
        return self._num_days() * self.hours_per_day

    @property
    def conditioned_input_channels(self) -> int:
        layout = self.config.get("conditioning_layout")
        if layout in {
            STRUCTURED_TRAJECTORY_CONDITIONING_LAYOUT,
            STRUCTURED_ASSIMILATION_CONDITIONING_LAYOUT,
        }:
            trajectory_steps = len(self.trajectory_lead_days)
            state_channels = 4 * trajectory_steps
            condition_channels = (
                3 * trajectory_steps + 3 * 5 + 1 + 4
                if layout == STRUCTURED_TRAJECTORY_CONDITIONING_LAYOUT
                else 3 + 3 * 3 + 1 + 4
            )
            return state_channels + 2 + condition_channels
        if layout == STRUCTURED_DYNAMICS_CONDITIONING_LAYOUT:
            state_channels = 4 * len(self.trajectory_lead_days)
            # Initial SIC/SIT, valid-ocean mask, forcing values and per-field
            # availability masks, plus four cyclic calendar channels.
            return state_channels + 2 + 2 + 1 + 2 * len(self.dynamic_forcing_indices) + 4
        if self.config.get("conditioning_layout") == STRUCTURED_CONDITIONING_LAYOUT:
            # Four structured state coordinates, two xy coordinates, and the
            # immutable 17-channel condition tensor defined below.
            return 4 + 2 + STRUCTURED_CONDITIONING_CHANNELS
        field_channels = len(self.indices)
        static_channels = 1 + 2 * len(self.calendar_features)
        return 5 * field_channels + 2 + static_channels

    def provenance(self) -> dict:
        structured_layout = self.config.get("conditioning_layout") in STRUCTURED_SIC_SIT_LAYOUTS
        if self.background_strategy == "calendar_year_ago" or structured_layout:
            pair_lines = []
            for background, target in self.calendar_pairs:
                trajectory = ",".join(
                    (target.date + timedelta(days=lead)).isoformat() for lead in self.trajectory_lead_days
                )
                pair_lines.append(
                    f"{self.background_strategy},{background.date.isoformat()},"
                    f"{target.date.isoformat()},{trajectory}\n"
                )
            target_candidates = sum(
                self.target_start <= record.date <= self.target_end for record in self.obs_data
            )
            first_target = self.calendar_pairs[0][1].date.isoformat()
            last_target = self.calendar_pairs[-1][1].date.isoformat()
        else:
            pair_lines = []
            target_candidates = self._num_days()
            first_target = None
            last_target = None
        return {
            "split": self.split,
            "background_strategy": self.background_strategy,
            "num_pairs": self._num_days(),
            "target_candidates": int(target_candidates),
            "dropped_target_dates": int(target_candidates - self._num_days()),
            "first_target_date": first_target,
            "last_target_date": last_target,
            "pair_manifest_sha256": (
                hashlib.sha256("".join(pair_lines).encode("utf-8")).hexdigest() if pair_lines else None
            ),
            "calendar_features": list(self.calendar_features),
            "resample_observation_masks_each_epoch": self.resample_observation_masks_each_epoch,
            "future_horizon_days": self.future_horizon_days,
            "trajectory_lead_days": list(self.trajectory_lead_days),
            "target_slice_index": self.hour_index,
            "utc_time_coordinate_verified": self.config.get("utc_time_coordinate_verified"),
            "trajectory_semantics": self.config.get("trajectory_semantics", "unspecified"),
            "time_claim_policy": self.config.get("time_claim_policy", "unspecified"),
            "source_array_dtype": self.config.get("source_array_dtype", "unspecified"),
            "model_nan_semantics": self.config.get("model_nan_semantics", "unspecified"),
            "sral_source": {
                "declared_root": self.config.get("sral_dir"),
                "resolved_root": (
                    str(Path(self.config["sral_dir"]).resolve()) if self.config.get("sral_dir") else None
                ),
                "dated_record_count": len(self.sral_records),
                "file_count": sum(len(paths) for paths in self.sral_records.values()),
            },
            "sral_availability_by_lag": self.structured_sral_availability,
        }

    def validate_structured_sral_audit_contract(self, audit: Mapping) -> None:
        """Match runtime source inventory/availability to the admitted full audit."""
        if self.config.get("conditioning_layout") not in {
            STRUCTURED_TRAJECTORY_CONDITIONING_LAYOUT,
            STRUCTURED_ASSIMILATION_CONDITIONING_LAYOUT,
        }:
            return
        expected_source = audit["sral_provenance"]
        actual_source = self.provenance()["sral_source"]
        for key in ("resolved_root", "dated_record_count", "file_count"):
            if actual_source[key] != expected_source[key]:
                raise ValueError(f"runtime SRAL source {key} differs from archive audit")
        expected_lags = expected_source["availability_by_split_and_lag"][self.split]
        for lag, actual in self.structured_sral_availability.items():
            expected = expected_lags[lag]
            comparisons = {
                "candidate_date_count": expected["candidate_date_count"],
                "available_file_date_count": expected["available_file_date_count"],
                "missing_date_count": expected["missing_date_count"],
            }
            if actual != {**comparisons, "missing_fraction": expected["missing_fraction"]}:
                raise ValueError(f"runtime SRAL availability differs for {self.split}/{lag}")

    def _validate_structured_raw_field(self, raw: np.ndarray, source: Path) -> None:
        """Fail closed on archive assumptions used by the structured codec."""
        raw = np.asarray(raw)
        if raw.ndim != 3 or raw.shape[0] != 2:
            raise ValueError(f"structured SIC/SIT field must be [2,H,W], got {raw.shape} in {source}")
        expected_dtype = self.config.get("source_array_dtype")
        if expected_dtype is not None and raw.dtype.name != str(expected_dtype):
            raise ValueError(
                f"structured source dtype {raw.dtype.name} != declared {expected_dtype!r} in {source}"
            )
        height, width = raw.shape[-2:]
        valid = self.base_valid_mask[0, :height, :width].detach().cpu().numpy() > 0
        sic_nan = np.isnan(raw[0]) & valid
        sit_nan = np.isnan(raw[1]) & valid
        if not np.array_equal(sic_nan, sit_nan):
            raise ValueError(f"mismatched SIC/SIT missingness on water in {source}")
        if np.any(np.isinf(raw)):
            raise ValueError(f"Inf is forbidden in structured SIC/SIT source {source}")

    def _resolve_index(self, idx: int) -> tuple[int, int]:
        day_idx, hour_in_day = divmod(int(idx), self.hours_per_day)
        hour = hour_in_day if self.hour_mode == "all" else self.hour_index
        return day_idx, hour

    def strided_case_indices(self, max_cases: int = 12, stride_days: int = 30) -> list[int]:
        max_cases = max(int(max_cases), 0)
        if max_cases == 0:
            return []

        indices = []
        next_date = None
        stride = timedelta(days=max(int(stride_days), 1))
        for day_idx in range(self._num_days()):
            target_date = (
                self.calendar_pairs[day_idx][1].date
                if self.background_strategy == "calendar_year_ago"
                else self.obs_data[day_idx + self.obs_shift].date
            )
            if next_date is not None and target_date < next_date:
                continue
            indices.append(day_idx * self.hours_per_day)
            next_date = target_date + stride
            if len(indices) >= max_cases:
                break
        return indices

    def _observation_diagnostics(
        self,
        mask_kind: str,
        obs_mask: torch.Tensor,
        valid_mask: torch.Tensor,
        sral_files_used: int = 0,
        empty_obs_days: int = 0,
    ) -> dict:
        if self.observed_channels:
            valid_count = float(valid_mask[self.observed_channels].sum().item())
            obs_count = int(obs_mask[self.observed_channels].sum().item())
        else:
            valid_count = 0.0
            obs_count = 0
        return {
            "mask_kind": mask_kind,
            "sral_files_used": int(sral_files_used),
            "empty_obs_days": int(empty_obs_days),
            "obs_count": obs_count,
            "observed_fraction": obs_count / max(valid_count, 1.0),
        }

    def _empty_observations(
        self,
        valid_mask: torch.Tensor,
        mask_kind: str = "empty",
        empty_obs_days: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        obs_values = torch.zeros((len(self.indices), *self.image_size), dtype=torch.float32)
        obs_mask = torch.zeros_like(obs_values)
        return (
            obs_values,
            obs_mask,
            self._observation_diagnostics(
                mask_kind,
                obs_mask,
                valid_mask,
                empty_obs_days=empty_obs_days,
            ),
        )

    def _sral_track_observations(
        self,
        target_date: date,
        hour: int,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        obs_values = torch.zeros((len(self.indices), *self.image_size), dtype=torch.float32)
        obs_mask = torch.zeros_like(obs_values)
        sral_files_used = 0
        empty_obs_days = 0
        if not self.sral_records:
            return self._empty_observations(
                valid_mask,
                mask_kind="sral_tracks",
                empty_obs_days=self.assimilation_range,
            )

        mask_config = self.config.get("observation_mask", {})
        require_finite_model_values = bool(mask_config.get("require_finite_model_values", True))
        transform_index = int(
            mask_config.get("sral_transform_index", self.config.get("sral_transform_index", 11))
        )
        observed_source_indices = [self.indices[channel] for channel in self.observed_channels]
        observed_means = [self.means[channel] for channel in self.observed_channels]
        observed_stds = [self.stds[channel] for channel in self.observed_channels]
        observed_padding = [self.padding_values[channel] for channel in self.observed_channels]
        for offset in range(self.assimilation_range):
            current_date = target_date - timedelta(days=offset)
            sral_paths = self.sral_records.get(current_date, [])
            model_record = self.records_by_date.get(current_date)
            if not sral_paths or model_record is None:
                empty_obs_days += 1
                continue

            combined_sral = _combine_sral_files(sral_paths)
            if combined_sral is None:
                empty_obs_days += 1
                continue
            transformed_sral = _sral_transform(combined_sral, transform_index)
            spatial_mask = torch.as_tensor(np.isfinite(transformed_sral).astype(np.float32))
            spatial_mask = pad_to_size(spatial_mask, self.image_size, fill_value=0.0)[0]
            spatial_mask = spatial_mask * valid_mask[0]
            if not torch.any(spatial_mask > 0):
                empty_obs_days += 1
                continue

            track_field, track_finite = prepare_model_field(
                field_at_hour(model_record.path, hour, observed_source_indices),
                None,
                observed_means,
                observed_stds,
                observed_padding,
                self.image_size,
            )
            sral_files_used += len(sral_paths)
            for observed_idx, channel in enumerate(self.observed_channels):
                fill_mask = (obs_mask[channel] == 0) & (spatial_mask > 0)
                if require_finite_model_values:
                    fill_mask = fill_mask & (track_finite[observed_idx] > 0)
                obs_values[channel][fill_mask] = track_field[observed_idx][fill_mask]
                obs_mask[channel][fill_mask] = 1.0

        return (
            obs_values,
            obs_mask,
            self._observation_diagnostics(
                "sral_tracks",
                obs_mask,
                valid_mask,
                sral_files_used=sral_files_used,
                empty_obs_days=empty_obs_days,
            ),
        )

    def _selected_sral_conditioning(self, mask_config: dict, rng) -> str:
        draw = float(rng.random())
        empty_probability = float(mask_config.get("empty_probability", 0.0))
        synthetic_probability = float(mask_config.get("synthetic_probability", 0.0))
        if draw < empty_probability:
            return "empty"
        if draw < empty_probability + synthetic_probability:
            return "synthetic"
        return "sral_tracks"

    def _synthetic_observations(
        self,
        truth: torch.Tensor,
        valid_mask: torch.Tensor,
        mask_config: dict,
        rng,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        synthetic_config = mask_config.get("synthetic", {})
        if not synthetic_config:
            synthetic_config = {"kind": "generated_track", "n_tracks_range": [1, 4]}
        synthetic_kind = synthetic_config.get("kind", "generated_track")
        spatial_mask = _make_mask(self.image_size, valid_mask, self.config, rng, mask_config=synthetic_config)
        spatial_mask = spatial_mask * valid_mask[0]
        obs_values, obs_mask = make_observation_tensors(truth, spatial_mask, self.observed_channels)
        return (
            obs_values,
            obs_mask,
            self._observation_diagnostics(
                f"synthetic_{synthetic_kind}",
                obs_mask,
                valid_mask,
            ),
        )

    def sral_spatial_mask(
        self,
        target_date: date,
        *,
        day_offsets: Iterable[int] = (0,),
        transform_index: int | None = None,
    ) -> torch.Tensor | None:
        """Return the union of SRAL footprints for explicit relative days.

        ``day_offsets=(0,)`` is the footprint on ``target_date``. Positive
        offsets address previous days, matching the observation merge order.
        Values from SRAL are deliberately ignored; only finite pixels define
        the footprint.
        """
        if not self.sral_records:
            return None
        mask = torch.zeros(self.image_size, dtype=torch.float32)
        if transform_index is None:
            transform_index = int(
                self.config.get("observation_mask", {}).get(
                    "sral_transform_index", self.config.get("sral_transform_index", 11)
                )
            )
        for raw_offset in day_offsets:
            offset = int(raw_offset)
            if offset < 0:
                raise ValueError(f"SRAL day offsets must be non-negative, got {offset}")
            current_date = target_date - timedelta(days=offset)
            sral_paths = self.sral_records.get(current_date, [])
            combined_sral = _combine_sral_files(sral_paths)
            if combined_sral is None:
                continue
            transformed_sral = _sral_transform(combined_sral, int(transform_index))
            finite = torch.as_tensor(np.isfinite(transformed_sral).astype(np.float32))
            finite = pad_to_size(finite, self.image_size, fill_value=0.0)[0]
            mask = torch.maximum(mask, finite * self.base_valid_mask[0])
        return mask if torch.any(mask > 0) else None

    def _sral_spatial_mask(self, target_date: date) -> torch.Tensor | None:
        return self.sral_spatial_mask(
            target_date,
            day_offsets=range(self.assimilation_range),
        )

    def _structured_lag_conditioning(
        self,
        target_date: date,
        hour: int,
        background: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Build the strict SIC-only lagged conditioning tensor.

        Each lag is represented separately as ``(value, innovation, mask)``.
        A missing exact calendar-year counterpart (notably February 29) makes
        that entire lag unavailable; no nearest-date substitution is allowed.
        Raw model NaNs are the archive's open-water encoding and therefore map
        to physical zero rather than deleting an otherwise valid observation.
        """
        zero = torch.zeros((1, *self.image_size), dtype=torch.float32)
        lag_channels: list[torch.Tensor] = []
        lag_available: list[bool] = []
        lag_obs_count: list[int] = []
        for lag in range(3):
            current_date = target_date - timedelta(days=lag)
            previous_date = previous_calendar_date(current_date)
            current_record = self.records_by_date.get(current_date)
            previous_record = self.records_by_date.get(previous_date) if previous_date else None
            spatial_mask = self.sral_spatial_mask(
                target_date,
                day_offsets=(lag,),
                transform_index=1,
            )
            if current_record is None or previous_record is None or spatial_mask is None:
                lag_channels.extend((zero.clone(), zero.clone(), zero.clone()))
                lag_available.append(False)
                lag_obs_count.append(0)
                continue

            current_sic, _ = prepare_model_field(
                field_at_hour(current_record.path, hour, [self.indices[0]]),
                None,
                [self.means[0]],
                [self.stds[0]],
                [self.padding_values[0]],
                self.image_size,
            )
            previous_sic, _ = prepare_model_field(
                field_at_hour(previous_record.path, hour, [self.indices[0]]),
                None,
                [self.means[0]],
                [self.stds[0]],
                [self.padding_values[0]],
                self.image_size,
            )
            mask = ((spatial_mask > 0) & (valid_mask[:1] > 0)).to(torch.float32)
            observed = torch.where(mask > 0, current_sic, zero)
            innovation = torch.where(mask > 0, current_sic - previous_sic, zero)
            lag_channels.extend((observed, innovation, mask))
            lag_available.append(True)
            lag_obs_count.append(int(mask.sum().item()))

        feature_values = calendar_feature_values(target_date, hour, self.calendar_features)
        calendar_channels = [
            torch.full((1, *self.image_size), value, dtype=torch.float32) for value in feature_values
        ]
        background_condition = torch.where(valid_mask > 0, background, torch.zeros_like(background))
        condition = torch.cat(
            [
                background_condition,
                valid_mask[:1],
                *lag_channels,
                valid_mask[:1],
                *calendar_channels,
            ],
            dim=0,
        )
        if condition.shape[0] != STRUCTURED_CONDITIONING_CHANNELS:
            raise RuntimeError(
                "structured conditioning channel invariant failed: "
                f"expected {STRUCTURED_CONDITIONING_CHANNELS}, got {condition.shape[0]}"
            )

        obs_values = torch.zeros((len(self.indices), *self.image_size), dtype=torch.float32)
        obs_mask = torch.zeros_like(obs_values)
        obs_values[0] = lag_channels[0][0]
        obs_mask[0] = lag_channels[2][0]
        diagnostics = self._observation_diagnostics(
            "structured_sral_tracks",
            obs_mask,
            valid_mask,
            empty_obs_days=sum(not available for available in lag_available),
        )
        diagnostics.update(
            {
                "lag_available": lag_available,
                "lag_obs_count": lag_obs_count,
                "structured_conditioning_channels": STRUCTURED_CONDITIONING_CHANNELS,
            }
        )
        return condition, obs_values, obs_mask, diagnostics

    def _structured_trajectory_conditioning(
        self,
        target_date: date,
        hour: int,
        background_trajectory: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict,
    ]:
        zero_field = torch.zeros((2, *self.image_size), dtype=torch.float32)
        zero_mask = torch.zeros((1, *self.image_size), dtype=torch.float32)
        lag_channels: list[torch.Tensor] = []
        lag_available: list[bool] = []
        lag_obs_count: list[int] = []
        lag_zero_mask = None
        lag0_physical = zero_field.clone()
        for lag in range(3):
            value_date = target_date - timedelta(days=lag)
            background_date = previous_calendar_date(value_date)
            value_record = self.records_by_date.get(value_date)
            background_record = self.records_by_date.get(background_date) if background_date else None
            if value_record is None or background_record is None or value_date not in self.sral_records:
                lag_channels.extend((zero_field.clone(), zero_field.clone(), zero_mask.clone()))
                lag_available.append(False)
                lag_obs_count.append(0)
                if lag == 0:
                    lag_zero_mask = zero_mask.clone()
                continue
            spatial_mask = self.sral_spatial_mask(
                target_date,
                day_offsets=(lag,),
                transform_index=1,
            )
            if spatial_mask is None:
                raise ValueError(
                    f"SRAL geometry has no finite valid coverage for required lag date {value_date}"
                )
            value_raw = field_at_hour(value_record.path, hour, self.indices)
            background_raw = field_at_hour(background_record.path, hour, self.indices)
            self._validate_structured_raw_field(value_raw, value_record.path)
            self._validate_structured_raw_field(background_raw, background_record.path)
            value, _ = prepare_model_field(
                value_raw,
                None,
                self.means,
                self.stds,
                self.padding_values,
                self.image_size,
            )
            lag_background, _ = prepare_model_field(
                background_raw,
                None,
                self.means,
                self.stds,
                self.padding_values,
                self.image_size,
            )
            mask = ((spatial_mask > 0) & (valid_mask[:1] > 0)).to(torch.float32)
            expanded = mask.expand_as(value)
            observed = torch.where(expanded > 0, value, zero_field)
            innovation = torch.where(expanded > 0, value - lag_background, zero_field)
            lag_channels.extend((observed, innovation, mask))
            lag_available.append(True)
            lag_obs_count.append(int(mask.sum().item()))
            if lag == 0:
                lag_zero_mask = mask
                raw_physical = torch.as_tensor(value_raw, dtype=torch.float32)
                raw_physical = torch.where(
                    torch.isfinite(raw_physical),
                    raw_physical,
                    torch.zeros_like(raw_physical),
                )
                raw_physical = pad_to_size(raw_physical, self.image_size, fill_value=(0.0, 0.0))
                lag0_physical = torch.where(
                    mask.expand_as(raw_physical) > 0,
                    raw_physical,
                    torch.zeros_like(raw_physical),
                )

        feature_values = calendar_feature_values(target_date, hour, self.calendar_features)
        calendar_channels = [
            torch.full((1, *self.image_size), value, dtype=torch.float32) for value in feature_values
        ]
        background_channels: list[torch.Tensor] = []
        background_steps = background_trajectory.shape[0] // 2
        if background_trajectory.shape[0] != 2 * background_steps:
            raise ValueError("structured background trajectory must contain SIC/SIT pairs")
        for lead_index in range(background_steps):
            lead_background = background_trajectory[2 * lead_index : 2 * lead_index + 2]
            background_channels.extend(
                (
                    torch.where(
                        valid_mask[:1].expand_as(lead_background) > 0,
                        lead_background,
                        torch.zeros_like(lead_background),
                    ),
                    valid_mask[:1],
                )
            )
        condition = torch.cat(
            [
                *background_channels,
                *lag_channels,
                valid_mask[:1],
                *calendar_channels,
            ],
            dim=0,
        )
        expected_channels = 3 * background_steps + 15 + 1 + 4
        if condition.shape[0] != expected_channels:
            raise RuntimeError(
                f"trajectory condition expected {expected_channels} channels, got {condition.shape[0]}"
            )
        obs_values = torch.zeros((len(self.indices), *self.image_size), dtype=torch.float32)
        obs_mask = torch.zeros_like(obs_values)
        obs_values[:] = lag_channels[0]
        obs_mask[:] = lag_channels[2].expand_as(obs_mask)
        assert lag_zero_mask is not None
        trajectory_steps = len(self.trajectory_lead_days)
        flow_mask = valid_mask[:1].repeat(4 * trajectory_steps, 1, 1)
        flow_mask[:4] *= 1.0 - lag_zero_mask
        diagnostics = self._observation_diagnostics(
            "structured_model_truth_pseudo_obs_sic_sit_on_real_sral_geometry",
            obs_mask,
            valid_mask,
            empty_obs_days=sum(not available for available in lag_available),
        )
        diagnostics.update({"lag_available": lag_available, "lag_obs_count": lag_obs_count})
        return condition, obs_values, obs_mask, flow_mask, lag0_physical, diagnostics

    def _structured_assimilation_conditioning(
        self,
        target_date: date,
        hour: int,
        background: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict,
    ]:
        """Build d0 assimilation inputs without hidden lag-background fields."""
        zero_field = torch.zeros((2, *self.image_size), dtype=torch.float32)
        zero_mask = torch.zeros((1, *self.image_size), dtype=torch.float32)
        lag_channels: list[torch.Tensor] = []
        lag_available: list[bool] = []
        lag_obs_count: list[int] = []
        lag0_mask = zero_mask.clone()
        lag0_physical = zero_field.clone()
        for lag in range(3):
            value_date = target_date - timedelta(days=lag)
            value_record = self.records_by_date.get(value_date)
            if value_record is None or value_date not in self.sral_records:
                lag_channels.extend((zero_field.clone(), zero_mask.clone()))
                lag_available.append(False)
                lag_obs_count.append(0)
                continue
            spatial_mask = self.sral_spatial_mask(target_date, day_offsets=(lag,), transform_index=1)
            if spatial_mask is None:
                raise ValueError(
                    f"SRAL geometry has no finite valid coverage for required lag date {value_date}"
                )
            value_raw = field_at_hour(value_record.path, hour, self.indices)
            self._validate_structured_raw_field(value_raw, value_record.path)
            value, _ = prepare_model_field(
                value_raw,
                None,
                self.means,
                self.stds,
                self.padding_values,
                self.image_size,
            )
            mask = ((spatial_mask > 0) & (valid_mask[:1] > 0)).to(torch.float32)
            observed = torch.where(mask.expand_as(value) > 0, value, zero_field)
            lag_channels.extend((observed, mask))
            lag_available.append(True)
            lag_obs_count.append(int(mask.sum().item()))
            if lag == 0:
                lag0_mask = mask
                raw_physical = torch.as_tensor(value_raw, dtype=torch.float32)
                raw_physical = torch.where(
                    torch.isfinite(raw_physical),
                    raw_physical,
                    torch.zeros_like(raw_physical),
                )
                raw_physical = pad_to_size(
                    raw_physical,
                    self.image_size,
                    fill_value=(0.0, 0.0),
                )
                lag0_physical = torch.where(
                    mask.expand_as(raw_physical) > 0,
                    raw_physical,
                    zero_field,
                )

        feature_values = calendar_feature_values(target_date, hour, self.calendar_features)
        calendar_channels = [
            torch.full((1, *self.image_size), value, dtype=torch.float32) for value in feature_values
        ]
        background_condition = torch.where(
            valid_mask[:1].expand_as(background) > 0,
            background,
            torch.zeros_like(background),
        )
        condition = torch.cat(
            [
                background_condition,
                valid_mask[:1],
                *lag_channels,
                valid_mask[:1],
                *calendar_channels,
            ],
            dim=0,
        )
        if condition.shape[0] != 17:
            raise RuntimeError(f"assimilation conditioning expected 17 channels, got {condition.shape[0]}")
        obs_values = torch.zeros((2, *self.image_size), dtype=torch.float32)
        obs_mask = torch.zeros_like(obs_values)
        obs_values[:] = lag_channels[0]
        obs_mask[:] = lag_channels[1].expand_as(obs_mask)
        flow_mask = valid_mask[:1].repeat(4, 1, 1)
        flow_mask *= 1.0 - lag0_mask
        diagnostics = self._observation_diagnostics(
            "structured_model_truth_pseudo_obs_sic_sit_on_real_sral_geometry",
            obs_mask,
            valid_mask,
            empty_obs_days=sum(not available for available in lag_available),
        )
        diagnostics.update(
            {
                "lag_available": lag_available,
                "lag_obs_count": lag_obs_count,
                "lag_background_innovations_used": False,
                "background_used_by_model": True,
            }
        )
        return condition, obs_values, obs_mask, flow_mask, lag0_physical, diagnostics

    def _structured_dynamics_conditioning(
        self,
        target_date: date,
        hour: int,
        initial_state: torch.Tensor,
        forcing_values: torch.Tensor,
        forcing_masks: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Condition a state-only forecast on the exact physical analysis at d0.

        No previous-year background or observation-track tensor is included.
        The target contains only the declared future lead snapshots.
        """
        initial_condition = torch.where(
            valid_mask[:1].expand_as(initial_state) > 0,
            initial_state,
            torch.zeros_like(initial_state),
        )
        # In all-hour mode every sample must carry its resolved archive-slice
        # index.  Using ``self.hour_index`` here silently labelled all 24 slices
        # as slice 23 even though d0, forcing, and targets came from ``hour``.
        feature_values = calendar_feature_values(target_date, hour, self.calendar_features)
        calendar_channels = [
            torch.full((1, *self.image_size), value, dtype=torch.float32) for value in feature_values
        ]
        condition = torch.cat(
            [
                initial_condition,
                valid_mask[:1],
                forcing_values,
                forcing_masks,
                *calendar_channels,
            ],
            dim=0,
        )
        expected_channels = 7 + 2 * len(self.dynamic_forcing_indices)
        if condition.shape[0] != expected_channels:
            raise RuntimeError(
                f"dynamics conditioning expected {expected_channels} channels, got {condition.shape[0]}"
            )
        obs_values = torch.zeros((2, *self.image_size), dtype=torch.float32)
        obs_mask = torch.zeros_like(obs_values)
        flow_mask = valid_mask[:1].repeat(4 * len(self.trajectory_lead_days), 1, 1)
        diagnostics = {
            "observation_semantics": "none_state_only_dynamics",
            "trajectory_lead_days": list(self.trajectory_lead_days),
            "initial_state_is_exact_truth": True,
            "background_used_by_model": False,
        }
        return condition, obs_values, obs_mask, flow_mask, diagnostics

    def _normalized_dynamic_forcing(
        self, record, hour: int, valid_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        count = len(self.dynamic_forcing_indices)
        if count == 0:
            empty = torch.empty((0, *self.image_size), dtype=torch.float32)
            return empty, empty.clone()
        normalization = self.config.get("dynamic_forcing_stats")
        if not isinstance(normalization, Mapping):
            raise ValueError("non-empty dynamic_forcing_indices require train-only dynamic_forcing_stats")
        means = tuple(float(value) for value in normalization.get("means", ()))
        stds = tuple(float(value) for value in normalization.get("stds", ()))
        normalized_indices = tuple(int(value) for value in normalization.get("indices", ()))
        if (
            normalized_indices != self.dynamic_forcing_indices
            or len(means) != count
            or len(stds) != count
            or any(not np.isfinite(value) or value <= 0.0 for value in stds)
        ):
            raise ValueError("dynamic_forcing_stats do not match the forcing channels")
        raw = np.asarray(
            field_at_hour(record.path, hour, self.dynamic_forcing_indices),
            dtype=np.float32,
        )
        if np.any(np.isinf(raw)):
            raise ValueError(f"Inf is forbidden in dynamic forcing source {record.path}")
        finite = torch.as_tensor(np.isfinite(raw), dtype=torch.float32)
        values = torch.as_tensor(np.where(np.isfinite(raw), raw, 0.0), dtype=torch.float32)
        mean = torch.as_tensor(means, dtype=torch.float32).view(-1, 1, 1)
        std = torch.as_tensor(stds, dtype=torch.float32).view(-1, 1, 1)
        values = torch.where(finite > 0, (values - mean) / std, torch.zeros_like(values))
        values = pad_to_size(values, self.image_size, fill_value=[0.0] * count)
        masks = pad_to_size(finite, self.image_size, fill_value=[0.0] * count)
        ocean = valid_mask[:1].expand_as(values)
        masks *= ocean
        values = torch.where(masks > 0, values, torch.zeros_like(values))
        return values, masks

    def _nearest_record(self, target: date):
        if not self._sorted_record_dates:
            return None
        import bisect

        pos = bisect.bisect_left(self._sorted_record_dates, target)
        options = []
        if pos < len(self._sorted_record_dates):
            options.append(self._sorted_record_dates[pos])
        if pos > 0:
            options.append(self._sorted_record_dates[pos - 1])
        if not options:
            return None
        closest = min(options, key=lambda d: abs((d - target).days))
        return self.records_by_date[closest]

    def _select_background_record(self, target_date, idx: int, day_idx: int, rng=None):
        if self.background_strategy == "indexed":
            return self.back_data[day_idx], (target_date - self.back_data[day_idx].date).days
        base = target_date - timedelta(days=self.background_year_offset_days)
        candidates = []
        for offset in range(-self.background_jitter_days, self.background_jitter_days + 1):
            record = self.records_by_date.get(base + timedelta(days=offset))
            if record is not None:
                candidates.append(record)
        if not candidates:
            fallback = self._nearest_record(base)
            if fallback is None:
                raise FileNotFoundError(
                    f"No background record available anywhere in the archive for target {target_date}"
                )
            candidates = [fallback]
        if self.split == "train":
            rng = self._rng_for_index(idx, stream=0) if rng is None else rng
            choice = int(rng.integers(len(candidates)))
        else:
            rng = np.random.default_rng((self.seed + idx) & 0xFFFFFFFF)
            choice = int(rng.integers(len(candidates)))
        record = candidates[choice]
        return record, (target_date - record.date).days

    def __getitem__(self, idx):
        day_idx, hour = self._resolve_index(idx)
        observation_rng = self._rng_for_index(idx, stream=1)
        layout = self.config.get("conditioning_layout")
        dynamics_layout = layout == STRUCTURED_DYNAMICS_CONDITIONING_LAYOUT
        if self.background_strategy == "calendar_year_ago" or dynamics_layout:
            back_record, target_record = self.calendar_pairs[day_idx]
            background_offset_days = (target_record.date - back_record.date).days
        else:
            target_record = self.obs_data[day_idx + self.obs_shift]
            back_record, background_offset_days = self._select_background_record(
                target_record.date, idx, day_idx
            )
        trajectory_layout = layout in STRUCTURED_SIC_SIT_LAYOUTS
        trajectory_leads = self.trajectory_lead_days if trajectory_layout else (0,)
        trajectory_steps = len(trajectory_leads)
        if trajectory_layout:
            if dynamics_layout:
                # Persistence is retained only as an evaluation baseline.  It is
                # never present in structured_conditioning.
                background_records = [target_record] * trajectory_steps
            elif layout == STRUCTURED_ASSIMILATION_CONDITIONING_LAYOUT:
                background_records = [back_record]
            else:
                background_records = [
                    self.records_by_date[previous_calendar_date(target_record.date + timedelta(days=lead))]
                    for lead in trajectory_leads
                ]
        else:
            background_records = [back_record]
        target_records = [
            self.records_by_date[target_record.date + timedelta(days=lead)] for lead in trajectory_leads
        ]
        background_fields = [field_at_hour(record.path, hour, self.indices) for record in background_records]
        target_fields = [field_at_hour(record.path, hour, self.indices) for record in target_records]
        if trajectory_layout:
            for record, field in zip(background_records, background_fields, strict=True):
                self._validate_structured_raw_field(field, record.path)
            for record, field in zip(target_records, target_fields, strict=True):
                self._validate_structured_raw_field(field, record.path)
        normalized_background = [
            prepare_model_field(
                field,
                None,
                self.means,
                self.stds,
                self.padding_values,
                self.image_size,
            )[0]
            for field in background_fields
        ]
        normalized_truth = [
            prepare_model_field(
                field,
                None,
                self.means,
                self.stds,
                self.padding_values,
                self.image_size,
            )[0]
            for field in target_fields
        ]
        background = torch.cat(normalized_background, dim=0)
        truth = torch.cat(normalized_truth, dim=0)
        valid_mask = self.base_valid_mask.clone()

        mask_config = self.config.get("observation_mask", {})
        mask_kind = mask_config.get("kind", "generated_track")
        structured_conditioning = None
        structured_flow_mask = None
        structured_lag0_physical_values = None
        if dynamics_layout:
            initial_raw = field_at_hour(target_record.path, hour, self.indices)
            self._validate_structured_raw_field(initial_raw, target_record.path)
            initial_state = prepare_model_field(
                initial_raw,
                None,
                self.means,
                self.stds,
                self.padding_values,
                self.image_size,
            )[0]
            forcing_values, forcing_masks = self._normalized_dynamic_forcing(target_record, hour, valid_mask)
            (
                structured_conditioning,
                obs_values,
                obs_mask,
                structured_flow_mask,
                obs_diagnostics,
            ) = self._structured_dynamics_conditioning(
                target_record.date,
                hour,
                initial_state,
                forcing_values,
                forcing_masks,
                valid_mask,
            )
        elif layout == STRUCTURED_ASSIMILATION_CONDITIONING_LAYOUT:
            (
                structured_conditioning,
                obs_values,
                obs_mask,
                structured_flow_mask,
                structured_lag0_physical_values,
                obs_diagnostics,
            ) = self._structured_assimilation_conditioning(
                target_record.date,
                hour,
                background,
                valid_mask,
            )
        elif trajectory_layout:
            (
                structured_conditioning,
                obs_values,
                obs_mask,
                structured_flow_mask,
                structured_lag0_physical_values,
                obs_diagnostics,
            ) = self._structured_trajectory_conditioning(
                target_record.date,
                hour,
                background,
                valid_mask,
            )
        elif self.config.get("conditioning_layout") == STRUCTURED_CONDITIONING_LAYOUT:
            structured_conditioning, obs_values, obs_mask, obs_diagnostics = (
                self._structured_lag_conditioning(
                    target_record.date,
                    hour,
                    background,
                    valid_mask,
                )
            )
        elif mask_kind == "sral_tracks":
            conditioning_kind = self._selected_sral_conditioning(mask_config, observation_rng)
            if conditioning_kind == "sral_tracks":
                obs_values, obs_mask, obs_diagnostics = self._sral_track_observations(
                    target_record.date,
                    hour,
                    valid_mask,
                )
            elif conditioning_kind == "synthetic":
                obs_values, obs_mask, obs_diagnostics = self._synthetic_observations(
                    truth,
                    valid_mask,
                    mask_config,
                    observation_rng,
                )
            else:
                obs_values, obs_mask, obs_diagnostics = self._empty_observations(
                    valid_mask,
                    empty_obs_days=self.assimilation_range,
                )
        else:
            spatial_mask = _make_mask(
                self.image_size,
                valid_mask,
                self.config,
                observation_rng,
                mask_config=mask_config,
            )
            spatial_mask = spatial_mask * valid_mask[0]
            obs_values, obs_mask = make_observation_tensors(truth, spatial_mask, self.observed_channels)
            obs_diagnostics = self._observation_diagnostics(mask_kind, obs_mask, valid_mask)

        feature_values = calendar_feature_values(target_record.date, hour, self.calendar_features)
        static_conditions = [self.base_valid_mask[:1]]
        static_conditions.extend(
            torch.full((1, *self.image_size), value, dtype=torch.float32) for value in feature_values
        )
        item = {
            "truth": truth,
            "background": background,
            "obs_values": obs_values,
            "obs_mask": obs_mask,
            "valid_mask": valid_mask,
            "water_mask": torch.cat(static_conditions, dim=0),
            "meta": {
                "case_id": (
                    f"{target_record.date.isoformat()}_slice{hour:02d}"
                    if trajectory_layout
                    else f"{target_record.date.isoformat()}_t{hour:02d}"
                ),
                "background_date": back_record.date.isoformat(),
                "target_date": target_record.date.isoformat(),
                "background_offset_days": int(background_offset_days),
                "background_strategy": self.background_strategy,
                "calendar_feature_values": list(feature_values),
                "archive_slice_index": hour,
                # Legacy generic consumers still read hour/time_index.  For the
                # structured trajectory contract these are archive indices,
                # never a UTC or issue-time claim.
                "hour": hour,
                "time_index": hour,
                "utc_time_coordinate_verified": bool(self.config.get("utc_time_coordinate_verified", False)),
                "trajectory_semantics": self.config.get("trajectory_semantics", "unspecified"),
                "time_claim_policy": self.config.get("time_claim_policy", "unspecified"),
                "trajectory_archive_date_labels": [
                    *[
                        ("analysis_snapshot_d" if lead == 0 else f"forecast_archive_day_d_plus_{lead}")
                        for lead in trajectory_leads
                    ]
                ],
                "background_path": str(back_record.path),
                "target_path": str(target_record.path),
                "background_trajectory_paths": [str(record.path) for record in background_records],
                "target_trajectory_paths": [str(record.path) for record in target_records],
                "background_role": (
                    "persistence_baseline_not_model_condition" if dynamics_layout else "model_condition"
                ),
                "trajectory_lead_days": list(trajectory_leads),
                "split": self.split,
                **obs_diagnostics,
            },
        }
        if structured_conditioning is not None:
            item["structured_conditioning"] = structured_conditioning
            physical_targets = []
            physical_backgrounds = []
            padding = torch.as_tensor(self.padding_values, dtype=torch.float32).view(-1, 1, 1)
            for target_field, background_field in zip(target_fields, background_fields, strict=True):
                physical_truth = torch.as_tensor(target_field, dtype=torch.float32)
                physical_truth = torch.where(torch.isfinite(physical_truth), physical_truth, padding)
                physical_targets.append(
                    pad_to_size(physical_truth, self.image_size, fill_value=self.padding_values)
                )
                physical_background = torch.as_tensor(background_field, dtype=torch.float32)
                physical_background = torch.where(
                    torch.isfinite(physical_background), physical_background, padding
                )
                physical_backgrounds.append(
                    pad_to_size(
                        physical_background,
                        self.image_size,
                        fill_value=self.padding_values,
                    )
                )
            item["structured_physical_truth"] = torch.cat(physical_targets, dim=0)
            item["structured_physical_background"] = torch.cat(physical_backgrounds, dim=0)
        if structured_flow_mask is not None:
            item["structured_flow_mask"] = structured_flow_mask
            if dynamics_layout:
                # Sampling/evaluation APIs accept a common optional-hard-data
                # interface.  Zero masks make this an exact no-op for D.
                item["structured_lag0_mask"] = torch.zeros((1, *self.image_size), dtype=torch.float32)
                item["structured_lag0_physical_values"] = torch.zeros(
                    (2, *self.image_size), dtype=torch.float32
                )
            else:
                lag0_mask = 1.0 - structured_flow_mask[:1]
                lag0_mask *= valid_mask[:1]
                item["structured_lag0_mask"] = lag0_mask
                if structured_lag0_physical_values is None:
                    raise RuntimeError("structured lag0 physical values were not preserved")
                item["structured_lag0_physical_values"] = structured_lag0_physical_values
        return item


DATASETS = {
    M2MForecastDataset.name: M2MForecastDataset,
}


def build_dataset(config, split: str = "train") -> Dataset:
    dataset_name = config["dataset_name"]
    if dataset_name not in DATASETS:
        raise KeyError(f"Unknown dataset_name: {dataset_name}")
    return DATASETS[dataset_name](config, split=split)
