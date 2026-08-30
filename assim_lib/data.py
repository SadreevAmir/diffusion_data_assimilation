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
        elif name == "hour":
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
    for path in paths:
        arr = np.load(path)
        if combined is None or combined.shape != arr.shape:
            combined = np.full(arr.shape, np.nan, dtype=np.float32)
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

    def __init__(self, config, split: str = "train"):
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
        self.hour_index = int(self.config.get("target_hour_index", 23))
        self.hour_mode = str(self.config.get("hour_mode", "fixed"))
        if self.hour_mode not in ("fixed", "all"):
            raise ValueError(f"Unknown hour_mode={self.hour_mode!r}; expected 'fixed' or 'all'")
        self.hours_per_day = 24 if self.hour_mode == "all" else 1
        self.calendar_features = tuple(self.config.get("calendar_features", ()))
        unexpected_calendar_features = set(self.calendar_features) - {"day_of_year", "hour"}
        if unexpected_calendar_features:
            raise ValueError(f"Unknown calendar_features: {sorted(unexpected_calendar_features)}")
        self.observed_channels = [int(v) for v in self.config.get("observed_channels", [0])]
        self.assimilation_range = int(self.config.get("assimilation_range", 1))
        self.obs_shift = max(0, self.assimilation_range - 1)
        split_seed_offset = {"train": 0, "valid": 10000, "test": 20000}.get(split, 30000)
        self.seed = int(self.config.get("seed", 1234)) + split_seed_offset
        self.resample_observation_masks_each_epoch = bool(
            self.config.get("resample_observation_masks_each_epoch", False)
        )
        self._epoch = torch.zeros((), dtype=torch.int64).share_memory_()
        self.background_strategy = str(self.config.get("background_strategy", "indexed"))
        if self.background_strategy not in ("indexed", "year_ago_jitter", "calendar_year_ago"):
            raise ValueError(
                f"Unknown background_strategy={self.background_strategy!r}; "
                "expected 'indexed', 'year_ago_jitter', or 'calendar_year_ago'"
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
        if len(self.back_data) == 0 or len(self.obs_data) == 0:
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
        self.calendar_pairs = (
            self._build_calendar_pairs() if self.background_strategy == "calendar_year_ago" else []
        )
        self.base_valid_mask = self._load_base_valid_mask()
        self._validate_observation_mask_config()

    def _build_calendar_pairs(self):
        pairs = []
        for target_record in self.obs_data:
            if not self.target_start <= target_record.date <= self.target_end:
                continue
            background_date = previous_calendar_date(target_record.date)
            if background_date is None or not self.background_start <= background_date <= self.background_end:
                continue
            background_record = self.records_by_date.get(background_date)
            if background_record is not None:
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
        sral_dir = self.config.get("sral_dir")
        if not sral_dir:
            return {}
        sral_path = Path(sral_dir)
        if not sral_path.is_dir():
            return {}
        out: dict[date, list[Path]] = {}
        for path in sorted(sral_path.glob("*.npy")):
            parsed = parse_date(path.name)
            if parsed is not None:
                out.setdefault(parsed, []).append(path)
        return out

    def _num_days(self) -> int:
        if self.background_strategy == "calendar_year_ago":
            return len(self.calendar_pairs)
        if self.background_strategy == "year_ago_jitter":
            return max(len(self.obs_data) - self.obs_shift, 0)
        return len(self.back_data)

    def __len__(self):
        return self._num_days() * self.hours_per_day

    @property
    def conditioned_input_channels(self) -> int:
        field_channels = len(self.indices)
        static_channels = 1 + 2 * len(self.calendar_features)
        return 5 * field_channels + 2 + static_channels

    def provenance(self) -> dict:
        if self.background_strategy == "calendar_year_ago":
            pair_lines = [
                f"{background.date.isoformat()},{target.date.isoformat()}\n"
                for background, target in self.calendar_pairs
            ]
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
        }

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
        if self.background_strategy == "calendar_year_ago":
            back_record, target_record = self.calendar_pairs[day_idx]
            background_offset_days = (target_record.date - back_record.date).days
        else:
            target_record = self.obs_data[day_idx + self.obs_shift]
            back_record, background_offset_days = self._select_background_record(
                target_record.date, idx, day_idx
            )
        background, _ = prepare_model_field(
            field_at_hour(back_record.path, hour, self.indices),
            None,
            self.means,
            self.stds,
            self.padding_values,
            self.image_size,
        )
        truth, _ = prepare_model_field(
            field_at_hour(target_record.path, hour, self.indices),
            None,
            self.means,
            self.stds,
            self.padding_values,
            self.image_size,
        )
        valid_mask = self.base_valid_mask.clone()

        mask_config = self.config.get("observation_mask", {})
        mask_kind = mask_config.get("kind", "generated_track")
        if mask_kind == "sral_tracks":
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
        return {
            "truth": truth,
            "background": background,
            "obs_values": obs_values,
            "obs_mask": obs_mask,
            "valid_mask": valid_mask,
            "water_mask": torch.cat(static_conditions, dim=0),
            "meta": {
                "case_id": f"{target_record.date.isoformat()}_h{hour:02d}",
                "background_date": back_record.date.isoformat(),
                "target_date": target_record.date.isoformat(),
                "background_offset_days": int(background_offset_days),
                "background_strategy": self.background_strategy,
                "calendar_feature_values": list(feature_values),
                "hour": hour,
                "background_path": str(back_record.path),
                "target_path": str(target_record.path),
                "split": self.split,
                **obs_diagnostics,
            },
        }


DATASETS = {
    M2MForecastDataset.name: M2MForecastDataset,
}


def build_dataset(config, split: str = "train") -> Dataset:
    dataset_name = config["dataset_name"]
    if dataset_name not in DATASETS:
        raise KeyError(f"Unknown dataset_name: {dataset_name}")
    return DATASETS[dataset_name](config, split=split)
