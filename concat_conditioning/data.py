from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .transforms import load_valid_mask, make_observation_tensors, pad_to_size, prepare_model_field


@dataclass(frozen=True)
class ForecastRecord:
    date: date
    path: Path


def _parse_date(value: str) -> date | None:
    dashed = re.search(r"\d{4}-\d{2}-\d{2}", value)
    if dashed is not None:
        return datetime.strptime(dashed.group(0), "%Y-%m-%d").date()

    compact = re.search(r"\d{8}", value)
    if compact is not None:
        return datetime.strptime(compact.group(0), "%Y%m%d").date()

    return None


def _date(value: str | date) -> date:
    if isinstance(value, date):
        return value
    return datetime.fromisoformat(value).date()


def _records_in_range(records: list[ForecastRecord], start: date, end: date) -> list[ForecastRecord]:
    return [record for record in records if start <= record.date <= end]


def _forecast_records(preds_dir: Path, lead_time_hours: int | None) -> list[ForecastRecord]:
    if not preds_dir.is_dir():
        raise FileNotFoundError(preds_dir)

    files = sorted(preds_dir.glob("*.npy"))
    if lead_time_hours is not None:
        prefix = f"ocean+atmosphere_{lead_time_hours}"
        files = [path for path in files if path.name.startswith(prefix)]

    records = []
    for path in files:
        parsed = _parse_date(path.name)
        if parsed is not None:
            records.append(ForecastRecord(parsed, path))
    if not records:
        raise FileNotFoundError(f"No dated .npy forecast files found in {preds_dir}")
    return records


def _field_at_hour(path: Path, hour_index: int):
    field = np.load(path)
    if field.ndim == 4:
        return field[:, hour_index]
    if field.ndim == 3:
        return field
    raise ValueError(f"Expected forecast array [V,T,H,W] or [V,H,W], got {field.shape} in {path}")


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


def _generate_track_mask(image_size: tuple[int, int], valid_mask: torch.Tensor, n_tracks_range, rng) -> torch.Tensor:
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

    valid_2d = valid_mask[0].detach().cpu().numpy() if valid_mask.ndim == 3 else valid_mask.detach().cpu().numpy()
    return torch.from_numpy(mask * (valid_2d > 0)).to(dtype=torch.float32)


def _random_mask(image_size: tuple[int, int], valid_mask: torch.Tensor, density: float, rng) -> torch.Tensor:
    mask = (rng.random(image_size) < float(density)).astype(np.float32)
    valid_2d = valid_mask[0].detach().cpu().numpy() if valid_mask.ndim == 3 else valid_mask.detach().cpu().numpy()
    return torch.from_numpy(mask * (valid_2d > 0)).to(dtype=torch.float32)


def _make_mask(image_size, valid_mask, config, rng) -> torch.Tensor:
    mask_config = config.get("observation_mask", {})
    kind = mask_config.get("kind", "generated_track")
    if kind == "generated_track":
        return _generate_track_mask(image_size, valid_mask, mask_config.get("n_tracks_range", [1, 4]), rng)
    if kind == "random":
        return _random_mask(image_size, valid_mask, float(mask_config.get("density", 0.05)), rng)
    if kind == "uniform_points":
        valid_2d = valid_mask[0].detach().cpu().numpy() > 0 if valid_mask.ndim == 3 else valid_mask.detach().cpu().numpy() > 0
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


class SmokeM2MDataset(Dataset):
    name = "SmokeM2MDataset"

    def __init__(self, config, split: str = "train"):
        self.config = config
        self.split = split
        split_config = config.get(split, {})
        self.n_cases = int(split_config.get("n_cases", config.get("n_cases", 8)))
        self.channels = int(config.get("channels", len(config.get("fields", ["var0", "var1", "var2", "var3"]))))
        self.image_size = tuple(int(v) for v in config.get("image_size", [32, 32]))
        self.background_noise_std = float(config.get("background_noise_std", 0.25))
        self.seed = int(config.get("seed", 42)) + (10000 if split == "valid" else 0)
        self.observed_channels = [int(v) for v in config.get("observed_channels", [0])]

        valid_mask = torch.ones((self.channels, *self.image_size), dtype=torch.float32)
        self.valid_mask = valid_mask

    def __len__(self):
        return self.n_cases

    def __getitem__(self, idx):
        rng = np.random.default_rng(self.seed + idx)
        height, width = self.image_size
        yy, xx = np.meshgrid(
            np.linspace(0.0, 1.0, height, dtype=np.float32),
            np.linspace(0.0, 1.0, width, dtype=np.float32),
            indexing="ij",
        )
        fields = []
        for channel in range(self.channels):
            freq = channel + 1
            base = np.sin(freq * np.pi * xx) + np.cos(freq * np.pi * yy)
            fields.append(base.astype(np.float32))
        truth = torch.from_numpy(np.stack(fields, axis=0))
        truth = truth + torch.from_numpy(rng.normal(0.0, 0.03, size=truth.shape).astype(np.float32))
        background = truth + torch.from_numpy(
            rng.normal(0.0, self.background_noise_std, size=truth.shape).astype(np.float32)
        )
        spatial_mask = _make_mask(self.image_size, self.valid_mask, self.config, rng)
        obs_values, obs_mask = make_observation_tensors(truth, spatial_mask, self.observed_channels)
        return {
            "truth": truth,
            "background": background,
            "obs_values": obs_values,
            "obs_mask": obs_mask,
            "valid_mask": self.valid_mask,
            "meta": {"case_id": f"{self.split}_{idx:04d}", "split": self.split},
        }


class M2MForecastDataset(Dataset):
    name = "M2MForecastDataset"

    def __init__(self, config, split: str = "train"):
        self.config = {**config, **config.get(split, {})}
        self.split = split
        self.indices = [int(v) for v in self.config["indices"]]
        self.means = self.config["means"]
        self.stds = self.config["stds"]
        self.padding_values = self.config["padding_values"]
        self.image_size = tuple(int(v) for v in self.config.get("image_size", [320, 256]))
        self.hour_index = int(self.config.get("target_hour_index", 23))
        self.observed_channels = [int(v) for v in self.config.get("observed_channels", [0])]
        self.assimilation_range = int(self.config.get("assimilation_range", 1))
        self.obs_shift = max(0, self.assimilation_range - 1)
        self.seed = int(self.config.get("seed", 1234)) + (10000 if split == "valid" else 0)

        preds_dir = Path(self.config["dataset_dir"]) / "preds"
        lead = self.config.get("lead_time_hours", None)
        records = _forecast_records(preds_dir, None if lead is None else int(lead))
        self.back_data = _records_in_range(records, _date(self.config["back_start_day"]), _date(self.config["back_end_day"]))

        obs_start = _date(self.config["obs_start_day"]) - timedelta(days=self.obs_shift)
        obs_end = _date(self.config["obs_end_day"]) + timedelta(days=1)
        self.obs_data = _records_in_range(records, obs_start, obs_end)
        if len(self.back_data) == 0 or len(self.obs_data) == 0:
            raise ValueError("No forecast records selected for M2M date ranges")
        if len(self.obs_data) < len(self.back_data) + self.obs_shift:
            raise ValueError("Observation/target date range is too short for background range and assimilation_range")

        self.sral_records = self._load_sral_records()
        self.records_by_date = {record.date: record for record in records}
        self.base_valid_mask = self._load_base_valid_mask()

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
            parsed = _parse_date(path.name)
            if parsed is not None:
                out.setdefault(parsed, []).append(path)
        return out

    def __len__(self):
        return len(self.back_data)

    def _sral_track_observations(self, target_date: date, valid_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict]:
        obs_values = torch.zeros((len(self.indices), *self.image_size), dtype=torch.float32)
        obs_mask = torch.zeros_like(obs_values)
        diagnostics = {
            "mask_kind": "sral_tracks",
            "sral_files_used": 0,
            "empty_obs_days": 0,
            "obs_count": 0,
            "observed_fraction": 0.0,
        }
        if not self.sral_records:
            diagnostics["empty_obs_days"] = self.assimilation_range
            return obs_values, obs_mask, diagnostics

        mask_config = self.config.get("observation_mask", {})
        transform_index = int(mask_config.get("sral_transform_index", self.config.get("sral_transform_index", 11)))
        for offset in range(self.assimilation_range):
            current_date = target_date - timedelta(days=offset)
            sral_paths = self.sral_records.get(current_date, [])
            model_record = self.records_by_date.get(current_date)
            if not sral_paths or model_record is None:
                diagnostics["empty_obs_days"] += 1
                continue

            combined_sral = _combine_sral_files(sral_paths)
            if combined_sral is None:
                diagnostics["empty_obs_days"] += 1
                continue
            transformed_sral = _sral_transform(combined_sral, transform_index)
            spatial_mask = torch.as_tensor(np.isfinite(transformed_sral).astype(np.float32))
            spatial_mask = pad_to_size(spatial_mask, self.image_size, fill_value=0.0)[0]
            spatial_mask = spatial_mask * valid_mask[0]
            if not torch.any(spatial_mask > 0):
                diagnostics["empty_obs_days"] += 1
                continue

            track_field, track_finite = prepare_model_field(
                _field_at_hour(model_record.path, self.hour_index),
                self.indices,
                self.means,
                self.stds,
                self.padding_values,
                self.image_size,
            )
            diagnostics["sral_files_used"] += len(sral_paths)
            for channel in self.observed_channels:
                fill_mask = (obs_mask[channel] == 0) & (spatial_mask > 0) & (track_finite[channel] > 0)
                obs_values[channel][fill_mask] = track_field[channel][fill_mask]
                obs_mask[channel][fill_mask] = 1.0

        valid_count = float(valid_mask[0].sum().item())
        obs_count = int(obs_mask[self.observed_channels].sum().item()) if self.observed_channels else 0
        diagnostics["obs_count"] = obs_count
        diagnostics["observed_fraction"] = obs_count / max(valid_count, 1.0)
        return obs_values, obs_mask, diagnostics

    def _sral_spatial_mask(self, target_date: date) -> torch.Tensor | None:
        if not self.sral_records:
            return None
        mask = torch.zeros(self.image_size, dtype=torch.float32)
        for offset in range(self.assimilation_range):
            current_date = target_date - timedelta(days=offset)
            sral_paths = self.sral_records.get(current_date, [])
            combined_sral = _combine_sral_files(sral_paths)
            if combined_sral is None:
                continue
            transformed_sral = _sral_transform(
                combined_sral,
                int(self.config.get("observation_mask", {}).get("sral_transform_index", self.config.get("sral_transform_index", 11))),
            )
            finite = torch.as_tensor(np.isfinite(transformed_sral).astype(np.float32))
            finite = pad_to_size(finite, self.image_size, fill_value=0.0)[0]
            mask = torch.maximum(mask, finite)
        return mask if torch.any(mask > 0) else None

    def __getitem__(self, idx):
        back_record = self.back_data[idx]
        target_record = self.obs_data[idx + self.obs_shift]
        background, background_finite = prepare_model_field(
            _field_at_hour(back_record.path, self.hour_index),
            self.indices,
            self.means,
            self.stds,
            self.padding_values,
            self.image_size,
        )
        truth, truth_finite = prepare_model_field(
            _field_at_hour(target_record.path, self.hour_index),
            self.indices,
            self.means,
            self.stds,
            self.padding_values,
            self.image_size,
        )
        valid_mask = self.base_valid_mask * background_finite * truth_finite

        mask_kind = self.config.get("observation_mask", {}).get("kind", "generated_track")
        if mask_kind == "sral_tracks":
            obs_values, obs_mask, obs_diagnostics = self._sral_track_observations(target_record.date, valid_mask)
        else:
            rng = np.random.default_rng(self.seed + idx)
            spatial_mask = _make_mask(self.image_size, valid_mask, self.config, rng)
            spatial_mask = spatial_mask * valid_mask[0]
            obs_values, obs_mask = make_observation_tensors(truth, spatial_mask, self.observed_channels)
            valid_count = float(valid_mask[0].sum().item())
            obs_diagnostics = {
                "mask_kind": mask_kind,
                "sral_files_used": 0,
                "empty_obs_days": 0,
                "obs_count": int(obs_mask[self.observed_channels].sum().item()) if self.observed_channels else 0,
                "observed_fraction": float(obs_mask[self.observed_channels].sum().item()) / max(valid_count, 1.0) if self.observed_channels else 0.0,
            }

        return {
            "truth": truth,
            "background": background,
            "obs_values": obs_values,
            "obs_mask": obs_mask,
            "valid_mask": valid_mask,
            "meta": {
                "case_id": target_record.date.isoformat(),
                "background_date": back_record.date.isoformat(),
                "target_date": target_record.date.isoformat(),
                "background_path": str(back_record.path),
                "target_path": str(target_record.path),
                "split": self.split,
                **obs_diagnostics,
            },
        }


DATASETS = {
    SmokeM2MDataset.name: SmokeM2MDataset,
    M2MForecastDataset.name: M2MForecastDataset,
}


def build_dataset(config, split: str = "train") -> Dataset:
    dataset_name = config["dataset_name"]
    if dataset_name not in DATASETS:
        raise KeyError(f"Unknown dataset_name: {dataset_name}")
    return DATASETS[dataset_name](config, split=split)
