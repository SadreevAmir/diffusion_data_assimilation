"""Immutable scientific contract shared by both dynamics-cascade stages."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any

from .direct_dynamics_training import DIRECT_LEADS


def forecast_contract_from_data_config(data_config: Mapping[str, Any]) -> dict[str, Any]:
    fields = tuple(str(value) for value in data_config.get("fields", ()))
    indices = tuple(int(value) for value in data_config.get("indices", ()))
    means = tuple(float(value) for value in data_config.get("means", ()))
    stds = tuple(float(value) for value in data_config.get("stds", ()))
    leads = tuple(int(value) for value in data_config.get("trajectory_lead_days", ()))
    image_size = tuple(int(value) for value in data_config.get("image_size", ()))
    forcing_indices = tuple(int(value) for value in data_config.get("dynamic_forcing_indices", ()))
    forcing = data_config.get("dynamic_forcing_stats")
    if fields != ("siconc", "sithic") or indices != (0, 1):
        raise ValueError("cascade contract requires ordered siconc/sithic fields at indices 0/1")
    if len(means) != 2 or len(stds) != 2 or any(not math.isfinite(value) for value in (*means, *stds)):
        raise ValueError("cascade contract requires finite two-field normalization")
    if any(value <= 0 for value in stds):
        raise ValueError("cascade normalization standard deviations must be positive")
    if leads != DIRECT_LEADS or image_size != (320, 256):
        raise ValueError("cascade contract requires d+3/d+6/d+9 on the 320x256 grid")
    if data_config.get("conditioning_layout") != "structured_sic_sit_dynamics_v1":
        raise ValueError("cascade contract requires the causal dynamics conditioning layout")
    if data_config.get("background_strategy") != "none":
        raise ValueError("cascade dynamics cannot use a previous-year background")
    if data_config.get("hour_mode") != "all":
        raise ValueError("cascade dynamics requires all 24 archive slices")
    if data_config.get("mask_true_is_invalid") is not True:
        raise ValueError("cascade contract requires the declared invalid-land mask semantics")
    if data_config.get("model_nan_semantics") != "joint_sic_sit_nan_means_open_water_zero":
        raise ValueError("cascade contract requires the audited joint SIC/SIT NaN semantics")
    if tuple(float(value) for value in data_config.get("padding_values", ())) != (0.0, 0.0):
        raise ValueError("cascade contract requires zero padding outside the physical grid")
    if forcing_indices != (6, 7, 13, 14) or not isinstance(forcing, Mapping):
        raise ValueError("cascade contract requires the declared wind/current channels and statistics")
    forcing_means = tuple(float(value) for value in forcing.get("means", ()))
    forcing_stds = tuple(float(value) for value in forcing.get("stds", ()))
    if tuple(int(value) for value in forcing.get("indices", ())) != forcing_indices:
        raise ValueError("dynamic forcing normalization indices differ from the conditioning channels")
    if forcing.get("source_split") != "train":
        raise ValueError("dynamic forcing normalization must be estimated on train only")
    if (
        len(forcing_means) != 4
        or len(forcing_stds) != 4
        or any(not math.isfinite(value) for value in (*forcing_means, *forcing_stds))
        or any(value <= 0 for value in forcing_stds)
    ):
        raise ValueError("cascade contract requires finite four-field forcing normalization")
    return {
        "schema_version": 1,
        "state_fields": list(fields),
        "state_indices": list(indices),
        "state_means": list(means),
        "state_stds": list(stds),
        "trajectory_lead_days": list(leads),
        "trajectory_channel_order": [f"d{lead}_{field}" for lead in leads for field in fields],
        "image_size": list(image_size),
        "conditioning_layout": data_config["conditioning_layout"],
        "background_strategy": data_config["background_strategy"],
        "dynamic_forcing_indices": list(forcing_indices),
        "dynamic_forcing_means": list(forcing_means),
        "dynamic_forcing_stds": list(forcing_stds),
        "mask_true_is_invalid": bool(data_config.get("mask_true_is_invalid")),
        "model_nan_semantics": str(data_config.get("model_nan_semantics")),
        "cascade_factor": 2,
        "coarse_operator": "masked_valid_ocean_2x2_average_v1",
        "right_inverse": "bilinear_plus_masked_cell_mean_correction_v1",
    }


def forecast_contract_sha256(contract: Mapping[str, Any]) -> str:
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
