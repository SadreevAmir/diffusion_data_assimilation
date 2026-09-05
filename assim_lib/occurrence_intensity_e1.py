"""Fail-closed executable handoff for the E1 engineering sentinel."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REVIEWED_MODE = "occurrence_intensity_e1_engineering_sentinel"


def validate_config(config: dict[str, Any]) -> None:
    if config.get("mode") != REVIEWED_MODE:
        raise ValueError("unreviewed E1 mode")
    if config.get("clearml") != {"enabled": True}:
        raise ValueError("E1 requires literal clearml.enabled=true")
    if config.get("protocol") != "primary_real":
        raise ValueError("E1 primary sentinel permits real observations only")
    if config.get("cases") != 8 or config.get("selection_role") != "engineering_only":
        raise ValueError("E1 must be the fixed eight-case engineering sentinel")
    if config.get("rank_gate") is not False:
        raise ValueError("E1 engineering sentinel cannot contain a rank gate")
    dataset = config.get("dataset", {})
    if dataset.get("lags_days") != [0, 1, 2] or dataset.get("observed_finite_only") is not True:
        raise ValueError("invalid lag or finite-mask contract")
    target = config.get("target", {})
    expected = {
        "occurrence": "continuous_dequantized_exact_zero",
        "zero_intensity_auxiliary": "uniform_0_1",
        "exact_one_policy_source": "train_inventory",
    }
    if target != expected:
        raise ValueError("invalid occurrence/intensity target law")


def controller_request(config_path: str | Path) -> dict[str, Any]:
    """Validate the literal local handoff; this function never launches work."""
    path = Path(config_path)
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    validate_config(config)
    return {
        "reviewed_mode": REVIEWED_MODE,
        "resource_kind": "server_gpu",
        "admission_required": True,
        "admission_status": "PENDING_INDEPENDENT_REVIEW",
        "launch_authorized": False,
        "clearml_enabled": True,
        "cases": 8,
        "scientific_gate": False,
    }
