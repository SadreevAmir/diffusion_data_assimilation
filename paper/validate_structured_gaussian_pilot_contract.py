#!/usr/bin/env python3
"""Fail-closed validation for the immutable checkpointed Gaussian pilot."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "paper/STRUCTURED_GAUSSIAN_CHECKPOINTED_PILOT_CONTRACT.json"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain an object")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(root: Path = ROOT) -> dict[str, Any]:
    contract = _load(root / CONTRACT.relative_to(ROOT))
    if contract.get("schema_version") != "structured_gaussian_checkpointed_pilot_contract_v1":
        raise ValueError("unexpected contract schema")
    if contract.get("immutable") is not True or contract.get("launch_authorized") is not False:
        raise ValueError("contract must be immutable and not launch-authorized")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if head != contract.get("science_head"):
        raise ValueError("science HEAD mismatch")

    identities = contract.get("required_identity_sha256")
    if not isinstance(identities, dict) or not identities:
        raise ValueError("required identities are absent")
    for relative, expected in identities.items():
        path = root / relative
        if path.is_symlink() or not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"immutable identity mismatch: {relative}")

    experiment = _load(root / contract["paths"]["experiment"])
    method = _load((root / contract["paths"]["experiment"]).parent / experiment["model_config"])
    frozen = contract["training_semantics"]
    required = {
        "train_batch_size": 16,
        "eval_batch_size": 8,
        "gradient_accumulation_steps": 1,
        "mixed_precision": "bf16",
        "minimum_optimizer_steps": 2128,
        "lr_scheduler_total_steps": 4123,
        "learning_rate": 0.0001,
        "ema_decay": 0.999,
        "ema_use_warmup": False,
        "ema_update_after_step": 0,
        "validation_weight_source": "ema",
        "structured_velocity_parameterization": "gaussian_path_preconditioned",
        "activation_checkpointing": True,
    }
    if frozen != required:
        raise ValueError("frozen training semantics changed")
    for key, expected in required.items():
        if method.get(key) != expected:
            raise ValueError(f"method config changed: {key}")
    if experiment.get("clearml", {}).get("enabled") is not True:
        raise ValueError("ClearML must remain enabled")

    evaluation = contract.get("evaluation")
    if evaluation != {
        "cases": 8,
        "ensemble_size": 8,
        "paired_initial_noise": True,
        "strict_fp32": True,
        "rk4_intervals": 64,
        "rank_histograms": ["sic", "sit"],
        "coverage_levels": [0.5, 0.8, 0.9],
        "spatial_variogram_lags": [1, 2, 4],
        "temporal_increment_transitions": 3,
        "physical_support_hard_veto": True,
        "large_actual_sic_sit_visual_gate": True,
        "visual_review_required": True,
    }:
        raise ValueError("evaluation contract changed")
    return contract


if __name__ == "__main__":
    validated = validate()
    print(json.dumps({
        "status": "passed",
        "science_head": validated["science_head"],
        "launch_authorized": validated["launch_authorized"],
        "contract": CONTRACT.name,
    }, sort_keys=True))
