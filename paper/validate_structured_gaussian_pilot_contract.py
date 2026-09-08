#!/usr/bin/env python3
"""Fail-closed validation for the immutable checkpointed Gaussian pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "paper/STRUCTURED_GAUSSIAN_CHECKPOINTED_PILOT_CONTRACT.json"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
GIT_OID_RE = re.compile(r"[0-9a-f]{40,64}")


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain an object")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sanitized_git_environment() -> dict[str, str]:
    """Bind Git discovery to ``cwd``, never inherited remote checkout overrides."""

    return {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}


def _git_output(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        env=_sanitized_git_environment(),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _validate_git_contract(root: Path, contract: dict[str, Any]) -> str:
    base = contract.get("base_science_head")
    if not isinstance(base, str) or GIT_OID_RE.fullmatch(base) is None:
        raise ValueError("invalid base science HEAD")
    if contract.get("execution_commit_pinned_externally") is not True:
        raise ValueError("execution commit must be pinned externally")
    try:
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", base, "HEAD"],
            cwd=root,
            env=_sanitized_git_environment(),
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise ValueError("base science HEAD is not an ancestor of current HEAD") from exc
    head = _git_output(root, "rev-parse", "HEAD")
    changed = set(_git_output(root, "diff", "--name-only", base, "--").splitlines())
    untracked = _git_output(root, "ls-files", "--others", "--exclude-standard")
    changed.update(untracked.splitlines())
    changed.discard("")
    required = contract.get("required_changes_from_base")
    if not isinstance(required, list) or not all(isinstance(item, str) for item in required):
        raise ValueError("required changes from base are absent")
    if changed != set(required):
        raise ValueError(
            f"changes from base differ: actual={sorted(changed)} expected={sorted(required)}"
        )
    return head


def _validate_memory_admission(
    path: Path,
    supplied_sha256: str,
    contract: dict[str, Any],
) -> dict[str, Any]:
    frozen = contract.get("memory_admission")
    if not isinstance(frozen, dict):
        raise ValueError("memory admission contract is absent")
    expected_sha256 = frozen.get("result_sha256")
    if (
        path.is_symlink()
        or not path.is_file()
        or SHA256_RE.fullmatch(supplied_sha256) is None
        or supplied_sha256 != expected_sha256
        or _sha256(path) != expected_sha256
    ):
        raise ValueError("memory admission result path/hash mismatch")
    result = _load(path)
    for key in (
        "schema_version", "status", "purpose", "method_config",
        "method_config_sha256", "protocol", "protocol_sha256",
    ):
        if result.get(key) != frozen.get(key):
            raise ValueError(f"memory admission {key} mismatch")
    if result.get("test_2023_used") is not False or result.get("full_training_permitted") is not False:
        raise ValueError("memory admission exceeded its validation-only authority")

    parity = result.get("cpu_parity")
    required_parity = {
        "status": "passed",
        "forward_bitwise_equal": True,
        "rng_state_equal": True,
        "gradient_none_pattern_equal": True,
        "gradients_finite": True,
        "gradient_atol": 0.000001,
        "gradient_rtol": 0.00001,
        "gradient_max_abs_difference": 0.0,
        "gradient_max_relative_difference": 0.0,
        "post_adam_parameters_bitwise_equal": True,
        "post_ema_parameters_bitwise_equal": True,
        "optimizer_steps": 1,
        "scheduler_steps": 1,
        "ema_updates": 1,
    }
    if not isinstance(parity, dict) or any(
        parity.get(key) != expected for key, expected in required_parity.items()
    ):
        raise ValueError("memory admission CPU parity evidence is incomplete")

    comparison = result.get("activation_memory_comparison")
    if not isinstance(comparison, dict) or comparison.get("checkpointed_peak_strictly_lower") is not True:
        raise ValueError("memory admission activation comparison did not pass")
    unchecked = comparison.get("uncheckpointed")
    checked = comparison.get("checkpointed")
    if not isinstance(unchecked, dict) or not isinstance(checked, dict):
        raise ValueError("memory admission activation probes are absent")
    if (
        unchecked.get("checkpointing") is not False
        or checked.get("checkpointing") is not True
        or unchecked.get("loss_finite") is not True
        or checked.get("loss_finite") is not True
    ):
        raise ValueError("memory admission activation probes are invalid")
    unchecked_peak = unchecked.get("memory", {}).get("peak_reserved_bytes")
    checked_peak = checked.get("memory", {}).get("peak_reserved_bytes")
    if (
        not isinstance(unchecked_peak, int)
        or not isinstance(checked_peak, int)
        or checked_peak >= unchecked_peak
    ):
        raise ValueError("checkpointing did not reduce activation peak")

    production = result.get("production_path")
    expected_trace = ["backward", "optimizer", "scheduler", "ema"] * 2
    if not isinstance(production, dict) or any((
        production.get("status") != "passed",
        production.get("train_batches") != 2,
        production.get("validation_batches") != 1,
        production.get("batch_size") != 16,
        production.get("validation_batch_size") != 8,
        production.get("image_size") != [320, 256],
        production.get("mixed_precision") != "bf16",
        production.get("gradient_accumulation_steps") != 1,
        production.get("validation_weight_source") != "ema",
        production.get("call_trace") != expected_trace,
        production.get("clearml_online") is not True,
        not isinstance(production.get("clearml_task_id"), str),
        not production.get("clearml_task_id"),
        not isinstance(production.get("checkpointed_modules"), list),
        not production.get("checkpointed_modules"),
    )):
        raise ValueError("memory admission production evidence is incomplete")
    memory = production.get("memory")
    if not isinstance(memory, dict):
        raise ValueError("memory admission production memory evidence is absent")
    peak = memory.get("peak_reserved_bytes")
    total = memory.get("total_bytes")
    if (
        not isinstance(peak, int)
        or not isinstance(total, int)
        or peak <= 0
        or total <= 0
        or peak > 0.9 * total
    ):
        raise ValueError("memory admission production peak exceeds its gate")
    return result


def validate(
    root: Path = ROOT,
    *,
    validate_git: bool = True,
    memory_admission_result: Path | None = None,
    memory_admission_sha256: str | None = None,
) -> dict[str, Any]:
    contract = _load(root / CONTRACT.relative_to(ROOT))
    if contract.get("schema_version") != "structured_gaussian_checkpointed_pilot_contract_v1":
        raise ValueError("unexpected contract schema")
    if contract.get("immutable") is not True or contract.get("launch_authorized") is not False:
        raise ValueError("contract must be immutable and not launch-authorized")
    if contract.get("status") != "frozen_awaiting_controller_assigned_experiment_id":
        raise ValueError("contract must await a controller-assigned experiment id")
    if contract.get("experiment_id") is not None:
        raise ValueError("unapproved experiment id embedded in frozen contract")
    if contract.get("experiment_id_authority") != "trusted_controller_allowlist_only":
        raise ValueError("experiment id authority is not fail-closed")
    identity_audit = contract.get("durable_identity_audit")
    expected_consumed = {
        "structured_joint_gaussian_preconditioned_pilot_retry1",
        "structured_joint_gaussian_checkpointed_pilot_valid",
    }
    if not isinstance(identity_audit, dict) or any((
        identity_audit.get("status") != "conflict_confirmed",
        identity_audit.get("conflicting_experiment_id")
        != "structured_joint_gaussian_checkpointed_pilot_valid",
        set(identity_audit.get("consumed_experiment_ids", [])) != expected_consumed,
        identity_audit.get("retry_suffix_permitted") is not False,
        identity_audit.get("required_resolution")
        != "controller_assigns_one_globally_new_allowlisted_experiment_id",
    )):
        raise ValueError("durable experiment identity audit is incomplete")
    head = _validate_git_contract(root, contract) if validate_git else None

    identities = contract.get("required_identity_sha256")
    if not isinstance(identities, dict) or not identities:
        raise ValueError("required identities are absent")
    for relative, expected in identities.items():
        path = root / relative
        if (
            not isinstance(relative, str)
            or not isinstance(expected, str)
            or SHA256_RE.fullmatch(expected) is None
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or path.is_symlink()
            or not path.is_file()
            or _sha256(path) != expected
        ):
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
    quantitative_stop_go = contract.get("quantitative_stop_go")
    if not isinstance(quantitative_stop_go, dict) or quantitative_stop_go.get(
        "boundary"
    ) != (
        "each lead ice_occurrence and sic_archive_cap brier_score and "
        "reliability_l1 must not exceed the paired raw reference"
    ):
        raise ValueError("boundary stop/go contract changed")
    if (memory_admission_result is None) != (memory_admission_sha256 is None):
        raise ValueError("memory admission path and SHA-256 must be supplied together")
    if memory_admission_result is not None and memory_admission_sha256 is not None:
        _validate_memory_admission(
            memory_admission_result,
            memory_admission_sha256,
            contract,
        )
    contract["validated_current_head"] = head
    return contract


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory-admission-result", type=Path)
    parser.add_argument("--memory-admission-sha256")
    arguments = parser.parse_args()
    validated = validate(
        memory_admission_result=arguments.memory_admission_result,
        memory_admission_sha256=arguments.memory_admission_sha256,
    )
    print(json.dumps({
        "status": "passed",
        "base_science_head": validated["base_science_head"],
        "validated_current_head": validated["validated_current_head"],
        "launch_authorized": validated["launch_authorized"],
        "contract": CONTRACT.name,
    }, sort_keys=True))
