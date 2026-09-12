"""Frozen CPU audit of gradient directions retained by the SIC decoder."""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from .clearml_tracking import ClearMLTracker
from .config import load_json
from .direct_dynamics_cascade_coarse_fine_boundary_audit import (
    _finish_success,
    _record_failure,
    _sha256,
    _strict_atomic_json,
    _terminate,
)
from .direct_dynamics_cascade_coarse_proper_refinement import proper_objective
from .direct_dynamics_cascade_fine_training import _clean_code_identity
from .direct_dynamics_fine_support_proper_admission import (
    _pre_correction_kkt_free_set,
    apply_training_sic_decoder,
)
from .direct_dynamics_sic_support_decoder_scoring import canonical_physical_decode
from .direct_dynamics_sit_support_decoder import _to_blocks


SIC_CHANNELS = (0, 2, 4)
LEADS = ("d3_sic", "d6_sic", "d9_sic")


def _verified_load(path: Path, sha256: str) -> dict[str, Any]:
    if not path.is_file() or _sha256(path) != sha256:
        raise ValueError(f"frozen evidence SHA mismatch: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(payload, dict):
        raise TypeError(f"frozen evidence is not a mapping: {path}")
    return payload


def _jacobian_statistics(
    raw: torch.Tensor,
    coarse: torch.Tensor,
    mask: torch.Tensor,
    decoded_gradient: torch.Tensor,
) -> dict[str, Any]:
    """Measure the exact implementation-selected active-set Jacobian."""
    if raw.shape != decoded_gradient.shape or raw.ndim != 5 or raw.shape[2] != 1:
        raise ValueError("raw and decoded_gradient must be matching [case,member,1,y,x]")
    cases, members = raw.shape[:2]
    member_mask = mask[:, None].expand(-1, members, -1, -1, -1).flatten(0, 1)
    raw_flat = raw.flatten(0, 1)
    coarse_flat = coarse.flatten(0, 1)
    gradient_flat = decoded_gradient.flatten(0, 1).double()
    free = _pre_correction_kkt_free_set(raw_flat, coarse_flat, member_mask, 2)
    free_blocks = _to_blocks(free, 2)
    mask_blocks = _to_blocks(member_mask > 0, 2)
    gradient_blocks = _to_blocks(gradient_flat, 2)
    valid_count = mask_blocks.sum(dim=-1)
    free_count = free_blocks.sum(dim=-1)
    active = valid_count > 0
    target = coarse_flat.double().clamp(0.0, 1.0)
    interior_budget = active & (target > 0.0) & (target < 1.0)
    endpoint_zero = active & (target == 0.0)
    endpoint_one = active & (target == 1.0)
    rank = torch.where(free_count > 0, free_count - 1, torch.zeros_like(free_count))
    potential_rank = torch.where(
        interior_budget & (valid_count > 0), valid_count - 1, torch.zeros_like(valid_count)
    )
    free_mean = (
        (gradient_blocks * free_blocks).sum(dim=-1, keepdim=True)
        / free_count.unsqueeze(-1).clamp_min(1)
    )
    retained = torch.where(free_blocks, gradient_blocks - free_mean, 0.0)
    valid_mean = (
        (gradient_blocks * mask_blocks).sum(dim=-1, keepdim=True)
        / valid_count.unsqueeze(-1).clamp_min(1)
    )
    fixed_budget = torch.where(
        mask_blocks, gradient_blocks - valid_mean, torch.zeros_like(gradient_blocks)
    )
    bounded_budget = torch.where(
        interior_budget.unsqueeze(-1), fixed_budget, torch.zeros_like(fixed_budget)
    )
    components = {
        "coarse_budget_change": torch.where(
            mask_blocks, gradient_blocks - fixed_budget, torch.zeros_like(gradient_blocks)
        ),
        "endpoint_budget_restriction": fixed_budget - bounded_budget,
        "interior_box_saturation": bounded_budget - retained,
        "available_fine_direction": retained,
    }
    raw_sq = torch.where(mask_blocks, gradient_blocks.square(), 0.0).sum()
    retained_sq = retained.square().sum()
    component_sq = {name: value.square().sum() for name, value in components.items()}
    reconstructed_sq = sum(component_sq.values())
    closure_error = abs(float(reconstructed_sq - raw_sq))
    closure_relative = closure_error / float(raw_sq) if float(raw_sq) else None
    if closure_relative is not None and closure_relative > 1e-10:
        raise RuntimeError("decoder gradient energy decomposition did not close")
    raw_norm = math.sqrt(float(raw_sq))
    retained_norm = math.sqrt(float(retained_sq))
    block_raw_sq = torch.where(mask_blocks, gradient_blocks.square(), 0.0).sum(dim=-1)
    block_retained_sq = retained.square().sum(dim=-1)
    informative = active & (block_raw_sq > 0)
    dead = informative & (block_retained_sq == 0)
    free_hist = Counter(map(int, free_count[active].tolist()))
    rank_hist = Counter(map(int, rank[active].tolist()))
    active_blocks = int(active.sum())
    potential = int(potential_rank.sum())
    return {
        "active_blocks": active_blocks,
        "valid_coordinates": int(valid_count[active].sum()),
        "interior_budget_blocks": int(interior_budget.sum()),
        "zero_budget_blocks": int(endpoint_zero.sum()),
        "one_budget_blocks": int(endpoint_one.sum()),
        "interior_budget_fraction": float(interior_budget.sum()) / active_blocks,
        "zero_budget_fraction": float(endpoint_zero.sum()) / active_blocks,
        "one_budget_fraction": float(endpoint_one.sum()) / active_blocks,
        "free_coordinate_count": int(free_count[active].sum()),
        "jacobian_rank_sum": int(rank[active].sum()),
        "potential_fixed_budget_rank_sum": potential,
        "rank_fraction_of_valid_coordinates": float(rank[active].sum())
        / float(valid_count[active].sum()),
        "rank_fraction_after_box_saturation": (
            float(rank[active].sum()) / potential if potential else None
        ),
        "free_count_histogram": {str(i): free_hist.get(i, 0) for i in range(5)},
        "jacobian_rank_histogram": {str(i): rank_hist.get(i, 0) for i in range(4)},
        "proper_gradient_norm_before_decoder_jacobian": raw_norm,
        "proper_gradient_norm_after_decoder_jacobian": retained_norm,
        "proper_gradient_norm_fraction_retained": (
            retained_norm / raw_norm if raw_norm else None
        ),
        "proper_gradient_energy_fraction_retained": (
            float(retained_sq / raw_sq) if float(raw_sq) else None
        ),
        "proper_gradient_energy": float(raw_sq),
        "orthogonal_energy_decomposition": {
            name: {
                "energy": float(value),
                "fraction": float(value / raw_sq) if float(raw_sq) else None,
            }
            for name, value in component_sq.items()
        },
        "orthogonal_energy_closure_absolute_error": closure_error,
        "orthogonal_energy_closure_relative_error": closure_relative,
        "informative_blocks": int(informative.sum()),
        "informative_blocks_with_zero_retained_gradient": int(dead.sum()),
        "informative_block_fraction_with_zero_retained_gradient": (
            float(dead.sum()) / float(informative.sum()) if torch.any(informative) else None
        ),
        "derivative_interpretation": (
            "Local derivative selected by the reviewed pre-correction KKT active set; "
            "at projection kinks this is not a global impossibility result."
        ),
    }


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != "sic_decoder_jacobian_audit_v1":
        raise ValueError("unreviewed SIC decoder Jacobian audit schema")
    if config.get("split") != "valid" or config.get("test_2023") != "closed":
        raise ValueError("audit is restricted to frozen validation-2022 evidence")
    if config.get("cases") != 12 or config.get("members") != 8:
        raise ValueError("audit requires the frozen 12 x 8 panel")
    if config.get("optimizer_steps") != 0 or config.get("sampling_performed") is not False:
        raise ValueError("Jacobian audit must not train or sample")
    if tuple(config.get("branches", {}).keys()) != ("control", "candidate"):
        raise ValueError("audit requires frozen control and candidate branches")
    if not isinstance(config.get("source_contract"), dict):
        raise ValueError("audit requires a frozen source contract")
    if len(config.get("implementation_bindings", {})) != 4:
        raise ValueError("audit requires four frozen implementation bindings")


def _validate_provenance(config: dict[str, Any], repo: Path) -> dict[str, Any]:
    contract_spec = config["source_contract"]
    contract_path = Path(contract_spec["path"])
    if not contract_path.is_file() or _sha256(contract_path) != contract_spec["sha256"]:
        raise ValueError("source evaluator contract SHA mismatch")
    contract = load_json(contract_path)
    if (
        contract.get("code_identity", {}).get("git_commit")
        != config["source_evaluator_commit"]
        or contract.get("split") != "valid"
        or contract.get("members") != 8
        or len(contract.get("case_ids", [])) != 12
        or contract.get("test_2023_used") is not False
    ):
        raise ValueError("source evaluator contract differs from frozen panel")
    expected = {"fixed_inputs": config["fixed_inputs"], **config["branches"]}
    evidence_names = config["source_evidence_names"]
    for label, spec in expected.items():
        filename = evidence_names[label]
        if contract.get("evidence_sha256", {}).get(filename) != spec["sha256"]:
            raise ValueError(f"source contract does not bind {label}")
    implementations: dict[str, Any] = {}
    for label, spec in config["implementation_bindings"].items():
        path = repo / spec["path"]
        if not path.is_file() or _sha256(path) != spec["sha256"]:
            raise ValueError(f"audit implementation SHA mismatch: {label}")
        implementations[label] = {"path": str(path), "sha256": spec["sha256"]}
    return {
        "source_contract": {"path": str(contract_path), "sha256": contract_spec["sha256"]},
        "source_evaluator_commit": config["source_evaluator_commit"],
        "case_ids": contract["case_ids"],
        "member_indices": list(range(8)),
        "implementations": implementations,
    }


def _validate_embedded_identity(
    payload: dict[str, Any], fixed_sha: str, case_ids: list[str], branch: str
) -> None:
    if payload.get("fixed_inputs_sha256") != fixed_sha:
        raise ValueError(f"{branch} evidence is not bound to fixed inputs")
    if list(payload.get("case_ids", ())) != case_ids:
        raise ValueError(f"{branch} case identities differ")
    if list(payload.get("member_indices", ())) != list(range(8)):
        raise ValueError(f"{branch} member identities differ")


def run(config_path: Path, output: Path) -> dict[str, Any]:
    torch.set_num_threads(6)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_available() or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Jacobian audit requires CUDA_VISIBLE_DEVICES empty")
    metrics_path = output.with_name(f"{output.stem}.metrics.json")
    if output.exists() or output.is_symlink() or metrics_path.exists() or metrics_path.is_symlink():
        raise FileExistsError(f"refusing to overwrite {output}")
    config = load_json(config_path)
    _validate_config(config)
    output.parent.mkdir(parents=True, exist_ok=True)
    reservation = {
        "status": "reserved",
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
    }
    with output.open("x") as handle:
        json.dump(reservation, handle, indent=2, allow_nan=False)
        handle.write("\n")
    tracker = None
    try:
        repo = Path(__file__).resolve().parents[1]
        provenance = _validate_provenance(config, repo)
        code_identity = _clean_code_identity(repo)
        tracker = ClearMLTracker(
            config["project_name"],
            f'{config["task_name"]}-{output.parent.name}',
            tags=config["clearml"]["tags"],
            env_path=config["clearml"]["env_path"],
        )
        tracker.connect("jacobian_audit_contract", config)
        _strict_atomic_json(
            output,
            {
                **reservation,
                "status": "running",
                "code_identity": code_identity,
                "clearml_task_id": str(tracker.task.id),
            },
        )
        result = _run_bound(config, provenance, code_identity)
        _strict_atomic_json(metrics_path, result)
        for branch, branch_result in result["branches"].items():
            for lead, row in branch_result["sic_leads"].items():
                for name in (
                    "rank_fraction_of_valid_coordinates",
                    "rank_fraction_after_box_saturation",
                    "proper_gradient_norm_fraction_retained",
                    "proper_gradient_energy_fraction_retained",
                ):
                    if row[name] is not None:
                        tracker.report_single_value(f"{branch}/{lead}/{name}", row[name])
                for name, component in row["orthogonal_energy_decomposition"].items():
                    if component["fraction"] is not None:
                        tracker.report_single_value(
                            f"{branch}/{lead}/energy_fraction/{name}",
                            component["fraction"],
                        )
        tracker.upload_artifact("sic_decoder_jacobian_audit", metrics_path)
        _finish_success(
            tracker,
            output,
            metrics_path,
            {
                **reservation,
                "code_identity": code_identity,
                "clearml_task_id": str(tracker.task.id),
                "scientific_role": result["scientific_role"],
            },
        )
        tracker = None
        return result
    except BaseException as error:
        try:
            _record_failure(output, reservation, error)
        except Exception:
            pass
        if tracker is not None:
            try:
                tracker.task.mark_failed(status_reason=str(error)[:1000])
            except Exception:
                pass
        raise
    finally:
        if tracker is not None:
            try:
                tracker.close()
            except Exception:
                pass


def _run_bound(
    config: dict[str, Any], provenance: dict[str, Any], code_identity: dict[str, Any]
) -> dict[str, Any]:
    fixed_spec = config["fixed_inputs"]
    fixed = _verified_load(Path(fixed_spec["path"]), fixed_spec["sha256"])
    mask = fixed.pop("valid_mask").float()
    truth = fixed.pop("truth_physical").double()
    means_vector = fixed.pop("normalization_means")
    stds_vector = fixed.pop("normalization_stds")
    del fixed
    stds = stds_vector.double().reshape(1, 1, 6, 1, 1)
    if mask.shape != (12, 1, 320, 256) or truth.shape != (12, 6, 320, 256):
        raise ValueError("frozen input shapes differ from reviewed panel")
    branches: dict[str, Any] = {}
    for branch, spec in config["branches"].items():
        evidence = _verified_load(Path(spec["path"]), spec["sha256"])
        _validate_embedded_identity(
            evidence, fixed_spec["sha256"], provenance["case_ids"], branch
        )
        forecast_normalized = evidence.pop("forecast_normalized")
        coarse_normalized = evidence.pop("coarse_normalized")
        del evidence
        if forecast_normalized.shape != (12, 8, 6, 320, 256):
            raise ValueError("frozen fine ensemble shape differs")
        if coarse_normalized.shape != (12, 8, 6, 160, 128):
            raise ValueError("frozen coarse ensemble shape differs")
        accumulators: dict[str, list[dict[str, Any]]] = {lead: [] for lead in LEADS}
        objective_values: list[float] = []
        crps_values: list[float] = []
        energy_values: list[float] = []
        for case in range(12):
            raw = canonical_physical_decode(
                forecast_normalized[case : case + 1], means_vector, stds_vector
            )
            coarse = canonical_physical_decode(
                coarse_normalized[case : case + 1], means_vector, stds_vector
            )
            decoded = apply_training_sic_decoder(
                raw, coarse, mask[case : case + 1]
            ).detach().requires_grad_(True)
            standardized = decoded / stds
            standardized_truth = truth[case : case + 1] / stds[:, 0]
            objective, crps, energy = proper_objective(
                standardized, standardized_truth, mask[case : case + 1].double()
            )
            objective.backward()
            gradient = decoded.grad
            if gradient is None or not torch.isfinite(gradient).all():
                raise FloatingPointError("proper-score decoded gradient is invalid")
            objective_values.append(float(objective.detach()))
            crps_values.append(float(crps.detach()))
            energy_values.append(float(energy.detach()))
            for lead, channel in zip(LEADS, SIC_CHANNELS):
                accumulators[lead].append(
                    _jacobian_statistics(
                        raw[:, :, channel : channel + 1],
                        coarse[:, :, channel : channel + 1],
                        mask[case : case + 1],
                        gradient[:, :, channel : channel + 1],
                    )
                )
            del raw, coarse, decoded, gradient, objective, crps, energy
        per_lead: dict[str, Any] = {}
        for lead, rows in accumulators.items():
            summed = {
                key: sum(float(row[key]) for row in rows)
                for key in (
                    "active_blocks", "valid_coordinates", "free_coordinate_count",
                    "jacobian_rank_sum", "potential_fixed_budget_rank_sum",
                    "interior_budget_blocks", "zero_budget_blocks", "one_budget_blocks",
                    "informative_blocks", "informative_blocks_with_zero_retained_gradient",
                )
            }
            before_sq = sum(row["proper_gradient_norm_before_decoder_jacobian"] ** 2 for row in rows)
            after_sq = sum(row["proper_gradient_norm_after_decoder_jacobian"] ** 2 for row in rows)
            decomposition_energy = {
                name: sum(
                    row["orthogonal_energy_decomposition"][name]["energy"]
                    for row in rows
                )
                for name in (
                    "coarse_budget_change",
                    "endpoint_budget_restriction",
                    "interior_box_saturation",
                    "available_fine_direction",
                )
            }
            decomposition_total = sum(decomposition_energy.values())
            closure_error = abs(decomposition_total - before_sq)
            closure_relative = closure_error / before_sq if before_sq else None
            if closure_relative is not None and closure_relative > 1e-10:
                raise RuntimeError(
                    "aggregate decoder gradient energy decomposition did not close"
                )
            active_blocks = summed["active_blocks"]
            potential = summed["potential_fixed_budget_rank_sum"]
            per_lead[lead] = {
                **{key: int(value) for key, value in summed.items()},
                "rank_fraction_of_valid_coordinates": summed["jacobian_rank_sum"] / summed["valid_coordinates"],
                "rank_fraction_after_box_saturation": summed["jacobian_rank_sum"] / potential if potential else None,
                "proper_gradient_norm_fraction_retained": math.sqrt(after_sq / before_sq) if before_sq else None,
                "proper_gradient_energy_fraction_retained": after_sq / before_sq if before_sq else None,
                "orthogonal_energy_decomposition": {
                    name: {
                        "energy": value,
                        "fraction": value / before_sq if before_sq else None,
                    }
                    for name, value in decomposition_energy.items()
                },
                "orthogonal_energy_closure_absolute_error": closure_error,
                "orthogonal_energy_closure_relative_error": closure_relative,
                "interior_budget_fraction": summed["interior_budget_blocks"] / active_blocks,
                "zero_budget_fraction": summed["zero_budget_blocks"] / active_blocks,
                "one_budget_fraction": summed["one_budget_blocks"] / active_blocks,
                "informative_block_fraction_with_zero_retained_gradient": (
                    summed["informative_blocks_with_zero_retained_gradient"]
                    / summed["informative_blocks"]
                    if summed["informative_blocks"] else None
                ),
                "free_count_histogram": {
                    str(i): sum(row["free_count_histogram"][str(i)] for row in rows)
                    for i in range(5)
                },
                "jacobian_rank_histogram": {
                    str(i): sum(row["jacobian_rank_histogram"][str(i)] for row in rows)
                    for i in range(4)
                },
                "per_case": rows,
                "derivative_interpretation": rows[0]["derivative_interpretation"],
            }
        branches[branch] = {
            "mean_proper_objective": sum(objective_values) / 12,
            "mean_fair_crps": sum(crps_values) / 12,
            "mean_joint_energy": sum(energy_values) / 12,
            "sic_leads": per_lead,
        }
        del forecast_normalized, coarse_normalized
    result = {
        "schema_version": config["schema_version"],
        "status": "complete",
        "scientific_role": "frozen_local_decoder_gradient_attribution_not_calibration",
        "split": "valid",
        "panel": "frozen_12_case_validation_2022_development_reuse",
        "cases": 12,
        "members": 8,
        "optimizer_steps": 0,
        "sampling_performed": False,
        "test_2023_used": False,
        "decoder_backward": "exact active-set Jacobian I-11^T/k; no STE",
        "absolute_value_tie_subgradient": "PyTorch abs implementation-selected subgradient 0 at exact CRPS ties",
        "code_identity": code_identity,
        "provenance": provenance,
        "source_sha256": {
            "fixed_inputs": fixed_spec["sha256"],
            **{name: spec["sha256"] for name, spec in config["branches"].items()},
        },
        "branches": branches,
    }
    return result


def main() -> None:
    signal.signal(signal.SIGTERM, _terminate)
    signal.signal(signal.SIGINT, _terminate)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
