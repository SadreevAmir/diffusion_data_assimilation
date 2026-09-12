"""Frozen attribution of SIC zero-atom deficit to coarse budget and allocation."""

from __future__ import annotations

import argparse
import json
import os
import signal
from pathlib import Path
from typing import Any

import torch

from .clearml_tracking import ClearMLTracker
from .config import load_json
from .direct_dynamics_cascade_coarse_fine_boundary_audit import (
    _finish_success,
    _load_frozen,
    _load_normalization,
    _record_failure,
    _require_equal,
    _sha256,
    _strict_atomic_json,
    _terminate,
    _validate_payload,
)
from .direct_dynamics_cascade_fine_training import _clean_code_identity
from .direct_dynamics_sic_support_decoder import (
    project_masked_blocks_to_unit_interval_mean,
)
from .direct_dynamics_sic_support_decoder_scoring import (
    SIC_CHANNELS,
    _physical,
    _source_gate_is_complete,
    canonical_physical_decode,
)
from .direct_dynamics_sit_support_decoder import _to_blocks


HORIZONS = ("d3_sic", "d6_sic", "d9_sic")


def maximum_zero_fraction(
    coarse_sic: torch.Tensor, valid_count: torch.Tensor
) -> torch.Tensor:
    """Maximum feasible zero fraction under sum=n*clip(C,0,1), 0<=z<=1."""
    if coarse_sic.shape != valid_count.shape:
        raise ValueError("coarse SIC and valid count must have identical shapes")
    if torch.any(valid_count < 0) or torch.any(valid_count > 4):
        raise ValueError("valid count must be in [0,4] for 2x2 blocks")
    count = valid_count.double()
    budget = count * coarse_sic.double().clamp(0.0, 1.0)
    required_positive = torch.ceil(budget)
    result = torch.where(count > 0, 1.0 - required_positive / count, torch.zeros_like(count))
    if torch.any((result[count > 0] < 0) | (result[count > 0] > 1)):
        raise RuntimeError("invalid maximum zero fraction")
    return result


def fully_zero_truth_blocks(
    truth_sic: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select active 2x2 blocks whose every valid truth SIC cell is exactly zero."""
    if truth_sic.ndim != 4 or truth_sic.shape[1] != 1 or mask.shape != truth_sic.shape:
        raise ValueError("truth SIC and mask must be matching [case,1,y,x]")
    truth_blocks = _to_blocks(truth_sic.double(), 2)
    mask_blocks = _to_blocks(mask.double(), 2) > 0
    count = mask_blocks.sum(dim=-1)
    selector = (count > 0) & ~torch.any(mask_blocks & (truth_blocks != 0), dim=-1)
    return selector, count


def actual_zero_fraction(decoded: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-member exact-zero fraction within each active 2x2 block."""
    if decoded.ndim != 5 or decoded.shape[2] != 1:
        raise ValueError("decoded SIC must be [case,member,1,y,x]")
    if mask.shape != (decoded.shape[0], 1, *decoded.shape[-2:]):
        raise ValueError("mask shape differs from decoded SIC")
    cases, members = decoded.shape[:2]
    expanded_mask = mask[:, None].expand(-1, members, -1, -1, -1)
    blocks = _to_blocks(decoded.flatten(0, 1).double(), 2).unflatten(0, (cases, members))
    mask_blocks = _to_blocks(expanded_mask.flatten(0, 1).double(), 2).unflatten(
        0, (cases, members)
    ) > 0
    count = mask_blocks.sum(dim=-1)
    zeros = (mask_blocks & (blocks == 0)).sum(dim=-1)
    return torch.where(count > 0, zeros.double() / count.double(), torch.zeros_like(zeros, dtype=torch.float64))


def attribute_zero_deficit(
    decoded: torch.Tensor,
    coarse_sic: torch.Tensor,
    truth_sic: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, Any]:
    """Case-equal decomposition on blocks whose valid truth cells are all zero."""
    selector, count_case = fully_zero_truth_blocks(truth_sic, mask)
    cases, members = decoded.shape[:2]
    selector_member = selector[:, None].expand(-1, members, -1, -1, -1)
    count = count_case[:, None].expand(-1, members, -1, -1, -1)
    if coarse_sic.shape != selector_member.shape:
        raise ValueError("coarse SIC shape differs from expanded truth-block selector")
    selected_cells_per_case = (selector * count_case).sum(dim=(-3, -2, -1))
    if torch.any(selected_cells_per_case == 0):
        raise ValueError("every frozen case must contain fully-zero truth SIC ocean cells")
    actual = actual_zero_fraction(decoded, mask)
    possible = maximum_zero_fraction(coarse_sic, count)
    actual_nonzero = 1.0 - actual
    mandatory_nonzero = 1.0 - possible
    allocation_excess_nonzero = possible - actual
    identity_error = actual_nonzero - mandatory_nonzero - allocation_excess_nonzero
    selected_identity_error = identity_error[selector_member]
    selected_allocation_excess = allocation_excess_nonzero[selector_member]
    if selected_identity_error.numel() == 0:
        raise ValueError("no fully-zero truth SIC blocks were selected")
    if float(selected_identity_error.abs().max()) > 1e-14:
        raise RuntimeError("zero-deficit decomposition identity failed")
    if float(selected_allocation_excess.min()) < -1e-14:
        raise RuntimeError("actual allocation exceeds mathematical maximum zero fraction")

    def case_member_mean(values: torch.Tensor) -> torch.Tensor:
        # Weight blocks by their valid-ocean cell count so this attributes the
        # pixel-level zero mass measured by the parent calibration audit.
        weight = selector_member * count
        numerator = (values * weight).sum(dim=(-3, -2, -1))
        denominator = weight.sum(dim=(-3, -2, -1))
        if torch.any(denominator == 0):
            raise ValueError("a frozen case has no fully-zero truth SIC blocks")
        return numerator / denominator

    rows = {
        "actual_zero_fraction": case_member_mean(actual),
        "maximum_feasible_zero_fraction": case_member_mean(possible),
        "actual_nonzero_fraction": case_member_mean(actual_nonzero),
        "mandatory_nonzero_from_coarse_budget": case_member_mean(mandatory_nonzero),
        "excess_nonzero_from_within_block_allocation": case_member_mean(allocation_excess_nonzero),
    }
    case_equal: dict[str, float | None | bool] = {
        name: float(values.mean()) for name, values in rows.items()
    }
    denominator = case_equal["actual_nonzero_fraction"]
    if not isinstance(denominator, float) or denominator == 0.0:
        case_equal["zero_deficit"] = True
        case_equal["mandatory_share_of_actual_nonzero"] = None
        case_equal["allocation_share_of_actual_nonzero"] = None
    else:
        case_equal["zero_deficit"] = False
        mandatory = case_equal["mandatory_nonzero_from_coarse_budget"]
        allocation = case_equal["excess_nonzero_from_within_block_allocation"]
        if not isinstance(mandatory, float) or not isinstance(allocation, float):
            raise RuntimeError("nonzero attribution components are not numeric")
        case_equal["mandatory_share_of_actual_nonzero"] = mandatory / denominator
        case_equal["allocation_share_of_actual_nonzero"] = allocation / denominator
    return {
        "fully_zero_truth_block_count_per_case": selector.sum(dim=(-3, -2, -1)).tolist(),
        "fully_zero_truth_ocean_cell_count_per_case": selected_cells_per_case.tolist(),
        "case_member": {name: values.tolist() for name, values in rows.items()},
        "case_equal": case_equal,
        "maximum_identity_error": float(selected_identity_error.abs().max()),
    }


def _load_bound_source(config: dict[str, Any], repo: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    spec = config["source_scoring_config"]
    path = repo / spec["path"]
    if not path.is_file() or _sha256(path) != spec["sha256"]:
        raise ValueError("source scoring config SHA mismatch")
    source_config = load_json(path)
    source_gate = _source_gate_is_complete(source_config)
    audit = config["source_decoder_audit"]
    for name in ("status", "metrics"):
        item = audit[name]
        item_path = Path(item["path"])
        if not item_path.is_file() or _sha256(item_path) != item["sha256"]:
            raise ValueError(f"source decoder audit {name} SHA mismatch")
    status = load_json(audit["status"]["path"])
    metrics = load_json(audit["metrics"]["path"])
    if status.get("status") != "complete" or status.get("metrics_sha256") != audit["metrics"]["sha256"]:
        raise ValueError("source decoder audit is not durably complete")
    if metrics.get("code_identity", {}).get("git_commit") != audit["code_commit"]:
        raise ValueError("source decoder audit code identity differs")
    if metrics.get("gate", {}).get("passed_pending_visual_review") is not False:
        raise ValueError("source decoder audit did not preserve the expected rejection")
    if metrics.get("gate", {}).get("sic_exact_support_passed") is not True:
        raise ValueError("source decoder audit did not establish exact SIC support")
    return source_config, {"source_gate": source_gate, "source_decoder_audit": audit}


def run(config_path: Path, output: Path) -> dict[str, Any]:
    torch.set_num_threads(6)
    torch.set_num_interop_threads(1)
    if torch.get_num_threads() != 6 or torch.get_num_interop_threads() != 1:
        raise RuntimeError("failed to enforce reviewed 6/1 CPU thread envelope")
    if torch.cuda.is_available() or os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
        raise RuntimeError("coarse-budget attribution requires CUDA_VISIBLE_DEVICES empty")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite {output}")
    config = load_json(config_path)
    if config.get("schema_version") != "sic_coarse_budget_attribution_v1":
        raise ValueError("unreviewed coarse-budget attribution schema")
    output.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = output.with_name(f"{output.stem}.metrics.json")
    if metrics_path.exists() or metrics_path.is_symlink():
        raise FileExistsError(f"refusing to overwrite {metrics_path}")
    reservation = {
        "status": "reserved", "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
    }
    with output.open("x") as handle:
        json.dump(reservation, handle, indent=2, allow_nan=False)
        handle.write("\n")
    tracker = None
    try:
        repo = Path(__file__).resolve().parents[1]
        source_config, provenance = _load_bound_source(config, repo)
        code_identity = _clean_code_identity(repo)
        tracker = ClearMLTracker(
            config["project_name"], f'{config["task_name"]}-{output.parent.name}',
            tags=config["clearml"]["tags"], env_path=config["clearml"]["env_path"],
        )
        tracker.connect("audit_contract", config)
        _strict_atomic_json(output, {
            **reservation, "status": "running", "code_identity": code_identity,
            "clearml_task_id": str(tracker.task.id),
        })

        evidence = {
            name: _load_frozen(spec) for name, spec in source_config["evidence"].items()
        }
        raw = evidence["raw"]
        for name, payload in evidence.items():
            _validate_payload(name, payload, source_config["evidence"][name], source_config)
            _require_equal(raw, payload, name)
        means, stds, normalization = _load_normalization(source_config)
        mask = raw["valid_mask"].double()
        truth = canonical_physical_decode(raw["truth_normalized"], means, stds)
        variants: dict[str, Any] = {}
        for name, payload in evidence.items():
            physical, coarse = _physical(payload, means, stds)
            horizon_rows = {}
            for channel, horizon in zip(SIC_CHANNELS, HORIZONS):
                cases, members = physical.shape[:2]
                expanded_mask = mask[:, None].expand(-1, members, -1, -1, -1).flatten(0, 1)
                decoded = project_masked_blocks_to_unit_interval_mean(
                    physical[:, :, channel : channel + 1].flatten(0, 1),
                    coarse[:, :, channel : channel + 1].flatten(0, 1),
                    expanded_mask,
                ).unflatten(0, (cases, members))
                horizon_rows[horizon] = attribute_zero_deficit(
                    decoded,
                    coarse[:, :, channel : channel + 1],
                    truth[:, channel : channel + 1],
                    mask,
                )
            variants[name] = horizon_rows

        comparison = {}
        for horizon in HORIZONS:
            raw_row = variants["raw"][horizon]["case_equal"]
            ordinary_row = variants["ordinary"][horizon]["case_equal"]
            comparison[horizon] = {
                f"ordinary_minus_raw_{key}": ordinary_row[key] - raw_row[key]
                for key in (
                    "actual_zero_fraction",
                    "maximum_feasible_zero_fraction",
                    "mandatory_nonzero_from_coarse_budget",
                    "excess_nonzero_from_within_block_allocation",
                )
            }
        result = {
            "status": "complete",
            "scientific_role": "frozen_development_attribution_not_calibrator",
            "training_performed": False,
            "sampling_performed": False,
            "code_identity": code_identity,
            "provenance": provenance,
            "normalization_source": normalization,
            "identity": "actual_nonzero = mandatory_nonzero_from_coarse_budget + excess_nonzero_from_within_block_allocation",
            "variants": variants,
            "ordinary_minus_raw": comparison,
        }
        _strict_atomic_json(metrics_path, result)
        for variant, horizons in variants.items():
            for iteration, horizon in enumerate(HORIZONS):
                row = horizons[horizon]["case_equal"]
                for metric in (
                    "actual_zero_fraction", "maximum_feasible_zero_fraction",
                    "mandatory_share_of_actual_nonzero", "allocation_share_of_actual_nonzero",
                ):
                    value = row[metric]
                    if value is not None:
                        tracker.report_scalar(
                            "sic_zero_atom_attribution", f"{variant}/{metric}", value, iteration
                        )
        tracker.upload_artifact("sic_coarse_budget_attribution", metrics_path)
        _finish_success(
            tracker, output, metrics_path,
            {
                **reservation, "code_identity": code_identity,
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
        raise
    finally:
        if tracker is not None:
            try:
                tracker.close()
            except Exception:
                pass


def main() -> None:
    signal.signal(signal.SIGTERM, _terminate)
    signal.signal(signal.SIGINT, _terminate)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve(), args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
