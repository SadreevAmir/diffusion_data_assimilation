"""Frozen CPU decomposition of d6 SIT ranks at the physical zero atom."""

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
from .direct_dynamics_cascade import masked_block_average
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


def _conditional_distribution(
    members_normalized: torch.Tensor,
    truth_normalized: torch.Tensor,
    fraction: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> dict[str, Any]:
    if members_normalized.ndim != 5 or members_normalized.shape[2] != 1:
        raise ValueError("rank-atom audit requires [case,member,1,y,x]")
    if truth_normalized.shape != members_normalized[:, 0].shape:
        raise ValueError("rank-atom truth shape differs from members")
    if fraction.shape != truth_normalized.shape:
        raise ValueError("rank-atom fraction shape differs from truth")
    if not torch.isfinite(members_normalized).all() or not torch.isfinite(truth_normalized).all():
        raise FloatingPointError("rank-atom inputs contain NaN/Inf")
    if not torch.isfinite(fraction).all() or torch.any((fraction < 0) | (fraction > 1)):
        raise ValueError("rank-atom weights must lie in [0,1]")
    encoded_zero = (
        torch.zeros((), dtype=truth_normalized.dtype) - mean.to(truth_normalized)
    ) / std.to(truth_normalized)
    valid = fraction > 0
    if torch.any(valid & (truth_normalized < encoded_zero)):
        raise ValueError("frozen d6 SIT truth contains negative physical values")
    selectors = {
        "exact_zero_truth": valid & (truth_normalized == encoded_zero),
        "positive_truth": valid & (truth_normalized > encoded_zero),
    }
    if not torch.equal(selectors["exact_zero_truth"] | selectors["positive_truth"], valid):
        raise ValueError("zero/positive truth selectors do not partition valid d6 SIT")

    encoded_0p01 = (
        torch.as_tensor(0.01, dtype=truth_normalized.dtype) - mean.to(truth_normalized)
    ) / std.to(truth_normalized)
    members = members_normalized.double()
    truth = truth_normalized.double()
    count = members.shape[1]
    less = (members < truth[:, None]).sum(dim=1)
    equal = (members == truth[:, None]).sum(dim=1)
    result: dict[str, Any] = {
        "encoded_zero_normalized": float(encoded_zero),
        "conditional_uniformity_claim_permitted": False,
        "groups": {},
        "per_case_direct_rank_frequencies": [],
        "per_case_reconstructed_rank_frequencies": [],
    }
    reconstructed_cases = []
    direct_cases = []
    group_case_data: dict[str, list[dict[str, Any] | None]] = {name: [] for name in selectors}
    for case in range(members.shape[0]):
        total_weight = fraction[case].double().sum()
        if total_weight <= 0:
            raise ValueError("rank-atom case has no valid ocean")

        def rank_frequency(weight: torch.Tensor) -> torch.Tensor:
            mass = torch.zeros(count + 1, dtype=torch.float64)
            denominator = weight.sum()
            if denominator <= 0:
                return torch.full_like(mass, float("nan"))
            for rank in range(count + 1):
                selected_rank = (rank >= less[case]) & (rank <= less[case] + equal[case])
                mass[rank] = torch.where(
                    selected_rank,
                    weight / (equal[case].double() + 1),
                    torch.zeros_like(weight),
                ).sum()
            return mass / denominator

        direct = rank_frequency(fraction[case].double())
        direct_cases.append(direct)
        reconstructed = torch.zeros_like(direct)
        for group, selector in selectors.items():
            weight = fraction[case].double() * selector[case]
            selected_weight = weight.sum()
            if selected_weight <= 0:
                group_case_data[group].append(None)
                continue
            rank = rank_frequency(weight)
            group_mass = selected_weight / total_weight
            selected_members = selector[case][None].expand(count, -1, -1, -1)
            selected_weight_members = weight[None].expand(count, -1, -1, -1)
            denominator = selected_weight_members.sum()
            categories = {
                "lt_0": members_normalized[case] < encoded_zero,
                "eq_0": members_normalized[case] == encoded_zero,
                "gt_0_le_0p01": (
                    (members_normalized[case] > encoded_zero)
                    & (members_normalized[case] <= encoded_0p01)
                ),
                "gt_0p01": members_normalized[case] > encoded_0p01,
            }
            probabilities = {
                name: float((selected_weight_members * event * selected_members).sum() / denominator)
                for name, event in categories.items()
            }
            if abs(sum(probabilities.values()) - 1.0) > 1e-12:
                raise ValueError("member value categories do not partition the selected group")
            group_case_data[group].append(
                {
                    "truth_weight_fraction": float(group_mass),
                    "weighted_truth_point_count": float(selected_weight),
                    "rank_frequencies": rank.tolist(),
                    "member_value_probabilities": probabilities,
                }
            )
            reconstructed += group_mass * rank
        reconstructed_cases.append(reconstructed)
    direct_tensor = torch.stack(direct_cases)
    reconstructed_tensor = torch.stack(reconstructed_cases)
    reconstruction_error = float((direct_tensor - reconstructed_tensor).abs().max())
    if reconstruction_error > 2e-15:
        raise ValueError("conditional rank mixture does not reconstruct the direct histogram")
    result["rank_mixture_reconstruction_max_abs"] = reconstruction_error
    result["per_case_direct_rank_frequencies"] = direct_tensor.tolist()
    result["per_case_reconstructed_rank_frequencies"] = reconstructed_tensor.tolist()
    result["case_equal_direct_rank_frequencies"] = direct_tensor.mean(dim=0).tolist()
    result["case_equal_reconstructed_rank_frequencies"] = reconstructed_tensor.mean(dim=0).tolist()
    for group, rows in group_case_data.items():
        nonempty = [row for row in rows if row is not None]
        result["groups"][group] = {
            "case_count": len(nonempty),
            "per_case": rows,
            "case_equal_rank_frequencies_over_nonempty_cases": [
                sum(row["rank_frequencies"][rank] for row in nonempty) / len(nonempty)
                for rank in range(count + 1)
            ] if nonempty else None,
            "case_equal_member_value_probabilities_over_nonempty_cases": {
                category: sum(row["member_value_probabilities"][category] for row in nonempty) / len(nonempty)
                for category in ("lt_0", "eq_0", "gt_0_le_0p01", "gt_0p01")
            } if nonempty else None,
        }
    return result


def run(config_path: Path, output: Path) -> dict[str, Any]:
    torch.set_num_threads(6)
    torch.set_num_interop_threads(1)
    if torch.get_num_threads() != 6 or torch.get_num_interop_threads() != 1:
        raise RuntimeError("failed to enforce the reviewed 6/1 CPU thread envelope")
    if torch.cuda.is_available() or os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
        raise RuntimeError("d6 SIT rank-atom audit must run with CUDA_VISIBLE_DEVICES empty")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite {output}")
    config = load_json(config_path)
    if config.get("schema_version") != "d6_sit_rank_atom_audit_v1":
        raise ValueError("unreviewed d6 SIT rank-atom audit schema")
    source = config["source_audit"]
    for key in ("config", "status", "metrics"):
        path = Path(source[key]["path"])
        if not path.is_file() or _sha256(path) != source[key]["sha256"]:
            raise ValueError(f"source audit {key} SHA mismatch")
    source_status = load_json(source["status"]["path"])
    if source_status.get("status") != "complete" or source_status.get("metrics_sha256") != source["metrics"]["sha256"]:
        raise ValueError("source coarse/fine audit is not terminal complete")
    evidence_config = load_json(source["config"]["path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = output.with_name(f"{output.stem}.metrics.json")
    if metrics_path.exists() or metrics_path.is_symlink():
        raise FileExistsError(f"refusing to overwrite {metrics_path}")
    reservation = {"status": "reserved", "config_path": str(config_path), "config_sha256": _sha256(config_path)}
    with output.open("x") as handle:
        json.dump(reservation, handle, indent=2, allow_nan=False); handle.write("\n")
    tracker = None
    try:
        code_identity = _clean_code_identity(Path(__file__).resolve().parents[1])
        tracker = ClearMLTracker(
            config["project_name"], f'{config["task_name"]}-{output.parent.name}',
            tags=config["clearml"]["tags"], env_path=config["clearml"]["env_path"],
        )
        tracker.connect("audit_contract", config)
        _strict_atomic_json(output, {**reservation, "status": "running", "code_identity": code_identity, "clearml_task_id": str(tracker.task.id)})
        evidence = {name: _load_frozen(spec) for name, spec in evidence_config["evidence"].items()}
        canonical = evidence["raw"]
        for name, payload in evidence.items():
            _validate_payload(name, payload, evidence_config["evidence"][name], evidence_config)
            _require_equal(canonical, payload, name)
        means, stds, normalization = _load_normalization(evidence_config)
        mask = canonical["valid_mask"].float()
        truth = canonical["truth_normalized"][:, 3:4].float()
        truth_coarse, fraction = masked_block_average(truth, mask)
        result: dict[str, Any] = {
            "status": "numerical_results_complete_pending_finalization",
            "scientific_role": "frozen_development_diagnostic_not_champion_selection",
            "code_identity": code_identity, "source_audit": source,
            "normalization_source": normalization, "output": "d6_sit", "candidates": {},
        }
        for name in ("raw", "ordinary64", "threshold64"):
            payload = evidence[name]
            result["candidates"][name] = {
                "coarse": _conditional_distribution(
                    payload["coarse_normalized"][:, :, 3:4].float(), truth_coarse,
                    fraction, means[3], stds[3],
                ),
                "full_resolution": _conditional_distribution(
                    payload["forecast_normalized"][:, :, 3:4].float(), truth,
                    mask, means[3], stds[3],
                ),
            }
        _strict_atomic_json(metrics_path, result)
        tracker.upload_artifact("d6_sit_rank_atom_audit", metrics_path)
        _finish_success(
            tracker, output, metrics_path,
            {**reservation, "code_identity": code_identity, "clearml_task_id": str(tracker.task.id), "scientific_role": result["scientific_role"]},
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
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.config.resolve(), arguments.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
