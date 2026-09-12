"""CPU-only d9-SIC rank decomposition for a frozen paired cascade replay."""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

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


def _rank_frequency(
    less: torch.Tensor, equal: torch.Tensor, weight: torch.Tensor, members: int
) -> torch.Tensor:
    denominator = weight.sum()
    if denominator <= 0:
        raise ValueError("rank group has no weight")
    result = torch.zeros(members + 1, dtype=torch.float64)
    for rank in range(members + 1):
        selected = (rank >= less) & (rank <= less + equal)
        result[rank] = torch.where(
            selected, weight / (equal.double() + 1.0), torch.zeros_like(weight)
        ).sum()
    return result / denominator


def _fine_atom_selectors(
    truth: torch.Tensor,
    valid: torch.Tensor,
    *,
    physical_mean: float,
    physical_std: float,
) -> dict[str, torch.Tensor]:
    """Partition fine-grid truth without tolerances or value clipping."""
    mean = torch.as_tensor(physical_mean, dtype=truth.dtype, device=truth.device)
    std = torch.as_tensor(physical_std, dtype=truth.dtype, device=truth.device)
    if not math.isfinite(float(mean)) or not math.isfinite(float(std)) or float(std) <= 0:
        raise ValueError("SIC normalization must have finite mean and positive finite std")
    encoded_zero = -mean / std
    encoded_one = (torch.ones((), dtype=truth.dtype, device=truth.device) - mean) / std
    active = valid > 0
    if torch.any(active & ((truth < encoded_zero) | (truth > encoded_one))):
        raise ValueError("truth lies outside physical SIC [0,1]")
    selectors = {
        "exact_zero": active & (truth == encoded_zero),
        "interior": active & (truth > encoded_zero) & (truth < encoded_one),
        "exact_one": active & (truth == encoded_one),
    }
    if not torch.equal(
        selectors["exact_zero"] | selectors["interior"] | selectors["exact_one"], active
    ):
        raise ValueError("SIC atom selectors do not partition valid truth")
    return selectors


def _coarse_atom_selectors(
    fine_selectors: dict[str, torch.Tensor], fine_valid: torch.Tensor, factor: int = 2
) -> dict[str, torch.Tensor]:
    """Derive exact coarse atoms from fine membership, never pooled FP values."""
    active_fraction = F.avg_pool2d(
        (fine_valid > 0).to(torch.float32), kernel_size=factor, stride=factor
    )
    active = active_fraction > 0

    def all_active(name: str) -> torch.Tensor:
        group_fraction = F.avg_pool2d(
            fine_selectors[name].to(torch.float32), kernel_size=factor, stride=factor
        )
        return active & (group_fraction == active_fraction)

    exact_zero = all_active("exact_zero")
    exact_one = all_active("exact_one")
    return {
        "exact_zero": exact_zero,
        "interior": active & ~exact_zero & ~exact_one,
        "exact_one": exact_one,
    }


def conditional_rank_atom_metrics(
    members: torch.Tensor,
    truth: torch.Tensor,
    fraction: torch.Tensor,
    *,
    physical_mean: float,
    physical_std: float,
    selectors: dict[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    if members.ndim != 5 or members.shape[2] != 1 or members.shape[1] < 2:
        raise ValueError("expected members [case,member,1,y,x]")
    if truth.shape != members[:, 0].shape or fraction.shape != truth.shape:
        raise ValueError("truth/fraction shape mismatch")
    valid = fraction > 0
    if not torch.isfinite(members).all() or not torch.isfinite(truth).all():
        raise FloatingPointError("rank audit contains NaN/Inf")
    if not torch.isfinite(fraction).all() or torch.any((fraction < 0) | (fraction > 1)):
        raise ValueError("invalid ocean fractions")
    mean = torch.as_tensor(physical_mean, dtype=truth.dtype)
    std = torch.as_tensor(physical_std, dtype=truth.dtype)
    if not math.isfinite(float(mean)) or not math.isfinite(float(std)) or float(std) <= 0:
        raise ValueError("SIC normalization must have finite mean and positive finite std")
    encoded_zero = -mean / std
    encoded_one = (torch.ones((), dtype=truth.dtype) - mean) / std
    if selectors is None:
        selectors = _fine_atom_selectors(
            truth,
            fraction,
            physical_mean=physical_mean,
            physical_std=physical_std,
        )
    if set(selectors) != {"exact_zero", "interior", "exact_one"}:
        raise ValueError("unexpected SIC atom selector names")
    for selector in selectors.values():
        if selector.shape != truth.shape or selector.dtype != torch.bool:
            raise ValueError("SIC atom selectors must be boolean and match truth")
    partition = selectors["exact_zero"] | selectors["interior"] | selectors["exact_one"]
    if not torch.equal(partition, valid):
        raise ValueError("SIC atom selectors do not partition valid truth")

    values = members.double()
    target = truth.double()
    less = (values < target[:, None]).sum(dim=1)
    equal = (values == target[:, None]).sum(dim=1)
    member_count = members.shape[1]
    direct_cases = []
    reconstructed_cases = []
    groups: dict[str, list[dict[str, Any] | None]] = {name: [] for name in selectors}
    for case in range(members.shape[0]):
        total_weight = fraction[case].double().sum()
        direct = _rank_frequency(
            less[case], equal[case], fraction[case].double(), member_count
        )
        reconstructed = torch.zeros_like(direct)
        for name, selector in selectors.items():
            weight = fraction[case].double() * selector[case]
            selected_weight = weight.sum()
            if selected_weight <= 0:
                groups[name].append(None)
                continue
            rank = _rank_frequency(less[case], equal[case], weight, member_count)
            group_fraction = selected_weight / total_weight
            observation = (values[case] - target[case]).abs().mean(dim=0)
            pair_sum = torch.zeros_like(observation)
            for left in range(member_count):
                for right in range(left + 1, member_count):
                    pair_sum += (values[case, left] - values[case, right]).abs()
            fair_crps = observation - pair_sum / (member_count * (member_count - 1))
            bias = values[case].mean(dim=0) - target[case]
            groups[name].append({
                "truth_weight_fraction": float(group_fraction),
                "rank_frequencies": rank.tolist(),
                "signed_bias_normalized": float((bias * weight).sum() / weight.sum()),
                "signed_bias_physical": float((bias * weight).sum() / weight.sum() * std.double()),
                "fair_crps_normalized": float((fair_crps * weight).sum() / weight.sum()),
                "fair_crps_physical": float((fair_crps * weight).sum() / weight.sum() * std.double()),
            })
            reconstructed += group_fraction * rank
        direct_cases.append(direct)
        reconstructed_cases.append(reconstructed)
    direct_tensor = torch.stack(direct_cases)
    reconstructed_tensor = torch.stack(reconstructed_cases)
    error = float((direct_tensor - reconstructed_tensor).abs().max())
    if error > 2e-15:
        raise RuntimeError("conditional groups do not reconstruct the direct histogram")
    uniform = torch.full((member_count + 1,), 1.0 / (member_count + 1), dtype=torch.float64)
    aggregate = direct_tensor.mean(dim=0)
    result = {
        "encoded_bounds_normalized": [float(encoded_zero), float(encoded_one)],
        "case_equal_rank_frequencies": aggregate.tolist(),
        "rank_tv_to_uniform": float(0.5 * (aggregate - uniform).abs().sum()),
        "rank_mixture_reconstruction_max_abs": error,
        "groups": {},
    }
    for name, rows in groups.items():
        nonempty = [row for row in rows if row is not None]
        result["groups"][name] = {
            "case_count": len(nonempty),
            "per_case": rows,
            "case_equal_rank_frequencies_over_nonempty_cases": (
                [
                    sum(row["rank_frequencies"][rank] for row in nonempty) / len(nonempty)
                    for rank in range(member_count + 1)
                ]
                if nonempty else None
            ),
            "mean_truth_weight_fraction_over_all_cases": float(
                sum(0.0 if row is None else row["truth_weight_fraction"] for row in rows)
                / len(rows)
            ),
            "case_equal_signed_bias_physical_over_nonempty_cases": (
                sum(row["signed_bias_physical"] for row in nonempty) / len(nonempty)
                if nonempty else None
            ),
            "case_equal_fair_crps_physical_over_nonempty_cases": (
                sum(row["fair_crps_physical"] for row in nonempty) / len(nonempty)
                if nonempty else None
            ),
        }
    return result


def run(config_path: Path, output: Path) -> dict[str, Any]:
    torch.set_num_threads(6)
    torch.set_num_interop_threads(1)
    if torch.get_num_threads() != 6 or torch.get_num_interop_threads() != 1:
        raise RuntimeError("failed to enforce the reviewed 6/1 CPU thread envelope")
    if torch.cuda.is_available() or os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
        raise RuntimeError("d9 SIC rank/atom audit requires CUDA_VISIBLE_DEVICES empty")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite {output}")
    config = load_json(config_path)
    if config.get("schema_version") != "d9_sic_paired_rank_atom_audit_v1":
        raise ValueError("unreviewed d9 SIC audit schema")
    output.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = output.with_name(f"{output.stem}.metrics.json")
    if metrics_path.exists() or metrics_path.is_symlink():
        raise FileExistsError(f"refusing to overwrite {metrics_path}")
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
        source_bindings: dict[str, Any] = {}
        for name, spec in config["source_gate"].items():
            path = Path(spec["path"])
            if not path.is_file() or _sha256(path) != spec["sha256"]:
                raise ValueError(f"source gate SHA mismatch: {name}")
            source_bindings[name] = {
                "path": str(path),
                "sha256": spec["sha256"],
            }
        contract = load_json(config["source_gate"]["contract"]["path"])
        if contract.get("schema_version") != "terminal_proper_refinement_paired_e2e_v1":
            raise ValueError("source gate contract schema mismatch")
        for key in ("case_indices", "case_ids", "coarse", "fine"):
            if contract.get(key) != config[key]:
                raise ValueError(f"source gate contract differs in {key}")
        if contract.get("evidence_sha256", {}).get("raw_ema9711_colored2048_raw.pt") != config["evidence"]["raw"]["sha256"]:
            raise ValueError("raw evidence is not bound by source gate contract")
        if contract.get("evidence_sha256", {}).get("multiscale64_ema9711_colored2048_raw.pt") != config["evidence"]["candidate"]["sha256"]:
            raise ValueError("candidate evidence is not bound by source gate contract")
        means, stds, normalization = _load_normalization(config)
        sic_mean, sic_std = float(means[4]), float(stds[4])
        if not math.isfinite(sic_mean) or not math.isfinite(sic_std) or sic_std <= 0:
            raise ValueError("invalid SHA-bound SIC normalization")

        payloads = {
            label: _load_frozen(spec) for label, spec in config["evidence"].items()
        }
        raw = payloads["raw"]
        for label, payload in payloads.items():
            _validate_payload(label, payload, config["evidence"][label], config)
            _require_equal(raw, payload, label)

        code_identity = _clean_code_identity(Path(__file__).resolve().parents[1])
        tracker = ClearMLTracker(
            config["project_name"], f'{config["task_name"]}-{output.parent.name}',
            tags=config["clearml"]["tags"], env_path=config["clearml"]["env_path"],
        )
        tracker.connect("audit_contract", config)
        _strict_atomic_json(
            output,
            {
                **reservation,
                "status": "running",
                "code_identity": code_identity,
                "clearml_task_id": str(tracker.task.id),
            },
        )
        truth = raw["truth_normalized"][:, 4:5].float()
        valid = raw["valid_mask"].float()
        truth_coarse, fraction_coarse = masked_block_average(truth, valid)
        fine_selectors = _fine_atom_selectors(
            truth, valid, physical_mean=sic_mean, physical_std=sic_std
        )
        coarse_selectors = _coarse_atom_selectors(fine_selectors, valid)
        result: dict[str, Any] = {
            "status": "numerical_results_complete_pending_finalization",
            "scientific_role": "frozen_development_diagnostic_not_selection",
            "code_identity": code_identity,
            "config_sha256": reservation["config_sha256"],
            "source_gate": source_bindings,
            "normalization_source": normalization,
            "channel": "d9_sic",
            "candidates": {},
        }
        for label, payload in payloads.items():
            kwargs = {
                "physical_mean": sic_mean,
                "physical_std": sic_std,
            }
            result["candidates"][label] = {
                "coarse": conditional_rank_atom_metrics(
                    payload["coarse_normalized"][:, :, 4:5].float(),
                    truth_coarse, fraction_coarse, selectors=coarse_selectors, **kwargs,
                ),
                "final": conditional_rank_atom_metrics(
                    payload["forecast_normalized"][:, :, 4:5].float(), truth, valid,
                    selectors=fine_selectors, **kwargs,
                ),
            }
        _strict_atomic_json(metrics_path, result)
        tracker.upload_artifact("d9_sic_rank_atom_audit", metrics_path)
        _finish_success(
            tracker,
            output,
            metrics_path,
            {
                **reservation,
                "code_identity": code_identity,
                "clearml_task_id": str(tracker.task.id),
                "scientific_role": result["scientific_role"],
                "source_gate": source_bindings,
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
