"""Frozen CPU probe for nonnegative fine SIT under an unchanged coarse mean."""

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
from .direct_dynamics_sit_left_censor_audit import decode_with_exact_sit_zero


SIT_CHANNELS = (1, 3, 5)
OUTPUTS = ("d3_sic", "d3_sit", "d6_sic", "d6_sit", "d9_sic", "d9_sit")


def support_decomposition(
    fine_sit: torch.Tensor,
    fine_mask: torch.Tensor,
    source_coarse_sit: torch.Tensor,
    fp32_ulps: int,
) -> dict[str, torch.Tensor | float]:
    """Decompose the coarse change caused by projecting fine SIT to [0, inf)."""
    if fine_sit.ndim != 4 or fine_sit.shape[1] != 1:
        raise ValueError("fine_sit must have shape [sample,1,y,x]")
    if fine_mask.shape != fine_sit.shape:
        raise ValueError("fine mask and SIT shapes differ")
    if source_coarse_sit.ndim != 4 or source_coarse_sit.shape[1] != 1:
        raise ValueError("source_coarse_sit must have shape [sample,1,y,x]")
    if fp32_ulps <= 0:
        raise ValueError("fp32_ulps must be positive")
    if not torch.isfinite(fine_sit).all() or not torch.isfinite(source_coarse_sit).all():
        raise FloatingPointError("support probe received NaN/Inf")

    fine32 = fine_sit.float()
    mask32 = fine_mask.float()
    source32 = source_coarse_sit.float()
    coarse32, fraction32 = masked_block_average(fine32, mask32)
    negative32, negative_fraction32 = masked_block_average((-fine32).clamp_min(0), mask32)
    projected32, projected_fraction32 = masked_block_average(fine32.clamp_min(0), mask32)
    if not torch.equal(fraction32, negative_fraction32) or not torch.equal(
        fraction32, projected_fraction32
    ):
        raise RuntimeError("masked coarse supports differ")

    coarse = coarse32.double()
    source = source32.double()
    q = negative32.double()
    projected = projected32.double()
    fraction = fraction32.double()
    unavoidable = (-coarse).clamp_min(0)
    mixed_sign = q - unavoidable
    active = fraction > 0
    scale = max(
        1.0,
        float(fine32.abs().max()),
        float(source32.abs().max()),
        float(projected32.abs().max()),
    )
    tolerance = fp32_ulps * torch.finfo(torch.float32).eps * scale

    errors = {
        "source_coarse": (coarse - source).abs(),
        "projection_identity": (projected - coarse - q).abs(),
        "decomposition_identity": (q - unavoidable - mixed_sign).abs(),
    }
    for name, error in errors.items():
        if float(error[active].max()) > tolerance:
            raise RuntimeError(f"{name} identity exceeds FP32-source-aware tolerance")
    if float(q[active].min()) < -tolerance:
        raise RuntimeError("q is negative beyond tolerance")
    if float(unavoidable[active].min()) < -tolerance:
        raise RuntimeError("unavoidable term is negative beyond tolerance")
    if float(mixed_sign[active].min()) < -tolerance:
        raise RuntimeError("mixed-sign term is negative beyond tolerance")

    return {
        "coarse": coarse,
        "q": q,
        "projected_coarse": projected,
        "unavoidable": unavoidable,
        "mixed_sign": mixed_sign,
        "fraction": fraction,
        "negative_coarse": coarse < -tolerance,
        "tolerance_metres": tolerance,
        "max_source_coarse_error_metres": float(errors["source_coarse"][active].max()),
        "max_projection_identity_error_metres": float(
            errors["projection_identity"][active].max()
        ),
        "max_decomposition_identity_error_metres": float(
            errors["decomposition_identity"][active].max()
        ),
        "minimum_mixed_sign_metres": float(mixed_sign[active].min()),
    }


def _case_equal_summary(
    decomposition: dict[str, torch.Tensor | float], cases: int, members: int
) -> dict[str, Any]:
    fraction = decomposition["fraction"].unflatten(0, (cases, members))
    active = fraction > 0
    weight = fraction

    def per_case_weighted(value: torch.Tensor) -> torch.Tensor:
        value = value.unflatten(0, (cases, members))
        numerator = (value * weight).sum(dim=(1, 2, 3, 4))
        denominator = weight.sum(dim=(1, 2, 3, 4))
        return numerator / denominator

    negative = decomposition["negative_coarse"].unflatten(0, (cases, members))
    weighted_negative = per_case_weighted(decomposition["negative_coarse"].double())
    unweighted_negative = negative.sum(dim=(1, 2, 3, 4)) / active.sum(
        dim=(1, 2, 3, 4)
    )
    result: dict[str, Any] = {
        "fp32_source_tolerance_metres": decomposition["tolerance_metres"],
        "max_source_coarse_error_metres": decomposition["max_source_coarse_error_metres"],
        "max_projection_identity_error_metres": decomposition[
            "max_projection_identity_error_metres"
        ],
        "max_decomposition_identity_error_metres": decomposition[
            "max_decomposition_identity_error_metres"
        ],
        "minimum_mixed_sign_metres": decomposition["minimum_mixed_sign_metres"],
        "negative_coarse_block_fraction_case_equal": float(unweighted_negative.mean()),
        "negative_coarse_ocean_fraction_weighted_case_equal": float(
            weighted_negative.mean()
        ),
        "per_case_negative_coarse_block_fraction": unweighted_negative.tolist(),
        "per_case_negative_coarse_ocean_fraction_weighted": weighted_negative.tolist(),
    }
    for key in ("q", "unavoidable", "mixed_sign"):
        per_case = per_case_weighted(decomposition[key])
        result[f"{key}_metres_case_equal"] = float(per_case.mean())
        result[f"per_case_{key}_metres"] = per_case.tolist()
    q = result["q_metres_case_equal"]
    result["unavoidable_fraction_of_q"] = (
        result["unavoidable_metres_case_equal"] / q if q > 0 else 0.0
    )
    result["mixed_sign_fraction_of_q"] = (
        result["mixed_sign_metres_case_equal"] / q if q > 0 else 0.0
    )
    return result


def run(config_path: Path, output: Path) -> dict[str, Any]:
    torch.set_num_threads(6)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_available() or os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
        raise RuntimeError("SIT support feasibility audit requires CUDA_VISIBLE_DEVICES empty")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite {output}")
    config = load_json(config_path)
    if config.get("schema_version") != "sit_support_feasibility_audit_v1":
        raise ValueError("unreviewed SIT support feasibility schema")
    fp32_ulps = int(config["numerics"]["fp32_source_tolerance_ulps"])
    for group in ("evidence_config", "source_censor_status", "source_censor_metrics"):
        source = config[group]
        path = Path(source["path"])
        if not path.is_file() or _sha256(path) != source["sha256"]:
            raise ValueError(f"{group} SHA mismatch")
    source_status = load_json(config["source_censor_status"]["path"])
    if source_status.get("status") != "complete" or source_status.get(
        "metrics_sha256"
    ) != config["source_censor_metrics"]["sha256"]:
        raise ValueError("left-censor source audit is not terminal complete")
    evidence_config = load_json(config["evidence_config"]["path"])

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
        code_identity = _clean_code_identity(Path(__file__).resolve().parents[1])
        tracker = ClearMLTracker(
            config["project_name"],
            f'{config["task_name"]}-{output.parent.name}',
            tags=config["clearml"]["tags"],
            env_path=config["clearml"]["env_path"],
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
        evidence = {
            name: _load_frozen(spec) for name, spec in evidence_config["evidence"].items()
        }
        canonical = evidence["raw"]
        for name, payload in evidence.items():
            _validate_payload(name, payload, evidence_config["evidence"][name], evidence_config)
            _require_equal(canonical, payload, name)
        means, stds, normalization = _load_normalization(evidence_config)
        mask = canonical["valid_mask"].float()
        cases = int(mask.shape[0])
        members = int(canonical["forecast_normalized"].shape[1])
        expanded_mask = mask[:, None].expand(-1, members, -1, -1, -1).flatten(0, 1)

        result: dict[str, Any] = {
            "status": "numerical_results_complete_pending_finalization",
            "scientific_role": "frozen_support_feasibility_probe_not_champion_selection",
            "code_identity": code_identity,
            "normalization_source": normalization,
            "sources": {
                key: config[key]
                for key in ("evidence_config", "source_censor_status", "source_censor_metrics")
            },
            "identity": "q=D((-Y)_+); D(Y_+)=D(Y)+q; q=(-D(Y))_+ + mixed_sign",
            "candidates": {},
        }
        means5 = means.float().reshape(1, 1, 6, 1, 1)
        stds5 = stds.float().reshape(1, 1, 6, 1, 1)
        for name in ("raw", "ordinary64", "threshold64"):
            payload = evidence[name]
            physical = decode_with_exact_sit_zero(
                payload["forecast_normalized"].float(), means, stds
            )
            coarse_physical = payload["coarse_normalized"].float() * stds5 + means5
            candidate: dict[str, Any] = {}
            for channel in SIT_CHANNELS:
                decomposition = support_decomposition(
                    physical[:, :, channel : channel + 1].flatten(0, 1),
                    expanded_mask,
                    coarse_physical[:, :, channel : channel + 1].flatten(0, 1),
                    fp32_ulps,
                )
                candidate[OUTPUTS[channel]] = _case_equal_summary(
                    decomposition, cases, members
                )
            result["candidates"][name] = candidate

        _strict_atomic_json(metrics_path, result)
        tracker.upload_artifact("sit_support_feasibility_audit", metrics_path)
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
