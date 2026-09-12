"""Frozen CPU mechanics gate for the support-consistent SIT decoder."""

from __future__ import annotations

import argparse
import json
import os
import signal
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
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
from .direct_dynamics_cascade_paired_evaluation import _save_contact_sheet
from .direct_dynamics_sit_left_censor_audit import decode_with_exact_sit_zero
from .direct_dynamics_sit_support_decoder import (
    project_masked_blocks_to_nonnegative_mean,
    support_decoder_checks,
)


SIT_CHANNELS = (1, 3, 5)
OUTPUTS = ("d3_sic", "d3_sit", "d6_sic", "d6_sit", "d9_sic", "d9_sit")


def _edge_means(value: torch.Tensor, mask: torch.Tensor, factor: int = 2) -> dict[str, Any]:
    """Case-equal absolute gradients across block boundaries and interiors."""
    if value.ndim != 5 or value.shape[2] != 1:
        raise ValueError("edge audit value must be [case,member,1,y,x]")
    expanded = mask[:, None].expand(-1, value.shape[1], -1, -1, -1) > 0
    horizontal = (value[..., 1:] - value[..., :-1]).abs().double()
    horizontal_valid = expanded[..., 1:] & expanded[..., :-1]
    vertical = (value[..., 1:, :] - value[..., :-1, :]).abs().double()
    vertical_valid = expanded[..., 1:, :] & expanded[..., :-1, :]
    x_boundary = (
        torch.arange(1, value.shape[-1], device=value.device) % factor == 0
    ).reshape(1, 1, 1, 1, -1)
    y_boundary = (
        torch.arange(1, value.shape[-2], device=value.device) % factor == 0
    ).reshape(1, 1, 1, -1, 1)

    def case_mean(boundary: bool) -> tuple[torch.Tensor, torch.Tensor]:
        hx = horizontal_valid & (x_boundary if boundary else ~x_boundary)
        vy = vertical_valid & (y_boundary if boundary else ~y_boundary)
        numerator = (horizontal * hx).sum(dim=(1, 2, 3, 4)) + (
            vertical * vy
        ).sum(dim=(1, 2, 3, 4))
        denominator = hx.sum(dim=(1, 2, 3, 4)) + vy.sum(dim=(1, 2, 3, 4))
        if torch.any(denominator == 0):
            raise ValueError("edge audit has a case without valid edges")
        return numerator / denominator, denominator

    boundary, boundary_count = case_mean(True)
    interior, interior_count = case_mean(False)
    defined = interior > 0
    ratios = [
        float(boundary[index] / interior[index]) if bool(defined[index]) else None
        for index in range(value.shape[0])
    ]
    mean_ratio = (
        float((boundary[defined] / interior[defined]).mean())
        if torch.any(defined)
        else None
    )
    return {
        "boundary_abs_gradient_case_equal": float(boundary.mean()),
        "interior_abs_gradient_case_equal": float(interior.mean()),
        "boundary_to_interior_ratio_case_equal_defined_cases": mean_ratio,
        "per_case_boundary_abs_gradient": boundary.tolist(),
        "per_case_interior_abs_gradient": interior.tolist(),
        "per_case_boundary_edge_count": boundary_count.tolist(),
        "per_case_interior_edge_count": interior_count.tolist(),
        "per_case_zero_interior_gradient": (~defined).tolist(),
        "per_case_boundary_to_interior_ratio": ratios,
    }


def _seam_audit(original: torch.Tensor, decoded: torch.Tensor, mask: torch.Tensor) -> dict[str, Any]:
    before = _edge_means(original, mask)
    after = _edge_means(decoded, mask)
    perturbation = _edge_means(decoded - original, mask)
    boundary_change = after["boundary_abs_gradient_case_equal"] - before[
        "boundary_abs_gradient_case_equal"
    ]
    interior_change = after["interior_abs_gradient_case_equal"] - before[
        "interior_abs_gradient_case_equal"
    ]
    before_ratios = before["per_case_boundary_to_interior_ratio"]
    after_ratios = after["per_case_boundary_to_interior_ratio"]
    paired_indices = [
        index
        for index, (before_ratio, after_ratio) in enumerate(
            zip(before_ratios, after_ratios, strict=True)
        )
        if before_ratio is not None and after_ratio is not None
    ]
    paired_changes = [
        after_ratios[index] - before_ratios[index] for index in paired_indices
    ]
    return {
        "original": before,
        "decoded": after,
        "decoder_perturbation": perturbation,
        "decoded_minus_original_boundary_gradient": boundary_change,
        "decoded_minus_original_interior_gradient": interior_change,
        "new_block_boundary_excess": boundary_change - interior_change,
        "paired_ratio_defined_case_indices": paired_indices,
        "paired_ratio_defined_case_count": len(paired_indices),
        "per_case_boundary_to_interior_ratio_change_paired": paired_changes,
        "boundary_to_interior_ratio_change_paired_case_equal": (
            sum(paired_changes) / len(paired_changes) if paired_changes else None
        ),
    }


def _gradient_probe() -> dict[str, Any]:
    mask = torch.ones((1, 1, 2, 2), dtype=torch.float64)
    positive = torch.tensor(
        [[[[-1.25, 0.25], [1.0, 3.0]]]], dtype=torch.float64, requires_grad=True
    )
    positive_coarse, _ = masked_block_average(positive, mask)
    positive_decoded = project_masked_blocks_to_nonnegative_mean(
        positive, positive_coarse, mask
    )
    weights = torch.tensor([[[[0.5, 1.0], [2.0, 4.0]]]], dtype=torch.float64)
    positive_gradient = torch.autograd.grad((positive_decoded * weights).sum(), positive)[0]
    negative = torch.tensor(
        [[[[-4.0, -3.0], [-2.0, -1.0]]]], dtype=torch.float64, requires_grad=True
    )
    negative_coarse, _ = masked_block_average(negative, mask)
    negative_decoded = project_masked_blocks_to_nonnegative_mean(
        negative, negative_coarse, mask
    )
    negative_gradient = torch.autograd.grad(negative_decoded.sum(), negative)[0]
    if not torch.isfinite(positive_gradient).all() or not torch.isfinite(negative_gradient).all():
        raise FloatingPointError("support decoder gradient is non-finite")
    if float(positive_gradient.abs().sum()) <= 0:
        raise RuntimeError("support decoder gradient is dead on a positive active example")
    if not torch.equal(negative_gradient, torch.zeros_like(negative_gradient)):
        raise RuntimeError("fully negative block must have zero local gradient")
    return {
        "straight_through_estimator": False,
        "positive_example_gradient_l1": float(positive_gradient.abs().sum()),
        "positive_example_gradient": positive_gradient.tolist(),
        "fully_negative_gradient_l1": float(negative_gradient.abs().sum()),
        "fully_negative_gradient": negative_gradient.tolist(),
    }


def _save_delta_sheet(
    path: Path,
    title: str,
    original: torch.Tensor,
    decoded: torch.Tensor,
    mask: torch.Tensor,
    limits: tuple[float, float],
) -> None:
    delta = (decoded[:, SIT_CHANNELS] - original[:, SIT_CHANNELS]).detach().cpu().numpy()
    valid = mask.detach().cpu().numpy()[0] > 0
    figure, axes = plt.subplots(delta.shape[0], 3, figsize=(7.2, 2.25 * delta.shape[0]), squeeze=False)
    for member in range(delta.shape[0]):
        for horizon in range(3):
            image = np.where(valid, delta[member, horizon], np.nan)
            axes[member, horizon].imshow(
                image, origin="upper", cmap="coolwarm", vmin=limits[0], vmax=limits[1]
            )
            axes[member, horizon].set_xticks([])
            axes[member, horizon].set_yticks([])
            if member == 0:
                axes[member, horizon].set_title(("d3", "d6", "d9")[horizon] + " SIT delta")
            if horizon == 0:
                axes[member, horizon].set_ylabel(f"member {member}")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def run(config_path: Path, output: Path) -> dict[str, Any]:
    torch.set_num_threads(6)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_available() or os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
        raise RuntimeError("SIT support decoder audit requires CUDA_VISIBLE_DEVICES empty")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite {output}")
    config = load_json(config_path)
    if config.get("schema_version") != "sit_support_decoder_audit_v1":
        raise ValueError("unreviewed SIT support decoder audit schema")
    for group in ("evidence_config", "source_probe_status", "source_probe_metrics"):
        source = config[group]
        path = Path(source["path"])
        if not path.is_file() or _sha256(path) != source["sha256"]:
            raise ValueError(f"{group} SHA mismatch")
    source_status = load_json(config["source_probe_status"]["path"])
    if source_status.get("status") != "complete" or source_status.get(
        "metrics_sha256"
    ) != config["source_probe_metrics"]["sha256"]:
        raise ValueError("support feasibility source probe is not terminal complete")
    evidence_config = load_json(config["evidence_config"]["path"])
    fp32_ulps = int(config["numerics"]["fp32_source_tolerance_ulps"])
    delta_limits = tuple(float(value) for value in config["visuals"]["sit_delta_limits_metres"])

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
        means5 = means.float().reshape(1, 1, 6, 1, 1)
        stds5 = stds.float().reshape(1, 1, 6, 1, 1)
        mask = canonical["valid_mask"].float()
        members = int(canonical["forecast_normalized"].shape[1])
        expanded_mask = mask[:, None].expand(-1, members, -1, -1, -1).flatten(0, 1)
        truth_physical = decode_with_exact_sit_zero(
            canonical["truth_normalized"].float(), means, stds
        )
        persistence_physical = decode_with_exact_sit_zero(
            canonical["persistence_normalized"].float(), means, stds
        )
        visuals = output.parent / "matched_support_decoder_members"
        visuals.mkdir()
        result: dict[str, Any] = {
            "status": "numerical_results_complete_pending_visual_review",
            "scientific_role": "support_decoder_cpu_mechanics_gate_not_calibrator",
            "code_identity": code_identity,
            "normalization_source": normalization,
            "sources": {
                key: config[key]
                for key in ("evidence_config", "source_probe_status", "source_probe_metrics")
            },
            "decoder_law": "per-block Euclidean simplex projection; physical coarse=max(source coarse,0)",
            "gradient_probe": _gradient_probe(),
            "candidates": {},
        }
        for name in ("raw", "ordinary64", "threshold64"):
            payload = evidence[name]
            original = decode_with_exact_sit_zero(
                payload["forecast_normalized"].float(), means, stds
            )
            source_coarse = payload["coarse_normalized"].float() * stds5 + means5
            decoded = original.clone()
            channel_result: dict[str, Any] = {}
            for channel in SIT_CHANNELS:
                fine = original[:, :, channel : channel + 1].flatten(0, 1)
                coarse = source_coarse[:, :, channel : channel + 1].flatten(0, 1)
                projected = project_masked_blocks_to_nonnegative_mean(
                    fine, coarse, expanded_mask
                )
                checks = support_decoder_checks(
                    fine, coarse, expanded_mask, fp32_ulps=fp32_ulps
                )
                reprojected = project_masked_blocks_to_nonnegative_mean(
                    projected, coarse.clamp_min(0), expanded_mask
                )
                idempotence = float((reprojected - projected).abs().max())
                if idempotence > checks["fp32_source_tolerance"]:
                    raise RuntimeError("support decoder is not idempotent")
                decoded[:, :, channel : channel + 1] = projected.unflatten(
                    0, original.shape[:2]
                )
                delta = projected - fine
                valid = expanded_mask.expand_as(delta) > 0
                channel_result[OUTPUTS[channel]] = {
                    **checks,
                    "idempotence_max_abs": idempotence,
                    "perturbation_rms_metres": float(torch.sqrt(delta[valid].double().square().mean())),
                    "perturbation_max_abs_metres": float(delta[valid].abs().max()),
                    "seam_audit": _seam_audit(
                        original[:, :, channel : channel + 1],
                        decoded[:, :, channel : channel + 1],
                        mask,
                    ),
                }
            if not torch.equal(decoded[:, :, 0::2], original[:, :, 0::2]):
                raise RuntimeError("support decoder changed SIC")
            result["candidates"][name] = channel_result
            for position in (0, 4, 7, 11):
                member_path = visuals / f"{name}_decoded_case{position:02d}_all8_fixed.png"
                _save_contact_sheet(
                    member_path,
                    f"{name}/support_decoder",
                    canonical["case_ids"][position],
                    decoded[position],
                    truth_physical[position],
                    persistence_physical[position],
                    mask[position],
                    {"sic": (0.0, 1.0), "sit": (0.0, 3.5)},
                )
                tracker.report_image(
                    "sit_support_decoder/matched_members",
                    f"{name}/case{position:02d}",
                    member_path,
                    0,
                )
                delta_path = visuals / f"{name}_delta_case{position:02d}_all8_fixed.png"
                _save_delta_sheet(
                    delta_path,
                    f"{name}/support_decoder delta; {canonical['case_ids'][position]}",
                    original[position],
                    decoded[position],
                    mask[position],
                    delta_limits,
                )
                tracker.report_image(
                    "sit_support_decoder/member_deltas",
                    f"{name}/case{position:02d}",
                    delta_path,
                    0,
                )
        _strict_atomic_json(metrics_path, result)
        tracker.upload_artifact("sit_support_decoder_audit", metrics_path)
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
