"""CPU admission for support-aware fine-stage proper-score refinement."""

from __future__ import annotations

import argparse
import json
import math
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
    _load_normalization,
    _record_failure,
    _sha256,
    _strict_atomic_json,
    _terminate,
)
from .direct_dynamics_cascade_coarse_proper_refinement import proper_objective
from .direct_dynamics_cascade_fine_training import _clean_code_identity
from .direct_dynamics_sic_support_decoder import (
    project_masked_blocks_to_unit_interval_mean,
)
from .direct_dynamics_sic_support_decoder_scoring import SIC_CHANNELS
from .direct_dynamics_sit_support_decoder import _from_blocks, _to_blocks


def _pre_correction_kkt_free_set(
    fine: torch.Tensor,
    coarse: torch.Tensor,
    mask: torch.Tensor,
    factor: int,
) -> torch.Tensor:
    """Recover the projection's KKT free set before roundoff repair.

    The frozen decoder makes a final O(eps) budget correction.  Classifying
    activity from that repaired value can turn a saturated endpoint into a
    false free coordinate, changing its Jacobian by O(1).  This reproduces the
    decoder's bisection and classifies the uncorrected clipped argument.
    """
    values = fine.double()
    coarse_values = coarse.to(device=fine.device, dtype=torch.float64)
    support = mask.to(device=fine.device).expand_as(fine) > 0
    blocks = _to_blocks(values, factor)
    block_mask = _to_blocks(support, factor)
    count = block_mask.sum(dim=-1)
    active = count > 0
    target_sum = count.double() * coarse_values.clamp(0.0, 1.0)
    positive_infinity = torch.full_like(blocks, torch.inf)
    negative_infinity = torch.full_like(blocks, -torch.inf)
    maximum = torch.where(block_mask, blocks, negative_infinity).max(dim=-1).values
    maximum = torch.where(active, maximum, torch.zeros_like(maximum))
    centered = torch.where(
        block_mask, blocks - maximum.unsqueeze(-1), torch.zeros_like(blocks)
    )
    minimum = torch.where(block_mask, centered, positive_infinity).min(dim=-1).values
    lower = torch.where(active, minimum - 1.0, torch.zeros_like(minimum))
    upper = torch.zeros_like(lower)
    for _ in range(80):
        theta = 0.5 * (lower + upper)
        candidate = torch.where(
            block_mask,
            (centered - theta.unsqueeze(-1)).clamp(0.0, 1.0),
            torch.zeros_like(blocks),
        )
        too_large = candidate.sum(dim=-1) > target_sum
        lower = torch.where(too_large, theta, lower)
        upper = torch.where(too_large, upper, theta)
    theta = 0.5 * (lower + upper)
    unconstrained = centered - theta.unsqueeze(-1)
    free_blocks = block_mask & (unconstrained > 0.0) & (unconstrained < 1.0)
    endpoint_budget = (target_sum == 0.0) | (target_sum == count.double())
    free_blocks = free_blocks & ~endpoint_budget.unsqueeze(-1)
    return _from_blocks(free_blocks, factor)


class _FixedBudgetCappedSimplex(torch.autograd.Function):
    """Exact-a.e. backward for the fixed-budget Euclidean projection.

    For a fixed active set, free coordinates have Jacobian I-11^T/k and
    saturated coordinates have zero derivative.  This is the true projection
    Jacobian almost everywhere, not a straight-through estimator.  The coarse
    budget is fixed conditioning and therefore deliberately non-trainable.
    """

    @staticmethod
    def forward(
        ctx: Any,
        fine: torch.Tensor,
        coarse: torch.Tensor,
        mask: torch.Tensor,
        factor: int,
    ) -> torch.Tensor:
        if coarse.requires_grad:
            raise ValueError("support-aware fine refinement requires a fixed coarse budget")
        with torch.no_grad():
            decoded = project_masked_blocks_to_unit_interval_mean(
                fine, coarse, mask, factor
            )
            free = _pre_correction_kkt_free_set(fine, coarse, mask, factor)
        ctx.save_for_backward(free)
        ctx.factor = factor
        ctx.input_dtype = fine.dtype
        return decoded

    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None]:
        (free,) = ctx.saved_tensors
        factor = ctx.factor
        grad_blocks = _to_blocks(grad_output.double(), factor)
        free_blocks = _to_blocks(free, factor)
        count = free_blocks.sum(dim=-1, keepdim=True)
        centered = grad_blocks - (
            (grad_blocks * free_blocks).sum(dim=-1, keepdim=True)
            / count.clamp_min(1)
        )
        grad_fine = _from_blocks(
            torch.where(free_blocks, centered, torch.zeros_like(centered)), factor
        )
        return grad_fine.to(dtype=ctx.input_dtype), None, None, None


def differentiable_fixed_budget_projection(
    fine: torch.Tensor,
    coarse: torch.Tensor,
    mask: torch.Tensor,
    factor: int = 2,
) -> torch.Tensor:
    return _FixedBudgetCappedSimplex.apply(fine, coarse, mask, factor)


def apply_training_sic_decoder(
    physical: torch.Tensor,
    coarse_physical: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Apply the reviewed decoder with an exact-a.e. memory-bounded gradient."""
    if physical.ndim != 5 or physical.shape[2] != 6:
        raise ValueError("physical ensemble must be [case,member,6,y,x]")
    expected = (
        physical.shape[0], physical.shape[1], 6,
        physical.shape[-2] // 2, physical.shape[-1] // 2,
    )
    if coarse_physical.shape != expected:
        raise ValueError("coarse ensemble shape differs from fine ensemble")
    if mask.shape != (physical.shape[0], 1, *physical.shape[-2:]):
        raise ValueError("mask shape differs from fine ensemble")
    result = physical.double().clone()
    cases, members = physical.shape[:2]
    member_mask = mask[:, None].expand(-1, members, -1, -1, -1).flatten(0, 1)
    for channel in SIC_CHANNELS:
        decoded = differentiable_fixed_budget_projection(
            physical[:, :, channel : channel + 1].flatten(0, 1),
            coarse_physical[:, :, channel : channel + 1].flatten(0, 1),
            member_mask,
        ).unflatten(0, (cases, members))
        result[:, :, channel : channel + 1] = decoded
    return result


def support_aware_physical_objective(
    physical: torch.Tensor,
    coarse_physical: torch.Tensor,
    truth_physical: torch.Tensor,
    mask: torch.Tensor,
    channel_stds: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode first, then score the joint physical ensemble in standardized units."""
    if truth_physical.shape != physical.shape[:1] + physical.shape[2:]:
        raise ValueError("truth shape differs from ensemble")
    if channel_stds.numel() != 6 or not torch.isfinite(channel_stds).all() or torch.any(channel_stds <= 0):
        raise ValueError("six finite positive channel standard deviations are required")
    decoded = apply_training_sic_decoder(physical, coarse_physical, mask)
    scale = channel_stds.to(device=decoded.device, dtype=decoded.dtype).reshape(
        1, 1, 6, 1, 1
    )
    standardized_members = decoded / scale
    standardized_truth = truth_physical.to(decoded).unsqueeze(1) / scale
    objective, crps, energy = proper_objective(
        standardized_members,
        standardized_truth[:, 0],
        mask.to(decoded),
    )
    return objective, crps, energy, decoded


def _load_bound_provenance(config: dict[str, Any], repo: Path) -> dict[str, Any]:
    bound: dict[str, Any] = {}
    for name, spec in config["sources"].items():
        path = repo / spec["path"] if spec.get("repo_relative") else Path(spec["path"])
        if not path.is_file() or _sha256(path) != spec["sha256"]:
            raise ValueError(f"support-aware admission source SHA mismatch: {name}")
        bound[name] = {"path": str(path), "sha256": spec["sha256"]}
    support = load_json(bound["support_decoder_metrics"]["path"])
    attribution = load_json(bound["coarse_budget_attribution"]["path"])
    if support.get("code_identity", {}).get("git_commit") != config["sources"]["support_decoder_metrics"]["code_commit"]:
        raise ValueError("source decoder code identity differs from frozen contract")
    if attribution.get("code_identity", {}).get("git_commit") != config["sources"]["coarse_budget_attribution"]["code_commit"]:
        raise ValueError("source attribution code identity differs from frozen contract")
    if support.get("gate", {}).get("sic_exact_support_passed") is not True:
        raise ValueError("source decoder did not establish exact physical support")
    if support.get("gate", {}).get("passed_pending_visual_review") is not False:
        raise ValueError("source post-hoc decoder rejection is not bound")
    if support.get("training_performed") is not False or support.get("sampling_performed") is not False:
        raise ValueError("source decoder unexpectedly trained or sampled")
    if attribution.get("scientific_role") != "frozen_development_attribution_not_calibrator":
        raise ValueError("source attribution scientific role differs")
    if attribution.get("training_performed") is not False or attribution.get("sampling_performed") is not False:
        raise ValueError("source attribution unexpectedly trained or sampled")
    return bound


def _analytic_finite_member_check() -> dict[str, float]:
    values = torch.arange(4, dtype=torch.float64).reshape(1, 4, 1, 1, 1).expand(-1, -1, 6, -1, -1)
    truth = torch.full((1, 6, 1, 1), 1.5, dtype=torch.float64)
    mask = torch.ones(1, 1, 1, 1, dtype=torch.float64)
    objective, crps, energy = proper_objective(values, truth, mask)
    expected = 1.0 / 6.0
    for name, value in (("objective", objective), ("fair_crps", crps), ("joint_energy", energy)):
        if abs(float(value) - expected) > 1e-14:
            raise RuntimeError(f"finite-member {name} differs from analytic unbiased score")
    return {"expected": expected, "objective": float(objective), "fair_crps": float(crps), "joint_energy": float(energy)}


def _synthetic_gradient_admission(channel_stds: torch.Tensor) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(20260912)
    leaf = torch.randn((2, 4, 6, 4, 4), generator=generator, dtype=torch.float64)
    leaf.requires_grad_(True)
    mask = torch.ones(2, 1, 4, 4, dtype=torch.float64)
    truth = torch.rand((2, 6, 4, 4), generator=generator, dtype=torch.float64)
    truth[:, 0::2] = truth[:, 0::2].clamp(0, 1)
    truth[:, 0::2, :2, :2] = 0.0
    truth[:, 1::2] = 2.5 * truth[:, 1::2]
    teacher_coarse, _ = masked_block_average(truth, mask)
    coarse = teacher_coarse[:, None].expand(-1, 4, -1, -1, -1).clone()
    if not torch.equal(coarse[:, 0], teacher_coarse) or not torch.equal(
        coarse, coarse[:, :1].expand_as(coarse)
    ):
        raise RuntimeError("synthetic teacher-coarse contract was not constructed exactly")

    objective, crps, energy, decoded = support_aware_physical_objective(
        leaf, coarse, truth, mask, channel_stds
    )
    with torch.no_grad():
        direct = leaf.double().clone()
        cases, members = leaf.shape[:2]
        member_mask = mask[:, None].expand(-1, members, -1, -1, -1).flatten(0, 1)
        for channel in SIC_CHANNELS:
            projected = project_masked_blocks_to_unit_interval_mean(
                leaf[:, :, channel : channel + 1].flatten(0, 1),
                coarse[:, :, channel : channel + 1].flatten(0, 1),
                member_mask,
            ).unflatten(0, (cases, members))
            direct[:, :, channel : channel + 1] = projected
    if not torch.equal(decoded, direct):
        raise RuntimeError("training scored path differs from reviewed decoder forward")
    for channel in SIC_CHANNELS:
        recovered, fraction = masked_block_average(
            decoded[:, :, channel : channel + 1].flatten(0, 1), member_mask
        )
        expected = coarse[:, :, channel : channel + 1].flatten(0, 1).clamp(0, 1)
        if float((recovered - expected)[fraction > 0].abs().max()) > 2e-14:
            raise RuntimeError("training decoder lost coarse consistency")
    if not all(torch.isfinite(value) for value in (objective, crps, energy)):
        raise FloatingPointError("support-aware proper score is non-finite")
    objective.backward()
    gradient = leaf.grad
    if gradient is None or not torch.isfinite(gradient).all():
        raise FloatingPointError("support-aware objective produced invalid gradient")
    sic_gradient = gradient[:, :, 0::2]
    coarse_sic = coarse[:, :, 0::2]
    zero_budget = (coarse_sic == 0).repeat_interleave(2, -2).repeat_interleave(2, -1)
    mixed_budget = ((coarse_sic > 0) & (coarse_sic < 1)).repeat_interleave(2, -2).repeat_interleave(2, -1)
    zero_norm = float(sic_gradient[zero_budget].norm())
    mixed_norm = float(sic_gradient[mixed_budget].norm())
    sit_norm = float(gradient[:, :, 1::2].norm())
    if zero_norm != 0.0:
        raise RuntimeError("zero-budget SIC block has a nonzero fine gradient")
    if not math.isfinite(mixed_norm) or mixed_norm <= 0:
        raise RuntimeError("mixed-budget SIC gradient is dead")
    if not math.isfinite(sit_norm) or sit_norm <= 0:
        raise RuntimeError("SIT gradient is dead")
    return {
        "objective": float(objective), "fair_crps": float(crps),
        "joint_energy": float(energy), "mixed_budget_sic_gradient_norm": mixed_norm,
        "zero_budget_sic_gradient_norm": zero_norm, "sit_gradient_norm": sit_norm,
        "decoder_forward_bitwise_equal": True, "exact_support_and_coarse_consistency": True,
        "teacher_coarse_equals_downsampled_truth": True,
        "teacher_coarse_identical_across_members": True,
        "backward": "exact active-set Jacobian I-11^T/k; no straight-through estimator",
    }


def run(config_path: Path, output: Path) -> dict[str, Any]:
    torch.set_num_threads(6)
    torch.set_num_interop_threads(1)
    if torch.get_num_threads() != 6 or torch.get_num_interop_threads() != 1:
        raise RuntimeError("failed to enforce reviewed 6/1 CPU thread envelope")
    if torch.cuda.is_available() or os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
        raise RuntimeError("support-aware admission requires CUDA_VISIBLE_DEVICES empty")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite {output}")
    config = load_json(config_path)
    if config.get("schema_version") != "fine_support_proper_cpu_admission_v1":
        raise ValueError("unreviewed support-aware admission schema")
    output.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = output.with_name(f"{output.stem}.metrics.json")
    if metrics_path.exists() or metrics_path.is_symlink():
        raise FileExistsError(f"refusing to overwrite {metrics_path}")
    reservation = {"status": "reserved", "config_path": str(config_path), "config_sha256": _sha256(config_path)}
    with output.open("x") as handle:
        json.dump(reservation, handle, indent=2, allow_nan=False)
        handle.write("\n")
    tracker = None
    try:
        repo = Path(__file__).resolve().parents[1]
        provenance = _load_bound_provenance(config, repo)
        code_identity = _clean_code_identity(repo)
        tracker = ClearMLTracker(
            config["project_name"], f'{config["task_name"]}-{output.parent.name}',
            tags=config["clearml"]["tags"], env_path=config["clearml"]["env_path"],
        )
        tracker.connect("admission_contract", config)
        _strict_atomic_json(output, {**reservation, "status": "running", "code_identity": code_identity, "clearml_task_id": str(tracker.task.id)})
        source_config = load_json(provenance["source_scoring_config"]["path"])
        _, channel_stds, normalization = _load_normalization(source_config)
        result = {
            "status": "admission_passed_pending_astra_review",
            "scientific_role": "synthetic_cpu_admission_not_training_or_evaluation",
            "training_performed": False, "optimizer_steps": 0,
            "sampling_performed": False, "gpu_used": False,
            "code_identity": code_identity, "provenance": provenance,
            "normalization": normalization,
            "candidate_contract": config["candidate_contract"],
            "finite_member_score_check": _analytic_finite_member_check(),
            "synthetic_gradient_check": _synthetic_gradient_admission(channel_stds),
        }
        _strict_atomic_json(metrics_path, result)
        for group in ("finite_member_score_check", "synthetic_gradient_check"):
            for name, value in result[group].items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    tracker.report_single_value(f"{group}/{name}", value)
        tracker.upload_artifact("fine_support_proper_cpu_admission", metrics_path)
        _finish_success(tracker, output, metrics_path, {**reservation, "code_identity": code_identity, "clearml_task_id": str(tracker.task.id), "scientific_role": result["scientific_role"]})
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
