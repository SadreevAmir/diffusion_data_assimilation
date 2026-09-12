"""CPU integration of the frozen-allocation coarse-budget refinement law."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import signal
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .clearml_tracking import ClearMLTracker
from .config import load_json
from .direct_dynamics_cascade import (
    masked_block_average,
    project_detail,
    smooth_right_inverse,
)
from .direct_dynamics_cascade_coarse_fine_boundary_audit import (
    _finish_success,
    _record_failure,
    _sha256,
    _strict_atomic_json,
    _terminate,
)
from .direct_dynamics_cascade_coarse_proper_refinement import (
    frozen_prefix,
    hybrid_terminal_sample,
    proper_objective,
)
from .direct_dynamics_cascade_fine_training import _clean_code_identity
from .direct_dynamics_fine_support_proper_admission import (
    _pre_correction_kkt_free_set,
    apply_training_sic_decoder,
)
from .direct_dynamics_sic_support_decoder import (
    project_masked_blocks_to_unit_interval_mean,
)
from .direct_dynamics_sic_coarse_budget_sensitivity import _projection_unconstrained
from .direct_dynamics_sic_support_decoder_scoring import canonical_physical_decode
from .direct_dynamics_sit_support_decoder import _from_blocks, _to_blocks
from .runtime import make_normalized_xy_grid


SIC_CHANNELS = (0, 2, 4)


def anchored_canonical_physical_coarse(
    base_normalized: torch.Tensor,
    candidate_normalized: torch.Tensor,
    means: torch.Tensor,
    stds: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Anchor a continuous train-time physical chart at a canonical coarse draw.

    The frozen draw is decoded by the canonical scoring law, including exact
    restoration of physical SIC atoms.  The trainable draw is then represented
    by a continuous physical displacement from that same normalized draw::

        C_0 = canonical(z_0)
        C_theta = C_0 + sigma_32 * (z_theta - z_0)

    This deliberately declares the chart optimized by coarse-budget training;
    it is not an STE and does not claim equality to an independently atom-fixed
    ``canonical(z_theta)`` away from the anchor.
    """
    if base_normalized.requires_grad:
        raise ValueError("anchored physical base must be detached")
    if base_normalized.shape != candidate_normalized.shape:
        raise ValueError("anchored coarse tensors must have identical shapes")
    if not base_normalized.is_floating_point() or base_normalized.shape[-3] != 6:
        raise ValueError("anchored coarse chart requires six floating-point channels")
    base_source = base_normalized.float()
    candidate_source = candidate_normalized.float()
    if not torch.isfinite(base_source).all() or not torch.isfinite(candidate_source).all():
        raise FloatingPointError("anchored coarse chart received NaN/Inf")
    base_physical = canonical_physical_decode(base_source, means, stds).detach()
    stds32 = stds.float().reshape(
        (1,) * (candidate_source.ndim - 3) + (6, 1, 1)
    )
    displacement = (
        candidate_source.double() - base_source.double()
    ) * stds32.double()
    candidate_physical = base_physical + displacement
    return base_physical, candidate_physical


def _selected_joint_free_set(
    fine: torch.Tensor,
    coarse: torch.Tensor,
    mask: torch.Tensor,
    factor: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select an exact-a.e. active set plus a declared endpoint continuation.

    Smooth interior active sets use the strict pre-correction KKT set.  Exact
    interior KKT kinks and impossible no-free interiors fail closed rather
    than receiving a silent surrogate derivative.  At C=0 the
    right-continuation distributes the infinitesimal budget over tied maxima;
    at C=1 the left-continuation removes it over tied minima.  Outside [0,1]
    clipping is locally flat.  This is a train-only generalized-gradient law,
    not a claim of classical differentiability at the endpoints.
    """
    support = mask.expand_as(fine) > 0
    strict = _pre_correction_kkt_free_set(fine, coarse, mask, factor)
    blocks = _to_blocks(fine.double(), factor)
    block_mask = _to_blocks(support, factor)
    unconstrained, checked_mask, valid_count = _projection_unconstrained(
        fine, coarse, mask, factor
    )
    if not torch.equal(block_mask, checked_mask):
        raise RuntimeError("projection support changed while selecting the backward")
    strict_count = _to_blocks(strict, factor).sum(dim=-1)
    interior = (valid_count > 0) & (coarse.double() > 0.0) & (coarse.double() < 1.0)
    interior_kink = interior & (
        block_mask
        & ((unconstrained == 0.0) | (unconstrained == 1.0))
    ).any(dim=-1)
    no_free_interior = interior & (strict_count == 0)
    if torch.any(interior_kink) or torch.any(no_free_interior):
        raise RuntimeError(
            "frozen-allocation backward is undefined at an exact interior KKT kink"
        )
    positive_infinity = torch.full_like(blocks, torch.inf)
    negative_infinity = torch.full_like(blocks, -torch.inf)
    maximum = torch.where(block_mask, blocks, negative_infinity).max(dim=-1).values
    minimum = torch.where(block_mask, blocks, positive_infinity).min(dim=-1).values
    maxima = block_mask & (blocks == maximum.unsqueeze(-1))
    minima = block_mask & (blocks == minimum.unsqueeze(-1))
    exact_zero = coarse.double() == 0.0
    exact_one = coarse.double() == 1.0
    selected_blocks = _to_blocks(strict, factor)
    selected_blocks = torch.where(exact_zero.unsqueeze(-1), maxima, selected_blocks)
    selected_blocks = torch.where(exact_one.unsqueeze(-1), minima, selected_blocks)
    selected = _from_blocks(selected_blocks, factor)
    budget_branch = (coarse.double() >= 0.0) & (coarse.double() <= 1.0)
    return selected, budget_branch


class _FrozenAllocationProjection(torch.autograd.Function):
    """Exact forward and declared full lift-plus-budget generalized backward."""

    @staticmethod
    def forward(
        ctx: Any,
        base_fine: torch.Tensor,
        base_coarse: torch.Tensor,
        candidate_coarse: torch.Tensor,
        mask: torch.Tensor,
        factor: int,
    ) -> torch.Tensor:
        if base_fine.requires_grad or base_coarse.requires_grad:
            raise ValueError("frozen-allocation base state must be detached")
        delta = candidate_coarse.double() - base_coarse.double()
        with torch.no_grad():
            shifted = base_fine.double() + smooth_right_inverse(delta, mask, factor)
            decoded = project_masked_blocks_to_unit_interval_mean(
                shifted, candidate_coarse.double(), mask, factor
            )
            free, budget_branch = _selected_joint_free_set(
                shifted, candidate_coarse, mask, factor
            )
        ctx.save_for_backward(free, budget_branch, mask)
        ctx.factor = factor
        ctx.coarse_dtype = candidate_coarse.dtype
        return decoded

    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[None, None, torch.Tensor, None, None]:
        free, budget_branch, mask = ctx.saved_tensors
        factor = ctx.factor
        free_blocks = _to_blocks(free, factor)
        grad_blocks = _to_blocks(grad_output.double(), factor)
        count = free_blocks.sum(dim=-1, keepdim=True)
        free_mean = (
            (grad_blocks * free_blocks).sum(dim=-1, keepdim=True)
            / count.clamp_min(1)
        )
        projected_gradient = _from_blocks(
            torch.where(
                free_blocks,
                grad_blocks - free_mean,
                torch.zeros_like(grad_blocks),
            ),
            factor,
        )
        with torch.enable_grad():
            lift_input = torch.zeros(
                budget_branch.shape,
                device=grad_output.device,
                dtype=torch.float64,
                requires_grad=True,
            )
            lift = smooth_right_inverse(lift_input, mask, factor)
            lift_vjp = torch.autograd.grad(
                (lift * projected_gradient).sum(), lift_input
            )[0]
        valid_count = _to_blocks(mask.expand_as(grad_output) > 0, factor).sum(dim=-1)
        free_count = count.squeeze(-1)
        budget_vjp = (
            valid_count.double()
            / free_count.double().clamp_min(1)
            * (grad_blocks * free_blocks).sum(dim=-1)
        )
        budget_vjp = torch.where(
            budget_branch & (free_count > 0),
            budget_vjp,
            torch.zeros_like(budget_vjp),
        )
        candidate_gradient = lift_vjp + budget_vjp
        if not torch.isfinite(candidate_gradient).all():
            raise FloatingPointError("frozen-allocation coarse gradient is invalid")
        return None, None, candidate_gradient.to(ctx.coarse_dtype), None, None


def differentiable_frozen_allocation_projection(
    base_fine: torch.Tensor,
    base_coarse: torch.Tensor,
    candidate_coarse: torch.Tensor,
    mask: torch.Tensor,
    factor: int = 2,
) -> torch.Tensor:
    return _FrozenAllocationProjection.apply(
        base_fine, base_coarse, candidate_coarse, mask, factor
    )


def frozen_allocation_physical_law(
    base_physical: torch.Tensor,
    base_coarse: torch.Tensor,
    candidate_coarse: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Apply Y=Y0+U(C-C0), then the same bounded SIC decoder on all water."""
    if base_physical.ndim != 5 or base_physical.shape[2] != 6:
        raise ValueError("base ensemble must be [case,member,6,y,x]")
    expected = (
        base_physical.shape[0],
        base_physical.shape[1],
        6,
        base_physical.shape[-2] // 2,
        base_physical.shape[-1] // 2,
    )
    if base_coarse.shape != expected or candidate_coarse.shape != expected:
        raise ValueError("coarse ensemble shapes differ from the base ensemble")
    if mask.shape != (base_physical.shape[0], 1, *base_physical.shape[-2:]):
        raise ValueError("fine mask shape differs from the base ensemble")
    if base_physical.requires_grad or base_coarse.requires_grad:
        raise ValueError("base allocation must remain frozen")
    cases, members = base_physical.shape[:2]
    member_mask = mask[:, None].expand(-1, members, -1, -1, -1).flatten(0, 1)
    channels: list[torch.Tensor] = []
    for channel in range(6):
        base_fine_channel = base_physical[:, :, channel : channel + 1].flatten(0, 1)
        base_coarse_channel = base_coarse[:, :, channel : channel + 1].flatten(0, 1)
        candidate_channel = candidate_coarse[:, :, channel : channel + 1].flatten(0, 1)
        if channel in SIC_CHANNELS:
            value = differentiable_frozen_allocation_projection(
                base_fine_channel,
                base_coarse_channel,
                candidate_channel,
                member_mask,
            )
        else:
            value = base_fine_channel.double() + smooth_right_inverse(
                candidate_channel.double() - base_coarse_channel.double(),
                member_mask,
            )
        channels.append(value.unflatten(0, (cases, members)))
    return torch.cat(channels, dim=2)


class _CompactTerminalModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input = nn.Conv2d(14, 12, 3, padding=1)
        self.output = nn.Conv2d(12, 6, 3, padding=1)

    def forward(
        self, value: torch.Tensor, timestamp: torch.Tensor, return_dict: bool = False
    ) -> tuple[torch.Tensor]:
        del return_dict
        time = timestamp.reshape(-1, 1, 1, 1) / 1000.0
        return (self.output(torch.tanh(self.input(value))) + 0.01 * time,)


def compact_coarse_budget_integration_check() -> dict[str, Any]:
    """Exercise the complete train-only law without loading data or a GPU."""
    torch.manual_seed(20260912)
    frozen_model = _CompactTerminalModel().eval()
    for parameter in frozen_model.parameters():
        parameter.requires_grad_(False)
    trainable_model = copy.deepcopy(frozen_model).train()
    for parameter in trainable_model.parameters():
        parameter.requires_grad_(True)
    if any(
        not torch.equal(frozen_model.state_dict()[name], value)
        for name, value in trainable_model.state_dict().items()
    ):
        raise RuntimeError("step-zero terminal copy differs from frozen model")

    members = 4
    fine_mask = torch.ones((1, 1, 8, 8), dtype=torch.float32)
    fine_mask[:, :, :2, :1] = 0.0
    coarse_mask = F.avg_pool2d(fine_mask, 2, 2)
    active = (coarse_mask > 0).float().expand(members, -1, -1, -1)
    condition = torch.randn((members, 6, 4, 4), dtype=torch.float32) * 0.1
    grid = make_normalized_xy_grid(4, 4, device=torch.device("cpu"), dtype=torch.float32)
    noise = torch.randn((members, 6, 4, 4), dtype=torch.float32) * 0.1 + 0.5
    prefix = frozen_prefix(frozen_model, noise, condition, active, grid)
    if prefix.requires_grad:
        raise RuntimeError("frozen prefix unexpectedly retains a graph")
    candidate_coarse = hybrid_terminal_sample(
        trainable_model, prefix, condition, active, grid
    ).unflatten(0, (1, members))
    with torch.no_grad():
        control_coarse = hybrid_terminal_sample(
            frozen_model, prefix, condition, active, grid
        ).unflatten(0, (1, members))
    candidate_control = float((candidate_coarse.detach() - control_coarse).abs().max())
    if candidate_control > 1e-7:
        raise RuntimeError("step-zero candidate coarse differs from control")

    member_mask = fine_mask[:, None].expand(-1, members, -1, -1, -1).flatten(0, 1)
    lifted = smooth_right_inverse(control_coarse.flatten(0, 1), member_mask).unflatten(
        0, (1, members)
    )
    residual_seed = torch.randn((1, members, 6, 8, 8), dtype=torch.float32) * 0.04
    residual = project_detail(residual_seed.flatten(0, 1), member_mask).unflatten(
        0, (1, members)
    ).detach()
    base_physical = (lifted + residual).detach()
    decoded_candidate = frozen_allocation_physical_law(
        base_physical, control_coarse.detach(), candidate_coarse, fine_mask
    )
    with torch.no_grad():
        decoded_control = apply_training_sic_decoder(
            base_physical, control_coarse.detach(), fine_mask
        )
    replay = float((decoded_candidate.detach() - decoded_control).abs().max())
    if replay != 0.0:
        raise RuntimeError("frozen-allocation step zero did not replay the scored law")

    truth = torch.rand((1, 6, 8, 8), dtype=torch.float64)
    truth[:, 0::2] = truth[:, 0::2].clamp(0.05, 0.95)
    channel_stds = torch.tensor([0.3, 0.5] * 3, dtype=torch.float64).reshape(
        1, 1, 6, 1, 1
    )
    objective, crps, energy = proper_objective(
        decoded_candidate / channel_stds,
        truth / channel_stds[:, 0],
        fine_mask.double(),
    )
    objective.backward()
    gradients = [
        parameter.grad
        for parameter in trainable_model.parameters()
        if parameter.grad is not None
    ]
    if not gradients or not all(torch.isfinite(value).all() for value in gradients):
        raise FloatingPointError("terminal parameters lack finite gradients")
    gradient_norm = float(
        torch.sqrt(sum(value.double().square().sum() for value in gradients))
    )
    if not math.isfinite(gradient_norm) or gradient_norm <= 0:
        raise FloatingPointError("terminal parameter gradient is dead")
    if any(parameter.grad is not None for parameter in frozen_model.parameters()):
        raise RuntimeError("frozen terminal model received gradients")
    if residual.requires_grad:
        raise RuntimeError("frozen residual unexpectedly received a graph")
    return {
        "status": "pass",
        "candidate_control_max_abs": candidate_control,
        "step0_replay_max_abs": replay,
        "objective": float(objective.detach()),
        "fair_crps": float(crps.detach()),
        "joint_energy": float(energy.detach()),
        "terminal_parameter_gradient_norm": gradient_norm,
        "frozen_prefix_has_no_graph": True,
        "frozen_residual_has_no_graph": True,
        "frozen_parameter_gradients_absent": True,
        "loss_domain": "all valid ocean pixels; no unstable-block exclusion",
        "endpoint_generalized_gradient": (
            "C=0 right-continuation over tied fine maxima; C=1 left-continuation "
            "over tied fine minima; C outside [0,1] locally flat; strict KKT on "
            "smooth interior active sets; exact interior kinks fail closed; no STE"
        ),
    }


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != "coarse_budget_refinement_cpu_integration_v1":
        raise ValueError("unreviewed coarse-budget integration schema")
    if config.get("optimizer_steps") != 0 or config.get("sampling_performed") is not False:
        raise ValueError("CPU integration must not train or sample")
    if config.get("test_2023") != "closed":
        raise ValueError("test-2023 must remain closed")
    contract = config.get("candidate_contract", {})
    if contract.get("objective") != (
        "0.75 standardized fair CRPS + 0.25 joint energy score on all valid water"
    ):
        raise ValueError("candidate objective differs from the reviewed proper score")
    if contract.get("fine_allocation") != "frozen residual and frozen conditioning":
        raise ValueError("candidate must freeze the fine allocation")


def _validate_sources(config: dict[str, Any], repo: Path) -> dict[str, Any]:
    bound: dict[str, Any] = {}
    for name, spec in config["sources"].items():
        path = repo / spec["path"] if spec.get("repo_relative") else Path(spec["path"])
        if not path.is_file() or _sha256(path) != spec["sha256"]:
            raise ValueError(f"coarse-budget integration source SHA mismatch: {name}")
        bound[name] = {"path": str(path), "sha256": spec["sha256"]}
    admission = load_json(bound["coarse_budget_sensitivity_metrics"]["path"])
    if (
        admission.get("status") != "complete"
        or admission.get("scientific_role")
        != "frozen_local_coarse_budget_sensitivity_not_calibration"
        or admission.get("optimizer_steps") != 0
        or admission.get("sampling_performed") is not False
        or admission.get("test_2023_used") is not False
    ):
        raise ValueError("source coarse-budget sensitivity admission is not complete")
    return bound


def run(config_path: Path, output: Path) -> dict[str, Any]:
    torch.set_num_threads(6)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_available() or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("coarse-budget integration requires CUDA_VISIBLE_DEVICES empty")
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
        sources = _validate_sources(config, repo)
        code_identity = _clean_code_identity(repo)
        tracker = ClearMLTracker(
            config["project_name"],
            f'{config["task_name"]}-{output.parent.name}',
            tags=config["clearml"]["tags"],
            env_path=config["clearml"]["env_path"],
        )
        tracker.connect("coarse_budget_refinement_contract", config)
        _strict_atomic_json(
            output,
            {
                **reservation,
                "status": "running",
                "code_identity": code_identity,
                "clearml_task_id": str(tracker.task.id),
            },
        )
        check = compact_coarse_budget_integration_check()
        result = {
            "schema_version": config["schema_version"],
            "status": "admission_passed_pending_astra_review",
            "scientific_role": "synthetic_cpu_integration_not_training_or_evaluation",
            "training_performed": False,
            "optimizer_steps": 0,
            "sampling_performed": False,
            "gpu_used": False,
            "test_2023_used": False,
            "code_identity": code_identity,
            "sources": sources,
            "candidate_contract": config["candidate_contract"],
            "integration_check": check,
        }
        _strict_atomic_json(metrics_path, result)
        for name, value in check.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                tracker.report_single_value(f"integration/{name}", value)
        tracker.upload_artifact("coarse_budget_refinement_cpu_integration", metrics_path)
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


def main() -> None:
    signal.signal(signal.SIGTERM, _terminate)
    signal.signal(signal.SIGINT, _terminate)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
