"""Frozen CPU admission of SIC sensitivity to the coarse block budget."""

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
from .direct_dynamics_cascade_coarse_fine_boundary_audit import (
    _finish_success,
    _record_failure,
    _sha256,
    _strict_atomic_json,
    _terminate,
)
from .direct_dynamics_cascade import smooth_right_inverse
from .direct_dynamics_cascade_coarse_proper_refinement import proper_objective
from .direct_dynamics_cascade_fine_training import _clean_code_identity
from .direct_dynamics_fine_support_proper_admission import (
    _pre_correction_kkt_free_set,
    apply_training_sic_decoder,
)
from .direct_dynamics_sic_decoder_jacobian_audit import (
    _validate_embedded_identity,
    _validate_provenance,
    _verified_load,
)
from .direct_dynamics_sic_support_decoder import (
    project_masked_blocks_to_unit_interval_mean,
)
from .direct_dynamics_sic_support_decoder_scoring import canonical_physical_decode
from .direct_dynamics_sit_support_decoder import _from_blocks, _to_blocks


SIC_CHANNELS = (0, 2, 4)
LEADS = ("d3_sic", "d6_sic", "d9_sic")


def _projection_unconstrained(
    fine: torch.Tensor,
    coarse: torch.Tensor,
    mask: torch.Tensor,
    factor: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the pre-clamp argument, valid mask, and valid count per block."""
    values = fine.double()
    support = mask.to(device=fine.device).expand_as(fine) > 0
    blocks = _to_blocks(values, factor)
    block_mask = _to_blocks(support, factor)
    count = block_mask.sum(dim=-1)
    active = count > 0
    target_sum = count.double() * coarse.double().clamp(0.0, 1.0)
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
    return centered - (0.5 * (lower + upper)).unsqueeze(-1), block_mask, count


def _decode_with_shifted_budget(
    raw: torch.Tensor,
    coarse: torch.Tensor,
    coarse_prime: torch.Tensor,
    mask: torch.Tensor,
    factor: int = 2,
) -> torch.Tensor:
    """Hold the fine residual fixed while changing the coarse state and budget."""
    delta = coarse_prime.double() - coarse.double()
    shifted = raw.double() + smooth_right_inverse(delta, mask, factor)
    return project_masked_blocks_to_unit_interval_mean(
        shifted, coarse_prime.double(), mask, factor
    )


def _finite_difference_gate(
    predicted: float, observed: float, *, atol: float, rtol: float
) -> dict[str, float]:
    if not all(math.isfinite(value) for value in (predicted, observed, atol, rtol)):
        raise FloatingPointError("coarse-budget finite difference is not finite")
    if atol < 0 or rtol < 0:
        raise ValueError("finite-difference tolerances must be nonnegative")
    absolute_error = abs(observed - predicted)
    tolerance = atol + rtol * abs(predicted)
    if absolute_error > tolerance:
        raise RuntimeError(
            "coarse-budget VJP failed finite-difference admission: "
            f"error={absolute_error:.6e} tolerance={tolerance:.6e}"
        )
    return {
        "predicted_directional_derivative": predicted,
        "observed_directional_derivative": observed,
        "absolute_error": absolute_error,
        "relative_error": absolute_error / max(abs(predicted), 1e-30),
        "admission_tolerance": tolerance,
    }


def _coarse_budget_vjp_statistics(
    raw: torch.Tensor,
    coarse: torch.Tensor,
    mask: torch.Tensor,
    decoded_gradient: torch.Tensor,
    *,
    kink_margin: float,
    finite_difference_epsilon: float,
    finite_difference_atol: float,
    finite_difference_rtol: float,
) -> dict[str, Any]:
    """Audit the exact-a.e. VJP through lifted coarse state and decoder budget.

    The production smooth right-inverse changes fine values across neighboring
    blocks, while the decoder budget changes the target block sum.  On a fixed
    active set the full VJP is U^T Jg plus n/k times the free-gradient sum.
    The reported finite difference uses only stable output blocks and stable
    coarse directions, so no derivative is invented at a projection kink.
    """
    if raw.shape != decoded_gradient.shape or raw.ndim != 4 or raw.shape[1] != 1:
        raise ValueError("raw and decoded_gradient must be matching [batch,1,y,x]")
    if coarse.shape != (raw.shape[0], 1, raw.shape[-2] // 2, raw.shape[-1] // 2):
        raise ValueError("coarse shape is incompatible with raw")
    if kink_margin <= finite_difference_epsilon or finite_difference_epsilon <= 0:
        raise ValueError("finite-difference epsilon must be positive and below kink margin")

    member_mask = mask.to(raw.device).expand_as(raw)
    free = _pre_correction_kkt_free_set(raw, coarse, member_mask, 2)
    free_blocks = _to_blocks(free, 2)
    gradient_blocks = _to_blocks(decoded_gradient.double(), 2)
    unconstrained, block_mask, valid_count = _projection_unconstrained(
        raw, coarse, member_mask, 2
    )
    free_count = free_blocks.sum(dim=-1)
    active = valid_count > 0
    budget = coarse.double().clamp(0.0, 1.0)
    interior = active & (budget > 0.0) & (budget < 1.0)
    clipping_flat = active & ((coarse.double() < 0.0) | (coarse.double() > 1.0))
    exact_clipping_kink = active & (
        (coarse.double() == 0.0) | (coarse.double() == 1.0)
    )
    distance_to_active_set_kink = torch.where(
        block_mask,
        torch.minimum(unconstrained.abs(), (unconstrained - 1.0).abs()),
        torch.full_like(unconstrained, torch.inf),
    ).min(dim=-1).values
    exact_active_set_kink = interior & (distance_to_active_set_kink == 0.0)
    near_active_set = (
        interior
        & (distance_to_active_set_kink > 0.0)
        & (distance_to_active_set_kink <= kink_margin)
    )
    near_clipping_kink = (
        interior
        & ((budget <= kink_margin) | (budget >= 1.0 - kink_margin))
    )
    no_free_interior = interior & (free_count == 0)
    stable = (
        interior
        & (free_count > 0)
        & (budget > kink_margin)
        & (budget < 1.0 - kink_margin)
        & (distance_to_active_set_kink > kink_margin)
    )
    excluded = active & ~stable
    free_mean = (
        (gradient_blocks * free_blocks).sum(dim=-1, keepdim=True)
        / free_count.unsqueeze(-1).clamp_min(1)
    )
    retained_blocks = torch.where(
        free_blocks & stable.unsqueeze(-1),
        gradient_blocks - free_mean,
        torch.zeros_like(gradient_blocks),
    )
    retained = _from_blocks(retained_blocks, 2)
    lift_input = torch.zeros_like(coarse, dtype=torch.float64, requires_grad=True)
    lift = smooth_right_inverse(lift_input, member_mask, 2)
    (lift * retained).sum().backward()
    if lift_input.grad is None or not torch.isfinite(lift_input.grad).all():
        raise FloatingPointError("production lift VJP is invalid")
    lift_vjp = lift_input.grad.detach()
    free_gradient_sum = (gradient_blocks * free_blocks).sum(dim=-1)
    budget_vjp = (
        valid_count.double() / free_count.double().clamp_min(1) * free_gradient_sum
    )
    budget_vjp = torch.where(stable, budget_vjp, torch.zeros_like(budget_vjp))
    total_vjp = lift_vjp + budget_vjp
    defined_total_vjp = torch.where(stable, total_vjp, torch.zeros_like(total_vjp))
    stable_count = int(stable.sum())
    active_count = int(active.sum())
    if stable_count == 0:
        finite_difference = None
        vjp_norm = None
        vjp_rms = None
        lift_vjp_norm = None
        budget_vjp_norm = None
        nonzero_fraction = None
    else:
        scale = float(defined_total_vjp[stable].abs().max())
        if scale == 0.0:
            direction = torch.zeros_like(defined_total_vjp)
            finite_difference = None
        else:
            direction = torch.where(stable, -defined_total_vjp / scale, 0.0)
            plus_coarse = coarse + finite_difference_epsilon * direction
            minus_coarse = coarse - finite_difference_epsilon * direction
            plus_raw = raw.double() + smooth_right_inverse(
                plus_coarse.double() - coarse.double(), member_mask, 2
            )
            minus_raw = raw.double() + smooth_right_inverse(
                minus_coarse.double() - coarse.double(), member_mask, 2
            )
            plus_free = _to_blocks(
                _pre_correction_kkt_free_set(
                    plus_raw, plus_coarse, member_mask, 2
                ),
                2,
            )
            minus_free = _to_blocks(
                _pre_correction_kkt_free_set(
                    minus_raw, minus_coarse, member_mask, 2
                ),
                2,
            )
            compared = stable.unsqueeze(-1).expand_as(free_blocks)
            if torch.any((plus_free != free_blocks) & compared) or torch.any(
                (minus_free != free_blocks) & compared
            ):
                raise RuntimeError(
                    "finite-difference perturbation changed a reviewed stable active set"
                )
            plus = _decode_with_shifted_budget(
                raw, coarse, plus_coarse, member_mask
            )
            minus = _decode_with_shifted_budget(
                raw, coarse, minus_coarse, member_mask
            )
            restricted_gradient = _from_blocks(
                torch.where(
                    stable.unsqueeze(-1),
                    gradient_blocks,
                    torch.zeros_like(gradient_blocks),
                ),
                2,
            )
            observed = float(
                (restricted_gradient * (plus - minus)).sum()
                / (2.0 * finite_difference_epsilon)
            )
            predicted = float((total_vjp * direction).sum())
            finite_difference = {
                "epsilon": finite_difference_epsilon,
                **_finite_difference_gate(
                    predicted,
                    observed,
                    atol=finite_difference_atol,
                    rtol=finite_difference_rtol,
                ),
            }
        vjp_norm = float(torch.linalg.vector_norm(defined_total_vjp[stable]))
        vjp_rms = math.sqrt(float(defined_total_vjp[stable].square().mean()))
        lift_vjp_norm = float(torch.linalg.vector_norm(lift_vjp[stable]))
        budget_vjp_norm = float(torch.linalg.vector_norm(budget_vjp[stable]))
        nonzero_fraction = float((defined_total_vjp[stable] != 0).sum()) / stable_count

    return {
        "active_blocks": active_count,
        "stable_differentiable_blocks": stable_count,
        "stable_differentiable_fraction_of_active": stable_count / active_count,
        "excluded_from_restricted_probe_blocks": int(excluded.sum()),
        "clipping_flat_blocks": int(clipping_flat.sum()),
        "exact_clipping_kink_blocks": int(exact_clipping_kink.sum()),
        "exact_active_set_kink_blocks": int(exact_active_set_kink.sum()),
        "near_active_set_excluded_blocks": int(near_active_set.sum()),
        "near_clipping_kink_excluded_blocks": int(near_clipping_kink.sum()),
        "no_free_interior_blocks": int(no_free_interior.sum()),
        "coarse_budget_vjp_l2_norm": vjp_norm,
        "coarse_budget_vjp_rms": vjp_rms,
        "production_lift_vjp_l2_norm": lift_vjp_norm,
        "decoder_budget_vjp_l2_norm": budget_vjp_norm,
        "stable_blocks_with_nonzero_vjp_fraction": nonzero_fraction,
        "finite_difference": finite_difference,
        "local_derivative": (
            "Restricted functional: decoded gradients outside stable output blocks "
            "and coarse directions outside stable input blocks are held at zero. "
            "Within it, dL/dC = U^T Jg + (n/k) sum_free(g), using the production "
            "smooth right-inverse. Exact kinks and numerical neighborhoods are "
            "reported separately; no STE or surrogate zero derivative is used."
        ),
    }


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != "sic_coarse_budget_sensitivity_v1":
        raise ValueError("unreviewed SIC coarse-budget sensitivity schema")
    if config.get("split") != "valid" or config.get("test_2023") != "closed":
        raise ValueError("probe is restricted to frozen validation-2022 evidence")
    if config.get("cases") != 12 or config.get("members") != 8:
        raise ValueError("probe requires the frozen 12 x 8 panel")
    if config.get("optimizer_steps") != 0 or config.get("sampling_performed") is not False:
        raise ValueError("probe must not train or sample")
    if tuple(config.get("branches", {}).keys()) != ("control", "candidate"):
        raise ValueError("probe requires frozen control and candidate branches")
    epsilon = config.get("finite_difference_epsilon")
    margin = config.get("kink_margin")
    if not isinstance(epsilon, (int, float)) or not isinstance(margin, (int, float)):
        raise ValueError("probe requires numeric finite-difference controls")
    atol = config.get("finite_difference_atol")
    rtol = config.get("finite_difference_rtol")
    if not isinstance(atol, (int, float)) or not isinstance(rtol, (int, float)):
        raise ValueError("probe requires numeric finite-difference tolerances")
    if not 0 < epsilon < margin < 0.01 or atol < 0 or not 0 < rtol < 0.01:
        raise ValueError("finite-difference controls are outside the reviewed range")
    if len(config.get("implementation_bindings", {})) != 5:
        raise ValueError("probe requires five frozen implementation bindings")


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    active = sum(row["active_blocks"] for row in rows)
    stable = sum(row["stable_differentiable_blocks"] for row in rows)
    norm_rows = [
        row["coarse_budget_vjp_l2_norm"]
        for row in rows
        if row["coarse_budget_vjp_l2_norm"] is not None
    ]
    norms_sq = sum(
        row["coarse_budget_vjp_l2_norm"] ** 2
        for row in rows
        if row["coarse_budget_vjp_l2_norm"] is not None
    )
    fd_rows = [row["finite_difference"] for row in rows if row["finite_difference"]]
    return {
        "active_blocks": active,
        "stable_differentiable_blocks": stable,
        "stable_differentiable_fraction_of_active": stable / active,
        "excluded_from_restricted_probe_blocks": sum(
            row["excluded_from_restricted_probe_blocks"] for row in rows
        ),
        "clipping_flat_blocks": sum(row["clipping_flat_blocks"] for row in rows),
        "exact_clipping_kink_blocks": sum(
            row["exact_clipping_kink_blocks"] for row in rows
        ),
        "exact_active_set_kink_blocks": sum(
            row["exact_active_set_kink_blocks"] for row in rows
        ),
        "near_active_set_excluded_blocks": sum(
            row["near_active_set_excluded_blocks"] for row in rows
        ),
        "near_clipping_kink_excluded_blocks": sum(
            row["near_clipping_kink_excluded_blocks"] for row in rows
        ),
        "no_free_interior_blocks": sum(row["no_free_interior_blocks"] for row in rows),
        "coarse_budget_vjp_l2_norm": math.sqrt(norms_sq) if norm_rows else None,
        "finite_difference_max_relative_error": (
            max(row["relative_error"] for row in fd_rows) if fd_rows else None
        ),
        "finite_difference_max_absolute_error": (
            max(row["absolute_error"] for row in fd_rows) if fd_rows else None
        ),
        "per_case": rows,
    }


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
    branches: dict[str, Any] = {}
    for branch, spec in config["branches"].items():
        evidence = _verified_load(Path(spec["path"]), spec["sha256"])
        _validate_embedded_identity(
            evidence, fixed_spec["sha256"], provenance["case_ids"], branch
        )
        forecast_normalized = evidence.pop("forecast_normalized")
        coarse_normalized = evidence.pop("coarse_normalized")
        del evidence
        accumulators: dict[str, list[dict[str, Any]]] = {lead: [] for lead in LEADS}
        replay_max = 0.0
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
            objective, _, _ = proper_objective(
                decoded / stds,
                truth[case : case + 1] / stds[:, 0],
                mask[case : case + 1].double(),
            )
            objective.backward()
            gradient = decoded.grad
            if gradient is None or not torch.isfinite(gradient).all():
                raise FloatingPointError("proper-score decoded gradient is invalid")
            member_mask = mask[case : case + 1, None].expand(-1, 8, -1, -1, -1).flatten(0, 1)
            for lead, channel in zip(LEADS, SIC_CHANNELS):
                raw_channel = raw[:, :, channel : channel + 1].flatten(0, 1)
                coarse_channel = coarse[:, :, channel : channel + 1].flatten(0, 1)
                replay = _decode_with_shifted_budget(
                    raw_channel, coarse_channel, coarse_channel, member_mask
                ).unflatten(0, (1, 8))
                replay_max = max(
                    replay_max,
                    float((replay - decoded[:, :, channel : channel + 1]).abs().max()),
                )
                accumulators[lead].append(
                    _coarse_budget_vjp_statistics(
                        raw_channel,
                        coarse_channel,
                        member_mask,
                        gradient[:, :, channel : channel + 1].flatten(0, 1),
                        kink_margin=float(config["kink_margin"]),
                        finite_difference_epsilon=float(
                            config["finite_difference_epsilon"]
                        ),
                        finite_difference_atol=float(config["finite_difference_atol"]),
                        finite_difference_rtol=float(config["finite_difference_rtol"]),
                    )
                )
            del raw, coarse, decoded, gradient, objective
        if replay_max != 0.0:
            raise RuntimeError("coarse-budget path did not exactly replay the scored law")
        branches[branch] = {
            "exact_replay_max_abs": replay_max,
            "sic_leads": {
                lead: _aggregate_rows(rows) for lead, rows in accumulators.items()
            },
        }
        del forecast_normalized, coarse_normalized
    return {
        "schema_version": config["schema_version"],
        "status": "complete",
        "scientific_role": "frozen_local_coarse_budget_sensitivity_not_calibration",
        "split": "valid",
        "panel": "frozen_12_case_validation_2022_development_reuse",
        "cases": 12,
        "members": 8,
        "optimizer_steps": 0,
        "sampling_performed": False,
        "test_2023_used": False,
        "finite_difference_epsilon": config["finite_difference_epsilon"],
        "kink_margin": config["kink_margin"],
        "code_identity": code_identity,
        "provenance": provenance,
        "source_sha256": {
            "fixed_inputs": fixed_spec["sha256"],
            **{name: spec["sha256"] for name, spec in config["branches"].items()},
        },
        "branches": branches,
    }


def run(config_path: Path, output: Path) -> dict[str, Any]:
    torch.set_num_threads(6)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_available() or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("coarse-budget probe requires CUDA_VISIBLE_DEVICES empty")
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
        tracker.connect("coarse_budget_sensitivity_contract", config)
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
                    "stable_differentiable_fraction_of_active",
                    "coarse_budget_vjp_l2_norm",
                    "finite_difference_max_relative_error",
                ):
                    if row[name] is not None:
                        tracker.report_single_value(f"{branch}/{lead}/{name}", row[name])
        tracker.upload_artifact("sic_coarse_budget_sensitivity", metrics_path)
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
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
