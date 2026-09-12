"""Variance-preconditioned flow parameterization for the fine cascade.

The target and base laws are unchanged::

    R = (I - U D) Y,  eta = (I - U D) xi,
    z_t = (1 - t) R + t eta.

The target law and population-optimal conditional velocity are unchanged.  A
closed-form Gaussian velocity handles the large linear contraction from
projected white noise to fine-scale sea-ice detail; the UNet learns the
remaining conditional structure.  Training the neural branch with ordinary
MSE is equivalent to a channel/time-dependent ``b(t)^-2`` weighted velocity
MSE, so finite-budget optimization is intentionally changed and recorded.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .direct_dynamics_cascade import project_detail
from .direct_dynamics_cascade_fine import (
    CASCADE_FACTOR,
    DIRECT_OUTPUT_CHANNELS,
    FINE_INPUT_CHANNELS,
    FineCascadeDynamicsTrainer,
    FineCascadeSampler,
    ProjectedDetailModel,
    VALID_MASK_MODEL_INPUT_CHANNEL,
    _sha256,
    load_fine_cascade_sampler,
)
from .sampler import Sampler
from .trainer import _atomic_json


PRECONDITIONING_KIND = "variance_preconditioned_projected_gaussian_v1"
PROJECTED_WHITE_BASE_KIND = "projected_white_gaussian_v1"


def _validate_scales(values: Any, label: str) -> tuple[float, ...]:
    if not isinstance(values, list) or len(values) != DIRECT_OUTPUT_CHANNELS:
        raise ValueError(f"{label} requires exactly six channel scales")
    scales = tuple(float(value) for value in values)
    if any(not math.isfinite(value) or value <= 0 for value in scales):
        raise ValueError(f"{label} scales must be finite and strictly positive")
    return scales


def validate_preconditioning_contract(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("kind") != PRECONDITIONING_KIND:
        raise ValueError("fine preconditioning contract has the wrong kind")
    sigma = _validate_scales(value.get("target_residual_rms"), "target residual RMS")
    beta = _validate_scales(value.get("projected_base_rms"), "projected base RMS")
    source = Path(str(value.get("source_audit_path", "")))
    expected_sha256 = str(value.get("source_audit_sha256", ""))
    if not source.is_file() or _sha256(source) != expected_sha256:
        raise ValueError("fine preconditioning source audit is missing or differs")
    audit = json.loads(source.read_text(encoding="utf-8"))
    if audit.get("split") != "train_only":
        raise ValueError("fine preconditioning statistics must be train-only")
    measured_sigma = _validate_scales(audit.get("measures", {}).get("target", {}).get("rms"), "audit target RMS")
    measured_beta = _validate_scales(audit.get("measures", {}).get("base", {}).get("rms"), "audit base RMS")
    if sigma != measured_sigma or beta != measured_beta:
        raise ValueError("declared preconditioning scales differ from the source audit")
    return {
        "kind": PRECONDITIONING_KIND,
        "target_residual_rms": list(sigma),
        "projected_base_rms": list(beta),
        "source_audit_path": str(source.resolve()),
        "source_audit_sha256": expected_sha256,
        "source_split": "train_only",
        "source_case_ids_sha256": audit.get("selected_case_ids_sha256"),
    }


def write_preconditioning_manifest(
    output_dir: str | Path,
    contract: dict[str, Any],
    code_commit: str,
) -> None:
    root = Path(output_dir)
    primary_path = root / "fine_cascade_manifest.json"
    if not primary_path.is_file():
        raise ValueError("preconditioned fine run lacks its primary cascade manifest")
    parameterization_path = root / "fine_cascade_preconditioning_manifest.json"
    _atomic_json(
        parameterization_path,
        {
            "schema_version": 1,
            "sampler": "VariancePreconditionedFineCascadeSampler",
            "preconditioning": contract,
            "training_loss": "neural_branch_mse_equivalent_to_b_t_inverse_squared_velocity_mse",
            "population_optimum": "unchanged_conditional_flow_velocity",
            "implementation_sha256": _sha256(Path(__file__)),
            "code_commit": code_commit,
        },
    )
    primary = json.loads(primary_path.read_text(encoding="utf-8"))
    if primary.get("code_commit") != code_commit:
        raise ValueError("primary and preconditioning manifest commits differ")
    primary.update(
        {
            "velocity_parameterization": PRECONDITIONING_KIND,
            "parameterization_manifest": parameterization_path.name,
            "parameterization_manifest_sha256": _sha256(parameterization_path),
        }
    )
    _atomic_json(primary_path, primary)


def _channel_tensor(values: tuple[float, ...], reference: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(values, dtype=reference.dtype, device=reference.device).view(
        1, DIRECT_OUTPUT_CHANNELS, 1, 1
    )


def preconditioning_coefficients(
    state: torch.Tensor,
    timesteps: torch.Tensor,
    target_residual_rms: tuple[float, ...],
    projected_base_rms: tuple[float, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``sqrt(q_t)``, analytic skip ``a_t`` and neural scale ``b_t``."""
    if state.ndim != 4 or state.shape[1] != DIRECT_OUTPUT_CHANNELS:
        raise ValueError("preconditioned fine state must have six channels")
    time = torch.as_tensor(timesteps, dtype=state.dtype, device=state.device)
    if time.ndim == 0:
        time = time.expand(state.shape[0])
    if time.ndim != 1 or time.shape[0] != state.shape[0]:
        raise ValueError("timesteps must be scalar or have one value per case")
    if not torch.isfinite(time).all() or torch.any((time < 0) | (time > 1)):
        raise ValueError("timesteps must be finite and lie in [0,1]")
    time = time.view(-1, 1, 1, 1)
    sigma = _channel_tensor(target_residual_rms, state)
    beta = _channel_tensor(projected_base_rms, state)
    q = (1 - time).square() * sigma.square() + time.square() * beta.square()
    if not torch.isfinite(q).all() or torch.any(q <= 0):
        raise FloatingPointError("fine preconditioning variance is not positive finite")
    scale = torch.sqrt(q)
    analytic = (time * beta.square() - (1 - time) * sigma.square()) / q
    neural = sigma * beta / scale
    return scale, analytic, neural


def preconditioned_model_state(
    state: torch.Tensor,
    timesteps: torch.Tensor,
    target_residual_rms: tuple[float, ...],
    projected_base_rms: tuple[float, ...],
) -> torch.Tensor:
    scale, _, _ = preconditioning_coefficients(
        state, timesteps, target_residual_rms, projected_base_rms
    )
    return state / scale


def reconstruct_preconditioned_velocity(
    model_output: torch.Tensor,
    state: torch.Tensor,
    timesteps: torch.Tensor,
    target_residual_rms: tuple[float, ...],
    projected_base_rms: tuple[float, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    if model_output.shape != state.shape:
        raise ValueError("fine preconditioned model output and state shapes differ")
    _, analytic, neural = preconditioning_coefficients(
        state, timesteps, target_residual_rms, projected_base_rms
    )
    return analytic * state + neural * model_output, neural


def preconditioned_flow_target(
    velocity: torch.Tensor,
    state: torch.Tensor,
    timesteps: torch.Tensor,
    target_residual_rms: tuple[float, ...],
    projected_base_rms: tuple[float, ...],
) -> torch.Tensor:
    """Return the exact regression target F* for the neural branch."""
    _, analytic, neural = preconditioning_coefficients(
        state, timesteps, target_residual_rms, projected_base_rms
    )
    if velocity.shape != state.shape:
        raise ValueError("fine velocity and state shapes differ")
    return (velocity - analytic * state) / neural


class _PreconditionedVelocityModel(nn.Module):
    """Adapter exposing reconstructed velocity to the unchanged ODE sampler."""

    def __init__(
        self,
        model: nn.Module,
        target_residual_rms: tuple[float, ...],
        projected_base_rms: tuple[float, ...],
    ):
        super().__init__()
        self.model = model if isinstance(model, ProjectedDetailModel) else ProjectedDetailModel(model)
        self.target_residual_rms = target_residual_rms
        self.projected_base_rms = projected_base_rms

    def forward(self, model_input: torch.Tensor, timestep: torch.Tensor, **_: Any):
        if model_input.ndim != 4 or model_input.shape[1] != FINE_INPUT_CHANNELS:
            raise ValueError("preconditioned fine sampler received an incompatible input")
        state = model_input[:, :DIRECT_OUTPUT_CHANNELS]
        normalized_time = timestep.to(dtype=state.dtype, device=state.device) / 1000.0
        normalized_state = preconditioned_model_state(
            state,
            normalized_time,
            self.target_residual_rms,
            self.projected_base_rms,
        )
        conditioned_input = torch.cat(
            (normalized_state, model_input[:, DIRECT_OUTPUT_CHANNELS:]), dim=1
        )
        neural = self.model(conditioned_input, timestep)[0]
        velocity, _ = reconstruct_preconditioned_velocity(
            neural,
            state,
            normalized_time,
            self.target_residual_rms,
            self.projected_base_rms,
        )
        valid_mask = model_input[
            :, VALID_MASK_MODEL_INPUT_CHANNEL : VALID_MASK_MODEL_INPUT_CHANNEL + 1
        ]
        return (project_detail(velocity.float(), valid_mask.float(), CASCADE_FACTOR),)


class VariancePreconditionedFineCascadeSampler(FineCascadeSampler):
    def __init__(
        self,
        model: nn.Module,
        target_residual_rms: tuple[float, ...],
        projected_base_rms: tuple[float, ...],
    ):
        self.target_residual_rms = target_residual_rms
        self.projected_base_rms = projected_base_rms
        self.sampler = Sampler(
            _PreconditionedVelocityModel(model, target_residual_rms, projected_base_rms)
        )
        self.capture_evidence = False
        self.evidence: list[dict[str, torch.Tensor]] = []


class VariancePreconditionedFineCascadeDynamicsTrainer(FineCascadeDynamicsTrainer):
    """Fine trainer with an exact channel-wise Gaussian-path parameterization."""

    def __init__(self, *args, **kwargs):
        experiment = kwargs.get("experiment_config")
        if not isinstance(experiment, dict):
            raise TypeError("variance-preconditioned trainer requires experiment_config")
        contract = validate_preconditioning_contract(experiment.get("fine_preconditioning"))
        self._target_residual_rms = tuple(contract["target_residual_rms"])
        self._projected_base_rms = tuple(contract["projected_base_rms"])
        self._preconditioning_contract = contract
        self._loss_inverse_neural_scale: torch.Tensor | None = None
        super().__init__(*args, **kwargs)
        write_preconditioning_manifest(
            self.output_dir,
            contract,
            self._fine_code_identity["git_commit"],
        )

    def _structured_model_state(
        self, state: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        return preconditioned_model_state(
            state,
            timesteps,
            self._target_residual_rms,
            self._projected_base_rms,
        )

    def _reconstruct_model_velocity(
        self,
        model_output: torch.Tensor,
        state: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        velocity, neural_scale = reconstruct_preconditioned_velocity(
            model_output,
            state,
            timesteps,
            self._target_residual_rms,
            self._projected_base_rms,
        )
        self._loss_inverse_neural_scale = neural_scale.reciprocal()
        return velocity

    def _flow_matching_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        inverse = self._loss_inverse_neural_scale
        self._loss_inverse_neural_scale = None
        if inverse is None:
            raise RuntimeError("preconditioned fine loss lacks its matching neural scale")
        valid_mask = batch["valid_mask"][:, :1].float()
        error = project_detail((pred.float() - target.float()) * inverse, valid_mask)
        return self._masked_mse(error, torch.zeros_like(error), valid_mask)

    def _make_sampler(self, model) -> VariancePreconditionedFineCascadeSampler:
        sampler = VariancePreconditionedFineCascadeSampler(
            model,
            self._target_residual_rms,
            self._projected_base_rms,
        )
        sampler.capture_evidence = True
        self._latest_fine_sampler = sampler
        return sampler


def load_variance_preconditioned_fine_cascade_sampler(
    run_dir: str,
    checkpoint_name: str,
    model_config: dict,
    expected_checkpoint_sha256: str,
    expected_code_commit: str,
    expected_forecast_contract_sha256: str,
    device=None,
    *,
    expected_base_parameterization: str = PROJECTED_WHITE_BASE_KIND,
) -> VariancePreconditionedFineCascadeSampler:
    root = Path(run_dir)
    manifest_path = root / "fine_cascade_preconditioning_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("fine checkpoint lacks its preconditioning manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    primary_path = root / "fine_cascade_manifest.json"
    if not primary_path.is_file():
        raise ValueError("fine checkpoint lacks its primary cascade manifest")
    primary = json.loads(primary_path.read_text(encoding="utf-8"))
    actual_base_parameterization = primary.get(
        "base_parameterization", PROJECTED_WHITE_BASE_KIND
    )
    if (
        manifest.get("schema_version") != 1
        or manifest.get("sampler") != "VariancePreconditionedFineCascadeSampler"
        or manifest.get("code_commit") != expected_code_commit
        or manifest.get("implementation_sha256") != _sha256(Path(__file__))
        or primary.get("velocity_parameterization") != PRECONDITIONING_KIND
        or primary.get("parameterization_manifest") != manifest_path.name
        or primary.get("parameterization_manifest_sha256") != _sha256(manifest_path)
        or actual_base_parameterization != expected_base_parameterization
    ):
        raise ValueError("fine preconditioning manifest is incompatible")
    contract = validate_preconditioning_contract(manifest.get("preconditioning"))
    base = load_fine_cascade_sampler(
        run_dir,
        checkpoint_name,
        model_config,
        expected_checkpoint_sha256,
        expected_code_commit,
        expected_forecast_contract_sha256,
        device=device,
        expected_velocity_parameterization=PRECONDITIONING_KIND,
    )
    return VariancePreconditionedFineCascadeSampler(
        base.sampler.model,
        tuple(contract["target_residual_rms"]),
        tuple(contract["projected_base_rms"]),
    )
