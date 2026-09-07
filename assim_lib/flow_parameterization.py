"""Exact parameterizations of the conditional-flow velocity field.

The structured model transports a clean latent ``z`` at ``t=0`` to an
independent standard-normal draw ``eps`` at ``t=1`` along the linear path

    x_t = (1 - t) z + t eps.

The Gaussian-path preconditioning below is only a reparameterization of the
velocity network.  It does not change that path, the target law, or the MSE on
the reconstructed latent velocity.
"""

from __future__ import annotations

import torch


RAW_VELOCITY = "raw"
GAUSSIAN_PATH_PRECONDITIONED = "gaussian_path_preconditioned"
VELOCITY_PARAMETERIZATIONS = frozenset(
    {RAW_VELOCITY, GAUSSIAN_PATH_PRECONDITIONED}
)


def validate_velocity_parameterization(value: str) -> str:
    value = str(value)
    if value not in VELOCITY_PARAMETERIZATIONS:
        raise ValueError(
            "unknown structured velocity parameterization "
            f"{value!r}; expected one of {sorted(VELOCITY_PARAMETERIZATIONS)}"
        )
    return value


def _broadcast_time(state: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
    time = torch.as_tensor(timesteps, device=state.device, dtype=state.dtype)
    if time.ndim == 0:
        time = time.expand(state.shape[0])
    if time.ndim != 1 or time.shape[0] != state.shape[0]:
        raise ValueError(
            "timesteps must be scalar or have one entry per state batch item"
        )
    if not torch.all(torch.isfinite(time)):
        raise ValueError("timesteps must be finite")
    if not torch.all((time >= 0.0) & (time <= 1.0)):
        raise ValueError("timesteps must lie in [0, 1]")
    return time.view(-1, *([1] * (state.ndim - 1)))


def gaussian_path_scale(
    state: torch.Tensor, timesteps: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return broadcast ``t`` and ``s(t)=sqrt((1-t)^2+t^2)``."""
    time = _broadcast_time(state, timesteps)
    scale_squared = (1.0 - time).square() + time.square()
    # Analytically s(t)^2 >= 1/2.  Keep the assertion fail-closed in case a
    # future dtype or formula change violates the contract.
    if not torch.all(scale_squared >= 0.5 - 8.0 * torch.finfo(state.dtype).eps):
        raise RuntimeError("linear Gaussian path scale violated its lower bound")
    return time, torch.sqrt(scale_squared)


def velocity_model_state(
    state: torch.Tensor,
    timesteps: torch.Tensor,
    parameterization: str,
) -> torch.Tensor:
    """Map the ODE state to the state channels consumed by the UNet."""
    parameterization = validate_velocity_parameterization(parameterization)
    if parameterization == RAW_VELOCITY:
        return state
    _, scale = gaussian_path_scale(state, timesteps)
    return state / scale


def reconstruct_velocity(
    model_output: torch.Tensor,
    state: torch.Tensor,
    timesteps: torch.Tensor,
    parameterization: str,
) -> torch.Tensor:
    """Reconstruct the full velocity used by both loss and ODE integration."""
    parameterization = validate_velocity_parameterization(parameterization)
    if model_output.shape != state.shape:
        raise ValueError(
            f"model output shape {tuple(model_output.shape)} does not match "
            f"state shape {tuple(state.shape)}"
        )
    if parameterization == RAW_VELOCITY:
        return model_output
    time, scale = gaussian_path_scale(state, timesteps)
    analytic = (2.0 * time - 1.0) * state / scale.square()
    return analytic + model_output / scale
