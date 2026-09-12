"""Deployment path for the exact coarse-to-fine dynamics cascade.

This module deliberately has no ``truth`` argument.  A forecast member is
sampled as ``C ~ p(C|c)`` followed by ``R ~ p(R|C,c)`` and reconstructed as
``Y = U(C) + R``.  Case-local, stage-local random streams make the result
independent of batch order and ensemble chunking.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import torch

from . import direct_dynamics_cascade as cascade_core
from . import direct_dynamics_cascade_coarse as coarse_implementation
from . import direct_dynamics_cascade_fine as fine_implementation
from .direct_dynamics_cascade import masked_block_average, project_detail, smooth_right_inverse
from .direct_dynamics_cascade_coarse import (
    CASCADE_FACTOR,
    CoarseCascadeDynamicsTrainer,
    load_coarse_cascade_sampler,
)
from .direct_dynamics_cascade_fine import generated_coarse_condition, load_fine_cascade_sampler
from .direct_dynamics_training import DIRECT_CONDITION_CHANNELS, DIRECT_OUTPUT_CHANNELS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class CascadeMember:
    """Tensors required to audit one generated trajectory member."""

    forecast: torch.Tensor
    coarse: torch.Tensor
    residual: torch.Tensor
    raw_coarse_noise: torch.Tensor
    raw_fine_noise: torch.Tensor
    projected_fine_noise: torch.Tensor
    coarse_seeds: tuple[int, ...]
    fine_seeds: tuple[int, ...]
    case_ids: tuple[str, ...]
    member_index: int
    solver: dict[str, int | float | str]
    replay_identity: dict[str, str] | None


def _casewise_noise(
    case_ids: tuple[str, ...],
    member: int,
    stage: str,
    shape: tuple[int, int, int],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, tuple[int, ...]]:
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("cascade inference requires unique case identities")
    noises = []
    seeds = []
    for case_id in case_ids:
        seed = CoarseCascadeDynamicsTrainer._member_seed(case_id, member, stage=stage)
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        noises.append(torch.randn((1, *shape), device=device, dtype=dtype, generator=generator))
        seeds.append(seed)
    return torch.cat(noises, dim=0), tuple(seeds)


class CascadePredictor:
    """Compose already-loaded coarse and fine samplers without oracle inputs."""

    def __init__(self, coarse_sampler, fine_sampler, replay_identity: dict[str, str] | None = None):
        self.coarse_sampler = coarse_sampler
        self.fine_sampler = fine_sampler
        self.replay_identity = None if replay_identity is None else dict(replay_identity)
        self.coarse_sampler.sampler.model.eval()
        self.fine_sampler.sampler.model.eval()

    @torch.no_grad()
    def sample_member(
        self,
        *,
        structured_conditioning: torch.Tensor,
        valid_mask: torch.Tensor,
        case_ids: tuple[str, ...],
        member: int,
        coarse_num_timesteps: int,
        fine_num_timesteps: int,
        device: torch.device | None = None,
        method: str = "rk4",
        rtol: float = 1e-5,
        atol: float = 1e-6,
        end_time: float = 0.0,
    ) -> CascadeMember:
        if structured_conditioning.ndim != 4 or structured_conditioning.shape[1] != DIRECT_CONDITION_CHANNELS:
            raise ValueError("cascade predictor requires the exact 15-channel causal condition")
        if valid_mask.ndim != 4 or valid_mask.shape[1] < 1:
            raise ValueError("cascade predictor requires a [batch,channel,y,x] valid mask")
        if structured_conditioning.shape[0] != len(case_ids):
            raise ValueError("case identity count differs from the conditioning batch")
        if (
            valid_mask.shape[0] != len(case_ids)
            or valid_mask.shape[-2:] != structured_conditioning.shape[-2:]
        ):
            raise ValueError("valid mask shape differs from the conditioning batch")
        if member < 0:
            raise ValueError("member index must be non-negative")
        if float(end_time) != 0.0:
            raise ValueError("production cascade inference requires the exact flow endpoint end_time=0")
        if coarse_num_timesteps < 2 or fine_num_timesteps < 2:
            raise ValueError("each cascade solver requires at least two timesteps")
        if (
            structured_conditioning.shape[-2] % CASCADE_FACTOR
            or structured_conditioning.shape[-1] % CASCADE_FACTOR
        ):
            raise ValueError("cascade grid dimensions must be divisible by the cascade factor")

        device = torch.device(device or structured_conditioning.device)
        condition = structured_conditioning.to(device=device, dtype=torch.float32)
        mask = valid_mask[:, :1].to(device=device, dtype=torch.float32)
        batch, _, height, width = condition.shape
        coarse_shape = (
            DIRECT_OUTPUT_CHANNELS,
            height // CASCADE_FACTOR,
            width // CASCADE_FACTOR,
        )
        coarse_noise, coarse_seeds = _casewise_noise(
            case_ids,
            member,
            "coarse",
            coarse_shape,
            device=device,
            dtype=torch.float32,
        )
        coarse = self.coarse_sampler.sample_conditioned(
            structured_conditioning=condition,
            valid_mask=mask,
            initial_noise=coarse_noise,
            num_timesteps=coarse_num_timesteps,
            device=device,
            method=method,
            rtol=rtol,
            atol=atol,
            end_time=end_time,
        ).float()

        fine_condition = generated_coarse_condition(condition, coarse, mask)
        fine_noise, fine_seeds = _casewise_noise(
            case_ids,
            member,
            "fine",
            (DIRECT_OUTPUT_CHANNELS, height, width),
            device=device,
            dtype=torch.float32,
        )
        if hasattr(self.fine_sampler, "project_initial_noise"):
            projected_noise = self.fine_sampler.project_initial_noise(fine_noise, mask)
        else:
            projected_noise = project_detail(fine_noise, mask, CASCADE_FACTOR)
        zeros = torch.zeros((batch, DIRECT_OUTPUT_CHANNELS, height, width), device=device)
        empty_obs = torch.zeros((batch, 2, height, width), device=device)
        forecast = self.fine_sampler.sample_conditioned(
            background=zeros,
            background_mask=torch.ones_like(zeros),
            obs_values=empty_obs,
            obs_mask=empty_obs,
            water_mask=mask,
            valid_mask=mask,
            state_mask=mask.expand(-1, DIRECT_OUTPUT_CHANNELS, -1, -1),
            size=(height, width),
            num_timesteps=fine_num_timesteps,
            device=device,
            method=method,
            rtol=rtol,
            atol=atol,
            start_mode="noise",
            initial_noise=fine_noise,
            sample_target="state",
            model_conditioning=fine_condition,
            state_channels=DIRECT_OUTPUT_CHANNELS,
            end_time=end_time,
        ).float()
        recovered_coarse, fraction = masked_block_average(forecast, mask, CASCADE_FACTOR)
        active = fraction > 0
        scale = max(float(coarse[active].abs().max()), 1.0)
        tolerance = 64 * torch.finfo(torch.float32).eps * scale
        if not torch.allclose(recovered_coarse[active], coarse[active], atol=tolerance, rtol=0):
            raise RuntimeError("end-to-end fine stage changed its matched coarse member")
        lift = smooth_right_inverse(coarse, mask, CASCADE_FACTOR)
        residual = project_detail(forecast - lift, mask, CASCADE_FACTOR)
        return CascadeMember(
            forecast=forecast,
            coarse=coarse,
            residual=residual,
            raw_coarse_noise=coarse_noise,
            raw_fine_noise=fine_noise,
            projected_fine_noise=projected_noise,
            coarse_seeds=coarse_seeds,
            fine_seeds=fine_seeds,
            case_ids=case_ids,
            member_index=member,
            solver={
                "method": method,
                "coarse_num_timesteps": coarse_num_timesteps,
                "fine_num_timesteps": fine_num_timesteps,
                "rtol": float(rtol),
                "atol": float(atol),
                "end_time": 0.0,
            },
            replay_identity=self.replay_identity,
        )

    @torch.no_grad()
    def sample_ensemble(
        self,
        *,
        member_indices: tuple[int, ...],
        storage_device: torch.device | str = "cpu",
        **kwargs,
    ) -> dict[str, torch.Tensor | tuple | dict | None]:
        if not member_indices or any(index < 0 for index in member_indices):
            raise ValueError("member_indices must contain non-negative ensemble identities")
        if len(member_indices) != len(set(member_indices)):
            raise ValueError("member_indices must be unique")
        stored_members = []
        for index in member_indices:
            member = self.sample_member(member=index, **kwargs)
            stored_members.append(
                CascadeMember(
                    forecast=member.forecast.detach().to(storage_device),
                    coarse=member.coarse.detach().to(storage_device),
                    residual=member.residual.detach().to(storage_device),
                    raw_coarse_noise=member.raw_coarse_noise.detach().to(storage_device),
                    raw_fine_noise=member.raw_fine_noise.detach().to(storage_device),
                    projected_fine_noise=member.projected_fine_noise.detach().to(storage_device),
                    coarse_seeds=member.coarse_seeds,
                    fine_seeds=member.fine_seeds,
                    case_ids=member.case_ids,
                    member_index=member.member_index,
                    solver=member.solver,
                    replay_identity=member.replay_identity,
                )
            )
        return {
            "forecast": torch.stack([item.forecast for item in stored_members], dim=1),
            "coarse": torch.stack([item.coarse for item in stored_members], dim=1),
            "residual": torch.stack([item.residual for item in stored_members], dim=1),
            "raw_coarse_noise": torch.stack([item.raw_coarse_noise for item in stored_members], dim=1),
            "raw_fine_noise": torch.stack([item.raw_fine_noise for item in stored_members], dim=1),
            "projected_fine_noise": torch.stack(
                [item.projected_fine_noise for item in stored_members], dim=1
            ),
            "coarse_seeds": tuple(item.coarse_seeds for item in stored_members),
            "fine_seeds": tuple(item.fine_seeds for item in stored_members),
            "member_indices": member_indices,
            "case_ids": stored_members[0].case_ids,
            "solver": stored_members[0].solver,
            "replay_identity": self.replay_identity,
        }


def load_cascade_predictor(
    *,
    coarse_run_dir: str,
    coarse_checkpoint_name: str,
    coarse_model_config: dict,
    coarse_checkpoint_sha256: str,
    fine_run_dir: str,
    fine_checkpoint_name: str,
    fine_model_config: dict,
    fine_checkpoint_sha256: str,
    expected_coarse_code_commit: str,
    expected_fine_code_commit: str,
    replay_code_commit: str,
    expected_forecast_contract_sha256: str,
    device: torch.device | None = None,
) -> CascadePredictor:
    """Strictly reload two stage-specific commits under one forecast contract."""
    if re.fullmatch(r"[0-9a-f]{40}", replay_code_commit) is None:
        raise ValueError("cascade replay requires an immutable current code commit")
    coarse = load_coarse_cascade_sampler(
        coarse_run_dir,
        coarse_checkpoint_name,
        coarse_model_config,
        coarse_checkpoint_sha256,
        expected_coarse_code_commit,
        expected_forecast_contract_sha256,
        device=device,
    )
    matched_manifest = Path(fine_run_dir) / "fine_cascade_matched_base_manifest.json"
    preconditioned_manifest = (
        Path(fine_run_dir) / "fine_cascade_preconditioning_manifest.json"
    )
    colored_manifest = Path(fine_run_dir) / "fine_cascade_colored_base_manifest.json"
    fine_primary = json.loads(
        (Path(fine_run_dir) / "fine_cascade_manifest.json").read_text(encoding="utf-8")
    )
    if (
        fine_primary.get("base_parameterization")
        == "projected_binomial_blend_gaussian_v1"
        and not colored_manifest.is_file()
    ):
        raise ValueError("colored fine checkpoint is missing its colored-base manifest")
    if matched_manifest.is_file() and (
        preconditioned_manifest.is_file() or colored_manifest.is_file()
    ):
        raise ValueError("fine checkpoint declares conflicting sampler parameterizations")
    if colored_manifest.is_file() and not preconditioned_manifest.is_file():
        raise ValueError("colored fine checkpoint lacks variance preconditioning")
    if colored_manifest.is_file():
        from .direct_dynamics_cascade_fine_colored import (
            load_colored_variance_preconditioned_fine_cascade_sampler,
        )

        fine = load_colored_variance_preconditioned_fine_cascade_sampler(
            fine_run_dir,
            fine_checkpoint_name,
            fine_model_config,
            fine_checkpoint_sha256,
            expected_fine_code_commit,
            expected_forecast_contract_sha256,
            device=device,
        )
    elif preconditioned_manifest.is_file():
        from .direct_dynamics_cascade_fine_preconditioned import (
            load_variance_preconditioned_fine_cascade_sampler,
        )

        fine = load_variance_preconditioned_fine_cascade_sampler(
            fine_run_dir,
            fine_checkpoint_name,
            fine_model_config,
            fine_checkpoint_sha256,
            expected_fine_code_commit,
            expected_forecast_contract_sha256,
            device=device,
        )
    elif matched_manifest.is_file():
        from .direct_dynamics_cascade_fine_matched_scale import (
            load_matched_scale_fine_cascade_sampler,
        )

        fine = load_matched_scale_fine_cascade_sampler(
            fine_run_dir,
            fine_checkpoint_name,
            fine_model_config,
            fine_checkpoint_sha256,
            expected_fine_code_commit,
            expected_forecast_contract_sha256,
            device=device,
        )
    else:
        fine = load_fine_cascade_sampler(
            fine_run_dir,
            fine_checkpoint_name,
            fine_model_config,
            fine_checkpoint_sha256,
            expected_fine_code_commit,
            expected_forecast_contract_sha256,
            device=device,
        )
    replay_identity = {
        "coarse_checkpoint_sha256": coarse_checkpoint_sha256,
        "fine_checkpoint_sha256": fine_checkpoint_sha256,
        "coarse_training_commit": expected_coarse_code_commit,
        "fine_training_commit": expected_fine_code_commit,
        "replay_code_commit": replay_code_commit,
        "forecast_contract_sha256": expected_forecast_contract_sha256,
        "coarse_manifest_sha256": _sha256(Path(coarse_run_dir) / "coarse_cascade_manifest.json"),
        "fine_manifest_sha256": _sha256(Path(fine_run_dir) / "fine_cascade_manifest.json"),
        "cascade_core_sha256": _sha256(Path(cascade_core.__file__)),
        "coarse_loader_sha256": _sha256(Path(coarse_implementation.__file__)),
        "fine_loader_sha256": _sha256(Path(fine_implementation.__file__)),
    }
    if matched_manifest.is_file():
        replay_identity["fine_matched_base_manifest_sha256"] = _sha256(matched_manifest)
    if preconditioned_manifest.is_file():
        replay_identity["fine_preconditioning_manifest_sha256"] = _sha256(
            preconditioned_manifest
        )
    if colored_manifest.is_file():
        replay_identity["fine_colored_base_manifest_sha256"] = _sha256(colored_manifest)
    return CascadePredictor(coarse, fine, replay_identity=replay_identity)
