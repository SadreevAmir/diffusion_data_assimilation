"""Fine residual-flow stage for the two-resolution dynamics cascade."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import torch
from diffusers.training_utils import EMAModel
from torch import nn

from . import direct_dynamics_cascade as cascade_core
from .config import TrainingConfig
from .direct_dynamics_cascade import (
    decompose,
    masked_block_average,
    project_detail,
    residual_flow_pair,
    smooth_right_inverse,
)
from .direct_dynamics_cascade_contract import (
    forecast_contract_from_data_config,
    forecast_contract_sha256,
)
from .direct_dynamics_training import (
    DIRECT_CONDITION_CHANNELS,
    DIRECT_OUTPUT_CHANNELS,
    DirectDynamicsTrainer,
)
from .model_io import build_unet, resolve_checkpoint_name
from .runtime import make_normalized_xy_grid
from .sampler import Sampler
from .trainer import UNetTrainer, _atomic_json

CASCADE_FACTOR = 2
FINE_CONDITION_CHANNELS = DIRECT_CONDITION_CHANNELS + DIRECT_OUTPUT_CHANNELS
FINE_INPUT_CHANNELS = DIRECT_OUTPUT_CHANNELS + 2 + FINE_CONDITION_CHANNELS
VALID_MASK_CONDITION_CHANNEL = 2
VALID_MASK_MODEL_INPUT_CHANNEL = DIRECT_OUTPUT_CHANNELS + 2 + VALID_MASK_CONDITION_CHANNEL
RAW_FINE_VELOCITY_PARAMETERIZATION = "raw_projected_velocity_v1"


def _validate_causal_condition(
    structured_conditioning: torch.Tensor, valid_mask: torch.Tensor
) -> torch.Tensor:
    if structured_conditioning.ndim != 4 or structured_conditioning.shape[1] != DIRECT_CONDITION_CHANNELS:
        raise ValueError("fine stage requires the unchanged 15-channel dynamics condition")
    if valid_mask.ndim != 4 or valid_mask.shape[1] < 1:
        raise ValueError("fine stage requires a [batch,channel,y,x] valid mask")
    mask = valid_mask[:, :1]
    if (
        mask.shape[0] != structured_conditioning.shape[0]
        or mask.shape[-2:] != structured_conditioning.shape[-2:]
    ):
        raise ValueError("causal condition and valid mask shapes differ")
    if not torch.isfinite(structured_conditioning).all():
        raise FloatingPointError("causal condition contains NaN/Inf")
    if not torch.isfinite(mask).all() or torch.any((mask != 0) & (mask != 1)):
        raise ValueError("valid mask must be finite and binary")
    embedded = structured_conditioning[:, VALID_MASK_CONDITION_CHANNEL : VALID_MASK_CONDITION_CHANNEL + 1]
    if not torch.equal(embedded, mask.to(dtype=embedded.dtype, device=embedded.device)):
        raise ValueError("embedded dynamics mask differs from dataset valid_mask")
    return mask


def generated_coarse_condition(
    structured_conditioning: torch.Tensor,
    coarse_member: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Append a generated member's canonical fine-grid lift to the causal condition."""
    mask = _validate_causal_condition(structured_conditioning, valid_mask)
    if coarse_member.ndim != 4 or coarse_member.shape[:2] != (
        structured_conditioning.shape[0],
        DIRECT_OUTPUT_CHANNELS,
    ):
        raise ValueError("generated coarse member must have shape [batch,6,y/2,x/2]")
    expected = (
        structured_conditioning.shape[-2] // CASCADE_FACTOR,
        structured_conditioning.shape[-1] // CASCADE_FACTOR,
    )
    if coarse_member.shape[-2:] != expected:
        raise ValueError("generated coarse member has an incompatible spatial shape")
    lift = smooth_right_inverse(coarse_member.float(), mask.float(), CASCADE_FACTOR)
    condition = torch.cat((structured_conditioning.float(), lift), dim=1)
    validate_fine_condition(condition, mask.float())
    return condition


def validate_fine_condition(
    condition: torch.Tensor, valid_mask: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reject detail-contaminated or mask-inconsistent coarse conditioning."""
    if condition.ndim != 4 or condition.shape[1] != FINE_CONDITION_CHANNELS:
        raise ValueError("fine sampler requires 21-channel coarse conditioning")
    mask = condition[:, VALID_MASK_CONDITION_CHANNEL : VALID_MASK_CONDITION_CHANNEL + 1]
    _validate_causal_condition(condition[:, :DIRECT_CONDITION_CHANNELS], mask)
    if valid_mask is not None and not torch.equal(
        mask, valid_mask[:, :1].to(dtype=mask.dtype, device=mask.device)
    ):
        raise ValueError("fine condition mask differs from the supplied valid_mask")
    lift = condition[:, -DIRECT_OUTPUT_CHANNELS:]
    if not torch.isfinite(lift).all():
        raise FloatingPointError("coarse lift contains NaN/Inf")
    coarse, _ = masked_block_average(lift.float(), mask.float(), CASCADE_FACTOR)
    canonical = smooth_right_inverse(coarse, mask.float(), CASCADE_FACTOR)
    scale = max(float(canonical.abs().max()), 1.0)
    tolerance = 128 * torch.finfo(torch.float32).eps * scale
    if not torch.allclose(lift.float(), canonical, atol=tolerance, rtol=0):
        raise ValueError("fine condition contains a non-canonical or detail-contaminated coarse lift")
    return mask, lift


def teacher_coarse_condition(
    truth: torch.Tensor,
    structured_conditioning: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Append the exact lifted coarse target to the unchanged causal condition."""
    mask = _validate_causal_condition(structured_conditioning, valid_mask)
    state = decompose(truth, mask, CASCADE_FACTOR)
    condition = generated_coarse_condition(structured_conditioning, state.coarse, mask)
    return condition, state.coarse, state.residual


class ProjectedDetailModel(nn.Module):
    """Project every neural velocity into ker(D) during training and sampling."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, model_input: torch.Tensor, timestep: torch.Tensor, **_: Any):
        if model_input.ndim != 4 or model_input.shape[1] != FINE_INPUT_CHANNELS:
            raise ValueError("projected fine model received an incompatible input")
        valid_mask = model_input[:, VALID_MASK_MODEL_INPUT_CHANNEL : VALID_MASK_MODEL_INPUT_CHANNEL + 1]
        raw = self.model(model_input, timestep, return_dict=False)[0]
        if raw.shape[1] != DIRECT_OUTPUT_CHANNELS:
            raise ValueError("fine model must emit six joint residual velocities")
        return (project_detail(raw.float(), valid_mask.float(), CASCADE_FACTOR),)


class FineCascadeSampler:
    """Compose one matched coarse member with a projected fine residual draw."""

    def __init__(self, model: nn.Module):
        projected = model if isinstance(model, ProjectedDetailModel) else ProjectedDetailModel(model)
        self.sampler = Sampler(projected)
        self.capture_evidence = False
        self.evidence: list[dict[str, torch.Tensor]] = []

    def project_initial_noise(
        self, raw_noise: torch.Tensor, valid_mask: torch.Tensor
    ) -> torch.Tensor:
        """Map member-bound white noise to the exact reverse-ODE endpoint."""
        return project_detail(raw_noise.float(), valid_mask.float(), CASCADE_FACTOR)

    @torch.no_grad()
    def sample_conditioned(self, **kwargs: Any) -> torch.Tensor:
        condition = kwargs.get("model_conditioning")
        initial_noise = kwargs.get("initial_noise")
        if condition is None:
            raise ValueError("fine sampler requires teacher/generated coarse conditioning")
        if initial_noise is None:
            raise ValueError("fine sampler requires an explicit member-bound noise tensor")
        valid_mask, lift = validate_fine_condition(condition)
        kwargs = dict(kwargs)
        external_valid = kwargs.get("valid_mask")
        if external_valid is not None and not torch.equal(
            external_valid[:, :1].to(device=valid_mask.device, dtype=valid_mask.dtype),
            valid_mask,
        ):
            raise ValueError("sampler valid_mask differs from embedded fine-condition mask")
        expected_state_mask = valid_mask.expand(-1, DIRECT_OUTPUT_CHANNELS, -1, -1)
        external_state_mask = kwargs.get("state_mask")
        if external_state_mask is not None and not torch.equal(
            external_state_mask.to(device=expected_state_mask.device, dtype=expected_state_mask.dtype),
            expected_state_mask,
        ):
            raise ValueError("sampler state_mask differs from embedded fine-condition mask")
        raw_initial_noise = initial_noise.detach().cpu() if self.capture_evidence else None
        kwargs["initial_noise"] = self.project_initial_noise(initial_noise, valid_mask)
        kwargs["model_conditioning"] = condition.float()
        kwargs["valid_mask"] = valid_mask.float()
        kwargs["state_mask"] = expected_state_mask.float()
        for key in (
            "background",
            "background_mask",
            "obs_values",
            "obs_mask",
            "water_mask",
        ):
            value = kwargs.get(key)
            if torch.is_tensor(value):
                kwargs[key] = value.float()
        residual = self.sampler.sample_conditioned(**kwargs)
        residual = project_detail(residual.float(), valid_mask.float())
        result = lift + residual
        if not torch.isfinite(result[valid_mask.expand_as(result) > 0]).all():
            raise FloatingPointError("fine cascade sampling produced NaN/Inf on valid ocean")
        expected_coarse, expected_fraction = masked_block_average(lift.float(), valid_mask.float())
        actual_coarse, actual_fraction = masked_block_average(result.float(), valid_mask.float())
        if not torch.equal(expected_fraction, actual_fraction) or not torch.allclose(
            expected_coarse, actual_coarse, atol=3e-6, rtol=0
        ):
            raise RuntimeError("fine sampling changed its matched coarse member")
        if self.capture_evidence:
            self.evidence.append(
                {
                    "raw_initial_noise": raw_initial_noise,
                    "projected_initial_noise": kwargs["initial_noise"].detach().cpu(),
                    "matched_coarse_lift": lift.detach().cpu(),
                    "sampled_residual": residual.detach().cpu(),
                }
            )
        return result


class FineCascadeDynamicsTrainer(DirectDynamicsTrainer):
    """Train p(R | C, d0, forcing, calendar) with teacher coarse targets."""

    diagnostic_steps = frozenset({63, 255, 511, 1023, 2047, 4095, 6473})

    def __init__(self, *args, **kwargs):
        self._fine_code_identity = kwargs.pop("fine_code_identity", None)
        data_config = kwargs.get("data_config")
        if "model" not in kwargs:
            raise TypeError("fine cascade trainer requires model as a keyword argument")
        config = kwargs.get("config", args[0] if args else None)
        if not isinstance(config, TrainingConfig):
            raise TypeError("fine cascade trainer requires an explicit TrainingConfig")
        if config.training_objective != "flow":
            raise ValueError("fine cascade mechanics pilot requires training_objective='flow'")
        if config.in_channels != FINE_INPUT_CHANNELS or config.out_channels != DIRECT_OUTPUT_CHANNELS:
            raise ValueError("fine cascade mechanics pilot requires an exact 29-to-6 model")
        if config.activation_checkpointing:
            raise ValueError("fine cascade mechanics pilot forbids activation checkpointing")
        if config.gradient_accumulation_steps != 1:
            raise ValueError("fine cascade update accounting requires gradient_accumulation_steps=1")
        if config.metric_every_n_epochs != 0 or config.sample_every_n_epochs != 0:
            raise ValueError("fine cascade uses only audited oracle diagnostics during the pilot")
        if not isinstance(data_config, dict):
            raise TypeError("fine cascade trainer requires an explicit data_config")
        self._forecast_contract = forecast_contract_from_data_config(data_config)
        model = kwargs["model"]
        kwargs["model"] = model if isinstance(model, ProjectedDetailModel) else ProjectedDetailModel(model)
        super().__init__(*args, **kwargs)
        if self.config.num_epochs * len(self.train_dataloader) not in {512, 2048, 6474}:
            raise ValueError("fine cascade run must contain exactly 512, 2048, or 6474 optimizer updates")
        self._latest_fine_sampler: FineCascadeSampler | None = None
        self._fine_validation_case_ids: tuple[str, ...] = ()
        if self.accelerator.is_main_process:
            write_fine_manifest(
                self.output_dir,
                config,
                self._fine_code_identity,
                self._forecast_contract,
            )

    def _make_training_pair(
        self,
        truth: torch.Tensor,
        batch: dict[str, torch.Tensor],
        timesteps: torch.Tensor,
        residual_background: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del residual_background
        noise = torch.randn(
            truth.shape,
            dtype=truth.dtype,
            device=truth.device,
            generator=generator,
        )
        state, velocity, _, _ = residual_flow_pair(
            truth, noise, batch["valid_mask"][:, :1], timesteps, CASCADE_FACTOR
        )
        return state, velocity

    def _make_model_input(
        self,
        noisy_truth: torch.Tensor,
        batch: dict[str, torch.Tensor],
        **_: object,
    ) -> torch.Tensor:
        condition, _, _ = teacher_coarse_condition(
            batch["truth"], batch["structured_conditioning"], batch["valid_mask"]
        )
        grid = self._grid.expand(noisy_truth.shape[0], -1, -1, -1)
        model_input = torch.cat((noisy_truth, grid, condition), dim=1)
        if model_input.shape[1] != FINE_INPUT_CHANNELS:
            raise RuntimeError("fine cascade model input has the wrong channel count")
        return model_input

    def _flow_matching_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute loss only after both velocities are projected into ker(D)."""
        valid_mask = batch["valid_mask"][:, :1].float()
        projected_pred = project_detail(pred.float(), valid_mask, CASCADE_FACTOR)
        projected_target = project_detail(target.float(), valid_mask, CASCADE_FACTOR)
        return self._masked_mse(projected_pred, projected_target, valid_mask)

    def _fixed_validation_batch(self) -> dict[str, torch.Tensor]:
        if self._direct_validation_batch is None:
            raw = next(iter(self.val_dataloader))
            moved = self._batch_to_device(raw)
            self._direct_validation_batch = {
                key: value[: min(3, value.shape[0])] for key, value in moved.items() if torch.is_tensor(value)
            }
            raw_case_ids = raw.get("meta", {}).get("case_id", ())
            if isinstance(raw_case_ids, str):
                raw_case_ids = (raw_case_ids,)
            self._fine_validation_case_ids = tuple(str(value) for value in raw_case_ids[:3])
        batch = self._direct_validation_batch
        if batch["structured_conditioning"].shape[1] == DIRECT_CONDITION_CHANNELS:
            condition, _, _ = teacher_coarse_condition(
                batch["truth"], batch["structured_conditioning"], batch["valid_mask"]
            )
            batch["structured_conditioning"] = condition
        return batch

    def _make_sampler(self, model) -> FineCascadeSampler:
        sampler = FineCascadeSampler(model)
        sampler.capture_evidence = True
        self._latest_fine_sampler = sampler
        return sampler

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def save_model_custom(self, name: str = "last_model.pth"):
        """Keep checkpoints compatible with the raw six-output UNet constructor."""
        os.makedirs(self.output_dir, exist_ok=True)
        unwrapped = self.accelerator.unwrap_model(self.model)
        if not isinstance(unwrapped, ProjectedDetailModel):
            raise TypeError("fine cascade checkpoint requires the projected model wrapper")
        for path, state in (
            (Path(self.output_dir) / name, unwrapped.model.state_dict()),
            (Path(self.output_dir) / f"ema_{name}", self.ema_model.state_dict()),
        ):
            temporary = path.with_name(f".{path.name}.{os.getpid()}.incomplete")
            torch.save(state, temporary)
            os.replace(temporary, path)

    def _direct_diagnostic(self, step: int, label: str, *, case_count: int = 1) -> dict[str, float]:
        oracle_label = f"oracle_coarse_{label}"
        return super()._direct_diagnostic(step, oracle_label, case_count=case_count)

    def _augment_direct_diagnostic_payload(
        self, payload: dict[str, Any], *, step: int, label: str
    ) -> dict[str, Any]:
        """Attach complete oracle provenance before plotting or external reporting."""
        if not label.startswith("oracle_coarse_"):
            raise ValueError("fine diagnostics must be explicitly labeled oracle_coarse")
        sampler = self._latest_fine_sampler
        if sampler is None or len(sampler.evidence) != 2:
            raise RuntimeError("oracle diagnostic did not capture both member-bound noise records")
        payload["metrics"]["oracle_true_coarse_mechanics_only"] = 1.0
        payload.update(
            {
                "diagnostic_role": "oracle_true_coarse_mechanics_only",
                "future_truth_used_for_coarse_condition": True,
                "forecast_claim_permitted": False,
                "member_noise_seed_rule": "99173 + 1009*member_index",
                "validation_case_ids": list(self._fine_validation_case_ids[: payload["truth"].shape[0]]),
                "raw_initial_noise_normalized": torch.stack(
                    [entry["raw_initial_noise"] for entry in sampler.evidence], dim=1
                ),
                "projected_initial_noise_normalized": torch.stack(
                    [entry["projected_initial_noise"] for entry in sampler.evidence], dim=1
                ),
                "matched_coarse_lift_normalized": torch.stack(
                    [entry["matched_coarse_lift"] for entry in sampler.evidence], dim=1
                ),
                "sampled_residual_normalized": torch.stack(
                    [entry["sampled_residual"] for entry in sampler.evidence], dim=1
                ),
                "sampling_weight_role": self._sampling_weight_label(),
                "optimizer_updates": step + 1,
            }
        )
        checkpoint_name = f"mechanics_update_{step + 1:04d}.pth"
        checkpoint_path = Path(self.output_dir) / checkpoint_name
        ema_checkpoint_path = Path(self.output_dir) / f"ema_{checkpoint_name}"
        if not checkpoint_path.is_file() or not ema_checkpoint_path.is_file():
            raise FileNotFoundError("oracle diagnostic requires durable pre-sampling checkpoints")
        payload.update(
            {
                "checkpoint": checkpoint_name,
                "checkpoint_sha256": self._file_sha256(checkpoint_path),
                "ema_checkpoint": ema_checkpoint_path.name,
                "ema_checkpoint_sha256": self._file_sha256(ema_checkpoint_path),
            }
        )
        return payload

    def _report_train_metrics(self, loss, loss_full, loss_obs, loss_smooth, step: int):
        UNetTrainer._report_train_metrics(self, loss, loss_full, loss_obs, loss_smooth, step)
        if (
            self.accelerator.is_main_process
            and step in self.diagnostic_steps
            and step not in self._direct_diagnostic_steps
        ):
            self._direct_diagnostic_steps.add(step)
            checkpoint_name = f"mechanics_update_{step + 1:04d}.pth"
            self.save_model_custom(checkpoint_name)
            checkpoint_path = Path(self.output_dir) / checkpoint_name
            _atomic_json(
                Path(self.output_dir) / f"mechanics_update_{step + 1:04d}_identity.json",
                {
                    "optimizer_updates": step + 1,
                    "checkpoint": checkpoint_name,
                    "checkpoint_sha256": self._file_sha256(checkpoint_path),
                    "ema_checkpoint": f"ema_{checkpoint_name}",
                    "ema_checkpoint_sha256": self._file_sha256(
                        Path(self.output_dir) / f"ema_{checkpoint_name}"
                    ),
                    "diagnostic_role": "oracle_true_coarse_mechanics_only",
                },
            )
            self._direct_diagnostic(step, f"update_{step + 1:04d}")

    def _after_training_epoch(self, epoch: int, global_step: int) -> None:
        del epoch, global_step


def fine_model_input_for_test(
    state: torch.Tensor,
    structured_conditioning: torch.Tensor,
    truth: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Pure CPU constructor used by admission tests."""
    condition, _, _ = teacher_coarse_condition(truth, structured_conditioning, valid_mask)
    grid = make_normalized_xy_grid(*state.shape[-2:], dtype=state.dtype).expand(state.shape[0], -1, -1, -1)
    return torch.cat((state, grid, condition), dim=1)


def _architecture_signature(config: TrainingConfig) -> dict[str, Any]:
    return {
        "image_size": list(config.image_size),
        "in_channels": config.in_channels,
        "out_channels": config.out_channels,
        "block_out_channels": list(config.block_out_channels),
        "layers_per_block": config.layers_per_block,
        "down_block_types": list(config.down_block_types),
        "up_block_types": list(config.up_block_types),
        "norm_num_groups": config.norm_num_groups,
        "dropout": config.dropout,
        "add_attention": config.add_attention,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_fine_manifest(
    output_dir: str | Path,
    config: TrainingConfig,
    code_identity: dict[str, str] | None,
    forecast_contract: dict[str, Any],
) -> None:
    commit = None if code_identity is None else code_identity.get("git_commit")
    if commit is None or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("fine manifest requires an exact immutable git commit")
    _atomic_json(
        Path(output_dir) / "fine_cascade_manifest.json",
        {
            "schema_version": 2,
            "sampler": "FineCascadeSampler",
            "velocity_parameterization": RAW_FINE_VELOCITY_PARAMETERIZATION,
            "representation": "Y=U(C)+R; D(U(C))=C; D(R)=0",
            "conditional_law": "p(R|C_generated,c_causal)",
            "condition_channels": FINE_CONDITION_CHANNELS,
            "model_input_channels": FINE_INPUT_CHANNELS,
            "model_output_channels": DIRECT_OUTPUT_CHANNELS,
            "coarse_factor": CASCADE_FACTOR,
            "image_size": list(config.image_size),
            "architecture": _architecture_signature(config),
            "code_commit": commit,
            "conditioning_module_sha256": _sha256(Path(__file__)),
            "cascade_core_sha256": _sha256(Path(cascade_core.__file__)),
            "training_condition": "teacher_coarse",
            "forecast_condition": "generated_coarse_only",
            "training_diagnostics": "oracle_true_coarse_mechanics_only",
            "forecast_contract": forecast_contract,
            "forecast_contract_sha256": forecast_contract_sha256(forecast_contract),
        },
    )


def load_fine_cascade_sampler(
    run_dir: str,
    checkpoint_name: str,
    model_config: dict,
    expected_checkpoint_sha256: str,
    expected_code_commit: str,
    expected_forecast_contract_sha256: str,
    device=None,
    *,
    expected_velocity_parameterization: str = RAW_FINE_VELOCITY_PARAMETERIZATION,
) -> FineCascadeSampler:
    """Reload a fine-stage checkpoint without silently falling back to plain sampling."""
    root = Path(run_dir)
    manifest_path = root / "fine_cascade_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("fine cascade checkpoint is missing its sampler manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 2,
        "sampler": "FineCascadeSampler",
        "conditional_law": "p(R|C_generated,c_causal)",
        "condition_channels": FINE_CONDITION_CHANNELS,
        "model_input_channels": FINE_INPUT_CHANNELS,
        "model_output_channels": DIRECT_OUTPUT_CHANNELS,
        "coarse_factor": CASCADE_FACTOR,
        "image_size": [320, 256],
        "forecast_condition": "generated_coarse_only",
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("fine cascade sampler manifest is incompatible")
    actual_parameterization = manifest.get(
        "velocity_parameterization", RAW_FINE_VELOCITY_PARAMETERIZATION
    )
    if actual_parameterization != expected_velocity_parameterization:
        raise ValueError(
            "fine velocity parameterization differs from the requested loader"
        )
    if manifest.get("conditioning_module_sha256") != _sha256(Path(__file__)):
        raise ValueError("fine conditioning implementation differs from the checkpoint manifest")
    if manifest.get("code_commit") != expected_code_commit:
        raise ValueError("fine cascade code commit differs from the required identity")
    stored_contract = manifest.get("forecast_contract")
    if not isinstance(stored_contract, dict):
        raise ValueError("fine cascade manifest lacks its forecast contract body")
    stored_contract_sha256 = forecast_contract_sha256(stored_contract)
    if (
        manifest.get("forecast_contract_sha256") != stored_contract_sha256
        or stored_contract_sha256 != expected_forecast_contract_sha256
    ):
        raise ValueError("fine cascade forecast contract differs from the required identity")
    if manifest.get("cascade_core_sha256") != _sha256(Path(cascade_core.__file__)):
        raise ValueError("fine D/U implementation differs from the checkpoint manifest")
    config = TrainingConfig.from_dict(model_config)
    if (
        config.in_channels != FINE_INPUT_CHANNELS
        or config.out_channels != DIRECT_OUTPUT_CHANNELS
        or tuple(config.image_size) != (320, 256)
    ):
        raise ValueError("fine cascade checkpoint requires an exact 320x256 29-to-6 UNet")
    if manifest.get("architecture") != _architecture_signature(config):
        raise ValueError("fine cascade architecture differs from its sampler manifest")
    if checkpoint_name == "auto":
        raise ValueError("fine cascade production reload requires an explicit checkpoint")
    resolved_name = resolve_checkpoint_name(run_dir, checkpoint_name)
    checkpoint_path = root / resolved_name
    if _sha256(checkpoint_path) != expected_checkpoint_sha256:
        raise ValueError("fine cascade checkpoint SHA256 differs from the required identity")
    model = build_unet(config)
    try:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "shadow_params" in state:
        ema_model = EMAModel(model.parameters())
        ema_model.load_state_dict(state)
        ema_model.copy_to(model.parameters())
    else:
        model.load_state_dict(state)
    model.eval()
    if device is not None:
        model.to(device)
    return FineCascadeSampler(model)
