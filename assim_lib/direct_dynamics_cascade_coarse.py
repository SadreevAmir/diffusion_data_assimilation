"""Coarse conditional-flow stage for the exact two-resolution dynamics cascade.

The coarse target is ``C = D(Y)``.  Conditioning is moved to the coarse grid
without discarding any causal spatial information: every 2x2 subpixel of the
non-calendar dynamics condition becomes a separate channel (space-to-depth),
while the four spatially constant calendar channels are stored once.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from diffusers.training_utils import EMAModel

from .config import TrainingConfig
from .direct_dynamics_cascade import masked_block_average
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
SPATIAL_CONDITION_CHANNELS = 11
CALENDAR_CONDITION_CHANNELS = 4
COARSE_CONDITION_CHANNELS = SPATIAL_CONDITION_CHANNELS * CASCADE_FACTOR**2 + CALENDAR_CONDITION_CHANNELS
COARSE_INPUT_CHANNELS = DIRECT_OUTPUT_CHANNELS + 2 + COARSE_CONDITION_CHANNELS
VALID_MASK_CONDITION_CHANNEL = 2


def _validate_full_condition(
    structured_conditioning: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    if structured_conditioning.ndim != 4 or structured_conditioning.shape[1] != DIRECT_CONDITION_CHANNELS:
        raise ValueError("coarse stage requires the exact 15-channel dynamics condition")
    if valid_mask.ndim != 4 or valid_mask.shape[1] < 1:
        raise ValueError("valid_mask must have shape [batch,channel,y,x]")
    mask = valid_mask[:, :1]
    if (
        mask.shape[0] != structured_conditioning.shape[0]
        or mask.shape[-2:] != structured_conditioning.shape[-2:]
    ):
        raise ValueError("dynamics condition and valid_mask shapes differ")
    if (
        structured_conditioning.shape[-2] % CASCADE_FACTOR
        or structured_conditioning.shape[-1] % CASCADE_FACTOR
    ):
        raise ValueError("dynamics condition dimensions must be divisible by the cascade factor")
    if not torch.isfinite(structured_conditioning).all():
        raise FloatingPointError("dynamics condition contains NaN/Inf")
    if not torch.isfinite(mask).all() or torch.any((mask != 0) & (mask != 1)):
        raise ValueError("valid_mask must be finite and binary")
    embedded = structured_conditioning[:, VALID_MASK_CONDITION_CHANNEL : VALID_MASK_CONDITION_CHANNEL + 1]
    if not torch.equal(embedded, mask.to(dtype=embedded.dtype, device=embedded.device)):
        raise ValueError("embedded dynamics mask differs from valid_mask")
    ocean = mask.expand(-1, 2, -1, -1) > 0
    if torch.any(structured_conditioning[:, :2][~ocean] != 0):
        raise ValueError("initial-state condition must be zero outside valid ocean")
    forcing_values = structured_conditioning[:, 3:7]
    forcing_masks = structured_conditioning[:, 7:11]
    if torch.any((forcing_masks != 0) & (forcing_masks != 1)):
        raise ValueError("forcing masks must be binary")
    if torch.any(forcing_masks > mask):
        raise ValueError("forcing masks cannot extend outside valid ocean")
    if torch.any(forcing_values[forcing_masks == 0] != 0):
        raise ValueError("forcing values must be zero wherever their mask is absent")
    return mask


def lossless_coarse_condition(
    structured_conditioning: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return an information-preserving causal condition on the coarse grid.

    Channels 0:11 (d0 SIC/SIT, ocean mask, four forcing values and four
    forcing masks) use a fixed 2x2 space-to-depth transform.  This preserves
    coastal geometry and field-specific missingness exactly.  Channels 11:15
    are required to be spatial constants and are therefore represented once.
    """
    mask = _validate_full_condition(structured_conditioning, valid_mask)
    spatial = F.pixel_unshuffle(
        structured_conditioning[:, :SPATIAL_CONDITION_CHANNELS],
        CASCADE_FACTOR,
    )
    calendar = structured_conditioning[:, SPATIAL_CONDITION_CHANNELS:]
    reference = calendar[..., :1, :1]
    if not torch.equal(calendar, reference.expand_as(calendar)):
        raise ValueError("calendar dynamics channels must be spatial constants")
    coarse_calendar = reference.expand(
        -1,
        -1,
        structured_conditioning.shape[-2] // CASCADE_FACTOR,
        structured_conditioning.shape[-1] // CASCADE_FACTOR,
    )
    encoded = torch.cat((spatial, coarse_calendar), dim=1)
    if encoded.shape[1] != COARSE_CONDITION_CHANNELS:
        raise RuntimeError("lossless coarse condition has the wrong channel count")
    _, ocean_fraction = masked_block_average(mask.float(), mask.float(), CASCADE_FACTOR)
    active = (ocean_fraction > 0).to(dtype=structured_conditioning.dtype)
    return encoded, active, ocean_fraction


def recover_full_condition_for_test(encoded: torch.Tensor) -> torch.Tensor:
    """Invert the declared coarse encoding; intended for contract tests only."""
    if encoded.ndim != 4 or encoded.shape[1] != COARSE_CONDITION_CHANNELS:
        raise ValueError("encoded condition has the wrong shape")
    spatial = F.pixel_shuffle(
        encoded[:, : SPATIAL_CONDITION_CHANNELS * CASCADE_FACTOR**2],
        CASCADE_FACTOR,
    )
    calendar = (
        encoded[:, -CALENDAR_CONDITION_CHANNELS:]
        .repeat_interleave(CASCADE_FACTOR, -2)
        .repeat_interleave(CASCADE_FACTOR, -1)
    )
    return torch.cat((spatial, calendar), dim=1)


def coarse_target(
    truth: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct C=D(Y), its binary support, and fractional ocean coverage."""
    if truth.ndim != 4 or truth.shape[1] != DIRECT_OUTPUT_CHANNELS:
        raise ValueError("coarse target requires six SIC/SIT trajectory channels")
    coarse, fractions = masked_block_average(truth.float(), valid_mask[:, :1].float(), CASCADE_FACTOR)
    fraction = fractions[:, :1]
    active = (fraction > 0).to(dtype=coarse.dtype)
    return coarse, active, fraction


def coarse_flow_pair(
    truth: torch.Tensor,
    noise: torch.Tensor,
    valid_mask: torch.Tensor,
    time: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the standard noise-to-C conditional-flow pair on active cells."""
    clean, active, fraction = coarse_target(truth, valid_mask)
    if noise.shape != clean.shape:
        raise ValueError("coarse noise and target shapes differ")
    if time.ndim != 1 or time.shape[0] != clean.shape[0]:
        raise ValueError("time must contain one scalar per case")
    if not torch.isfinite(time).all() or torch.any((time < 0) | (time > 1)):
        raise ValueError("flow time must lie in [0,1]")
    support = active.expand_as(clean) > 0
    if not torch.isfinite(noise[support]).all():
        raise FloatingPointError("coarse noise contains NaN/Inf on active ocean")
    safe_noise = torch.where(support, noise.float(), torch.zeros_like(clean))
    view = time.to(device=clean.device, dtype=clean.dtype).reshape(-1, 1, 1, 1)
    state = (1.0 - view) * clean + view * safe_noise
    velocity = safe_noise - clean
    if not torch.isfinite(state[support]).all() or not torch.isfinite(velocity[support]).all():
        raise FloatingPointError("coarse flow pair produced NaN/Inf on active ocean")
    return state, velocity, clean, fraction


def ocean_fraction_weighted_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    ocean_fraction: torch.Tensor,
) -> torch.Tensor:
    """Case-equal loss preserving the fine-grid valid-ocean pixel geometry."""
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("coarse prediction and target must have identical 4D shapes")
    if ocean_fraction.shape != (prediction.shape[0], 1, *prediction.shape[-2:]):
        raise ValueError("ocean_fraction has an incompatible shape")
    if not torch.isfinite(ocean_fraction).all() or torch.any((ocean_fraction < 0) | (ocean_fraction > 1)):
        raise ValueError("ocean_fraction must be finite and lie in [0,1]")
    weights = ocean_fraction.to(device=prediction.device, dtype=prediction.dtype)
    support = weights.expand_as(prediction) > 0
    if not torch.isfinite(prediction[support]).all() or not torch.isfinite(target[support]).all():
        raise FloatingPointError("coarse loss received NaN/Inf on active ocean")
    safe_prediction = torch.where(support, prediction, torch.zeros_like(prediction))
    safe_target = torch.where(support, target, torch.zeros_like(target))
    squared = (safe_prediction - safe_target).square()
    if not torch.isfinite(squared[support]).all():
        raise FloatingPointError("coarse squared error overflowed on active ocean")
    numerator = (squared * weights).sum(dim=(1, 2, 3))
    denominator = prediction.shape[1] * weights.sum(dim=(1, 2, 3))
    if torch.any(denominator <= 0):
        raise ValueError("every coarse training case must contain valid ocean")
    result = (numerator / denominator).mean()
    if not torch.isfinite(result):
        raise FloatingPointError("coarse weighted loss reduction overflowed")
    return result


class CoarseCascadeSampler:
    """Sample C from the losslessly encoded full causal dynamics condition."""

    def __init__(self, model):
        self.sampler = Sampler(model)

    @torch.no_grad()
    def sample_conditioned(
        self,
        *,
        structured_conditioning: torch.Tensor,
        valid_mask: torch.Tensor,
        initial_noise: torch.Tensor,
        num_timesteps: int,
        device=None,
        method: str = "rk4",
        rtol: float = 1e-5,
        atol: float = 1e-6,
        end_time: float = 0.0,
    ) -> torch.Tensor:
        condition, active, _ = lossless_coarse_condition(structured_conditioning, valid_mask)
        condition = condition.float()
        active = active.float()
        expected = (condition.shape[0], DIRECT_OUTPUT_CHANNELS, *condition.shape[-2:])
        if tuple(initial_noise.shape) != expected:
            raise ValueError(f"coarse initial_noise must have shape {expected}")
        device = device or initial_noise.device
        support = active.expand(-1, DIRECT_OUTPUT_CHANNELS, -1, -1) > 0
        raw_noise = initial_noise.to(device=active.device)
        if not torch.isfinite(raw_noise[support]).all():
            raise FloatingPointError("coarse initial noise contains NaN/Inf on active ocean")
        initial_noise = torch.where(support, raw_noise.float(), torch.zeros_like(raw_noise.float()))
        zeros = torch.zeros(expected, dtype=torch.float32, device=condition.device)
        empty_obs = torch.zeros((expected[0], 2, *expected[-2:]), dtype=torch.float32, device=zeros.device)
        coarse = self.sampler.sample_conditioned(
            background=zeros,
            background_mask=torch.ones_like(zeros),
            obs_values=empty_obs,
            obs_mask=empty_obs,
            water_mask=active,
            size=expected[-2:],
            num_timesteps=num_timesteps,
            device=device,
            method=method,
            rtol=rtol,
            atol=atol,
            start_mode="noise",
            initial_noise=initial_noise,
            sample_target="state",
            model_conditioning=condition,
            state_channels=DIRECT_OUTPUT_CHANNELS,
            end_time=end_time,
            valid_mask=active,
            state_mask=active.expand(-1, DIRECT_OUTPUT_CHANNELS, -1, -1),
        )
        support = active.to(device=coarse.device).expand_as(coarse) > 0
        if not torch.isfinite(coarse[support]).all():
            raise FloatingPointError("coarse sampler produced NaN/Inf on active ocean")
        return torch.where(support, coarse.float(), torch.zeros_like(coarse.float()))


class CoarseLearningCurveEarlyStop(RuntimeError):
    """Expected scientific stop after a bounded, non-improving learning curve."""


class CoarseCascadeDynamicsTrainer(DirectDynamicsTrainer):
    """Train p(C | full causal d0/forcing/calendar condition)."""

    diagnostic_steps = frozenset({63, 255, 511})

    def __init__(self, *args, **kwargs):
        self._coarse_diagnostic_batch_raw = kwargs.pop("coarse_diagnostic_batch", None)
        self._coarse_code_identity = kwargs.pop("coarse_code_identity", None)
        config = kwargs.get("config", args[0] if args else None)
        if not isinstance(config, TrainingConfig):
            raise TypeError("coarse cascade trainer requires an explicit TrainingConfig")
        if config.training_objective != "flow":
            raise ValueError("coarse cascade requires training_objective='flow'")
        if config.in_channels != COARSE_INPUT_CHANNELS or config.out_channels != DIRECT_OUTPUT_CHANNELS:
            raise ValueError("coarse cascade requires an exact 56-to-6 model")
        if tuple(config.image_size) != (160, 128):
            raise ValueError("coarse cascade model grid must be 160x128")
        if config.activation_checkpointing:
            raise ValueError("coarse cascade pilot forbids activation checkpointing")
        if config.gradient_accumulation_steps != 1:
            raise ValueError("coarse cascade pilot requires gradient_accumulation_steps=1")
        if config.timestep_sampler != "stratified_uniform":
            raise ValueError("coarse cascade pilot requires stratified_uniform flow times")
        if config.metric_every_n_epochs != 0 or config.sample_every_n_epochs != 0:
            raise ValueError("coarse cascade pilot forbids incompatible generic diagnostics")
        super().__init__(*args, **kwargs)
        planned_updates = self.config.num_epochs * len(self.train_dataloader)
        if planned_updates not in {512, 2048}:
            raise ValueError("coarse cascade pilot must contain exactly 512 or 2048 optimizer updates")
        self._planned_updates = planned_updates
        self.diagnostic_steps = (
            frozenset({63, 255, 511}) if planned_updates == 512 else frozenset({511, 1023, 1535, 2047})
        )
        self._coarse_diagnostic_batch = None
        self._coarse_diagnostic_case_ids: tuple[str, ...] = ()
        self._coarse_diagnostic_steps: set[int] = set()
        self._coarse_diagnostic_history: list[dict[str, float]] = []
        if self.accelerator.is_main_process:
            write_coarse_manifest(self.output_dir, config, self._coarse_code_identity)

    def _full_state_recovery_enabled(self) -> bool:
        """Persist model, optimizer, scheduler, scaler and RNG at every epoch boundary."""
        return True

    def _make_training_pair(
        self,
        truth: torch.Tensor,
        batch: dict[str, torch.Tensor],
        timesteps: torch.Tensor,
        residual_background: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del residual_background
        shape = (truth.shape[0], DIRECT_OUTPUT_CHANNELS, truth.shape[-2] // 2, truth.shape[-1] // 2)
        noise = torch.randn(shape, dtype=truth.dtype, device=truth.device, generator=generator)
        state, velocity, _, _ = coarse_flow_pair(truth, noise, batch["valid_mask"][:, :1], timesteps)
        return state, velocity

    def _make_model_input(
        self,
        noisy_truth: torch.Tensor,
        batch: dict[str, torch.Tensor],
        **_: object,
    ) -> torch.Tensor:
        condition, _, _ = lossless_coarse_condition(
            batch["structured_conditioning"], batch["valid_mask"][:, :1]
        )
        grid = self._grid.expand(noisy_truth.shape[0], -1, -1, -1)
        model_input = torch.cat((noisy_truth, grid, condition), dim=1)
        if model_input.shape[1] != COARSE_INPUT_CHANNELS:
            raise RuntimeError("coarse cascade model input has the wrong channel count")
        return model_input

    def _flow_matching_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        _, _, ocean_fraction = coarse_target(batch["truth"], batch["valid_mask"][:, :1])
        return ocean_fraction_weighted_mse(pred, target, ocean_fraction)

    def _make_sampler(self, model) -> CoarseCascadeSampler:
        return CoarseCascadeSampler(model)

    def _report_train_metrics(self, loss, loss_full, loss_obs, loss_smooth, step: int):
        UNetTrainer._report_train_metrics(self, loss, loss_full, loss_obs, loss_smooth, step)
        if (
            self.accelerator.is_main_process
            and self._planned_updates == 512
            and step in self.diagnostic_steps
            and step not in self._coarse_diagnostic_steps
        ):
            self._coarse_diagnostic_steps.add(step)
            checkpoint = f"coarse_update_{step + 1:04d}.pth"
            self.save_model_custom(checkpoint)
            self._coarse_diagnostic(step, f"update_{step + 1:04d}", checkpoint)

    def _after_training_epoch(self, epoch: int, global_step: int) -> None:
        if not self.accelerator.is_main_process:
            return
        planned_updates = getattr(self, "_planned_updates", global_step)
        expected_steps = [64, 256, 512] if planned_updates == 512 else [512, 1024, 1536, 2048]
        if planned_updates == 2048 and global_step in expected_steps:
            diagnostic_step = global_step - 1
            if diagnostic_step not in self._coarse_diagnostic_steps:
                self._require_full_state_recovery(global_step)
                self._coarse_diagnostic_steps.add(diagnostic_step)
                checkpoint = f"coarse_update_{global_step:04d}.pth"
                self.save_model_custom(checkpoint)
                self._coarse_diagnostic(
                    diagnostic_step,
                    f"update_{global_step:04d}",
                    checkpoint,
                )
        actual_steps = [int(record["step"]) for record in self._coarse_diagnostic_history]
        expected_so_far = [step for step in expected_steps if step <= global_step]
        if actual_steps != expected_so_far:
            raise RuntimeError(
                f"coarse mechanics gate requires diagnostics {expected_so_far}, got {actual_steps}"
            )
        if planned_updates == 2048 and global_step < planned_updates:
            progress = self._learning_curve_progress(epoch, global_step)
            _atomic_json(Path(self.output_dir) / "coarse_learning_curve_progress.json", progress)
            if progress["decision"] == "stop_no_learning_with_persistent_speckle":
                _atomic_json(Path(self.output_dir) / "coarse_mechanics_gate.json", progress)
                raise CoarseLearningCurveEarlyStop(progress["reason"])
            return
        if global_step != planned_updates:
            raise RuntimeError(
                f"coarse mechanics terminal gate expected step {planned_updates}, got {global_step}"
            )
        early = self._coarse_diagnostic_history[0]
        final = self._coarse_diagnostic_history[-1]
        baseline_step = int(early["step"])
        improvement_key = f"rmse_improvement_from_update{baseline_step}_by_output"
        mean_improvement_key = f"mean_dimensionless_rmse_improvement_from_update{baseline_step}"
        rmse_keys = sorted(key for key in final if key.endswith("_raw_mean_rmse"))
        persistence_keys = [key.replace("_raw_mean_rmse", "_persistence_rmse") for key in rmse_keys]
        roughness_keys = sorted(key for key in final if key.endswith("_member_roughness"))
        if any(early[key] <= 0 for key in rmse_keys) or any(final[key] <= 0 for key in persistence_keys):
            raise ValueError("coarse mechanics ratio denominators must be strictly positive")
        improvement_by_output = {
            key.removesuffix("_raw_mean_rmse"): 1.0 - final[key] / early[key] for key in rmse_keys
        }
        skill_by_output = {
            key.removesuffix("_persistence_rmse"): 1.0
            - final[key.replace("_persistence_rmse", "_raw_mean_rmse")] / final[key]
            for key in persistence_keys
        }
        roughness_ratios = [
            final[key] / max(final[key.replace("_member_roughness", "_truth_roughness")], 1e-8)
            for key in roughness_keys
        ]
        summary = {
            "rmse_improvement_reference_step": baseline_step,
            improvement_key: improvement_by_output,
            "rmse_skill_over_persistence_by_output": skill_by_output,
            mean_improvement_key: sum(improvement_by_output.values()) / len(improvement_by_output),
            "mean_dimensionless_rmse_skill_over_persistence": sum(skill_by_output.values())
            / len(skill_by_output),
            "max_member_to_truth_roughness_ratio": max(roughness_ratios),
            "joint_raw_support_violation_fraction": final["joint_raw_support_violation_fraction"],
        }
        flattened_summary = [
            *improvement_by_output.values(),
            *skill_by_output.values(),
            summary[mean_improvement_key],
            summary["mean_dimensionless_rmse_skill_over_persistence"],
            summary["max_member_to_truth_roughness_ratio"],
            summary["joint_raw_support_violation_fraction"],
        ]
        if not all(math.isfinite(value) for value in flattened_summary):
            raise FloatingPointError("coarse mechanics gate produced non-finite dimensionless ratios")
        reasons = []
        if summary[mean_improvement_key] < 0.20:
            reasons.append(
                f"mean dimensionless RMSE ratio improved by less than 20% from update{baseline_step}"
            )
        if summary["mean_dimensionless_rmse_skill_over_persistence"] <= 0:
            reasons.append("mean dimensionless coarse skill did not beat persistence")
        if summary["max_member_to_truth_roughness_ratio"] >= 10:
            reasons.append("raw coarse member roughness reached at least 10x truth")
        if summary["joint_raw_support_violation_fraction"] >= 0.95:
            reasons.append("at least 95% of raw coarse predictions violate physical support")
        gate = {
            "status": "failed" if reasons else "passed",
            "decision": "reject_mechanics_candidate" if reasons else "eligible_for_paired_review",
            "epoch": epoch + 1,
            "optimizer_updates": global_step,
            "criteria": {
                "rmse_improvement_reference_step": baseline_step,
                f"min_rmse_improvement_from_update{baseline_step}": 0.20,
                "strictly_positive_rmse_skill_over_persistence": True,
                "max_roughness_ratio_exclusive": 10.0,
                "max_joint_support_violation_exclusive": 0.95,
            },
            "summary": summary,
            "reasons": reasons,
            "publication_ready": False,
        }
        _atomic_json(Path(self.output_dir) / "coarse_mechanics_gate.json", gate)
        self.val_history[-1]["coarse_mechanics_gate"] = gate
        _atomic_json(Path(self.output_dir) / "metrics.json", self.val_history)
        if self.clearml is not None:
            self.clearml.report_single_value("coarse_mechanics_gate_passed", float(not reasons))

    def _require_full_state_recovery(self, global_step: int) -> None:
        """Fail closed unless the epoch state was committed before diagnostics."""
        if global_step % len(self.train_dataloader):
            raise ValueError("coarse diagnostics require an exact epoch boundary")
        epoch = global_step // len(self.train_dataloader)
        root = Path(self.output_dir) / self.config.recovery_checkpoint_name
        latest_path = root / "latest.json"
        if latest_path.is_symlink() or not latest_path.is_file():
            raise RuntimeError("coarse diagnostic requires a committed full-state recovery pointer")
        latest = json.loads(latest_path.read_text(encoding="utf-8"))
        state_dir = root / f"epoch_{epoch:04d}"
        state_path = state_dir / "resume.json"
        if (
            latest.get("state_dir") != state_dir.name
            or latest.get("resume_sha256") != self._sha256(state_path)
            or not (state_dir / "ema_state.pth").is_file()
            or not (state_dir / "accelerator").is_dir()
        ):
            raise RuntimeError("coarse diagnostic full-state recovery is incomplete or stale")

    @staticmethod
    def _mean_rmse_ratio(record: dict[str, float]) -> float:
        rmse_keys = sorted(key for key in record if key.endswith("_raw_mean_rmse"))
        ratios = []
        for key in rmse_keys:
            denominator = record[key.replace("_raw_mean_rmse", "_persistence_rmse")]
            if denominator <= 0:
                raise ValueError("persistence RMSE must be positive for learning-curve ratios")
            ratios.append(record[key] / denominator)
        if not ratios or not all(math.isfinite(value) for value in ratios):
            raise FloatingPointError("learning-curve RMSE ratios are empty or non-finite")
        return sum(ratios) / len(ratios)

    def _learning_curve_progress(self, epoch: int, global_step: int) -> dict[str, Any]:
        ratios = [self._mean_rmse_ratio(record) for record in self._coarse_diagnostic_history]
        latest = self._coarse_diagnostic_history[-1]
        roughness = max(
            latest[key] / max(latest[key.replace("_member_roughness", "_truth_roughness")], 1e-8)
            for key in latest
            if key.endswith("_member_roughness")
        )
        two_flat_intervals = len(ratios) >= 3 and ratios[-1] >= ratios[-2] and ratios[-2] >= ratios[-3]
        persistent_speckle = roughness >= 3.0
        stop = two_flat_intervals and persistent_speckle
        return {
            "status": "stopped" if stop else "continue",
            "decision": ("stop_no_learning_with_persistent_speckle" if stop else "continue_learning_curve"),
            "epoch": epoch + 1,
            "optimizer_updates": global_step,
            "mean_dimensionless_rmse_ratio_history": ratios,
            "latest_max_member_to_truth_roughness_ratio": roughness,
            "criteria": {
                "two_consecutive_nonimproving_rmse_ratio_intervals": True,
                "persistent_speckle_proxy_roughness_ratio_at_least": 3.0,
            },
            "reason": (
                "two consecutive diagnostics did not improve forecast RMSE while roughness remained high"
                if stop
                else "learning curve remains eligible to continue"
            ),
            "publication_ready": False,
        }

    @classmethod
    def _raw_support_magnitude_metrics(
        cls, raw_ensemble: torch.Tensor, valid: torch.Tensor
    ) -> dict[str, float]:
        expanded_valid = valid[:, None].expand_as(raw_ensemble)
        cls._require_finite_on_valid(raw_ensemble, expanded_valid, label="raw coarse ensemble")
        support = valid[:, None].expand_as(raw_ensemble[:, :, 0::2]) > 0
        sic = raw_ensemble[:, :, 0::2]
        sit = raw_ensemble[:, :, 1::2]
        sic_excess = torch.relu(-sic) + torch.relu(sic - 1.0)
        sit_excess = torch.relu(-sit)

        def summarize(prefix: str, excess: torch.Tensor) -> dict[str, float]:
            values = excess[support]
            positive = values[values > 0]
            return {
                f"{prefix}_raw_support_excess_mean_all": float(values.mean().item()),
                f"{prefix}_raw_support_excess_mean_violating": (
                    float(positive.mean().item()) if positive.numel() else 0.0
                ),
                f"{prefix}_raw_support_excess_p95_all": float(torch.quantile(values, 0.95).item()),
                f"{prefix}_raw_support_excess_max": float(values.max().item()),
            }

        return {**summarize("sic", sic_excess), **summarize("sit", sit_excess)}

    def save_model_custom(self, name: str = "last_model.pth"):
        """Atomically save raw and EMA weights used by coarse sampling."""
        os.makedirs(self.output_dir, exist_ok=True)
        model = self.accelerator.unwrap_model(self.model)
        for path, state in (
            (Path(self.output_dir) / name, model.state_dict()),
            (Path(self.output_dir) / f"ema_{name}", self.ema_model.state_dict()),
        ):
            temporary = path.with_name(f".{path.name}.{os.getpid()}.incomplete")
            torch.save(state, temporary)
            os.replace(temporary, path)

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _diagnostic_batch(self) -> dict[str, Any]:
        if self._coarse_diagnostic_batch is None:
            raw = self._coarse_diagnostic_batch_raw
            if raw is None:
                raw = next(iter(self.val_dataloader))
            raw_case_ids = raw.get("meta", {}).get("case_id", ())
            if isinstance(raw_case_ids, str):
                raw_case_ids = (raw_case_ids,)
            self._coarse_diagnostic_case_ids = tuple(str(value) for value in raw_case_ids)
            if not self._coarse_diagnostic_case_ids or len(set(self._coarse_diagnostic_case_ids)) != len(
                self._coarse_diagnostic_case_ids
            ):
                raise ValueError("coarse diagnostics require unique non-empty case identities")
            self._coarse_diagnostic_batch = self._batch_to_device(raw)
            if len(self._coarse_diagnostic_case_ids) != self._coarse_diagnostic_batch["truth"].shape[0]:
                raise ValueError("coarse diagnostic case identities do not match its tensor batch")
        return self._coarse_diagnostic_batch

    @staticmethod
    def _member_seed(case_id: str, member: int, stage: str = "coarse") -> int:
        if member < 0 or not case_id or not stage:
            raise ValueError("member seed requires a case identity, stage, and non-negative member")
        payload = f"cascade-v1|{stage}|{case_id}|member={member}".encode()
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)

    @staticmethod
    def _fraction_weighted_rmse(
        value: torch.Tensor,
        truth: torch.Tensor,
        ocean_fraction: torch.Tensor,
    ) -> float:
        if value.shape != truth.shape or value.shape[1] != 1:
            raise ValueError("fraction-weighted RMSE expects matching one-channel fields")
        result = torch.sqrt(ocean_fraction_weighted_mse(value, truth, ocean_fraction))
        if not torch.isfinite(result):
            raise FloatingPointError("fraction-weighted RMSE is non-finite")
        return float(result.item())

    @torch.no_grad()
    def _coarse_diagnostic(self, step: int, label: str, checkpoint_name: str) -> dict[str, float]:
        from .structured_trajectory_evaluation import make_structured_trajectory_figure

        batch = self._diagnostic_batch()
        condition = batch["structured_conditioning"]
        valid = batch["valid_mask"][:, :1]
        truth_normalized, active, fraction = coarse_target(batch["truth"], valid)
        persistence_normalized, _, _ = coarse_target(batch["background"], valid)
        case_count = truth_normalized.shape[0]
        raw_members = []
        raw_noises = []
        noise_seeds = []
        was_training = self.model.training
        with self._sampling_model() as sample_model:
            sample_model.eval()
            sampler = CoarseCascadeSampler(sample_model)
            for member in range(2):
                noises = []
                member_seeds = []
                # Case-local streams make results independent of batch/chunk order.
                for case_id in self._coarse_diagnostic_case_ids:
                    seed = self._member_seed(case_id, member)
                    generator = torch.Generator(device=self.accelerator.device)
                    generator.manual_seed(seed)
                    member_seeds.append(seed)
                    noises.append(
                        torch.randn(
                            (1, DIRECT_OUTPUT_CHANNELS, *self.config.image_size),
                            device=self.accelerator.device,
                            dtype=condition.dtype,
                            generator=generator,
                        )
                    )
                noise = torch.cat(noises, dim=0)
                raw_noises.append(noise.detach().cpu())
                noise_seeds.append(member_seeds)
                raw_members.append(
                    sampler.sample_conditioned(
                        structured_conditioning=condition,
                        valid_mask=valid,
                        initial_noise=noise,
                        num_timesteps=self.config.num_sample_timesteps,
                        device=self.accelerator.device,
                        method=self.config.sample_method,
                        rtol=self.config.sample_rtol,
                        atol=self.config.sample_atol,
                        end_time=0.0,
                    )
                )
        if was_training:
            self.model.train()
        normalized_ensemble = torch.stack(raw_members, dim=1)
        physical_ensemble = self._normalized_to_physical(normalized_ensemble.flatten(0, 1)).unflatten(
            0, (case_count, 2)
        )
        physical_truth = self._normalized_to_physical(truth_normalized)
        physical_persistence = self._normalized_to_physical(persistence_normalized)
        checkpoint_path = Path(self.output_dir) / checkpoint_name
        ema_path = Path(self.output_dir) / f"ema_{checkpoint_name}"
        if not checkpoint_path.is_file() or not ema_path.is_file():
            raise FileNotFoundError("coarse diagnostic requires durable raw and EMA checkpoints")
        payload = {
            "diagnostic_status": "raw_samples_saved_metrics_pending",
            "diagnostic_role": "generated_coarse_forecast_mechanics",
            "future_truth_used_for_condition": False,
            "forecast_claim_permitted": False,
            "optimizer_updates": step + 1,
            "validation_case_ids": list(self._coarse_diagnostic_case_ids),
            "coarse_noise_seeds_by_member_case": noise_seeds,
            "seed_rule": "sha256(cascade-v1|stage|case_id|member_id) first 63 bits",
            "raw_coarse_noise_normalized": torch.stack(raw_noises, dim=1),
            "raw_coarse_members_normalized": normalized_ensemble.detach().cpu(),
            "raw_coarse_members_physical": physical_ensemble.detach().cpu(),
            "coarse_truth_physical": physical_truth.detach().cpu(),
            "coarse_persistence_physical": physical_persistence.detach().cpu(),
            "coarse_active_mask": active.detach().cpu(),
            "coarse_ocean_fraction": fraction.detach().cpu(),
            "checkpoint": checkpoint_name,
            "checkpoint_sha256": self._sha256(checkpoint_path),
            "ema_checkpoint": ema_path.name,
            "ema_checkpoint_sha256": self._sha256(ema_path),
            "sampling_weight_role": self._sampling_weight_label(),
            "metrics": {},
        }
        diagnostics = Path(self.output_dir) / "coarse_diagnostics"
        diagnostics.mkdir(parents=True, exist_ok=True)
        payload_path = diagnostics / f"{label}_samples.pt"
        self._atomic_save_direct_payload(payload, payload_path)

        metrics: dict[str, float] = {
            "step": float(step + 1),
            "diagnostic_case_count": float(case_count),
            **self._raw_support_metrics(physical_ensemble, active),
            **self._raw_support_magnitude_metrics(physical_ensemble, active),
        }
        mean = physical_ensemble.mean(dim=1)
        for lead_index, lead in enumerate((3, 6, 9)):
            for offset, field in enumerate(("sic", "sit")):
                channel = 2 * lead_index + offset
                key = f"d{lead}_{field}"
                metrics[f"{key}_raw_mean_rmse"] = self._fraction_weighted_rmse(
                    mean[:, channel : channel + 1],
                    physical_truth[:, channel : channel + 1],
                    fraction,
                )
                metrics[f"{key}_persistence_rmse"] = self._fraction_weighted_rmse(
                    physical_persistence[:, channel : channel + 1],
                    physical_truth[:, channel : channel + 1],
                    fraction,
                )
                metrics[f"{key}_member_roughness"] = self._roughness(
                    physical_ensemble[:, 0, channel : channel + 1], active
                )
                metrics[f"{key}_truth_roughness"] = self._roughness(
                    physical_truth[:, channel : channel + 1], active
                )
        nonfinite_metrics = [key for key, value in metrics.items() if not math.isfinite(value)]
        if nonfinite_metrics:
            raise FloatingPointError(f"coarse diagnostic produced non-finite metrics: {nonfinite_metrics}")
        payload["diagnostic_status"] = "complete_before_plot"
        payload["metrics"] = metrics
        self._atomic_save_direct_payload(payload, payload_path)
        for case_index in range(case_count):
            figure = make_structured_trajectory_figure(
                physical_truth[case_index],
                physical_persistence[case_index],
                physical_ensemble[case_index, 0],
                active[case_index],
                title=f"coarse dynamics {label}: raw member 0; {payload['validation_case_ids'][case_index]}",
                origin="upper",
                lead_days=(3, 6, 9),
            )
            image_path = diagnostics / f"{label}_case{case_index:02d}_raw_member0.png"
            figure.savefig(image_path, dpi=160)
            import matplotlib.pyplot as plt

            plt.close(figure)
            if self.clearml is not None:
                self.clearml.report_image(
                    "coarse_dynamics/samples", f"case{case_index:02d}_raw_member0", image_path, step + 1
                )
        _atomic_json(diagnostics / f"{label}_metrics.json", metrics)
        if self.clearml is not None:
            for key, value in metrics.items():
                if key != "step" and math.isfinite(value):
                    self.clearml.report_scalar("coarse_dynamics/physical", key, value, step + 1)
        self._coarse_diagnostic_history.append(dict(metrics))
        return metrics


def coarse_model_input_for_test(
    state: torch.Tensor,
    structured_conditioning: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    condition, _, _ = lossless_coarse_condition(structured_conditioning, valid_mask)
    grid = make_normalized_xy_grid(*state.shape[-2:], device=state.device, dtype=state.dtype).expand(
        state.shape[0], -1, -1, -1
    )
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


def load_coarse_cascade_sampler(
    run_dir: str,
    checkpoint_name: str,
    model_config: dict[str, Any],
    expected_checkpoint_sha256: str,
    expected_code_commit: str,
    device=None,
) -> CoarseCascadeSampler:
    root = Path(run_dir)
    manifest_path = root / "coarse_cascade_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("coarse cascade checkpoint is missing its sampler manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "sampler": "CoarseCascadeSampler",
        "condition_encoding": "lossless_2x2_space_to_depth_v1",
        "condition_channels": COARSE_CONDITION_CHANNELS,
        "model_input_channels": COARSE_INPUT_CHANNELS,
        "model_output_channels": DIRECT_OUTPUT_CHANNELS,
        "coarse_factor": CASCADE_FACTOR,
        "image_size": [160, 128],
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("coarse cascade sampler manifest is incompatible")
    current_module_sha256 = CoarseCascadeDynamicsTrainer._sha256(Path(__file__))
    if manifest.get("conditioning_module_sha256") != current_module_sha256:
        raise ValueError("coarse conditioning implementation differs from the checkpoint manifest")
    if manifest.get("code_commit") != expected_code_commit:
        raise ValueError("coarse cascade code commit differs from the required identity")
    config = TrainingConfig.from_dict(model_config)
    if (
        config.in_channels != COARSE_INPUT_CHANNELS
        or config.out_channels != DIRECT_OUTPUT_CHANNELS
        or tuple(config.image_size) != (160, 128)
    ):
        raise ValueError("coarse cascade checkpoint requires an exact 160x128 56-to-6 UNet")
    if manifest.get("architecture") != _architecture_signature(config):
        raise ValueError("coarse cascade architecture differs from its sampler manifest")
    model = build_unet(config)
    resolved = resolve_checkpoint_name(run_dir, checkpoint_name)
    if checkpoint_name == "auto":
        raise ValueError("coarse cascade production reload requires an explicit checkpoint")
    checkpoint_path = root / resolved
    actual_sha256 = CoarseCascadeDynamicsTrainer._sha256(checkpoint_path)
    if actual_sha256 != expected_checkpoint_sha256:
        raise ValueError("coarse cascade checkpoint SHA256 differs from the required identity")
    try:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "shadow_params" in state:
        ema = EMAModel(model.parameters())
        ema.load_state_dict(state)
        ema.copy_to(model.parameters())
    else:
        model.load_state_dict(state)
    model.eval()
    if device is not None:
        model.to(device)
    return CoarseCascadeSampler(model)


def write_coarse_manifest(
    output_dir: str | Path,
    config: TrainingConfig,
    code_identity: dict[str, str] | None,
) -> None:
    commit = None if code_identity is None else code_identity.get("git_commit")
    if commit is None or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("coarse manifest requires an exact immutable git commit")
    _atomic_json(
        Path(output_dir) / "coarse_cascade_manifest.json",
        {
            "schema_version": 1,
            "sampler": "CoarseCascadeSampler",
            "target": "C=D(Y)",
            "conditional_law": "p(C|c_full) via lossless fixed encoding of c_full",
            "condition_encoding": "lossless_2x2_space_to_depth_v1",
            "condition_channels": COARSE_CONDITION_CHANNELS,
            "model_input_channels": COARSE_INPUT_CHANNELS,
            "model_output_channels": DIRECT_OUTPUT_CHANNELS,
            "coarse_factor": CASCADE_FACTOR,
            "image_size": [160, 128],
            "architecture": _architecture_signature(config),
            "code_commit": commit,
            "conditioning_module_sha256": CoarseCascadeDynamicsTrainer._sha256(Path(__file__)),
        },
    )
