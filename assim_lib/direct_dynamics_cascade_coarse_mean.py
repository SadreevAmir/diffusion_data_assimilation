"""Deterministic conditional-mean control for coarse SIC/SIT dynamics.

The model predicts the normalized persistence increment
``Delta(c) = E[C - P(c) | c]`` from causal conditioning only.  There is no
random state input and no ODE solve.  This isolates conditional-mean mechanics
before fitting a separate innovation law.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch

from .direct_dynamics_cascade_coarse import (
    COARSE_CONDITION_CHANNELS,
    CoarseCascadeDynamicsTrainer,
    coarse_target,
    lossless_coarse_condition,
)
from .direct_dynamics_cascade_coarse_residual import coarse_persistence_from_condition
from .direct_dynamics_training import DIRECT_LEADS, DIRECT_OUTPUT_CHANNELS
from .runtime import make_normalized_xy_grid
from .trainer import _atomic_json
COARSE_MEAN_INPUT_CHANNELS = 2 + COARSE_CONDITION_CHANNELS


def coarse_mean_target(
    truth: torch.Tensor,
    structured_conditioning: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return Delta=C-P, P, C and fine-ocean fractions in normalized space."""
    clean, active, fraction = coarse_target(truth, valid_mask)
    persistence, persistence_active, persistence_fraction = (
        coarse_persistence_from_condition(structured_conditioning, valid_mask)
    )
    if not torch.equal(active, persistence_active) or not torch.equal(
        fraction, persistence_fraction
    ):
        raise ValueError("mean target and persistence supports differ")
    support = active.expand_as(clean) > 0
    delta = torch.where(support, clean - persistence, torch.zeros_like(clean))
    if not torch.isfinite(delta[support]).all():
        raise FloatingPointError("deterministic mean target contains NaN/Inf")
    return delta, persistence, clean, fraction


def coarse_mean_model_input(
    structured_conditioning: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode only causal condition and fixed coordinates; no state/noise input."""
    condition, active, fraction = lossless_coarse_condition(
        structured_conditioning, valid_mask
    )
    grid = make_normalized_xy_grid(
        *condition.shape[-2:], device=condition.device, dtype=condition.dtype
    ).expand(condition.shape[0], -1, -1, -1)
    model_input = torch.cat((grid, condition), dim=1)
    if model_input.shape[1] != COARSE_MEAN_INPUT_CHANNELS:
        raise RuntimeError("deterministic mean model input has the wrong channel count")
    return model_input, active, fraction


class CoarseConditionalMeanPredictor:
    """One-pass conditional mean ``P(c) + Delta_theta(c)``."""

    def __init__(self, model):
        self.model = model

    @torch.no_grad()
    def predict_conditioned(
        self,
        *,
        structured_conditioning: torch.Tensor,
        valid_mask: torch.Tensor,
        device=None,
    ) -> torch.Tensor:
        model_input, active, _ = coarse_mean_model_input(
            structured_conditioning, valid_mask
        )
        persistence, persistence_active, _ = coarse_persistence_from_condition(
            structured_conditioning, valid_mask
        )
        if not torch.equal(active, persistence_active):
            raise ValueError("mean predictor and persistence supports differ")
        device = device or model_input.device
        model_input = model_input.to(device=device, dtype=torch.float32)
        timestep = torch.zeros(model_input.shape[0], device=device, dtype=torch.float32)
        delta = self.model(model_input, timestep, return_dict=False)[0].float()
        if tuple(delta.shape) != tuple(persistence.shape):
            raise ValueError(
                "deterministic mean delta shape differs from causal persistence"
            )
        support = active.to(device=device).expand_as(delta) > 0
        if not torch.isfinite(delta[support]).all():
            raise FloatingPointError("deterministic mean prediction contains NaN/Inf")
        forecast = torch.where(
            support,
            persistence.to(device=device, dtype=torch.float32) + delta,
            torch.zeros_like(delta),
        )
        if not torch.isfinite(forecast[support]).all():
            raise FloatingPointError(
                "deterministic mean reconstruction contains NaN/Inf"
            )
        return forecast


class CoarseConditionalMeanTrainer(CoarseCascadeDynamicsTrainer):
    """Fit the fraction-weighted conditional mean before any innovation model."""

    required_training_objective = "deterministic_mean"
    required_timestep_sampler = "constant_zero"
    required_input_channels = COARSE_MEAN_INPUT_CHANNELS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self._planned_updates != 2048:
            raise ValueError("conditional mean control requires exactly 2048 updates")
        if self.accelerator.is_main_process:
            self._write_mean_contract()

    def _write_mean_contract(self) -> None:
        manifest_path = Path(self.output_dir) / "coarse_cascade_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update(
            {
                "sampler": "CoarseConditionalMeanPredictor",
                "target": "Delta=C-P",
                "reconstruction": "m(c)=P(c)+Delta_theta(c)",
                "conditional_law": "conditional mean only; no distribution claim",
                "training_objective": (
                    "case-equal ocean-fraction-weighted MSE in normalized six-channel space"
                ),
                "persistence_source": (
                    "normalized d0 SIC/SIT condition only; mask-aware D"
                ),
                "stochastic_state_input": False,
                "ode_sampling": False,
                "model_input_channels": COARSE_MEAN_INPUT_CHANNELS,
                "mean_module_sha256": self._sha256(Path(__file__)),
                "clipping_or_postprocessing": False,
            }
        )
        _atomic_json(manifest_path, manifest)
        metadata_path = Path(self.output_dir) / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata.update(
            {
                "checkpoint_selection_rule": (
                    "minimum raw-weight validation case-equal ocean-fraction-weighted "
                    "normalized MSE"
                ),
                "deterministic_mean_semantics": "m(c)=P(c)+E[C-P(c)|c]",
                "stochastic_state_input": False,
                "ode_sampling": False,
            }
        )
        _atomic_json(metadata_path, metadata)

    def _make_training_pair(
        self,
        truth: torch.Tensor,
        batch: dict[str, torch.Tensor],
        timesteps: torch.Tensor,
        residual_background: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del residual_background, generator
        if torch.any(timesteps != 0):
            raise ValueError("conditional mean requires an identically zero timestep")
        delta, persistence, _, _ = coarse_mean_target(
            truth,
            batch["structured_conditioning"],
            batch["valid_mask"][:, :1],
        )
        background, _, _ = coarse_target(
            batch["background"], batch["valid_mask"][:, :1]
        )
        if not torch.equal(persistence, background):
            raise ValueError("causal d0 persistence differs from dataset persistence baseline")
        return torch.zeros_like(delta), delta

    def _make_model_input(
        self,
        noisy_truth: torch.Tensor,
        batch: dict[str, torch.Tensor],
        **_: object,
    ) -> torch.Tensor:
        del noisy_truth
        model_input, _, _ = coarse_mean_model_input(
            batch["structured_conditioning"], batch["valid_mask"][:, :1]
        )
        return model_input

    def _make_sampler(self, model) -> CoarseConditionalMeanPredictor:
        return CoarseConditionalMeanPredictor(model)

    def _after_training_epoch(self, epoch: int, global_step: int) -> None:
        """Save fixed mean diagnostics without inheriting flow-specific rejection rules."""
        if not self.accelerator.is_main_process:
            return
        expected_steps = [512, 1024, 1536, 2048]
        if global_step in expected_steps:
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
                f"conditional-mean review requires diagnostics {expected_so_far}, "
                f"got {actual_steps}"
            )
        progress = self._mean_review_record(epoch, global_step)
        _atomic_json(Path(self.output_dir) / "coarse_mean_progress.json", progress)
        if global_step < self._planned_updates:
            return
        if global_step != self._planned_updates:
            raise RuntimeError(
                f"conditional-mean terminal review expected step {self._planned_updates}, "
                f"got {global_step}"
            )
        # The generic runner consumes coarse_mechanics_gate.json.  This is an
        # explicit hold, not an automatic scientific acceptance or rejection.
        _atomic_json(Path(self.output_dir) / "coarse_mean_gate.json", progress)
        _atomic_json(Path(self.output_dir) / "coarse_mechanics_gate.json", progress)
        self.val_history[-1]["coarse_mean_gate"] = progress
        _atomic_json(Path(self.output_dir) / "metrics.json", self.val_history)
        if self.clearml is not None:
            self.clearml.report_single_value("coarse_mean_requires_frozen_review", 1.0)

    def _mean_review_record(self, epoch: int, global_step: int) -> dict[str, Any]:
        final = self._coarse_diagnostic_history[-1]
        first = self._coarse_diagnostic_history[0]
        rmse_keys = sorted(key for key in final if key.endswith("_raw_mean_rmse"))
        if len(rmse_keys) != DIRECT_OUTPUT_CHANNELS:
            raise ValueError("conditional-mean review requires all six output RMSE values")
        persistence_keys = [
            key.replace("_raw_mean_rmse", "_persistence_rmse") for key in rmse_keys
        ]
        roughness_keys = sorted(
            key for key in final if key.endswith("_member_roughness")
        )
        if (
            any(first[key] <= 0 for key in rmse_keys)
            or any(final[key] <= 0 for key in persistence_keys)
            or len(roughness_keys) != DIRECT_OUTPUT_CHANNELS
        ):
            raise ValueError("conditional-mean review has invalid ratio denominators")
        skill = {
            key.removesuffix("_raw_mean_rmse"): 1.0
            - final[key] / final[key.replace("_raw_mean_rmse", "_persistence_rmse")]
            for key in rmse_keys
        }
        learning = {
            key.removesuffix("_raw_mean_rmse"): 1.0 - final[key] / first[key]
            for key in rmse_keys
        }
        roughness = {
            key.removesuffix("_member_roughness"): final[key]
            / max(final[key.replace("_member_roughness", "_truth_roughness")], 1e-8)
            for key in roughness_keys
        }
        summary = {
            "rmse_skill_over_persistence_by_output": skill,
            "rmse_change_from_first_diagnostic_by_output": learning,
            "member_to_truth_roughness_ratio_by_output": roughness,
            "mean_dimensionless_rmse_skill_over_persistence": sum(skill.values())
            / len(skill),
            "minimum_output_rmse_skill_over_persistence": min(skill.values()),
            "maximum_member_to_truth_roughness_ratio": max(roughness.values()),
            "joint_raw_support_violation_fraction": final[
                "joint_raw_support_violation_fraction"
            ],
        }
        flattened = [
            *skill.values(),
            *learning.values(),
            *roughness.values(),
            *(
                value
                for value in summary.values()
                if isinstance(value, (float, int))
            ),
        ]
        if not all(math.isfinite(float(value)) for value in flattened):
            raise FloatingPointError("conditional-mean review produced non-finite ratios")
        terminal = global_step == self._planned_updates
        return {
            "schema_version": "coarse_deterministic_mean_review_v1",
            "status": (
                "pending_independent_frozen_review" if terminal else "training"
            ),
            "decision": (
                "hold_for_frozen_mean_review" if terminal else "continue_fixed_budget"
            ),
            "epoch": epoch + 1,
            "optimizer_updates": global_step,
            "descriptive_only": True,
            "automatic_acceptance_or_rejection": False,
            "innovations_stage_permitted": False,
            "summary": summary,
            "required_frozen_review": [
                "all-six-output skill against causal persistence",
                "raw spatial coherence on predeclared seasonal cases",
                "raw physical-support excess magnitude",
                "exact-zero and low-ice SIT false-mass diagnostics",
            ],
            "publication_ready": False,
        }

    @torch.no_grad()
    def _coarse_diagnostic(
        self, step: int, label: str, checkpoint_name: str
    ) -> dict[str, float]:
        from .structured_trajectory_evaluation import make_structured_trajectory_figure

        batch = self._diagnostic_batch()
        condition = batch["structured_conditioning"]
        valid = batch["valid_mask"][:, :1]
        _, persistence_normalized, truth_normalized, fraction = coarse_mean_target(
            batch["truth"], condition, valid
        )
        active = (fraction > 0).to(dtype=truth_normalized.dtype)
        was_training = self.model.training
        with self._sampling_model() as model:
            model.eval()
            forecast_normalized = CoarseConditionalMeanPredictor(model).predict_conditioned(
                structured_conditioning=condition,
                valid_mask=valid,
                device=self.accelerator.device,
            )
        if was_training:
            self.model.train()
        forecast_physical = self._normalized_to_physical(forecast_normalized)
        truth_physical = self._normalized_to_physical(truth_normalized)
        persistence_physical = self._normalized_to_physical(persistence_normalized)
        checkpoint_path = Path(self.output_dir) / checkpoint_name
        ema_path = Path(self.output_dir) / f"ema_{checkpoint_name}"
        if not checkpoint_path.is_file() or not ema_path.is_file():
            raise FileNotFoundError("mean diagnostic requires durable raw and EMA checkpoints")
        payload = {
            "diagnostic_status": "raw_samples_saved_metrics_pending",
            "diagnostic_role": "deterministic_conditional_mean_mechanics",
            "future_truth_used_for_condition": False,
            "distribution_claim_permitted": False,
            "optimizer_updates": step + 1,
            "validation_case_ids": list(self._coarse_diagnostic_case_ids),
            "structured_conditioning": condition.detach().cpu(),
            "fine_valid_mask": batch["valid_mask"].detach().cpu(),
            "coarse_mean_normalized": forecast_normalized.detach().cpu(),
            "coarse_mean_physical": forecast_physical.detach().cpu(),
            "coarse_truth_physical": truth_physical.detach().cpu(),
            "coarse_persistence_physical": persistence_physical.detach().cpu(),
            "coarse_active_mask": active.detach().cpu(),
            "coarse_ocean_fraction": fraction.detach().cpu(),
            "checkpoint": checkpoint_name,
            "checkpoint_sha256": self._sha256(checkpoint_path),
            "ema_checkpoint": ema_path.name,
            "ema_checkpoint_sha256": self._sha256(ema_path),
            "metrics": {},
        }
        diagnostics = Path(self.output_dir) / "coarse_mean_diagnostics"
        diagnostics.mkdir(parents=True, exist_ok=True)
        payload_path = diagnostics / f"{label}_mean.pt"
        self._atomic_save_direct_payload(payload, payload_path)
        ensemble_view = forecast_physical[:, None]
        metrics: dict[str, float] = {
            "step": float(step + 1),
            "diagnostic_case_count": float(forecast_physical.shape[0]),
            **self._raw_support_metrics(ensemble_view, active),
            **self._raw_support_magnitude_metrics(ensemble_view, active),
        }
        for lead_index, lead in enumerate(DIRECT_LEADS):
            for offset, field in enumerate(("sic", "sit")):
                channel = 2 * lead_index + offset
                key = f"d{lead}_{field}"
                forecast = forecast_physical[:, channel : channel + 1]
                truth = truth_physical[:, channel : channel + 1]
                persistence = persistence_physical[:, channel : channel + 1]
                metrics[f"{key}_raw_mean_rmse"] = self._fraction_weighted_rmse(
                    forecast, truth, fraction
                )
                metrics[f"{key}_persistence_rmse"] = self._fraction_weighted_rmse(
                    persistence, truth, fraction
                )
                metrics[f"{key}_member_roughness"] = self._roughness(forecast, active)
                metrics[f"{key}_truth_roughness"] = self._roughness(truth, active)
        if not all(math.isfinite(value) for value in metrics.values()):
            raise FloatingPointError("conditional mean diagnostic produced non-finite metrics")
        payload["metrics"] = metrics
        payload["diagnostic_status"] = "complete_before_plot"
        self._atomic_save_direct_payload(payload, payload_path)
        for case_index, case_id in enumerate(self._coarse_diagnostic_case_ids):
            figure = make_structured_trajectory_figure(
                truth_physical[case_index],
                persistence_physical[case_index],
                forecast_physical[case_index],
                active[case_index],
                title=(
                    f"coarse deterministic mean {label}; {case_id}; "
                    "truth / persistence / mean"
                ),
                origin="upper",
                lead_days=DIRECT_LEADS,
            )
            image_path = diagnostics / f"{label}_case{case_index:02d}_mean.png"
            figure.savefig(image_path, dpi=160)
            import matplotlib.pyplot as plt

            plt.close(figure)
            if self.clearml is not None:
                self.clearml.report_image(
                    "coarse_mean/raw_samples",
                    f"case{case_index:02d}",
                    image_path,
                    step + 1,
                )
        payload["diagnostic_status"] = "complete"
        self._atomic_save_direct_payload(payload, payload_path)
        _atomic_json(diagnostics / f"{label}_metrics.json", metrics)
        if self.clearml is not None:
            for key, value in metrics.items():
                if key != "step":
                    self.clearml.report_scalar(
                        "coarse_mean/physical", key, value, step + 1
                    )
        self._coarse_diagnostic_history.append(dict(metrics))
        return metrics
