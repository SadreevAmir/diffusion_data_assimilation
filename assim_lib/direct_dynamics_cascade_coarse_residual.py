"""Persistence-centred coarse dynamics flow with an unchanged conditional law.

The absolute coarse target is ``C=D(Y)`` and the causal persistence state is
``P=D(Y_d0 repeated at d+3,d+6,d+9)``.  This module trains the invertible
coordinate change ``R=C-P`` and reconstructs every raw sample as ``P+R``.
No residual re-scaling, clipping, smoothing, support head, or post-processing
is applied.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import torch

from .direct_dynamics_cascade_coarse import (
    COARSE_INPUT_CHANNELS,
    CoarseCascadeDynamicsTrainer,
    coarse_target,
    lossless_coarse_condition,
)
from .direct_dynamics_training import DIRECT_LEADS, DIRECT_OUTPUT_CHANNELS
from .runtime import make_normalized_xy_grid
from .sampler import Sampler
from .trainer import _atomic_json
from .transforms import channel_denormalize


def coarse_persistence_from_condition(
    structured_conditioning: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build P only from the available normalized d0 SIC/SIT condition."""
    if structured_conditioning.ndim != 4 or structured_conditioning.shape[1] != 15:
        raise ValueError("persistence construction requires the exact 15-channel condition")
    d0 = structured_conditioning[:, :2]
    repeated = d0.repeat(1, len(DIRECT_LEADS), 1, 1)
    return coarse_target(repeated, valid_mask)


def coarse_residual_flow_pair(
    truth: torch.Tensor,
    noise: torch.Tensor,
    structured_conditioning: torch.Tensor,
    valid_mask: torch.Tensor,
    time: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return r_t, epsilon-R, R, P, and ocean fraction on active ocean."""
    clean, active, fraction = coarse_target(truth, valid_mask)
    persistence, persistence_active, persistence_fraction = coarse_persistence_from_condition(
        structured_conditioning, valid_mask
    )
    if not torch.equal(active, persistence_active) or not torch.equal(fraction, persistence_fraction):
        raise ValueError("target and persistence coarse supports differ")
    if noise.shape != clean.shape:
        raise ValueError("residual noise and coarse target shapes differ")
    if time.ndim != 1 or time.shape[0] != clean.shape[0]:
        raise ValueError("time must contain one scalar per case")
    if not torch.isfinite(time).all() or torch.any((time < 0) | (time > 1)):
        raise ValueError("flow time must lie in [0,1]")
    support = active.expand_as(clean) > 0
    if not torch.isfinite(noise[support]).all():
        raise FloatingPointError("residual noise contains NaN/Inf on active ocean")
    residual = torch.where(support, clean - persistence, torch.zeros_like(clean))
    canonical_noise = torch.where(support, noise.float(), torch.zeros_like(clean))
    view = time.to(device=clean.device, dtype=clean.dtype).reshape(-1, 1, 1, 1)
    state = (1.0 - view) * residual + view * canonical_noise
    velocity = canonical_noise - residual
    if not torch.isfinite(state[support]).all() or not torch.isfinite(
        velocity[support]
    ).all():
        raise FloatingPointError("residual flow pair produced NaN/Inf on active ocean")
    return state, velocity, residual, persistence, fraction


def _residual_affine_tensors(
    statistics: Mapping[str, Any], reference: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    means = torch.as_tensor(statistics.get("means"), dtype=reference.dtype, device=reference.device)
    stds = torch.as_tensor(statistics.get("stds"), dtype=reference.dtype, device=reference.device)
    if means.shape != (DIRECT_OUTPUT_CHANNELS,) or stds.shape != (DIRECT_OUTPUT_CHANNELS,):
        raise ValueError("residual statistics require exactly six means and standard deviations")
    if not torch.isfinite(means).all() or not torch.isfinite(stds).all() or torch.any(stds <= 0):
        raise ValueError("residual statistics must be finite with strictly positive deviations")
    shape = (1, DIRECT_OUTPUT_CHANNELS, 1, 1)
    return means.reshape(shape), stds.reshape(shape)


def standardize_coarse_residual(
    residual: torch.Tensor,
    active: torch.Tensor,
    statistics: Mapping[str, Any],
) -> torch.Tensor:
    """Apply the fixed invertible train-only diagonal residual transform."""
    means, stds = _residual_affine_tensors(statistics, residual)
    support = active.expand_as(residual) > 0
    standardized = (residual - means) / stds
    if not torch.isfinite(standardized[support]).all():
        raise FloatingPointError("standardized residual contains NaN/Inf on active ocean")
    return torch.where(support, standardized, torch.zeros_like(standardized))


def unstandardize_coarse_residual(
    standardized: torch.Tensor,
    active: torch.Tensor,
    statistics: Mapping[str, Any],
) -> torch.Tensor:
    """Invert the fixed diagonal transform without clipping or smoothing."""
    means, stds = _residual_affine_tensors(statistics, standardized)
    support = active.expand_as(standardized) > 0
    residual = means + stds * standardized
    if not torch.isfinite(residual[support]).all():
        raise FloatingPointError("unstandardized residual contains NaN/Inf on active ocean")
    return torch.where(support, residual, torch.zeros_like(residual))


def coarse_standardized_residual_flow_pair(
    truth: torch.Tensor,
    noise: torch.Tensor,
    structured_conditioning: torch.Tensor,
    valid_mask: torch.Tensor,
    time: torch.Tensor,
    statistics: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return w_t, epsilon-W, raw R, persistence P and ocean fraction."""
    if time.ndim != 1 or time.shape[0] != truth.shape[0]:
        raise ValueError("time must contain one scalar per case")
    if not torch.isfinite(time).all() or torch.any((time < 0) | (time > 1)):
        raise ValueError("flow time must lie in [0,1]")
    _, _, residual, persistence, fraction = coarse_residual_flow_pair(
        truth,
        torch.zeros_like(noise),
        structured_conditioning,
        valid_mask,
        time,
    )
    active = (fraction > 0).to(dtype=residual.dtype)
    whitened = standardize_coarse_residual(residual, active, statistics)
    support = active.expand_as(whitened) > 0
    canonical_noise = torch.where(support, noise.float(), torch.zeros_like(whitened))
    if not torch.isfinite(canonical_noise[support]).all():
        raise FloatingPointError("whitened residual noise contains NaN/Inf on active ocean")
    view = time.to(device=whitened.device, dtype=whitened.dtype).reshape(-1, 1, 1, 1)
    state = (1.0 - view) * whitened + view * canonical_noise
    velocity = canonical_noise - whitened
    if not torch.isfinite(state[support]).all() or not torch.isfinite(velocity[support]).all():
        raise FloatingPointError("standardized residual flow pair produced NaN/Inf on active ocean")
    return state, velocity, residual, persistence, fraction


def residual_model_input_for_test(
    residual_state: torch.Tensor,
    structured_conditioning: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    condition, _, _ = lossless_coarse_condition(structured_conditioning, valid_mask)
    grid = make_normalized_xy_grid(
        *residual_state.shape[-2:], device=residual_state.device, dtype=residual_state.dtype
    ).expand(residual_state.shape[0], -1, -1, -1)
    return torch.cat((residual_state, grid, condition), dim=1)


class CoarsePersistenceResidualSampler:
    """Sample R and reconstruct C=P+R without altering the raw endpoint."""

    def __init__(self, model):
        self.sampler = Sampler(model)
        self.last_residual: torch.Tensor | None = None

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
        persistence, persistence_active, _ = coarse_persistence_from_condition(
            structured_conditioning, valid_mask
        )
        if not torch.equal(active, persistence_active):
            raise ValueError("encoded condition and persistence supports differ")
        expected = (condition.shape[0], DIRECT_OUTPUT_CHANNELS, *condition.shape[-2:])
        if tuple(initial_noise.shape) != expected:
            raise ValueError(f"coarse residual noise must have shape {expected}")
        support = active.expand_as(initial_noise) > 0
        raw_noise = initial_noise.to(device=active.device, dtype=torch.float32)
        if not torch.isfinite(raw_noise[support]).all():
            raise FloatingPointError("coarse residual noise contains NaN/Inf on active ocean")
        noise = torch.where(support, raw_noise, torch.zeros_like(raw_noise))
        zeros = torch.zeros(expected, dtype=torch.float32, device=condition.device)
        empty_obs = torch.zeros((expected[0], 2, *expected[-2:]), dtype=torch.float32, device=zeros.device)
        residual = self.sampler.sample_conditioned(
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
            initial_noise=noise,
            sample_target="state",
            model_conditioning=condition.float(),
            state_channels=DIRECT_OUTPUT_CHANNELS,
            end_time=end_time,
            valid_mask=active,
            state_mask=active.expand(-1, DIRECT_OUTPUT_CHANNELS, -1, -1),
        )
        residual = torch.where(support, residual.float(), torch.zeros_like(residual.float()))
        reconstructed = torch.where(
            support,
            persistence.to(device=residual.device, dtype=torch.float32) + residual,
            torch.zeros_like(residual),
        )
        if not torch.isfinite(reconstructed[support]).all():
            raise FloatingPointError("coarse residual reconstruction produced NaN/Inf")
        self.last_residual = residual
        return reconstructed


class CoarseStandardizedPersistenceResidualSampler:
    """Sample standardized residuals and exactly invert the train-only transform."""

    def __init__(self, model, statistics: Mapping[str, Any]):
        self.sampler = Sampler(model)
        self.statistics = dict(statistics)
        self.last_residual: torch.Tensor | None = None

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
        persistence, persistence_active, _ = coarse_persistence_from_condition(
            structured_conditioning, valid_mask
        )
        if not torch.equal(active, persistence_active):
            raise ValueError("encoded condition and persistence supports differ")
        expected = (condition.shape[0], DIRECT_OUTPUT_CHANNELS, *condition.shape[-2:])
        if tuple(initial_noise.shape) != expected:
            raise ValueError(f"standardized residual noise must have shape {expected}")
        support = active.expand_as(initial_noise) > 0
        raw_noise = initial_noise.to(device=active.device, dtype=torch.float32)
        if not torch.isfinite(raw_noise[support]).all():
            raise FloatingPointError("standardized residual noise contains NaN/Inf")
        noise = torch.where(support, raw_noise, torch.zeros_like(raw_noise))
        zeros = torch.zeros(expected, dtype=torch.float32, device=condition.device)
        empty_obs = torch.zeros((expected[0], 2, *expected[-2:]), dtype=torch.float32, device=zeros.device)
        whitened = self.sampler.sample_conditioned(
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
            initial_noise=noise,
            sample_target="state",
            model_conditioning=condition.float(),
            state_channels=DIRECT_OUTPUT_CHANNELS,
            end_time=end_time,
            valid_mask=active,
            state_mask=active.expand(-1, DIRECT_OUTPUT_CHANNELS, -1, -1),
        )
        residual = unstandardize_coarse_residual(whitened.float(), active, self.statistics)
        reconstructed = torch.where(
            support,
            persistence.to(device=residual.device, dtype=torch.float32) + residual,
            torch.zeros_like(residual),
        )
        if not torch.isfinite(reconstructed[support]).all():
            raise FloatingPointError("standardized residual reconstruction produced NaN/Inf")
        self.last_residual = residual
        return reconstructed


def residual_mechanics_gate(
    final: dict[str, float],
    baseline: dict[str, float],
) -> dict[str, Any]:
    """Compare the bounded residual pilot with the paired absolute-state run."""
    output_keys = [
        f"d{lead}_{field}_raw_mean_rmse"
        for lead in DIRECT_LEADS
        for field in ("sic", "sit")
    ]
    if any(key not in final or key not in baseline for key in output_keys):
        raise ValueError("residual gate cannot align six outputs with its baseline")
    persistence_keys = [key.replace("_raw_mean_rmse", "_persistence_rmse") for key in output_keys]
    required_numeric = [*output_keys, *persistence_keys]
    if any(key not in final for key in persistence_keys):
        raise ValueError("residual gate lacks persistence RMSE denominators")
    if any(
        not math.isfinite(final[key])
        or (key in baseline and not math.isfinite(baseline[key]))
        for key in required_numeric
    ):
        raise FloatingPointError("residual gate received non-finite RMSE inputs")
    if any(final[key] <= 0 or baseline[key] <= 0 for key in output_keys) or any(
        final[key] <= 0 for key in persistence_keys
    ):
        raise ValueError("residual gate RMSE denominators must be positive")
    ratios = {key.removesuffix("_raw_mean_rmse"): final[key] / baseline[key] for key in output_keys}
    skills = {
        key.removesuffix("_raw_mean_rmse"): 1.0
        - final[key] / final[key.replace("_raw_mean_rmse", "_persistence_rmse")]
        for key in output_keys
    }
    tail_keys = tuple(
        f"{field}_{suffix}"
        for field in ("sic", "sit")
        for suffix in (
            "raw_support_excess_mean_all",
            "raw_support_excess_p95_all",
            "raw_support_excess_max",
            "raw_support_violation_fraction",
        )
    )
    if any(key not in final or key not in baseline for key in tail_keys):
        raise ValueError("residual gate lacks paired support-tail baselines")
    if any(
        not math.isfinite(final[key])
        or not math.isfinite(baseline[key])
        or baseline[key] <= 0
        for key in tail_keys
    ):
        raise ValueError("residual gate support-tail baselines must be finite and positive")
    tail_ratios = {key: final[key] / baseline[key] for key in tail_keys}
    mean_ratio = sum(ratios.values()) / len(ratios)
    mean_skill = sum(skills.values()) / len(skills)
    criteria = {
        "mean_rmse_ratio_to_absolute_baseline_at_most_0p90": mean_ratio <= 0.90,
        "no_output_rmse_ratio_above_1p02": max(ratios.values()) <= 1.02,
        "positive_mean_skill_over_persistence": mean_skill > 0,
        "raw_support_mean_p95_max_and_frequency_not_worse": max(tail_ratios.values()) <= 1.0,
    }
    passed = all(criteria.values())
    return {
        "status": "numeric_pass_pending_visual_review" if passed else "failed",
        "decision": "hold_for_visual_review" if passed else "reject_residual_candidate",
        "rmse_ratio_to_absolute_baseline_by_output": ratios,
        "skill_over_persistence_by_output": skills,
        "mean_rmse_ratio_to_absolute_baseline": mean_ratio,
        "mean_skill_over_persistence": mean_skill,
        "raw_support_mean_tail_ratios": tail_ratios,
        "criteria": criteria,
        "visual_gate": "all fixed winter/summer raw members must lose pervasive speckle",
        "publication_ready": False,
    }


def publish_residual_mechanics_gate(
    *,
    output_dir: str | Path,
    val_history: list[dict[str, Any]],
    clearml: Any,
    gate: dict[str, Any],
) -> None:
    """Publish one authoritative residual terminal decision to every sink."""
    if not val_history:
        raise ValueError("residual terminal gate requires non-empty validation history")
    gate_copy = dict(gate)
    _atomic_json(Path(output_dir) / "coarse_mechanics_gate.json", gate_copy)
    val_history[-1]["coarse_mechanics_gate"] = gate_copy
    _atomic_json(Path(output_dir) / "metrics.json", val_history)
    if clearml is not None:
        clearml.report_single_value(
            "coarse_residual_mechanics_gate_passed",
            float(gate_copy["status"] == "numeric_pass_pending_visual_review"),
        )


class CoarsePersistenceResidualTrainer(CoarseCascadeDynamicsTrainer):
    """Train the exact R=C-P coordinate change with the baseline architecture."""

    def __init__(self, *args, **kwargs):
        baseline = kwargs.pop("residual_baseline", None)
        if not isinstance(baseline, dict):
            raise ValueError("residual trainer requires a SHA-bound baseline metric artifact")
        baseline_path = Path(str(baseline.get("path", "")))
        baseline_sha256 = str(baseline.get("sha256", ""))
        if not baseline_path.is_file() or self._sha256(baseline_path) != baseline_sha256:
            raise ValueError("residual baseline metric artifact is missing or differs")
        self._residual_baseline_path = baseline_path
        self._residual_baseline_sha256 = baseline_sha256
        self._residual_baseline_metrics = json.loads(baseline_path.read_text(encoding="utf-8"))
        bound_paths = {
            "samples": ("samples_path", "samples_sha256"),
            "metadata": ("metadata_path", "metadata_sha256"),
            "sentinel": ("sentinel_path", "sentinel_sha256"),
        }
        self._residual_baseline_binding: dict[str, dict[str, str]] = {}
        for name, (path_key, sha_key) in bound_paths.items():
            path = Path(str(baseline.get(path_key, "")))
            sha256 = str(baseline.get(sha_key, ""))
            if not path.is_file() or self._sha256(path) != sha256:
                raise ValueError(f"residual baseline {name} artifact is missing or differs")
            self._residual_baseline_binding[name] = {"path": str(path), "sha256": sha256}
        source_metadata = json.loads(
            Path(self._residual_baseline_binding["metadata"]["path"]).read_text(encoding="utf-8")
        )
        current_data = kwargs.get("data_config")
        if not isinstance(current_data, dict):
            raise ValueError("residual trainer requires effective data_config provenance")
        if (
            source_metadata.get("normalization_means") != current_data.get("means")
            or source_metadata.get("normalization_stds") != current_data.get("stds")
        ):
            raise ValueError("residual and absolute baseline normalization differs")
        source_sentinel = json.loads(
            Path(self._residual_baseline_binding["sentinel"]["path"]).read_text(encoding="utf-8")
        )["pilot_subset"]
        current_provenance = kwargs.get("dataset_provenance", {}).get("pilot_subset")
        for key in (
            "validation_indices",
            "validation_indices_sha256",
            "diagnostic_subset_positions",
            "diagnostic_case_ids",
        ):
            if not isinstance(current_provenance, dict) or current_provenance.get(
                key
            ) != source_sentinel.get(key):
                raise ValueError(f"residual and absolute baseline pilot provenance differs: {key}")
        self._residual_baseline_samples = torch.load(
            self._residual_baseline_binding["samples"]["path"],
            map_location="cpu",
            weights_only=True,
        )
        super().__init__(*args, **kwargs)
        if self._planned_updates != 2048:
            raise ValueError("residual mechanics pilot requires exactly 2048 optimizer updates")
        self._validate_paired_diagnostic_batch()
        self._write_residual_manifest()

    def _validate_paired_diagnostic_batch(self) -> None:
        raw = self._coarse_diagnostic_batch_raw
        if not isinstance(raw, dict):
            raise ValueError("residual trainer requires the fixed diagnostic batch")
        case_ids = [str(value) for value in raw.get("meta", {}).get("case_id", ())]
        if case_ids != list(self._residual_baseline_samples.get("validation_case_ids", ())):
            raise ValueError("residual diagnostic identities differ from the absolute baseline")
        valid = raw["valid_mask"][:, :1].float()
        truth, active, fraction = coarse_target(raw["truth"], valid)
        persistence, _, _ = coarse_persistence_from_condition(raw["structured_conditioning"], valid)
        means = tuple(float(value) for value in self.data_config["means"]) * len(DIRECT_LEADS)
        stds = tuple(float(value) for value in self.data_config["stds"]) * len(DIRECT_LEADS)
        paired = self._residual_baseline_samples
        checks = (
            ("coarse_active_mask", active, 0.0),
            ("coarse_ocean_fraction", fraction, 0.0),
            ("coarse_truth_physical", channel_denormalize(truth, means, stds), 1e-7),
            (
                "coarse_persistence_physical",
                channel_denormalize(persistence, means, stds),
                1e-7,
            ),
        )
        for key, current, atol in checks:
            reference = paired.get(key)
            if not torch.is_tensor(reference):
                raise ValueError(f"absolute baseline samples lack {key}")
            torch.testing.assert_close(current.cpu(), reference, rtol=0.0, atol=atol)

    def _write_residual_manifest(self) -> None:
        path = Path(self.output_dir) / "coarse_cascade_manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest.update(
            {
                "sampler": "CoarsePersistenceResidualSampler",
                "target": "R=C-P",
                "reconstruction": "C=P+R",
                "persistence_source": "normalized d0 SIC/SIT condition only; mask-aware D",
                "conditional_law": "p(C|c_full) via invertible translation R=C-P(c_full)",
                "residual_standardization": "none",
                "clipping_or_postprocessing": False,
                "residual_module_sha256": self._sha256(Path(__file__)),
                "baseline_metrics_path": str(self._residual_baseline_path),
                "baseline_metrics_sha256": self._residual_baseline_sha256,
                "paired_baseline_artifacts": self._residual_baseline_binding,
            }
        )
        _atomic_json(path, manifest)

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
        state, velocity, _, persistence, _ = coarse_residual_flow_pair(
            truth,
            noise,
            batch["structured_conditioning"],
            batch["valid_mask"][:, :1],
            timesteps,
        )
        background, _, _ = coarse_target(batch["background"], batch["valid_mask"][:, :1])
        if not torch.equal(persistence, background):
            raise ValueError("causal d0 persistence differs from dataset persistence baseline")
        return state, velocity

    def _make_model_input(
        self,
        noisy_truth: torch.Tensor,
        batch: dict[str, torch.Tensor],
        **_: object,
    ) -> torch.Tensor:
        model_input = residual_model_input_for_test(
            noisy_truth,
            batch["structured_conditioning"],
            batch["valid_mask"][:, :1],
        )
        if model_input.shape[1] != COARSE_INPUT_CHANNELS:
            raise RuntimeError("coarse residual model input has the wrong channel count")
        return model_input

    def _make_sampler(self, model) -> CoarsePersistenceResidualSampler:
        return CoarsePersistenceResidualSampler(model)

    @staticmethod
    def _low_ice_metrics(
        ensemble: torch.Tensor,
        truth: torch.Tensor,
        active: torch.Tensor,
    ) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for lead_index, lead in enumerate(DIRECT_LEADS):
            ocean = active[:, 0] > 0
            low_ice = ocean & (truth[:, 2 * lead_index] < 0.01)
            if not torch.any(low_ice):
                raise ValueError("residual diagnostics require low-ice validation cells")
            sit = ensemble[:, :, 2 * lead_index + 1]
            selected = sit[low_ice[:, None].expand_as(sit)]
            metrics.update(
                {
                    f"d{lead}_low_ice_raw_sit_rms": float(torch.sqrt(selected.square().mean()).item()),
                    f"d{lead}_low_ice_raw_sit_mae": float(selected.abs().mean().item()),
                    f"d{lead}_low_ice_raw_sit_positive_p95": float(
                        torch.quantile(selected.clamp_min(0), 0.95).item()
                    ),
                    f"d{lead}_low_ice_raw_sit_negative_p95": float(
                        torch.quantile((-selected).clamp_min(0), 0.95).item()
                    ),
                }
            )
        return metrics

    @torch.no_grad()
    def _coarse_diagnostic(self, step: int, label: str, checkpoint_name: str) -> dict[str, float]:
        from .structured_trajectory_evaluation import make_structured_trajectory_figure

        batch = self._diagnostic_batch()
        condition = batch["structured_conditioning"]
        valid = batch["valid_mask"][:, :1]
        truth_normalized, active, fraction = coarse_target(batch["truth"], valid)
        persistence_normalized, _, _ = coarse_persistence_from_condition(condition, valid)
        dataset_persistence, _, _ = coarse_target(batch["background"], valid)
        if not torch.equal(persistence_normalized, dataset_persistence):
            raise ValueError("diagnostic persistence differs from the causal d0 condition")
        case_count = truth_normalized.shape[0]
        raw_members = []
        raw_residuals = []
        paired_33_members = []
        paired_33_residuals = []
        raw_noises = []
        noise_seeds = []
        was_training = self.model.training
        with self._sampling_model() as sample_model:
            sample_model.eval()
            sampler = self._make_sampler(sample_model)
            for member in range(2):
                noises = []
                member_seeds = []
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
                reconstructed = sampler.sample_conditioned(
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
                if sampler.last_residual is None:
                    raise RuntimeError("residual sampler did not expose its raw residual endpoint")
                raw_members.append(reconstructed)
                raw_residuals.append(sampler.last_residual.detach().clone())
                if step + 1 == 2048:
                    paired = sampler.sample_conditioned(
                        structured_conditioning=condition,
                        valid_mask=valid,
                        initial_noise=noise,
                        num_timesteps=33,
                        device=self.accelerator.device,
                        method="rk4",
                        end_time=0.0,
                    )
                    paired_33_members.append(paired)
                    if sampler.last_residual is None:
                        raise RuntimeError("paired residual endpoint is missing")
                    paired_33_residuals.append(sampler.last_residual.detach().clone())
        if was_training:
            self.model.train()

        stacked_noises = torch.stack(raw_noises, dim=1)
        torch.testing.assert_close(
            stacked_noises,
            self._residual_baseline_samples["raw_coarse_noise_normalized"],
            rtol=0.0,
            atol=0.0,
        )
        normalized_ensemble = torch.stack(raw_members, dim=1)
        normalized_residuals = torch.stack(raw_residuals, dim=1)
        physical_ensemble = self._normalized_to_physical(normalized_ensemble.flatten(0, 1)).unflatten(
            0, (case_count, 2)
        )
        normalized_ensemble_33 = None
        physical_ensemble_33 = None
        if paired_33_members:
            normalized_ensemble_33 = torch.stack(paired_33_members, dim=1)
            physical_ensemble_33 = self._normalized_to_physical(
                normalized_ensemble_33.flatten(0, 1)
            ).unflatten(0, (case_count, 2))
        physical_truth = self._normalized_to_physical(truth_normalized)
        physical_persistence = self._normalized_to_physical(persistence_normalized)
        checkpoint_path = Path(self.output_dir) / checkpoint_name
        ema_path = Path(self.output_dir) / f"ema_{checkpoint_name}"
        if not checkpoint_path.is_file() or not ema_path.is_file():
            raise FileNotFoundError("residual diagnostic requires durable raw and EMA checkpoints")
        payload: dict[str, Any] = {
            "diagnostic_status": "raw_residual_and_reconstruction_saved_metrics_pending",
            "diagnostic_role": "generated_persistence_residual_coarse_forecast_mechanics",
            "future_truth_used_for_condition": False,
            "forecast_claim_permitted": False,
            "optimizer_updates": step + 1,
            "validation_case_ids": list(self._coarse_diagnostic_case_ids),
            "coarse_noise_seeds_by_member_case": noise_seeds,
            "raw_coarse_noise_normalized": stacked_noises,
            "paired_absolute_baseline_noise_exact_match": True,
            "raw_residual_members_normalized": normalized_residuals.detach().cpu(),
            "raw_reconstructed_members_normalized": normalized_ensemble.detach().cpu(),
            "raw_reconstructed_members_physical": physical_ensemble.detach().cpu(),
            "coarse_truth_physical": physical_truth.detach().cpu(),
            "coarse_persistence_physical": physical_persistence.detach().cpu(),
            "coarse_active_mask": active.detach().cpu(),
            "coarse_ocean_fraction": fraction.detach().cpu(),
            "checkpoint": checkpoint_name,
            "checkpoint_sha256": self._sha256(checkpoint_path),
            "ema_checkpoint": ema_path.name,
            "ema_checkpoint_sha256": self._sha256(ema_path),
            "sampling_weight_role": self._sampling_weight_label(),
            "sampling": {"primary": "rk4_17", "paired_final": step + 1 == 2048},
            "metrics": {},
        }
        if paired_33_members:
            payload["paired_rk4_33_reconstructed_normalized"] = normalized_ensemble_33.cpu()
            payload["paired_rk4_33_reconstructed_physical"] = physical_ensemble_33.cpu()
            payload["paired_rk4_33_residual_normalized"] = torch.stack(
                paired_33_residuals, dim=1
            ).cpu()
        diagnostics = Path(self.output_dir) / "coarse_diagnostics"
        diagnostics.mkdir(parents=True, exist_ok=True)
        payload_path = diagnostics / f"{label}_samples.pt"
        self._atomic_save_direct_payload(payload, payload_path)

        metrics: dict[str, float] = {
            "step": float(step + 1),
            "diagnostic_case_count": float(case_count),
            **self._raw_support_metrics(physical_ensemble, active),
            **self._raw_support_magnitude_metrics(physical_ensemble, active),
            **self._low_ice_metrics(physical_ensemble, physical_truth, active),
        }
        mean = physical_ensemble.mean(dim=1)
        anomalies = physical_ensemble - mean[:, None]
        for lead_index, lead in enumerate(DIRECT_LEADS):
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
                member_rmses = [
                    self._fraction_weighted_rmse(
                        physical_ensemble[:, member, channel : channel + 1],
                        physical_truth[:, channel : channel + 1],
                        fraction,
                    )
                    for member in range(2)
                ]
                metrics[f"{key}_individual_member_mean_rmse"] = sum(member_rmses) / len(
                    member_rmses
                )
                metrics[f"{key}_member_roughness"] = self._roughness(
                    physical_ensemble[:, 0, channel : channel + 1], active
                )
                metrics[f"{key}_anomaly_lag1_roughness"] = self._roughness(
                    anomalies[:, 0, channel : channel + 1], active
                )
                metrics[f"{key}_truth_roughness"] = self._roughness(
                    physical_truth[:, channel : channel + 1], active
                )
        if physical_ensemble_33 is not None and normalized_ensemble_33 is not None:
            paired_metrics = {
                **self._raw_support_metrics(physical_ensemble_33, active),
                **self._raw_support_magnitude_metrics(physical_ensemble_33, active),
                **self._low_ice_metrics(physical_ensemble_33, physical_truth, active),
            }
            metrics.update({f"rk4_33_{key}": value for key, value in paired_metrics.items()})
            mean_33 = physical_ensemble_33.mean(dim=1)
            anomalies_33 = physical_ensemble_33 - mean_33[:, None]
            for lead_index, lead in enumerate(DIRECT_LEADS):
                for offset, field in enumerate(("sic", "sit")):
                    channel = 2 * lead_index + offset
                    key = f"d{lead}_{field}"
                    metrics[f"rk4_33_{key}_raw_mean_rmse"] = self._fraction_weighted_rmse(
                        mean_33[:, channel : channel + 1],
                        physical_truth[:, channel : channel + 1],
                        fraction,
                    )
                    member_rmses_33 = [
                        self._fraction_weighted_rmse(
                            physical_ensemble_33[:, member, channel : channel + 1],
                            physical_truth[:, channel : channel + 1],
                            fraction,
                        )
                        for member in range(2)
                    ]
                    metrics[f"rk4_33_{key}_individual_member_mean_rmse"] = sum(
                        member_rmses_33
                    ) / len(member_rmses_33)
                    physical_differences = []
                    normalized_differences = []
                    for member in range(2):
                        physical_delta = (
                            physical_ensemble[:, member, channel : channel + 1]
                            - physical_ensemble_33[:, member, channel : channel + 1]
                        )
                        normalized_delta = (
                            normalized_ensemble[:, member, channel : channel + 1]
                            - normalized_ensemble_33[:, member, channel : channel + 1]
                        )
                        physical_differences.append(
                            self._fraction_weighted_rmse(
                                physical_delta, torch.zeros_like(physical_delta), fraction
                            )
                        )
                        normalized_differences.append(
                            self._fraction_weighted_rmse(
                                normalized_delta, torch.zeros_like(normalized_delta), fraction
                            )
                        )
                    metrics[f"{key}_rk4_17_vs_33_physical_rmse"] = sum(
                        physical_differences
                    ) / len(physical_differences)
                    metrics[f"{key}_rk4_17_vs_33_normalized_rmse"] = sum(
                        normalized_differences
                    ) / len(normalized_differences)
                    for member in range(2):
                        metrics[f"rk4_33_{key}_member{member}_roughness"] = self._roughness(
                            physical_ensemble_33[:, member, channel : channel + 1], active
                        )
                        metrics[
                            f"rk4_33_{key}_member{member}_anomaly_lag1_roughness"
                        ] = self._roughness(
                            anomalies_33[:, member, channel : channel + 1], active
                        )
        bad = [key for key, value in metrics.items() if not math.isfinite(value)]
        if bad:
            raise FloatingPointError(f"residual diagnostic produced non-finite metrics: {bad}")
        payload["diagnostic_status"] = "complete_before_plot"
        payload["metrics"] = metrics
        self._atomic_save_direct_payload(payload, payload_path)

        visual_ensembles = [("rk4_17", physical_ensemble)]
        if physical_ensemble_33 is not None:
            visual_ensembles.append(("rk4_33", physical_ensemble_33))
        for solver_label, visual_ensemble in visual_ensembles:
            for case_index in range(case_count):
                for member in range(2):
                    figure = make_structured_trajectory_figure(
                        physical_truth[case_index],
                        physical_persistence[case_index],
                        visual_ensemble[case_index, member],
                        active[case_index],
                        title=(
                            f"coarse residual {label}: {solver_label} raw member {member}; "
                            f"{payload['validation_case_ids'][case_index]}"
                        ),
                        origin="upper",
                        lead_days=DIRECT_LEADS,
                    )
                    image_path = diagnostics / (
                        f"{label}_case{case_index:02d}_{solver_label}_raw_member{member}.png"
                    )
                    figure.savefig(image_path, dpi=160)
                    plt.close(figure)
                    if self.clearml is not None:
                        series = f"case{case_index:02d}_{solver_label}_raw_member{member}"
                        self.clearml.report_image(
                            "coarse_residual/samples", series, image_path, step + 1
                        )
        _atomic_json(diagnostics / f"{label}_metrics.json", metrics)
        if self.clearml is not None:
            for key, value in metrics.items():
                if key != "step":
                    self.clearml.report_scalar("coarse_residual/physical", key, value, step + 1)
        self._coarse_diagnostic_history.append(dict(metrics))
        return metrics

    def _after_training_epoch(self, epoch: int, global_step: int) -> None:
        if not self.accelerator.is_main_process:
            return
        if global_step < 2048:
            super()._after_training_epoch(epoch, global_step)
            return
        if global_step != 2048:
            raise RuntimeError(f"residual terminal gate expected step 2048, got {global_step}")
        diagnostic_step = global_step - 1
        if diagnostic_step not in self._coarse_diagnostic_steps:
            self._require_full_state_recovery(global_step)
            self._coarse_diagnostic_steps.add(diagnostic_step)
            checkpoint = f"coarse_update_{global_step:04d}.pth"
            self.save_model_custom(checkpoint)
            self._coarse_diagnostic(diagnostic_step, f"update_{global_step:04d}", checkpoint)
        actual_steps = [int(record["step"]) for record in self._coarse_diagnostic_history]
        if actual_steps != [512, 1024, 1536, 2048]:
            raise RuntimeError(
                "residual mechanics gate requires diagnostics [512, 1024, 1536, 2048], "
                f"got {actual_steps}"
            )
        gate = residual_mechanics_gate(
            self._coarse_diagnostic_history[-1], self._residual_baseline_metrics
        )
        gate.update(
            {
                "epoch": epoch + 1,
                "optimizer_updates": global_step,
                "baseline_metrics_path": str(self._residual_baseline_path),
                "baseline_metrics_sha256": self._residual_baseline_sha256,
            }
        )
        publish_residual_mechanics_gate(
            output_dir=self.output_dir,
            val_history=self.val_history,
            clearml=self.clearml,
            gate=gate,
        )


class CoarseStandardizedPersistenceResidualTrainer(CoarsePersistenceResidualTrainer):
    """Train W=(C-P-mu)/s using immutable train-only diagonal statistics."""

    def __init__(self, *args, **kwargs):
        binding = kwargs.pop("residual_statistics", None)
        if not isinstance(binding, dict):
            raise ValueError("standardized residual trainer requires a statistics binding")
        path = Path(str(binding.get("path", "")))
        sha256 = str(binding.get("sha256", ""))
        if not path.is_file() or self._sha256(path) != sha256:
            raise ValueError("residual statistics artifact is missing or differs")
        statistics = json.loads(path.read_text(encoding="utf-8"))
        if statistics.get("status") != "complete" or statistics.get("split") != "train":
            raise ValueError("residual statistics must be a completed train-only artifact")
        if int(statistics.get("train_case_count", -1)) != 4096:
            raise ValueError("residual statistics must contain the exact 4096-case pilot subset")
        current_provenance = kwargs.get("dataset_provenance", {}).get("pilot_subset", {})
        if statistics.get("train_indices_sha256") != current_provenance.get(
            "train_indices_sha256"
        ):
            raise ValueError("residual statistics and training subset differ")
        if statistics.get("selected_case_ids") != current_provenance.get("train_case_ids"):
            raise ValueError("residual statistics and training case identities differ")
        dataset_provenance = kwargs.get("dataset_provenance", {})
        current_calendar = dataset_provenance.get("train_calendar", {})
        if statistics.get("ordered_inventory_sha256") != current_calendar.get(
            "ordered_inventory_sha256"
        ):
            raise ValueError("residual statistics and ordered train inventory differ")
        if statistics.get("static_valid_mask_sha256") != dataset_provenance.get(
            "static_valid_mask_sha256"
        ):
            raise ValueError("residual statistics and static valid mask differ")
        current_data = kwargs.get("data_config")
        if not isinstance(current_data, dict) or statistics.get(
            "data_config_sha256"
        ) != self._canonical_hash(current_data):
            raise ValueError("residual statistics and effective data configuration differ")
        reference = torch.empty((1, DIRECT_OUTPUT_CHANNELS, 1, 1))
        _residual_affine_tensors(statistics, reference)
        self._residual_statistics_path = path
        self._residual_statistics_sha256 = sha256
        self._residual_statistics = statistics
        super().__init__(*args, **kwargs)

    @staticmethod
    def _canonical_hash(payload: Mapping[str, Any]) -> str:
        import hashlib

        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def _write_residual_manifest(self) -> None:
        super()._write_residual_manifest()
        path = Path(self.output_dir) / "coarse_cascade_manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest.update(
            {
                "sampler": "CoarseStandardizedPersistenceResidualSampler",
                "target": "W=(C-P-mu)/s",
                "reconstruction": "C=P+mu+s*W",
                "conditional_law": "p(C|c_full) via fixed invertible train-only diagonal affine map",
                "base_noise_coordinate": "epsilon~N(0,I) in W-space",
                "residual_standardization": "six-channel train-only case-equal diagonal",
                "residual_statistics_path": str(self._residual_statistics_path),
                "residual_statistics_sha256": self._residual_statistics_sha256,
                "residual_statistics": {
                    "means": self._residual_statistics["means"],
                    "stds": self._residual_statistics["stds"],
                    "train_indices_sha256": self._residual_statistics[
                        "train_indices_sha256"
                    ],
                },
            }
        )
        _atomic_json(path, manifest)

    def _make_training_pair(
        self,
        truth: torch.Tensor,
        batch: dict[str, torch.Tensor],
        timesteps: torch.Tensor,
        residual_background: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del residual_background
        shape = (
            truth.shape[0],
            DIRECT_OUTPUT_CHANNELS,
            truth.shape[-2] // 2,
            truth.shape[-1] // 2,
        )
        noise = torch.randn(shape, dtype=truth.dtype, device=truth.device, generator=generator)
        state, velocity, _, persistence, _ = coarse_standardized_residual_flow_pair(
            truth,
            noise,
            batch["structured_conditioning"],
            batch["valid_mask"][:, :1],
            timesteps,
            self._residual_statistics,
        )
        background, _, _ = coarse_target(batch["background"], batch["valid_mask"][:, :1])
        if not torch.equal(persistence, background):
            raise ValueError("causal d0 persistence differs from dataset persistence baseline")
        return state, velocity

    def _make_sampler(self, model) -> CoarseStandardizedPersistenceResidualSampler:
        return CoarseStandardizedPersistenceResidualSampler(model, self._residual_statistics)
