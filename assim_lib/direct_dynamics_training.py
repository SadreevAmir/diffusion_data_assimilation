"""Clean all-hour direct-state flow for multi-day SIC/SIT dynamics.

This recovery path deliberately stays close to the accepted assimilation model:
the learned state is the six normalized physical fields at d+3/d+6/d+9.
There are no occurrence/cap heads and no inactive latent fillers.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path

import torch

from .config import TrainingConfig, load_json, merge_config_overrides, resolve_path
from .data import build_dataset
from .main import _debug
from .model_io import build_unet
from .runtime import add_noise, build_dataloader, seed_everything
from .runtime import make_normalized_xy_grid
from .trainer import UNetTrainer, _atomic_json
from .transforms import channel_denormalize


DIRECT_LEADS = (3, 6, 9)
DIRECT_OUTPUT_CHANNELS = 2 * len(DIRECT_LEADS)
DIRECT_CONDITION_CHANNELS = 15
DIRECT_INPUT_CHANNELS = DIRECT_OUTPUT_CHANNELS + 2 + DIRECT_CONDITION_CHANNELS


def _repeat_field_stats(values: list[float] | tuple[float, ...]) -> tuple[float, ...]:
    pair = tuple(float(value) for value in values)
    if len(pair) != 2:
        raise ValueError("direct dynamics requires exactly SIC/SIT normalization")
    return pair * len(DIRECT_LEADS)


def validate_direct_dataset(dataset) -> dict:
    if dataset.hour_mode != "all" or dataset.hours_per_day != 24:
        raise ValueError("direct dynamics requires all 24 archive slices")
    if tuple(dataset.trajectory_lead_days) != DIRECT_LEADS:
        raise ValueError(f"trajectory leads must be {DIRECT_LEADS}")
    if dataset.background_strategy != "none":
        raise ValueError("dynamics must not use a previous-year background")
    if len(dataset) != dataset._num_days() * 24:
        raise ValueError("all-hour dataset length is inconsistent")

    checked = []
    for hour in (0, 12, 23):
        item = dataset[hour]
        if int(item["meta"]["archive_slice_index"]) != hour:
            raise ValueError("resolved archive slice is inconsistent")
        if tuple(item["truth"].shape) != (DIRECT_OUTPUT_CHANNELS, *dataset.image_size):
            raise ValueError("direct target must contain six normalized physical channels")
        if tuple(item["background"].shape) != (DIRECT_OUTPUT_CHANNELS, *dataset.image_size):
            raise ValueError("persistence tensor must contain six channels")
        if tuple(item["structured_conditioning"].shape) != (
            DIRECT_CONDITION_CHANNELS,
            *dataset.image_size,
        ):
            raise ValueError("direct conditioning must contain fifteen channels")
        if not torch.isfinite(item["truth"]).all():
            raise ValueError("direct normalized target contains non-finite values")
        if not torch.isfinite(item["structured_conditioning"]).all():
            raise ValueError("direct conditioning contains non-finite values")
        checked.append(
            {
                "hour": hour,
                "case_id": item["meta"]["case_id"],
                "target_paths": item["meta"]["target_trajectory_paths"],
                "background_role": item["meta"]["background_role"],
            }
        )
    return {
        "status": "passed",
        "length": len(dataset),
        "days": dataset._num_days(),
        "hours_per_day": dataset.hours_per_day,
        "checked_slices": checked,
    }


class DirectDynamicsTrainer(UNetTrainer):
    """Use the proven flow objective with explicit d0/forcing conditioning."""

    diagnostic_steps = frozenset({99, 499})

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._direct_diagnostic_steps: set[int] = set()
        self._direct_validation_batch = None
        self._direct_means = _repeat_field_stats(self.channel_means)
        self._direct_stds = _repeat_field_stats(self.channel_stds)

    def _make_training_pair(
        self,
        truth: torch.Tensor,
        batch: dict[str, torch.Tensor],
        timesteps: torch.Tensor,
        residual_background: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        valid = batch["valid_mask"][:, :1].expand_as(truth) > 0
        clean = torch.where(valid, truth, torch.zeros_like(truth))
        noise = torch.randn(
            clean.shape,
            dtype=clean.dtype,
            device=clean.device,
            generator=generator,
        )
        noise = torch.where(valid, noise, torch.zeros_like(noise))
        time = timesteps.view(-1, 1, 1, 1)
        return (1.0 - time) * clean + time * noise, noise - clean

    def _make_model_input(
        self,
        noisy_truth: torch.Tensor,
        batch: dict[str, torch.Tensor],
        **_: object,
    ) -> torch.Tensor:
        condition = batch.get("structured_conditioning")
        if condition is None:
            raise KeyError("direct dynamics requires explicit structured_conditioning")
        grid = self._grid.expand(noisy_truth.shape[0], -1, -1, -1)
        model_input = torch.cat((noisy_truth, grid, condition), dim=1)
        if model_input.shape[1] != DIRECT_INPUT_CHANNELS:
            raise ValueError(
                f"direct model input has {model_input.shape[1]} channels, "
                f"expected {DIRECT_INPUT_CHANNELS}"
            )
        return model_input

    def _flow_matching_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return self._masked_mse(pred, target, batch["valid_mask"][:, :1])

    def _normalized_to_physical(self, value: torch.Tensor) -> torch.Tensor:
        return channel_denormalize(value.to(dtype=torch.float32), self._direct_means, self._direct_stds)

    @staticmethod
    def _project_for_display(value: torch.Tensor) -> torch.Tensor:
        projected = value.clone()
        projected[:, 0::2].clamp_(0.0, 1.0)
        projected[:, 1::2].clamp_(min=0.0)
        return projected

    @staticmethod
    def _masked_rmse(value: torch.Tensor, truth: torch.Tensor, valid: torch.Tensor) -> float:
        mask = valid.expand_as(value)
        return float((((value - truth).square() * mask).sum() / mask.sum().clamp(min=1)).sqrt().item())

    @staticmethod
    def _roughness(value: torch.Tensor, valid: torch.Tensor) -> float:
        mask_y = valid[..., 1:, :] * valid[..., :-1, :]
        mask_x = valid[..., :, 1:] * valid[..., :, :-1]
        dy = (value[..., 1:, :] - value[..., :-1, :]).abs()
        dx = (value[..., :, 1:] - value[..., :, :-1]).abs()
        numerator = (dy * mask_y).sum() + (dx * mask_x).sum()
        denominator = mask_y.sum() + mask_x.sum()
        return float((numerator / denominator.clamp(min=1)).item())

    @staticmethod
    def _require_finite_on_valid(
        value: torch.Tensor, valid: torch.Tensor, *, label: str
    ) -> None:
        mask = valid.expand_as(value) > 0
        if not torch.any(mask):
            raise ValueError(f"{label} has no valid ocean points")
        invalid = ~torch.isfinite(value)
        count = int((invalid & mask).sum().item())
        if count:
            raise FloatingPointError(
                f"{label} contains {count} NaN/Inf values on valid ocean points"
            )

    @classmethod
    def _raw_support_metrics(
        cls, raw_ensemble: torch.Tensor, valid: torch.Tensor
    ) -> dict[str, float]:
        expanded_valid = valid[:, None].expand_as(raw_ensemble)
        cls._require_finite_on_valid(
            raw_ensemble, expanded_valid, label="raw direct-dynamics ensemble"
        )
        sic = raw_ensemble[:, :, 0::2]
        sit = raw_ensemble[:, :, 1::2]
        support = valid[:, None].expand_as(sic) > 0
        denominator = max(int(support.sum().item()), 1)
        sic_bad = ((sic < 0.0) | (sic > 1.0)) & support
        sit_bad = (sit < 0.0) & support
        return {
            "sic_raw_support_violation_fraction": float(sic_bad.sum().item() / denominator),
            "sit_raw_support_violation_fraction": float(sit_bad.sum().item() / denominator),
            "joint_raw_support_violation_fraction": float(
                (sic_bad | sit_bad).sum().item() / denominator
            ),
        }

    def _fixed_validation_batch(self) -> dict[str, torch.Tensor]:
        if self._direct_validation_batch is None:
            raw = next(iter(self.val_dataloader))
            batch = self._batch_to_device(raw)
            self._direct_validation_batch = {
                key: value[: min(3, value.shape[0])]
                for key, value in batch.items()
                if torch.is_tensor(value)
            }
        return self._direct_validation_batch

    @torch.no_grad()
    def _direct_diagnostic(
        self, step: int, label: str, *, case_count: int = 1
    ) -> dict[str, float]:
        from .structured_trajectory_evaluation import make_structured_trajectory_figure

        batch = self._fixed_validation_batch()
        case_count = min(int(case_count), batch["truth"].shape[0])
        if case_count <= 0:
            raise ValueError("direct diagnostic requires at least one validation case")
        batch = {key: value[:case_count] for key, value in batch.items()}
        valid = batch["valid_mask"][:, :1]
        members = []
        was_training = self.model.training
        with self._sampling_model() as sample_model:
            sample_model.eval()
            sampler = self._make_sampler(sample_model)
            for member in range(2):
                generator = torch.Generator(device=self.accelerator.device)
                generator.manual_seed(99173 + 1009 * member)
                initial_noise = torch.randn(
                    (case_count, DIRECT_OUTPUT_CHANNELS, *self.config.image_size),
                    dtype=batch["truth"].dtype,
                    device=self.accelerator.device,
                    generator=generator,
                )
                normalized = sampler.sample_conditioned(
                    background=batch["background"],
                    background_mask=torch.ones_like(batch["background"]),
                    obs_values=batch["obs_values"],
                    obs_mask=batch["obs_mask"],
                    water_mask=batch["water_mask"],
                    size=self.config.image_size,
                    num_timesteps=self.config.num_sample_timesteps,
                    device=self.accelerator.device,
                    method=self.config.sample_method,
                    rtol=self.config.sample_rtol,
                    atol=self.config.sample_atol,
                    start_mode="noise",
                    initial_noise=initial_noise,
                    sample_target="state",
                    model_conditioning=batch["structured_conditioning"],
                    state_channels=DIRECT_OUTPUT_CHANNELS,
                    end_time=0.0,
                    state_mask=valid.expand(-1, DIRECT_OUTPUT_CHANNELS, -1, -1),
                )
                members.append(self._normalized_to_physical(normalized))
        if was_training:
            self.model.train()

        raw_ensemble = torch.stack(members, dim=1)
        metrics: dict[str, float] = {
            "step": float(step),
            "diagnostic_case_count": float(case_count),
            **self._raw_support_metrics(raw_ensemble, valid),
        }
        ensemble = self._project_for_display(raw_ensemble.flatten(0, 1)).unflatten(
            0, (case_count, 2)
        )
        truth = batch["structured_physical_truth"]
        persistence = batch["structured_physical_background"]
        raw_mean = raw_ensemble.mean(dim=1)
        display_mean = ensemble.mean(dim=1)
        self._require_finite_on_valid(truth, valid, label="direct diagnostic truth")
        self._require_finite_on_valid(persistence, valid, label="direct diagnostic persistence")
        for lead_index, lead_day in enumerate(DIRECT_LEADS):
            for field_offset, field_name in enumerate(("sic", "sit")):
                channel = 2 * lead_index + field_offset
                key = f"d{lead_day}_{field_name}"
                metrics[f"{key}_raw_mean_rmse"] = self._masked_rmse(
                    raw_mean[:, channel : channel + 1], truth[:, channel : channel + 1], valid
                )
                metrics[f"{key}_projected_mean_rmse_display_only"] = self._masked_rmse(
                    display_mean[:, channel : channel + 1],
                    truth[:, channel : channel + 1],
                    valid,
                )
                metrics[f"{key}_persistence_rmse"] = self._masked_rmse(
                    persistence[:, channel : channel + 1], truth[:, channel : channel + 1], valid
                )
                metrics[f"{key}_member_roughness"] = self._roughness(
                    raw_ensemble[:, 0, channel : channel + 1], valid
                )
                metrics[f"{key}_truth_roughness"] = self._roughness(
                    truth[0, channel : channel + 1], valid[0]
                )

        samples_dir = Path(self.output_dir) / "direct_diagnostics"
        samples_dir.mkdir(parents=True, exist_ok=True)
        payload_path = samples_dir / f"{label}_samples.pt"
        torch.save(
            {
                "raw_samples": raw_ensemble.detach().cpu(),
                "display_samples": ensemble.detach().cpu(),
                "truth": truth.detach().cpu(),
                "persistence": persistence.detach().cpu(),
                "valid_mask": valid.detach().cpu(),
                "metrics": metrics,
                "lead_days": DIRECT_LEADS,
            },
            payload_path,
        )
        figure_series = (
            ("raw_member0", raw_ensemble[0, 0]),
            ("projected_member0_display_only", ensemble[0, 0]),
            ("raw_ensemble_mean", raw_mean[0]),
            ("projected_ensemble_mean_display_only", display_mean[0]),
        )
        for series, sample in figure_series:
            figure = make_structured_trajectory_figure(
                truth[0],
                persistence[0],
                sample,
                valid[0],
                title=f"direct dynamics {label}: {series}",
                origin="upper",
                lead_days=DIRECT_LEADS,
            )
            image_path = samples_dir / f"{label}_{series}.png"
            figure.savefig(image_path, dpi=160)
            import matplotlib.pyplot as plt

            plt.close(figure)
            if self.clearml is not None:
                self.clearml.report_image("direct_dynamics/samples", series, image_path, step)
        _atomic_json(samples_dir / f"{label}_metrics.json", metrics)
        if self.clearml is not None:
            for key, value in metrics.items():
                if key != "step" and math.isfinite(value):
                    self.clearml.report_scalar("direct_dynamics/physical", key, value, step)
        return metrics

    def _report_train_metrics(self, loss, loss_full, loss_obs, loss_smooth, step: int):
        super()._report_train_metrics(loss, loss_full, loss_obs, loss_smooth, step)
        if (
            self.accelerator.is_main_process
            and step in self.diagnostic_steps
            and step not in self._direct_diagnostic_steps
        ):
            self._direct_diagnostic_steps.add(step)
            self._direct_diagnostic(step, f"step_{step + 1:06d}")

    def _after_training_epoch(self, epoch: int, global_step: int) -> None:
        if self.accelerator.is_main_process:
            metrics = self._direct_diagnostic(
                global_step, f"epoch_{epoch + 1:04d}", case_count=3
            )
            self.val_history[-1]["direct_physical_diagnostic"] = metrics
            if epoch == 0:
                gate_reasons = []
                if metrics["joint_raw_support_violation_fraction"] >= 0.95:
                    gate_reasons.append("at least 95% of raw predictions violate physical support")
                roughness_ratios = []
                for lead_day in DIRECT_LEADS:
                    for field_name in ("sic", "sit"):
                        prefix = f"d{lead_day}_{field_name}"
                        truth_roughness = metrics[f"{prefix}_truth_roughness"]
                        member_roughness = metrics[f"{prefix}_member_roughness"]
                        roughness_ratios.append(member_roughness / max(truth_roughness, 1e-8))
                if max(roughness_ratios) >= 50.0:
                    gate_reasons.append("raw sample roughness exceeds truth by at least 50x")
                gate = {
                    "status": "failed" if gate_reasons else "passed",
                    "epoch": 1,
                    "global_step": int(global_step),
                    "decision": "stop" if gate_reasons else "continue",
                    "criteria": {
                        "finite_raw_values_required": True,
                        "max_joint_support_violation_fraction_exclusive": 0.95,
                        "max_member_to_truth_roughness_ratio_exclusive": 50.0,
                    },
                    "reasons": gate_reasons,
                    "diagnostic_case_count": 3,
                }
                self.val_history[-1]["epoch_1_automatic_safety_gate"] = gate
                _atomic_json(Path(self.output_dir) / "epoch_1_gate.json", gate)
                if self.clearml is not None:
                    self.clearml.report_single_value(
                        "epoch_1_automatic_safety_gate_passed", float(not gate_reasons)
                    )
            _atomic_json(Path(self.output_dir) / "metrics.json", self.val_history)
            if epoch == 0 and gate_reasons:
                raise RuntimeError("epoch-1 automatic safety gate failed: " + "; ".join(gate_reasons))


def _gpu_batch_smoke(model, train_loader, config: TrainingConfig) -> dict:
    """Exercise the exact large-backbone batch shape without an optimizer step."""
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    model = model.to(device).train()
    raw = next(iter(train_loader))
    truth = raw["truth"].to(device)
    valid = raw["valid_mask"][:, :1].to(device).expand_as(truth) > 0
    condition = raw["structured_conditioning"].to(device)
    clean = torch.where(valid, truth, torch.zeros_like(truth))
    noise = torch.where(valid, torch.randn_like(clean), torch.zeros_like(clean))
    time = torch.linspace(0.05, 0.95, truth.shape[0], device=device)
    state = (1.0 - time[:, None, None, None]) * clean + time[:, None, None, None] * noise
    target = noise - clean
    grid = make_normalized_xy_grid(*config.image_size, device=device).expand(
        truth.shape[0], -1, -1, -1
    )
    model_input = torch.cat((state, grid, condition), dim=1)
    if model_input.shape[1] != DIRECT_INPUT_CHANNELS:
        raise ValueError("GPU smoke constructed the wrong direct model input")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        prediction = model(model_input, time * 1000, return_dict=False)[0]
        loss = ((prediction - target).square() * valid).sum() / valid.sum().clamp(min=1)
    if not torch.isfinite(loss):
        raise FloatingPointError("GPU smoke produced a non-finite loss")
    loss.backward()
    for parameter in model.parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise FloatingPointError("GPU smoke produced a non-finite gradient")
    result = {
        "status": "passed",
        "batch_size": int(truth.shape[0]),
        "model_input_shape": list(model_input.shape),
        "model_output_shape": list(prediction.shape),
        "loss": float(loss.detach().item()),
        "peak_gpu_memory_mib": float(torch.cuda.max_memory_allocated(device) / 2**20),
        "activation_checkpointing": bool(config.activation_checkpointing),
    }
    for parameter in model.parameters():
        parameter.grad = None
    model.to("cpu")
    del loss, prediction, model_input, state, target, clean, noise, condition, truth, raw, model
    torch.cuda.empty_cache()
    return result


def run(config_path: Path, *, preflight_only: bool = False) -> dict:
    experiment = load_json(config_path)
    config_dir = config_path.resolve().parent
    data_path = resolve_path(experiment["data_config"], config_dir)
    model_path = resolve_path(experiment["model_config"], config_dir)
    data_config = merge_config_overrides(load_json(data_path), experiment.get("data_overrides"))
    model_raw = load_json(model_path)
    model_config = {**model_raw, **experiment.get("training", {})}
    launch_id = os.environ.get("DIRECT_DYNAMICS_LAUNCH_ID", "").strip()
    if launch_id:
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", launch_id) is None:
            raise ValueError("DIRECT_DYNAMICS_LAUNCH_ID is not a safe identifier")
        base_run_name = str(model_config.get("run_name") or "run")
        model_config["run_name"] = f"{base_run_name}-{launch_id}"
        experiment = dict(experiment)
        experiment["task_name"] = f"{experiment['task_name']}-{launch_id}"
    clearml = experiment.get("clearml", {})
    model_config.update(
        {
            "clearml_project_name": experiment["project_name"],
            "clearml_task_name": experiment["task_name"],
            "clearml_enabled": bool(clearml.get("enabled", True)),
            "clearml_tags": clearml.get("tags", []),
            "clearml_env_path": clearml.get("env_path"),
            "clearml_upload_checkpoints": bool(clearml.get("upload_checkpoints", False)),
        }
    )
    train_config = TrainingConfig.from_dict(model_config)
    if train_config.training_objective != "flow":
        raise ValueError("direct dynamics recovery intentionally uses the accepted state-flow objective")
    if train_config.in_channels != DIRECT_INPUT_CHANNELS or train_config.out_channels != DIRECT_OUTPUT_CHANNELS:
        raise ValueError("direct dynamics model must use 23 inputs and 6 outputs")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("direct dynamics training requires exactly one visible GPU")

    seed_everything(train_config.seed)
    train_dataset = build_dataset(data_config, split="train")
    valid_dataset = build_dataset(data_config, split="valid")
    sentinel = {
        "train": validate_direct_dataset(train_dataset),
        "valid": validate_direct_dataset(valid_dataset),
    }
    train_loader = build_dataloader(
        train_dataset,
        train_config.train_batch_size,
        train_config.num_workers_train,
        shuffle=True,
    )
    valid_loader = build_dataloader(
        valid_dataset,
        train_config.eval_batch_size,
        train_config.num_workers_val,
        shuffle=False,
    )
    expected_steps = len(train_loader) * train_config.num_epochs
    if len(train_loader) < 3000 or expected_steps < 48000:
        raise ValueError(
            f"all-hour optimizer budget is too small: per_epoch={len(train_loader)}, total={expected_steps}"
        )
    _debug(
        f"direct all-hour dataset train={len(train_dataset)} valid={len(valid_dataset)} "
        f"batches={len(train_loader)} planned_steps={expected_steps}"
    )
    output_root = Path(train_config.base_output_dir) / (train_config.run_name or "run")
    if not preflight_only:
        if output_root.exists() or output_root.is_symlink():
            raise FileExistsError(f"refusing to reuse direct training output: {output_root}")
        _atomic_json(output_root / "direct_dataset_sentinel.json", sentinel)
    model = build_unet(train_config)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    _debug(f"direct dynamics parameters={parameter_count:,}")
    if preflight_only:
        result = {
            "status": "preflight_passed",
            "parameter_count": parameter_count,
            "steps_per_epoch": len(train_loader),
            "planned_optimizer_steps": expected_steps,
            "dataset_sentinel": sentinel,
            "gpu_batch_smoke": _gpu_batch_smoke(model, train_loader, train_config),
        }
        return result
    gpu_smoke = _gpu_batch_smoke(model, train_loader, train_config)
    _atomic_json(output_root / "gpu_batch_smoke.json", gpu_smoke)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate)
    from diffusers.optimization import get_cosine_schedule_with_warmup

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=train_config.lr_warmup_steps,
        num_training_steps=expected_steps,
    )
    trainer = DirectDynamicsTrainer(
        config=train_config,
        model=model,
        optimizer=optimizer,
        data_loader_train=train_loader,
        data_loader_val=valid_loader,
        lr_scheduler=scheduler,
        add_noise_func=add_noise,
        experiment_config=experiment,
        model_config=model_raw,
        data_config=data_config,
        dataset_provenance={"train": train_dataset.provenance(), "valid": valid_dataset.provenance()},
        dashboard_dataset=valid_dataset,
    )
    output_dir = trainer.train_loop()
    result = {
        "status": "completed",
        "output_dir": str(Path(output_dir).resolve()),
        "parameter_count": parameter_count,
        "steps_per_epoch": len(train_loader),
        "completed_optimizer_steps": expected_steps,
    }
    _atomic_json(Path(output_dir) / "direct_training_completion.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    arguments = parser.parse_args()
    print(
        json.dumps(
            run(arguments.config.resolve(), preflight_only=arguments.preflight_only),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
