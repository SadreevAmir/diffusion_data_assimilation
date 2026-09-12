"""Colored Gaussian reference law for the variance-preconditioned fine flow.

The physical target remains the exact cascade detail ``R=(I-UD)Y``.  Only the
unconditional Gaussian reference is changed from projected white noise to

    eta_K = S P ((1-alpha) I + alpha G) xi,

where ``G`` is a fixed separable binomial blur, ``P=I-UD`` and ``S`` contains
train-only channel scales.  For ``alpha < 1`` the reference retains non-zero
power at every spatial frequency; no detail band is removed.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .direct_dynamics_cascade import project_detail, residual_flow_pair
from .direct_dynamics_cascade_fine import (
    CASCADE_FACTOR,
    DIRECT_OUTPUT_CHANNELS,
    ProjectedDetailModel,
    _sha256,
    validate_fine_condition,
)
from .direct_dynamics_cascade_fine_preconditioned import (
    PRECONDITIONING_KIND,
    VariancePreconditionedFineCascadeDynamicsTrainer,
    VariancePreconditionedFineCascadeSampler,
    load_variance_preconditioned_fine_cascade_sampler,
)
from .trainer import _atomic_json


COLORED_BASE_KIND = "projected_binomial_blend_gaussian_v1"


def _channel_scales(values: Any) -> tuple[float, ...]:
    if not isinstance(values, list) or len(values) != DIRECT_OUTPUT_CHANNELS:
        raise ValueError("colored base requires exactly six channel scales")
    result = tuple(float(value) for value in values)
    if any(not math.isfinite(value) or value <= 0 for value in result):
        raise ValueError("colored base channel scales must be positive finite")
    return result


def validate_colored_base_contract(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("kind") != COLORED_BASE_KIND:
        raise ValueError("fine colored-base contract has the wrong kind")
    blend = float(value.get("blend", float("nan")))
    if not math.isfinite(blend) or not 0.0 <= blend < 1.0:
        raise ValueError("colored-base blend must be finite and lie in [0,1)")
    scales = _channel_scales(value.get("channel_scales"))
    source = Path(str(value.get("source_audit_path", "")))
    expected_sha256 = str(value.get("source_audit_sha256", ""))
    if not source.is_file() or _sha256(source) != expected_sha256:
        raise ValueError("colored-base source audit is missing or differs")
    audit = json.loads(source.read_text(encoding="utf-8"))
    selected = audit.get("selected", {})
    if (
        audit.get("split") != "train_only"
        or audit.get("schema_version") != 1
        or selected.get("kind") != COLORED_BASE_KIND
        or float(selected.get("blend", float("nan"))) != blend
        or tuple(float(item) for item in selected.get("channel_scales", ())) != scales
    ):
        raise ValueError("colored-base declaration differs from its train-only audit")
    return {
        "kind": COLORED_BASE_KIND,
        "blend": blend,
        "channel_scales": list(scales),
        "kernel_1d": [0.25, 0.5, 0.25],
        "minimum_unprojected_spectral_gain": 1.0 - blend,
        "source_audit_path": str(source.resolve()),
        "source_audit_sha256": expected_sha256,
        "source_split": "train_only",
    }


def binomial_blur(value: torch.Tensor) -> torch.Tensor:
    """Apply the fixed 3x3 separable binomial filter channel-wise."""
    if value.ndim != 4 or not value.is_floating_point():
        raise ValueError("binomial blur requires a floating [B,C,H,W] tensor")
    kernel_1d = torch.tensor((0.25, 0.5, 0.25), dtype=value.dtype, device=value.device)
    kernel = (kernel_1d[:, None] * kernel_1d[None, :]).view(1, 1, 3, 3)
    weight = kernel.expand(value.shape[1], 1, 3, 3)
    padded = F.pad(value, (1, 1, 1, 1), mode="reflect")
    return F.conv2d(padded, weight, groups=value.shape[1])


def colored_projected_gaussian(
    white: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    blend: float,
    channel_scales: tuple[float, ...],
) -> torch.Tensor:
    """Map white noise to the audited full-support colored reference."""
    if white.ndim != 4 or white.shape[1] != DIRECT_OUTPUT_CHANNELS:
        raise ValueError("colored fine base requires six-channel white noise")
    if not math.isfinite(blend) or not 0.0 <= blend < 1.0:
        raise ValueError("colored fine base blend must lie in [0,1)")
    if len(channel_scales) != DIRECT_OUTPUT_CHANNELS:
        raise ValueError("colored fine base requires six channel scales")
    filtered = (1.0 - blend) * white + blend * binomial_blur(white)
    projected = project_detail(filtered.float(), valid_mask.float(), CASCADE_FACTOR)
    scales = torch.as_tensor(
        channel_scales, dtype=projected.dtype, device=projected.device
    ).view(1, DIRECT_OUTPUT_CHANNELS, 1, 1)
    result = projected * scales
    return project_detail(result, valid_mask.float(), CASCADE_FACTOR)


def write_colored_base_manifest(
    output_dir: str | Path,
    contract: dict[str, Any],
    code_commit: str,
) -> None:
    root = Path(output_dir)
    primary_path = root / "fine_cascade_manifest.json"
    preconditioning_path = root / "fine_cascade_preconditioning_manifest.json"
    if not primary_path.is_file() or not preconditioning_path.is_file():
        raise ValueError("colored fine run lacks its primary preconditioning manifests")
    path = root / "fine_cascade_colored_base_manifest.json"
    _atomic_json(
        path,
        {
            "schema_version": 1,
            "sampler": "ColoredVariancePreconditionedFineCascadeSampler",
            "colored_base": contract,
            "target_law": "unchanged_exact_detail_R_equals_I_minus_UD_Y",
            "implementation_sha256": _sha256(Path(__file__)),
            "code_commit": code_commit,
        },
    )
    primary = json.loads(primary_path.read_text(encoding="utf-8"))
    if (
        primary.get("code_commit") != code_commit
        or primary.get("velocity_parameterization") != PRECONDITIONING_KIND
    ):
        raise ValueError("colored fine primary manifest is incompatible")
    primary.update(
        {
            "base_parameterization": COLORED_BASE_KIND,
            "colored_base_manifest": path.name,
            "colored_base_manifest_sha256": _sha256(path),
        }
    )
    _atomic_json(primary_path, primary)


class ColoredVariancePreconditionedFineCascadeSampler(
    VariancePreconditionedFineCascadeSampler
):
    def __init__(self, model, target_residual_rms, projected_base_rms, blend, channel_scales):
        super().__init__(model, target_residual_rms, projected_base_rms)
        self.blend = float(blend)
        self.channel_scales = tuple(float(value) for value in channel_scales)

    def project_initial_noise(
        self, raw_noise: torch.Tensor, valid_mask: torch.Tensor
    ) -> torch.Tensor:
        """Return the exact colored endpoint passed to the reverse ODE."""
        return colored_projected_gaussian(
            raw_noise.float(),
            valid_mask.float(),
            blend=self.blend,
            channel_scales=self.channel_scales,
        )

    @torch.no_grad()
    def sample_conditioned(self, **kwargs: Any) -> torch.Tensor:
        white = kwargs.get("initial_noise")
        condition = kwargs.get("model_conditioning")
        if white is None or condition is None:
            raise ValueError("colored fine sampler requires explicit noise and conditioning")
        valid_mask, _ = validate_fine_condition(condition)
        colored = self.project_initial_noise(white, valid_mask)
        previous = len(self.evidence)
        forwarded = dict(kwargs)
        forwarded["initial_noise"] = colored
        result = super().sample_conditioned(**forwarded)
        if self.capture_evidence:
            if len(self.evidence) != previous + 1:
                raise RuntimeError("colored fine sampler lost its evidence record")
            record = self.evidence[-1]
            record["white_initial_noise"] = white.detach().cpu()
            record["raw_initial_noise"] = white.detach().cpu()
            record["colored_projected_initial_noise"] = colored.detach().cpu()
        return result


class ColoredVariancePreconditionedFineCascadeDynamicsTrainer(
    VariancePreconditionedFineCascadeDynamicsTrainer
):
    """Use the same colored Gaussian law in training and reverse-ODE sampling."""

    def __init__(self, *args, **kwargs):
        experiment = kwargs.get("experiment_config")
        if not isinstance(experiment, dict):
            raise TypeError("colored fine trainer requires experiment_config")
        contract = validate_colored_base_contract(experiment.get("fine_colored_base"))
        self._colored_blend = float(contract["blend"])
        self._colored_channel_scales = tuple(contract["channel_scales"])
        super().__init__(*args, **kwargs)
        write_colored_base_manifest(
            self.output_dir,
            contract,
            self._fine_code_identity["git_commit"],
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
        white = torch.randn(
            truth.shape,
            dtype=truth.dtype,
            device=truth.device,
            generator=generator,
        )
        colored = colored_projected_gaussian(
            white,
            batch["valid_mask"][:, :1],
            blend=self._colored_blend,
            channel_scales=self._colored_channel_scales,
        )
        state, velocity, _, _ = residual_flow_pair(
            truth,
            colored,
            batch["valid_mask"][:, :1],
            timesteps,
            CASCADE_FACTOR,
        )
        return state, velocity

    def _make_sampler(self, model) -> ColoredVariancePreconditionedFineCascadeSampler:
        sampler = ColoredVariancePreconditionedFineCascadeSampler(
            model,
            self._target_residual_rms,
            self._projected_base_rms,
            self._colored_blend,
            self._colored_channel_scales,
        )
        sampler.capture_evidence = True
        self._latest_fine_sampler = sampler
        return sampler


def load_colored_variance_preconditioned_fine_cascade_sampler(
    run_dir: str,
    checkpoint_name: str,
    model_config: dict,
    expected_checkpoint_sha256: str,
    expected_code_commit: str,
    expected_forecast_contract_sha256: str,
    device=None,
) -> ColoredVariancePreconditionedFineCascadeSampler:
    root = Path(run_dir)
    manifest_path = root / "fine_cascade_colored_base_manifest.json"
    primary_path = root / "fine_cascade_manifest.json"
    if not manifest_path.is_file() or not primary_path.is_file():
        raise ValueError("colored fine checkpoint lacks its sampler manifests")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    primary = json.loads(primary_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != 1
        or manifest.get("sampler")
        != "ColoredVariancePreconditionedFineCascadeSampler"
        or manifest.get("code_commit") != expected_code_commit
        or manifest.get("implementation_sha256") != _sha256(Path(__file__))
        or primary.get("base_parameterization") != COLORED_BASE_KIND
        or primary.get("colored_base_manifest") != manifest_path.name
        or primary.get("colored_base_manifest_sha256") != _sha256(manifest_path)
    ):
        raise ValueError("colored fine sampler manifest is incompatible")
    contract = validate_colored_base_contract(manifest.get("colored_base"))
    base = load_variance_preconditioned_fine_cascade_sampler(
        run_dir,
        checkpoint_name,
        model_config,
        expected_checkpoint_sha256,
        expected_code_commit,
        expected_forecast_contract_sha256,
        device=device,
        expected_base_parameterization=COLORED_BASE_KIND,
    )
    velocity_model = base.sampler.model
    projected = getattr(velocity_model, "model", None)
    if not isinstance(projected, ProjectedDetailModel):
        raise TypeError("colored fine loader did not recover the projected neural model")
    return ColoredVariancePreconditionedFineCascadeSampler(
        projected,
        base.target_residual_rms,
        base.projected_base_rms,
        contract["blend"],
        contract["channel_scales"],
    )
