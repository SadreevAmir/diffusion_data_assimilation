"""Channel-scale-matched Gaussian base law for the fine cascade ablation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch

from .direct_dynamics_cascade import project_detail
from .direct_dynamics_cascade_fine import (
    CASCADE_FACTOR,
    DIRECT_OUTPUT_CHANNELS,
    FineCascadeDynamicsTrainer,
    FineCascadeSampler,
    _sha256,
    load_fine_cascade_sampler,
)
from .trainer import _atomic_json

BASE_LAW_KIND = "channel_scaled_projected_gaussian_v1"


def validate_channel_scales(values: Any) -> tuple[float, ...]:
    if not isinstance(values, list) or len(values) != DIRECT_OUTPUT_CHANNELS:
        raise ValueError("matched fine base law requires exactly six channel scales")
    scales = tuple(float(value) for value in values)
    if any(not math.isfinite(value) or value <= 0 or value >= 1 for value in scales):
        raise ValueError("matched fine base scales must be finite and lie strictly in (0,1)")
    return scales


def scaled_projected_noise(
    raw_noise: torch.Tensor,
    valid_mask: torch.Tensor,
    channel_scales: tuple[float, ...],
) -> torch.Tensor:
    scales = torch.as_tensor(channel_scales, dtype=raw_noise.dtype, device=raw_noise.device)
    scales = scales.reshape(1, DIRECT_OUTPUT_CHANNELS, 1, 1)
    return project_detail(raw_noise * scales, valid_mask, CASCADE_FACTOR)


def matched_residual_flow_pair(
    clean: torch.Tensor,
    raw_noise: torch.Tensor,
    mask: torch.Tensor,
    time: torch.Tensor,
    channel_scales: tuple[float, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if time.ndim != 1 or time.shape[0] != clean.shape[0]:
        raise ValueError("time must have one scalar per batch element")
    clean_detail = project_detail(clean, mask, CASCADE_FACTOR)
    base = scaled_projected_noise(raw_noise, mask, channel_scales)
    time_view = time.to(dtype=clean.dtype, device=clean.device).reshape(-1, 1, 1, 1)
    state = (1.0 - time_view) * clean_detail + time_view * base
    velocity = base - clean_detail
    valid = mask.expand_as(clean) > 0
    if not torch.isfinite(state[valid]).all() or not torch.isfinite(velocity[valid]).all():
        raise FloatingPointError("matched residual flow pair produced NaN/Inf on valid ocean")
    return state, velocity, clean_detail, base


class MatchedScaleFineCascadeSampler(FineCascadeSampler):
    def __init__(self, model, channel_scales: tuple[float, ...]):
        super().__init__(model)
        self.channel_scales = validate_channel_scales(list(channel_scales))

    def project_initial_noise(self, raw_noise: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        return scaled_projected_noise(raw_noise.float(), valid_mask.float(), self.channel_scales)

    @torch.no_grad()
    def sample_conditioned(self, **kwargs: Any) -> torch.Tensor:
        raw_noise = kwargs.get("initial_noise")
        if raw_noise is None:
            raise ValueError("matched fine sampler requires explicit unscaled Gaussian noise")
        result = super().sample_conditioned(**kwargs)
        if self.capture_evidence:
            self.evidence[-1]["unscaled_initial_noise"] = raw_noise.detach().cpu()
            self.evidence[-1]["base_noise_channel_scales"] = torch.tensor(self.channel_scales)
        return result


class MatchedScaleFineCascadeDynamicsTrainer(FineCascadeDynamicsTrainer):
    def __init__(self, *args, **kwargs):
        experiment = kwargs.get("experiment_config")
        if not isinstance(experiment, dict):
            raise TypeError("matched fine trainer requires experiment_config")
        law = experiment.get("fine_base_law")
        if not isinstance(law, dict) or law.get("kind") != BASE_LAW_KIND:
            raise ValueError("matched fine trainer requires its explicit base-law contract")
        self._base_channel_scales = validate_channel_scales(law.get("channel_scales"))
        source_audit = Path(str(law.get("source_audit_path", "")))
        expected_audit_sha256 = str(law.get("source_audit_sha256", ""))
        if not source_audit.is_file() or _sha256(source_audit) != expected_audit_sha256:
            raise ValueError("matched fine base-law source audit is missing or differs")
        audit = json.loads(source_audit.read_text(encoding="utf-8"))
        ordered = [audit["candidate_channel_scales"][name] for name in audit["channel_order"]]
        if validate_channel_scales(ordered) != self._base_channel_scales:
            raise ValueError("declared channel scales differ from the frozen source audit")
        self._base_law_contract = {
            "kind": BASE_LAW_KIND,
            "channel_order": list(audit["channel_order"]),
            "channel_scales": list(self._base_channel_scales),
            "source_audit_path": str(source_audit.resolve()),
            "source_audit_sha256": expected_audit_sha256,
            "source_case_ids_sha256": audit["selected_case_ids_sha256"],
            "source_split": audit["split"],
        }
        super().__init__(*args, **kwargs)
        _atomic_json(
            Path(self.output_dir) / "fine_cascade_matched_base_manifest.json",
            {
                "schema_version": 1,
                "sampler": "MatchedScaleFineCascadeSampler",
                "base_law": self._base_law_contract,
                "implementation_sha256": _sha256(Path(__file__)),
                "code_commit": self._fine_code_identity["git_commit"],
            },
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
        raw_noise = torch.randn(
            truth.shape, dtype=truth.dtype, device=truth.device, generator=generator
        )
        state, velocity, _, _ = matched_residual_flow_pair(
            truth, raw_noise, batch["valid_mask"][:, :1], timesteps, self._base_channel_scales
        )
        return state, velocity

    def _make_sampler(self, model) -> MatchedScaleFineCascadeSampler:
        sampler = MatchedScaleFineCascadeSampler(model, self._base_channel_scales)
        sampler.capture_evidence = True
        self._latest_fine_sampler = sampler
        return sampler

    def _augment_direct_diagnostic_payload(
        self, payload: dict[str, Any], *, step: int, label: str
    ) -> dict[str, Any]:
        result = super()._augment_direct_diagnostic_payload(payload, step=step, label=label)
        result["fine_base_law"] = self._base_law_contract
        return result


def load_matched_scale_fine_cascade_sampler(
    run_dir: str,
    checkpoint_name: str,
    model_config: dict,
    expected_checkpoint_sha256: str,
    expected_code_commit: str,
    expected_forecast_contract_sha256: str,
    device=None,
) -> MatchedScaleFineCascadeSampler:
    root = Path(run_dir)
    matched_path = root / "fine_cascade_matched_base_manifest.json"
    if not matched_path.is_file():
        raise ValueError("matched fine checkpoint lacks its base-law manifest")
    matched = json.loads(matched_path.read_text(encoding="utf-8"))
    if (
        matched.get("schema_version") != 1
        or matched.get("sampler") != "MatchedScaleFineCascadeSampler"
        or matched.get("code_commit") != expected_code_commit
        or matched.get("implementation_sha256") != _sha256(Path(__file__))
    ):
        raise ValueError("matched fine base-law manifest is incompatible")
    law = matched.get("base_law")
    if not isinstance(law, dict) or law.get("kind") != BASE_LAW_KIND:
        raise ValueError("matched fine manifest contains the wrong base law")
    scales = validate_channel_scales(law.get("channel_scales"))
    base = load_fine_cascade_sampler(
        run_dir,
        checkpoint_name,
        model_config,
        expected_checkpoint_sha256,
        expected_code_commit,
        expected_forecast_contract_sha256,
        device=device,
    )
    return MatchedScaleFineCascadeSampler(base.sampler.model, scales)
