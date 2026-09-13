"""CPU-integrable mask-only CFM runner for the IDEA-F1 matched A/B gate.

The candidate and control have identical parameters.  In the candidate, one
head is restricted to a deterministic neighbourhood of the d0 ice edge and
the other is an unrestricted source/sink head.  In the control both heads are
unrestricted.  Every spatial tensor is zeroed on inactive land before a
convolution and after every sampling step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .direct_dynamics_mixed_support_admission import (
    binary_dequantized_logit,
    coarse_occurrence,
    decode_binary_dequantized_logit,
)
from .transforms import channel_denormalize


MASK_LEADS = 3
CONDITION_CHANNELS = 15


def _binary_mask(value: torch.Tensor, *, label: str) -> torch.Tensor:
    if value.ndim != 4 or value.shape[1] != 1:
        raise ValueError(f"{label} must have shape [batch,1,y,x]")
    if not torch.isfinite(value).all() or torch.any((value != 0) & (value != 1)):
        raise ValueError(f"{label} must be finite and binary")
    return value


def _positive_case_denominator(valid: torch.Tensor, channels: int) -> torch.Tensor:
    count = valid.reshape(valid.shape[0], -1).sum(dim=1) * channels
    if not torch.isfinite(count).all() or torch.any(count <= 0):
        raise ValueError("each case must contain at least one active ocean cell")
    return count


def d0_edge_neighbourhood(d0_occurrence: torch.Tensor, valid: torch.Tensor, radius: int = 2) -> torch.Tensor:
    """Deterministic front support; empty/full fields have no artificial edge."""
    d0 = _binary_mask(d0_occurrence, label="d0_occurrence")
    ocean = _binary_mask(valid, label="valid")
    d0 = torch.where(ocean > 0, d0, torch.zeros_like(d0))
    edge = torch.zeros_like(d0)
    horizontal = (d0[..., :, 1:] != d0[..., :, :-1]) & (
        (ocean[..., :, 1:] > 0) & (ocean[..., :, :-1] > 0)
    )
    vertical = (d0[..., 1:, :] != d0[..., :-1, :]) & (
        (ocean[..., 1:, :] > 0) & (ocean[..., :-1, :] > 0)
    )
    edge[..., :, 1:] = torch.maximum(edge[..., :, 1:], horizontal.to(edge.dtype))
    edge[..., :, :-1] = torch.maximum(edge[..., :, :-1], horizontal.to(edge.dtype))
    edge[..., 1:, :] = torch.maximum(edge[..., 1:, :], vertical.to(edge.dtype))
    edge[..., :-1, :] = torch.maximum(edge[..., :-1, :], vertical.to(edge.dtype))
    if radius:
        edge = F.max_pool2d(edge, 2 * radius + 1, stride=1, padding=radius)
    return edge * ocean


class CompactMaskVelocity(nn.Module):
    """Parameter-matched candidate/control velocity field."""

    def __init__(self, *, hidden_channels: int = 16, front_source: bool):
        super().__init__()
        self.front_source = bool(front_source)
        inputs = MASK_LEADS + CONDITION_CHANNELS + 1 + 2
        self.stem = nn.Sequential(
            nn.Conv2d(inputs, hidden_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.SiLU(),
        )
        self.front_head = nn.Conv2d(hidden_channels, MASK_LEADS, 1)
        self.source_head = nn.Conv2d(hidden_channels, MASK_LEADS, 1)

    def forward(
        self,
        state: torch.Tensor,
        time: torch.Tensor,
        condition: torch.Tensor,
        d0_occurrence: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        ocean = _binary_mask(valid, label="valid")
        d0 = _binary_mask(d0_occurrence, label="d0_occurrence")
        if state.ndim != 4 or state.shape[1] != MASK_LEADS:
            raise ValueError("state must contain the three mask leads")
        if condition.ndim != 4 or condition.shape[1] != CONDITION_CHANNELS:
            raise ValueError("condition must contain fifteen causal channels")
        if state.shape[0] != condition.shape[0] or state.shape[-2:] != condition.shape[-2:]:
            raise ValueError("state and condition shapes differ")
        if time.ndim != 1 or time.shape[0] != state.shape[0]:
            raise ValueError("time must have one value per case")
        if not torch.isfinite(time).all() or torch.any((time < 0) | (time > 1)):
            raise ValueError("time must lie in [0,1]")
        _positive_case_denominator(ocean, MASK_LEADS)
        support = ocean > 0
        clean_state = torch.where(support, state, torch.zeros_like(state))
        clean_condition = torch.where(support, condition, torch.zeros_like(condition))
        clean_d0 = torch.where(support, d0, torch.zeros_like(d0))
        yy = torch.linspace(-1, 1, state.shape[-2], dtype=state.dtype, device=state.device)
        xx = torch.linspace(-1, 1, state.shape[-1], dtype=state.dtype, device=state.device)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
        grid = torch.stack((grid_x, grid_y)).unsqueeze(0).expand(state.shape[0], -1, -1, -1)
        grid = torch.where(support, grid, torch.zeros_like(grid))
        time_map = time[:, None, None, None].expand(-1, 1, *state.shape[-2:])
        time_map = torch.where(support, time_map, torch.zeros_like(time_map))
        features = self.stem(torch.cat((clean_state, clean_condition, time_map, grid), dim=1))
        front = self.front_head(features)
        source = self.source_head(features)
        if self.front_source:
            front = front * d0_edge_neighbourhood(clean_d0, ocean)
        prediction = front + source
        return torch.where(support, prediction, torch.zeros_like(prediction))


def build_matched_models(*, seed: int, hidden_channels: int = 16) -> tuple[CompactMaskVelocity, CompactMaskVelocity]:
    """Return candidate/control with identical initialization and parameter count."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        candidate = CompactMaskVelocity(hidden_channels=hidden_channels, front_source=True)
        torch.manual_seed(seed)
        control = CompactMaskVelocity(hidden_channels=hidden_channels, front_source=False)
    if sum(p.numel() for p in candidate.parameters()) != sum(p.numel() for p in control.parameters()):
        raise RuntimeError("matched models differ in parameter count")
    for left, right in zip(candidate.parameters(), control.parameters(), strict=True):
        if not torch.equal(left, right):
            raise RuntimeError("matched models differ at initialization")
    return candidate, control


@dataclass(frozen=True)
class CfmBatch:
    target: torch.Tensor
    condition: torch.Tensor
    d0_occurrence: torch.Tensor
    valid: torch.Tensor


def masked_cfm_loss(
    model: CompactMaskVelocity,
    batch: CfmBatch,
    *,
    time: torch.Tensor,
    noise: torch.Tensor,
    uniform: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-case normalized dequantized CFM loss, with no hidden STE."""
    ocean = _binary_mask(batch.valid, label="valid")
    if batch.target.shape != noise.shape or batch.target.shape != uniform.shape:
        raise ValueError("target, noise and dequantization filler shapes differ")
    clean = binary_dequantized_logit(batch.target, uniform)
    support = ocean > 0
    clean = torch.where(support, clean, torch.zeros_like(clean))
    noise = torch.where(support, noise, torch.zeros_like(noise))
    view = time[:, None, None, None]
    state = (1 - view) * clean + view * noise
    velocity = noise - clean
    prediction = model(state, time, batch.condition, batch.d0_occurrence, ocean)
    numerator = ((prediction - velocity).square() * ocean).reshape(state.shape[0], -1).sum(dim=1)
    denominator = _positive_case_denominator(ocean, state.shape[1])
    per_case = numerator / denominator
    return per_case.mean(), per_case


@torch.no_grad()
def sample_masks(
    model: CompactMaskVelocity,
    *,
    condition: torch.Tensor,
    d0_occurrence: torch.Tensor,
    valid: torch.Tensor,
    noise: torch.Tensor,
    steps: int = 4,
) -> torch.Tensor:
    if steps <= 0:
        raise ValueError("steps must be positive")
    ocean = _binary_mask(valid, label="valid")
    _positive_case_denominator(ocean, MASK_LEADS)
    state = torch.where(ocean > 0, noise, torch.zeros_like(noise))
    dt = -1.0 / steps
    for index in range(steps):
        time = torch.full((state.shape[0],), 1.0 - index / steps, dtype=state.dtype, device=state.device)
        state = state + dt * model(state, time, condition, d0_occurrence, ocean)
        state = torch.where(ocean > 0, state, torch.zeros_like(state))
    decoded = decode_binary_dequantized_logit(state)
    return torch.where(ocean > 0, decoded, torch.zeros_like(decoded))


def coarse_mask_batch_from_item(item: dict[str, Any], means: list[float], stds: list[float], factor: int = 4) -> CfmBatch:
    """Construct the exact 80x64 train input without future conditioning."""
    valid_native = item["valid_mask"][:1].unsqueeze(0).float()
    condition_native = item["structured_conditioning"].unsqueeze(0).float()
    condition_native = torch.where(valid_native > 0, condition_native, torch.zeros_like(condition_native))
    condition = F.avg_pool2d(condition_native, factor, stride=factor)
    valid = (F.avg_pool2d(valid_native, factor, stride=factor) > 0).float()
    condition = torch.where(valid > 0, condition, torch.zeros_like(condition))
    initial = channel_denormalize(condition_native[:, :2], means, stds)
    d0, _ = coarse_occurrence(initial[:, :1], valid_native, factor)
    future = item["structured_physical_truth"].unsqueeze(0).float()
    targets = [coarse_occurrence(future[:, 2*i:2*i+1], valid_native, factor)[0] for i in range(MASK_LEADS)]
    return CfmBatch(target=torch.cat(targets, dim=1), condition=condition, d0_occurrence=d0, valid=valid)
