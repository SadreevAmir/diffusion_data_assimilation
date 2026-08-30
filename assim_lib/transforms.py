from __future__ import annotations

import torch


def as_3d_tensor(x, dtype=torch.float32) -> torch.Tensor:
    tensor = torch.as_tensor(x, dtype=dtype)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3:
        raise ValueError(f"Expected [C,H,W] or [H,W], got shape {tuple(tensor.shape)}")
    return tensor


def select_channels(field, indices: list[int] | None) -> torch.Tensor:
    tensor = as_3d_tensor(field)
    if indices is None:
        return tensor
    index = torch.as_tensor(indices, dtype=torch.long)
    return tensor.index_select(0, index)


def channel_normalize(field: torch.Tensor, means, stds) -> torch.Tensor:
    means = torch.as_tensor(means, dtype=field.dtype, device=field.device).view(-1, 1, 1)
    stds = torch.as_tensor(stds, dtype=field.dtype, device=field.device).view(-1, 1, 1)
    return (field - means) / stds


def channel_denormalize(field: torch.Tensor, means, stds) -> torch.Tensor:
    means = torch.as_tensor(means, dtype=field.dtype, device=field.device).view(-1, 1, 1)
    stds = torch.as_tensor(stds, dtype=field.dtype, device=field.device).view(-1, 1, 1)
    return field * stds + means


def first_mask_channel(mask: torch.Tensor, reference: torch.Tensor, name: str) -> torch.Tensor:
    if mask.ndim != 4:
        raise ValueError(f"Expected {name} [B,C,H,W], got shape {tuple(mask.shape)}")
    if mask.shape[0] != reference.shape[0] or mask.shape[-2:] != reference.shape[-2:]:
        raise ValueError(
            f"Expected {name} batch/spatial shape {(reference.shape[0], *reference.shape[-2:])}, "
            f"got {(mask.shape[0], *mask.shape[-2:])}"
        )
    if mask.shape[1] < 1:
        raise ValueError(f"Expected {name} to have at least one channel")
    return mask[:, :1].to(device=reference.device, dtype=reference.dtype)


def make_conditioned_model_input(
    state: torch.Tensor,
    grid: torch.Tensor,
    background: torch.Tensor,
    background_mask: torch.Tensor | None,
    obs_values: torch.Tensor,
    obs_mask: torch.Tensor,
    water_mask: torch.Tensor,
) -> torch.Tensor:
    if background_mask is None:
        background_mask = torch.ones_like(background)
    if background_mask.shape != background.shape:
        raise ValueError(
            f"Expected background_mask shape {tuple(background.shape)}, got {tuple(background_mask.shape)}"
        )
    background_mask = background_mask.to(device=state.device, dtype=state.dtype)
    if water_mask.ndim != 4:
        raise ValueError(f"Expected water_mask [B,C,H,W], got shape {tuple(water_mask.shape)}")
    if water_mask.shape[0] != state.shape[0] or water_mask.shape[-2:] != state.shape[-2:]:
        raise ValueError(
            "Expected water_mask batch/spatial shape "
            f"{(state.shape[0], *state.shape[-2:])}, got "
            f"{(water_mask.shape[0], *water_mask.shape[-2:])}"
        )
    if water_mask.shape[1] < 1:
        raise ValueError("Expected water_mask to contain at least the physical water-domain channel")
    static_conditioning = water_mask.to(device=state.device, dtype=state.dtype)
    return torch.cat(
        [state, grid, background, background_mask, obs_values, obs_mask, static_conditioning],
        dim=1,
    )


def pad_to_size(field: torch.Tensor, image_size: tuple[int, int], fill_value=0.0) -> torch.Tensor:
    field = as_3d_tensor(field, dtype=field.dtype)
    channels, height, width = field.shape
    out_h, out_w = image_size
    if height > out_h or width > out_w:
        raise ValueError(f"Cannot pad shape {(height, width)} to smaller image_size {image_size}")

    if torch.is_tensor(fill_value):
        fill = fill_value.to(dtype=field.dtype, device=field.device).view(channels, 1, 1)
        out = fill.expand(channels, out_h, out_w).clone()
    elif isinstance(fill_value, (list, tuple)):
        fill = torch.as_tensor(fill_value, dtype=field.dtype, device=field.device).view(channels, 1, 1)
        out = fill.expand(channels, out_h, out_w).clone()
    else:
        out = torch.full((channels, out_h, out_w), float(fill_value), dtype=field.dtype, device=field.device)

    out[:, :height, :width] = field
    return out


def load_valid_mask(mask, channels: int, image_size: tuple[int, int]) -> torch.Tensor:
    mask = torch.as_tensor(mask, dtype=torch.float32)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).expand(channels, -1, -1)
    if mask.ndim != 3:
        raise ValueError(f"Expected 2D or 3D valid mask, got shape {tuple(mask.shape)}")
    if mask.shape[0] == 1 and channels > 1:
        mask = mask.expand(channels, -1, -1)
    if mask.shape[0] != channels:
        raise ValueError(f"Expected {channels} mask channels, got {mask.shape[0]}")
    return pad_to_size(mask, image_size, fill_value=0.0)


def prepare_model_field(
    field, indices, means, stds, padding_values, image_size
) -> tuple[torch.Tensor, torch.Tensor]:
    selected = select_channels(field, None if indices is None else list(indices))
    finite = torch.isfinite(selected).to(torch.float32)
    padding = torch.as_tensor(padding_values, dtype=selected.dtype).view(-1, 1, 1)
    selected = torch.where(torch.isfinite(selected), selected, padding)
    selected = pad_to_size(selected, tuple(image_size), fill_value=padding_values)
    finite = pad_to_size(finite, tuple(image_size), fill_value=0.0)
    return channel_normalize(selected, means, stds), finite


def make_observation_tensors(
    truth: torch.Tensor,
    spatial_mask: torch.Tensor,
    observed_channels: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    channels = truth.shape[0]
    spatial_mask = torch.as_tensor(spatial_mask, dtype=truth.dtype, device=truth.device)
    if spatial_mask.ndim == 3:
        spatial_mask = spatial_mask[0]
    if spatial_mask.ndim != 2:
        raise ValueError(f"Expected 2D observation mask, got shape {tuple(spatial_mask.shape)}")

    obs_mask = torch.zeros_like(truth)
    for channel in observed_channels:
        if channel < 0 or channel >= channels:
            raise ValueError(f"observed channel {channel} is outside [0, {channels})")
        obs_mask[channel] = spatial_mask

    obs_values = torch.where(obs_mask > 0, truth, torch.zeros_like(truth))
    return obs_values, obs_mask
