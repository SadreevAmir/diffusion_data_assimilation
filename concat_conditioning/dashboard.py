from __future__ import annotations

import numpy as np


def _as_numpy(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor
    return tensor.detach().cpu().numpy()


def _denormalize_channel(values, channel: int, means, stds):
    mean = float(means[channel]) if channel < len(means) else 0.0
    std = float(stds[channel]) if channel < len(stds) else 1.0
    return values * std + mean


def _clip_display(values, channel: int):
    if channel == 0:
        return np.clip(values, 0.0, 1.0)
    return values


def _channel_mask(mask, channel: int):
    if mask is None:
        return None
    if mask.ndim == 2:
        return mask > 0
    if mask.ndim == 3:
        if mask.shape[0] == 1:
            return mask[0] > 0
        if channel < mask.shape[0]:
            return mask[channel] > 0
    raise ValueError(f"Expected valid_mask [H,W] or [C,H,W], got shape {mask.shape}")


def make_background_condition_assim_figure(
    background,
    obs_values,
    obs_mask,
    assim,
    fields,
    means,
    stds,
    channels,
    title: str,
    panel_width: float = 7.0,
    panel_height: float = 6.0,
    valid_mask=None,
):
    import matplotlib.pyplot as plt

    background = _as_numpy(background)
    obs_values = _as_numpy(obs_values)
    obs_mask = _as_numpy(obs_mask) > 0
    assim = _as_numpy(assim)
    valid_mask = _as_numpy(valid_mask) if valid_mask is not None else None

    channels = list(channels)
    if not channels:
        channels = list(range(background.shape[0]))

    nrows = len(channels)
    fig, axes = plt.subplots(
        nrows,
        3,
        figsize=(3 * panel_width, nrows * panel_height),
        squeeze=False,
        constrained_layout=True,
    )
    fig.suptitle(title, fontsize=16)

    for row, channel in enumerate(channels):
        field_name = fields[channel] if channel < len(fields) else f"ch{channel}"
        bg = _clip_display(_denormalize_channel(background[channel], channel, means, stds), channel)
        an = _clip_display(_denormalize_channel(assim[channel], channel, means, stds), channel)
        cond = _clip_display(_denormalize_channel(obs_values[channel], channel, means, stds), channel)
        channel_valid = _channel_mask(valid_mask, channel)
        if channel_valid is not None:
            bg = np.where(channel_valid, bg, np.nan)
            an = np.where(channel_valid, an, np.nan)
        cond_mask = obs_mask[channel]
        if channel_valid is not None:
            cond_mask = cond_mask & channel_valid
        cond = np.where(cond_mask, cond, np.nan)

        finite_values = [arr[np.isfinite(arr)] for arr in (bg, an, cond)]
        finite_values = [arr for arr in finite_values if arr.size]
        if finite_values:
            combined = np.concatenate(finite_values)
            vmin = float(np.nanpercentile(combined, 1.0))
            vmax = float(np.nanpercentile(combined, 99.0))
            if vmin == vmax:
                vmin, vmax = None, None
        else:
            vmin, vmax = None, None

        cmap = "Blues_r" if channel == 0 else "viridis"
        panels = (("background", bg), ("condition", cond), ("assim", an))
        for col, (panel_name, values) in enumerate(panels):
            ax = axes[row, col]
            masked_values = np.ma.masked_invalid(values)
            image = ax.imshow(masked_values, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.set_title(f"{field_name} {panel_name}", fontsize=14)
            ax.axis("off")
            if panel_name == "condition" and not np.isfinite(values).any():
                ax.text(
                    0.5,
                    0.5,
                    "no direct obs",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    color="white",
                    fontsize=11,
                    bbox={"facecolor": "black", "alpha": 0.55, "pad": 4},
                )
            fig.colorbar(image, ax=ax, fraction=0.035, pad=0.015)

    return fig
