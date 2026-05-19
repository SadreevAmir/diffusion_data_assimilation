from __future__ import annotations

import numpy as np


def _as_numpy(tensor):
    return tensor.detach().cpu().numpy()


def _denormalize_channel(values, channel: int, means, stds):
    mean = float(means[channel]) if channel < len(means) else 0.0
    std = float(stds[channel]) if channel < len(stds) else 1.0
    return values * std + mean


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
):
    import matplotlib.pyplot as plt

    background = _as_numpy(background)
    obs_values = _as_numpy(obs_values)
    obs_mask = _as_numpy(obs_mask) > 0
    assim = _as_numpy(assim)

    channels = list(channels)
    if not channels:
        channels = list(range(background.shape[0]))

    nrows = len(channels)
    fig, axes = plt.subplots(nrows, 3, figsize=(12, 3.4 * nrows), squeeze=False)
    fig.suptitle(title)

    for row, channel in enumerate(channels):
        field_name = fields[channel] if channel < len(fields) else f"ch{channel}"
        bg = _denormalize_channel(background[channel], channel, means, stds)
        an = _denormalize_channel(assim[channel], channel, means, stds)
        cond = _denormalize_channel(obs_values[channel], channel, means, stds)
        cond = np.where(obs_mask[channel], cond, np.nan)

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
            image = ax.imshow(masked_values, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(f"{field_name} {panel_name}")
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
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig
