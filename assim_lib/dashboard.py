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


def _empty_display_value(channel: int):
    return 0.0 if channel == 0 else np.nan


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


def _display_panels(background, obs_values, obs_mask, assim, channel: int, means, stds,
                    valid_mask=None, water_mask=None, truth=None):
    bg = _clip_display(_denormalize_channel(background[channel], channel, means, stds), channel)
    an = _clip_display(_denormalize_channel(assim[channel], channel, means, stds), channel)
    cond = _clip_display(_denormalize_channel(obs_values[channel], channel, means, stds), channel)
    target = None
    if truth is not None:
        target = _clip_display(_denormalize_channel(truth[channel], channel, means, stds), channel)
    display_mask = water_mask if water_mask is not None else valid_mask
    channel_water = _channel_mask(display_mask, channel)
    if channel_water is not None:
        bg = np.where(channel_water, bg, np.nan)
        an = np.where(channel_water, an, np.nan)
        if target is not None:
            target = np.where(channel_water, target, np.nan)
    cond_mask = obs_mask[channel]
    if channel_water is not None:
        cond_mask = cond_mask & channel_water
    condition_has_obs = bool(np.any(cond_mask))
    cond = np.where(cond_mask, cond, _empty_display_value(channel))
    if channel_water is not None:
        cond = np.where(channel_water, cond, np.nan)

    mask_display = cond_mask.astype(np.float32)
    if channel_water is not None:
        mask_display = np.where(channel_water, mask_display, np.nan)

    if channel == 0:
        vmin, vmax = 0.0, 1.0
    else:
        display_values = [bg, an, cond]
        if target is not None:
            display_values.append(target)
        finite_values = [arr[np.isfinite(arr)] for arr in display_values]
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
    return bg, cond, mask_display, an, target, condition_has_obs, vmin, vmax, cmap


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
    water_mask=None,
    truth=None,
):
    import matplotlib.pyplot as plt

    background = _as_numpy(background)
    obs_values = _as_numpy(obs_values)
    obs_mask = _as_numpy(obs_mask) > 0
    assim = _as_numpy(assim)
    truth = _as_numpy(truth) if truth is not None else None
    valid_mask = _as_numpy(valid_mask) if valid_mask is not None else None
    water_mask = _as_numpy(water_mask) if water_mask is not None else None

    channels = list(channels)
    if not channels:
        channels = list(range(background.shape[0]))

    nrows = len(channels)
    ncols = 5 if truth is not None else 4
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * panel_width, nrows * panel_height),
        squeeze=False,
        constrained_layout=True,
    )
    fig.suptitle(title, fontsize=16)

    for row, channel in enumerate(channels):
        field_name = fields[channel] if channel < len(fields) else f"ch{channel}"
        bg, cond, mask_display, an, target, condition_has_obs, vmin, vmax, cmap = _display_panels(
            background,
            obs_values,
            obs_mask,
            assim,
            channel,
            means,
            stds,
            valid_mask=valid_mask,
            water_mask=water_mask,
            truth=truth,
        )
        panels = [
            ("background", bg, cmap, vmin, vmax),
            ("condition", cond, cmap, vmin, vmax),
            ("mask", mask_display, "gray", 0.0, 1.0),
            ("assim", an, cmap, vmin, vmax),
        ]
        if target is not None:
            panels.append(("truth", target, cmap, vmin, vmax))
        for col, (panel_name, values, panel_cmap, panel_vmin, panel_vmax) in enumerate(panels):
            ax = axes[row, col]
            masked_values = np.ma.masked_invalid(values)
            image = ax.imshow(
                masked_values, cmap=panel_cmap, vmin=panel_vmin, vmax=panel_vmax, interpolation="nearest"
            )
            ax.set_title(f"{field_name} {panel_name}", fontsize=14)
            ax.axis("off")
            if panel_name in ("condition", "mask") and not condition_has_obs:
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


def make_multi_case_background_condition_assim_figure(
    cases,
    fields,
    means,
    stds,
    channels,
    title: str,
    panel_width: float = 3.6,
    panel_height: float = 2.5,
):
    import matplotlib.pyplot as plt

    cases = list(cases)
    if not cases:
        raise ValueError("At least one dashboard case is required")

    channels = list(channels)
    first_background = _as_numpy(cases[0]["background"])
    if not channels:
        channels = list(range(first_background.shape[0]))

    analysis_columns = [("both", "assim")]
    optional_columns = [
        ("background only", "assim_background_only"),
        ("obs only / bg base", "assim_observation_only"),
        ("neither / bg base", "assim_neither"),
    ]
    analysis_columns.extend(
        (title, key) for title, key in optional_columns if all(case.get(key) is not None for case in cases)
    )
    nrows = len(cases) * len(channels)
    has_truth = all(case.get("truth") is not None for case in cases)
    ncols = 3 + len(analysis_columns) + int(has_truth)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * panel_width, nrows * panel_height),
        squeeze=False,
        constrained_layout=True,
    )
    fig.suptitle(title, fontsize=16)
    column_titles = ["background", "condition", "mask", *[title for title, _ in analysis_columns]]
    if has_truth:
        column_titles.append("truth")

    for case_row, case in enumerate(cases):
        background = _as_numpy(case["background"])
        obs_values = _as_numpy(case["obs_values"])
        obs_mask = _as_numpy(case["obs_mask"]) > 0
        assim = _as_numpy(case["assim"])
        truth = _as_numpy(case["truth"]) if has_truth else None
        valid_mask = _as_numpy(case["valid_mask"]) if case.get("valid_mask") is not None else None
        water_mask = _as_numpy(case["water_mask"]) if case.get("water_mask") is not None else None
        case_idx = int(case.get("case_idx", case_row))
        case_label = case.get("case_label") or f"case {case_idx:04d}"

        for channel_row, channel in enumerate(channels):
            row = case_row * len(channels) + channel_row
            field_name = fields[channel] if channel < len(fields) else f"ch{channel}"
            bg, cond, mask_display, an, target, condition_has_obs, vmin, vmax, cmap = _display_panels(
                background,
                obs_values,
                obs_mask,
                assim,
                channel,
                means,
                stds,
                valid_mask=valid_mask,
                water_mask=water_mask,
                truth=truth,
            )
            panels = [
                ("background", bg, cmap, vmin, vmax),
                ("condition", cond, cmap, vmin, vmax),
                ("mask", mask_display, "gray", 0.0, 1.0),
                (analysis_columns[0][0], an, cmap, vmin, vmax),
            ]
            for panel_name, key in analysis_columns[1:]:
                analysis = _as_numpy(case[key])
                values = _clip_display(_denormalize_channel(analysis[channel], channel, means, stds), channel)
                channel_water = _channel_mask(water_mask if water_mask is not None else valid_mask, channel)
                if channel_water is not None:
                    values = np.where(channel_water, values, np.nan)
                panels.append((panel_name, values, cmap, vmin, vmax))
            if target is not None:
                panels.append(("truth", target, cmap, vmin, vmax))
            for col, (panel_name, values, panel_cmap, panel_vmin, panel_vmax) in enumerate(panels):
                ax = axes[row, col]
                masked_values = np.ma.masked_invalid(values)
                ax.imshow(
                    masked_values, cmap=panel_cmap, vmin=panel_vmin, vmax=panel_vmax, interpolation="nearest"
                )
                if row == 0:
                    ax.set_title(column_titles[col], fontsize=12)
                if col == 0:
                    ax.text(
                        -0.06,
                        0.5,
                        f"{case_label}\n{field_name}",
                        ha="right",
                        va="center",
                        transform=ax.transAxes,
                        fontsize=10,
                        clip_on=False,
                    )
                ax.axis("off")
                if panel_name in ("condition", "mask") and not condition_has_obs:
                    ax.text(
                        0.5,
                        0.5,
                        "no direct obs",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        color="white",
                        fontsize=9,
                        bbox={"facecolor": "black", "alpha": 0.55, "pad": 4},
                    )

    return fig
