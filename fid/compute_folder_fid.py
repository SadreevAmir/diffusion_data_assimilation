import argparse
import json
import os

from .inception_fid import (
    InceptionFeatureExtractor,
    RenderConfig,
    compute_feature_stats_for_folder,
    compute_frechet_distance,
    default_data_root,
    default_inception_weights_path,
    default_stats_json_path,
    load_channel_stats,
    load_saved_stats,
    save_stats,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute Inception-FID between two folders of .npy sea-ice fields. "
            "Each field is rendered to RGB through a fixed matplotlib colormap before feature extraction."
        )
    )
    parser.add_argument("--real-dir", required=True, help="Folder with real .npy samples")
    parser.add_argument("--fake-dir", help="Folder with generated .npy samples")
    parser.add_argument("--data-root", default=default_data_root(), help="Base sea-ice data root")
    parser.add_argument("--stats-json", help="Path to stats.json. Required only if --input-normalized is used")
    parser.add_argument("--device", default=None, help="cuda, mps or cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-real", type=int, default=None)
    parser.add_argument("--max-fake", type=int, default=None)
    parser.add_argument("--resize-to", type=int, default=299, help="Inception input size")
    parser.add_argument("--weights-path", help="Optional local InceptionV3 weights .pth")
    parser.add_argument("--real-stats-in", help="Load cached real stats from .npz")
    parser.add_argument("--real-stats-out", help="Save real stats to .npz")
    parser.add_argument("--fake-stats-out", help="Save fake stats to .npz")
    parser.add_argument("--output-json", help="Optional path to save result JSON")
    parser.add_argument("--only-real-stats", action="store_true", help="Only compute and store real stats")

    parser.add_argument(
        "--render-mode",
        choices=("channel0", "channel1", "blend"),
        default="channel0",
        help="How to render a 2-channel field into RGB before Inception",
    )
    parser.add_argument("--channel0-cmap", default="viridis", help="Colormap for channel 0 rendering")
    parser.add_argument("--channel1-cmap", default="magma", help="Colormap for channel 1 rendering")
    parser.add_argument("--channel0-vmin", type=float, default=0.0)
    parser.add_argument("--channel0-vmax", type=float, default=1.0)
    parser.add_argument("--channel1-vmin", type=float, default=0.0)
    parser.add_argument("--channel1-vmax", type=float, default=1.0)
    parser.add_argument(
        "--blend-alpha",
        type=float,
        default=0.5,
        help="Weight of channel0 render in blend mode; channel1 gets (1 - alpha)",
    )
    parser.add_argument(
        "--input-normalized",
        action="store_true",
        help="Set this if .npy files are normalized and must be denormalized via stats.json before rendering",
    )
    return parser.parse_args()


def _build_render_config(args) -> tuple[RenderConfig, str | None]:
    stats_json = args.stats_json or default_stats_json_path(args.data_root)
    channel_mean = None
    channel_std = None
    if args.input_normalized:
        channel_mean, channel_std = load_channel_stats(stats_json)

    return (
        RenderConfig(
            mode=args.render_mode,
            channel0_cmap=args.channel0_cmap,
            channel1_cmap=args.channel1_cmap,
            channel0_vmin=args.channel0_vmin,
            channel0_vmax=args.channel0_vmax,
            channel1_vmin=args.channel1_vmin,
            channel1_vmax=args.channel1_vmax,
            blend_alpha=args.blend_alpha,
            input_normalized=args.input_normalized,
            channel_mean=channel_mean,
            channel_std=channel_std,
        ),
        stats_json if args.input_normalized else None,
    )


def _render_meta(args, stats_json_path: str | None) -> dict:
    return {
        "render_mode": args.render_mode,
        "channel0_cmap": args.channel0_cmap,
        "channel1_cmap": args.channel1_cmap,
        "channel0_vmin": args.channel0_vmin,
        "channel0_vmax": args.channel0_vmax,
        "channel1_vmin": args.channel1_vmin,
        "channel1_vmax": args.channel1_vmax,
        "blend_alpha": args.blend_alpha,
        "input_normalized": args.input_normalized,
        "stats_json": os.path.abspath(stats_json_path) if stats_json_path else None,
    }


def main():
    args = parse_args()

    if not args.only_real_stats and not args.fake_dir:
        raise SystemExit("--fake-dir is required unless --only-real-stats is used")

    render_config, stats_json_path = _build_render_config(args)
    weights_path = args.weights_path or default_inception_weights_path()

    extractor = InceptionFeatureExtractor(
        device=args.device,
        resize_to=args.resize_to,
        weights_path=weights_path,
    )

    if args.real_stats_in:
        real_stats, real_meta = load_saved_stats(args.real_stats_in)
    else:
        real_stats = compute_feature_stats_for_folder(
            folder=args.real_dir,
            extractor=extractor,
            render_config=render_config,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_items=args.max_real,
            desc="real features",
        )
        real_meta = {
            "folder": os.path.abspath(args.real_dir),
            "weights_path": os.path.abspath(weights_path) if weights_path else None,
            "resize_to": args.resize_to,
            "render": _render_meta(args, stats_json_path),
        }
        if args.real_stats_out:
            save_stats(args.real_stats_out, real_stats, meta=real_meta)

    if args.only_real_stats:
        payload = {
            "mode": "real_stats_only",
            "real_count": real_stats.count,
            "feature_dim": real_stats.feature_dim,
            "real_stats_out": args.real_stats_out,
            "real_meta": real_meta,
        }
        print(json.dumps(payload, indent=2))
        if args.output_json:
            with open(args.output_json, "w") as f:
                json.dump(payload, f, indent=2)
        return

    fake_stats = compute_feature_stats_for_folder(
        folder=args.fake_dir,
        extractor=extractor,
        render_config=render_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_items=args.max_fake,
        desc="fake features",
    )
    if args.fake_stats_out:
        save_stats(
            args.fake_stats_out,
            fake_stats,
            meta={
                "folder": os.path.abspath(args.fake_dir),
                "weights_path": os.path.abspath(weights_path) if weights_path else None,
                "resize_to": args.resize_to,
                "render": _render_meta(args, stats_json_path),
            },
        )

    fid_score = compute_frechet_distance(real_stats, fake_stats)
    payload = {
        "fid": fid_score,
        "real_count": real_stats.count,
        "fake_count": fake_stats.count,
        "feature_dim": real_stats.feature_dim,
        "real_dir": os.path.abspath(args.real_dir),
        "fake_dir": os.path.abspath(args.fake_dir),
        "weights_path": os.path.abspath(weights_path) if weights_path else None,
        "resize_to": args.resize_to,
        "render": _render_meta(args, stats_json_path),
        "real_stats_in": args.real_stats_in,
        "real_stats_out": args.real_stats_out,
        "fake_stats_out": args.fake_stats_out,
    }

    print(json.dumps(payload, indent=2))
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
