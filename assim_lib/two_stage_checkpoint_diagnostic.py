"""Post-hoc numerical and scientific diagnostics for the frozen two-stage pilot.

This module performs no optimization.  It keeps the failed learning-pilot
directory immutable, verifies every source artifact by SHA-256, replays fixed
validation cases with identical noise, and decodes physical fields in float64
so an interior SIC value cannot be mistaken for the discrete archive cap solely
because of float32 rounding.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch
from torch.utils.data._utils.collate import default_collate

from .clearml_tracking import ClearMLTracker
from .config import TrainingConfig
from .data import build_dataset
from .model_io import load_sampler
from .structured_archive_audit import validate_bound_archive_audit
from .structured_joint_state import (
    StructuredDecodeSaturationError,
    canonical_mapping_sha256,
    decode_structured_joint_trajectory,
    validate_conditioning_normalization,
    validate_structured_state_stats,
)
from .structured_trajectory_evaluation import (
    make_structured_trajectory_figure,
    sample_structured_batch,
    structured_trajectory_metrics,
)


SCHEMA_VERSION = "two_stage_checkpoint_diagnostic_v1"
SOURCE_SCHEMA = "two_stage_native_learning_pilot_v1"
SOURCE_STATUS_SHA256 = "ec294f6269a032a031b6595f41dddac136bcd9755aad2199c5ae278d1f23c48a"
SOURCE_SHA256 = {
    "assimilation/last_model.pth": "8f5b6e028283dae3b0ef7b1b317f5d1bfba5ab1dd1c43d381bf682b5631576b2",
    "assimilation/ema_last_model.pth": "993b08d17c941f82edac5d2174a2ff1041ea88af764f5709154ac74ba30e44f3",
    "assimilation/metadata.json": "41ac61c0abfe40fb3b2210ebbf7b0e97b29af1f3e1696d30ae752282486b52b6",
    "assimilation/config.json": "626a47334a6382f40022548c9d56bebc4909f0a9f92f60eb04a4633bee1b7979",
    "assimilation/metrics.json": "297a02b3722dbc8223b8e90860bc20f363de177c18470d4620c3cabb8eb54dd3",
    "dynamics/last_model.pth": "03343b5255132ac4c706aa72a32345ca676345866ded2a1eb749e8764ff70209",
    "dynamics/ema_last_model.pth": "2961a026844d18f9a22ddd00fc3af0ba42ac038bdbcddb8b480c4e6fb779f139",
    "dynamics/metadata.json": "053a11fd2b654a5c220d7fc89c8bc8520cd76e0b619f9fecc24e390d8d76de5f",
    "dynamics/config.json": "70d3f0731368b3b5603734e00a66b169737cfc2e3e0ffcf3640ab60c94b69885",
    "dynamics/metrics.json": "0d4a489096207a06301cc3d6130ce74cd8cf2ea7218f1ba0c9c3b6f416bc289f",
}
SOLVER_TIMEPOINTS = (17, 33, 65)
METRIC_CASES = 4
METRIC_MEMBERS = 5
METRIC_TIMEPOINTS = 17


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().to(device="cpu").contiguous().numpy()
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> Any:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"unsafe or absent JSON artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _source_paths(source: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for role in ("assimilation", "dynamics"):
        run = source / role / "training" / "seed1701"
        for name in (
            "last_model.pth",
            "ema_last_model.pth",
            "metadata.json",
            "config.json",
            "metrics.json",
        ):
            paths[f"{role}/{name}"] = run / name
    return paths


def _verify_source(source: Path) -> dict[str, str]:
    status_path = source / "run_status.json"
    if _sha256(status_path) != SOURCE_STATUS_SHA256:
        raise ValueError("source run-status SHA-256 differs")
    status = _read_json(status_path)
    if (
        status.get("schema_version") != SOURCE_SCHEMA
        or status.get("status") != "failed"
        or status.get("phase") != "dynamics"
        or status.get("error") != "dynamics final EMA sampling diagnostic did not pass"
    ):
        raise ValueError("source learning-pilot terminal state differs")
    actual: dict[str, str] = {}
    for key, path in _source_paths(source).items():
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"unsafe or absent source artifact: {key}")
        actual[key] = _sha256(path)
        if actual[key] != SOURCE_SHA256[key]:
            raise ValueError(f"source SHA-256 differs: {key}")
    return actual


def _load_contract(
    source: Path, role: str
) -> tuple[dict[str, Any], TrainingConfig, Path, dict[str, Any], dict[str, Any]]:
    run = source / role / "training" / "seed1701"
    metadata = _read_json(run / "metadata.json")
    data = metadata.get("data_config")
    training = metadata.get("training_config")
    if not isinstance(data, dict) or not isinstance(training, dict):
        raise ValueError(f"{role} metadata lacks frozen data/training contracts")
    config = TrainingConfig.from_dict(training)
    if (
        config.training_objective != "structured_joint_state_flow"
        or config.trajectory_lead_days
        != ((0,) if role == "assimilation" else (3, 6, 9))
        or data.get("split_protocol") != "3dvar_main_200d"
        or data.get("valid", {}).get("obs_start_day") != "2022-01-01"
        or data.get("valid", {}).get("obs_end_day") != "2022-12-31"
    ):
        raise ValueError(f"{role} validation contract differs")
    stats = config.structured_state_stats
    validate_structured_state_stats(stats)
    validate_conditioning_normalization(data, stats)
    if stats.get("data_config_sha256") != canonical_mapping_sha256(data):
        raise ValueError(f"{role} embedded stats/data identity differs")
    audit = validate_bound_archive_audit(data, run / "metadata.json")
    return data, config, run, metadata, audit


def _validate_dataset_provenance(
    dataset,
    metadata: dict[str, Any],
    audit: dict[str, Any],
) -> None:
    dataset.validate_structured_sral_audit_contract(audit)
    expected = metadata.get("dataset_provenance", {}).get("valid")
    actual = dataset.provenance()
    if not isinstance(expected, dict):
        raise ValueError("source metadata lacks validation provenance")
    required = (
        "pair_manifest_sha256",
        "split",
        "num_pairs",
        "first_target_date",
        "last_target_date",
        "trajectory_lead_days",
    )
    for key in required:
        if expected.get(key) != actual.get(key):
            raise ValueError(f"current validation provenance differs: {key}")


def _case_indices(dataset, count: int) -> list[int]:
    if hasattr(dataset, "strided_case_indices"):
        indices = dataset.strided_case_indices(count, stride_days=30)
    else:
        indices = list(range(min(count, len(dataset))))
    if len(indices) != count:
        raise ValueError("frozen diagnostic case selection is incomplete")
    return [int(value) for value in indices]


def _batch(dataset, index: int, device: torch.device) -> dict[str, Any]:
    raw = default_collate([dataset[index]])
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in raw.items()
    }


def _seed(config: TrainingConfig, case_index: int, member_index: int, stream: int) -> int:
    return int(
        (
            int(config.validation_seed)
            + 1_000_003 * 4
            + 10_007 * (int(case_index) + 1)
            + 101 * (int(member_index) + 1)
            + int(stream)
        )
        % (2**63 - 1)
    )


def _noise(config: TrainingConfig, seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(
        (1, config.out_channels, *config.image_size),
        device=device,
        dtype=torch.float32,
        generator=generator,
    )


def _sample(
    sampler,
    batch: dict[str, Any],
    config: TrainingConfig,
    noise: torch.Tensor,
    timepoints: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    physical, latent = sample_structured_batch(
        sampler,
        batch,
        stats=config.structured_state_stats,
        size=config.image_size,
        num_timesteps=timepoints,
        device=device,
        method="rk4",
        rtol=config.sample_rtol,
        atol=config.sample_atol,
        initial_noise=noise,
        physical_dtype=torch.float64,
        return_latent=True,
    )
    fp32: dict[str, Any]
    try:
        decode_structured_joint_trajectory(
            latent,
            config.structured_state_stats,
            physical_dtype=torch.float32,
        )
    except StructuredDecodeSaturationError as error:
        fp32 = {
            "status": "rounded_out_of_open_interval",
            "error": str(error),
            "diagnostics": error.diagnostics,
        }
    else:
        fp32 = {"status": "representable"}
    if physical.dtype != torch.float64 or not torch.isfinite(physical).all():
        raise RuntimeError("float64 physical replay is not finite float64")
    return physical, latent, fp32


def _masked_trajectory_difference(
    left: torch.Tensor,
    right: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    channels_per_lead: int,
    channel_names: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    if left.shape != right.shape or left.ndim != 4:
        raise ValueError("paired trajectory tensors must have identical [B,C,H,W] shape")
    if channels_per_lead != len(channel_names) or left.shape[1] % channels_per_lead:
        raise ValueError("trajectory channel labels do not match tensor shape")
    valid = valid_mask[:, :1] > 0
    result: dict[str, dict[str, float]] = {}
    for lead in range(left.shape[1] // channels_per_lead):
        for offset, name in enumerate(channel_names):
            channel = lead * channels_per_lead + offset
            delta = (
                left[:, channel : channel + 1].to(torch.float64)
                - right[:, channel : channel + 1].to(torch.float64)
            ).abs()[valid]
            result[f"lead{lead}_{name}"] = {
                "mean_absolute": float(delta.mean().item()),
                "root_mean_square": float(delta.square().mean().sqrt().item()),
                "maximum_absolute": float(delta.max().item()),
            }
    return result


def _spatial_diagnostics(
    name: str,
    values: torch.Tensor,
    valid_mask: torch.Tensor,
) -> dict[str, float]:
    result: dict[str, float] = {}
    valid = valid_mask[:, :1] > 0
    if values.ndim == 5:
        values = values.mean(dim=1)
    for lead in range(values.shape[1] // 2):
        for offset, field in enumerate(("sic", "sit")):
            channel = values[:, 2 * lead + offset : 2 * lead + offset + 1]
            parts = []
            for axis in (-2, -1):
                left = [slice(None)] * 4
                right = [slice(None)] * 4
                mask_left = [slice(None)] * 4
                mask_right = [slice(None)] * 4
                left[axis] = slice(None, -1)
                right[axis] = slice(1, None)
                mask_left[axis] = slice(None, -1)
                mask_right[axis] = slice(1, None)
                edge = valid[tuple(mask_left)] & valid[tuple(mask_right)]
                parts.append((channel[tuple(left)] - channel[tuple(right)]).abs()[edge])
            result[f"{name}_lead{lead}_{field}_neighbor_abs"] = float(
                torch.cat(parts).to(torch.float64).mean().item()
            )
        occurrence = values[:, 2 * lead : 2 * lead + 1] > 0
        transitions = []
        for axis in (-2, -1):
            left = [slice(None)] * 4
            right = [slice(None)] * 4
            mask_left = [slice(None)] * 4
            mask_right = [slice(None)] * 4
            left[axis] = slice(None, -1)
            right[axis] = slice(1, None)
            mask_left[axis] = slice(None, -1)
            mask_right[axis] = slice(1, None)
            edge = valid[tuple(mask_left)] & valid[tuple(mask_right)]
            transitions.append((occurrence[tuple(left)] != occurrence[tuple(right)])[edge])
        result[f"{name}_lead{lead}_occurrence_edge_fraction"] = float(
            torch.cat(transitions).to(torch.float64).mean().item()
        )
    return result


def _save_rank_figure(metrics: dict[str, Any], output: Path, lead_days: list[int]) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(lead_days), 2, figsize=(12, 4 * len(lead_days)))
    if len(lead_days) == 1:
        axes = axes[None, :]
    for lead, day in enumerate(lead_days):
        for column, field in enumerate(("sic", "sit")):
            counts = metrics[f"lead{lead}_{field}_rank_counts"]
            total = sum(counts)
            frequency = [value / total for value in counts]
            axis = axes[lead, column]
            axis.bar(range(len(frequency)), frequency)
            axis.axhline(1.0 / len(frequency), color="black", linestyle="--")
            axis.set_title(f"d+{day} {field.upper()} fractional ranks")
            axis.set_xlabel("rank")
            axis.set_ylabel("frequency")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _run_verified(source: Path, output: Path) -> dict[str, Any]:
    if torch.cuda.device_count() != 1:
        raise RuntimeError("checkpoint diagnostic requires exactly one visible GPU")
    source_hashes = _verify_source(source)
    data, config, run_dir, metadata, audit = _load_contract(source, "dynamics")
    dataset = build_dataset(data, split="valid")
    _validate_dataset_provenance(dataset, metadata, audit)
    indices = _case_indices(dataset, METRIC_CASES)
    device = torch.device("cuda")
    batches = [_batch(dataset, index, device) for index in indices]
    noise_manifest: list[dict[str, Any]] = []
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    tracker = ClearMLTracker(
        project_name="sea_ice_two_stage",
        task_name="two_stage_frozen_checkpoint_diagnostic_v1",
        tags=["two-stage", "post-hoc", "no-training", "float64-decode", "visual-qc"],
        env_path="/home/.env",
    )
    tracker.connect("diagnostic_contract", {
        "source_schema": SOURCE_SCHEMA,
        "source_sha256": SOURCE_SHA256,
        "source_status_sha256": SOURCE_STATUS_SHA256,
        "optimizer_steps": 0,
        "test_2023_used": False,
        "neural_forward": "strict_float32_tf32_disabled",
        "physical_decode_and_scoring": "float64",
        "original_pilot_forward": "accelerate_bf16_not_bitwise_reproduced",
    })

    solver_results: dict[str, Any] = {}
    solver_samples: dict[str, dict[int, tuple[torch.Tensor, torch.Tensor]]] = {}
    for checkpoint_label, checkpoint_name in (
        ("raw", "last_model.pth"),
        ("ema", "ema_last_model.pth"),
    ):
        sampler = load_sampler(
            str(run_dir), checkpoint_name, dict(config.__dict__), device=device
        )
        by_timepoint: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        seed = _seed(config, indices[0], 0, 11)
        fixed_noise = _noise(config, seed, device)
        noise_manifest.append(
            {
                "purpose": f"solver_{checkpoint_label}",
                "case_index": indices[0],
                "member_index": 0,
                "seed": seed,
                "sha256": _tensor_sha256(fixed_noise),
            }
        )
        representations: dict[str, Any] = {}
        for timepoints in SOLVER_TIMEPOINTS:
            physical, latent, fp32 = _sample(
                sampler,
                batches[0],
                config,
                fixed_noise.clone(),
                timepoints,
                device,
            )
            by_timepoint[timepoints] = (physical.detach().cpu(), latent.detach().cpu())
            representations[str(timepoints)] = fp32
        solver_samples[checkpoint_label] = by_timepoint
        solver_results[checkpoint_label] = {
            "float32_decode": representations,
            "latent_17_vs_33": _masked_trajectory_difference(
                by_timepoint[17][1],
                by_timepoint[33][1],
                batches[0]["valid_mask"].detach().cpu(),
                channels_per_lead=4,
                channel_names=("occurrence", "cap", "sic_interior", "sit_positive"),
            ),
            "latent_33_vs_65": _masked_trajectory_difference(
                by_timepoint[33][1],
                by_timepoint[65][1],
                batches[0]["valid_mask"].detach().cpu(),
                channels_per_lead=4,
                channel_names=("occurrence", "cap", "sic_interior", "sit_positive"),
            ),
            "physical_17_vs_33": _masked_trajectory_difference(
                by_timepoint[17][0],
                by_timepoint[33][0],
                batches[0]["valid_mask"].detach().cpu(),
                channels_per_lead=2,
                channel_names=("sic", "sit"),
            ),
            "physical_33_vs_65": _masked_trajectory_difference(
                by_timepoint[33][0],
                by_timepoint[65][0],
                batches[0]["valid_mask"].detach().cpu(),
                channels_per_lead=2,
                channel_names=("sic", "sit"),
            ),
        }
        del sampler
        gc.collect()
        torch.cuda.empty_cache()

    ema_sampler = load_sampler(
        str(run_dir), "ema_last_model.pth", dict(config.__dict__), device=device
    )
    ensemble_cases = []
    metric_decode_replays: list[dict[str, Any]] = []
    causal_replay: dict[str, torch.Tensor] | None = None
    for case_position, (case_index, batch) in enumerate(zip(indices, batches, strict=True)):
        members = []
        for member_index in range(METRIC_MEMBERS):
            seed = _seed(config, case_index, member_index, 11)
            member_noise = _noise(config, seed, device)
            noise_manifest.append(
                {
                    "purpose": "ema_metric",
                    "case_position": case_position,
                    "case_index": case_index,
                    "member_index": member_index,
                    "seed": seed,
                    "sha256": _tensor_sha256(member_noise),
                }
            )
            physical, latent, fp32 = _sample(
                ema_sampler,
                batch,
                config,
                member_noise,
                METRIC_TIMEPOINTS,
                device,
            )
            metric_decode_replays.append(
                {
                    "case_position": case_position,
                    "case_index": case_index,
                    "member_index": member_index,
                    "seed": seed,
                    "float32_decode": fp32,
                    "float64_decode": "valid",
                }
            )
            if case_index == 0 and member_index == 1:
                causal_replay = {
                    "latent": latent.detach().cpu(),
                    "physical_float64": physical.detach().cpu(),
                    "initial_noise": member_noise.detach().cpu(),
                    "float32_decode": fp32,
                }
            members.append(physical.detach().cpu())
        ensemble_cases.append(torch.stack(members, dim=1))
    ensemble = torch.cat(ensemble_cases, dim=0)
    truth = torch.cat(
        [batch["structured_physical_truth"].detach().cpu() for batch in batches]
    )
    background = torch.cat(
        [batch["structured_physical_background"].detach().cpu() for batch in batches]
    )
    valid = torch.cat([batch["valid_mask"].detach().cpu() for batch in batches])
    lag0 = torch.cat([batch["structured_lag0_mask"].detach().cpu() for batch in batches])
    metrics = structured_trajectory_metrics(
        ensemble,
        truth,
        background,
        valid,
        lag0_mask=lag0,
        sic_cap=float(config.structured_state_stats["sic_cap"]),
        exclude_lag0_from_day0_scores=False,
    )
    spatial = {}
    spatial.update(_spatial_diagnostics("ensemble_mean", ensemble, valid))
    spatial.update(_spatial_diagnostics("member0", ensemble[:, 0], valid))
    spatial.update(_spatial_diagnostics("truth", truth, valid))
    spatial.update(_spatial_diagnostics("persistence", background, valid))

    visual = output / "visual_qc"
    visual.mkdir()
    for label in ("raw", "ema"):
        figure = make_structured_trajectory_figure(
            truth[0],
            background[0],
            solver_samples[label][65][0][0],
            valid[0],
            title=f"dynamics {label}, frozen case {indices[0]}, member 0, RK4-65",
            origin="upper",
            lead_days=config.trajectory_lead_days,
        )
        path = visual / f"dynamics_{label}_case00_member00_rk4_65.png"
        figure.savefig(path, dpi=180, bbox_inches="tight")
        import matplotlib.pyplot as plt

        plt.close(figure)
    rank_path = visual / "dynamics_ema_rank_histograms.png"
    _save_rank_figure(metrics, rank_path, config.trajectory_lead_days)
    torch.save(
        {
            "latent": solver_samples["ema"][65][1],
            "physical_float64": solver_samples["ema"][65][0],
            "truth": truth[:1],
            "persistence": background[:1],
            "valid_mask": valid[:1],
            "case_index": indices[0],
            "member_index": 0,
            "timepoints": 65,
        },
        output / "frozen_ema_case00_member00_rk4_65.pt",
    )
    if causal_replay is None:
        raise RuntimeError("original dynamics failure case/member was not replayed")
    torch.save(
        {
            **causal_replay,
            "case_index": 0,
            "member_index": 1,
            "member_seed": 4012933,
            "timepoints": METRIC_TIMEPOINTS,
            "source_failure": "epoch3 validation case=0 member=1",
        },
        output / "causal_replay_epoch3_case00_member01.pt",
    )

    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "optimizer_steps": 0,
        "test_2023_used": False,
        "source_artifact_sha256": source_hashes,
        "case_indices": indices,
        "ensemble_size": METRIC_MEMBERS,
        "metric_timepoints": METRIC_TIMEPOINTS,
        "solver_timepoints": list(SOLVER_TIMEPOINTS),
        "inference_precision": {
            "neural_forward_and_ode": "float32_tf32_disabled",
            "physical_unstandardization_decode_scoring": "float64",
            "comparison_to_original_pilot": "not_bitwise_original_bf16_endpoint",
        },
        "noise_manifest": noise_manifest,
        "metric_decode_replays": metric_decode_replays,
        "solver": solver_results,
        "metrics": metrics,
        "spatial": spatial,
        "visual_review_status": "pending_human_review",
    }
    result_path = output / "checkpoint_diagnostic.json"
    _atomic_json(result_path, result)
    tracker.upload_artifact("checkpoint_diagnostic", result_path)
    for path in sorted(visual.glob("*.png")):
        tracker.report_image("two_stage_checkpoint_diagnostic", path.stem, path, 0)
    tracker.close()
    return result


def run(source: Path, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    if output.is_symlink() or any(output.iterdir()):
        raise ValueError("diagnostic output must be a new empty non-symlink directory")
    status_path = output / "run_status.json"
    _atomic_json(status_path, {"schema_version": SCHEMA_VERSION, "status": "running"})
    try:
        result = _run_verified(source, output)
    except Exception as error:
        _atomic_json(
            status_path,
            {
                "schema_version": SCHEMA_VERSION,
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error)[:1000],
            },
        )
        raise
    _atomic_json(status_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    print(
        json.dumps(
            run(arguments.source_dir.resolve(), arguments.output_dir.resolve()),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
