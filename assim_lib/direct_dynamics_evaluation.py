"""Paired validation evaluation for frozen direct-dynamics checkpoints.

The evaluator deliberately uses only the validation split.  It compares fixed
EMA checkpoints with identical cases and initial noise, scores raw (unclipped)
physical fields, and keeps projection out of all primary metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import torch

from .clearml_tracking import ClearMLTracker
from .data import build_dataset
from .direct_dynamics_training import (
    DIRECT_LEADS,
    DIRECT_OUTPUT_CHANNELS,
    _repeat_field_stats,
    validate_direct_dataset,
)
from .model_io import load_sampler
from .structured_trajectory_evaluation import (
    _fair_crps,
    fractional_rank_counts,
    make_structured_trajectory_figure,
)
from .trainer import _atomic_json
from .transforms import channel_denormalize


EXPECTED_INPUT_SHA256 = {
    "metadata.json": "73c530dc669b539237bc7b295a352ef6093f0936b970ecb9c96b321bc7745da0",
    "config.json": "898b5414cb921569b33dbd6bab89df01b0952742ed41648ab948f4ff9a5b05a6",
    "epoch_snapshots/epoch_0006/ema_last_model.pth": "9b8bb6954b7a19179aee2e58b0c6cdddfcb8b489b1e29d6f503086c704fa0b1b",
    "epoch_snapshots/epoch_0008/ema_last_model.pth": "daeab3e0ba6590f44d05ca3114cf7c91df898d481d27404664ba0a3aff7fc28e",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_finite_scalars(value: Any, path: str = "result") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _require_finite_scalars(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _require_finite_scalars(child, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise FloatingPointError(f"non-finite scalar metric at {path}: {value}")


def _selected_indices(length: int, count: int) -> list[int]:
    if count < 2 or length < count:
        raise ValueError("paired evaluation requires at least two distinct validation cases")
    # Integer arithmetic gives deterministic, endpoint-inclusive seasonal coverage.
    result = [round(index * (length - 1) / (count - 1)) for index in range(count)]
    if len(set(result)) != count:
        raise RuntimeError("validation case selection produced duplicate indices")
    return result


def _tensor_batch(item: dict[str, Any], members: int, device: torch.device) -> dict[str, torch.Tensor]:
    result = {}
    for key, value in item.items():
        if torch.is_tensor(value):
            result[key] = value.unsqueeze(0).repeat(members, *([1] * value.ndim)).to(device)
    return result


def _initial_noise(
    case_index: int,
    members: int,
    image_size: tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    values = []
    for member in range(members):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(314159 + 1000003 * case_index + 1009 * member)
        values.append(torch.randn((DIRECT_OUTPUT_CHANNELS, *image_size), generator=generator))
    return torch.stack(values).to(device)


@torch.no_grad()
def _sample_checkpoint(
    run_dir: Path,
    checkpoint: str,
    training_config: dict[str, Any],
    dataset,
    indices: list[int],
    members: int,
    steps: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    sampler = load_sampler(str(run_dir), checkpoint, training_config, device=device)
    means = _repeat_field_stats(dataset.means)
    stds = _repeat_field_stats(dataset.stds)
    sampled, truths, persistence, masks, identities = [], [], [], [], []
    image_size = tuple(int(value) for value in training_config["image_size"])
    for order, dataset_index in enumerate(indices):
        item = dataset[dataset_index]
        batch = _tensor_batch(item, members, device)
        valid = batch["valid_mask"][:, :1]
        noise = _initial_noise(order, members, image_size, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            normalized = sampler.sample_conditioned(
                background=batch["background"],
                background_mask=torch.ones_like(batch["background"]),
                obs_values=batch["obs_values"],
                obs_mask=batch["obs_mask"],
                water_mask=batch["water_mask"],
                size=image_size,
                num_timesteps=steps,
                device=device,
                method="rk4",
                rtol=1e-5,
                atol=1e-6,
                start_mode="noise",
                initial_noise=noise,
                sample_target="state",
                model_conditioning=batch["structured_conditioning"],
                state_channels=DIRECT_OUTPUT_CHANNELS,
                end_time=0.0,
                state_mask=valid.expand(-1, DIRECT_OUTPUT_CHANNELS, -1, -1),
            )
        physical = channel_denormalize(normalized.float(), means, stds).cpu()
        sampled.append(physical)
        truths.append(item["structured_physical_truth"].float())
        persistence.append(item["structured_physical_background"].float())
        masks.append(item["valid_mask"][:1].float())
        identities.append(
            {
                "dataset_index": dataset_index,
                "case_id": str(item["meta"]["case_id"]),
                "archive_slice_index": int(item["meta"]["archive_slice_index"]),
                "target_paths": list(item["meta"]["target_trajectory_paths"]),
            }
        )
    del sampler
    torch.cuda.empty_cache()
    return (
        torch.stack(sampled),
        torch.stack(truths),
        torch.stack(persistence),
        torch.stack(masks),
        identities,
    )


def _masked_rmse(values: torch.Tensor, truth: torch.Tensor, mask: torch.Tensor) -> float:
    valid = mask.expand_as(truth) > 0
    return float((values[valid] - truth[valid]).square().mean(dtype=torch.float64).sqrt().item())


def _masked_mae(values: torch.Tensor, truth: torch.Tensor, mask: torch.Tensor) -> float:
    valid = mask.expand_as(truth) > 0
    return float((values[valid] - truth[valid]).abs().mean(dtype=torch.float64).item())


def _roughness(values: torch.Tensor, mask: torch.Tensor) -> float:
    mask = mask[:, None].expand(values.shape[0], values.shape[1], values.shape[2], -1, -1)
    mask_y = (mask[..., 1:, :] > 0) & (mask[..., :-1, :] > 0)
    mask_x = (mask[..., :, 1:] > 0) & (mask[..., :, :-1] > 0)
    dy = (values[..., 1:, :] - values[..., :-1, :]).abs()[mask_y]
    dx = (values[..., :, 1:] - values[..., :, :-1]).abs()[mask_x]
    return float(torch.cat((dy, dx)).to(torch.float64).mean().item())


def _support_magnitude(values: torch.Tensor, mask: torch.Tensor, field: str) -> dict[str, float]:
    valid = mask[:, None].expand_as(values) > 0
    selected = values[valid].to(torch.float64)
    if not torch.isfinite(selected).all():
        raise FloatingPointError(f"{field} samples contain NaN/Inf on valid ocean points")
    if field == "sic":
        magnitude = torch.relu(-selected) + torch.relu(selected - 1.0)
    elif field == "sit":
        magnitude = torch.relu(-selected)
    else:
        raise ValueError(field)
    violating = magnitude > 0
    positive = magnitude[violating]
    return {
        "frequency": float(violating.to(torch.float64).mean().item()),
        "mean_among_violations": float(positive.mean().item()) if positive.numel() else 0.0,
        "p95_among_violations": float(torch.quantile(positive, 0.95).item()) if positive.numel() else 0.0,
        "max": float(positive.max().item()) if positive.numel() else 0.0,
    }


def _score(ensemble: torch.Tensor, truth: torch.Tensor, persistence: torch.Tensor, mask: torch.Tensor) -> dict[str, Any]:
    result: dict[str, Any] = {"leads": {}, "support": {}}
    for field_offset, field in enumerate(("sic", "sit")):
        result["support"][field] = _support_magnitude(ensemble[:, :, field_offset::2], mask, field)
    for lead_index, lead in enumerate(DIRECT_LEADS):
        lead_result = {}
        for field_offset, field in enumerate(("sic", "sit")):
            channel = 2 * lead_index + field_offset
            members = ensemble[:, :, channel : channel + 1]
            target = truth[:, channel : channel + 1]
            baseline = persistence[:, channel : channel + 1]
            rank = fractional_rank_counts(members, target, mask).cpu()
            rank_probability = rank / rank.sum()
            uniform = torch.full_like(rank_probability, 1.0 / len(rank_probability))
            spread = members.std(dim=1, unbiased=True)
            skill = _masked_rmse(members.mean(dim=1), target, mask)
            lead_result[field] = {
                "ensemble_mean_rmse": skill,
                "persistence_rmse": _masked_rmse(baseline, target, mask),
                "fair_crps": _fair_crps(members, target, mask),
                "persistence_point_mass_crps": _masked_mae(baseline, target, mask),
                "spread": float(torch.sqrt((spread.square() * mask).sum() / mask.sum().clamp(min=1)).item()),
                "spread_skill_ratio": float(spread.square().mul(mask).sum().div(mask.sum().clamp(min=1)).sqrt().item() / max(skill, 1e-12)),
                "fractional_rank_counts": [float(value) for value in rank],
                "rank_tv_to_uniform": float(0.5 * (rank_probability - uniform).abs().sum().item()),
                "normalized_mean_rank": float((rank_probability * torch.arange(len(rank), dtype=torch.float64)).sum().item() / (len(rank) - 1)),
                "member_roughness": _roughness(members, mask),
            }
        result["leads"][f"d{lead}"] = lead_result
    result["temporal_change"] = {}
    for earlier, later in ((0, 1), (1, 2)):
        for field_offset, field in enumerate(("sic", "sit")):
            c0, c1 = 2 * earlier + field_offset, 2 * later + field_offset
            member_delta = ensemble[:, :, c1 : c1 + 1] - ensemble[:, :, c0 : c0 + 1]
            truth_delta = truth[:, c1 : c1 + 1] - truth[:, c0 : c0 + 1]
            result["temporal_change"][f"d{DIRECT_LEADS[earlier]}_to_d{DIRECT_LEADS[later]}_{field}_mean_rmse"] = _masked_rmse(
                member_delta.mean(dim=1), truth_delta, mask
            )
    return result


def _save_visuals(output: Path, label: str, ensemble: torch.Tensor, truth: torch.Tensor, persistence: torch.Tensor, mask: torch.Tensor) -> list[Path]:
    paths = []
    visual_dir = output / "visuals" / label
    visual_dir.mkdir(parents=True, exist_ok=True)
    for case in range(min(2, ensemble.shape[0])):
        for member in range(min(4, ensemble.shape[1])):
            figure = make_structured_trajectory_figure(
                truth[case], persistence[case], ensemble[case, member], mask[case],
                title=f"{label}: case {case}, individual member {member}",
                origin="upper", lead_days=DIRECT_LEADS,
            )
            path = visual_dir / f"case_{case:02d}_member_{member:02d}.png"
            figure.savefig(path, dpi=170)
            import matplotlib.pyplot as plt
            plt.close(figure)
            paths.append(path)
    return paths


def run(run_dir: Path, output: Path, cases: int, members: int) -> dict[str, Any]:
    if torch.cuda.device_count() != 1:
        raise RuntimeError("paired direct evaluation requires exactly one visible GPU")
    if os.environ.get("CLEARML_REQUIRE_ONLINE") != "1":
        raise RuntimeError("paired direct evaluation requires CLEARML_REQUIRE_ONLINE=1")
    verified_inputs = {}
    for relative, expected in EXPECTED_INPUT_SHA256.items():
        path = run_dir / relative
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"frozen input SHA-256 mismatch for {relative}: {actual}")
        verified_inputs[relative] = actual
    if output.exists():
        raise FileExistsError(f"refusing to reuse evaluation output {output}")
    output.mkdir(parents=True)
    metadata = json.loads((run_dir / "metadata.json").read_text())
    training_config = metadata["training_config"]
    data_config = metadata["data_config"]
    dataset = build_dataset(data_config, split="valid")
    sentinel = validate_direct_dataset(dataset)
    indices = _selected_indices(len(dataset), cases)
    checkpoints = {
        "ema_epoch6": "epoch_snapshots/epoch_0006/ema_last_model.pth",
        "ema_epoch8": "epoch_snapshots/epoch_0008/ema_last_model.pth",
    }
    tracker = ClearMLTracker(
        project_name="sea_ice_two_stage",
        task_name="direct_dynamics_paired_val_epoch6_vs_epoch8_v1",
        tags=["paired-validation", "ema", "rank-histogram", "raw-unclipped", "one-gpu"],
        env_path="/home/.env",
    )
    tracker.connect("evaluation", {"cases": cases, "members": members, "indices": indices, "checkpoints": checkpoints})
    result: dict[str, Any] = {
        "schema_version": "direct_dynamics_paired_validation_v1",
        "split": "valid",
        "primary_scores_use_raw_unclipped_samples": True,
        "inference_precision": {
            "network_autocast": "bfloat16",
            "ODE_state_and_time_grid": "float32",
            "claim": "recovery evaluation; not strict-FP32 solver control",
        },
        "cases": cases,
        "members": members,
        "validation_case_indices": indices,
        "member_noise_seed_rule": "314159 + 1000003*case_order + 1009*member_index",
        "verified_input_sha256": verified_inputs,
        "dataset_sentinel": sentinel,
        "checkpoints": {},
    }
    reference = None
    for label, checkpoint in checkpoints.items():
        checkpoint_path = run_dir / checkpoint
        ensemble, truth, persistence, mask, identities = _sample_checkpoint(
            run_dir, checkpoint, training_config, dataset, indices, members, 17, torch.device("cuda")
        )
        if reference is None:
            reference = (truth, persistence, mask, identities)
        else:
            if not torch.equal(reference[0], truth) or identities != reference[3]:
                raise RuntimeError("paired checkpoint evaluation lost case identity")
        metrics = _score(ensemble, truth, persistence, mask)
        visuals = _save_visuals(output, label, ensemble, truth, persistence, mask)
        torch.save(
            {"ensemble": ensemble[:2], "truth": truth[:2], "persistence": persistence[:2], "valid_mask": mask[:2], "case_identities": identities[:2]},
            output / f"{label}_visual_samples.pt",
        )
        result["checkpoints"][label] = {
            "checkpoint": checkpoint,
            "checkpoint_sha256": _sha256(checkpoint_path),
            "case_identities": identities,
            "metrics": metrics,
        }
        for lead, fields in metrics["leads"].items():
            for field, values in fields.items():
                for name in ("ensemble_mean_rmse", "persistence_rmse", "fair_crps", "persistence_point_mass_crps", "spread_skill_ratio", "rank_tv_to_uniform"):
                    tracker.report_scalar(f"paired/{name}", f"{label}/{lead}/{field}", values[name], 0)
        for path in visuals:
            tracker.report_image("individual_samples", f"{label}/{path.stem}", path, 0)
        del ensemble
        torch.cuda.empty_cache()

    # A compact same-noise solver control; it is diagnostic, never a tuning loop.
    result["solver_sensitivity_17_vs_33"] = {}
    for label, checkpoint in checkpoints.items():
        solver = {}
        for steps in (17, 33):
            ensemble, truth, _, mask, _ = _sample_checkpoint(
                run_dir, checkpoint, training_config, dataset, indices[:2], members, steps, torch.device("cuda")
            )
            solver[str(steps)] = {"ensemble": ensemble, "truth": truth, "mask": mask}
        diff = solver["33"]["ensemble"] - solver["17"]["ensemble"]
        checkpoint_result: dict[str, Any] = {"cases": 2, "members": members, "leads": {}}
        for lead_index, lead in enumerate(DIRECT_LEADS):
            checkpoint_result["leads"][f"d{lead}"] = {}
            for field_offset, field in enumerate(("sic", "sit")):
                channel = 2 * lead_index + field_offset
                field_diff = diff[:, :, channel : channel + 1]
                valid = solver["17"]["mask"][:, None].expand_as(field_diff) > 0
                selected = field_diff[valid].to(torch.float64)
                checkpoint_result["leads"][f"d{lead}"][field] = {
                    "mean_absolute_difference": float(selected.abs().mean().item()),
                    "root_mean_square_difference": float(selected.square().mean().sqrt().item()),
                    "max_absolute_difference": float(selected.abs().max().item()),
                }
        result["solver_sensitivity_17_vs_33"][label] = checkpoint_result
    result["clearml_task_id"] = str(tracker.task.id)
    _require_finite_scalars(result)
    _atomic_json(output / "paired_evaluation.json", result)
    tracker.upload_artifact("paired_evaluation", output / "paired_evaluation.json")
    tracker.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cases", type=int, default=12)
    parser.add_argument("--members", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(run(args.run_dir.resolve(), args.output.resolve(), args.cases, args.members), indent=2))


if __name__ == "__main__":
    main()
