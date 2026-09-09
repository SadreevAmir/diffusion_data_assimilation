"""Frozen EMA-6 physical-tail and strict-FP32 solver diagnostic."""

from __future__ import annotations

import argparse
import json
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch

from .clearml_tracking import ClearMLTracker
from .data import build_dataset
from .direct_dynamics_evaluation import (
    EXPECTED_INPUT_SHA256,
    _case_identity,
    _initial_noise,
    _require_finite_scalars,
    _save_visuals,
    _score,
    _selected_indices,
    _sha256,
    _tensor_batch,
)
from .direct_dynamics_training import (
    DIRECT_LEADS,
    DIRECT_OUTPUT_CHANNELS,
    _repeat_field_stats,
    validate_direct_dataset,
)
from .model_io import load_sampler
from .trainer import _atomic_json
from .transforms import channel_denormalize


EMA6_CHECKPOINT = "epoch_snapshots/epoch_0006/ema_last_model.pth"


@torch.no_grad()
def _sample_cases(
    run_dir: Path,
    training_config: dict[str, Any],
    dataset,
    prepared_cases: list[tuple[int, int, dict[str, Any]]],
    *,
    members: int,
    steps: int,
    network_precision: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    device = torch.device("cuda")
    sampler = load_sampler(str(run_dir), EMA6_CHECKPOINT, training_config, device=device)
    means = _repeat_field_stats(dataset.means)
    stds = _repeat_field_stats(dataset.stds)
    samples, truths, baselines, masks, identities = [], [], [], [], []
    image_size = tuple(int(value) for value in training_config["image_size"])
    for position, (case_order, dataset_index, item) in enumerate(prepared_cases):
        print(
            f"[tail-diagnostic] precision={network_precision} steps={steps} "
            f"case={position + 1}/{len(prepared_cases)} index={dataset_index}",
            flush=True,
        )
        batch = _tensor_batch(item, members, device)
        valid = batch["valid_mask"][:, :1]
        noise = _initial_noise(case_order, members, image_size, device)
        precision_context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if network_precision == "bf16"
            else nullcontext()
        )
        with precision_context:
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
        samples.append(channel_denormalize(normalized.float(), means, stds).cpu())
        truths.append(item["structured_physical_truth"].float())
        baselines.append(item["structured_physical_background"].float())
        masks.append(item["valid_mask"][:1].float())
        identities.append(_case_identity(item, dataset_index))
    del sampler
    torch.cuda.empty_cache()
    return (
        torch.stack(samples),
        torch.stack(truths),
        torch.stack(baselines),
        torch.stack(masks),
        identities,
    )


def _top_support_violations(
    ensemble: torch.Tensor,
    mask: torch.Tensor,
    identities: list[dict[str, Any]],
    *,
    field: str,
    count: int = 8,
) -> list[dict[str, Any]]:
    offset = 0 if field == "sic" else 1
    values = ensemble[:, :, offset::2]
    valid = mask[:, None].expand_as(values) > 0
    if field == "sic":
        magnitude = torch.relu(-values) + torch.relu(values - 1.0)
    else:
        magnitude = torch.relu(-values)
    magnitude = torch.where(valid, magnitude, torch.full_like(magnitude, -1.0))
    shape = magnitude.shape
    top_values, top_indices = torch.topk(magnitude.flatten(), k=min(count, magnitude.numel()))
    records = []
    for violation, flat_index in zip(top_values, top_indices):
        index = int(flat_index.item())
        width = shape[4]
        column = index % width
        index //= width
        height = shape[3]
        row = index % height
        index //= height
        lead_index = index % shape[2]
        index //= shape[2]
        member = index % shape[1]
        case = index // shape[1]
        records.append(
            {
                "field": field,
                "violation_magnitude": float(violation.item()),
                "raw_value": float(values[case, member, lead_index, row, column].item()),
                "case_order": int(case),
                "case_id": identities[case]["case_id"],
                "dataset_index": identities[case]["dataset_index"],
                "member": int(member),
                "lead_day": int(DIRECT_LEADS[lead_index]),
                "row": int(row),
                "column": int(column),
            }
        )
    return records


def _solver_difference(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> dict[str, Any]:
    result = {}
    difference = b - a
    for lead_index, lead in enumerate(DIRECT_LEADS):
        result[f"d{lead}"] = {}
        for offset, field in enumerate(("sic", "sit")):
            selected = difference[:, :, 2 * lead_index + offset : 2 * lead_index + offset + 1]
            valid = mask[:, None].expand_as(selected) > 0
            values = selected[valid].to(torch.float64)
            result[f"d{lead}"][field] = {
                "mean_absolute_difference": float(values.abs().mean().item()),
                "root_mean_square_difference": float(values.square().mean().sqrt().item()),
                "max_absolute_difference": float(values.abs().max().item()),
            }
    return result


def run(run_dir: Path, output: Path) -> dict[str, Any]:
    if torch.cuda.device_count() != 1:
        raise RuntimeError("tail diagnostic requires exactly one visible GPU")
    if os.environ.get("CLEARML_REQUIRE_ONLINE") != "1":
        raise RuntimeError("tail diagnostic requires CLEARML_REQUIRE_ONLINE=1")
    if output.exists():
        raise FileExistsError(f"refusing to reuse diagnostic output {output}")
    verified = {}
    for relative, expected in EXPECTED_INPUT_SHA256.items():
        actual = _sha256(run_dir / relative)
        if actual != expected:
            raise ValueError(f"frozen input SHA-256 mismatch for {relative}")
        verified[relative] = actual
    output.mkdir(parents=True)
    metadata = json.loads((run_dir / "metadata.json").read_text())
    training_config = metadata["training_config"]
    dataset = build_dataset(metadata["data_config"], split="valid")
    indices = _selected_indices(len(dataset), 12)
    tracker = ClearMLTracker(
        project_name="sea_ice_two_stage",
        task_name=f"direct_dynamics_ema6_tail_fp32_{output.name}",
        tags=["ema6", "support-tail", "strict-fp32", "rk4-33-65", "validation-only", "one-gpu"],
        env_path="/home/.env",
    )
    tracker.connect("diagnostic", {"indices": indices, "members": 8, "checkpoint": EMA6_CHECKPOINT})
    preload = sorted(set(indices) | {0, 12, 23})
    cache = {}
    for position, dataset_index in enumerate(preload):
        print(f"[tail-diagnostic] preparing case={position + 1}/{len(preload)} index={dataset_index}", flush=True)
        cache[dataset_index] = dataset[dataset_index]
    sentinel = validate_direct_dataset(dataset, item_cache=cache)
    prepared = [(order, index, cache[index]) for order, index in enumerate(indices)]

    bf16_17, truth, persistence, mask, identities = _sample_cases(
        run_dir, training_config, dataset, prepared, members=8, steps=17, network_precision="bf16"
    )
    top = {
        field: _top_support_violations(bf16_17, mask, identities, field=field)
        for field in ("sic", "sit")
    }
    selected_orders = sorted({top["sic"][0]["case_order"], top["sit"][0]["case_order"]})
    selected_prepared = [prepared[order] for order in selected_orders]
    selected_truth = truth[selected_orders]
    selected_persistence = persistence[selected_orders]
    selected_mask = mask[selected_orders]

    previous_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    previous_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        fp32 = {}
        for steps in (33, 65):
            fp32[steps], fp_truth, fp_persistence, fp_mask, fp_identities = _sample_cases(
                run_dir, training_config, dataset, selected_prepared,
                members=8, steps=steps, network_precision="fp32",
            )
            if not torch.equal(fp_truth, selected_truth) or fp_identities != [identities[i] for i in selected_orders]:
                raise RuntimeError("strict-FP32 diagnostic lost frozen case pairing")
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_matmul_tf32
        torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32

    visuals = _save_visuals(output, "ema6_fp32_rk4_65", fp32[65], selected_truth, selected_persistence, selected_mask)
    result = {
        "schema_version": "direct_dynamics_ema6_tail_fp32_v1",
        "split": "valid",
        "verified_input_sha256": verified,
        "dataset_sentinel": sentinel,
        "checkpoint": EMA6_CHECKPOINT,
        "checkpoint_sha256": verified[EMA6_CHECKPOINT],
        "localization_precision": "NN bf16; ODE state float32; RK4 17",
        "strict_solver_precision": "NN float32; ODE float32; TF32 disabled",
        "top_support_violations_bf16_rk4_17": top,
        "selected_case_orders": selected_orders,
        "selected_case_identities": [identities[index] for index in selected_orders],
        "fp32_rk4_33_metrics": _score(fp32[33], selected_truth, selected_persistence, selected_mask),
        "fp32_rk4_65_metrics": _score(fp32[65], selected_truth, selected_persistence, selected_mask),
        "fp32_solver_sensitivity_33_vs_65": _solver_difference(fp32[33], fp32[65], selected_mask),
        "clearml_task_id": str(tracker.task.id),
    }
    _require_finite_scalars(result)
    _atomic_json(output / "tail_diagnostic.json", result)
    torch.save(
        {"rk4_33": fp32[33], "rk4_65": fp32[65], "truth": selected_truth, "persistence": selected_persistence, "valid_mask": selected_mask},
        output / "selected_tail_samples.pt",
    )
    for path in visuals:
        tracker.report_image("strict_fp32_tail_samples", path.stem, path, 0)
    tracker.upload_artifact("tail_diagnostic", output / "tail_diagnostic.json")
    tracker.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.run_dir.resolve(), args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
