"""Bounded frozen-weight temperature calibration for direct sea-ice dynamics.

Only the initial six-channel Gaussian noise is scaled.  The EMA checkpoint,
conditioning, masks, solver, and raw physical scoring remain unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
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
    _sha256,
)
from .direct_dynamics_tail_diagnostic import EMA6_CHECKPOINT
from .direct_dynamics_training import (
    DIRECT_LEADS,
    DIRECT_OUTPUT_CHANNELS,
    _repeat_field_stats,
    validate_direct_dataset,
)
from .model_io import load_sampler
from .trainer import _atomic_json
from .transforms import channel_denormalize


TEMPERATURES = (1.0, 1.05, 1.10)
CALIBRATION_INDICES = (0, 777, 1553, 2330, 3107, 3883, 4660, 5436, 6213, 6990, 7766, 8543)
CONFIRMATION_DAYS = (10, 20, 48, 80, 113, 145, 177, 210, 242, 275, 307, 339)
CONFIRMATION_HOURS = (7, 18, 5, 16, 3, 14, 1, 12, 23, 10, 21, 8)
CONFIRMATION_INDICES = tuple(
    day * 24 + hour for day, hour in zip(CONFIRMATION_DAYS, CONFIRMATION_HOURS)
)
STRESS_DATASET_INDICES = (3883, 7766)
SSR_REFERENCE = math.sqrt(8.0 / 9.0)
BOOTSTRAP_SEED = 20260909
BOOTSTRAP_REPLICATES = 10_000


def validate_panel_indices(dataset_length: int) -> dict[str, Any]:
    panels = {
        "calibration": list(CALIBRATION_INDICES),
        "confirmation": list(CONFIRMATION_INDICES),
    }
    flat = [index for values in panels.values() for index in values]
    if dataset_length != 8544:
        raise ValueError(f"frozen panel expects validation length 8544, got {dataset_length}")
    if len(set(flat)) != 24 or any(index < 0 or index >= dataset_length for index in flat):
        raise ValueError("calibration/confirmation indices are not distinct valid cases")
    anchors = sorted((index // 24, name, index) for name, values in panels.items() for index in values)
    for left, right in zip(anchors, anchors[1:]):
        if right[0] - left[0] < 10:
            raise ValueError(
                f"overlapping [d0,d+9] windows at dataset indices {left[2]} and {right[2]}"
            )
    if not set(STRESS_DATASET_INDICES).issubset(CALIBRATION_INDICES):
        raise ValueError("fixed stress cases must be contained in the development panel")
    return {
        "calibration_indices": panels["calibration"],
        "confirmation_indices": panels["confirmation"],
        "minimum_anchor_separation_days": min(
            right[0] - left[0] for left, right in zip(anchors, anchors[1:])
        ),
        "stress_dataset_indices": list(STRESS_DATASET_INDICES),
    }


def scaled_initial_noise(
    seed_order: int,
    members: int,
    image_size: tuple[int, int],
    device: torch.device,
    temperature: float,
) -> torch.Tensor:
    if temperature <= 0 or not math.isfinite(temperature):
        raise ValueError("temperature must be finite and positive")
    return _initial_noise(seed_order, members, image_size, device) * temperature


def _stack_batch(items: list[dict[str, Any]], members: int, device: torch.device) -> dict[str, torch.Tensor]:
    result = {}
    for key, value in items[0].items():
        if torch.is_tensor(value):
            stacked = torch.stack([item[key] for item in items]).to(device)
            result[key] = stacked.repeat_interleave(members, dim=0)
    return result


@torch.no_grad()
def sample_panel(
    sampler,
    dataset,
    prepared: list[tuple[int, int, dict[str, Any]]],
    *,
    members: int,
    temperature: float,
    steps: int,
    cases_per_batch: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    device = torch.device("cuda")
    means = _repeat_field_stats(dataset.means)
    stds = _repeat_field_stats(dataset.stds)
    image_size = tuple(int(value) for value in dataset.image_size)
    samples = []
    for start in range(0, len(prepared), cases_per_batch):
        chunk = prepared[start : start + cases_per_batch]
        print(
            f"[temperature] tau={temperature:.2f} steps={steps} "
            f"cases={start + 1}-{start + len(chunk)}/{len(prepared)}",
            flush=True,
        )
        items = [record[2] for record in chunk]
        batch = _stack_batch(items, members, device)
        noises = torch.stack(
            [
                scaled_initial_noise(record[0], members, image_size, device, temperature)
                for record in chunk
            ]
        ).flatten(0, 1)
        valid = batch["valid_mask"][:, :1]
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
            initial_noise=noises,
            sample_target="state",
            model_conditioning=batch["structured_conditioning"],
            state_channels=DIRECT_OUTPUT_CHANNELS,
            end_time=0.0,
            state_mask=valid.expand(-1, DIRECT_OUTPUT_CHANNELS, -1, -1),
        )
        if not torch.isfinite(normalized).all():
            raise FloatingPointError(f"non-finite raw sample at tau={temperature}")
        samples.append(
            channel_denormalize(normalized.float(), means, stds)
            .reshape(len(chunk), members, DIRECT_OUTPUT_CHANNELS, *image_size)
            .cpu()
        )
    items = [record[2] for record in prepared]
    return (
        torch.cat(samples),
        torch.stack([item["structured_physical_truth"].float() for item in items]),
        torch.stack([item["structured_physical_background"].float() for item in items]),
        torch.stack([item["valid_mask"][:1].float() for item in items]),
        [_case_identity(item, record[1]) for record, item in zip(prepared, items)],
    )


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values))


def _save_stress_samples(
    output: Path,
    label: str,
    ensemble: torch.Tensor,
    truth: torch.Tensor,
    persistence: torch.Tensor,
    mask: torch.Tensor,
    identities: list[dict[str, Any]],
    stress_positions: list[int],
    temperature: float,
    *,
    prefix: str = "calibration",
    solver_steps: int = 33,
) -> Path:
    path = output / f"{prefix}_{label}_stress_raw.pt"
    temporary = output / f".{path.name}.incomplete"
    torch.save(
        {
            "raw_ensemble": ensemble[stress_positions],
            "truth": truth[stress_positions],
            "persistence": persistence[stress_positions],
            "valid_mask": mask[stress_positions],
            "case_identities": [identities[position] for position in stress_positions],
            "temperature": temperature,
            "member_noise_seed_rule": "314159 + 1000003*calibration_case_order + 1009*member_index",
            "noise_transform": "initial_noise = temperature * fixed_base_gaussian_noise",
            "checkpoint": EMA6_CHECKPOINT,
            "solver": f"NN/ODE FP32; TF32 off; RK4-{solver_steps}",
            "primary_role": "raw_unclipped_stress_evidence",
        },
        temporary,
    )
    os.replace(temporary, path)
    return path


def score_cases_equal(
    ensemble: torch.Tensor,
    truth: torch.Tensor,
    persistence: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, Any]:
    per_case = [
        _score(ensemble[i : i + 1], truth[i : i + 1], persistence[i : i + 1], mask[i : i + 1])
        for i in range(ensemble.shape[0])
    ]
    aggregate: dict[str, Any] = {"leads": {}}
    mean_scalar_names = (
        "fair_crps",
        "persistence_point_mass_crps",
        "normalized_mean_rank",
        "member_roughness",
    )
    for lead in ("d3", "d6", "d9"):
        aggregate["leads"][lead] = {}
        for field in ("sic", "sit"):
            case_values = [case["leads"][lead][field] for case in per_case]
            values = {name: _mean([case[name] for case in case_values]) for name in mean_scalar_names}
            values["ensemble_mean_rmse"] = math.sqrt(
                _mean([case["ensemble_mean_rmse"] ** 2 for case in case_values])
            )
            values["persistence_rmse"] = math.sqrt(
                _mean([case["persistence_rmse"] ** 2 for case in case_values])
            )
            values["spread"] = math.sqrt(_mean([case["spread"] ** 2 for case in case_values]))
            values["spread_skill_ratio"] = values["spread"] / max(
                values["ensemble_mean_rmse"], 1e-12
            )
            probabilities = []
            for case in per_case:
                counts = torch.tensor(case["leads"][lead][field]["fractional_rank_counts"], dtype=torch.float64)
                probabilities.append(counts / counts.sum())
            equal_probabilities = torch.stack(probabilities).mean(dim=0)
            values["equal_case_rank_probabilities"] = [float(value) for value in equal_probabilities]
            uniform = torch.full_like(equal_probabilities, 1.0 / len(equal_probabilities))
            values["rank_tv_to_uniform"] = float(
                0.5 * (equal_probabilities - uniform).abs().sum().item()
            )
            aggregate["leads"][lead][field] = values
    return {"aggregate": aggregate, "per_case": per_case}


def tail_metrics(ensemble: torch.Tensor, mask: torch.Tensor) -> dict[str, Any]:
    result: dict[str, Any] = {"leads": {}}
    for lead_index, lead_day in enumerate(DIRECT_LEADS):
        lead = f"d{lead_day}"
        result["leads"][lead] = {}
        for offset, field in enumerate(("sic", "sit")):
            per_case = []
            for case in range(ensemble.shape[0]):
                values = ensemble[case, :, 2 * lead_index + offset : 2 * lead_index + offset + 1]
                valid = mask[case : case + 1].expand_as(values) > 0
                selected = values[valid].to(torch.float64)
                if field == "sic":
                    delta = torch.relu(-selected) + torch.relu(selected - 1.0)
                else:
                    delta = torch.relu(-selected)
                per_case.append(
                    {
                        "mean_delta": float(delta.mean().item()),
                        "frequency_gt_0p01": float((delta > 0.01).to(torch.float64).mean().item()),
                        "frequency_gt_0p10": float((delta > 0.10).to(torch.float64).mean().item()),
                        "max_delta": float(delta.max().item()),
                    }
                )
            result["leads"][lead][field] = {
                "mean_delta": _mean([case["mean_delta"] for case in per_case]),
                "frequency_gt_0p01": _mean([case["frequency_gt_0p01"] for case in per_case]),
                "frequency_gt_0p10": _mean([case["frequency_gt_0p10"] for case in per_case]),
                "max_delta": max(case["max_delta"] for case in per_case),
                "per_case": per_case,
            }
    return result


def tail_gate(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    failures = []
    for lead in ("d3", "d6", "d9"):
        for field in ("sic", "sit"):
            cand = candidate["leads"][lead][field]
            base = baseline["leads"][lead][field]
            for name, absolute_tolerance in (
                ("mean_delta", 1e-5),
                ("frequency_gt_0p01", 1e-6),
                ("frequency_gt_0p10", 1e-6),
            ):
                limit = 1.02 * base[name] + absolute_tolerance
                if cand[name] > limit:
                    failures.append({"lead": lead, "field": field, "metric": name, "value": cand[name], "limit": limit})
            max_limit = base["max_delta"] + 0.005
            if cand["max_delta"] > max_limit:
                failures.append({"lead": lead, "field": field, "metric": "max_delta", "value": cand["max_delta"], "limit": max_limit})
    return {"passed": not failures, "failures": failures}


def comparison_summary(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    ratios = {"fair_crps": {}, "ensemble_mean_rmse": {}}
    ssr_errors_candidate, ssr_errors_baseline = [], []
    rank_differences = []
    for lead in ("d3", "d6", "d9"):
        for field in ("sic", "sit"):
            key = f"{lead}_{field}"
            cand = candidate["aggregate"]["leads"][lead][field]
            base = baseline["aggregate"]["leads"][lead][field]
            for metric in ratios:
                ratios[metric][key] = cand[metric] / max(base[metric], 1e-12)
            ssr_errors_candidate.append(abs(cand["spread_skill_ratio"] - SSR_REFERENCE))
            ssr_errors_baseline.append(abs(base["spread_skill_ratio"] - SSR_REFERENCE))
            rank_differences.append(cand["rank_tv_to_uniform"] - base["rank_tv_to_uniform"])
    return {
        "J_mean_fair_crps_ratio": _mean(list(ratios["fair_crps"].values())),
        "fair_crps_ratios": ratios["fair_crps"],
        "rmse_ratios": ratios["ensemble_mean_rmse"],
        "mean_abs_ssr_error_candidate": _mean(ssr_errors_candidate),
        "mean_abs_ssr_error_baseline": _mean(ssr_errors_baseline),
        "mean_abs_ssr_error_reduction": _mean(ssr_errors_baseline) - _mean(ssr_errors_candidate),
        "mean_rank_tv_difference": _mean(rank_differences),
        "rank_tv_differences": dict(zip(ratios["fair_crps"], rank_differences)),
    }


def bootstrap_J_upper95(candidate: dict[str, Any], baseline: dict[str, Any]) -> float:
    pairs = [(lead, field) for lead in ("d3", "d6", "d9") for field in ("sic", "sit")]
    cand = torch.tensor(
        [[case["leads"][lead][field]["fair_crps"] for lead, field in pairs] for case in candidate["per_case"]],
        dtype=torch.float64,
    )
    base = torch.tensor(
        [[case["leads"][lead][field]["fair_crps"] for lead, field in pairs] for case in baseline["per_case"]],
        dtype=torch.float64,
    )
    generator = torch.Generator(device="cpu").manual_seed(BOOTSTRAP_SEED)
    indices = torch.randint(len(cand), (BOOTSTRAP_REPLICATES, len(cand)), generator=generator)
    values = (cand[indices].mean(dim=1) / base[indices].mean(dim=1).clamp(min=1e-12)).mean(dim=1)
    return float(torch.quantile(values, 0.95).item())


def confirmation_gate(
    comparison: dict[str, Any],
    bootstrap_upper95: float,
    *tail_results: dict[str, Any],
) -> dict[str, Any]:
    failures = []
    if comparison["J_mean_fair_crps_ratio"] > 0.99:
        failures.append("J_mean_fair_crps_ratio > 0.99")
    if bootstrap_upper95 >= 1.0:
        failures.append("paired date-bootstrap upper95 >= 1")
    if any(value > 1.01 for value in comparison["fair_crps_ratios"].values()):
        failures.append("an individual fair CRPS ratio > 1.01")
    if any(value > 1.01 for value in comparison["rmse_ratios"].values()):
        failures.append("an individual RMSE ratio > 1.01")
    if comparison["mean_abs_ssr_error_reduction"] < 0.01:
        failures.append("mean abs SSR-reference error reduction < 0.01")
    if comparison["mean_rank_tv_difference"] > 0:
        failures.append("mean rank TV increased")
    if any(value > 0.01 for value in comparison["rank_tv_differences"].values()):
        failures.append("an individual rank TV increased by > 0.01")
    if any(not result["passed"] for result in tail_results):
        failures.append("at least one raw tail gate failed")
    return {"passed": not failures, "failures": failures}


def _development_candidate_gate(comparison: dict[str, Any], *tail_results: dict[str, Any]) -> bool:
    return (
        comparison["J_mean_fair_crps_ratio"] < 1.0
        and max(comparison["fair_crps_ratios"].values()) <= 1.01
        and max(comparison["rmse_ratios"].values()) <= 1.01
        and comparison["mean_abs_ssr_error_reduction"] > 0
        and comparison["mean_rank_tv_difference"] <= 0
        and max(comparison["rank_tv_differences"].values()) <= 0.01
        and all(result["passed"] for result in tail_results)
    )


def run(run_dir: Path, output: Path) -> dict[str, Any]:
    if torch.cuda.device_count() != 1:
        raise RuntimeError("temperature calibration requires exactly one visible GPU")
    if os.environ.get("CLEARML_REQUIRE_ONLINE") != "1":
        raise RuntimeError("temperature calibration requires CLEARML_REQUIRE_ONLINE=1")
    # The server exposes 256 logical CPUs through cpuset but enforces a six-core
    # CFS quota.  PyTorch otherwise creates about 256 scoring threads, causing
    # severe oversubscription in rank/CRPS reductions.
    torch.set_num_threads(6)
    torch.set_num_interop_threads(1)
    if output.exists():
        raise FileExistsError(f"refusing to reuse calibration output {output}")
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
    panel_contract = validate_panel_indices(len(dataset))
    tracker = ClearMLTracker(
        project_name="sea_ice_two_stage",
        task_name=f"direct_dynamics_ema6_temperature_{output.name}",
        tags=["ema6", "noise-temperature", "strict-fp32", "rk4-33", "date-bootstrap", "raw-unclipped", "one-gpu"],
        env_path="/home/.env",
    )
    tracker.connect(
        "protocol",
        {
            **panel_contract,
            "temperatures": TEMPERATURES,
            "members": 8,
            "solver": "FP32 TF32-off RK4-33",
            "optimizer_steps": 0,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        },
    )
    # Confirmation data is deliberately loaded only after a candidate is selected.
    # This preserves the scientific firewall and avoids spending roughly half the
    # archive-I/O budget when tau=1.05 already fails the predeclared tail gate.
    all_indices = sorted(set(CALIBRATION_INDICES) | {12, 23})
    cache = {}
    for position, dataset_index in enumerate(all_indices):
        print(f"[temperature] preparing case={position + 1}/{len(all_indices)} index={dataset_index}", flush=True)
        cache[dataset_index] = dataset[dataset_index]
    sentinel = validate_direct_dataset(dataset, item_cache=cache)
    calibration = [(order, index, cache[index]) for order, index in enumerate(CALIBRATION_INDICES)]
    stress_positions = [CALIBRATION_INDICES.index(index) for index in STRESS_DATASET_INDICES]
    result: dict[str, Any] = {
        "schema_version": "direct_dynamics_ema6_temperature_calibration_v1",
        "status": "running",
        "split": "valid",
        "selection_uses_confirmation": False,
        "optimizer_steps": 0,
        "primary_scores_use_raw_unclipped_samples": True,
        "projection_role": "display_only",
        "inference_precision": "NN float32; ODE float32; TF32 disabled; RK4-33",
        "cpu_thread_contract": {"intraop": 6, "interop": 1},
        "checkpoint": EMA6_CHECKPOINT,
        "checkpoint_sha256": verified[EMA6_CHECKPOINT],
        "verified_input_sha256": verified,
        "dataset_sentinel": sentinel,
        "panel_contract": panel_contract,
        "clearml_task_id": str(tracker.task.id),
        "calibration": {},
        "confirmation": {},
    }
    previous_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    previous_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sampler = load_sampler(str(run_dir), EMA6_CHECKPOINT, training_config, device=torch.device("cuda"))
    try:
        calibration_samples = {}
        for temperature in TEMPERATURES:
            label = f"tau_{temperature:.2f}"
            ensemble, truth, persistence, mask, identities = sample_panel(
                sampler, dataset, calibration, members=8, temperature=temperature, steps=33
            )
            scores = score_cases_equal(ensemble, truth, persistence, mask)
            tails = tail_metrics(ensemble, mask)
            stress_tails = tail_metrics(ensemble[stress_positions], mask[stress_positions])
            entry: dict[str, Any] = {"case_identities": identities, "scores": scores, "tails": tails, "stress_tails": stress_tails}
            calibration_samples[temperature] = ensemble
            if temperature != 1.0:
                entry["comparison_to_tau_1"] = comparison_summary(scores, result["calibration"]["tau_1.00"]["scores"])
                entry["tail_gate"] = tail_gate(tails, result["calibration"]["tau_1.00"]["tails"])
                entry["stress_tail_gate"] = tail_gate(stress_tails, result["calibration"]["tau_1.00"]["stress_tails"])
            result["calibration"][label] = entry
            stress_path = _save_stress_samples(
                output,
                label,
                ensemble,
                truth,
                persistence,
                mask,
                identities,
                stress_positions,
                temperature,
            )
            entry["stress_raw_samples_path"] = str(stress_path)
            tracker.upload_artifact(f"{label}_stress_raw", stress_path)
            for path in _save_visuals(output, f"calibration_{label}_raw", ensemble[:1], truth[:1], persistence[:1], mask[:1]):
                tracker.report_image("temperature/raw_individual_samples", f"calibration/{label}/{path.stem}", path, 0)
            stress_ensemble = ensemble[stress_positions]
            stress_truth = truth[stress_positions]
            stress_persistence = persistence[stress_positions]
            stress_mask = mask[stress_positions]
            for path in _save_visuals(
                output,
                f"calibration_{label}_stress_raw",
                stress_ensemble,
                stress_truth,
                stress_persistence,
                stress_mask,
            ):
                tracker.report_image(
                    "temperature/raw_stress_samples",
                    f"calibration/{label}/{path.stem}",
                    path,
                    0,
                )
            _atomic_json(output / "progress.json", result)
            if temperature == 1.05 and (
                not entry["tail_gate"]["passed"] or not entry["stress_tail_gate"]["passed"]
            ):
                result["stopped_before_tau_1.10"] = "tau_1.05_failed_raw_tail_gate"
                break

        candidates = []
        baseline_entry = result["calibration"]["tau_1.00"]
        for temperature in (1.05, 1.10):
            label = f"tau_{temperature:.2f}"
            if label not in result["calibration"]:
                continue
            entry = result["calibration"][label]
            if _development_candidate_gate(entry["comparison_to_tau_1"], entry["tail_gate"], entry["stress_tail_gate"]):
                candidates.append((entry["comparison_to_tau_1"]["J_mean_fair_crps_ratio"], temperature))
        if not candidates:
            result["status"] = "complete_no_candidate"
            result["decision"] = "keep_tau_1.00"
        else:
            selected_temperature = min(candidates)[1]
            selected_label = f"tau_{selected_temperature:.2f}"
            result["selected_on_calibration"] = selected_label
            for position, dataset_index in enumerate(CONFIRMATION_INDICES):
                print(
                    f"[temperature] preparing confirmation case="
                    f"{position + 1}/{len(CONFIRMATION_INDICES)} index={dataset_index}",
                    flush=True,
                )
                cache[dataset_index] = dataset[dataset_index]
            confirmation = [
                (100 + order, index, cache[index])
                for order, index in enumerate(CONFIRMATION_INDICES)
            ]
            confirmation_samples = {}
            for temperature in (1.0, selected_temperature):
                label = f"tau_{temperature:.2f}"
                ensemble, truth, persistence, mask, identities = sample_panel(
                    sampler, dataset, confirmation, members=8, temperature=temperature, steps=33
                )
                confirmation_samples[temperature] = ensemble
                result["confirmation"][label] = {
                    "case_identities": identities,
                    "scores": score_cases_equal(ensemble, truth, persistence, mask),
                    "tails": tail_metrics(ensemble, mask),
                }
                for path in _save_visuals(output, f"confirmation_{label}_raw", ensemble[:1], truth[:1], persistence[:1], mask[:1]):
                    tracker.report_image("temperature/raw_individual_samples", f"confirmation/{label}/{path.stem}", path, 0)
                _atomic_json(output / "progress.json", result)
            base_confirm = result["confirmation"]["tau_1.00"]
            cand_confirm = result["confirmation"][selected_label]
            comparison = comparison_summary(cand_confirm["scores"], base_confirm["scores"])
            bootstrap_upper = bootstrap_J_upper95(cand_confirm["scores"], base_confirm["scores"])
            confirmation_tails = tail_gate(cand_confirm["tails"], base_confirm["tails"])

            stress_prepared = [calibration[position] for position in stress_positions]
            stress65 = {}
            stress65_paths = {}
            for temperature in (1.0, selected_temperature):
                stress_ensemble, stress_truth, stress_persistence, stress_mask, stress_identities = sample_panel(
                    sampler, dataset, stress_prepared, members=8, temperature=temperature, steps=65
                )
                stress65[temperature] = tail_metrics(stress_ensemble, stress_mask)
                label = f"tau_{temperature:.2f}"
                stress65_paths[temperature] = _save_stress_samples(
                    output,
                    label,
                    stress_ensemble,
                    stress_truth,
                    stress_persistence,
                    stress_mask,
                    stress_identities,
                    list(range(len(stress_prepared))),
                    temperature,
                    prefix="stress_rk4_65",
                    solver_steps=65,
                )
                tracker.upload_artifact(f"{label}_stress_rk4_65_raw", stress65_paths[temperature])
            stress65_gate = tail_gate(stress65[selected_temperature], stress65[1.0])
            calibration_entry = result["calibration"][selected_label]
            gate = confirmation_gate(
                comparison,
                bootstrap_upper,
                calibration_entry["tail_gate"],
                calibration_entry["stress_tail_gate"],
                confirmation_tails,
                stress65_gate,
            )
            result["confirmation_comparison"] = comparison
            result["paired_date_bootstrap_J_upper95"] = bootstrap_upper
            result["confirmation_tail_gate"] = confirmation_tails
            result["stress_rk4_65"] = {
                "baseline": stress65[1.0],
                "candidate": stress65[selected_temperature],
                "tail_gate": stress65_gate,
                "baseline_raw_samples_path": str(stress65_paths[1.0]),
                "candidate_raw_samples_path": str(stress65_paths[selected_temperature]),
            }
            result["acceptance_gate"] = gate
            result["status"] = "complete_candidate_accepted" if gate["passed"] else "complete_candidate_rejected"
            result["decision"] = selected_label if gate["passed"] else "keep_tau_1.00"
            torch.save(
                {
                    "calibration_tau_1": calibration_samples[1.0][:1],
                    "calibration_candidate": calibration_samples[selected_temperature][:1],
                    "confirmation_tau_1": confirmation_samples[1.0][:1],
                    "confirmation_candidate": confirmation_samples[selected_temperature][:1],
                },
                output / "selected_raw_sample_examples.pt",
            )
    finally:
        del sampler
        torch.cuda.empty_cache()
        torch.backends.cuda.matmul.allow_tf32 = previous_matmul_tf32
        torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32

    result["finished_at_unix"] = time.time()
    _require_finite_scalars(result)
    _atomic_json(output / "temperature_calibration.json", result)
    tracker.upload_artifact("temperature_calibration", output / "temperature_calibration.json")
    for temperature, entry in result["calibration"].items():
        if "comparison_to_tau_1" in entry:
            tracker.report_scalar("temperature/calibration", f"{temperature}/J", entry["comparison_to_tau_1"]["J_mean_fair_crps_ratio"], 0)
    if "acceptance_gate" in result:
        tracker.report_scalar("temperature/confirmation", "J", result["confirmation_comparison"]["J_mean_fair_crps_ratio"], 0)
        tracker.report_scalar("temperature/confirmation", "bootstrap_J_upper95", result["paired_date_bootstrap_J_upper95"], 0)
    tracker.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.run_dir.resolve(), args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
