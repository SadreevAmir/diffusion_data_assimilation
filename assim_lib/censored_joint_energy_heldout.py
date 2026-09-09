"""Bounded train-to-validation feasibility pilot for the censored joint law."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import default_collate

from .bounded_clean_state_overfit import diagnostic_indices
from .censored_joint_energy_overfit import (
    BASE_LEARNING_RATE,
    CONDITION_BATCH_SIZE,
    DIAGNOSTIC_CHUNK,
    DIAGNOSTIC_MEMBERS,
    ENSEMBLE_MEMBERS,
    PREFETCH_FACTOR,
    SOURCE_CONFIG_SHA256,
    TRAIN_WORKERS,
    WARMUP_UPDATES,
    _atomic_torch_save,
    _finite_scalars,
    _generate_members,
    _initialize_small_nonzero_head,
    _ipc_preflight,
    _sha256,
    joint_field_energy_score,
)
from .censored_joint_multiscale_score import (
    PATCHES_PER_CONDITION,
    multiscale_joint_energy_score,
    sample_valid_centres,
)
from .censored_joint_spatial_diagnostics import (
    OUTPUT_NAMES,
    SPATIAL_LAGS,
    _plot_case_maps,
    mean_absolute_increment,
)
from .clearml_tracking import ClearMLTracker
from .config import TrainingConfig, load_json, merge_config_overrides, resolve_path
from .data import build_dataset
from .direct_dynamics_training import (
    DIRECT_INPUT_CHANNELS,
    DIRECT_LEADS,
    DIRECT_OUTPUT_CHANNELS,
    _repeat_field_stats,
    validate_direct_dataset,
)
from .model_io import build_unet
from .runtime import build_dataloader, make_normalized_xy_grid, seed_everything
from .structured_trajectory_evaluation import make_structured_trajectory_figure
from .trainer import _atomic_json


MAX_UPDATES = 1024
DIAGNOSTIC_UPDATES = (0, 256, 512, 1024)
VALIDATION_INDICES = (
    120,
    846,
    1572,
    2298,
    3000,
    3726,
    4452,
    5178,
    5880,
    6606,
    7332,
    8058,
)
VALIDATION_MANIFEST = (
    (120, "2022-01-06", 0),
    (846, "2022-02-05", 6),
    (1572, "2022-03-07", 12),
    (2298, "2022-04-06", 18),
    (3000, "2022-05-06", 0),
    (3726, "2022-06-05", 6),
    (4452, "2022-07-05", 12),
    (5178, "2022-08-04", 18),
    (5880, "2022-09-03", 0),
    (6606, "2022-10-03", 6),
    (7332, "2022-11-02", 12),
    (8058, "2022-12-02", 18),
)
REFERENCE_COMMIT = "b3d534dca25484347565c3942b98c84759cd5c1e"
DIAGNOSTIC_SEED_OFFSET = 202
PATCH_TRAIN_SEED_OFFSET = 909
PATCH_DIAGNOSTIC_SEED_OFFSET = 1001
DIAGNOSTIC_PATCH_CENTRES = 128
EXPECTED_TRAIN_CASES = 51_792
EXPECTED_TRAIN_BATCHES = 6_474
EXPECTED_INVENTORY_SHA256 = {
    "train": "514249bfed2b98e7b70ca80f7749e3ed4086b3364b647719cea1c90b781117ea",
    "valid": "0298710a8b4cbb031dd78864281e4b049a60251bc8fccb664056118a3d1e1c97",
}


def _tensor_sha256(value: torch.Tensor) -> str:
    contiguous = value.detach().cpu().contiguous()
    return hashlib.sha256(contiguous.numpy().tobytes()).hexdigest()


def _calendar_inventory(
    dataset, *, split: str, first_year: int, last_year: int
) -> dict[str, Any]:
    lower = date(first_year, 1, 1)
    upper = date(last_year, 12, 31)
    first = None
    last = None
    digest = hashlib.sha256()
    for _, target in dataset.calendar_pairs:
        dates = [target.date, *(target.date + timedelta(days=lead) for lead in DIRECT_LEADS)]
        if any(value < lower or value > upper for value in dates):
            raise ValueError("trajectory date escapes its declared split year range")
        if any(value not in dataset.records_by_date for value in dates):
            raise FileNotFoundError("calendar inventory lacks a declared trajectory target")
        paths = [dataset.records_by_date[value].path.name for value in dates]
        digest.update(
            ("|".join([*(value.isoformat() for value in dates), *paths]) + "\n").encode()
        )
        first = dates[0] if first is None else min(first, dates[0])
        last = dates[-1] if last is None else max(last, dates[-1])
    if first is None or last is None:
        raise ValueError("calendar inventory is empty")
    fingerprint = digest.hexdigest()
    if fingerprint != EXPECTED_INVENTORY_SHA256[split]:
        raise ValueError(f"{split} ordered calendar inventory fingerprint differs")
    return {
        "calendar_pairs": len(dataset.calendar_pairs),
        "first_d0": first.isoformat(),
        "last_target": last.isoformat(),
        "allowed_years": [first_year, last_year],
        "ordered_inventory_sha256": fingerprint,
    }


def _enforce_train_envelope(train_cases: int, train_batches: int) -> None:
    if train_cases != EXPECTED_TRAIN_CASES or train_batches != EXPECTED_TRAIN_BATCHES:
        raise ValueError(
            "full train envelope differs: "
            f"cases={train_cases}, batches={train_batches}"
        )


def _validate_objective_setup(
    objective: str,
    step_zero_checkpoint: Path | None,
    expected_step_zero_sha256: str | None,
) -> None:
    if objective not in {"global", "multiscale"}:
        raise ValueError("objective must be global or multiscale")
    if objective == "multiscale":
        if step_zero_checkpoint is None or expected_step_zero_sha256 is None:
            raise ValueError("multiscale objective requires frozen step-zero identity")
        if _sha256(step_zero_checkpoint) != expected_step_zero_sha256:
            raise ValueError("step-zero checkpoint SHA differs from frozen baseline")
    elif step_zero_checkpoint is not None or expected_step_zero_sha256 is not None:
        raise ValueError("global baseline does not accept a step-zero override")


class _RunBoundary:
    """Close ClearML and retain exact progress for every post-tracker failure."""

    def __init__(self, tracker, output_dir: Path, progress: dict[str, Any]):
        self.tracker = tracker
        self.output_dir = output_dir
        self.progress = progress

    def __enter__(self):
        return self

    def _failure_payload(self, error_type, error, *, status: str) -> dict[str, Any]:
        return {
            "status": status,
            "attempted_update": self.progress["attempted_update"],
            "completed_optimizer_steps": self.progress[
                "completed_optimizer_steps"
            ],
            "error_type": error_type.__name__,
            "message": str(error),
            "losses": list(self.progress["losses"]),
            "checkpoints": self.progress["checkpoints"],
            "diagnostic_updates_complete": sorted(self.progress["diagnostics"]),
        }

    def _record_failure(self, failure: dict[str, Any], *, connect: bool) -> None:
        try:
            _atomic_json(self.output_dir / "failure.json", failure)
        except BaseException:
            pass
        if connect:
            try:
                self.tracker.connect(
                    "censored_joint_energy_heldout_failure", failure
                )
            except BaseException:
                pass

    def __exit__(self, error_type, error, traceback) -> bool:
        if error is not None:
            failure = self._failure_payload(error_type, error, status="failed")
            self._record_failure(failure, connect=True)
            try:
                self.tracker.close()
            except BaseException:
                # Keep the primary experiment error as the propagated exception.
                pass
            return False
        try:
            self.tracker.close()
        except BaseException as close_error:
            failure = self._failure_payload(
                type(close_error), close_error, status="failed_finalization"
            )
            self._record_failure(failure, connect=False)
            raise
        return False


def _save_then_evaluate_diagnostic(
    *,
    stage: Path,
    payload: dict[str, Any],
    identity: dict[str, Any],
    evaluator=None,
) -> dict[str, Any]:
    stage.mkdir(parents=True, exist_ok=False)
    _atomic_torch_save({**payload, **identity}, stage / "samples.pt")
    evaluate = evaluate_group if evaluator is None else evaluator
    try:
        return evaluate(payload)
    except BaseException as error:
        _atomic_json(
            stage / "diagnostic_failure.json",
            {"error_type": type(error).__name__, "message": str(error), **identity},
        )
        raise


def _validate_anchor_manifest(dataset) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    items = []
    records = []
    windows = []
    for expected_index, expected_date, expected_slice in VALIDATION_MANIFEST:
        item = dataset[expected_index]
        meta = item["meta"]
        if meta["split"] != "valid" or meta["target_date"] != expected_date:
            raise ValueError("validation anchor identity differs from frozen manifest")
        if int(meta["archive_slice_index"]) != expected_slice:
            raise ValueError("validation archive slice differs from frozen manifest")
        target_date = date.fromisoformat(expected_date)
        expected_dates = [target_date + timedelta(days=lead) for lead in DIRECT_LEADS]
        expected_basenames = [f"ocean+atmosphere_24_{value.isoformat()}.npy" for value in expected_dates]
        actual_basenames = [Path(value).name for value in meta["target_trajectory_paths"]]
        if actual_basenames != expected_basenames:
            raise ValueError("validation trajectory paths differ from frozen manifest")
        if meta["background_role"] != "persistence_baseline_not_model_condition":
            raise ValueError("persistence leaked into declared model conditioning role")
        for name in ("truth", "structured_conditioning"):
            if not torch.all(torch.isfinite(item[name])):
                raise FloatingPointError(f"validation anchor {name} contains NaN/Inf")
        windows.append((target_date, target_date + timedelta(days=max(DIRECT_LEADS))))
        records.append(
            {
                "dataset_index": expected_index,
                "case_id": meta["case_id"],
                "d0": expected_date,
                "archive_slice_index": expected_slice,
                "target_dates": [value.isoformat() for value in expected_dates],
                "target_basenames": expected_basenames,
            }
        )
        items.append(item)
    if len({record["case_id"] for record in records}) != len(records):
        raise ValueError("validation manifest has duplicate case IDs")
    for index, left in enumerate(windows):
        for right in windows[index + 1 :]:
            if max(left[0], right[0]) <= min(left[1], right[1]):
                raise ValueError("validation trajectory windows overlap")
    return items, {"status": "passed", "anchors": records}


def _case_fair_crps(members: torch.Tensor, truth: torch.Tensor) -> float:
    count = members.shape[0]
    if count < 2:
        raise ValueError("fair CRPS requires at least two members")
    observation = (members - truth).abs().mean(dim=0)
    pair = (members[:, None] - members[None, :]).abs().sum(dim=(0, 1))
    score = observation - pair / (2.0 * count * (count - 1))
    return float(score.mean())


def _fractional_rank_histogram(
    members: torch.Tensor, truth: torch.Tensor
) -> list[float]:
    count = members.shape[0]
    lower = (members < truth).sum(dim=0)
    equal = (members == truth).sum(dim=0)
    bins = torch.zeros(count + 1, dtype=torch.float64)
    for rank in range(count + 1):
        weight = ((lower <= rank) & (rank <= lower + equal)).to(torch.float64)
        weight /= (equal + 1).to(torch.float64)
        bins[rank] = weight.sum()
    bins /= truth.numel()
    return [float(value) for value in bins]


def _correlation_matrix(anomalies: torch.Tensor) -> dict[str, Any]:
    matrix: list[list[float | None]] = []
    reasons: list[list[str | None]] = []
    for left in range(anomalies.shape[1]):
        row = []
        reason_row = []
        for right in range(anomalies.shape[1]):
            x = anomalies[:, left]
            y = anomalies[:, right]
            if float(x.square().sum()) == 0.0 or float(y.square().sum()) == 0.0:
                row.append(None)
                reason_row.append("zero_member_variance")
            else:
                row.append(float((x * y).sum() / torch.sqrt(x.square().sum() * y.square().sum())))
                reason_row.append(None)
        matrix.append(row)
        reasons.append(reason_row)
    return {"correlation": matrix, "null_reasons": reasons}


def evaluate_group(payload: dict[str, torch.Tensor]) -> dict[str, Any]:
    ensemble = payload["physical_ensemble"].float()
    truth = payload["truth"].float()
    persistence = payload["persistence"].float()
    valid = payload["valid_mask"][:, 0].bool()
    if ensemble.ndim != 5 or ensemble.shape[1:3] != (DIAGNOSTIC_MEMBERS, 6):
        raise ValueError("diagnostic ensemble has an unexpected shape")
    active = valid[:, None, None].expand_as(ensemble)
    if not torch.all(torch.isfinite(ensemble[active])):
        raise FloatingPointError("physical diagnostic ensemble contains NaN/Inf")
    outputs = {}
    hard_failures = []
    case_records: dict[str, list[dict[str, Any]]] = {name: [] for name in OUTPUT_NAMES}
    for channel, name in enumerate(OUTPUT_NAMES):
        rank_histograms = []
        for case in range(ensemble.shape[0]):
            mask = valid[case]
            members = ensemble[case, :, channel][:, mask]
            target = truth[case, channel][mask]
            baseline = persistence[case, channel][mask]
            mean = members.mean(dim=0)
            member_variance = members.var(dim=0, unbiased=True)
            histogram = _fractional_rank_histogram(members, target)
            rank_histograms.append(histogram)
            heterogeneous = bool(float(target.max() - target.min()) > 0.0)
            exact_collapse = heterogeneous and torch.equal(
                members[1:], members[:1].expand_as(members[1:])
            )
            if exact_collapse:
                hard_failures.append(f"exact_collapse:{case}:{name}")
            increments = {}
            case_valid = valid[case : case + 1]
            for lag in SPATIAL_LAGS:
                increments[str(lag)] = {
                    "individual_members": [
                        mean_absolute_increment(
                            ensemble[case : case + 1, member, channel], case_valid, lag
                        )
                        for member in range(DIAGNOSTIC_MEMBERS)
                    ],
                    "ensemble_mean": mean_absolute_increment(
                        ensemble[case : case + 1, :, channel].mean(dim=1), case_valid, lag
                    ),
                    "truth": mean_absolute_increment(
                        truth[case : case + 1, channel], case_valid, lag
                    ),
                    "persistence": mean_absolute_increment(
                        persistence[case : case + 1, channel], case_valid, lag
                    ),
                }
            probability_zero = (members == 0).float().mean(dim=0)
            truth_zero = (target == 0).float()
            atom = {
                "member_zero": float(probability_zero.mean()),
                "truth_zero": float(truth_zero.mean()),
                "zero_brier": float((probability_zero - truth_zero).square().mean()),
            }
            if name.endswith("_sic"):
                probability_one = (members == 1).float().mean(dim=0)
                truth_one = (target == 1).float()
                atom.update(
                    {
                        "member_one": float(probability_one.mean()),
                        "truth_one": float(truth_one.mean()),
                        "one_brier": float(
                            (probability_one - truth_one).square().mean()
                        ),
                    }
                )
            case_records[name].append(
                {
                    "case": case,
                    "ensemble_mean_mse": float((mean - target).square().mean()),
                    "persistence_mse": float((baseline - target).square().mean()),
                    "member_variance": float(member_variance.mean()),
                    "fair_crps": _case_fair_crps(members, target),
                    "persistence_mae": float((baseline - target).abs().mean()),
                    "rank_histogram": histogram,
                    "atoms": atom,
                    "heterogeneous_truth": heterogeneous,
                    "exact_collapse": exact_collapse,
                    "increments": increments,
                }
            )
        records = case_records[name]
        rmse = math.sqrt(sum(record["ensemble_mean_mse"] for record in records) / len(records))
        persistence_rmse = math.sqrt(sum(record["persistence_mse"] for record in records) / len(records))
        spread = math.sqrt(sum(record["member_variance"] for record in records) / len(records))
        crps = sum(record["fair_crps"] for record in records) / len(records)
        persistence_mae = sum(record["persistence_mae"] for record in records) / len(records)
        outputs[name] = {
            "rmse": rmse,
            "persistence_rmse": persistence_rmse,
            "spread": spread,
            "spread_skill_ratio": spread / rmse if rmse > 0 else None,
            "fair_crps": crps,
            "persistence_mae": persistence_mae,
            "fair_crps_ratio": crps / persistence_mae if persistence_mae > 0 else None,
            "fair_crps_ratio_null_reason": None if persistence_mae > 0 else "zero_baseline_score",
            "rank_histogram": [sum(values) / len(values) for values in zip(*rank_histograms, strict=True)],
            "rank_total_variation_from_uniform": 0.5
            * sum(
                abs(value - 1.0 / (DIAGNOSTIC_MEMBERS + 1))
                for value in [
                    sum(values) / len(values)
                    for values in zip(*rank_histograms, strict=True)
                ]
            ),
            "boundary_events": {
                key: sum(record["atoms"][key] for record in records) / len(records)
                for key in records[0]["atoms"]
            },
            "cases": records,
        }

    dependence = []
    for case in range(ensemble.shape[0]):
        mask = valid[case]
        record: dict[str, Any] = {"case": case}
        for field, channels in (("sic", (0, 2, 4)), ("sit", (1, 3, 5))):
            member_means = torch.stack(
                [ensemble[case, :, channel][:, mask].mean(dim=1) for channel in channels],
                dim=1,
            )
            anomalies = member_means - member_means.mean(dim=0, keepdim=True)
            truth_means = [float(truth[case, channel][mask].mean()) for channel in channels]
            persistence_means = [float(persistence[case, channel][mask].mean()) for channel in channels]
            record[field] = {
                **_correlation_matrix(anomalies),
                "truth_spatial_means": truth_means,
                "truth_changes": [truth_means[1] - truth_means[0], truth_means[2] - truth_means[1], truth_means[2] - truth_means[0]],
                "persistence_spatial_means": persistence_means,
                "persistence_changes": [persistence_means[1] - persistence_means[0], persistence_means[2] - persistence_means[1], persistence_means[2] - persistence_means[0]],
            }
        dependence.append(record)
    support = {
        "sic_below_zero": int(((ensemble[:, :, 0::2] < 0) & valid[:, None, None]).sum()),
        "sic_above_one": int(((ensemble[:, :, 0::2] > 1) & valid[:, None, None]).sum()),
        "sit_below_zero": int(((ensemble[:, :, 1::2] < 0) & valid[:, None, None]).sum()),
    }
    if any(support.values()):
        hard_failures.append("physical_support")
    result = {
        "outputs": outputs,
        "cross_lead_dependence": dependence,
        "support": support,
        "hard_failures": hard_failures,
    }
    _finite_scalars(result)
    return result


def _case_equal_multiscale_diagnostic(
    payload: dict[str, torch.Tensor],
    *,
    stds: tuple[float, ...],
    centres: torch.Tensor,
) -> tuple[float, dict[str, float]]:
    totals = []
    component_values: dict[str, list[float]] = {}
    for case in range(payload["truth"].shape[0]):
        total, components = multiscale_joint_energy_score(
            payload["physical_ensemble"][case : case + 1],
            payload["truth"][case : case + 1],
            payload["valid_mask"][case : case + 1],
            stds=stds,
            centres=centres[case : case + 1],
            expected_centres=None,
        )
        totals.append(float(total))
        for name, value in components.items():
            component_values.setdefault(name, []).append(float(value))
    return sum(totals) / len(totals), {
        name: sum(values) / len(values)
        for name, values in component_values.items()
    }


def _slice_tensor_batch(batch: dict[str, Any], case: int) -> dict[str, Any]:
    return {
        key: value[case : case + 1] if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


@torch.no_grad()
def _diagnose_group(
    *,
    model,
    batch: dict[str, Any],
    metadata: list[dict[str, Any]],
    noises: torch.Tensor,
    group: str,
    update: int,
    output_dir: Path,
    tracker: ClearMLTracker,
    means: tuple[float, ...],
    stds: tuple[float, ...],
    checkpoint: dict[str, Any],
    device: torch.device,
    score_centres: torch.Tensor | None = None,
) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    try:
        grid = make_normalized_xy_grid(*batch["truth"].shape[-2:], device=device)
        case_ensembles = []
        case_latents = []
        for case in range(batch["truth"].shape[0]):
            one = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in _slice_tensor_batch(batch, case).items()
            }
            physical_chunks = []
            latent_chunks = []
            for start in range(0, DIAGNOSTIC_MEMBERS, DIAGNOSTIC_CHUNK):
                physical, latent = _generate_members(
                    model,
                    condition=one["structured_conditioning"].float(),
                    background=one["background"].float(),
                    valid=one["valid_mask"][:, :1].float(),
                    noise=noises[case : case + 1, start : start + DIAGNOSTIC_CHUNK].to(device),
                    grid=grid,
                    means=means,
                    stds=stds,
                )
                physical_chunks.append(physical.cpu())
                latent_chunks.append(latent.cpu())
            case_ensembles.append(torch.cat(physical_chunks, dim=1))
            case_latents.append(torch.cat(latent_chunks, dim=1))
        payload = {
            "physical_ensemble": torch.cat(case_ensembles, dim=0),
            "uncensored_latent_normalized": torch.cat(case_latents, dim=0),
            "truth": batch["structured_physical_truth"].float(),
            "persistence": batch["structured_physical_background"].float(),
            "valid_mask": batch["valid_mask"][:, :1].float(),
        }
        noise_sha256 = _tensor_sha256(noises)
        stage = output_dir / "diagnostics" / f"update_{update:04d}" / group
        identity = {
            "metadata": metadata,
            "checkpoint": checkpoint,
            "noise_sha256": noise_sha256,
            "group": group,
            "update": update,
        }
        if score_centres is not None:
            payload["multiscale_score_centres"] = score_centres
            identity["multiscale_score_centres_sha256"] = _tensor_sha256(
                score_centres
            )

        def diagnostic_evaluator(values: dict[str, torch.Tensor]) -> dict[str, Any]:
            evaluated = evaluate_group(values)
            if score_centres is not None:
                multiscale_score, multiscale_components = (
                    _case_equal_multiscale_diagnostic(
                        values,
                        stds=stds,
                        centres=score_centres,
                    )
                )
                evaluated["multiscale_score"] = multiscale_score
                evaluated["multiscale_components"] = multiscale_components
            return evaluated

        metrics = _save_then_evaluate_diagnostic(
            stage=stage,
            payload=payload,
            identity=identity,
            evaluator=diagnostic_evaluator,
        )
        metrics.update(
            {
                "group": group,
                "update": update,
                "checkpoint": checkpoint,
                "noise_sha256": noise_sha256,
                "metadata": metadata,
            }
        )
        _atomic_json(stage / "metrics.json", metrics)
        for case in range(payload["truth"].shape[0]):
            for member in (0, 1):
                figure = make_structured_trajectory_figure(
                    payload["truth"][case],
                    payload["persistence"][case],
                    payload["physical_ensemble"][case, member],
                    payload["valid_mask"][case],
                    title=f"heldout joint energy {group} update {update}; case {case}, member {member}",
                    origin="upper",
                    lead_days=DIRECT_LEADS,
                )
                image_path = stage / f"case_{case:02d}_member{member}.png"
                figure.savefig(image_path, dpi=160, bbox_inches="tight")
                import matplotlib.pyplot as plt

                plt.close(figure)
                tracker.report_image(
                    f"censored_joint_heldout/{group}/members",
                    f"case_{case:02d}_member{member}",
                    image_path,
                    update,
                )
            map_path = stage / f"case_{case:02d}_maps.png"
            _plot_case_maps(payload, case, map_path)
            tracker.report_image(
                f"censored_joint_heldout/{group}/maps",
                f"case_{case:02d}",
                map_path,
                update,
            )
        for name, values in metrics["outputs"].items():
            for metric in ("rmse", "persistence_rmse", "spread", "spread_skill_ratio", "fair_crps", "persistence_mae", "fair_crps_ratio"):
                value = values[metric]
                if value is not None:
                    tracker.report_scalar(f"censored_joint_heldout/{group}/{metric}", name, value, update)
        if "multiscale_score" in metrics:
            tracker.report_scalar(
                f"censored_joint_heldout/{group}/multiscale_score",
                "total",
                metrics["multiscale_score"],
                update,
            )
            for component_name, component_value in metrics[
                "multiscale_components"
            ].items():
                tracker.report_scalar(
                    f"censored_joint_heldout/{group}/multiscale_components",
                    component_name,
                    component_value,
                    update,
                )
        return metrics
    finally:
        model.train(was_training)


def run(
    config_path: Path,
    output_dir: Path,
    *,
    objective: str = "global",
    step_zero_checkpoint: Path | None = None,
    expected_step_zero_sha256: str | None = None,
) -> dict[str, Any]:
    _validate_objective_setup(
        objective, step_zero_checkpoint, expected_step_zero_sha256
    )
    if torch.cuda.device_count() != 1:
        raise RuntimeError("heldout feasibility pilot requires exactly one visible GPU")
    if os.environ.get("CLEARML_REQUIRE_ONLINE") != "1":
        raise RuntimeError("heldout feasibility pilot requires online ClearML")
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"refusing to reuse heldout output: {output_dir}")
    experiment = load_json(config_path)
    config_dir = config_path.resolve().parent
    data_path = resolve_path(experiment["data_config"], config_dir)
    model_path = resolve_path(experiment["model_config"], config_dir)
    for name, path in (("experiment", config_path), ("data", data_path), ("model", model_path)):
        if _sha256(path) != SOURCE_CONFIG_SHA256[name]:
            raise ValueError(f"{name} config differs from audited source")
    data_config = merge_config_overrides(load_json(data_path), experiment.get("data_overrides"))
    raw_model = load_json(model_path)
    effective_model = dict(raw_model)
    effective_model.update({"dropout": 0.0, "train_batch_size": CONDITION_BATCH_SIZE, "activation_checkpointing": False})
    model_config = TrainingConfig.from_dict(effective_model)
    if model_config.image_size != (320, 256) or model_config.in_channels != DIRECT_INPUT_CHANNELS or model_config.out_channels != DIRECT_OUTPUT_CHANNELS or model_config.dropout != 0.0 or model_config.activation_checkpointing:
        raise ValueError("heldout model differs from frozen contract")
    seed_everything(model_config.seed)
    torch.set_num_threads(6)
    torch.set_num_interop_threads(1)
    train_dataset = build_dataset(data_config, split="train")
    valid_dataset = build_dataset(data_config, split="valid")
    train_sentinel = validate_direct_dataset(train_dataset)
    valid_sentinel = validate_direct_dataset(valid_dataset)
    train_inventory = _calendar_inventory(
        train_dataset, split="train", first_year=2016, last_year=2021
    )
    valid_inventory = _calendar_inventory(
        valid_dataset, split="valid", first_year=2022, last_year=2022
    )
    validation_items, validation_manifest = _validate_anchor_manifest(valid_dataset)
    train_control_indices = diagnostic_indices(len(train_dataset))
    train_control_items = [train_dataset[index] for index in train_control_indices]
    ipc_preflight = _ipc_preflight(train_dataset, tuple(range(CONDITION_BATCH_SIZE)))
    loader_generator = torch.Generator(device="cpu").manual_seed(model_config.seed + 41)
    loader = build_dataloader(train_dataset, CONDITION_BATCH_SIZE, TRAIN_WORKERS, shuffle=True, prefetch_factor=PREFETCH_FACTOR, generator=loader_generator)
    _enforce_train_envelope(len(train_dataset), len(loader))
    output_dir.mkdir(parents=True, exist_ok=False)
    tracker = ClearMLTracker(
        "sea_ice_two_stage",
        f"censored_joint_{objective}_heldout_{output_dir.name}",
        tags=["censored-joint-law", f"{objective}-energy-score", "heldout-feasibility", "validation-2022-exploratory", "one-gpu", "no-ode", "no-member-mse", "dropout-zero"],
        env_path=experiment.get("clearml", {}).get("env_path"),
    )
    contract = {
        "purpose": "exploratory_heldout_feasibility",
        "terminal_scientific_status": "complete_pending_independent_review",
        "reference_commit": REFERENCE_COMMIT,
        "test_2023_accessed": False,
        "updates": MAX_UPDATES,
        "diagnostic_updates": DIAGNOSTIC_UPDATES,
        "validation_manifest": validation_manifest,
        "train_control_indices": train_control_indices,
        "train_sentinel": train_sentinel,
        "valid_sentinel": valid_sentinel,
        "train_inventory": train_inventory,
        "valid_inventory": valid_inventory,
        "condition_batch_size": CONDITION_BATCH_SIZE,
        "maximum_network_batch": CONDITION_BATCH_SIZE * ENSEMBLE_MEMBERS,
        "training_members": ENSEMBLE_MEMBERS,
        "diagnostic_members": DIAGNOSTIC_MEMBERS,
        "workers": TRAIN_WORKERS,
        "prefetch_factor": PREFETCH_FACTOR,
        "ipc_preflight": ipc_preflight,
        "objective": (
            "unbiased_joint_energy_score"
            if objective == "global"
            else "global_plus_multiscale_patch_unbiased_joint_energy_score"
        ),
        "objective_mode": objective,
        "step_zero_checkpoint": (
            None
            if step_zero_checkpoint is None
            else {
                "path": str(step_zero_checkpoint),
                "sha256": expected_step_zero_sha256,
            }
        ),
        "patch_training": (
            None
            if objective == "global"
            else {
                "sizes": [8, 16],
                "centres_per_condition": PATCHES_PER_CONDITION,
                "weights": {"global": 0.5, "patch_8": 0.25, "patch_16": 0.25},
                "seed_offset": PATCH_TRAIN_SEED_OFFSET,
                "selection": "uniform_valid_ocean_with_replacement_nested",
            }
        ),
        "diagnostic_patch_centres": (
            None if objective == "global" else DIAGNOSTIC_PATCH_CENTRES
        ),
        "generator": "persistence_centered_censored_joint_residual",
        "optimizer": "AdamW",
        "learning_rate": BASE_LEARNING_RATE,
        "warmup_updates": WARMUP_UPDATES,
        "precision": "bf16_network_fp32_score",
        "dropout": 0.0,
        "ema": False,
        "activation_checkpointing": False,
        "source_config_sha256": SOURCE_CONFIG_SHA256,
        "effective_data_config": data_config,
        "source_model_config": raw_model,
        "effective_model_config": effective_model,
    }
    losses: list[float] = []
    diagnostics: dict[int, dict[str, Any]] = {}
    checkpoints: dict[int, dict[str, Any]] = {}
    progress = {
        "attempted_update": 0,
        "completed_optimizer_steps": 0,
        "losses": losses,
        "diagnostics": diagnostics,
        "checkpoints": checkpoints,
    }
    with _RunBoundary(tracker, output_dir, progress):
        tracker.connect("censored_joint_energy_heldout_contract", contract)
        _atomic_json(output_dir / "contract.json", contract)
        device = torch.device("cuda:0")
        model = build_unet(model_config).to(device)
        head_init = _initialize_small_nonzero_head(model, seed=model_config.seed + 313)
        if objective == "multiscale":
            try:
                step_zero_state = torch.load(
                    step_zero_checkpoint, map_location="cpu", weights_only=True
                )
            except TypeError:
                step_zero_state = torch.load(step_zero_checkpoint, map_location="cpu")
            model.load_state_dict(step_zero_state)
            head_init = {
                **head_init,
                "overridden_by_frozen_step_zero": True,
                "frozen_step_zero_sha256": expected_step_zero_sha256,
            }
        optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LEARNING_RATE)
        means = _repeat_field_stats(train_dataset.means)
        stds = _repeat_field_stats(train_dataset.stds)
        grid = make_normalized_xy_grid(*model_config.image_size, device=device)
        training_noise = torch.Generator(device=device).manual_seed(model_config.seed + 101)
        patch_generator = torch.Generator(device=device).manual_seed(
            model_config.seed + PATCH_TRAIN_SEED_OFFSET
        )
        diagnostic_generator = torch.Generator(device="cpu").manual_seed(
            model_config.seed + DIAGNOSTIC_SEED_OFFSET
        )
        validation_noise = torch.randn(
            (
                len(validation_items),
                DIAGNOSTIC_MEMBERS,
                DIRECT_OUTPUT_CHANNELS,
                *model_config.image_size,
            ),
            generator=diagnostic_generator,
        )
        train_control_noise = torch.randn(
            (
                len(train_control_items),
                DIAGNOSTIC_MEMBERS,
                DIRECT_OUTPUT_CHANNELS,
                *model_config.image_size,
            ),
            generator=diagnostic_generator,
        )
        validation_batch = default_collate(validation_items)
        train_control_batch = default_collate(train_control_items)
        validation_metadata = [item["meta"] for item in validation_items]
        train_control_metadata = [item["meta"] for item in train_control_items]
        diagnostic_patch_generator = torch.Generator(device="cpu").manual_seed(
            model_config.seed + PATCH_DIAGNOSTIC_SEED_OFFSET
        )
        validation_score_centres = (
            None
            if objective == "global"
            else sample_valid_centres(
                validation_batch["valid_mask"][:, :1],
                DIAGNOSTIC_PATCH_CENTRES,
                generator=diagnostic_patch_generator,
            )
        )
        train_control_score_centres = (
            None
            if objective == "global"
            else sample_valid_centres(
                train_control_batch["valid_mask"][:, :1],
                DIAGNOSTIC_PATCH_CENTRES,
                generator=diagnostic_patch_generator,
            )
        )
        patch_centres_log: list[torch.Tensor] = []

        def snapshot(update: int) -> None:
            checkpoint_path = output_dir / f"model_step_{update:04d}.pth"
            _atomic_torch_save(model.state_dict(), checkpoint_path)
            checkpoint = {
                "path": str(checkpoint_path),
                "sha256": _sha256(checkpoint_path),
                "update": update,
                "weight_source": "raw",
            }
            if objective == "multiscale":
                patch_history_path = (
                    output_dir / f"patch_centres_through_step_{update:04d}.pth"
                )
                patch_history = (
                    torch.stack(patch_centres_log)
                    if patch_centres_log
                    else torch.empty(
                        0,
                        CONDITION_BATCH_SIZE,
                        PATCHES_PER_CONDITION,
                        2,
                        dtype=torch.int64,
                    )
                )
                _atomic_torch_save(patch_history, patch_history_path)
                checkpoint["patch_centres"] = {
                    "path": str(patch_history_path),
                    "sha256": _sha256(patch_history_path),
                    "completed_updates": len(patch_centres_log),
                }
            checkpoints[update] = checkpoint
            _atomic_json(output_dir / f"model_step_{update:04d}.json", checkpoint)
            tracker.connect(f"censored_joint_heldout_snapshot_{update:04d}", checkpoint)
            diagnostics[update] = {
                "validation": _diagnose_group(
                    model=model,
                    batch=validation_batch,
                    metadata=validation_metadata,
                    noises=validation_noise,
                    group="validation",
                    update=update,
                    output_dir=output_dir,
                    tracker=tracker,
                    means=means,
                    stds=stds,
                    checkpoint=checkpoint,
                    device=device,
                    score_centres=validation_score_centres,
                ),
                "train_control": _diagnose_group(
                    model=model,
                    batch=train_control_batch,
                    metadata=train_control_metadata,
                    noises=train_control_noise,
                    group="train_control",
                    update=update,
                    output_dir=output_dir,
                    tracker=tracker,
                    means=means,
                    stds=stds,
                    checkpoint=checkpoint,
                    device=device,
                    score_centres=train_control_score_centres,
                ),
            }

        iterator = iter(loader)
        snapshot(0)
        for update in range(1, MAX_UPDATES + 1):
            progress["attempted_update"] = update
            try:
                raw_batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                raw_batch = next(iterator)
            batch = {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in raw_batch.items()}
            valid = batch["valid_mask"][:, :1].float()
            noise = torch.randn((CONDITION_BATCH_SIZE, ENSEMBLE_MEMBERS, DIRECT_OUTPUT_CHANNELS, *model_config.image_size), generator=training_noise, device=device)
            members, _ = _generate_members(model, condition=batch["structured_conditioning"].float(), background=batch["background"].float(), valid=valid, noise=noise, grid=grid, means=means, stds=stds)
            if objective == "global":
                loss = joint_field_energy_score(
                    members,
                    batch["structured_physical_truth"].float(),
                    valid,
                    stds=stds,
                )
                loss_components = {"global": loss}
            else:
                patch_centres = sample_valid_centres(
                    valid,
                    PATCHES_PER_CONDITION,
                    generator=patch_generator,
                )
                patch_centres_log.append(patch_centres.detach().cpu())
                loss, loss_components = multiscale_joint_energy_score(
                    members,
                    batch["structured_physical_truth"].float(),
                    valid,
                    stds=stds,
                    centres=patch_centres,
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            if not torch.isfinite(gradient_norm) or (update == 1 and gradient_norm <= 0):
                raise FloatingPointError("heldout gradient is non-finite or dead")
            learning_rate = BASE_LEARNING_RATE * min(update / WARMUP_UPDATES, 1.0)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            optimizer.step()
            progress["completed_optimizer_steps"] = update
            value = float(loss.detach())
            losses.append(value)
            if update == 1 or update % 8 == 0:
                tracker.report_scalar("censored_joint_heldout/train", "energy_score", value, update)
                tracker.report_scalar("censored_joint_heldout/train", "learning_rate", learning_rate, update)
                for component_name, component_value in loss_components.items():
                    tracker.report_scalar(
                        "censored_joint_heldout/train_components",
                        component_name,
                        float(component_value.detach()),
                        update,
                    )
            if update in DIAGNOSTIC_UPDATES:
                snapshot(update)
                _atomic_json(output_dir / "losses.json", losses)
        result = {
            "status": "complete_pending_independent_review",
            "clearml_task_id": str(tracker.task.id),
            "objective_mode": objective,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "head_initialization": head_init,
            "contract": contract,
            "loss_first_32_mean": sum(losses[:32]) / 32,
            "loss_last_32_mean": sum(losses[-32:]) / 32,
            "diagnostics": diagnostics,
            "checkpoints": checkpoints,
        }
        _finite_scalars(result)
        _atomic_json(output_dir / "result.json", result)
        tracker.connect("censored_joint_energy_heldout_result", result)
        tracker.upload_artifact("censored_joint_energy_heldout_result", output_dir / "result.json")
        return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/experiments/train_direct_dynamics_all_hours_v1.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--objective", choices=("global", "multiscale"), default="global"
    )
    parser.add_argument("--step-zero-checkpoint", type=Path)
    parser.add_argument("--expected-step-zero-sha256")
    arguments = parser.parse_args()
    print(
        json.dumps(
            run(
                arguments.config,
                arguments.output,
                objective=arguments.objective,
                step_zero_checkpoint=arguments.step_zero_checkpoint,
                expected_step_zero_sha256=arguments.expected_step_zero_sha256,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
