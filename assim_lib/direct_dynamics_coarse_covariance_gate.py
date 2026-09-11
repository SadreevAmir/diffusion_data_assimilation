"""CPU-only blocked selection and safety gate for the coarse Gaussian base law."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Subset

from .clearml_tracking import ClearMLTracker
from .config import load_json, merge_config_overrides, resolve_path
from .data import build_dataset
from .direct_dynamics_cascade_coarse import coarse_target
from .direct_dynamics_cascade_coarse_correlated_reference import (
    ActiveCoarseLayout,
    fit_weighted_second_moment_basis,
    identity_reference,
    reference_from_basis,
    select_reference_leave_one_year_out,
    tensor_sha256,
)
from .direct_dynamics_cascade_coarse_residual import (
    coarse_persistence_from_condition,
    standardize_coarse_residual,
    unstandardize_coarse_residual,
)
from .direct_dynamics_cascade_fine_training import (
    EXPECTED_SPLIT_LENGTHS,
    _canonical_sha256,
    _clean_code_identity,
    _fine_collate,
    _sha256_file,
)
from .direct_dynamics_cascade_residual_law_audit import frozen_seasonal_indices
from .direct_dynamics_training import validate_direct_dataset
from .runtime import build_dataloader
from .trainer import _atomic_json


def _load_bound_statistics(spec: dict[str, Any]) -> dict[str, Any]:
    path = Path(spec["path"])
    if not path.is_file() or _sha256_file(path) != spec["sha256"]:
        raise ValueError("SHA-bound standardized-residual statistics differ")
    statistics = json.loads(path.read_text())
    if statistics.get("static_valid_mask_sha256") != spec["static_valid_mask_sha256"]:
        raise ValueError("statistics static-mask binding differs")
    return statistics


def _covariance_oracle(reference, *, seed: int, draws: int = 40000) -> dict[str, float]:
    if reference.is_identity:
        return {"draws": draws, "max_diagonal_error": 0.0, "mean_covariance_error": 0.0}
    # Test a deterministic 64-coordinate marginal; constructing dense d x d K is forbidden.
    count = min(64, reference.dimension)
    indices = torch.linspace(0, reference.dimension - 1, count).round().long().unique()
    basis = reference.low_rank_basis[indices].double()
    h = reference.diagonal_normalizer[indices].double()
    generator = torch.Generator().manual_seed(seed)
    latent_generator = torch.Generator().manual_seed(seed + 1)
    xi = torch.randn((draws, len(indices)), generator=generator, dtype=torch.float64)
    z = torch.randn((draws, reference.rank), generator=latent_generator, dtype=torch.float64)
    samples = h * (
        reference.alpha**0.5 * xi
        + (1.0 - reference.alpha) ** 0.5 * (z @ basis.T)
    )
    empirical = samples.T @ samples / draws
    expected = reference.alpha * torch.diag(h.square()) + (
        1.0 - reference.alpha
    ) * (h[:, None] * basis) @ (h[:, None] * basis).T
    return {
        "draws": draws,
        "max_diagonal_error": float((torch.diag(empirical) - 1).abs().max()),
        "mean_covariance_error": float((empirical - expected).abs().mean()),
        "minimum_expected_eigenvalue": float(torch.linalg.eigvalsh(expected).min()),
        "maximum_expected_offdiagonal": float(
            (expected - torch.diag(torch.diag(expected))).abs().max()
        ),
    }


def run(config_path: Path, output: Path) -> dict[str, Any]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.device_count() != 0:
        raise RuntimeError("coarse covariance gate must be CPU-only")
    if os.environ.get("CLEARML_REQUIRE_ONLINE") != "1":
        raise RuntimeError("coarse covariance gate requires online ClearML")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to replace {output}")
    experiment = load_json(config_path)
    statistics = _load_bound_statistics(experiment["residual_statistics"])
    config_dir = config_path.resolve().parent
    data_path = resolve_path(experiment["data_config"], config_dir)
    data_config = merge_config_overrides(load_json(data_path), experiment.get("data_overrides"))
    dataset = build_dataset(data_config, split="train")
    if len(dataset) != EXPECTED_SPLIT_LENGTHS["train"]:
        raise ValueError("train archive length differs from frozen protocol")
    validate_direct_dataset(dataset)
    indices, case_ids = frozen_seasonal_indices(dataset)
    loader = build_dataloader(
        Subset(dataset, indices), batch_size=int(experiment.get("batch_size", 4)),
        num_workers=int(experiment.get("num_workers", 4)), shuffle=False,
        collate_fn=_fine_collate, prefetch_factor=2,
    )
    layout = None
    vectors = []
    round_trip_max_error = 0.0
    for raw in loader:
        valid = raw["valid_mask"][:, :1].float()
        clean, active, fraction = coarse_target(raw["truth"].float(), valid)
        persistence, persistence_active, persistence_fraction = coarse_persistence_from_condition(
            raw["structured_conditioning"].float(), valid
        )
        if not torch.equal(active, persistence_active) or not torch.equal(fraction, persistence_fraction):
            raise ValueError("target and persistence supports differ")
        if layout is None:
            layout = ActiveCoarseLayout.build(active[:1], fraction[:1])
            if tensor_sha256(valid[:1].contiguous()) != statistics["static_valid_mask_sha256"]:
                raise ValueError("current dataset static mask differs from residual statistics")
        expected_active = active[:1].expand_as(active)
        expected_fraction = fraction[:1].expand_as(fraction)
        if not torch.equal(active, expected_active) or not torch.equal(fraction, expected_fraction):
            raise ValueError("coarse active mask/fraction is not static")
        residual = torch.where(active.expand_as(clean) > 0, clean - persistence, 0.0)
        standardized = standardize_coarse_residual(residual, active, statistics)
        reconstructed = unstandardize_coarse_residual(standardized, active, statistics)
        support = active.expand_as(residual).bool()
        round_trip_max_error = max(
            round_trip_max_error,
            float((reconstructed[support] - residual[support]).abs().max()),
        )
        vectors.append(layout.vectorize(standardized))
    if layout is None:
        raise RuntimeError("empty covariance fit panel")
    values = torch.cat(vectors).reshape(24, 24, layout.dimension)
    selection = select_reference_leave_one_year_out(
        values, layout.coordinate_fraction,
        ranks=tuple(experiment["ranks"]), alphas=tuple(experiment["alphas"]),
        seed=int(experiment["fit_seed"]),
    )
    chosen = selection["selected"]
    if chosen["rank"] == 0:
        reference = identity_reference(layout.dimension)
        candidate_permitted = False
    else:
        maximum_basis = fit_weighted_second_moment_basis(
            values.reshape(-1, layout.dimension), layout.coordinate_fraction,
            max_rank=max(experiment["ranks"]), seed=int(experiment["fit_seed"]),
        )
        reference = reference_from_basis(
            maximum_basis, rank=int(chosen["rank"]), alpha=float(chosen["alpha"])
        )
        candidate_permitted = True
    artifact_path = output.with_suffix(".pt")
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    if artifact_path.exists() or artifact_path.is_symlink():
        raise FileExistsError(f"refusing to replace {artifact_path}")
    torch.save({
        "alpha": reference.alpha,
        "rank": reference.rank,
        "low_rank_basis": reference.low_rank_basis,
        "diagonal_normalizer": reference.diagonal_normalizer,
        "active_cells": layout.active_cells,
        "ocean_fraction": layout.ocean_fraction,
        "channel_order": ["d3_sic", "d3_sit", "d6_sic", "d6_sit", "d9_sic", "d9_sit"],
    }, artifact_path)
    oracle = _covariance_oracle(reference, seed=int(experiment["oracle_seed"]))
    gate_passed = (
        round_trip_max_error <= 2e-6
        and oracle["max_diagonal_error"] <= 0.03
        and oracle["mean_covariance_error"] <= 0.015
        and oracle.get("minimum_expected_eigenvalue", 1.0) > 0
    )
    identity = _clean_code_identity(Path(__file__).resolve().parents[1])
    protocol = {
        "coordinate": "W=S^-1(C-P-mu)",
        "moment": "full second moment E[WW^T], without per-sample centering",
        "blocking": "leave-one-year-out; four dates/year; all 24 slices/date together",
        "train_case_count": 576,
        "train_date_count": 24,
        "case_ids_sha256": _canonical_sha256(case_ids),
        "residual_statistics_path": experiment["residual_statistics"]["path"],
        "residual_statistics_sha256": experiment["residual_statistics"]["sha256"],
        "static_valid_mask_sha256": statistics["static_valid_mask_sha256"],
        "data_config_sha256": _canonical_sha256(data_config),
        "source_config_sha256": _sha256_file(config_path),
        "code_identity": identity,
        "fit_seed": int(experiment["fit_seed"]),
        "oracle_seed": int(experiment["oracle_seed"]),
        "latent_seed_rule": "z_seed = declared_xi_seed + 1000003; never used for K=I",
    }
    tracker = ClearMLTracker(
        experiment["project_name"], experiment["task_name"],
        tags=experiment["clearml"]["tags"], env_path=experiment["clearml"]["env_path"],
    )
    result = {
        "status": "complete" if gate_passed else "failed_gate",
        "gpu_candidate_permitted": bool(gate_passed and candidate_permitted),
        "selection": selection,
        "round_trip_max_error": round_trip_max_error,
        "covariance_oracle": oracle,
        "reference_artifact": str(artifact_path),
        "reference_artifact_sha256": _sha256_file(artifact_path),
        "protocol": protocol,
        "clearml_task_id": str(tracker.task.id),
    }
    _atomic_json(output, result)
    tracker.connect("covariance_gate_protocol", protocol)
    tracker.upload_artifact("coarse_covariance_gate", output)
    tracker.upload_artifact("coarse_covariance_reference", artifact_path)
    tracker.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(6)
    torch.set_num_interop_threads(1)
    print(json.dumps(run(args.config.resolve(), args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
