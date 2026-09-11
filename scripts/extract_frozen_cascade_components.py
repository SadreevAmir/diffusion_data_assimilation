#!/usr/bin/env python3
"""Replay one frozen cascade in its compatible source worktree and save tensors.

Run this script with PYTHONPATH pointing at the exact historical replay
worktree.  It deliberately contains no model implementation, so checkpoint
manifest checks remain those of the historical code rather than being relaxed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import torch

from assim_lib.config import load_json
from assim_lib.data import build_dataset
from assim_lib.direct_dynamics_cascade_e2e_evaluation import _verify_inventory
from assim_lib.direct_dynamics_cascade_end_to_end import load_cascade_predictor
from assim_lib.direct_dynamics_cascade_fine_training import _fine_collate
from assim_lib.direct_dynamics_training import validate_direct_dataset


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_commit(worktree: Path) -> str:
    commit = subprocess.run(
        ["git", "-c", f"safe.directory={worktree}", "-C", str(worktree), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-c", f"safe.directory={worktree}", "-C", str(worktree),
         "status", "--porcelain=v1", "--untracked-files=all"],
        check=True, capture_output=True, text=True,
    ).stdout
    if status:
        raise RuntimeError("source replay worktree is not clean")
    return commit


def _atomic_torch_save(payload: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@torch.no_grad()
def run(config_path: Path, label: str, source_worktree: Path, expected_replay_commit: str, output: Path) -> None:
    if torch.cuda.device_count() != 1:
        raise RuntimeError("component extraction requires exactly one visible GPU")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to replace {output}")
    replay_commit = _source_commit(source_worktree)
    if replay_commit != expected_replay_commit:
        raise RuntimeError("source worktree is not at the declared replay commit")
    experiment = load_json(config_path)
    matches = [row for row in experiment["candidates"] if row["label"] == label]
    if len(matches) != 1:
        raise ValueError("candidate label is not unique")
    candidate = matches[0]
    coarse = experiment["coarse"]
    coarse_root = Path(coarse["run_dir"])
    fine_root = Path(candidate["fine_run_dir"])
    _verify_inventory(coarse_root, coarse["sha256"], "coarse")
    _verify_inventory(fine_root, candidate["fine_sha256"], label)
    checkpoint = fine_root / candidate["checkpoint"]
    if not checkpoint.is_file() or _sha256(checkpoint) != candidate["checkpoint_sha256"]:
        raise ValueError("frozen checkpoint mismatch")

    fine_metadata = load_json(fine_root / "metadata.json")
    dataset = build_dataset(fine_metadata["data_config"], split="valid")
    validate_direct_dataset(dataset)
    batch = _fine_collate([dataset[index] for index in experiment["case_indices"]])
    case_ids = tuple(batch["meta"]["case_id"])
    if list(case_ids) != experiment["case_ids"]:
        raise ValueError("frozen case identities differ from archive")
    device = torch.device("cuda:0")
    predictor = load_cascade_predictor(
        coarse_run_dir=str(coarse_root), coarse_checkpoint_name=coarse["checkpoint"],
        coarse_model_config=load_json(coarse_root / "config.json"),
        coarse_checkpoint_sha256=coarse["sha256"][coarse["checkpoint"]],
        fine_run_dir=str(fine_root), fine_checkpoint_name=candidate["checkpoint"],
        fine_model_config=load_json(fine_root / "config.json"),
        fine_checkpoint_sha256=candidate["checkpoint_sha256"],
        expected_coarse_code_commit=coarse["code_commit"],
        expected_fine_code_commit=candidate["fine_code_commit"],
        replay_code_commit=replay_commit,
        expected_forecast_contract_sha256=experiment["forecast_contract_sha256"], device=device,
    )
    ensemble = predictor.sample_ensemble(
        member_indices=tuple(range(8)),
        structured_conditioning=batch["structured_conditioning"].float().to(device),
        valid_mask=batch["valid_mask"][:, :1].float().to(device),
        case_ids=case_ids, coarse_num_timesteps=17, fine_num_timesteps=17,
        device=device, method="rk4", rtol=1e-5, atol=1e-6, end_time=0.0,
    )
    payload = {
        "schema_version": "frozen_cascade_components_v1",
        "candidate": label,
        "replay_commit": replay_commit,
        "fine_training_commit": candidate["fine_code_commit"],
        "fine_checkpoint_sha256": candidate["checkpoint_sha256"],
        "case_ids": case_ids,
        "member_indices": ensemble["member_indices"],
        "coarse_seeds": ensemble["coarse_seeds"],
        "fine_seeds": ensemble["fine_seeds"],
        "solver": ensemble["solver"],
        "coarse": ensemble["coarse"].float().cpu(),
        "residual": ensemble["residual"].float().cpu(),
        "truth": batch["truth"].float().cpu(),
        "valid": batch["valid_mask"][:, :1].float().cpu(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_torch_save(payload, output)
    print(json.dumps({
        "status": "complete", "candidate": label, "replay_commit": replay_commit,
        "output": str(output), "sha256": _sha256(output), "bytes": output.stat().st_size,
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--source-worktree", required=True, type=Path)
    parser.add_argument("--expected-replay-commit", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.config.resolve(), args.candidate, args.source_worktree.resolve(),
        args.expected_replay_commit, args.output.resolve())


if __name__ == "__main__":
    main()
