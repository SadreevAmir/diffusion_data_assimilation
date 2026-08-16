"""Dependency-free oracle for the frozen clean-checkpoint deep ensemble."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any


CHECKPOINT_SEEDS = (1701, 1702, 1703)
LATENT_SEEDS = (2401, 2402, 2403, 2404)
CASE_COUNT = 40
MEMBER_COUNT = 10
SOURCE_EXPERIMENT = "joint_full_condition_validation_2022"
MANIFEST_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class MemberPlan:
    case_index: int
    checkpoint_seed: int
    latent_seed: int

    @property
    def member_id(self) -> str:
        return f"case{self.case_index:02d}-ckpt{self.checkpoint_seed}-z{self.latent_seed}"


def case_plan(case_index: int) -> tuple[MemberPlan, ...]:
    if not 0 <= case_index < CASE_COUNT:
        raise ValueError(f"case_index must be in [0, {CASE_COUNT}), got {case_index}")
    extra_checkpoint = CHECKPOINT_SEEDS[case_index % len(CHECKPOINT_SEEDS)]
    plan = tuple(
        MemberPlan(case_index, checkpoint_seed, latent_seed)
        for checkpoint_seed in CHECKPOINT_SEEDS
        for latent_seed in (
            LATENT_SEEDS if checkpoint_seed == extra_checkpoint else LATENT_SEEDS[:3]
        )
    )
    if len(plan) != MEMBER_COUNT or len({member.member_id for member in plan}) != MEMBER_COUNT:
        raise AssertionError("invalid or duplicate member plan")
    return plan


def validate_contract() -> None:
    extra_counts: Counter[int] = Counter()
    for case_index in range(CASE_COUNT):
        plan = case_plan(case_index)
        counts = Counter(member.checkpoint_seed for member in plan)
        if sorted(counts.values()) != [3, 3, 4]:
            raise AssertionError(f"case {case_index}: expected 4/3/3, got {counts}")
        extra = [seed for seed, count in counts.items() if count == 4]
        if extra != [CHECKPOINT_SEEDS[case_index % 3]]:
            raise AssertionError(f"case {case_index}: incorrect extra checkpoint")
        extra_counts[extra[0]] += 1
    if tuple(extra_counts[seed] for seed in CHECKPOINT_SEEDS) != (14, 13, 13):
        raise AssertionError(f"incorrect forty-case balance: {extra_counts}")


def _require_exact_keys(record: dict[str, Any], expected: set[str], context: str) -> None:
    actual = set(record)
    if actual != expected:
        raise ValueError(
            f"{context}: keys differ; missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )


def validate_admission_manifest(manifest: dict[str, Any]) -> None:
    """Validate the compact pre-score manifest emitted by the trusted runner."""
    _require_exact_keys(manifest, {"schema_version", "source_experiment", "non_seed_configuration_hash", "training_runs", "cases"}, "manifest")
    if manifest["schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise ValueError("manifest: unsupported schema_version")
    if manifest["source_experiment"] != SOURCE_EXPERIMENT:
        raise ValueError("manifest: incorrect source_experiment")
    configuration_hash = manifest["non_seed_configuration_hash"]
    if not isinstance(configuration_hash, str) or not configuration_hash:
        raise ValueError("manifest: non_seed_configuration_hash must be non-empty")

    training_runs = manifest["training_runs"]
    if not isinstance(training_runs, list) or len(training_runs) != len(CHECKPOINT_SEEDS):
        raise ValueError("manifest: exactly three training_runs are required")
    checkpoint_hashes: dict[int, str] = {}
    for expected_seed, run in zip(CHECKPOINT_SEEDS, training_runs, strict=True):
        if not isinstance(run, dict):
            raise ValueError("training_run: expected an object")
        _require_exact_keys(run, {"training_seed", "non_seed_configuration_hash", "selected_checkpoint_hash", "training_completed_normally", "selected_checkpoint_finite", "selection_count"}, f"training_run[{expected_seed}]")
        if run["training_seed"] != expected_seed:
            raise ValueError("training_run: seeds must be exactly 1701,1702,1703 in order")
        if run["non_seed_configuration_hash"] != configuration_hash:
            raise ValueError("training_run: non-seed configuration hashes differ")
        checkpoint_hash = run["selected_checkpoint_hash"]
        if not isinstance(checkpoint_hash, str) or not checkpoint_hash:
            raise ValueError("training_run: selected_checkpoint_hash must be non-empty")
        if run["training_completed_normally"] is not True or run["selected_checkpoint_finite"] is not True:
            raise ValueError("training_run: training and selected checkpoint must be valid")
        if run["selection_count"] != 1:
            raise ValueError("training_run: frozen rule must select exactly one checkpoint")
        checkpoint_hashes[expected_seed] = checkpoint_hash
    if len(set(checkpoint_hashes.values())) != len(CHECKPOINT_SEEDS):
        raise ValueError("training_run: checkpoint identities must be distinct")

    cases = manifest["cases"]
    if not isinstance(cases, list) or len(cases) != CASE_COUNT:
        raise ValueError("manifest: exactly forty ordered cases are required")
    for expected_case_index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError("case: expected an object")
        _require_exact_keys(case, {"case_index", "members"}, f"case[{expected_case_index}]")
        if case["case_index"] != expected_case_index:
            raise ValueError("case: indices must be complete and ordered")
        members = case["members"]
        expected_plan = case_plan(expected_case_index)
        if not isinstance(members, list) or len(members) != MEMBER_COUNT:
            raise ValueError("case: exactly ten members are required")
        noise_by_latent_seed: dict[int, str] = {}
        for expected_member, member in zip(expected_plan, members, strict=True):
            if not isinstance(member, dict):
                raise ValueError("member: expected an object")
            _require_exact_keys(member, {"member_id", "checkpoint_seed", "latent_seed", "checkpoint_hash", "initial_noise_hash", "finite"}, expected_member.member_id)
            identity = (member["member_id"], member["checkpoint_seed"], member["latent_seed"])
            expected_identity = (expected_member.member_id, expected_member.checkpoint_seed, expected_member.latent_seed)
            if identity != expected_identity:
                raise ValueError("member: identity or frozen order differs")
            if member["checkpoint_hash"] != checkpoint_hashes[expected_member.checkpoint_seed]:
                raise ValueError("member: checkpoint identity differs from admitted training run")
            noise_hash = member["initial_noise_hash"]
            if not isinstance(noise_hash, str) or not noise_hash:
                raise ValueError("member: initial_noise_hash must be non-empty")
            prior_hash = noise_by_latent_seed.setdefault(expected_member.latent_seed, noise_hash)
            if prior_hash != noise_hash:
                raise ValueError("member: common-noise hash differs across checkpoints")
            if member["finite"] is not True:
                raise ValueError("member: non-finite output")


def _example_manifest() -> dict[str, Any]:
    """Create deterministic synthetic metadata for oracle self-testing only."""
    checkpoint_hashes = {seed: f"oracle-checkpoint-{seed}" for seed in CHECKPOINT_SEEDS}
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "source_experiment": SOURCE_EXPERIMENT,
        "non_seed_configuration_hash": "oracle-common-configuration",
        "training_runs": [{"training_seed": seed, "non_seed_configuration_hash": "oracle-common-configuration", "selected_checkpoint_hash": checkpoint_hashes[seed], "training_completed_normally": True, "selected_checkpoint_finite": True, "selection_count": 1} for seed in CHECKPOINT_SEEDS],
        "cases": [{"case_index": case_index, "members": [{"member_id": member.member_id, "checkpoint_seed": member.checkpoint_seed, "latent_seed": member.latent_seed, "checkpoint_hash": checkpoint_hashes[member.checkpoint_seed], "initial_noise_hash": f"oracle-noise-case{case_index:02d}-z{member.latent_seed}", "finite": True} for member in case_plan(case_index)]} for case_index in range(CASE_COUNT)],
    }


def validate_fail_closed_examples() -> None:
    manifest = _example_manifest()
    validate_admission_manifest(manifest)
    manifest["cases"][0]["members"][4]["initial_noise_hash"] = "mismatch"
    try:
        validate_admission_manifest(manifest)
    except ValueError as error:
        if "common-noise hash differs" not in str(error):
            raise
    else:
        raise AssertionError("metadata oracle accepted a common-noise mismatch")


if __name__ == "__main__":
    validate_contract()
    validate_fail_closed_examples()
    print("PASS: 40 cases, 10 members per case, checkpoint extras 14/13/13; metadata admission fails closed")
