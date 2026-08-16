"""Dependency-free oracle for the frozen clean-checkpoint deep ensemble."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


CHECKPOINT_SEEDS = (1701, 1702, 1703)
LATENT_SEEDS = (2401, 2402, 2403, 2404)
CASE_COUNT = 40
MEMBER_COUNT = 10


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


if __name__ == "__main__":
    validate_contract()
    print("PASS: 40 cases, 10 members per case, checkpoint extras 14/13/13")
