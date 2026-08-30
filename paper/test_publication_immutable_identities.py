#!/usr/bin/env python3
"""Negative fixtures for the publication immutable-identity boundary."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from paper.check_publication_artifacts import (
    CASEWISE_LOCAL_HANDOFF_ARTIFACTS,
    PAPER_DIR,
    RANK_COHERENT_IMMUTABLE_DIGESTS,
    validate_casewise_local_handoff,
    validate_rank_coherent_immutable_identities,
)


class ImmutableIdentityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.handoff = (PAPER_DIR / "RANK_COHERENT_CONTROLLER_HANDOFF.md").read_text(
            encoding="utf-8"
        )

    def test_current_identity_set_passes(self) -> None:
        validate_rank_coherent_immutable_identities(PAPER_DIR, self.handoff)

    def test_each_single_file_mismatch_fails_closed(self) -> None:
        for mutated_name in RANK_COHERENT_IMMUTABLE_DIGESTS:
            with self.subTest(mutated_name=mutated_name):
                with tempfile.TemporaryDirectory() as temporary:
                    fixture_dir = Path(temporary)
                    for name in RANK_COHERENT_IMMUTABLE_DIGESTS:
                        content = (PAPER_DIR / name).read_bytes()
                        if name == mutated_name:
                            content += b"\n# deliberate identity mismatch\n"
                        (fixture_dir / name).write_bytes(content)
                    with self.assertRaisesRegex(
                        ValueError,
                        rf"{mutated_name} immutable digest mismatch",
                    ):
                        validate_rank_coherent_immutable_identities(
                            fixture_dir, self.handoff
                        )

    def test_documented_identity_mismatch_fails_closed(self) -> None:
        first_digest = next(iter(RANK_COHERENT_IMMUTABLE_DIGESTS.values()))
        mismatched_handoff = self.handoff.replace(first_digest, "0" * 64, 1)
        with self.assertRaisesRegex(ValueError, "does not record the verified digest"):
            validate_rank_coherent_immutable_identities(
                PAPER_DIR, mismatched_handoff
            )

    def test_casewise_local_handoff_current_identities_pass(self) -> None:
        validate_casewise_local_handoff(PAPER_DIR)

    def test_casewise_local_handoff_single_file_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture_dir = Path(temporary)
            for name in CASEWISE_LOCAL_HANDOFF_ARTIFACTS:
                (fixture_dir / name).write_bytes((PAPER_DIR / name).read_bytes())
            (fixture_dir / "CASEWISE_SAFETY_SELECTOR_LOCAL_HANDOFF.json").write_bytes(
                (PAPER_DIR / "CASEWISE_SAFETY_SELECTOR_LOCAL_HANDOFF.json").read_bytes()
            )
            changed = CASEWISE_LOCAL_HANDOFF_ARTIFACTS[0]
            (fixture_dir / changed).write_bytes(
                (fixture_dir / changed).read_bytes() + b"\n# identity drift\n"
            )
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                validate_casewise_local_handoff(fixture_dir)


if __name__ == "__main__":
    unittest.main()
