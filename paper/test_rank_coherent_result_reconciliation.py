"""Fixtures for atomic rank-coherent result reconciliation."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from paper import test_rank_coherent_runner_prototype as prototype_fixture
from paper.rank_coherent_result_reconciliation import canonical_marker, publish_reconciliation
from paper.test_validate_rank_coherent_admission import valid_record


class RankCoherentReconciliationTests(unittest.TestCase):
    decisions = {"proper_score": True, "reliability": False, "boundary": True,
                 "spatial_physical": True, "operational": True}
    def admitted_directory(self):
        temporary, output, process = prototype_fixture.PrototypeTests.invoke(self, prototype_fixture.synthetic_source())
        self.assertEqual(process.returncode, 0, process.stderr)
        metadata_path = output / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["decision_bearing"] = True
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        record = Path(temporary.name) / "admission.json"
        record.write_text(json.dumps(valid_record()), encoding="utf-8")
        return temporary, output, record

    def marker(self, output: Path, record: Path) -> str:
        return canonical_marker(
            record, output, experiment_id="rank_coherent_valid", candidate="rank_coherent_candidate",
            family_decisions=self.decisions, overall_eligible=False,
        )

    def test_marker_is_bound_to_admitted_directory_and_gate_conjunction(self) -> None:
        temporary, output, record = self.admitted_directory()
        self.addCleanup(temporary.cleanup)
        self.assertIn("status=RECONCILED_NEGATIVE", self.marker(output, record))
        with self.assertRaisesRegex(ValueError, "contradicts mandatory families"):
            canonical_marker(
                record, output, experiment_id="rank_coherent_valid", candidate="candidate",
                family_decisions={name: True for name in ("proper_score", "reliability", "boundary", "spatial_physical", "operational")},
                overall_eligible=False,
            )

    def test_partial_surface_transition_fails_before_writes(self) -> None:
        temporary, output, record = self.admitted_directory()
        self.addCleanup(temporary.cleanup)
        marker = self.marker(output, record)
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            names = ("PAPER_DRAFT.md", "CLAIM_LEDGER.md", "REPRODUCIBILITY.md", "PUBLICATION_READINESS.md", "RANK_COHERENT_RESULT_RECONCILIATION.md")
            paths = [directory / name for name in names]
            for path in paths:
                path.write_text("old", encoding="utf-8")
            documents = {path: marker for path in paths}
            documents[paths[2]] = "partial"
            with self.assertRaisesRegex(ValueError, "canonical marker"):
                publish_reconciliation(
                    documents, reconciliation_path=paths[-1], record_path=record,
                    compact_directory=output, experiment_id="rank_coherent_valid",
                    candidate="rank_coherent_candidate", family_decisions=self.decisions,
                    overall_eligible=False,
                )
            self.assertTrue(all(path.read_text(encoding="utf-8") == "old" for path in paths))

    def test_mid_publish_failure_rolls_back_all_surfaces(self) -> None:
        temporary, output, record = self.admitted_directory()
        self.addCleanup(temporary.cleanup)
        marker = self.marker(output, record)
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            names = ("PAPER_DRAFT.md", "CLAIM_LEDGER.md", "REPRODUCIBILITY.md", "PUBLICATION_READINESS.md", "RANK_COHERENT_RESULT_RECONCILIATION.md")
            paths = [directory / name for name in names]
            for path in paths:
                path.write_text("old", encoding="utf-8")
            real_replace, calls = os.replace, 0
            def fail_after_partial_publish(*args):
                nonlocal calls
                calls += 1
                if calls == 6:
                    raise OSError("injected partial publication failure")
                real_replace(*args)
            with patch("paper.atomic_publish.os.replace", side_effect=fail_after_partial_publish):
                with self.assertRaisesRegex(OSError, "partial publication"):
                    publish_reconciliation(
                        {path: marker for path in paths}, reconciliation_path=paths[-1],
                        record_path=record, compact_directory=output,
                        experiment_id="rank_coherent_valid", candidate="rank_coherent_candidate",
                        family_decisions=self.decisions, overall_eligible=False,
                    )
            self.assertTrue(all(path.read_text(encoding="utf-8") == "old" for path in paths))


if __name__ == "__main__":
    unittest.main()
