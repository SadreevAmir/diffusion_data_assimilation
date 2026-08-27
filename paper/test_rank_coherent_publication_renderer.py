"""Negative and positive fixtures for compact-to-publication rendering."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from paper import test_rank_coherent_runner_prototype as prototype_fixture
from paper.rank_coherent_publication_renderer import render_publication_documents
from paper.test_validate_rank_coherent_admission import valid_record


class PublicationRendererTests(unittest.TestCase):
    def inputs(self):
        temporary, output, process = prototype_fixture.PrototypeTests.invoke(self, prototype_fixture.synthetic_source())
        self.assertEqual(process.returncode, 0, process.stderr)
        metadata_path = output / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata.update(decision_bearing=True, experiment_id="rank_coherent_valid", candidate="rank_coherent_candidate")
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        record = Path(temporary.name) / "admission.json"
        record.write_text(json.dumps(valid_record()), encoding="utf-8")
        names = ("PAPER_DRAFT.md", "CLAIM_LEDGER.md", "REPRODUCIBILITY.md", "PUBLICATION_READINESS.md", "RANK_COHERENT_RESULT_RECONCILIATION.md")
        paths = [Path(temporary.name) / name for name in names]
        return temporary, output, record, paths, {path: f"base {path.name}\n" for path in paths}

    def test_renders_identity_effect_sizes_decisions_and_negative_interpretation(self):
        temporary, output, record, paths, bases = self.inputs()
        self.addCleanup(temporary.cleanup)
        rendered, call = render_publication_documents(bases, reconciliation_path=paths[-1], record_path=record, compact_directory=output)
        self.assertEqual(call["experiment_id"], "rank_coherent_valid")
        for text in rendered.values():
            self.assertIn("analysis_fair_crps", text)
            self.assertIn("proper_score=", text)
            self.assertIn("Отрицательная ветвь обязательна", text)
            self.assertIn("blocker не закрывается", text)

    def test_missing_effect_size_fails_closed(self):
        temporary, output, record, paths, bases = self.inputs()
        self.addCleanup(temporary.cleanup)
        aggregate_path = output / "aggregate_case_mean_metrics.json"
        aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
        aggregate["aggregate_metrics"].pop("analysis_fair_crps")
        aggregate_path.write_text(json.dumps(aggregate), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "proper-score anchor"):
            render_publication_documents(bases, reconciliation_path=paths[-1], record_path=record, compact_directory=output)

    def test_missing_failed_family_interpretation_is_not_caller_controllable(self):
        temporary, output, record, paths, bases = self.inputs()
        self.addCleanup(temporary.cleanup)
        bases[paths[0]] += "Ни одного провала нет.\n"
        rendered, _ = render_publication_documents(bases, reconciliation_path=paths[-1], record_path=record, compact_directory=output)
        self.assertIn("Отрицательная ветвь обязательна", rendered[paths[0]])

    def test_negative_branch_cannot_close_blocker(self):
        temporary, output, record, paths, bases = self.inputs()
        self.addCleanup(temporary.cleanup)
        rendered, call = render_publication_documents(bases, reconciliation_path=paths[-1], record_path=record, compact_directory=output)
        self.assertFalse(call["overall_eligible"])
        self.assertTrue(all("blocker не закрывается" in text for text in rendered.values()))


if __name__ == "__main__":
    unittest.main()
