"""Negative and positive fixtures for compact-to-publication rendering."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from paper import test_rank_coherent_runner_prototype as prototype_fixture
from paper.rank_coherent_publication_renderer import render_publication_documents
from paper.rank_coherent_result_reconciliation import publish_reconciliation
from paper.check_publication_artifacts import validate_eligible_calibration_transition
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
        bases = {path: f"base {path.name}\n" for path in paths}
        guard = "| `eligible_calibration` | `MISSING_ELIGIBLE_RESULT` | `NONE` | `BLOCKED` |\n"
        for path in paths[:4]:
            bases[path] += guard
        bases[paths[3]] += (
            "Publication status: NOT_READY\n"
            "Required scientific blockers: an eligible spatially preserving calibration and\n"
            "the remaining minimum-tier comparisons\n"
            "| Eligible spatially preserving calibration | `RESEARCH_PLAN.md` | "
            "MISSING_ELIGIBLE_RESULT | One frozen candidate passes every mandatory "
            "no-compensation gate family |\n"
            "| Conformal intervals | `MINIMUM_TIER_COMPARISON_AUDIT.md` | MISSING | "
            "A valid trusted comparison remains required |\n"
        )
        bases[paths[-1]] += "Status: PRE_RESULT_NO_TRUSTED_MODE\n"
        return temporary, output, record, paths, bases

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

    def test_negative_render_publish_pipeline_changes_reconciliation_state(self):
        temporary, output, record, paths, bases = self.inputs()
        self.addCleanup(temporary.cleanup)
        rendered, call = render_publication_documents(
            bases, reconciliation_path=paths[-1], record_path=record,
            compact_directory=output,
        )
        publish_reconciliation(
            rendered, reconciliation_path=paths[-1], record_path=record,
            compact_directory=output, **call,
        )
        self.assertIn("Status: RECONCILED_NEGATIVE", paths[-1].read_text(encoding="utf-8"))
        self.assertNotIn("Status: PRE_RESULT_NO_TRUSTED_MODE", paths[-1].read_text(encoding="utf-8"))

    def test_positive_render_publish_updates_normative_transition(self):
        temporary, output, record, paths, bases = self.inputs()
        self.addCleanup(temporary.cleanup)
        aggregate_path = output / "aggregate_case_mean_metrics.json"
        aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
        for criteria in aggregate["gate_inputs"].values():
            for criterion in criteria:
                criteria[criterion] = True
        for family in ("proper_score", "finite_ensemble_reliability", "boundary", "spatial_physical", "operational"):
            aggregate["gate"][family] = True
        aggregate["gate"]["overall_eligible"] = True
        aggregate_path.write_text(json.dumps(aggregate), encoding="utf-8")
        status_path = output / "run_status.json"
        status = json.loads(status_path.read_text(encoding="utf-8"))
        status["overall_eligible"] = True
        status_path.write_text(json.dumps(status), encoding="utf-8")
        rendered, call = render_publication_documents(
            bases, reconciliation_path=paths[-1], record_path=record,
            compact_directory=output,
        )
        publish_reconciliation(
            rendered, reconciliation_path=paths[-1], record_path=record,
            compact_directory=output, **call,
        )
        validate_eligible_calibration_transition(
            *(paths[index].read_text(encoding="utf-8") for index in (0, 1, 3, 2))
        )
        self.assertIn("Status: RECONCILED_POSITIVE", paths[-1].read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
