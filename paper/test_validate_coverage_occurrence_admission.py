import json
import tempfile
import unittest
from pathlib import Path

from validate_coverage_occurrence_admission import EXPECTED, FAMILIES, SOURCE, admit


class CoverageOccurrenceAdmissionTests(unittest.TestCase):
    def make_record(self, label, **gate_overrides):
        gate = {family: True for family in FAMILIES}
        gate["overall_eligible"] = True
        gate.update(gate_overrides)
        return {
            "identity": {
                **EXPECTED[label],
                "source_experiment": SOURCE,
                "cases": 40,
                "ensemble_size": 10,
            },
            "gate": gate,
        }

    def write(self, record):
        handle = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(record, handle)
        handle.close()
        self.addCleanup(Path(handle.name).unlink)
        return Path(handle.name)

    def test_accepts_exact_positive_and_negative_conjunctions(self):
        for label in EXPECTED:
            with self.subTest(label=label, outcome="positive"):
                self.assertTrue(admit(self.write(self.make_record(label)), label)["gate"]["overall_eligible"])
            negative = self.make_record(label, proper_score=False, overall_eligible=False)
            with self.subTest(label=label, outcome="negative"):
                self.assertFalse(admit(self.write(negative), label)["gate"]["overall_eligible"])

    def test_rejects_identity_substitution(self):
        record = self.make_record("threshold")
        record["identity"]["variant_id"] = EXPECTED["joint_rank"]["variant_id"]
        with self.assertRaisesRegex(ValueError, "variant_id"):
            admit(self.write(record), "threshold")

    def test_rejects_non_boolean_family(self):
        record = self.make_record("joint_rank", reliability=1)
        with self.assertRaisesRegex(ValueError, "Boolean required"):
            admit(self.write(record), "joint_rank")

    def test_rejects_false_overall_with_all_families_true(self):
        record = self.make_record("joint_rank", overall_eligible=False)
        with self.assertRaisesRegex(ValueError, "family conjunction"):
            admit(self.write(record), "joint_rank")


if __name__ == "__main__":
    unittest.main()
