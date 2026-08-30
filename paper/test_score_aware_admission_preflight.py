import json
import tempfile
import unittest
from pathlib import Path

from paper.score_aware_admission_preflight import preflight


class ScoreAwareAdmissionPreflightTest(unittest.TestCase):
    def setUp(self):
        self.repository = Path(__file__).resolve().parents[1]
        self.handoff = self.repository / "paper" / "SCORE_AWARE_RAW_REWEIGHTING_LOCAL_HANDOFF.json"

    def test_current_boundary_has_one_exact_reason(self):
        result = preflight(self.repository, self.handoff, frozenset({"validation_strict_full"}))
        self.assertEqual(result["preflight"], "PASS_LOCAL_BOUNDARY")
        self.assertEqual(result["reasons"], ["MISSING_LITERAL_REVIEWED_MODE"])
        self.assertFalse(result["proposal_authorized"])

    def test_artifact_drift_fails_local_boundary(self):
        handoff = json.loads(self.handoff.read_text())
        handoff["artifact_sha256"]["NEXT_SCORE_AWARE_RAW_REWEIGHTING_CONTRACT.md"] = "0" * 64
        with tempfile.TemporaryDirectory() as directory:
            mutated = Path(directory) / "handoff.json"
            mutated.write_text(json.dumps(handoff))
            result = preflight(self.repository, mutated, frozenset())
        self.assertEqual(result["preflight"], "FAIL_LOCAL_BOUNDARY")
        self.assertTrue(any(reason.startswith("WORKTREE_ARTIFACT_DRIFT:") for reason in result["reasons"]))


if __name__ == "__main__":
    unittest.main()
