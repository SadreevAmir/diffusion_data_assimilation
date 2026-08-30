from __future__ import annotations

import copy, hashlib, json, tempfile, unittest
from pathlib import Path

from paper import observation_likelihood_reweighting_runner as runner
from paper.observation_likelihood_reweighting_server_adapter import FrozenRequest, validate_request
from paper.validate_observation_likelihood_compact_outputs import directory_sha256, validate_directory


def payloads():
    cases = []
    for index in range(40):
        weights = [0.1] * 10
        cases.append({"case_id": f"case-{index:02d}", "fold": index // 8, "weights": weights,
          "effective_sample_size": 10.0, "source_hash_match": True, "mask_invariants_pass": True,
          "analysis_fair_crps_delta": -0.04, "analysis_crps_delta": 0.0})
    return {
      "case_weights.json": {"schema_version":"observation-likelihood-case-weights-v1", "cases":cases},
      "aggregate_weights.json": {"schema_version":"observation-likelihood-aggregate-v1", "num_cases":40, "median_effective_sample_size":10.0, "minimum_effective_sample_size":10.0, "all_source_hashes_match":True, "all_mask_invariants_pass":True},
      "paired_uncertainty.json": {"schema_version":"observation-likelihood-paired-uncertainty-v1", "analysis_fair_crps":{"point_delta":-0.04,"date_interval":[-0.06,-0.02],"four_case_block_interval":[-0.07,-0.01]}, "analysis_crps":{"point_delta":0.0,"date_interval":[-0.01,0.01],"four_case_block_interval":[-0.02,0.02]}},
      "gate_decision.json": {"schema_version":"observation-likelihood-no-compensation-gate-v1", "families":{name:True for name in ("proper_score","reliability","boundary","spatial_physical","operational")}, "overall_eligible":True},
    }


class ObservationLikelihoodAdmissionTests(unittest.TestCase):
    def write(self, documents):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        directory = Path(temporary.name)
        for name, document in documents.items(): (directory/name).write_text(json.dumps(document, sort_keys=True))
        return directory

    def test_runner_equal_scores_and_copy_only(self):
        raw = [object() for _ in range(10)]
        candidate, diagnostics = runner.construct_case(raw, [[0.5] for _ in range(10)], [0.5], 0.2)
        self.assertEqual(diagnostics["weights"], [0.1] * 10)
        self.assertEqual(candidate, raw)
        self.assertTrue(all(candidate[i] is raw[i] for i in range(10)))

    def test_runner_rejects_zero_bandwidth_and_nonfinite_input(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            runner.training_bandwidth([0.0, 0.0])
        with self.assertRaisesRegex(ValueError, "finite"):
            runner.likelihood_weights([[float("nan")] for _ in range(10)], [0.5], 0.2)

    def test_exact_compact_schema_and_negative_fixtures(self):
        good = payloads(); directory = self.write(good); validate_directory(directory)
        for mutation, message in (("weight", "invalid weights"), ("ess", "ESS mismatch"), ("gate", "conjunction"), ("extra", "exactly four")):
            bad = copy.deepcopy(good)
            if mutation == "weight": bad["case_weights.json"]["cases"][0]["weights"][0] = -0.1
            elif mutation == "ess": bad["case_weights.json"]["cases"][0]["effective_sample_size"] = 9.0
            elif mutation == "gate": bad["gate_decision.json"]["overall_eligible"] = False
            target = self.write(bad)
            if mutation == "extra": (target/"unexpected.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, message): validate_directory(target)

    def test_adapter_rejects_no_go_and_schema_drift(self):
        request = FrozenRequest("validation_reviewed", {"source_experiment":runner.SOURCE_EXPERIMENT}, "valid", "2022-01-01", "2022-07-15", 40, 10, "server_cpu", "summary_only")
        with self.assertRaisesRegex(ValueError, "admission=GO"): validate_request(request, {"admission":"NO_GO"})
        admission = {"admission":"GO", "reviewed_mode":"validation_reviewed", "publication_commit":"a"*40,
          "runner_sha256":"b"*64, "contract_sha256":"c"*64, "rank_reference_sha256":"d"*64,
          "admission_record_sha256":"e"*64, "compact_directory_sha256":"f"*64, "extra":"drift"}
        with self.assertRaisesRegex(ValueError, "admission=GO"): validate_request(request, admission)


if __name__ == "__main__": unittest.main()
