from __future__ import annotations

import copy
import unittest

from paper.probabilistic_da_contract_oracle import (
    CASE_COLUMNS, EXPECTED_ARTIFACTS, INFLATIONS, RADII_KM, evaluate_decision,
    purged_folds, validate_case_rows, validate_manifest,
)


H = "a" * 64


def fixture():
    folds = []
    for index, (held_out, training) in enumerate(purged_folds()):
        candidates = [
            {"radius_km": radius, "inflation": inflation, "feasible": True, "training_fair_crps": 1.0 + radius / 1000 + inflation / 100}
            for radius in RADII_KM for inflation in INFLATIONS
        ]
        folds.append({"fold": index, "held_out": list(held_out), "training": list(training), "candidates": candidates, "selected": {"radius_km": 50, "inflation": 1.0}})
    return {
        "contract_version": "probabilistic_da_letkf_v1", "source_experiment": "joint_full_condition_validation_2022",
        "artifact_policy": "summary_only", "case_count": 40, "ensemble_size": 10, "artifacts": list(EXPECTED_ARTIFACTS),
        "observation_operator": {"name": "sealed_common_operator", "output_hashes": [{"case_index": i, "letkf": H, "learned_joint": H} for i in range(40)]},
        "localization": {"taper": "Gaspari-Cohn", "radii_km": list(RADII_KM)},
        "assimilation": {"filter": "LETKF", "variable": "physical_SIC", "square_root": "deterministic", "projection": "after_complete_analysis_update", "observation_error_source": "sealed_common_contract", "inflations": list(INFLATIONS)},
        "background": {"configuration_digest": H, "member_hashes": [[H] * 10 for _ in range(40)]},
        "folds": folds,
        "invariant_checks": {"common_information": True, "finite": True, "fold_leakage_absent": True, "background_independent": True, "solver_success": True},
    }


class ProbabilisticDAContractOracleTests(unittest.TestCase):
    def case_rows(self):
        rows = []
        for case_index in range(40):
            for method in ("LETKF", "raw_learned_joint", "3D-Var"):
                row = {key: 0.1 for key in CASE_COLUMNS[4:]}
                row.update({"case_index": case_index, "target_date": f"sealed-case-{case_index:02d}", "fold": case_index // 8, "method": method})
                rows.append(row)
        return rows

    def test_literal_folds_and_valid_fixture(self):
        self.assertEqual([len(train) for _, train in purged_folds()], [29, 26, 26, 26, 29])
        validate_manifest(fixture())

    def test_observation_identity_fails_closed(self):
        value = fixture(); value["observation_operator"]["output_hashes"][3]["letkf"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "hash-identical"): validate_manifest(value)

    def test_localization_and_assimilation_drift_fail_closed(self):
        value = fixture(); value["localization"]["radii_km"].append(800)
        with self.assertRaisesRegex(ValueError, "localization"): validate_manifest(value)
        value = fixture(); value["assimilation"]["projection"] = "memberwise_before_update"
        with self.assertRaisesRegex(ValueError, "assimilation"): validate_manifest(value)

    def test_purge_and_selection_tie_break_fail_closed(self):
        value = fixture(); value["folds"][2]["training"].append(15)
        with self.assertRaisesRegex(ValueError, "purge"): validate_manifest(value)
        value = fixture()
        for candidate in value["folds"][0]["candidates"]: candidate["training_fair_crps"] = 1.0
        value["folds"][0]["selected"] = {"radius_km": 400, "inflation": 1.2}
        with self.assertRaisesRegex(ValueError, "tie-break"): validate_manifest(value)

    def test_compact_boundary_and_invariant_fail_closed(self):
        value = fixture(); value["artifacts"].append("raw_analyses.nc")
        with self.assertRaisesRegex(ValueError, "artifact"): validate_manifest(value)
        value = fixture(); value["invariant_checks"]["fold_leakage_absent"] = False
        with self.assertRaisesRegex(ValueError, "invariant"): validate_manifest(value)

    def test_case_metric_schema_and_identity_fail_closed(self):
        rows = self.case_rows(); validate_case_rows(rows)
        rows[-1] = dict(rows[0])
        with self.assertRaisesRegex(ValueError, "unique"): validate_case_rows(rows)
        rows = self.case_rows(); rows[0]["unreviewed_metric"] = 1.0
        with self.assertRaisesRegex(ValueError, "keys must be exact"): validate_case_rows(rows)

    def test_useful_and_negative_decisions(self):
        summary = {"case_count": 40, "letkf_fair_crps": 1.1, "raw_learned_joint_fair_crps": 1.0, "rank_uniformity_pass": True, "letkf_mean_rmse": 1.02, "var3d_mean_rmse": 1.0}
        self.assertEqual(evaluate_decision(summary), "PROBABILISTIC_DA_USEFUL")
        negative = copy.deepcopy(summary); negative["rank_uniformity_pass"] = False
        self.assertEqual(evaluate_decision(negative), "PROBABILISTIC_DA_NEGATIVE")


if __name__ == "__main__": unittest.main()
