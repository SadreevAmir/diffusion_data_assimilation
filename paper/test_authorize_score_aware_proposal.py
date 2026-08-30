#!/usr/bin/env python3
from __future__ import annotations

import unittest
from dataclasses import replace

from paper.authorize_score_aware_proposal import authorize
from paper.validate_score_aware_raw_reweighting_admission import AdmissionResult


MODE = "validation_literal_reviewed_score_aware_fixture"


def admitted() -> AdmissionResult:
    return AdmissionResult(
        admission="GO",
        reviewed_mode=MODE,
        publication_commit="a" * 40,
        runner_sha256="b" * 64,
        contract_sha256="c" * 64,
        reference_sha256="d" * 64,
        admission_record_sha256="e" * 64,
        compact_directory_sha256="f" * 64,
        decision_bearing_validation="PASS",
        deviations=(),
    )


class ScoreAwareProposalAuthorizationTests(unittest.TestCase):
    def test_go_admission_emits_only_frozen_request(self):
        result = authorize(admitted())
        self.assertTrue(result["proposal_authorized"])
        self.assertEqual(result["reviewed_mode"], MODE)
        self.assertEqual(result["request"]["mode"], MODE)
        self.assertEqual(
            result["request"]["parameters"],
            {"source_experiment": "joint_full_condition_validation_2022"},
        )
        self.assertEqual(result["request"]["resource_kind"], "server_cpu")
        self.assertEqual(result["request"]["artifact_policy"], "summary_only")

    def test_non_go_or_review_deviation_fails_closed(self):
        for changed in (
            replace(admitted(), admission="NO_GO"),
            replace(admitted(), decision_bearing_validation="FAIL"),
            replace(admitted(), deviations=("waiver",)),
        ):
            with self.assertRaises(ValueError):
                authorize(changed)

    def test_unbound_mapping_is_not_an_admission_result(self):
        with self.assertRaisesRegex(ValueError, "combined admission result"):
            authorize(admitted().__dict__)


if __name__ == "__main__":
    unittest.main()
