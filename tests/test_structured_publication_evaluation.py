from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from assim_lib.structured_trajectory_evaluation import (
    _brier_reliability,
    _fair_crps,
    _fraction_skill_score,
    _local_spatial_variogram_score,
    _symmetric_edge_displacement_l1,
    _temporal_metrics,
    build_structured_publication_visual_qc_metadata,
    evaluate_structured_deterministic_baseline,
    evaluate_structured_publication_ensemble,
    load_structured_publication_contract,
    validate_structured_publication_tensors,
)


ROOT = Path(__file__).resolve().parents[1]
CASE_MANIFEST = ROOT / "config/evaluation/structured_joint_d0_d3_test_cases.json"
PROTOCOL = ROOT / "config/evaluation/structured_joint_d0_d3_publication_protocol.json"


class StructuredPublicationContractTests(unittest.TestCase):
    def test_checked_in_pending_protocol_is_visible_but_scoring_fails_closed(self) -> None:
        contract = load_structured_publication_contract(
            CASE_MANIFEST, PROTOCOL, require_frozen=False
        )
        self.assertEqual(len(contract.cases), 197)
        self.assertEqual(contract.cases[0].anchor_date, "2023-01-01")
        self.assertEqual(contract.cases[-1].anchor_date, "2023-07-16")
        self.assertEqual(
            contract.manifest_sha256,
            "ff6c4d44d35de9c4a94d62bb1821033eac031d6d2a5c07fbf0f223eb8a7be4ef",
        )
        self.assertTrue(any("protocol status" in item for item in contract.blockers))
        with self.assertRaisesRegex(ValueError, "not frozen"):
            load_structured_publication_contract(CASE_MANIFEST, PROTOCOL)

    def test_date_pairing_is_validated_before_a_manifest_can_be_scored(self) -> None:
        manifest = json.loads(CASE_MANIFEST.read_text(encoding="utf-8"))
        protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        manifest["cases"][0]["background_archive_dates"][0] = "2022-01-02"
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "cases.json"
            protocol_path = Path(directory) / "protocol.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            protocol["case_manifest_sha256"] = hashlib.sha256(
                manifest_path.read_bytes()
            ).hexdigest()
            protocol["status"] = "frozen_for_test"
            protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "calendar-year paired"):
                load_structured_publication_contract(manifest_path, protocol_path)


class StructuredPublicationEvaluatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        protocol["case_manifest_sha256"] = hashlib.sha256(
            CASE_MANIFEST.read_bytes()
        ).hexdigest()
        protocol["status"] = "frozen_for_cpu_unit_test"
        cls.protocol_path = Path(cls.temporary.name) / "protocol.json"
        cls.protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
        cls.contract = load_structured_publication_contract(
            CASE_MANIFEST, cls.protocol_path
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def setUp(self) -> None:
        case_count = len(self.contract.cases)
        truth = torch.zeros((case_count, 8, 1, 2), dtype=torch.float32)
        truth[:, 0::2, :, 1] = 0.5
        truth[:, 1::2, :, 1] = 1.0
        self.truth = truth
        self.background = truth.clone()
        self.ensemble = truth[:, None].repeat(1, self.contract.ensemble_size, 1, 1, 1)
        self.valid = torch.ones((case_count, 1, 1, 2), dtype=torch.float32)
        self.lag0 = torch.zeros_like(self.valid)
        self.lag0[..., 0] = 1.0
        self.case_ids = [case.case_id for case in self.contract.cases]

    def test_publication_metrics_cover_all_leads_and_fractional_atom_ranks(self) -> None:
        result = evaluate_structured_publication_ensemble(
            self.ensemble,
            self.truth,
            self.background,
            self.valid,
            self.lag0,
            contract=self.contract,
            case_ids=self.case_ids,
            member_seeds=self.contract.member_seeds,
            sic_cap=0.75,
        )
        self.assertEqual(set(result["leads"]), {"d0", "d+1", "d+2", "d+3"})
        day0_sic = result["leads"]["d0"]["fields"]["sic"]
        self.assertEqual(day0_sic["evaluated_points"], 197)
        self.assertEqual(day0_sic["fair_crps"], 0.0)
        self.assertAlmostEqual(day0_sic["rank_tv_to_uniform"], 0.0, places=12)
        self.assertAlmostEqual(sum(day0_sic["fractional_rank_counts"]), 197.0)
        self.assertEqual(
            day0_sic["central_coverage"],
            {"central_50": 1.0, "central_80": 1.0, "central_90": 1.0},
        )
        self.assertEqual(result["leads"]["d+1"]["fields"]["sit"]["evaluated_points"], 394)
        self.assertEqual(result["day0_exact_track"]["max_abs_error"], 0.0)
        self.assertEqual(
            result["leads"]["d+1"]["events"]["ice_occurrence"]["brier_score"],
            0.0,
        )
        spatial = result["leads"]["d+1"]["spatial_occurrence"]
        self.assertEqual(spatial["ensemble_expected_extent_mae"], 0.0)
        self.assertEqual(spatial["expected_iiee_fraction"], 0.0)
        self.assertEqual(spatial["edge_displacement_mean"], 0.0)
        self.assertTrue(
            all(
                item["fss"] == 1.0
                for item in spatial["fraction_skill_score"]["scales"]
            )
        )
        self.assertEqual(
            result["leads"]["d+1"]["spatial_variogram"]["sic"][
                "equal_lag_mean"
            ],
            0.0,
        )
        increment = result["temporal"]["transitions"]["d+0_to_d+1"]["fields"]["sic"]
        self.assertEqual(increment["increment_fair_crps"], 0.0)
        self.assertEqual(increment["ensemble_mean_tendency_accuracy"], 1.0)
        self.assertEqual(increment["member_tendency_agreement_probability"], 1.0)
        self.assertEqual(
            result["temporal"]["trajectory_variogram"]["sic"][
                "trajectory_variogram_score"
            ],
            0.0,
        )
        json.dumps(result, allow_nan=False)

    def test_fixed_bin_reliability_uses_empirical_member_probability(self) -> None:
        member_event = torch.tensor(
            [[[[[True, False]]], [[[False, True]]]]], dtype=torch.bool
        )
        truth_event = torch.tensor([[[[False, True]]]], dtype=torch.bool)
        metrics = _brier_reliability(
            member_event, truth_event, torch.ones_like(truth_event)
        )
        self.assertEqual(metrics["brier_score"], 0.25)
        self.assertEqual(metrics["reliability_l1"], 0.0)
        self.assertEqual(metrics["reliability_bins"][5]["count"], 2)
        self.assertEqual(
            metrics["reliability_bins"][5]["forecast_probability_mean"], 0.5
        )
        self.assertEqual(metrics["reliability_bins"][5]["observed_frequency"], 0.5)

    def test_symmetric_edge_displacement_has_known_one_cell_shift(self) -> None:
        truth = torch.zeros((3, 4), dtype=torch.bool)
        truth[:, 2:] = True
        forecast = torch.zeros_like(truth)
        forecast[:, 3:] = True
        distance, state = _symmetric_edge_displacement_l1(
            forecast.numpy(), truth.numpy(), torch.ones_like(truth).numpy()
        )
        self.assertEqual(state, "both_present")
        self.assertEqual(distance, 0.5)

    def test_fss_and_local_variogram_match_closed_form_two_pixel_case(self) -> None:
        member_sic = torch.tensor(
            [[[[0.0, 1.0]], [[1.0, 0.0]]]], dtype=torch.float32
        )
        truth_sic = torch.tensor([[[0.0, 1.0]]], dtype=torch.float32)
        valid = torch.ones_like(truth_sic)
        fss = _fraction_skill_score(member_sic, truth_sic, valid)
        self.assertAlmostEqual(fss["scales"][0]["fss"], 2.0 / 3.0)

        flat_members = torch.zeros_like(member_sic)
        variogram = _local_spatial_variogram_score(
            flat_members, truth_sic, valid
        )
        self.assertEqual(variogram["lags"][0]["valid_pairs"], 1)
        self.assertEqual(variogram["lags"][0]["score"], 1.0)
        self.assertIsNone(variogram["lags"][1]["score"])

    def test_tendency_consistency_uses_member_aligned_exact_signs(self) -> None:
        truth = torch.zeros((1, 8, 1, 2), dtype=torch.float32)
        truth[:, 2, 0, 0] = 1.0
        ensemble = torch.zeros((1, 2, 8, 1, 2), dtype=torch.float32)
        ensemble[:, 0, 2, 0, 0] = 1.0
        ensemble[:, 1, 2, 0, 0] = -1.0
        metrics = _temporal_metrics(
            ensemble, truth, torch.ones((1, 1, 1, 2), dtype=torch.float32)
        )
        tendency = metrics["transitions"]["d+0_to_d+1"]["fields"]["sic"]
        self.assertEqual(tendency["ensemble_mean_tendency_accuracy"], 0.5)
        self.assertEqual(tendency["member_tendency_agreement_probability"], 0.75)

    def test_sorted_fair_crps_matches_two_member_closed_form(self) -> None:
        members = torch.tensor([[[[[0.0]]], [[[2.0]]]]], dtype=torch.float32)
        truth = torch.ones((1, 1, 1, 1), dtype=torch.float32)
        mask = torch.ones_like(truth)
        # Mean absolute error is 1; the fair two-member spread correction is 1.
        self.assertEqual(_fair_crps(members, truth, mask), 0.0)

    def test_pair_and_member_identity_are_strict(self) -> None:
        reversed_ids = list(reversed(self.case_ids))
        with self.assertRaisesRegex(ValueError, "manifest order"):
            validate_structured_publication_tensors(
                self.ensemble,
                self.truth,
                self.background,
                self.valid,
                self.lag0,
                contract=self.contract,
                case_ids=reversed_ids,
                member_seeds=self.contract.member_seeds,
                sic_cap=0.75,
            )
        with self.assertRaisesRegex(ValueError, "member_seeds"):
            validate_structured_publication_tensors(
                self.ensemble,
                self.truth,
                self.background,
                self.valid,
                self.lag0,
                contract=self.contract,
                case_ids=self.case_ids,
                member_seeds=tuple(reversed(self.contract.member_seeds)),
                sic_cap=0.75,
            )

    def test_exact_day0_track_violation_fails_closed(self) -> None:
        broken = self.ensemble.clone()
        broken[0, 0, 0, 0, 0] = 0.25
        broken[0, 0, 1, 0, 0] = 1.0
        with self.assertRaisesRegex(ValueError, "exact paired day-0"):
            validate_structured_publication_tensors(
                broken,
                self.truth,
                self.background,
                self.valid,
                self.lag0,
                contract=self.contract,
                case_ids=self.case_ids,
                member_seeds=self.contract.member_seeds,
                sic_cap=0.75,
            )

    def test_deterministic_baseline_has_no_probabilistic_scores(self) -> None:
        result = evaluate_structured_deterministic_baseline(
            self.background,
            self.truth,
            self.valid,
            self.lag0,
            self.background,
            baseline_name="exact_previous_calendar_year_background",
            conditioning_policy="frozen manifest background_archive_dates only",
            contract=self.contract,
            case_ids=self.case_ids,
            sic_cap=0.75,
        )
        self.assertEqual(result["score_family"], "deterministic_only")
        self.assertEqual(result["leads"]["d+3"]["fields"]["sic"]["rmse"], 0.0)
        self.assertEqual(
            result["leads"]["d+2"]["event_classification"][
                "ice_occurrence_error_fraction"
            ],
            0.0,
        )
        self.assertEqual(
            result["temporal"]["transitions"]["d+1_to_d+2"]["fields"]["sit"][
                "tendency_accuracy"
            ],
            1.0,
        )
        serialized = json.dumps(result, allow_nan=False)
        self.assertNotIn("crps", serialized.lower())
        self.assertNotIn("coverage", serialized.lower())
        self.assertNotIn("rank", serialized.lower())

    def test_deterministic_baseline_source_construction_is_fail_closed(self) -> None:
        broken = self.background.clone()
        broken[:, 2, :, 1] = 0.25
        broken[:, 3, :, 1] = 1.0
        with self.assertRaisesRegex(ValueError, "approved source construction"):
            evaluate_structured_deterministic_baseline(
                broken,
                self.truth,
                self.valid,
                self.lag0,
                self.background,
                baseline_name="exact_previous_calendar_year_background",
                conditioning_policy="frozen manifest background_archive_dates only",
                contract=self.contract,
                case_ids=self.case_ids,
                sic_cap=0.75,
            )

        persistence = torch.cat([self.background[:, :2]] * 4, dim=1)
        result = evaluate_structured_deterministic_baseline(
            persistence,
            self.truth,
            self.valid,
            self.lag0,
            self.background,
            baseline_name="deterministic_background_persistence",
            conditioning_policy=(
                "lead-0 previous-calendar-year background repeated through d0..d3"
            ),
            contract=self.contract,
            case_ids=self.case_ids,
            sic_cap=0.75,
        )
        self.assertEqual(result["baseline_name"], "deterministic_background_persistence")
        with self.assertRaisesRegex(ValueError, "not an approved"):
            evaluate_structured_deterministic_baseline(
                self.background,
                self.truth,
                self.valid,
                self.lag0,
                self.background,
                baseline_name="test_tuned_baseline",
                conditioning_policy="anything",
                contract=self.contract,
                case_ids=self.case_ids,
                sic_cap=0.75,
            )

    def test_visual_qc_selection_is_fixed_and_auditable(self) -> None:
        metadata = build_structured_publication_visual_qc_metadata(
            self.contract, sic_cap=0.75
        )
        self.assertEqual(len(metadata["cases"]), 8)
        self.assertEqual(metadata["cases"][0]["case_id"], self.case_ids[0])
        self.assertEqual(metadata["cases"][-1]["case_id"], self.case_ids[-1])
        self.assertEqual(metadata["cases"][0]["member_indices"], [0, 16, 33, 49])
        json.dumps(metadata, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
