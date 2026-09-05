import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from assim_lib.clean_baseline_diagnostics import (
    WORST_CASE_METRICS,
    analyze,
    fair_crps_sum,
    make_all_case_contact_sheet,
    make_comparison_panels,
    reconcile_case_inputs,
    select_worst_cases,
    tie_aware_rank_counts,
)


class CleanBaselineDiagnosticsTest(unittest.TestCase):
    @staticmethod
    def _sample_payload(value: float, *, members: int = 10) -> dict:
        truth = np.full((1, 2, 2), value, dtype=np.float32)
        ensemble = np.repeat(truth[None, ...], members, axis=0)
        return {
            "analysis_ensemble": ensemble,
            "truth": truth,
            "background": truth.copy(),
            "obs_values": np.zeros((1, 2, 2), dtype=np.float32),
            "obs_mask": np.zeros((1, 2, 2), dtype=bool),
            "valid_mask": np.ones((1, 2, 2), dtype=bool),
            "fields": np.asarray(["siconc"]),
        }

    def test_ties_are_split_over_all_admissible_ranks(self):
        ensemble = np.asarray([[[0.0, 0.0]], [[0.0, 1.0]]], dtype=np.float64)
        truth = np.asarray([[0.0, 1.0]], dtype=np.float64)
        valid = np.ones_like(truth, dtype=bool)
        counts = tie_aware_rank_counts(ensemble, truth, valid)
        np.testing.assert_allclose(counts, [1.0 / 3.0, 1.0 / 3.0 + 0.5, 1.0 / 3.0 + 0.5])
        self.assertAlmostEqual(float(counts.sum()), 2.0)

    def test_tie_aware_ranks_are_deterministic(self):
        ensemble = np.zeros((10, 2, 2), dtype=np.float64)
        truth = np.zeros((2, 2), dtype=np.float64)
        valid = np.ones((2, 2), dtype=bool)
        first = tie_aware_rank_counts(ensemble, truth, valid)
        second = tie_aware_rank_counts(ensemble, truth, valid)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_allclose(first, np.full(11, 4.0 / 11.0))

    def test_fair_crps_uses_off_diagonal_pair_correction(self):
        ensemble = np.asarray([[[0.0]], [[1.0]]], dtype=np.float64)
        truth = np.asarray([[0.5]], dtype=np.float64)
        valid = np.ones_like(truth, dtype=bool)
        self.assertAlmostEqual(fair_crps_sum(ensemble, truth, valid), 0.0)

    def test_non_finite_valid_member_fails_closed(self):
        ensemble = np.asarray([[[0.0]], [[np.nan]]])
        truth = np.asarray([[0.0]])
        valid = np.ones_like(truth, dtype=bool)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            tie_aware_rank_counts(ensemble, truth, valid)

    def test_fixed_comparison_rejects_different_observation_conditions(self):
        with TemporaryDirectory() as root:
            candidate = Path(root) / "candidate"
            reference = Path(root) / "reference"
            output = Path(root) / "panels"
            candidate.mkdir()
            reference.mkdir()
            base = {
                "analysis_ensemble": np.zeros((2, 1, 2, 2), dtype=np.float32),
                "truth": np.zeros((1, 2, 2), dtype=np.float32),
                "background": np.zeros((1, 2, 2), dtype=np.float32),
                "obs_values": np.zeros((1, 2, 2), dtype=np.float32),
                "obs_mask": np.zeros((1, 2, 2), dtype=bool),
                "valid_mask": np.ones((1, 2, 2), dtype=bool),
                "fields": np.asarray(["siconc"]),
            }
            np.savez_compressed(candidate / "0000_2022-01-02_h23.npz", **base)
            changed = dict(base)
            changed["obs_mask"] = base["obs_mask"].copy()
            changed["obs_mask"][0, 0, 0] = True
            np.savez_compressed(reference / "0000_2022-01-02_h23.npz", **changed)

            with self.assertRaisesRegex(ValueError, "observation masks differ"):
                make_comparison_panels(
                    candidate,
                    reference,
                    output,
                    case_orders=(0,),
                    member_ids=(0, 1),
                )

    def test_all_case_reconciliation_rejects_hidden_condition_mismatch(self):
        with TemporaryDirectory() as root:
            candidate = Path(root) / "candidate" / "samples"
            reference = Path(root) / "reference" / "samples"
            candidate.mkdir(parents=True)
            reference.mkdir(parents=True)
            cases = [{"case_order": 0, "dataset_index": 0, "target_date": "2022-01-02", "hour": 23}]
            (candidate.parent / "metadata.json").write_text(json.dumps({"cases": cases}))
            (reference.parent / "metadata.json").write_text(json.dumps({"cases": cases}))
            base = {
                "analysis_ensemble": np.zeros((2, 1, 2, 2), dtype=np.float32),
                "truth": np.zeros((1, 2, 2), dtype=np.float32),
                "background": np.zeros((1, 2, 2), dtype=np.float32),
                "obs_values": np.zeros((1, 2, 2), dtype=np.float32),
                "obs_mask": np.zeros((1, 2, 2), dtype=bool),
                "valid_mask": np.ones((1, 2, 2), dtype=bool),
                "fields": np.asarray(["siconc"]),
            }
            filename = "0000_2022-01-02_h23.npz"
            np.savez_compressed(candidate / filename, **base)
            changed = dict(base)
            changed["truth"] = base["truth"].copy()
            changed["truth"][0, 1, 1] = 0.5
            np.savez_compressed(reference / filename, **changed)

            with self.assertRaisesRegex(ValueError, "truth differs"):
                reconcile_case_inputs(candidate, reference)

    def test_worst_case_selection_is_predeclared_and_deterministic(self):
        cases = []
        for index, target_date in enumerate(("2022-02-16", "2022-01-02", "2022-04-02")):
            case = {"target_date": target_date, "path": f"{index:04d}_{target_date}_h23.npz"}
            case.update({metric: 1.0 for metric in WORST_CASE_METRICS})
            cases.append(case)
        first = select_worst_cases(cases)
        second = select_worst_cases(list(reversed(cases)))
        self.assertEqual(first, second)
        self.assertEqual(first["metrics"], list(WORST_CASE_METRICS))
        self.assertEqual(first["selected_dates"], ["2022-01-02"])
        for metric in WORST_CASE_METRICS:
            self.assertEqual(first["selections"][metric][0]["target_date"], "2022-01-02")

    def test_all_case_contact_sheet_covers_eight_dates_and_fixed_members(self):
        with TemporaryDirectory() as root:
            candidate = Path(root) / "candidate"
            reference = Path(root) / "reference"
            output = Path(root) / "all_cases.png"
            candidate.mkdir()
            reference.mkdir()
            dates = (
                "2022-01-02",
                "2022-02-16",
                "2022-04-02",
                "2022-05-17",
                "2022-07-01",
                "2022-08-15",
                "2022-09-29",
                "2022-11-13",
            )
            for index, target_date in enumerate(dates):
                filename = f"{index:04d}_{target_date}_h23.npz"
                payload = self._sample_payload(index / 10.0)
                np.savez_compressed(candidate / filename, **payload)
                np.savez_compressed(reference / filename, **payload)

            evidence = make_all_case_contact_sheet(candidate, reference, output)
            self.assertTrue(output.is_file())
            self.assertEqual(evidence["dates"], list(dates))
            self.assertEqual(evidence["rows"], 16)
            self.assertEqual(evidence["member_ids"], [0, 3, 6, 9])

    def test_all_case_contact_sheet_fails_if_a_case_can_be_omitted(self):
        with TemporaryDirectory() as root:
            candidate = Path(root) / "candidate"
            reference = Path(root) / "reference"
            candidate.mkdir()
            reference.mkdir()
            for index in range(7):
                target_date = f"2022-01-{index + 1:02d}"
                filename = f"{index:04d}_{target_date}_h23.npz"
                payload = self._sample_payload(0.0)
                np.savez_compressed(candidate / filename, **payload)
                np.savez_compressed(reference / filename, **payload)
            with self.assertRaisesRegex(ValueError, "exactly 8 cases"):
                make_all_case_contact_sheet(candidate, reference, Path(root) / "bad.png")

    def test_all_case_contact_sheet_rejects_member_cherry_picking(self):
        with TemporaryDirectory() as root:
            candidate = Path(root) / "candidate"
            reference = Path(root) / "reference"
            candidate.mkdir()
            reference.mkdir()
            for index in range(8):
                target_date = f"2022-01-{index + 1:02d}"
                filename = f"{index:04d}_{target_date}_h23.npz"
                payload = self._sample_payload(0.0)
                np.savez_compressed(candidate / filename, **payload)
                np.savez_compressed(reference / filename, **payload)
            with self.assertRaisesRegex(ValueError, "fixed member ids"):
                make_all_case_contact_sheet(
                    candidate,
                    reference,
                    Path(root) / "bad-members.png",
                    member_ids=(1, 2, 4, 8),
                )

    def test_analyze_writes_complete_non_selective_visual_evidence(self):
        with TemporaryDirectory() as root:
            candidate = Path(root) / "candidate" / "samples"
            reference = Path(root) / "reference" / "samples"
            output = Path(root) / "diagnostics"
            candidate.mkdir(parents=True)
            reference.mkdir(parents=True)
            dates = (
                "2022-01-02",
                "2022-02-16",
                "2022-04-02",
                "2022-05-17",
                "2022-07-01",
                "2022-08-15",
                "2022-09-29",
                "2022-11-13",
            )
            case_metadata = []
            for index, target_date in enumerate(dates):
                filename = f"{index:04d}_{target_date}_h23.npz"
                payload = self._sample_payload(index / 10.0)
                np.savez_compressed(candidate / filename, **payload)
                np.savez_compressed(reference / filename, **payload)
                case_metadata.append(
                    {
                        "case_order": index,
                        "dataset_index": index * 45,
                        "target_date": target_date,
                        "hour": 23,
                    }
                )
            metadata = json.dumps({"cases": case_metadata})
            (candidate.parent / "metadata.json").write_text(metadata)
            (reference.parent / "metadata.json").write_text(metadata)

            summary = analyze(candidate, reference, output)
            visual_gate = summary["visual_gate"]
            self.assertTrue(Path(visual_gate["all_case_contact_sheet"]["path"]).is_file())
            self.assertEqual(visual_gate["all_case_contact_sheet"]["dates"], list(dates))
            self.assertEqual(visual_gate["worst_case_protocol"]["metrics"], list(WORST_CASE_METRICS))
            self.assertEqual(visual_gate["worst_case_protocol"]["selected_dates"], [dates[0]])
            self.assertEqual(len(visual_gate["worst_case_panels"]), 1)
            self.assertEqual(len(visual_gate["panels"]), 4)
            self.assertTrue((output / "diagnostics.json").is_file())


if __name__ == "__main__":
    unittest.main()
