#!/usr/bin/env python3
from __future__ import annotations

import unittest

from paper.raw_member_reweighting_server_adapter import (
    DATASET_SPLIT,
    END_DATE,
    RESOURCE_KIND,
    START_DATE,
    FrozenRequest,
    ReviewedModeInventory,
    construct_case,
    validate_request,
)
from paper.raw_member_reweighting_runner import ARTIFACT_POLICY, SOURCE_EXPERIMENT


MODE = "validation_reviewed_raw_member_reweighting"


def admission(mode=MODE):
    return {
        "admission": "GO",
        "reviewed_mode": mode,
        "review_record_sha256": "a" * 64,
        "publication_commit": "b" * 40,
        "runner_sha256": "c" * 64,
        "contract_sha256": "d" * 64,
        "synthetic_result_sha256": "e" * 64,
    }


def request(**changes):
    values = {
        "mode": MODE,
        "parameters": {"source_experiment": SOURCE_EXPERIMENT},
        "dataset_split": DATASET_SPLIT,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "cases": 40,
        "ensemble_size": 10,
        "resource_kind": RESOURCE_KIND,
        "artifact_policy": ARTIFACT_POLICY,
    }
    values.update(changes)
    return FrozenRequest(**values)


class ServerAdapterTests(unittest.TestCase):
    def reviewed_inventory(self):
        inventory = ReviewedModeInventory()
        inventory.register(admission())
        return inventory

    def test_exact_reviewed_request_is_accepted(self):
        validate_request(request(), admission())

    def test_mode_or_envelope_drift_fails_closed(self):
        for changed in (
            {"mode": MODE + "_other"},
            {"cases": 39},
            {"ensemble_size": 9},
            {"artifact_policy": "selected_artifacts"},
            {"resource_kind": "server_gpu"},
        ):
            with self.assertRaises(ValueError, msg=changed):
                validate_request(request(**changed), admission())

    def test_missing_or_unbound_admission_fails_closed(self):
        with self.assertRaises(ValueError):
            validate_request(request(), {"admission": "GO", "reviewed_mode": MODE})
        with self.assertRaises(ValueError):
            validate_request(request(), admission(MODE + "_other"))

    def test_inventory_registers_only_go_and_is_immutable(self):
        inventory = ReviewedModeInventory()
        with self.assertRaises(ValueError):
            inventory.register({"admission": "NO_GO", "reviewed_mode": MODE})
        self.assertEqual(inventory.register(admission()), MODE)
        self.assertEqual(inventory.resolve(MODE)["runner_sha256"], "c" * 64)
        with self.assertRaises(KeyError):
            inventory.resolve(MODE + "_other")
        changed = admission()
        changed["runner_sha256"] = "f" * 64
        with self.assertRaises(ValueError):
            inventory.register(changed)

    def test_dispatch_preserves_complete_source_fields(self):
        ranks = [0.0] * 10
        means = list(range(10))
        raw = [[[index / 10]] for index in range(10)]
        candidate, diagnostics = construct_case(
            self.reviewed_inventory(), request(), ranks, means, raw
        )
        self.assertEqual(len(candidate), 10)
        self.assertTrue(diagnostics["exact_source_copy"])
        self.assertTrue(all(diagnostics["mask_invariants"].values()))

    def test_dispatch_rejects_unregistered_or_direct_admission(self):
        ranks = [0.0] * 10
        means = list(range(10))
        raw = [[[index / 10]] for index in range(10)]
        with self.assertRaises(KeyError):
            construct_case(ReviewedModeInventory(), request(), ranks, means, raw)
        with self.assertRaises(ValueError):
            construct_case(admission(), request(), ranks, means, raw)


if __name__ == "__main__":
    unittest.main()
