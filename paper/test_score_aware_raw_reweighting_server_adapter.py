#!/usr/bin/env python3
from __future__ import annotations

import unittest

try:
    import numpy as np
except ModuleNotFoundError:  # Minimal publication-check environment.
    np = None

from paper.score_aware_raw_reweighting_runner import ARTIFACT_POLICY, SOURCE_EXPERIMENT
from paper.score_aware_raw_reweighting_server_adapter import (
    DATASET_SPLIT,
    END_DATE,
    RESOURCE_KIND,
    START_DATE,
    FrozenRequest,
    ReviewedModeInventory,
    construct_case,
    validate_request,
)


MODE = "validation_literal_reviewed_score_aware_fixture"


def admission(mode=MODE):
    return {
        "admission": "GO",
        "reviewed_mode": mode,
        "publication_commit": "a" * 40,
        "runner_sha256": "b" * 64,
        "contract_sha256": "c" * 64,
        "reference_sha256": "d" * 64,
        "admission_record_sha256": "e" * 64,
        "compact_directory_sha256": "f" * 64,
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
    def inventory(self):
        inventory = ReviewedModeInventory()
        inventory.register(admission())
        return inventory

    def test_exact_request_is_accepted(self):
        validate_request(request(), admission())

    def test_interface_and_envelope_drift_fail_closed(self):
        changes = (
            {"mode": MODE + "_other"},
            {"parameters": {"source_experiment": "other"}},
            {"cases": 39},
            {"ensemble_size": 9},
            {"resource_kind": "server_gpu"},
            {"artifact_policy": "selected_artifacts"},
        )
        for change in changes:
            with self.assertRaises(ValueError, msg=change):
                validate_request(request(**change), admission())

    def test_admission_schema_and_inventory_fail_closed(self):
        bad = admission(); bad["extra"] = "field"
        with self.assertRaisesRegex(ValueError, "schema drift"):
            validate_request(request(), bad)
        inventory = self.inventory()
        changed = admission(); changed["runner_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            inventory.register(changed)
        with self.assertRaises(KeyError):
            inventory.resolve(MODE + "_other")

    def test_dispatch_copies_only_selected_raw_members(self):
        if np is None:
            self.skipTest("numpy is not installed in the minimal local environment")
        rows = np.arange(1, 41, dtype=float)[:, None]
        columns = np.arange(1, 14, dtype=float)[None, :]
        training = np.sin(rows / columns) + rows * columns / 1000.0
        targets = 0.2 + training @ np.linspace(-0.1, 0.1, 13)
        heldout = training[:10] + np.linspace(0.0, 0.2, 10)[:, None]
        raw = np.arange(10 * 5 * 6, dtype=float).reshape(10, 5, 6)
        raw /= raw.max()
        candidate, diagnostics = construct_case(
            self.inventory(), request(), training, targets, heldout, raw
        )
        for position, source in enumerate(diagnostics["source_raw_member_indices"]):
            self.assertTrue(np.array_equal(candidate[position], raw[source]))
        self.assertTrue(diagnostics["exact_source_copy"])


if __name__ == "__main__":
    unittest.main()
