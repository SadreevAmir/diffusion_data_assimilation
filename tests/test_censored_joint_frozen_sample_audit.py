from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

import torch

from assim_lib.censored_joint_frozen_sample_audit import (
    _phase_map,
    _plot_pipeline,
    _safe_ratio,
    audit,
    decode_latent,
    decoder_invariants,
    paired_identity,
)


MEANS = (0.2, 0.4)
STDS = (0.5, 0.25)


def _payload() -> dict:
    valid = torch.ones(1, 1, 4, 4)
    latent = torch.linspace(-2.0, 2.0, 1 * 2 * 6 * 4 * 4).reshape(1, 2, 6, 4, 4)
    _, physical = decode_latent(latent, valid, means=MEANS, stds=STDS)
    persistence = torch.zeros(1, 6, 4, 4)
    persistence[:, 0::2] = 0.5
    persistence[:, 1::2] = 0.75
    return {
        "uncensored_latent_normalized": latent,
        "physical_ensemble": physical,
        "truth": persistence.clone(),
        "persistence": persistence,
        "valid_mask": valid,
        "metadata": [{"case": 0}],
        "noise_sha256": "same",
    }


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FrozenSampleAuditTests(unittest.TestCase):
    def test_decoder_invariants_reconstruct_and_are_nonexpansive(self) -> None:
        result = decoder_invariants(_payload(), means=MEANS, stds=STDS)
        self.assertIs(result["passed"], True)
        self.assertEqual(result["independent_decoder_max_abs_error"], 0.0)
        self.assertEqual(result["one_lipschitz_violation_count"], 0)


    def test_decoder_matches_independent_manual_six_channel_reference(self) -> None:
        valid = torch.ones(1, 1, 1, 2)
        latent = torch.tensor(
            [[[[[-1.0, 3.0]], [[-2.0, 2.0]], [[0.0, 1.0]], [[-4.0, 4.0]], [[2.0, -2.0]], [[1.0, -3.0]]]]]
        )
        uncensored, censored = decode_latent(latent, valid, means=MEANS, stds=STDS)
        manual = latent.clone()
        for channel in range(6):
            field = channel % 2
            manual[:, :, channel] = latent[:, :, channel] * STDS[field] + MEANS[field]
            if field == 0:
                manual[:, :, channel].clamp_(0.0, 1.0)
            else:
                manual[:, :, channel].clamp_(min=0.0)
        self.assertTrue(torch.equal(censored, manual))
        self.assertAlmostEqual(float(uncensored[0, 0, 0, 0, 0]), -0.3, places=6)


    def test_decoder_rejects_active_infinity(self) -> None:
        payload = _payload()
        payload["uncensored_latent_normalized"][0, 0, 0, 0, 0] = torch.inf
        with self.assertRaisesRegex(FloatingPointError, "NaN/Inf"):
            decoder_invariants(payload, means=MEANS, stds=STDS)


    def test_paired_identity_detects_truth_change(self) -> None:
        baseline = _payload()
        candidate = _payload()
        self.assertIs(paired_identity(baseline, candidate)["passed"], True)
        candidate["truth"] = candidate["truth"].clone()
        candidate["truth"][0, 0, 0, 0] = 1.0
        result = paired_identity(baseline, candidate)
        self.assertIs(result["passed"], False)
        self.assertIs(result["tensor_exact"]["truth"], False)


    def test_paired_identity_rejects_missing_provenance(self) -> None:
        baseline = _payload()
        candidate = _payload()
        del candidate["noise_sha256"]
        result = paired_identity(baseline, candidate)
        self.assertIs(result["passed"], False)
        self.assertEqual(result["missing_required_keys"], ["noise_sha256"])


    def test_zero_denominator_is_explicitly_null(self) -> None:
        self.assertEqual(
            _safe_ratio(0.0, 0.0),
            {"value": None, "null_reason": "zero_denominator"},
        )


    def test_phase_map_detects_known_period_two_pattern(self) -> None:
        row = torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0])
        field = row.view(1, 1, 1, 8).expand(1, 1, 4, 8).clone()
        valid = torch.ones_like(field, dtype=torch.bool)
        result = _phase_map(field, valid, 2)
        self.assertIsNone(result["relative_phase_contrast"]["null_reason"])
        self.assertGreater(result["relative_phase_contrast"]["value"], 0.5)
    def test_end_to_end_audit_fails_closed_on_metadata_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            base = _payload()
            candidate = _payload()
            candidate["metadata"] = [{"case": "different"}]
            for payload, checkpoint in ((base, "a" * 64), (candidate, "b" * 64)):
                payload["noise_sha256"] = "c" * 64
                payload["group"] = "validation"
                payload["update"] = 7
                payload["checkpoint"] = {"sha256": checkpoint}
            historical = {
                "ensemble": base["physical_ensemble"],
                "truth": base["truth"],
                "persistence": base["persistence"],
                "valid_mask": base["valid_mask"],
                "case_identities": [{"case": 0}],
            }
            baseline_path = tmp_path / "baseline.pt"
            candidate_path = tmp_path / "candidate.pt"
            historical_path = tmp_path / "historical.pt"
            torch.save(base, baseline_path)
            torch.save(candidate, candidate_path)
            torch.save(historical, historical_path)
            manifest = {
                "baseline_sha256": _digest(baseline_path),
                "candidate_sha256": _digest(candidate_path),
                "historical_sha256": _digest(historical_path),
                "baseline_checkpoint_sha256": "a" * 64,
                "candidate_checkpoint_sha256": "b" * 64,
                "noise_sha256": "c" * 64,
                "group": "validation",
                "update": 7,
                "means": MEANS,
                "stds": STDS,
                "baseline_cases": 1,
                "historical_cases": 1,
                "members": 2,
                "image_size": (4, 4),
            }
            result = audit(
                baseline_path,
                candidate_path,
                historical_path,
                tmp_path / "output",
                expected_manifest=manifest,
                create_plots=False,
            )
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["stage"], "paired_identity_or_decoder")
            self.assertTrue((tmp_path / "output" / "failure.json").is_file())
            self.assertFalse((tmp_path / "output" / "audit.json").exists())


    def test_pipeline_figure_is_saved_with_frozen_orientation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            payload = _payload()
            payload = {
                key: value.expand(6, *value.shape[1:]).clone()
                if torch.is_tensor(value)
                else value
                for key, value in payload.items()
            }
            _plot_pipeline(
                payload,
                tmp_path,
                label="fixture",
                means=MEANS,
                stds=STDS,
                physical_limits={"sic": (-1.0, 2.0), "sit": (-1.0, 2.0)},
                residual_limits={"sic": (-2.0, 2.0), "sit": (-2.0, 2.0)},
            )
            self.assertGreater(
                (tmp_path / "fixture_case05_member0_sic_pipeline.png").stat().st_size,
                0,
            )
            self.assertGreater(
                (tmp_path / "fixture_case05_member0_sit_pipeline.png").stat().st_size,
                0,
            )


if __name__ == "__main__":
    unittest.main()
