from __future__ import annotations

import math
import subprocess
import sys
import unittest
from pathlib import Path

import torch

from assim_lib import latent_temperature_ensemble as latent
from assim_lib.config import TrainingConfig
from assim_lib.evaluate import generate_ensemble


class _IdentitySampler:
    def sample_conditioned(self, **kwargs):
        return kwargs["initial_noise"]


class LatentTemperatureEnsembleTests(unittest.TestCase):
    def test_contract_is_exact_and_predeclared(self) -> None:
        self.assertEqual(latent.MODE, "validation_latent_temperature_sampling")
        self.assertEqual(latent.SOURCE_EXPERIMENT, "joint_full_condition_validation_2022")
        self.assertEqual(latent.CASE_COUNT, 40)
        self.assertEqual(latent.MEMBER_COUNT, 10)
        self.assertEqual(latent.BASE_NOISE_SEED, 1234)
        self.assertEqual(latent.INITIAL_NOISE_SCALE, 1.30)
        self.assertEqual(latent.CHECKPOINT_NAME, "ema_last_model.pth")
        self.assertEqual(len(latent.CHECKPOINT_SHA256), 64)

    def test_noise_schedule_matches_baseline_generator(self) -> None:
        seeds = [
            latent.noise_seed(case_index, member_index)
            for case_index in range(latent.CASE_COUNT)
            for member_index in range(latent.MEMBER_COUNT)
        ]
        self.assertEqual(seeds, list(range(1234, 1234 + 400)))
        self.assertEqual(len(seeds), len(set(seeds)))

    def test_scale_is_finite_positive_and_not_identity(self) -> None:
        self.assertTrue(math.isfinite(latent.INITIAL_NOISE_SCALE))
        self.assertGreater(latent.INITIAL_NOISE_SCALE, 0.0)
        self.assertNotEqual(latent.INITIAL_NOISE_SCALE, 1.0)

    def test_generator_hashes_base_then_scales_exact_tensor(self) -> None:
        config = TrainingConfig(image_size=(2, 2), sample_start_mode="noise")
        tensor = torch.zeros((1, 1, 2, 2), dtype=torch.float32)

        def sample(scale: float):
            base_hashes: list[str] = []
            scaled_hashes: list[str] = []
            output = generate_ensemble(
                sampler=_IdentitySampler(),
                background=tensor,
                obs_values=tensor,
                obs_mask=tensor,
                water_mask=tensor,
                valid_mask=tensor,
                config=config,
                ensemble_size=2,
                sample_batch_size=2,
                num_timesteps=25,
                method="dopri5",
                device=torch.device("cpu"),
                seed=1234,
                case_order=0,
                sample_target="state",
                initial_noise_hashes=base_hashes,
                initial_noise_scale=scale,
                scaled_initial_noise_hashes=scaled_hashes,
            )
            return output, base_hashes, scaled_hashes

        baseline, baseline_base, baseline_scaled = sample(1.0)
        candidate, candidate_base, candidate_scaled = sample(1.30)
        self.assertTrue(torch.equal(candidate, baseline * 1.30))
        self.assertEqual(candidate_base, baseline_base)
        self.assertEqual(baseline_base, baseline_scaled)
        self.assertNotEqual(candidate_base, candidate_scaled)

    def test_standalone_help(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "assim_lib.latent_temperature_ensemble", "--help"],
            cwd=Path(latent.__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--source-experiment", result.stdout)


if __name__ == "__main__":
    unittest.main()
