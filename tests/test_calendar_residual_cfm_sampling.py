from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from assim_lib import calendar_residual_cfm_sampling as sampling


class CalendarResidualSamplingTest(unittest.TestCase):
    def test_frozen_schedule_is_shifted_exact_calendar_pair_schedule(self) -> None:
        self.assertEqual(len(sampling.EXPECTED_DATES), 40)
        self.assertEqual(sampling.EXPECTED_DATES[0].isoformat(), "2022-01-02")
        self.assertEqual(sampling.EXPECTED_DATES[-1].isoformat(), "2022-07-16")
        self.assertTrue(all((b - a).days == 5 for a, b in zip(sampling.EXPECTED_DATES, sampling.EXPECTED_DATES[1:])))

    def test_validator_rejects_out_of_range_sample(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "samples").mkdir()
            cases = []
            for index, target in enumerate(sampling.EXPECTED_DATES):
                cases.append({"case_order": index, "target_date": target.isoformat()})
                ensemble = np.zeros((10, 1, 320, 256), dtype=np.float32)
                if index == 0:
                    ensemble[0, 0, 0, 0] = 1.1
                np.savez_compressed(
                    root / "samples" / f"{index:04d}_{target.isoformat()}_h23.npz",
                    analysis_ensemble=ensemble,
                    truth=np.zeros((1, 320, 256), dtype=np.float32),
                    valid_mask=np.ones((1, 320, 256), dtype=np.bool_),
                )
            sampling._atomic_json(
                root / "metadata.json",
                {
                    "split": "valid",
                    "num_cases": 40,
                    "stride_days": 5,
                    "ensemble_size": 10,
                    "sample_batch_size": 10,
                    "num_timesteps": 25,
                    "sampling_method": "dopri5",
                    "sampling_rtol": 1e-5,
                    "sampling_atol": 1e-6,
                    "inference_precision": "float32",
                    "sample_target": "residual",
                    "sample_cfg_mode": "none",
                    "cases": cases,
                },
            )
            with self.assertRaisesRegex(ValueError, "outside"):
                sampling.validate_evaluation(root)


if __name__ == "__main__":
    unittest.main()
