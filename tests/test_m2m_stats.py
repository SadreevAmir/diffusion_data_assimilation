import tempfile
import unittest
from pathlib import Path

import numpy as np

from assim_lib.m2m_stats import compute_m2m_channel_stats


class M2MStatsTests(unittest.TestCase):
    def test_channel_stats_use_truth_channels_hour_finite_values_and_water_mask(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            preds = root / "dataset" / "preds"
            preds.mkdir(parents=True)

            first = np.zeros((3, 2, 2, 2), dtype=np.float32)
            second = np.zeros((3, 2, 2, 2), dtype=np.float32)
            first[0, 1] = np.array([[1.0, 3.0], [np.nan, 5.0]], dtype=np.float32)
            first[2, 1] = np.array([[2.0, 4.0], [6.0, np.inf]], dtype=np.float32)
            second[0, 1] = np.array([[7.0, 9.0], [11.0, 13.0]], dtype=np.float32)
            second[2, 1] = np.array([[8.0, 10.0], [12.0, 14.0]], dtype=np.float32)

            np.save(preds / "ocean+atmosphere_24_20200101.npy", first)
            np.save(preds / "ocean+atmosphere_24_20200102.npy", second)

            land_mask = np.array([[False, True], [False, False]], dtype=np.float32)
            land_mask_path = root / "land_mask.npy"
            np.save(land_mask_path, land_mask)

            payload = compute_m2m_channel_stats(
                {
                    "dataset_name": "M2MForecastDataset",
                    "dataset_dir": str(root / "dataset"),
                    "mask_path": str(land_mask_path),
                    "mask_true_is_invalid": True,
                    "lead_time_hours": 24,
                    "fields": ["siconc", "sitemp"],
                    "indices": [0, 2],
                    "image_size": [2, 2],
                    "target_hour_index": 1,
                    "assimilation_range": 1,
                    "train": {
                        "back_start_day": "2020-01-01",
                        "back_end_day": "2020-01-01",
                        "obs_start_day": "2020-01-02",
                        "obs_end_day": "2020-01-02",
                    },
                },
            )

        expected = [
            np.array([7.0, 11.0, 13.0]),
            np.array([8.0, 12.0, 14.0]),
        ]
        self.assertEqual(payload["record_set"], "truth")
        self.assertEqual(payload["records_seen"], 1)
        self.assertEqual(payload["counts"], [3, 3])
        np.testing.assert_allclose(payload["mean"], [values.mean() for values in expected])
        np.testing.assert_allclose(payload["variance"], [values.var() for values in expected])
        np.testing.assert_allclose(payload["std"], [values.std() for values in expected])

    def test_channel_variance_is_stable_for_large_truth_values(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            preds = root / "dataset" / "preds"
            preds.mkdir(parents=True)

            truth_values = np.array(
                [[1.0e12 + 1.0, 1.0e12 + 2.0], [1.0e12 + 3.0, 1.0e12 + 4.0]],
                dtype=np.float64,
            )
            background = np.zeros((1, 2, 2), dtype=np.float64)
            truth = truth_values[None, ...]
            np.save(preds / "ocean+atmosphere_24_20200101.npy", background)
            np.save(preds / "ocean+atmosphere_24_20200102.npy", truth)

            payload = compute_m2m_channel_stats(
                {
                    "dataset_name": "M2MForecastDataset",
                    "dataset_dir": str(root / "dataset"),
                    "lead_time_hours": 24,
                    "fields": ["value"],
                    "indices": [0],
                    "image_size": [2, 2],
                    "train": {
                        "back_start_day": "2020-01-01",
                        "back_end_day": "2020-01-01",
                        "obs_start_day": "2020-01-02",
                        "obs_end_day": "2020-01-02",
                    },
                },
                workers=1,
            )

        np.testing.assert_allclose(payload["mean"], [truth_values.mean(dtype=np.float64)])
        np.testing.assert_allclose(payload["variance"], [truth_values.var(dtype=np.float64)])
        np.testing.assert_allclose(payload["std"], [truth_values.std(dtype=np.float64)])


if __name__ == "__main__":
    unittest.main()
