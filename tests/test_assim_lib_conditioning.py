import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from assim_lib.data import M2MForecastDataset
from assim_lib.transforms import make_conditioned_model_input


class AssimLibConditioningTests(unittest.TestCase):
    def test_water_mask_is_single_conditioning_channel(self):
        state = torch.zeros((1, 4, 3, 2))
        grid = torch.zeros((1, 2, 3, 2))
        background = torch.ones_like(state)
        obs_values = torch.full_like(state, 2.0)
        obs_mask = torch.full_like(state, 3.0)
        water_mask = torch.stack(
            [
                torch.ones((3, 2)),
                torch.zeros((3, 2)),
                torch.full((3, 2), 2.0),
                torch.full((3, 2), 3.0),
            ],
            dim=0,
        ).unsqueeze(0)

        model_input = make_conditioned_model_input(
            state,
            grid,
            background,
            obs_values,
            obs_mask,
            water_mask,
        )

        self.assertEqual(tuple(model_input.shape), (1, 19, 3, 2))
        torch.testing.assert_close(model_input[:, -1:], water_mask[:, :1])

    def test_m2m_masks_are_zero_in_land_invalid_and_padding_regions(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            preds = root / "dataset" / "preds"
            preds.mkdir(parents=True)

            background = np.array(
                [
                    [[1.0, 2.0], [3.0, 4.0]],
                    [[5.0, 6.0], [7.0, 8.0]],
                ],
                dtype=np.float32,
            )
            truth = np.array(
                [
                    [[10.0, 11.0], [12.0, np.nan]],
                    [[20.0, 21.0], [22.0, 23.0]],
                ],
                dtype=np.float32,
            )
            np.save(preds / "ocean+atmosphere_24_20200101.npy", background)
            np.save(preds / "ocean+atmosphere_24_20200102.npy", truth)

            land_mask = np.array([[False, True], [False, False]], dtype=np.float32)
            land_mask_path = root / "land_mask.npy"
            np.save(land_mask_path, land_mask)

            dataset = M2MForecastDataset(
                {
                    "dataset_name": "M2MForecastDataset",
                    "dataset_dir": str(root / "dataset"),
                    "mask_path": str(land_mask_path),
                    "mask_true_is_invalid": True,
                    "lead_time_hours": 24,
                    "fields": ["var0", "var1"],
                    "indices": [0, 1],
                    "means": [0.0, 0.0],
                    "stds": [1.0, 1.0],
                    "padding_values": [-1.0, -2.0],
                    "image_size": [3, 4],
                    "observed_channels": [0, 1],
                    "observation_mask": {"kind": "random", "density": 1.0},
                    "train": {
                        "back_start_day": "2020-01-01",
                        "back_end_day": "2020-01-01",
                        "obs_start_day": "2020-01-02",
                        "obs_end_day": "2020-01-02",
                    },
                },
                split="train",
            )
            sample = dataset[0]

        self.assertEqual(tuple(sample["water_mask"].shape), (2, 3, 4))
        self.assertEqual(tuple(sample["valid_mask"].shape), (2, 3, 4))
        self.assertEqual(tuple(sample["obs_mask"].shape), (2, 3, 4))

        # Land mask and padded bottom/right extent are invalid for every mask.
        self.assertEqual(float(sample["water_mask"][:, 0, 1].sum()), 0.0)
        self.assertEqual(float(sample["water_mask"][:, 2, :].sum()), 0.0)
        self.assertEqual(float(sample["water_mask"][:, :, 2:].sum()), 0.0)
        self.assertEqual(float(sample["valid_mask"][:, 0, 1].sum()), 0.0)
        self.assertEqual(float(sample["valid_mask"][:, 2, :].sum()), 0.0)
        self.assertEqual(float(sample["valid_mask"][:, :, 2:].sum()), 0.0)
        self.assertEqual(float(sample["obs_mask"][:, 0, 1].sum()), 0.0)
        self.assertEqual(float(sample["obs_mask"][:, 2, :].sum()), 0.0)
        self.assertEqual(float(sample["obs_mask"][:, :, 2:].sum()), 0.0)

        # A NaN in truth invalidates only its field channel. The generated observation
        # footprint is spatial and follows channel 0 validity for every observed channel.
        self.assertEqual(float(sample["valid_mask"][0, 1, 1]), 0.0)
        self.assertEqual(float(sample["valid_mask"][1, 1, 1]), 1.0)
        self.assertEqual(float(sample["obs_mask"][0, 1, 1]), 0.0)
        self.assertEqual(float(sample["obs_mask"][1, 1, 1]), 0.0)


if __name__ == "__main__":
    unittest.main()
