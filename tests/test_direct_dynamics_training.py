import unittest
from datetime import date
from unittest.mock import patch

import torch

from assim_lib.data import M2MForecastDataset
from assim_lib.direct_dynamics_training import (
    DIRECT_CONDITION_CHANNELS,
    DIRECT_INPUT_CHANNELS,
    DIRECT_OUTPUT_CHANNELS,
)


class DirectDynamicsContractTests(unittest.TestCase):
    def test_channel_contract(self):
        self.assertEqual(DIRECT_OUTPUT_CHANNELS, 6)
        self.assertEqual(DIRECT_CONDITION_CHANNELS, 15)
        self.assertEqual(DIRECT_INPUT_CHANNELS, 23)

    def test_dynamics_calendar_uses_resolved_slice(self):
        dataset = object.__new__(M2MForecastDataset)
        dataset.image_size = (2, 2)
        dataset.calendar_features = ("day_of_year", "time_index")
        dataset.dynamic_forcing_indices = (6, 7, 13, 14)
        initial = torch.zeros((2, 2, 2))
        forcing = torch.zeros((4, 2, 2))
        masks = torch.ones((4, 2, 2))
        valid = torch.ones((2, 2, 2))

        with patch("assim_lib.data.calendar_feature_values", return_value=(1.0, 2.0, 3.0, 4.0)) as fn:
            condition, *_ = dataset._structured_dynamics_conditioning(
                date(2020, 1, 2), 12, initial, forcing, masks, valid
            )
        fn.assert_called_once_with(date(2020, 1, 2), 12, dataset.calendar_features)
        self.assertEqual(tuple(condition.shape), (15, 2, 2))


if __name__ == "__main__":
    unittest.main()
