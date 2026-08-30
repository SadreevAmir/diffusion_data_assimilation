import unittest

from paper.weighted_rank_cell_reference import weighted_rank_cell


class WeightedRankCellReferenceTest(unittest.TestCase):
    def assert_one_hot(self, counts, rank):
        self.assertEqual(counts, [1.0 if index == rank else 0.0 for index in range(11)])

    def test_equal_weight_parity_without_ties(self):
        members = [index / 9 for index in range(10)]
        truths = [-0.1] + [(members[index] + members[index + 1]) / 2 for index in range(9)] + [1.1]
        for expected, truth in enumerate(truths):
            coordinate, counts = weighted_rank_cell(members, truth, [0.1] * 10, tie_draw=0)
            self.assertAlmostEqual(coordinate, expected)
            self.assert_one_hot(counts, expected)

    def test_equal_weight_parity_for_duplicate_and_boundary_ties(self):
        members = [0.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0, 1.0]
        for truth, draw, expected in [(0.0, 0, 0), (0.0, 2, 2), (1.0, 0, 6), (1.0, 4, 10)]:
            coordinate, counts = weighted_rank_cell(members, truth, [0.1] * 10, tie_draw=draw)
            self.assertAlmostEqual(coordinate, expected)
            self.assert_one_hot(counts, expected)

    def test_unequal_weight_coordinate_splits_one_count(self):
        coordinate, counts = weighted_rank_cell(
            [0.0, 0.5, 1.0], 0.75, [0.2, 0.3, 0.5], tie_draw=0
        )
        self.assertAlmostEqual(coordinate, 1.5)
        self.assertEqual(counts, [0.0, 0.5, 0.5, 0.0])
        self.assertAlmostEqual(sum(counts), 1.0)

    def test_invalid_inputs_fail_closed(self):
        invalid = [
            ([0.0], 0.0, [0.9], 1),
            ([0.0], 0.0, [1.0], 2),
            ([float("nan")], 0.0, [1.0], 0),
            ([0.0], float("inf"), [1.0], 0),
        ]
        for members, truth, weights, draw in invalid:
            with self.subTest(members=members, truth=truth, weights=weights, draw=draw):
                with self.assertRaises(ValueError):
                    weighted_rank_cell(members, truth, weights, tie_draw=draw)


if __name__ == "__main__":
    unittest.main()
