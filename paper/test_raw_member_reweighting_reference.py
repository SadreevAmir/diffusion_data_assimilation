import unittest

from paper.raw_member_reweighting_reference import construct_selection


class RawMemberReweightingReferenceTest(unittest.TestCase):
    def test_uniform_ranks_preserve_all_rank_positions(self):
        result = construct_selection([(index + 0.5) / 10 for index in range(10)])
        self.assertEqual(result["source_raw_member_indices"], list(range(10)))
        self.assertEqual(result["source_multiplicities"], [1] * 10)
        self.assertAlmostEqual(result["effective_sample_size"], 10.0)

    def test_concentrated_ranks_create_frozen_repetitions(self):
        result = construct_selection([0.01] * 10)
        self.assertEqual(result["analog_rank_bin_counts"], [10] + [0] * 9)
        self.assertEqual(result["source_raw_member_indices"], [0] * 7 + [2, 5, 8])
        self.assertAlmostEqual(sum(result["analog_rank_probabilities"]), 1.0)

    def test_invalid_rank_fails_closed(self):
        for ranks in ([0.5] * 9, [0.5] * 9 + [float("nan")], [0.5] * 9 + [1.1]):
            with self.subTest(ranks=ranks):
                with self.assertRaises(ValueError):
                    construct_selection(ranks)


if __name__ == "__main__":
    unittest.main()
