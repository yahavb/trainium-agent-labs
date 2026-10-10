"""Independent-world packing, isolation and correctness gates."""

import unittest
import numpy as np
from batch_reference import fixed_batch, prepare_batch
from check_batch import check_result
from contact import build_fixture
from update_reference import fixed_steps, prepare


class BatchTests(unittest.TestCase):
    def test_packed_updates_match_each_independent_world(self):
        for scene in ("plane", "pairs", "stack"):
            fixtures = [build_fixture(scene, 33, w) for w in range(3)]
            inputs = prepare_batch(fixtures)
            actual = fixed_batch(*inputs, 32)
            for w, fixture in enumerate(fixtures):
                np.testing.assert_array_equal(actual[:, w:w + 1], fixed_steps(*prepare(fixture), 32))
            self.assertTrue(np.all(actual[33:] == 0))

    def test_changing_one_world_does_not_change_others(self):
        inputs = prepare_batch([build_fixture("plane", 8, w) for w in range(3)])
        expected = fixed_batch(*inputs, 8)
        inputs[1][:, 1] *= -3
        actual = fixed_batch(*inputs, 8)
        np.testing.assert_array_equal(actual[:, [0, 2]], expected[:, [0, 2]])
        self.assertFalse(np.array_equal(actual[:, 1], expected[:, 1]))

    def test_padding_corruption_and_input_mutation_rejected(self):
        fixtures = [build_fixture("plane", 33, w) for w in range(2)]
        inputs = prepare_batch(fixtures)
        originals = [x.copy() for x in inputs]
        expected = fixed_batch(*inputs, 32)
        self.assertEqual(check_result(expected, expected, fixtures, originals, inputs)["score"], 1)
        bad = expected.copy()
        bad[40, 1] = 2
        self.assertEqual(check_result(bad, expected, fixtures, originals, inputs)["score"], 0)
        inputs[3][0, 0] *= 2
        self.assertEqual(check_result(expected, expected, fixtures, originals, inputs)["score"], 0)

    def test_empty_or_mixed_contact_batches_rejected(self):
        for fixtures in ([], [build_fixture("plane", 8, 0), build_fixture("plane", 33, 1)]):
            with self.assertRaises(ValueError):
                prepare_batch(fixtures)


if __name__ == "__main__":
    unittest.main()
