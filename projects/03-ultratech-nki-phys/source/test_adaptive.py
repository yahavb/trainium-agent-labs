import unittest

import numpy as np

from adaptive_reference import solve


class AdaptiveTests(unittest.TestCase):
    def problem(self):
        a = np.diag([.5, 1]).astype(np.float32)
        b = np.array([[-.5], [0]], dtype=np.float32)
        return a.T.copy(), b, np.zeros_like(b), np.ones_like(b)

    def test_hand_worked_fixed_beta_restart(self):
        inputs = self.problem()
        value, trace = solve(*inputs, 3, fixed_beta=.9)
        np.testing.assert_allclose(value[:, 0], [1.20125, 0], atol=2e-7)
        self.assertEqual(trace[0]["restart_steps"], [3])
        next_value, _ = solve(*inputs, 4, fixed_beta=.9)
        self.assertLess(abs(next_value[0, 0] - 1), abs(value[0, 0] - 1))

    def test_fista_first_step_has_no_momentum(self):
        value, _ = solve(*self.problem(), 2)
        np.testing.assert_array_equal(value[:, 0], [.75, 0])

    def test_world_isolation_padding_and_inputs(self):
        a, b, x, alpha = self.problem()
        matrix = np.concatenate([a, a], axis=1)
        bias = np.concatenate([b, b * 3], axis=1)
        initial = np.zeros_like(bias)
        rate = np.ones_like(bias)
        originals = [v.copy() for v in (matrix, bias, initial, rate)]
        values, trace = solve(matrix, bias, initial, rate, 100)
        expected, _ = solve(a, b, x, alpha, 100)
        np.testing.assert_array_equal(values[:, :1], expected)
        np.testing.assert_allclose(values[0], [1, 3], atol=1e-6)
        self.assertTrue(np.all(values[1] == 0))
        for actual, original in zip((matrix, bias, initial, rate), originals):
            np.testing.assert_array_equal(actual, original)
        self.assertEqual(len(trace), 2)


if __name__ == "__main__":
    unittest.main()
