import ast
import unittest

import numpy as np

from math_tasks import TASKS, check, cpu, fixture, sources


class MathTaskTests(unittest.TestCase):
    def test_all_cpu_cases(self):
        for task in TASKS:
            for seed in range(16):
                inputs = fixture(task, seed)
                originals = tuple(x.copy() for x in inputs)
                for optimized in (False, True):
                    with self.subTest(task=task, seed=seed, optimized=optimized):
                        self.assertEqual(check(task, cpu(task, inputs, optimized), inputs, originals)["score"], 1)

    def test_rejects_wrong_sign_nonfinite_shape_and_input_changes(self):
        for task in TASKS:
            inputs = fixture(task, 3)
            originals = tuple(x.copy() for x in inputs)
            correct = cpu(task, inputs)
            for wrong in (-correct, correct + 1, correct[:, 0], np.full_like(correct, np.nan)):
                self.assertEqual(check(task, wrong, inputs, originals)["score"], 0)
            inputs[0].flat[0] += 1
            self.assertEqual(check(task, correct, inputs, originals)["score"], 0)

    def test_analytic_cases(self):
        x, v, k, c = fixture("spring", 3)
        x.fill(2); v.fill(-1); k.fill(3); c.fill(4)
        np.testing.assert_array_equal(cpu("spring", (x, v, k, c)), -2)
        forces, = fixture("net-force", 3)
        forces.fill(1)
        np.testing.assert_array_equal(cpu("net-force", (forces,)), 8)

    def test_reviewed_source_syntax_and_distinctness(self):
        for task in TASKS:
            forms = sources(task)
            self.assertEqual(len(forms), 2)
            self.assertEqual(len(set(forms.values())), 2)
            for source in forms.values():
                ast.parse(source)


if __name__ == "__main__":
    unittest.main()
