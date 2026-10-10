import tempfile
import unittest
from pathlib import Path

import numpy as np

from checker import grade_snapshot
from math_tasks import TASKS, check, cpu, fixture


class CheckerTests(unittest.TestCase):
    def test_all_public_equation_cases(self):
        for task in TASKS:
            for seed in range(16):
                inputs = fixture(task, seed)
                self.assertEqual(check(task, cpu(task, inputs), inputs, inputs)["score"], 1)

    def test_rejects_wrong_sign_dtype_shape_nan_and_changed_inputs(self):
        for task in TASKS:
            inputs = fixture(task, 3)
            originals = tuple(value.copy() for value in inputs)
            actual = cpu(task, inputs)
            for wrong in (-actual, actual + 1, actual.astype(np.float64),
                          actual[:, 0], np.full_like(actual, np.nan)):
                self.assertEqual(check(task, wrong, inputs, originals)["score"], 0)
            inputs[0].flat[0] += 1
            self.assertEqual(check(task, actual, inputs, originals)["score"], 0)

    def test_snapshot_does_not_claim_input_preservation(self):
        task = "spring"
        inputs = fixture(task, 3)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "output.npz"
            np.savez(path, actual=cpu(task, inputs),
                     **dict(zip(TASKS[task]["input_names"], inputs)))
            result = grade_snapshot(task, path)
        self.assertEqual(result["score"], 1)
        self.assertFalse(result["input_preservation_independently_verified"])


if __name__ == "__main__":
    unittest.main()
