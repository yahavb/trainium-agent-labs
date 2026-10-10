"""Validate timing serialization without requiring the Neuron SDK."""

from types import SimpleNamespace
import unittest

import numpy as np
from benchmark_update import execute_runtime, runtime_inputs, runtime_outputs, timing_stats


class BenchmarkTests(unittest.TestCase):
    def result(self, **changes):
        return SimpleNamespace(**(dict(mean_ms=2, min_ms=1, max_ms=3,
                                       std_dev_ms=0.1, iterations=200,
                                       warmup_iterations=20, durations_ms=[2.0] * 200) | changes))

    def test_stats_preserve_milliseconds(self):
        self.assertEqual(timing_stats(self.result()), dict(mean_ms=2, min_ms=1,
                         max_ms=3, std_dev_ms=0.1, iterations=200,
                         warmup_iterations=20, durations_ms=[2.0] * 200))

    def test_invalid_samples_rejected(self):
        for samples in ([], [float("nan")] * 200, [-1] * 200):
            with self.subTest(samples=samples[:1]), self.assertRaises(ValueError):
                timing_stats(self.result(durations_ms=samples))

    def test_runtime_writes_explicit_outputs_without_return_value(self):
        data = np.zeros((2, 1), dtype=np.float32)
        tensor = SimpleNamespace(name="output_0", numpy=lambda: data)
        class Model:
            def allocate_output_tensors(self):
                return [tensor]

            def __call__(self, inputs, outputs):
                self_check = outputs["output_0"] is tensor
                if not self_check:
                    raise ValueError("Wrong output binding")
                data[:] = inputs["value"]
        model = Model()
        outputs = runtime_outputs(model)
        actual = execute_runtime(model, {"value": 3}, outputs)
        np.testing.assert_array_equal(actual, np.full((2, 1), 3, dtype=np.float32))
        data[:] = 4
        self.assertEqual(actual[0, 0], 3)

    def test_ambiguous_output_interface_rejected(self):
        for tensors in ([], [SimpleNamespace(name=None)], [1, 2]):
            with self.subTest(count=len(tensors)), self.assertRaises(ValueError):
                runtime_outputs(SimpleNamespace(allocate_output_tensors=lambda: tensors))

    def test_invalid_latencies_rejected(self):
        for invalid in (0, -1, float("nan"), float("inf")):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                timing_stats(self.result(mean_ms=invalid))

    def test_invalid_ranges_and_counts_rejected(self):
        for changes in (dict(min_ms=3), dict(max_ms=1), dict(std_dev_ms=-1), dict(iterations=0)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                timing_stats(self.result(**changes))

    def test_input_names_not_position_determine_mapping(self):
        arrays = [np.full((2, 2), i, dtype=np.float32) for i in range(3)]
        model = SimpleNamespace(input_tensors_info=dict(initial=None, bias=None, a_transposed=None))
        tensors = SimpleNamespace(from_numpy=lambda a, name: a.copy())
        bound = runtime_inputs(model, arrays, tensors)
        np.testing.assert_array_equal(bound["a_transposed"], arrays[0])
        np.testing.assert_array_equal(bound["initial"], arrays[2])

    def test_unknown_input_interface_rejected(self):
        with self.assertRaises(ValueError):
            runtime_inputs(SimpleNamespace(input_tensors_info={"input_0": None}),
                           [np.zeros(1)] * 3, None)


if __name__ == "__main__":
    unittest.main()
