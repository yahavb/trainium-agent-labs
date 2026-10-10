import unittest
import numpy as np

from plane_loop import canonical, review, reviewed_forms


class PlaneReviewTests(unittest.TestCase):
    def test_approved_forms(self):
        for name, source in reviewed_forms().items():
            self.assertEqual(review(source + "\n# harmless comment\n"), name)

    def test_missing_step_size_rejected(self):
        source = reviewed_forms()["copyfree"].replace(
            "            nisa.tensor_tensor(dst=scaled, data1=summed, data2=rate, op=nl.multiply)\n", "")
        with self.assertRaises(ValueError):
            review(source)

    def test_host_side_effects_rejected(self):
        with self.assertRaises(ValueError):
            review("import os\nos.system('id')\n" + reviewed_forms()["copyfree"])

    def test_local_rename_is_not_new_program(self):
        source = reviewed_forms()["copyfree"]
        self.assertEqual(canonical(source), canonical(source.replace("impulses", "state")))

    def test_docstrings_do_not_change_review(self):
        for name, source in reviewed_forms().items():
            without_module_docstring = source.split("\n", 1)[1]
            with_function_docstring = without_module_docstring.replace(
                "    contacts, worlds = bias.shape", '    """Different description."""\n    contacts, worlds = bias.shape')
            self.assertEqual(review(without_module_docstring), name)
            self.assertEqual(review(with_function_docstring), name)

    def test_scaling_fusion_preserves_update_mathematics(self):
        rng = np.random.default_rng(260)
        x = rng.random((64, 8)).astype(np.float32)
        gradient = rng.normal(size=(64, 8)).astype(np.float32)
        rate = rng.random((64, 8)).astype(np.float32)
        original = np.maximum(x - gradient * rate, 0)
        transformed = np.maximum(gradient * (-rate) + x, 0)
        np.testing.assert_array_equal(original, transformed)
        self.assertEqual(review(reviewed_forms()["scale-fused"]), "scale-fused")


if __name__ == "__main__":
    unittest.main()
