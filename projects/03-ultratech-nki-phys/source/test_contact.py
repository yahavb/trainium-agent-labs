import unittest
import numpy as np
from contact import build_fixture, diagnostics, oracle, projected_gradient, velocity_error_bound


class ContactTests(unittest.TestCase):
    def test_physical_stopping_repairs_premature_stack_exit(self):
        f = build_fixture("stack", 8, 1)
        reference = oracle(f["A"], f["b"])
        early, _ = projected_gradient(f["A"], f["b"])
        self.assertFalse(diagnostics(f["A"], f["b"], early, reference, f)["passed"])
        repaired, _ = projected_gradient(f["A"], f["b"], fixture=f)
        self.assertTrue(diagnostics(f["A"], f["b"], repaired, reference, f)["passed"])

    def test_velocity_bound_dominates_actual_error(self):
        for scene in ("plane", "pairs", "stack"):
            f = build_fixture(scene, 8, 1)
            reference = oracle(f["A"], f["b"])
            candidate = reference * 0.9
            actual = np.max(np.abs(f["inverse_mass"] * (f["J"].T @ (candidate - reference))))
            self.assertLessEqual(actual, velocity_error_bound(f["A"], f["b"], candidate, f) + 1e-12)

    def test_single_contact_analytic_solution(self):
        for speed in (-2.0, 0.0, 1.0):
            a, b = np.array([[0.5]]), np.array([speed])
            np.testing.assert_allclose(oracle(a, b), [max(0, -speed / 0.5)], atol=1e-12)

    def test_two_body_inelastic_momentum(self):
        masses = np.array([2.0, 3.0])
        velocities = np.array([1.0, -1.0])
        j = np.array([[-1.0, 1.0]])
        a = (j / masses) @ j.T
        impulses = oracle(a, j @ velocities)
        after = velocities + (j.T @ impulses) / masses
        np.testing.assert_allclose(after, [-0.2, -0.2], atol=1e-12)
        self.assertAlmostEqual(float(masses @ after), float(masses @ velocities))

    def test_rejects_planted_failures(self):
        f = build_fixture("stack", 8, 3)
        reference = oracle(f["A"], f["b"])
        self.assertGreater(np.linalg.norm(reference), 0)
        for wrong in (-reference, np.zeros_like(reference), reference * 1.1,
                      np.full_like(reference, np.nan), reference[:-1]):
            self.assertFalse(diagnostics(f["A"], f["b"], wrong, reference, f)["passed"])

    def test_all_scene_oracles_certified(self):
        for scene in ("plane", "pairs", "stack"):
            f = build_fixture(scene, 33, 19)
            reference = oracle(f["A"], f["b"])
            self.assertLess(diagnostics(f["A"], f["b"], reference, reference, f)["residual"], 1e-9)


if __name__ == "__main__":
    unittest.main()
