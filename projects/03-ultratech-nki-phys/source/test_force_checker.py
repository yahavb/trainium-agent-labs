"""Frozen physics gates and trusted-controller public qualification checks."""

import unittest
import numpy as np
from force_checker import check_forces, public_gate


class ForceCheckerTests(unittest.TestCase):
    def setUp(self):
        self.fixture = dict(A=np.array([[2.0]]), b=np.array([-4.0]),
                            qacc_smooth=np.array([-2.0]), response=np.array([[1.0]]))
        self.reference = np.array([2.0])

    def test_correct_force_passes_and_zero_force_fails(self):
        self.assertTrue(check_forces(self.fixture, self.reference.astype(np.float32), self.reference)["passed"])
        self.assertFalse(check_forces(self.fixture, np.zeros(1), self.reference)["passed"])

    def test_acceleration_gate_catches_amplified_force_error(self):
        self.fixture["response"][:] = 100
        self.fixture["qacc_smooth"][:] = -200
        check = check_forces(self.fixture, np.array([2 + 1e-5]), self.reference)
        self.assertTrue(check["numerical"]["passed"])
        self.assertLess(check["force_error"], 1e-4)
        self.assertGreater(check["acceleration_error"], 1e-4)
        self.assertFalse(check["passed"])

    def test_invalid_outputs_rejected(self):
        for value in (np.array([np.nan]), np.array([-2.0]), np.array([[2.0]])):
            with self.subTest(value=value):
                self.assertFalse(check_forces(self.fixture, value, self.reference)["passed"])

    def test_gate_requires_complete_report_and_same_candidate(self):
        manifest = dict(suite_id="suite", contract_sha256="rules", fixtures=[dict(case_id="a"), dict(case_id="b")])
        report = dict(candidate_sha256="candidate", suite_id="suite", contract_sha256="rules",
                      cases=[dict(case_id="a", passed=True), dict(case_id="b", passed=True)])
        self.assertTrue(public_gate(report, manifest, "candidate"))
        self.assertFalse(public_gate(report, manifest, "changed-candidate"))
        for cases in ([dict(case_id="a", passed=True)],
                      [dict(case_id="a", passed=True), dict(case_id="a", passed=True)],
                      [dict(case_id="a", passed=True), dict(case_id="b", passed=False)]):
            self.assertFalse(public_gate(dict(report, cases=cases), manifest, "candidate"))
        self.assertFalse(public_gate(dict(report, contract_sha256="changed"), manifest, "candidate"))
        self.assertFalse(public_gate(dict(report, suite_id="other-suite"), manifest, "candidate"))


if __name__ == "__main__":
    unittest.main()
