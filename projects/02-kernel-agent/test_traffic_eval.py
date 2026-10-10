"""Focused regression checks for the acceptance gate: no false passes.

These run without the Neuron SDK. They exercise the fail-closed traffic gate and the single
acceptance decision (nkibench.accept_case) that both the CLI verify path and the agent loop
use, so the two can never disagree about what passed.

    python3 -m unittest test_traffic_eval -v        # from projects/02-kernel-agent
"""
import unittest

import nkibench


class TrafficGateTests(unittest.TestCase):
    """The optimization-level gate must fail closed on missing or fake measurements."""

    LEVEL = 5  # first level with a traffic bar (1.60x)

    def setUp(self):
        self.case = nkibench.LEVELS[self.LEVEL]["shapes"][0]  # K=128 M=128 N=512
        self.args, _ = nkibench.make_inputs(self.case, self.LEVEL, seed=0)
        self.want = nkibench.LEVELS[self.LEVEL]["ref"](*self.args)
        self.floor = nkibench.minimum_hbm_bytes(self.args, self.want)

    def counted(self, bytes_, **kw):
        c = dict(bytes=bytes_, transfers=3, dtypes={"float32"}, warnings=[])
        c.update(kw)
        return c

    def test_floor_verified_passes(self):
        self.assertIsNone(
            nkibench.check_traffic_bar(self.LEVEL, self.counted(self.floor), self.args, self.want))

    def test_zero_traffic_fails_closed(self):
        m = nkibench.check_traffic_bar(self.LEVEL, self.counted(0), self.args, self.want)
        self.assertIsNotNone(m)
        self.assertIn("TRAFFIC UNMEASURED", m)

    def test_missing_bytes_key_fails_closed(self):
        m = nkibench.check_traffic_bar(self.LEVEL, dict(transfers=0), self.args, self.want)
        self.assertIsNotNone(m)
        self.assertIn("TRAFFIC UNMEASURED", m)

    def test_unmeasured_transfers_fail(self):
        m = nkibench.check_traffic_bar(
            self.LEVEL, self.counted(self.floor, unmeasured=2), self.args, self.want)
        self.assertIsNotNone(m)
        self.assertIn("TRAFFIC UNVERIFIED", m)

    def test_above_bar_fails(self):
        m = nkibench.check_traffic_bar(
            self.LEVEL, self.counted(int(self.floor * 1.7)), self.args, self.want)
        self.assertIsNotNone(m)
        self.assertIn("TOO MUCH HBM TRAFFIC", m)

    def test_below_floor_fails_as_accounting(self):
        m = nkibench.check_traffic_bar(
            self.LEVEL, self.counted(self.floor - 1), self.args, self.want)
        self.assertIsNotNone(m)
        self.assertIn("BELOW THE BYTE FLOOR", m)

    def test_levels_without_bar_ignore_traffic(self):
        case = nkibench.LEVELS[4]["shapes"][0]
        args, _ = nkibench.make_inputs(case, 4, seed=0)
        want = nkibench.LEVELS[4]["ref"](*args)
        self.assertIsNone(nkibench.check_traffic_bar(4, dict(bytes=0, transfers=0), args, want))


class AcceptCaseTests(unittest.TestCase):
    """One decision function for the CLI and the loop; every component can fail it."""

    LEVEL = 5

    def setUp(self):
        self.case = nkibench.LEVELS[self.LEVEL]["shapes"][0]
        self.args, _ = nkibench.make_inputs(self.case, self.LEVEL, seed=0)
        self.before = [x.copy() for x in self.args]
        self.want = nkibench.LEVELS[self.LEVEL]["ref"](*self.args)
        self.got = self.want.copy()
        self.floor = nkibench.minimum_hbm_bytes(self.args, self.want)

    def counted(self, bytes_=None, **kw):
        c = dict(bytes=self.floor if bytes_ is None else bytes_,
                 transfers=3, dtypes={"float32"}, warnings=[])
        c.update(kw)
        return c

    def test_correct_at_floor_is_accepted(self):
        ok, m, checks = nkibench.accept_case(
            self.LEVEL, self.got, self.counted(), self.args, self.before, self.want)
        self.assertTrue(ok)
        self.assertIsNone(m)
        self.assertTrue(all(checks.values()))

    def test_numerical_error_is_rejected(self):
        got = self.got.copy()
        got[0, 0] += 10.0
        ok, m, checks = nkibench.accept_case(
            self.LEVEL, got, self.counted(), self.args, self.before, self.want)
        self.assertFalse(ok)
        self.assertFalse(checks["numerics_ok"])

    def test_input_mutation_is_rejected(self):
        mutated = [x.copy() for x in self.args]
        mutated[0][0, 0] += 1.0
        ok, m, checks = nkibench.accept_case(
            self.LEVEL, self.got, self.counted(), mutated, self.before, self.want)
        self.assertFalse(ok)
        self.assertIn("MODIFIED ITS INPUT", m)
        self.assertFalse(checks["inputs_ok"])

    def test_traffic_over_bar_is_rejected(self):
        ok, m, checks = nkibench.accept_case(
            self.LEVEL, self.got, self.counted(int(self.floor * 2)), self.args, self.before,
            self.want)
        self.assertFalse(ok)
        self.assertIn("TOO MUCH HBM TRAFFIC", m)
        self.assertFalse(checks["traffic_ok"])

    def test_hardware_hazard_is_rejected(self):
        c = self.counted(warnings=["this pattern produces incorrect results on hardware"])
        ok, m, checks = nkibench.accept_case(
            self.LEVEL, self.got, c, self.args, self.before, self.want)
        self.assertFalse(ok)
        self.assertIn("WRONG ON HARDWARE", m)
        self.assertFalse(checks["hazard_ok"])


if __name__ == "__main__":
    unittest.main()
