"""Focused regression checks for the acceptance gate: no false passes.

These run without the Neuron SDK. They exercise the fail-closed traffic gate and the single
acceptance decision (nkibench.accept_case) that both the CLI verify path and the agent loop
use, so the two can never disagree about what passed.

    python3 -m unittest test_traffic_eval -v        # from projects/02-kernel-agent
"""
import unittest

import nkibench
import test_time_learning as ttl


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


class BanditTests(unittest.TestCase):
    def test_untried_arms_in_priority_order(self):
        b = ttl.Bandit()
        seen = []
        for _ in range(len(ttl.ARMS)):
            arm = b.select()
            seen.append(arm)
            b.update(arm, 0.0)
        self.assertEqual(seen, list(ttl.ARMS))

    def test_equal_means_break_ties_in_priority_order(self):
        b1, b2 = ttl.Bandit(), ttl.Bandit()
        for arm in ttl.ARMS:
            b1.update(arm, 0.5)
            b2.update(arm, 0.5)
        self.assertEqual(b1.select(), ttl.ARMS[0])
        self.assertEqual(b2.select(), ttl.ARMS[0])

    def test_best_mean_wins_without_exploration(self):
        b = ttl.Bandit(exploration=0.0)
        for arm in ttl.ARMS:
            b.update(arm, 0.0)
        b.update("retain_both", 1.0)
        self.assertEqual(b.select(), "retain_both")

    def test_reward_clamped_to_unit_interval(self):
        b = ttl.Bandit()
        b.update("retain_both", 5.0)
        self.assertEqual(b.totals["retain_both"], 1.0)
        b.update("retain_both", -2.0)
        self.assertEqual(b.totals["retain_both"], 1.0)
        self.assertEqual(b.counts["retain_both"], 2)


class MemoryTests(unittest.TestCase):
    def test_best_lines_and_recent_failures(self):
        m = ttl.Memory(cap=10)
        m.append(dict(ok=True, improvement=0.5, lesson="big win"))
        m.append(dict(ok=True, improvement=0.1, lesson="small win"))
        m.append(dict(ok=False, improvement=0.0, lesson="failed A"))
        m.append(dict(ok=False, improvement=0.0, lesson="failed B"))
        lines = m.prompt_lines(best_k=1, failures_k=1)
        self.assertEqual(lines["best"], ["big win"])
        self.assertEqual(lines["failures"], ["failed B"])

    def test_cap_enforced(self):
        m = ttl.Memory(cap=3)
        for i in range(5):
            m.append(dict(ok=True, improvement=i, lesson=f"l{i}"))
        self.assertEqual(len(m.items), 3)
        self.assertEqual(m.items[0]["lesson"], "l2")

    def test_best_lines_rank_by_progress_not_only_wins(self):
        m = ttl.Memory(cap=10)
        m.append(dict(ok=False, improvement=0.0, progress=0.6, lesson="near miss A"))
        m.append(dict(ok=False, improvement=0.0, progress=0.3, lesson="near miss B"))
        lines = m.prompt_lines(best_k=1, failures_k=0)
        self.assertEqual(lines["best"], ["near miss A"])


class PopulationTests(unittest.TestCase):
    @staticmethod
    def member(name, worst, valid=True):
        return dict(hash=name, source=name, strategy="tidy_only", worst_waste=worst,
                    valid=valid)

    def test_invalid_never_admitted(self):
        p = ttl.Population(size=2)
        self.assertFalse(p.insert(self.member("a", 2.0, valid=False)))
        self.assertEqual(p.members, [])

    def test_worst_replaced_only_if_better(self):
        p = ttl.Population(size=2)
        self.assertTrue(p.insert(self.member("a", 1.5)))
        self.assertTrue(p.insert(self.member("b", 2.0)))
        self.assertFalse(p.insert(self.member("c", 2.5)))
        self.assertTrue(p.insert(self.member("d", 1.2)))
        self.assertEqual([m["hash"] for m in p.members], ["d", "a"])

    def test_duplicate_hash_rejected(self):
        p = ttl.Population(size=2)
        p.insert(self.member("a", 1.5))
        self.assertFalse(p.insert(self.member("a", 1.0)))
        self.assertEqual(len(p.members), 1)

    def test_parent_deterministic_without_diversity(self):
        import random
        p = ttl.Population(size=2)
        p.insert(self.member("a", 1.5))
        p.insert(self.member("b", 2.0))
        rng = random.Random(0)
        self.assertEqual(p.pick_parent(rng, second_prob=0.0)["hash"], "a")


class RewardTests(unittest.TestCase):
    """The loop's reward and stopping rule, exercised on synthetic evaluations."""

    def test_feedback_enrichment_translates_the_partition_wall(self):
        import traffic_agent
        raw = "raised AssertionError: dma_copy dst partition dimension 256 exceeds maximum 128"
        msg = traffic_agent.enrich_feedback(raw)
        self.assertIn("free (second) dimension", msg)
        self.assertIn("[128, k_tiles * M]", msg)
        psum = "raised AssertionError: dma_copy requires HBM or SBUF tensors, got " \
               "src=MemoryRegion.psum, dst=MemoryRegion.shared_hbm"
        msg2 = traffic_agent.enrich_feedback(psum)
        self.assertIn("nisa.tensor_copy", msg2)

    def test_feedback_enrichment_delegates_element_mismatch(self):
        import traffic_agent
        raw = ("raised AssertionError: dma_copy requires src and dst to have the same number "
               "of elements, got src=65536, dst=16384")
        msg = traffic_agent.enrich_feedback(raw)
        self.assertIn("EXACTLY the shape", msg)

    @staticmethod
    def _ev(bytes_, floor, checks_ok=True, accepted=False, worst=1.0):
        checks = dict(inputs_ok=checks_ok, numerics_ok=checks_ok, hazard_ok=checks_ok,
                      traffic_ok=False)
        return dict(rules_ok=checks_ok, numerics_ok=checks_ok, inputs_ok=checks_ok,
                    hazards_ok=checks_ok, passed=1 if checks_ok else 0, accepted=accepted,
                    per_case=[dict(bytes=bytes_, floor=floor, waste=bytes_ / floor,
                                   checks=checks)],
                    worst_waste=worst if checks_ok else None)

    def test_invalid_candidate_earns_zero_reward(self):
        import traffic_agent
        bad = self._ev(2_000_000, 2_000_000, checks_ok=False)
        self.assertEqual(traffic_agent.improvement(2.0, bad), 0.0)

    def test_improvement_clamped_to_unit_interval(self):
        import traffic_agent
        good = self._ev(2_000_000, 2_000_000, worst=1.0)
        self.assertEqual(traffic_agent.improvement(2.0, good), 1.0)
        self.assertEqual(traffic_agent.improvement(1.5, good), 0.5)
        self.assertEqual(traffic_agent.improvement(None, good), 0.0)

    def test_below_floor_candidate_is_not_valid(self):
        import traffic_agent
        under = self._ev(1_000_000, 2_000_000, worst=0.5)
        self.assertFalse(traffic_agent.candidate_valid(under))
        self.assertEqual(traffic_agent.improvement(2.0, under), 0.0)

    def test_at_floor_requires_acceptance_and_exact_floor(self):
        import traffic_agent
        self.assertTrue(traffic_agent.at_floor(dict(accepted=True,
                                                    per_case=[dict(bytes=100, floor=100)])))
        self.assertFalse(traffic_agent.at_floor(dict(accepted=True,
                                                     per_case=[dict(bytes=101, floor=100)])))
        self.assertFalse(traffic_agent.at_floor(dict(accepted=False,
                                                     per_case=[dict(bytes=100, floor=100)])))


class ProgressTests(unittest.TestCase):
    """Tiered progress: the bandit's gradient without counting wrong kernels as wins."""

    @staticmethod
    def _ev(ran, correct, waste, total=4, rules=True):
        cases = []
        for i in range(total):
            if i < ran:
                ch = dict(inputs_ok=True, numerics_ok=(i < correct), hazard_ok=True,
                          traffic_ok=False)
                cases.append(dict(bytes=None, floor=None,
                                  waste=(waste if i < correct else None), checks=ch))
            else:
                cases.append(dict(bytes=None, floor=None, waste=None, checks=None))
        return dict(rules_ok=rules, numerics_ok=rules, inputs_ok=rules, hazards_ok=rules,
                    passed=correct, accepted=False, total=total, per_case=cases,
                    worst_waste=None, failure_kind=None, feedback="")

    def test_seed_like_scores_0_8(self):
        import traffic_agent as ta
        self.assertAlmostEqual(ta.progress(self._ev(4, 4, 2.0)), 0.8, places=2)

    def test_floor_kernel_scores_1_0(self):
        import traffic_agent as ta
        self.assertAlmostEqual(ta.progress(self._ev(4, 4, 1.0)), 1.0, places=2)

    def test_rule_failure_scores_zero(self):
        import traffic_agent as ta
        self.assertEqual(ta.progress(self._ev(0, 0, None, rules=False)), 0.0)

    def test_partial_near_miss_beats_zero_but_not_the_seed(self):
        import traffic_agent as ta
        partial = ta.progress(self._ev(1, 1, 1.0))
        self.assertGreater(partial, 0.0)
        self.assertLess(partial, ta.progress(self._ev(4, 4, 2.0)))

    def test_three_correct_shapes_at_floor_beat_the_seed(self):
        import traffic_agent as ta
        self.assertGreater(ta.progress(self._ev(4, 3, 1.0)),
                           ta.progress(self._ev(4, 4, 2.0)))

    def test_bandit_reward_is_progress_gain(self):
        import traffic_agent as ta
        self.assertEqual(ta.bandit_reward(None, self._ev(4, 4, 1.0)), 0.0)
        self.assertEqual(ta.bandit_reward(0.8, self._ev(4, 4, 2.0)), 0.0)
        self.assertGreater(ta.bandit_reward(0.8, self._ev(4, 4, 1.0)), 0.0)


if __name__ == "__main__":
    unittest.main()
