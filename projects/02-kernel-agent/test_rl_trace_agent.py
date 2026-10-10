#!/usr/bin/env python3
"""
Offline tests for rl_trace_agent.py (v5). No Neuron SDK, no model server.

    python test_rl_trace_agent.py

When the real `nki` package is missing, a small NumPy stand-in is installed so the real harness
(agent.grade, nkibench.simulate_and_count) still runs end to end. The stand-in checks THE LOOP, not
NKI semantics: the signatures it enforces (nl.ds(start, size), tensor_copy(dst, src, ...)) are the
ones the real verifier printed in the v4.1 trace. With the real SDK installed the stand-in is not
used and the same tests run against the real simulator.
"""
import json
import os
import sys
import tempfile
import types
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


# ---------------------------------------------------------------------------- nki stand-in

def install_stub():
    class T:
        def __init__(self, arr):
            self.arr = arr

        shape = property(lambda s: s.arr.shape)
        dtype = property(lambda s: s.arr.dtype)
        nbytes = property(lambda s: s.arr.nbytes)

        def __getitem__(self, idx):
            return T(self.arr[idx])

        def __setitem__(self, idx, v):
            self.arr[idx] = v.arr if isinstance(v, T) else v

    nki = types.ModuleType("nki")
    nl = types.ModuleType("nki.language")
    nisa = types.ModuleType("nki.isa")
    typing_ = types.ModuleType("nki.typing")
    typing_.tensor = object

    def ds(start, size):
        return slice(start, start + size)

    def ndarray(shape, dtype=None, buffer=None):
        return T(np.zeros(shape, dtype or np.float32))

    def tensor_copy(dst, src, engine=None, name=None):
        assert dst.shape == src.shape, f"tensor_copy shape mismatch {dst.shape} vs {src.shape}"
        dst.arr[...] = src.arr

    def dma_copy(dst=None, src=None, **kw):
        assert dst.arr.size == src.arr.size, (
            f"dma_copy requires src and dst to have the same number of elements, "
            f"got src={src.arr.size}, dst={dst.arr.size}")
        dst.arr[...] = src.arr.reshape(dst.arr.shape)

    nl.ds, nl.ndarray, nl.affine_range = ds, ndarray, range
    nl.sbuf, nl.psum, nl.shared_hbm, nl.float32 = "sbuf", "psum", "shared_hbm", np.float32
    nl.sum, nl.min, nl.max, nl.exp, nl.divide = (lambda *a, **k: None,) * 5
    nisa.tensor_copy, nisa.dma_copy = tensor_copy, dma_copy

    def jit(fn):
        def wrapper(*args):
            conv = [T(a) if isinstance(a, np.ndarray) else a for a in args]
            out = fn(*conv)
            return out.arr if isinstance(out, T) else out
        wrapper.__name__ = fn.__name__
        return wrapper

    nki.jit = jit
    nki.simulate = lambda kernel: kernel
    nki.language, nki.isa, nki.typing = nl, nisa, typing_
    sys.modules.update({"nki": nki, "nki.language": nl, "nki.isa": nisa, "nki.typing": typing_})


try:
    import nki  # noqa: F401
    USING_STUB = False
except ImportError:
    install_stub()
    USING_STUB = True

import agent as base_agent            # noqa: E402
import nkibench                       # noqa: E402
import rl_trace_agent as rl           # noqa: E402

REFERENCE = open(os.path.join(HERE, "reference_level2.py"), encoding="utf-8").read()

# The three failure shapes from the v4.1 trace.
BAD_TUPLE = '''import nki
import nki.language as nl
import nki.isa as nisa

@nki.jit
def tensor_transpose2D_kernel_(in_tensor, shape2D):
    out = nl.ndarray(in_tensor.shape, dtype=in_tensor.dtype, buffer=nl.shared_hbm)
    t = nl.ndarray(in_tensor.shape, dtype=in_tensor.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=t, src=in_tensor)
    o = nl.ndarray(in_tensor.shape, dtype=in_tensor.dtype, buffer=nl.sbuf)
    F1, F2 = shape2D
    for a in nl.affine_range(F1):
        for b in nl.affine_range(F2):
            nisa.tensor_copy(dst=o[:, nl.ds((b * F1 + a, 1))], src=t[:, nl.ds(a * F2 + b, 1)])
    nisa.dma_copy(dst=out, src=o)
    return out
'''
BAD_KEYWORD = BAD_TUPLE.replace(
    "nisa.tensor_copy(dst=o[:, nl.ds((b * F1 + a, 1))], src=t[:, nl.ds(a * F2 + b, 1)])",
    "nisa.tensor_copy(dst=o, src=t, ds=nl.ds(b * F1 + a, 1))")
NO_TRANSPOSE = BAD_TUPLE.replace(
    "nisa.tensor_copy(dst=o[:, nl.ds((b * F1 + a, 1))], src=t[:, nl.ds(a * F2 + b, 1)])",
    "nisa.tensor_copy(dst=o[:, nl.ds(a * F2 + b, 1)], src=t[:, nl.ds(a * F2 + b, 1)])")
NEVER_WRITTEN = BAD_TUPLE.replace(
    "nisa.tensor_copy(dst=o[:, nl.ds((b * F1 + a, 1))], src=t[:, nl.ds(a * F2 + b, 1)])", "pass")


def fenced(code):
    return f"```python\n{code}\n```"


class Args:
    """The parts of argparse.Namespace that rl_trace_agent reads."""
    def __init__(self, **kw):
        self.mode, self.hints, self.terse, self.trace = "reflect", "api", 0, False
        self.rounds, self.samples, self.episodes, self.seed = 4, 4, 1, 7
        self.patience, self.no_variants, self.no_flow, self.verbose = 0, False, False, False
        self.seed_references, self.sft_min_reward, self.no_seed = False, 0.999, False
        self.same_temp = False
        self.model, self.base, self.context, self.max_tokens, self.think = "m", "http://x", 8192, 2500, False
        self.__dict__.update(kw)


def make_run(args, tmpdir):
    strategy = rl.ShrunkUCB(rl.STRATEGIES)
    temperature = rl.ShrunkUCB(rl.TEMPERATURES)
    state = rl.State(os.path.join(tmpdir, "state.json"))
    state.strategy, state.temperature = strategy, temperature
    log = open(os.path.join(tmpdir, "log.jsonl"), "w", encoding="utf-8")
    groups = open(os.path.join(tmpdir, "groups.jsonl"), "w", encoding="utf-8")
    import random
    return rl.Run(args, strategy, temperature, state, log, None, groups, random.Random(1), HERE)


def read_rows(tmpdir):
    with open(os.path.join(tmpdir, "log.jsonl"), encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# ---------------------------------------------------------------------------- tests

class TestLint(unittest.TestCase):
    def test_tuple_to_ds_is_located(self):
        issues = rl.lint_api(BAD_TUPLE)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]["line"], 14)
        self.assertIn("ONE tuple", issues[0]["msg"])
        self.assertIn("nl.ds(start, size)", issues[0]["sig"])

    def test_invented_keyword_is_located(self):
        issues = rl.lint_api(BAD_KEYWORD)
        self.assertTrue(any("ds" in i["msg"] and "unexpected keyword" in i["msg"] for i in issues))

    def test_reference_is_clean(self):
        self.assertEqual(rl.lint_api(REFERENCE), [])

    def test_unknown_names_are_not_flagged_without_evidence(self):
        code = "import nki.language as nl\nx = nl.something_new(1, 2, 3)\n"
        # With the stand-in nl module the name is provably missing; with no module it is unknown.
        issues = rl.lint_api(code)
        self.assertTrue(all("does not exist" in i["msg"] for i in issues))

    def test_fallback_table_works_without_the_sdk(self):
        saved = {k: sys.modules.pop(k) for k in ("nki", "nki.language", "nki.isa") if k in sys.modules}
        rl._lookup.cache_clear()
        try:
            self.assertEqual(len(rl.lint_api(BAD_TUPLE)), 1)
        finally:
            sys.modules.update(saved)
            rl._lookup.cache_clear()

    def test_model_written_change_that_breaks_the_api_is_rejected(self):
        self.assertFalse(rl.diagnosis_is_sane("Replace with `nl.ds((1, 1))`"))
        self.assertFalse(rl.diagnosis_is_sane("Use `nisa.tensor_copy(dst=a, src=b, ds=1)`"))
        self.assertTrue(rl.diagnosis_is_sane("Use `nl.ds(1, 1)` instead"))
        self.assertTrue(rl.diagnosis_is_sane("no code spans at all"))


class TestClassify(unittest.TestCase):
    def test_trace_messages(self):
        c = rl.classify_feedback
        self.assertEqual(c("0 of 4 shapes passed. On shape=(32, 12) as 3x4: raised TypeError: ds() "
                           "missing 1 required positional argument: 'size'", False), "api_signature")
        self.assertEqual(c("0 of 4 shapes passed. On s: raised TypeError: unsupported operand "
                           "type(s) for +: 'int' and 'tuple'", False), "runtime_error")
        self.assertEqual(c("0 of 4 shapes passed. On s: NUMERICAL MISMATCH: worst error 4.49", False),
                         "correctness")
        self.assertEqual(c("raised TypeError: tensor_copy() got an unexpected keyword argument 'ds'",
                           False), "api_name")
        self.assertEqual(c("anything", True), "verified")


class TestFlow(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.inp = rng.standard_normal((8, 12)).astype(np.float32)
        self.shape = (3, 4)
        self.want = nkibench.ref_transpose2d(self.inp, self.shape)

    def test_correct_output_has_no_diagnosis(self):
        self.assertEqual(rl.explain_permutation(self.want, self.inp, self.shape), "")

    def test_identity_is_named(self):
        text = rl.explain_permutation(self.inp, self.inp, self.shape)
        self.assertIn("SAME index", text)
        self.assertIn("should hold input[0, 4]", text)

    def test_swapped_dimensions_are_named(self):
        got = nkibench.ref_transpose2d(self.inp, (4, 3))
        self.assertIn("wrong way round", rl.explain_permutation(got, self.inp, self.shape))

    def test_unwritten_destinations_are_named(self):
        got = self.want.copy()
        got[:, 5:] = 0.0
        self.assertIn("never written", rl.explain_permutation(got, self.inp, self.shape))


class TestReward(unittest.TestCase):
    def test_progress_orders_failures(self):
        crash = rl.progress_score({"runs": False}, "raised TypeError", False)
        close = rl.progress_score({"runs": True}, "NUMERICAL MISMATCH 10.0% of elements are outside "
                                                  "tolerance", False)
        far = rl.progress_score({"runs": True}, "NUMERICAL MISMATCH 90.0% of elements are outside "
                                                "tolerance", False)
        self.assertEqual((crash, rl.progress_score({}, "", True)), (0.0, 1.0))
        self.assertGreater(close, far)
        self.assertGreater(far, crash)

    def test_rewards_stay_in_range_and_verifier_dominates(self):
        ok = rl.shaped_reward(1.0, 0, 1.0, 1.0, 1.0, 1.0, False)
        bad = rl.shaped_reward(0.3, 0, 0.0, 0.0, 1.0, 0.0, False)
        self.assertAlmostEqual(ok, 1.0)
        self.assertLess(bad, 0.35)

    def test_advantages(self):
        raw, norm = rl.group_advantages([0.3, 0.3, 0.3, 0.3])
        self.assertEqual((raw, norm), ([0.0] * 4, [0.0] * 4))
        raw, norm = rl.group_advantages([0.0, 1.0])
        self.assertAlmostEqual(norm[1], 1.0)
        self.assertAlmostEqual(sum(norm), 0.0)


class TestSampling(unittest.TestCase):
    def test_temperatures_spread(self):
        t = rl.sample_temps(0.6, 4)
        self.assertEqual(len(set(t)), 4)
        self.assertTrue(all(0.2 <= x <= 1.1 for x in t))

    def test_first_sample_keeps_the_canonical_prompt(self):
        p = rl.sample_prompts("P", 4, True)
        self.assertEqual(p[0], "P")
        self.assertEqual(len(set(p)), 4)
        self.assertEqual(rl.sample_prompts("P", 4, False), ["P"] * 4)


class TestProbe(unittest.TestCase):
    def test_exception_gets_a_line_number(self):
        probe = rl.probe_kernel(2, BAD_TUPLE)
        self.assertEqual(probe["kind"], "exception")
        self.assertEqual(probe["lineno"], 14)
        self.assertIn("nl.ds", probe["src"])

    def test_mismatch_gets_a_flow_check(self):
        probe = rl.probe_kernel(2, NO_TRANSPOSE)
        self.assertEqual(probe["kind"], "mismatch")
        self.assertIn("SAME index", rl.located_analysis(2, NO_TRANSPOSE, probe))
        self.assertEqual(rl.located_analysis(2, NO_TRANSPOSE, probe, reveal_flow=False), "")

    def test_reference_passes(self):
        self.assertEqual(rl.probe_kernel(2, REFERENCE)["kind"], "ok")


class ScriptedModel:
    """Stands in for the endpoint. Which kernel it returns depends on what the prompt contains, so
    the test exercises the real prompt-building path."""
    def __init__(self, solve_when_marker="CHECKER ANALYSIS"):
        self.calls, self.marker = [], solve_when_marker
        self.dup_variants = [BAD_KEYWORD, NO_TRANSPOSE, NEVER_WRITTEN]

    def __call__(self, args, prompt, temperature, max_tokens=None, seed=None):
        self.calls.append(prompt)
        if self.marker in prompt:
            return fenced(REFERENCE), "stop", 100, 100
        if rl.DUP_NOTE.strip()[:30] in prompt:
            code = self.dup_variants[len(self.calls) % len(self.dup_variants)]
            return fenced(code), "stop", 100, 100
        return fenced(BAD_TUPLE), "stop", 100, 100


class TestEpisode(unittest.TestCase):
    def run_episode(self, args, model, tmpdir):
        rl.ask_model = model
        run = make_run(args, tmpdir)
        out = rl.run_episode(run, 2, 0, "test")
        run.log.close()
        run.groups.close()
        return out

    def test_grounded_analysis_solves_what_v41_could_not(self):
        with tempfile.TemporaryDirectory() as d:
            model = ScriptedModel()
            best, verified = self.run_episode(Args(rounds=4), model, d)
            rows = read_rows(d)
            self.assertTrue(verified)
            self.assertEqual(best, 1.0)
            r0 = [r for r in rows if r["round"] == 0]
            # identical replies in round 0 are noticed and re-asked, so the group has diversity
            self.assertTrue(all(r["resampled"] >= 1 for r in r0))
            self.assertGreater(r0[0]["group_distinct"], 1)
            # the lint finding rides in the feedback, the located analysis in the repair prompt
            self.assertTrue(any(r["lint"] for r in r0))
            r1 = [r for r in rows if r["round"] == 1]
            self.assertEqual(r1[0]["diagnosis_source"], "grounded")
            self.assertIn("nl.ds(start, size)", r1[0]["prompt"])
            self.assertIn("CHECKER ANALYSIS", r1[0]["prompt"])
            # no model call was spent on a reflection that could hallucinate
            self.assertTrue(all("REFLECTION TASK" not in p for p in model.calls))
            # categories are no longer all "correctness"
            self.assertTrue(any(r["failure_category"] in ("api_signature", "api_name") for r in r0))
            # group export is written, with advantages
            with open(os.path.join(d, "groups.jsonl"), encoding="utf-8") as f:
                g = [json.loads(x) for x in f]
            self.assertTrue(g and "advantage" in g[0]["completions"][0])

    def test_state_learns_api_facts(self):
        with tempfile.TemporaryDirectory() as d:
            rl.ask_model = ScriptedModel()
            run = make_run(Args(rounds=3), d)
            rl.run_episode(run, 2, 0, "t")
            run.log.close()
            run.groups.close()
            self.assertTrue(any("nl.ds" in k for k in run.state.facts))
            self.assertTrue(any("nl.ds" in f for f in run.state.relevant_facts()))

    def test_plain_mode_is_the_baseline_prompt(self):
        with tempfile.TemporaryDirectory() as d:
            model = ScriptedModel(solve_when_marker="@@never@@")
            self.run_episode(Args(mode="plain", rounds=2, samples=1), model, d)
            self.assertEqual(model.calls[0], base_agent.first_prompt(2, 0))
            self.assertIn("is not right yet", model.calls[1])
            self.assertNotIn("CALL SIGNATURES", model.calls[1])

    def test_identical_samples_give_zero_signal_and_are_logged_as_such(self):
        with tempfile.TemporaryDirectory() as d:
            self.run_episode(Args(rounds=1, no_variants=True), ScriptedModel("@@never@@"), d)
            rows = read_rows(d)
            self.assertTrue(all(r["group_distinct"] == 1 for r in rows))
            self.assertTrue(all(r["advantage_norm"] == 0.0 for r in rows))

    def test_grade_cache_skips_duplicate_simulations(self):
        with tempfile.TemporaryDirectory() as d:
            self.run_episode(Args(rounds=1, no_variants=True), ScriptedModel("@@never@@"), d)
            self.assertEqual(sum(r["grade_cached"] for r in read_rows(d)), 3)


if __name__ == "__main__":
    print(f"nki: {'NumPy stand-in' if USING_STUB else 'real SDK'}")
    unittest.main(verbosity=2)