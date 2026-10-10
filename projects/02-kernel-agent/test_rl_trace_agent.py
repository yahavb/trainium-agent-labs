#!/usr/bin/env python3
"""
Offline tests for rl_trace_agent.py (v6), kernel_lint.py, holdout.py, summarize_rl_trace.py
and run_matrix.py. No Neuron SDK, no model server.

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
import holdout as holdout_mod         # noqa: E402
import kernel_lint                    # noqa: E402
import nkibench                       # noqa: E402
import rl_trace_agent as rl           # noqa: E402
import run_matrix                     # noqa: E402
import summarize_rl_trace as summ     # noqa: E402

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


# Correct only when F1 == F2: the stride uses F2 where F1 belongs.
SQUARE_ONLY = BAD_TUPLE.replace(
    "nisa.tensor_copy(dst=o[:, nl.ds((b * F1 + a, 1))], src=t[:, nl.ds(a * F2 + b, 1)])",
    "nisa.tensor_copy(dst=o[:, nl.ds(b * F2 + a, 1)], src=t[:, nl.ds(a * F2 + b, 1)])")


def fenced(code):
    return f"```python\n{code}\n```"


class Args:
    """The parts of argparse.Namespace that rl_trace_agent reads."""
    def __init__(self, **kw):
        self.mode, self.hints, self.terse, self.trace = "reflect", "api", 0, False
        self.rounds, self.samples, self.episodes, self.seed = 4, 4, 1, 7
        self.patience, self.no_variants, self.no_flow, self.verbose = 0, False, False, False
        self.max_restarts, self.escalate, self.escalate_algo = 2, True, False
        self.plain_restart, self.no_analysis, self.probe_all = True, False, True
        self.holdout, self.independent_episodes = "report", False
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
        self.assertEqual(rl.located_analysis(2, NO_TRANSPOSE, probe, reveal_flow=False,
                                             all_shapes=False), "")

    def test_square_only_kernel_is_recognised(self):
        probe = rl.probe_kernel(2, SQUARE_ONLY)
        self.assertEqual(len(probe["cases"]), 4)
        text = rl.located_analysis(2, SQUARE_ONLY, probe)
        self.assertIn("SHAPE PATTERN", text)
        self.assertIn("F1 == F2", text)
        self.assertNotIn("F1 == F2", rl.located_analysis(2, SQUARE_ONLY, probe, reveal_flow=False,
                                                         all_shapes=False))

    def test_out_of_bound_on_the_wrong_axis_is_explained(self):
        msg = ("Out-of-bound access for tensor `unnamed` on dimension 1: index range [3, 3] exceed "
               "dimension size of 3.")
        text = rl._oob_analysis(msg, (3, 4))
        self.assertIn("NOT a tile-size limit", text)
        self.assertIn("F1", text)
        self.assertEqual(rl._oob_analysis("raised TypeError", (3, 4)), "")
        # the single-index form seen in the real run, on the partition axis
        real = "Out-of-bound access for tensor `unnamed` on dimension 0: index 3 exceed dimension size of 3."
        text = rl._oob_analysis(real, (3, 4))
        self.assertIn("F1", text)
        self.assertIn("PARTITION axis", text)

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


# ============================================================================ v6 additions

# The v5 failure: a fixed tile used on the 32x12 input. The stand-in's dma_copy asserts the same
# message the real simulator printed ("same number of elements, got src=384, dst=...").
HARDCODED_TILE = REFERENCE.replace(
    "in_tile = nl.ndarray(in_tensor.shape, dtype=in_tensor.dtype, buffer=nl.sbuf)",
    "in_tile = nl.ndarray((128, 128), dtype=in_tensor.dtype, buffer=nl.sbuf)")
SHAPES2 = [(32, 12), (128, 64), (64, 128), (8, 35)]

# Correct on every checker shape, wrong when the first extent is 1 (a held-out shape).
OVERFIT = REFERENCE.replace("for i_f1 in nl.affine_range(sz_f1):",
                            "for i_f1 in nl.affine_range(sz_f1 if sz_f1 > 1 else 0):")


class TestHardcodedLint(unittest.TestCase):
    def test_fixed_tile_is_flagged_with_the_line_and_the_smaller_shapes(self):
        found = kernel_lint.lint_hardcoded_dims(HARDCODED_TILE, SHAPES2)
        self.assertEqual(len(found), 2)                       # both dimensions are literals
        self.assertTrue(all(f["severity"] == "likely" for f in found))
        text = kernel_lint.format_hardcoded(found)
        self.assertIn("HARD-CODED SIZE", text)
        self.assertIn("(32, 12)", text)
        self.assertIn("in_tile = nl.ndarray((128, 128)", text)

    def test_reference_is_clean(self):
        self.assertEqual(kernel_lint.lint_hardcoded_dims(REFERENCE, SHAPES2), [])

    def test_sizes_derived_from_the_tensor_are_not_flagged(self):
        code = ("import nki\\nimport nki.language as nl\\n@nki.jit\\ndef k(x):\\n"
                "    P, F = x.shape\\n    n = min(128, P)\\n    half = F // 2\\n"
                "    t = nl.ndarray((n, half * 2), dtype=x.dtype, buffer=nl.sbuf)\\n"
                "    return t\\n").replace("\\n", "\n")
        self.assertEqual(kernel_lint.lint_hardcoded_dims(code, SHAPES2), [])

    def test_named_constants_are_followed(self):
        code = ("import nki\nimport nki.language as nl\nTILE = 128\n@nki.jit\ndef k(x):\n"
                "    t = nl.ndarray((TILE, TILE * 2), dtype=x.dtype, buffer=nl.sbuf)\n    return t\n")
        found = kernel_lint.lint_hardcoded_dims(code, SHAPES2)
        self.assertEqual({f["value"] for f in found}, {128, 256})

    def test_hbm_outputs_and_literal_one_are_ignored(self):
        code = ("import nki\nimport nki.language as nl\n@nki.jit\ndef k(x):\n"
                "    o = nl.ndarray((128, 512), dtype=x.dtype, buffer=nl.shared_hbm)\n"
                "    t = nl.ndarray((x.shape[0], 1), dtype=x.dtype, buffer=nl.sbuf)\n    return o\n")
        self.assertEqual(kernel_lint.lint_hardcoded_dims(code, SHAPES2), [])

    def test_literal_slice_bound_on_a_parameter_is_flagged(self):
        code = ("import nki\nimport nki.isa as nisa\nimport nki.language as nl\n@nki.jit\n"
                "def k(x):\n    t = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.sbuf)\n"
                "    nisa.dma_copy(dst=t, src=x[0:128, 0:512])\n    return t\n")
        found = kernel_lint.lint_hardcoded_dims(code, SHAPES2)
        self.assertTrue(any(f["kind"] == "slice" and f["value"] == 128 for f in found))

    def test_unparseable_code_returns_nothing(self):
        self.assertEqual(kernel_lint.lint_hardcoded_dims("def (", SHAPES2), [])


class TestTileFeedback(unittest.TestCase):
    def test_misleading_128x512_advice_is_replaced(self):
        raw = base_agent.enrich("raised AssertionError: dma_copy requires src and dst to have the "
                                "same number of elements, got src=384, dst=512")
        self.assertIn("128x512", raw)                         # the advice v5 sent the model
        fixed = rl.rewrite_tile_feedback(raw, SHAPES2)
        self.assertNotIn("128x512", fixed)
        self.assertIn("x.shape", fixed)
        self.assertIn("got src=384, dst=512", fixed)
        self.assertIn("(32, 12)", fixed)

    def test_other_feedback_is_untouched(self):
        self.assertEqual(rl.rewrite_tile_feedback("NUMERICAL MISMATCH", SHAPES2), "NUMERICAL MISMATCH")

    def test_level_input_shapes_come_from_the_checker(self):
        self.assertEqual(rl.level_input_shapes(2), SHAPES2)


class TestDedup(unittest.TestCase):
    def test_duplicates_do_not_dominate_the_group(self):
        rewards, fps = [0.375, 0.375, 0.375, 1.0], ["a", "a", "a", "b"]
        old_raw, _ = rl.group_advantages(rewards)
        raw, norm, dup = rl.dedup_advantages(rewards, fps)
        self.assertEqual(dup, [False, True, True, False])
        self.assertAlmostEqual(raw[3], 0.3125)                # (1.0 - 0.6875), over 2 kernels
        self.assertNotAlmostEqual(old_raw[3], raw[3])           # v5 weighted the duplicate 3x
        self.assertAlmostEqual(raw[0], raw[1])                # duplicates inherit
        self.assertAlmostEqual(norm[0], -1.0)
        self.assertAlmostEqual(norm[3], 1.0)

    def test_all_identical_is_zero_signal(self):
        raw, norm, dup = rl.dedup_advantages([0.3] * 4, ["x"] * 4)
        self.assertEqual((raw, norm, dup), ([0.0] * 4, [0.0] * 4, [False, True, True, True]))

    def test_empty_code_counts_as_one_kernel(self):
        _, _, dup = rl.dedup_advantages([0.0, 0.0], ["", ""])
        self.assertEqual(dup, [False, True])


class TestEscalation(unittest.TestCase):
    def test_restarts_climb_the_tiers_but_stop_before_algo(self):
        t = rl.effective_tier
        self.assertEqual([t("none", e) for e in range(4)], ["none", "api", "shape", "shape"])
        self.assertEqual([t("api", e) for e in range(3)], ["api", "shape", "shape"])
        self.assertEqual(t("shape", 5), "shape")
        self.assertEqual(t("api", 5, escalate_algo=True), "algo")
        self.assertEqual(t("none", 3, escalate=False), "none")
        self.assertEqual(t("algo", 0), "algo")

    def test_shape_card_is_in_the_shape_tier_only(self):
        for tier, want in (("none", False), ("api", False), ("shape", True), ("algo", True)):
            prompt = rl.build_prompt(Args(hints=tier), 2, "direct")
            self.assertEqual("SHAPES: the checker runs" in prompt, want, tier)
        self.assertIn("LEVEL 2 TRANSPOSE NOTE", rl.build_prompt(Args(hints="algo"), 2, "direct"))
        self.assertNotIn("LEVEL 2 TRANSPOSE NOTE", rl.build_prompt(Args(hints="shape"), 2, "direct"))


class TestProbeDetails(unittest.TestCase):
    def test_every_shape_is_reported_and_progress_is_graded(self):
        crash = rl.probe_kernel(2, HARDCODED_TILE)
        self.assertEqual(crash["kind"], "exception")
        self.assertEqual(len(crash["details"]), 4)
        self.assertTrue(all(d["status"] == "raised" for d in crash["details"]))
        wrong = rl.probe_kernel(2, NO_TRANSPOSE)
        square = rl.probe_kernel(2, SQUARE_ONLY)
        ok = rl.probe_kernel(2, REFERENCE)
        self.assertEqual(rl.probe_progress(crash), 0.0)
        self.assertEqual(rl.probe_progress(ok), 1.0)
        self.assertGreater(rl.probe_progress(wrong), rl.probe_progress(crash))
        self.assertGreater(rl.probe_progress(square), rl.probe_progress(wrong))   # 1 shape ok > all wrong
        self.assertLess(rl.probe_progress(square), 1.0)
        self.assertIsNone(rl.probe_progress({"kind": "unavailable"}))

    def test_table_names_each_shape(self):
        table = rl.shape_table(rl.probe_kernel(2, SQUARE_ONLY))
        self.assertIn("PER-SHAPE RESULT", table)
        self.assertIn("shape=(128, 64) as 8x8: passes", table)
        self.assertIn("shape=(32, 12) as 3x4: RAISES", table)
        wrong = rl.shape_table(rl.probe_kernel(2, NO_TRANSPOSE))
        self.assertIn("shape=(32, 12) as 3x4: wrong:", wrong)
        self.assertIn("of elements right", wrong)
        self.assertEqual(rl.shape_table(rl.probe_kernel(2, REFERENCE)), "")


class TestHoldout(unittest.TestCase):
    def test_reference_passes_every_holdout_shape(self):
        res = holdout_mod.check_holdout(2, REFERENCE)
        self.assertTrue(res["available"])
        self.assertTrue(res["ok"], res)
        self.assertEqual(res["total"], len(holdout_mod.HOLDOUT[2]))

    def test_overfit_kernel_is_caught_and_explained(self):
        self.assertEqual(rl.probe_kernel(2, OVERFIT)["kind"], "ok")     # passes the grader's shapes
        res = holdout_mod.check_holdout(2, OVERFIT)
        self.assertFalse(res["ok"])
        self.assertIn("HELD-OUT SHAPES", holdout_mod.format_holdout(res))
        self.assertEqual(holdout_mod.format_holdout({"available": True, "ok": True}), "")

    def test_holdout_shapes_differ_from_the_grading_shapes(self):
        for level, cases in holdout_mod.HOLDOUT.items():
            graded = [nkibench.label(c, level) for c in nkibench.LEVELS[level]["shapes"]]
            for c in cases:
                self.assertNotIn(nkibench.label(c, level), graded)


class ConstantModel:
    """Always the same kernel, whatever the prompt: the v5 'stuck' trace."""
    def __init__(self, code):
        self.code, self.calls = code, []

    def __call__(self, args, prompt, temperature, max_tokens=None, seed=None):
        self.calls.append(prompt)
        return fenced(self.code), "stop", 100, 100


class MarkerModel:
    """`bad` until the prompt contains `marker`, then `good`."""
    def __init__(self, bad, good, marker):
        self.bad, self.good, self.marker, self.calls = bad, good, marker, []

    def __call__(self, args, prompt, temperature, max_tokens=None, seed=None):
        self.calls.append(prompt)
        return fenced(self.good if self.marker in prompt else self.bad), "stop", 100, 100


class TestEpisodeV6(unittest.TestCase):
    def go(self, args, model, d, level=2):
        rl.ask_model = model
        run = make_run(args, d)
        out = run.args and rl.run_episode(run, level, 0, "t")
        run.log.close()
        run.groups.close()
        return out, run

    def test_hardcoded_tile_gets_a_located_hardcode_finding_and_the_fix_lands(self):
        with tempfile.TemporaryDirectory() as d:
            model = MarkerModel(HARDCODED_TILE, REFERENCE, "HARD-CODED SIZE")
            (best, verified), _ = self.go(Args(rounds=4, hints="shape"), model, d)
            rows = read_rows(d)
            self.assertTrue(verified)
            r0 = [r for r in rows if r["round"] == 0][0]
            self.assertEqual(r0["failure_category"], "tile_limits")
            self.assertTrue(r0["shape_lint"])
            self.assertNotIn("128x512", r0["feedback"])
            self.assertIn("PER-SHAPE RESULT", " ".join(r["feedback"] for r in rows
                                                       if r["round"] == 0))
            self.assertTrue(any("HARD-CODED SIZE" in p for p in model.calls))

    def test_progress_comes_from_the_probe(self):
        with tempfile.TemporaryDirectory() as d:
            self.go(Args(rounds=1, no_variants=True, samples=1),
                    ConstantModel(SQUARE_ONLY), d)
            row = read_rows(d)[0]
            # SQUARE_ONLY passes the one square shape and raises on the other three
            self.assertAlmostEqual(row["progress"], 0.25)
            self.assertEqual(len(row["probe_cases"]), 4)

    def test_stuck_episode_ends_after_the_restart_budget(self):
        with tempfile.TemporaryDirectory() as d:
            model = ConstantModel(HARDCODED_TILE)
            (best, verified), _ = self.go(Args(rounds=8, patience=0, max_restarts=2,
                                               no_variants=True, hints="none"), model, d)
            rows = read_rows(d)
            rounds = 1 + max(r["round"] for r in rows)
            self.assertFalse(verified)
            self.assertLessEqual(rounds, 3)                   # v5 ran 7+ rounds on this
            tiers = [r["hints_effective"] for r in rows if r["sample"] == 0]
            self.assertEqual(tiers[0], "none")
            self.assertEqual(tiers[1], "api")                 # escalated on the first restart
            self.assertEqual(tiers[-1], "shape")

    def test_restart_prompt_is_fresh_and_carries_the_analysis(self):
        with tempfile.TemporaryDirectory() as d:
            model = ConstantModel(HARDCODED_TILE)
            self.go(Args(rounds=3, patience=0, no_variants=True, samples=2, hints="api"), model, d)
            restart = model.calls[2]                          # round 1, sample 0
            self.assertIn("different structure", restart)
            self.assertIn("Your previous reply failed", restart)
            self.assertIn("CHECKER ANALYSIS", restart)
            self.assertIn("SHAPES: the checker runs", restart)   # api -> shape on the restart

    def test_plain_mode_restarts_with_a_fresh_first_prompt(self):
        with tempfile.TemporaryDirectory() as d:
            model = ConstantModel(HARDCODED_TILE)
            self.go(Args(mode="plain", rounds=4, patience=0, max_restarts=0, samples=1,
                         hints="none"), model, d)
            first = base_agent.first_prompt(2, 0)
            self.assertEqual(model.calls[0], first)
            self.assertIn("is not right yet", model.calls[1])  # a repair, before it is stuck
            self.assertEqual(model.calls[2], first)            # v5 re-sent the repair prompt here

    def test_plain_restart_can_be_switched_off(self):
        with tempfile.TemporaryDirectory() as d:
            model = ConstantModel(HARDCODED_TILE)
            self.go(Args(mode="plain", rounds=3, patience=0, max_restarts=0, samples=1,
                         plain_restart=False), model, d)
            self.assertIn("is not right yet", model.calls[2])

    def test_duplicates_are_flagged_not_exported_and_do_not_update_the_policy(self):
        with tempfile.TemporaryDirectory() as d:
            _, run = self.go(Args(rounds=1, no_variants=True, samples=4),
                             ConstantModel(HARDCODED_TILE), d)
            rows = read_rows(d)
            self.assertEqual([r["duplicate"] for r in rows], [False, True, True, True])
            with open(os.path.join(d, "groups.jsonl"), encoding="utf-8") as f:
                g = [json.loads(x) for x in f]
            self.assertEqual(len(g[0]["completions"]), 1)
            pulls = sum(v[0] for k, v in run.strategy.stats.items() if k.startswith("*||"))
            self.assertEqual(pulls, 1)

    def test_analysis_can_be_switched_off_for_an_ablation(self):
        with tempfile.TemporaryDirectory() as d:
            self.go(Args(rounds=1, no_variants=True, samples=1, no_analysis=True),
                    ConstantModel(HARDCODED_TILE), d)
            row = read_rows(d)[0]
            self.assertEqual((row["lint"], row["shape_lint"], row["probe_cases"]), ([], [], None))
            self.assertNotIn("HARD-CODED", row["feedback"])
            self.assertFalse(row["config"]["analysis"])

    def test_holdout_is_reported_and_can_be_required(self):
        with tempfile.TemporaryDirectory() as d:                    # report: still verified
            (best, verified), _ = self.go(Args(rounds=2, no_variants=True, samples=1,
                                               holdout="report"), ConstantModel(OVERFIT), d)
            self.assertTrue(verified)
            row = read_rows(d)[0]
            self.assertFalse(row["holdout"]["ok"])
        with tempfile.TemporaryDirectory() as d:                    # require: a failure, then repaired
            model = MarkerModel(OVERFIT, REFERENCE, "HELD-OUT SHAPES")
            (best, verified), _ = self.go(Args(rounds=3, no_variants=True, samples=1,
                                               holdout="require"), model, d)
            rows = read_rows(d)
            self.assertEqual(rows[0]["failure_category"], "holdout")
            self.assertLess(rows[0]["kernel_reward"], 0.999)
            self.assertTrue(verified)
            self.assertEqual(rows[-1]["kernel_reward"], 1.0)

    def test_independent_episodes_reset_the_policy(self):
        with tempfile.TemporaryDirectory() as d:
            rl.ask_model = ConstantModel(HARDCODED_TILE)
            run = make_run(Args(rounds=1, no_variants=True, samples=1,
                                independent_episodes=True), d)
            rl.run_episode(run, 2, 0, "t")
            first = sum(v[0] for k, v in run.strategy.stats.items() if k.startswith("*||"))
            rl.run_episode(run, 2, 1, "t")
            second = sum(v[0] for k, v in run.strategy.stats.items() if k.startswith("*||"))
            run.log.close()
            run.groups.close()
            self.assertEqual((first, second), (1, 1))


class TestSummarizer(unittest.TestCase):
    @staticmethod
    def rows(mode, hints, outcomes, level=2, **cfg):
        """outcomes: per episode a list of per-round lists of per-sample kernel rewards."""
        out = []
        for ep, rounds in enumerate(outcomes):
            for rnd, samples in enumerate(rounds):
                for i, k in enumerate(samples):
                    out.append({"_file": f"{mode}.jsonl", "run_id": "r", "level": level, "episode": ep,
                                "round": rnd, "sample": i, "kernel_reward": k, "mode": mode,
                                "hints": hints, "config": cfg or None, "rl_reward": k,
                                "progress": 0.0})
        return out

    def test_pass_at_k(self):
        self.assertAlmostEqual(summ.pass_at_k(4, 1, 1), 0.25)
        self.assertAlmostEqual(summ.pass_at_k(4, 1, 4), 1.0)
        self.assertAlmostEqual(summ.pass_at_k(4, 0, 4), 0.0)
        self.assertAlmostEqual(summ.pass_at_k(4, 2, 2), 1 - 1 / 6)

    def test_wilson_is_wide_for_five_episodes(self):
        lo, hi = summ.wilson(5, 5)
        self.assertGreater(lo, 0.5)
        self.assertLess(lo, 0.6)                              # 5/5 does not mean 100%
        lo3, hi3 = summ.wilson(3, 5)
        self.assertLess(lo3, 0.25)
        self.assertGreater(hi3, 0.8)                          # 3/5 and 5/5 overlap heavily
        self.assertLess(lo3, lo)
        self.assertTrue(all(map(lambda v: v != v, summ.wilson(0, 0))))

    def test_arm_table_reports_per_sample_rates_not_best_of_n(self):
        # Plain: 5 episodes, round 0 solves 1 of 4 samples in 3 of them; the other two never solve.
        plain = self.rows("plain", "none", [[[1, .3, .3, .3]]] * 3 + [[[.3] * 4, [.5] * 4]] * 2)
        lines, stats = summ.arm_table(plain)
        st = stats[("plain/none", 2)]
        self.assertEqual((st["episodes"], st["solved"]), (5, 3))
        self.assertAlmostEqual(st["pass1"], 3 * 0.25 / 5)
        self.assertAlmostEqual(st["pass4"], 3 / 5)

    def test_arm_labels_show_ablations_and_confounds_warn(self):
        a = self.rows("plain", "none", [[[1, 0, 0, 0]]])
        b = self.rows("reflect", "api", [[[1, 1, 1, 0]]], variants=True, analysis=True)
        c = self.rows("reflect", "api", [[[1, 0, 0, 0]]], variants=True, analysis=False)
        self.assertEqual(summ.arm_label(c[0]), "reflect/api+no-analysis")
        notes = "\n".join(summ.confound_report(a + b + c))
        self.assertIn("WARNING: 'plain' differs", notes)
        self.assertIn("NO analysis", notes)

    def test_main_runs_on_a_v6_style_log(self):
        rows = self.rows("reflect", "shape", [[[.3, .3, 1, .3], [1, 1, 1, 1]]] * 3)
        for r in rows:
            r.pop("_file")
            r.update({"action": "reflect", "temperature": .6, "failure_category": "verified"
                      if r["kernel_reward"] >= .999 else "tile_limits", "repeated": False,
                      "group_distinct": 3, "advantage": 0.1, "duplicate": False,
                      "grade_cached": False, "trace_enabled": False})
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(json.dumps(r) for r in rows) + "\n")
            import io
            from contextlib import redirect_stdout
            buf = io.StringIO()
            old = sys.argv
            sys.argv = ["summarize", path]
            try:
                with redirect_stdout(buf):
                    summ.main()
            finally:
                sys.argv = old
            text = buf.getvalue()
            self.assertIn("Arm comparison", text)
            self.assertIn("reflect/shape", text)
            self.assertIn("trace_reward: not measured", text)
            self.assertIn("verified in 3/3 episodes", text)


class TestMatrix(unittest.TestCase):
    def test_arm_parsing(self):
        self.assertEqual(run_matrix.parse_arm("plain/none"), ("plain", "none", []))
        self.assertEqual(run_matrix.parse_arm("reflect/api+no-analysis+no-variants"),
                         ("reflect", "api", ["--no-analysis", "--no-variants"]))
        for bad in ("reflect", "bogus/api", "reflect/zzz", "reflect/api+nope"):
            with self.assertRaises(SystemExit):
                run_matrix.parse_arm(bad)

    def test_command_is_independent_and_seeded_per_chunk(self):
        a = run_matrix.argparse.Namespace(rounds=4, samples=4, patience=3, seed=7, out="o",
                                          export_groups=False)
        c0 = run_matrix.build_command(a, "reflect/shape", 2, 5, 0, ["--context", "8192"])
        c1 = run_matrix.build_command(a, "reflect/shape", 2, 5, 1, [])
        self.assertIn("--independent-episodes", c0)
        self.assertEqual(c0[c0.index("--seed") + 1], "7")
        self.assertEqual(c1[c1.index("--seed") + 1], "1007")
        self.assertEqual(c0[-2:], ["--context", "8192"])

    def test_resume_counts_distinct_episodes(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "l.jsonl")
            with open(path, "w") as f:
                for run_id, ep, rnd in (("a", 0, 0), ("a", 0, 1), ("a", 1, 0), ("b", 0, 0)):
                    f.write(json.dumps({"run_id": run_id, "episode": ep, "round": rnd}) + "\n")
            self.assertEqual(run_matrix.episodes_done(path), 3)
            self.assertEqual(run_matrix.episodes_done(os.path.join(d, "missing.jsonl")), 0)


if __name__ == "__main__":
    print(f"nki: {'NumPy stand-in' if USING_STUB else 'real SDK'}")
    unittest.main(verbosity=2)