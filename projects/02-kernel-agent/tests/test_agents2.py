"""Tests for agents2 that need neither nki nor a model: a fake nki, a fake model, fake checks.

    python tests/test_agents2.py          (or pytest tests/test_agents2.py)

The real thing is tested in a seat pod: `python agent2.py --offline --all` (every role, real checks)
and `python agent2.py --classify <old attempts.jsonl>` (lint must have no false positives).
"""

import os
import sys
import types

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
os.chdir(HERE)


# ---------------------------------------------------------------- a fake nki, installed first

def _fake_nki():
    if "nki" in sys.modules and getattr(sys.modules["nki"], "__version__", "") != "0.0-fake":
        return                                    # a real nki is installed: use it
    nki = types.ModuleType("nki")
    nki.__version__ = "0.0-fake"
    isa = types.ModuleType("nki.isa")
    lang = types.ModuleType("nki.language")

    def dma_copy(dst, src, dge_mode=None):
        """Copy data between HBM and SBUF. The shapes must match."""

    def tensor_scalar(dst, data, op0, operand0, reverse0=False):
        """Apply an operator with a scalar to every element of a tile."""

    def nc_matmul(dst, stationary, moving):
        """Multiply two SBUF tiles into a PSUM tile."""

    def ndarray(shape, dtype=None, buffer=None):
        """Allocate a tensor."""

    def sum(x, axis, keepdims=False):  # noqa: A001
        """Sum over axes."""

    def multiply(x, y):
        """Multiply elementwise."""

    for f in (dma_copy, tensor_scalar, nc_matmul):
        setattr(isa, f.__name__, f)
    for f in (ndarray, sum, multiply):
        setattr(lang, f.__name__, f)
    lang.sbuf, lang.psum, lang.shared_hbm, lang.float32 = object(), object(), object(), "float32"
    nki.jit = lambda f: f
    nki.isa, nki.language = isa, lang
    tensor = types.ModuleType("nki.language.tensor")

    class NkiTensor:
        def ap(self, pattern, offset=None):
            """Low-level access pattern override."""

        def permute(self, dims):
            """Reorder tensor dimensions."""

        def reshape(self, shape):
            """Reshape the tensor to a new shape without copying data."""
    tensor.NkiTensor = NkiTensor
    lang.tensor = tensor
    sys.modules.update({"nki": nki, "nki.isa": isa, "nki.language": lang,
                        "nki.language.tensor": tensor})


_fake_nki()

import agent                                          # noqa: E402
from agents2 import errors, lint                      # noqa: E402
from agents2.config import Config, Role, ROLE_DEFAULTS  # noqa: E402
from agents2.debugger import parse_change             # noqa: E402
from agents2.llm import PromptTooLong, Section, pack  # noqa: E402
from agents2.manager import Manager                   # noqa: E402
from agents2.planner import parse_plan                # noqa: E402
from agents2.retriever import Retriever               # noqa: E402

HEADER = "import nki\nimport nki.isa as nisa\nimport nki.language as nl\n\n"


# ---------------------------------------------------------------- lint

def test_lint_names_and_kwargs():
    src = HEADER + ("@nki.jit\ndef k(a):\n"
                    "    t = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.sbuf)\n"
                    "    nisa.multiply(dst=t, data=t)\n"
                    "    nisa.nc_matmul(dst=t, stationary=t, moving=t, transpose_moving=True)\n"
                    "    nisa.tensor_scalar(dst=t, op0=nl.multiply, operand0=0.5)\n"
                    "    return t\n")
    found = lint.lint(src)
    kinds = [(f["kind"], f["line"]) for f in found]
    assert ("NAME", 8) in kinds, found
    assert ("KWARG", 9) in kinds, found
    assert ("KWARG", 10) in kinds, found                     # missing data=
    assert "module 'nki.isa' has no attribute 'multiply'" in found[0]["msg"]


def test_lint_clean_and_conservative():
    src = HEADER + ("@nki.jit\ndef k(a, pool_size):\n"
                    "    out = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.shared_hbm)\n"
                    "    t = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.sbuf)\n"
                    "    nisa.dma_copy(dst=t, src=a)\n"
                    "    s = nl.sum(t, axis=[1], keepdims=True)\n"
                    "    x = a.whatever.deep(1, foo=2)\n"
                    "    nisa.dma_copy(dst=out, src=t)\n"
                    "    return out\n")
    assert lint.lint(src) == []
    assert lint.lint("def broken(:\n") == []


def test_lint_alias_and_modules():
    src = "import nki\nfrom nki.isa import dma_copy\nimport nki.nl\n"
    assert [f["kind"] for f in lint.lint(src)] == ["NAME"]
    assert sorted(f["kind"] for f in lint.lint(src, traffic_level=True)) == ["ALIAS", "NAME"]


# ---------------------------------------------------------------- errors

def test_runtime_kinds_on_recorded_messages():
    cases = {
        "AttributeError: module 'nki.isa' has no attribute 'multiply'": "NAME",
        "AssertionError: Out-of-bound access for tensor `unnamed` on dimension 1: index range [0, 7] "
        "exceed dimension size of 4": "BOUNDS",
        "AssertionError: dma_copy requires src and dst to have the same number of elements, got src=4, "
        "dst=16384": "DMA_SHAPE",
        "ValueError: shape mismatch: value array of shape (4,) could not be broadcast": "DMA_SHAPE",
        "AssertionError: SBUF and PSUM tensors must have at least 2 dimensions": "TILE_RANK",
        "TypeError: nc_matmul() got an unexpected keyword argument 'transpose_moving'": "KWARG",
        "AssertionError: dst must be in ['psum'], got sbuf": "MEMSPACE",
        "AssertionError: dma_copy dst partition dimension 256 exceeds maximum 128": "PARTITION",
        "AssertionError: Matmul contraction dimension mismatch: stationary[0]=128 != moving[0]=64":
            "MATMUL_SHAPE",
        "AssertionError: tensor_reduce axis must be the last contiguous dim(s) of the tile": "REDUCE_AXIS",
        "AssertionError: dma_transpose axes must be one of [(1, 0), (2, 1, 0), (3, 1, 2, 0)], got (0, 2)":
            "TRANSPOSE",
        "AssertionError: dma_copy priority is only supported for NeuronCore-v4 or newer": "UNSUPPORTED",
        "TypeError: '_TileSize' object is not callable": "TILE_SIZE",
        "AttributeError: '_TileSize' object has no attribute 'value'": "TILE_SIZE",
        "SomethingNew: never seen": "OTHER",
    }
    for msg, want in cases.items():
        assert errors.classify_runtime(msg) == want, (msg, errors.classify_runtime(msg))


def test_feedback_kinds():
    assert errors.classify_feedback("No code came back. Reply...") == "EMPTY"
    assert errors.classify_feedback("The code does not parse: x") == "PARSE"
    assert errors.classify_feedback("Rule violations, which ...") == "RULES"
    assert errors.classify_feedback("0 of 4 shapes passed. On K=1: raised AttributeError: module "
                                    "'nki.isa' has no attribute 'x'") == "NAME"
    assert errors.classify_feedback("2 of 4 shapes passed. On K=1: OUTPUT IS 98% ZEROS while") == "ZEROS"
    assert errors.classify_feedback("Correct on every shape.") == "CORRECT"


def test_rule_change_name_suggests_other_module():
    retr = Retriever()
    rule = errors.rule_change(dict(kind="NAME", error="module 'nki.isa' has no attribute 'multiply'"),
                              retr, 1, agent.enrich, agent.fragment_note)
    assert rule and rule["names"][0] == "nl.multiply", rule
    assert "scale" in rule["example"] or "tensor_scalar" in rule["example"]
    assert errors.rule_change(dict(kind="VALUES", error="NUMERICAL MISMATCH"), retr, 1, agent.enrich,
                              agent.fragment_note) is None


# ---------------------------------------------------------------- retriever

def test_cards_from_the_cheatsheet_and_ours():
    from agents2.retriever import load_cards
    cards = load_cards()
    assert "core" in cards and "nisa.dma_copy" in cards, sorted(cards)
    assert "tile.permute" in cards and cards["tile.permute"][2] == {2}
    retr = Retriever()
    assert "op0=nl.multiply" in retr.card("nisa.tensor_scalar", 1)      # the checked card wins
    assert retr.card("tile.permute", 1) and retr.card("tile.permute", 2) == ""
    assert "t.permute" in retr.api_map(1) and "t.permute" not in retr.api_map(2)
    assert "t.ap" in retr.api_map(2) and "topics: rules" in retr.api_map(1)
    assert "views" not in retr.api_map(1)                                 # an index, not documentation
    assert retr.card("tile.ap", 1).startswith("t.ap(pattern")             # introspected, self dropped
    if "matmul-k-loop" in cards:
        assert "SAME psum" not in retr.cards(["nisa.nc_matmul"], 4)        # withheld at 3-7
        assert "SAME psum" in retr.cards(["nisa.nc_matmul"], 8)
    assert retr.core(1).startswith("NKI rules")
    assert Retriever(cards=False).card("nisa.tensor_scalar", 1).startswith("nisa.tensor_scalar(")


def test_rules_never_restate_or_name_a_withheld_card():
    retr = Retriever()
    t = dict(kind="TRANSPOSE", error="AssertionError: dma_transpose axes must be one of [(1, 0)], got (0, 2)")
    assert errors.rule_change(t, retr, 2, agent.enrich, agent.fragment_note) is None   # model path
    assert "(1, 0)" in errors.rule_change(t, retr, 1, agent.enrich, agent.fragment_note)["change"]
    r = dict(kind="REDUCE_AXIS", error="AssertionError: tensor_reduce axis must be the last dim(s)")
    at2 = errors.rule_change(r, retr, 2, agent.enrich, agent.fragment_note)
    at1 = errors.rule_change(r, retr, 1, agent.enrich, agent.fragment_note)
    assert "tile.permute" not in at2["names"] and "view" not in at2["change"], at2
    assert "tile.permute" in at1["names"] and "view" in at1["change"], at1


def test_alias_is_its_own_kind_with_a_rule():
    msg = ("dma_copy is imported by name, so the traffic check cannot count its bytes. Import the module "
           "(import nki.isa as nisa) and call nisa.dma_copy.")
    rule = errors.rule_change(dict(kind="ALIAS", error=msg), Retriever(), 5, agent.enrich,
                              agent.fragment_note)
    assert rule["change"] == msg and rule["names"] == ["nisa.dma_copy"]


def test_tile_methods_in_close_names_and_kwarg_rule():
    retr = Retriever()
    assert "tile.permute" in retr.close("nl.permute")
    rule = errors.rule_change(dict(kind="KWARG", error="TypeError: NkiTensor.reshape() takes 2 positional "
                                   "arguments but 6 were given"), retr, 1, agent.enrich, agent.fragment_note)
    assert rule["names"] == ["tile.reshape"], rule


def test_plan_names_tile_methods():
    p = parse_plan("APPROACH: view windows\nCALLS: t.permute, .ap, nl.sum, tile.reshape\nSTEPS: 1 x")
    assert p.calls == ["tile.permute", "tile.ap", "nl.sum", "tile.reshape"], p.calls
    assert Retriever().short("t.permute") == "tile.permute" and Retriever().exists("t.permute")


# ---------------------------------------------------------------- lookup

def test_parse_lookup():
    from agents2.lookup import parse_lookup
    assert parse_lookup("LOOKUP: nl.sum, t.permute and rules") == ["nl.sum", "t.permute", "rules"]
    assert parse_lookup("**LOOKUP:** `nisa.dma_copy`") == ["nisa.dma_copy"]
    assert parse_lookup("APPROACH: x\nLOOKUP: nl.sum") == []                # an answer, not a lookup
    assert parse_lookup("```python\nLOOKUP: x\n```") == []
    assert parse_lookup("no lookup here") == []


def test_lookup_round_trip_and_cap():
    from agents2 import lookup
    from agents2.ledger import Ledger
    cfg = _cfg()
    replies = iter(["LOOKUP: nisa.tensor_scalar, nl.nonexistent", "LOOKUP: rules", "APPROACH: done"])
    seen = []

    class L:
        def chat(self, role, sections, tags=None, **kw):
            text, _ = pack(sections, 10 ** 6, len)
            seen.append(text)
            return next(replies), dict(role=role, error=None)
    ev = NullEvents()
    text, meta, pulled = lookup.ask(L(), "planner", [Section("task", "TASK", required=True),
                                                     Section("format", "FORMAT", required=True)],
                                    Retriever(), 1, 2, Ledger(1, 0), ev, tags=dict(level=1))
    assert text == "APPROACH: done" and pulled == ["nisa.tensor_scalar", "nl.nonexistent", "rules"]
    assert "op0=nl.multiply" in seen[1] and "no such name" in seen[1]      # the docs came back
    assert "NKI rules" in seen[2] and "Reply in one of two ways" not in seen[2]  # last round: answer
    assert "No more lookups" in seen[2] and seen[2].rstrip().endswith("FORMAT")
    assert "Reply in one of two ways" in seen[0] and seen[0].rstrip().endswith("FORMAT")  # one format
    assert [e for e, f in ev.rows] == ["lookup", "lookup"]
    assert "Documentation you looked up" not in seen[0]                   # nothing pushed up front


def _ask_scripted(replies, rounds):
    from agents2 import lookup
    from agents2.ledger import Ledger
    replies, seen = iter(replies), []

    class L:
        def chat(self, role, sections, tags=None, **kw):
            seen.append(pack(sections, 10 ** 6, len)[0])
            return next(replies), dict(role=role, error=None)
    out = lookup.ask(L(), "debugger", [Section("task", "TASK", required=True),
                                       Section("format", "FORMAT", required=True)],
                     Retriever(), 1, rounds, Ledger(1, 0), NullEvents())
    return out, seen


def test_lookup_after_answer_now_gets_one_more_call():
    # asked again in the last round: its docs and one forced answer
    (text, _, pulled), seen = _ask_scripted(["LOOKUP: nl.sum", "LOOKUP: nisa.dma_copy", "CHANGE: x"], 1)
    assert text == "CHANGE: x" and len(seen) == 3 and "nisa.dma_copy" in pulled
    assert "No more lookups" in seen[1] and "No more lookups" in seen[2]
    # still asking after that: the caller gets the lookup back, and no fourth call is made
    (text, _, _), seen = _ask_scripted(["LOOKUP: nl.sum", "LOOKUP: nisa.dma_copy", "LOOKUP: rules"], 1)
    assert text == "LOOKUP: rules" and len(seen) == 3
    # rounds=0 (--no-lookup): no invitation, and a lookup is not answered
    (text, _, pulled), seen = _ask_scripted(["LOOKUP: nl.sum"], 0)
    assert len(seen) == 1 and pulled == [] and "LOOKUP" not in seen[0] and "No more" not in seen[0]


def test_debugger_never_passes_a_lookup_on_as_its_change():
    from agents2.debugger import Debugger
    from agents2.ledger import Ledger, Plan

    class L:
        def chat(self, role, sections, tags=None, **kw):
            return "LOOKUP: nl.sum", dict(role=role, error=None)
    d = Debugger(L(), Retriever(), _cfg(), agent.enrich, lambda e, l: "", None)
    out = d.debug(1, Plan(approach="x", calls=[]), "x = 1\n",
                  dict(kind="VALUES", error="NUMERICAL MISMATCH: worst 3"), [], Ledger(1, 0), 3)
    assert out["path"] == "fallback" and "LOOKUP" not in out["change"], out


# ---------------------------------------------------------------- packer and parsers

def test_pack_drops_lowest_priority_first():
    count = lambda t: len(t)
    secs = [Section("a", "x" * 50, required=True), Section("b", "y" * 30, priority=1),
            Section("c", "z" * 30, priority=5), Section("d", "w" * 10, required=True)]
    text, rep = pack(secs, 100, count)
    assert rep["dropped"] == ["b"] and "z" * 30 in text and text.index("x") < text.index("w")
    try:
        pack([Section("a", "x" * 200, required=True)], 100, count)
        raise AssertionError("expected PromptTooLong")
    except PromptTooLong:
        pass


def test_parse_plan_variants():
    p = parse_plan("**APPROACH:** sum each window\n**CALLS:** `nl.sum`, nisa.dma_copy(), nki.isa.tensor_scalar"
                   "\nLAYOUT: channels on partitions\nSTEPS:\n1. load\n2. sum\n")
    assert p.approach == "sum each window"
    assert p.calls == ["nl.sum", "nisa.dma_copy", "nisa.tensor_scalar"], p.calls
    assert p.steps.startswith("1. load")
    q = parse_plan("I would just use nl.sum over windows.")
    assert q.approach.startswith("I would") and q.calls == ["nl.sum"]
    assert parse_plan("") is None


def test_parse_change():
    c = parse_change("CAUSE: wrong op\nLINE: 12\nCHANGE: use nl.multiply as op0")
    assert c == dict(cause="wrong op", change="use nl.multiply as op0", line=12)
    assert parse_change("**APPROACH WRONG:** matmul cannot pool")["approach_wrong"] == "matmul cannot pool"


# ---------------------------------------------------------------- the manager, end to end

class FakeLLM:
    """Scripted answers per role. coder_answers: a list of callables (prompt, mode) -> reply."""

    def __init__(self, cfg, plans, coder, debugger=None):
        self.cfg, self.plans, self.coder, self.debugger = cfg, list(plans), coder, debugger
        self.calls = []
        self.context = {}

    def chat(self, role, sections, tags=None, temperature=None, max_tokens=None):
        text, rep = pack(sections, self.cfg.roles[role].prompt_cap, lambda t: len(t) // 3 + 1)
        tags = tags or {}
        self.calls.append((role, tags.get("mode"), text))
        meta = dict(role=role, error=None, gen_seconds=0.0, completion_tokens=10, **tags)
        if role == "planner":
            return (self.plans.pop(0) if self.plans else "APPROACH: same\nCALLS: nl.sum"), meta
        if role == "coder":
            return self.coder(text, tags.get("mode")), meta
        return (self.debugger(text) if self.debugger else "CAUSE: x\nLINE: 1\nCHANGE: y"), meta


class FakeChecks:
    """A kernel's text decides its check: GOOD passes, BADNAME/VALUES/SHAPE fail as named."""

    def run(self, code, level):
        base = dict(parts=dict(parses=True, rules=True, runs=False, correct=False), passed=0, total=4,
                    correct=False, line=7, shape="s", corner=None, shapes=[], lint=[], warnings=[])
        if "GOOD" in code:
            base.update(kind="CORRECT", stage="correct", error=None, reward=1.0, correct=True,
                        parts=dict(parses=True, rules=True, runs=True, correct=True))
        elif "BADNAME" in code:
            base.update(kind="NAME", stage="lint", reward=0.3,
                        error="module 'nki.isa' has no attribute 'multiply'")
        elif "VALUES" in code:
            base.update(kind="VALUES", stage="values", reward=0.5, error="NUMERICAL MISMATCH: worst 3")
        else:
            base.update(kind="DMA_SHAPE", stage="run", reward=0.3,
                        error="AssertionError: dma_copy requires src and dst to have the same number "
                              "of elements, got src=4, dst=16")
        return base

    def close(self):
        pass


class NullEvents:
    def __init__(self):
        self.rows = []

    def write(self, event, **f):
        self.rows.append((event, f))

    def attempt(self, **f):
        self.rows.append(("attempt", f))


def _cfg(**kw):
    roles = {n: Role(name=n, base="http://x", model="m", **d) for n, d in ROLE_DEFAULTS.items()}
    return Config(roles=roles, **kw)


def _run(llm_factory, **kw):
    cfg = _cfg(**kw)
    llm = llm_factory(cfg)
    events = NullEvents()
    mgr = Manager(cfg, llm, FakeChecks(), Retriever(), events, agent, say=lambda m: None)
    return mgr.run_level(1), llm, events


def test_solves_after_a_rule_fix_without_debugger_model():
    code = lambda prompt, mode: "```python\n# BADNAME\n```" if mode == "write" else "```python\n# GOOD\n```"
    res, llm, ev = _run(lambda cfg: FakeLLM(cfg, ["APPROACH: a\nCALLS: nl.sum"], code), threads=1)
    assert res["solved"] and res["stop_reason"] == "solved", res
    roles = [c[0] for c in llm.calls]
    assert roles == ["planner", "coder", "coder"], roles          # the rule path made no model call
    apply_prompt = llm.calls[2][2]
    assert "nl.multiply" in apply_prompt and "BADNAME" in apply_prompt


def test_coder_prompts_example_kernel_and_index():
    code = lambda prompt, mode: "```python\n# BADNAME\n```" if mode == "write" else "```python\n# GOOD\n```"
    res, llm, ev = _run(lambda cfg: FakeLLM(cfg, ["APPROACH: a\nCALLS: nl.sum"], code), threads=1)
    write, apply_ = llm.calls[1][2], llm.calls[2][2]
    assert "def copy_kernel" in write and "buffer=nl.shared_hbm" in write    # the organisers' example
    assert "NKI names" in write and "NKI names" not in apply_                # no index in a repair
    assert "Reply in one of two ways" in apply_                             # it may still look up
    res, llm, ev = _run(lambda cfg: FakeLLM(cfg, ["APPROACH: a\nCALLS: nl.sum"], code), threads=1,
                        skeleton=False)
    assert "def copy_kernel" not in llm.calls[1][2]


def test_call_budget_stops_a_running_thread():
    n = [0]

    def code(prompt, mode):
        n[0] += 1
        return f"```python\n# SHAPE {n[0]}\n```"
    res, llm, ev = _run(lambda cfg: FakeLLM(cfg, ["APPROACH: a\nCALLS: nl.sum"], code), threads=1,
                        max_approaches=1, max_calls=3)
    ends = [f["reason"] for e, f in ev.rows if e == "thread_end"]
    assert ends == ["budget"] and res["stop_reason"] == "budget", (ends, res["stop_reason"])
    assert len(llm.calls) == 3                       # planner, write, one apply; then it stopped


def test_cycling_ends_threads_and_the_level_runs_out_of_approaches():
    n = [0]

    def code(prompt, mode):
        n[0] += 1
        return f"```python\n# SHAPE {n[0]}\n```"                   # always the same failure, new text
    plans = [f"APPROACH: plan {i}\nCALLS: nl.sum, nisa.op{i}" for i in range(10)]
    plans = [f"APPROACH: plan {i}\nCALLS: {c}" for i, c in
             enumerate(["nl.sum", "nisa.dma_copy", "nisa.tensor_scalar", "nl.multiply", "nl.ndarray"])]
    res, llm, ev = _run(lambda cfg: FakeLLM(cfg, plans, code), threads=2, max_approaches=3)
    assert not res["solved"] and res["stop_reason"] == "exhausted", res
    ends = [f["reason"] for e, f in ev.rows if e == "thread_end"]
    assert ends.count("cycling") == 3, ends
    assert len(res["approaches"]) == 3


def test_progress_counts_stages_not_only_reward():
    from agents2.manager import progress
    lint_fail = dict(reward=0.3, stage="lint", passed=0)
    run_fail = dict(reward=0.3, stage="run", passed=0)
    values = dict(reward=0.5, stage="values", passed=0)
    assert progress(run_fail) > progress(lint_fail)             # same 0.30, but further along
    assert progress(values) > progress(run_fail)
    assert progress(dict(reward=0.3, stage="lint")) == progress(lint_fail)


def test_echo_ends_the_thread():
    code = lambda prompt, mode: "```python\n# SHAPE same\n```"
    res, llm, ev = _run(lambda cfg: FakeLLM(cfg, ["APPROACH: a\nCALLS: nl.sum"], code), threads=1,
                        max_approaches=1)
    ends = [f["reason"] for e, f in ev.rows if e == "thread_end"]
    assert ends == ["echo"], ends
    # two echoes, one rescue from the debugger, two more echoes, then the thread ends
    assert sum(1 for e, f in ev.rows if e == "echo") == 4
    assert sum(1 for e, f in ev.rows if e == "change" and f.get("rescue")) == 1
    echo_prompt = [c for c in llm.calls if c[0] == "coder"][-1][2]
    assert "unchanged" in echo_prompt


def test_debugger_model_path_and_approach_wrong():
    n = [0]

    def code(prompt, mode):
        n[0] += 1
        return f"```python\n# VALUES {n[0]}\n```"
    dbg = lambda prompt: "APPROACH WRONG: matmul cannot pool"
    res, llm, ev = _run(lambda cfg: FakeLLM(cfg, ["APPROACH: a\nCALLS: nl.sum"], code, dbg), threads=1,
                        max_approaches=1)
    ends = [f["reason"] for e, f in ev.rows if e == "thread_end"]
    assert ends == ["approach_wrong"], ends
    # the first APPROACH WRONG came too early (1 check) and was turned into a fix
    assert [c[0] for c in llm.calls].count("debugger") == 2


def test_duplicate_plan_is_replanned():
    code = lambda prompt, mode: "```python\n# GOOD\n```"
    plans = ["APPROACH: sum each window\nCALLS: nl.sum",
             "APPROACH: sum each window again\nCALLS: nl.sum",
             "APPROACH: matmul with a ones vector\nCALLS: nl.sum"]
    cfg = _cfg(threads=1, max_approaches=2)
    llm = FakeLLM(cfg, plans, code)
    from agents2.ledger import Ledger, Plan
    led = Ledger(1, 0)
    led.register(Plan(approach="sum each window", calls=["nl.sum"]), 9)
    mgr = Manager(cfg, llm, FakeChecks(), Retriever(), NullEvents(), agent, say=lambda m: None)
    mgr._thread(led, __import__("threading").Event(), 0)
    # same calls but a different algorithm is a new approach; the near-copies are not
    assert [a.plan.approach for a in led.approaches] == ["sum each window",
                                                         "matmul with a ones vector"], led.approaches
    replans = [c[2] for c in llm.calls if c[0] == "planner"]
    assert len(replans) == 3 and "already trying" in replans[1] and "already trying" in replans[2]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"ok    {t.__name__}")
        except Exception as e:
            failed += 1
            import traceback
            print(f"FAIL  {t.__name__}: {e!r}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
