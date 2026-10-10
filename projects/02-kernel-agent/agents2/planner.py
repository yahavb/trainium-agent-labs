"""PLANNER: chooses the algorithm before any code exists.

On level 1, 100% of the old loop's attempts called nisa.nc_matmul for pooling, because the API card
talked about matmul, and every later round repaired that same wrong algorithm. A plan is cheap (about
120 output tokens) and is the one place threads are made to differ: two threads with the same plan are
one thread twice.
"""

import inspect
import re

from agents2.ledger import Plan
from agents2.llm import Section

FIELDS = ("APPROACH", "CALLS", "LAYOUT", "STEPS")
NAME_RE = re.compile(r"\b(?:nisa|nl|nki\.isa|nki\.language)\.\w+"
                     r"|(?<![\w.])(?:t|tile)?\.(?:ap|permute|reshape|rearrange|slice|select|broadcast|"
                     r"flatten_dims|expand_dim|squeeze_dim)\b")


def hardware_note(level):
    """The problem statement's own hardware line, as agent.py's first prompt has it, except that levels
    1-2 get no matmul limits: on seat-38, matmul text in a level-1 prompt was enough to make 12 of 84
    attempts call nc_matmul. Anything more is for the agents to look up (agents2/lookup.py)."""
    import nkibench
    s = f"Hardware limits: a tile's partition dimension is at most {nkibench.PMAX}."
    if level >= 3:
        s += (f" For matmul, the stationary free dimension is at most {nkibench.GEMM_STATIONARY_FMAX} and "
              f"the moving free dimension at most {nkibench.GEMM_MOVING_FMAX}.")
    return s + "\n\nImport nki, nki.language as nl, and nki.isa as nisa."


def task_text(spec, level):
    import nkibench
    shapes = "; ".join(nkibench.label(c, level) for c in spec["shapes"])
    args = ", ".join(inspect.signature(spec["ref"]).parameters)
    return (f"Operation: {spec['op']}\n"
            f"Entry point: `{spec['entry']}({args})`, decorated with @nki.jit.\n"
            f"It must compute exactly what this NumPy reference computes:\n\n"
            f"{inspect.getsource(spec['ref'])}\n"
            f"Test shapes: {shapes}")


def parse_plan(text):
    """Lenient: bold markers, lower case, extra prose and multi-line STEPS are all accepted."""
    clean = re.sub(r"[*`#]", "", text or "")
    found = {}
    for i, f in enumerate(FIELDS):
        nxt = "|".join(FIELDS[i + 1:]) or "$^"
        m = re.search(rf"(?im)^\s*{f}\s*:\s*(.*?)(?=^\s*(?:{nxt})\s*:|\Z)", clean, re.S)
        if m:
            found[f] = re.sub(r"\s+", " ", m.group(1)).strip()
    if not found.get("APPROACH"):
        first = re.sub(r"\s+", " ", clean).strip()
        if not first:
            return None
        found["APPROACH"] = first[:300]
    calls = []
    for n in NAME_RE.findall(found.get("CALLS", "") or clean):
        n = n.replace("nki.isa.", "nisa.").replace("nki.language.", "nl.")
        if not n.startswith(("nisa.", "nl.")):
            n = "tile." + n.rpartition(".")[2]
        if n not in calls:
            calls.append(n)
    return Plan(approach=found["APPROACH"], calls=calls, layout=found.get("LAYOUT", ""),
                steps=found.get("STEPS", ""), raw=text)


class Planner:
    def __init__(self, llm, retriever, cfg, events=None):
        self.llm, self.retriever, self.cfg, self.events = llm, retriever, cfg, events

    def plan(self, level, ledger, note=""):
        """One plan, or None if the request failed. `note` carries a correction (unknown names, or an
        approach another thread already has). The prompt is the problem statement and an index of
        names. The first call only asks which documentation the planner needs (cfg.docs_request); then
        it plans with that documentation, and may look up more within cfg.lookup["planner"] rounds in
        all. What it pulled travels with the plan to the coder."""
        import nkibench
        from agents2 import lookup
        spec = nkibench.LEVELS[level]
        tried = ledger.tried()
        sections = [
            Section("intro", "You are planning an AWS Neuron NKI kernel before anyone writes it. Do not "
                             "write code: choose the algorithm.", required=True),
            Section("task", task_text(spec, level), required=True),
            Section("hardware", hardware_note(level), required=True),
            Section("hint", f"Hint: {spec.get('notes')}" if self.cfg.hint and spec.get("notes") else "",
                    priority=5),
            lookup.index_section(self.retriever, level),
            Section("tried", ("These approaches are taken: tried already, or being tried by another "
                              "thread. Choose a different algorithm:\n"
                              + "\n".join(f"- {t}" for t in tried)) if tried else "",
                    priority=4),
            Section("note", note, required=bool(note)),
            Section("format", "Reply with exactly these four lines:\n"
                              "APPROACH: <one sentence: the operation that does the main work, and on "
                              "what view or layout of the data>\n"
                              "CALLS: <the nisa and nl functions and tile methods (t.xxx) it uses, "
                              "comma-separated>\n"
                              "LAYOUT: <what goes on the partition axis, and the tile shapes for the first "
                              "test shape>\n"
                              "STEPS: <3 to 6 short numbered steps>", required=True),
        ]
        rounds = self.cfg.lookup.get("planner", 0)
        pulled, text = [], ""
        if self.cfg.docs_request and rounds > 0:
            # The first call only chooses documentation (lookup.REQUEST); it counts as one round.
            pulled, text, meta = lookup.request(self.llm, "planner", sections, level, ledger, self.events,
                                                tags=dict(level=level))
            if meta.get("error"):
                return None, meta
            rounds -= 1
        # Unless it already answered with a whole plan (one cut at the request's cap is asked for again).
        if not (re.search(r"(?im)^\W*APPROACH\W*:", text) and meta.get("finish_reason") != "length"):
            text, meta, pulled = lookup.ask(self.llm, "planner", sections, self.retriever, level, rounds,
                                            ledger, self.events, pulled=pulled, tags=dict(level=level))
        if meta.get("error") or lookup.parse_lookup(text):     # still asking after "answer now"
            return None, meta
        plan = parse_plan(text)
        if plan is None:
            return None, meta
        # A real name under the wrong prefix (nl.tensor_reduce, nl.reshape) is corrected, not dropped:
        # 31% of all names in names-only plans on seat-198 (agents2/index.py).
        from agents2.index import fix_name
        plan.calls = list(dict.fromkeys(fix_name(self.retriever, c, level) for c in plan.calls))
        plan.unknown = [c for c in plan.calls if not self.retriever.exists(c)]
        plan.calls = [c for c in plan.calls if c not in plan.unknown]
        plan.docs = pulled
        return plan, meta
