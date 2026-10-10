"""The repair loop: generate freely, verify, translate the verdict into one change, repeat.

    attempt 1   one-sentence task + signature. No rules: rules live in the verifier.
    attempt k   previous code + ONE instruction (+ evidence), and the ledger only when a failure
                class repeats. Never the reference, never the rules list.
    no code     (truncated / empty / no block) never reaches the verifier; the prompt changes.
    pass        the kernel is re-checked on holdout cases the model never saw; that decides the
                confidence it reports.

Every attempt is logged with its input tokens split by section, so where the budget goes is data.

    python -m kagent.agent --levels 1 2 3 --attempts 8 --repeat 5
    python -m kagent.agent --levels 1 --scripted tests/scripts/l1.json     # no model
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import client
from .extract import extract_code
from .harness import verify
from .levels import LEVELS
from . import judge
from .translate import Directive, judge_directive, rewrite_directive, translate

def code_version():
    """Hash of the agent/harness sources, logged with every result so runs on different pods can
    be checked to have used the same harness."""
    h = hashlib.sha1()
    for f in sorted(Path(__file__).parent.glob("*.py")):
        if not f.name.startswith("."):  # macOS tar can leave ._*.py AppleDouble files on a pod
            h.update(f.read_bytes())
    return h.hexdigest()[:10]


VERSION = code_version()
ANSWER_TOKENS = 3000
HOLDOUT_TIMEOUT_S = 120.0   # holdout shapes are larger: a correct per-element kernel must not read as wrong (E21)   # room reserved for the reply; the rest of the context is the prompt budget
SECTIONS = ("task", "code", "feedback", "ledger")


# ---------------------------------------------------------------- models

class LiveModel:
    def __init__(self, temperature=0.6, think=False):
        self.temperature, self.think = temperature, think
        self.exact_tokens = True

    def count(self, text):
        return client.count_text(text)

    def complete(self, prompt, sample):
        msgs = [{"role": "user", "content": prompt}]
        return client.chat(msgs, max_tokens=ANSWER_TOKENS, temperature=self.temperature,
                           sample=sample, think=self.think)


class ScriptedModel:
    """Replays canned replies in order, to exercise the loop without a server. Token counts are
    estimates (chars/4) and are marked as such in the log."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.exact_tokens = False
        self.prompts = []

    def count(self, text):
        return max(1, len(text) // 4)

    def complete(self, prompt, sample):
        self.prompts.append(prompt)
        r = self.replies.pop(0) if self.replies else ""
        finish = "length" if r.startswith("<TRUNCATED>") else "stop"
        r = r.removeprefix("<TRUNCATED>")
        return client.Reply(content=r, reasoning="", finish_reason=finish,
                            prompt_tokens=self.count(prompt) + 10,
                            completion_tokens=max(1, len(r) // 4), max_tokens=ANSWER_TOKENS,
                            latency_s=0.0)


# ---------------------------------------------------------------- prompts

# Same meaning, different wording. The server is near-deterministic (E10), so identical prompts
# make repeated runs identical; run k uses wording k % 5, which turns "5 runs" into a rate over
# phrasings instead of five copies of one run.
FIRST_WORDINGS = [
    "Write a Python function `{sig}` using numpy. {task} Process the array in tiles of at most "
    "128 rows x 512 columns with explicit python loops over the tiles; the last tile in each "
    "direction may be smaller. Reply with one ```python code block and nothing else.",
    "Implement `{sig}` in Python with numpy. {task} Work tile by tile, each tile at most 128 rows "
    "by 512 columns, using explicit python loops; edge tiles can be smaller. Answer with a single "
    "```python code block.",
    "{task} Write this as a numpy function `{sig}` that walks over the input in tiles of up to "
    "128 x 512 elements with plain python loops (the last tiles may be partial). Only output one "
    "```python block.",
    "Task: {task}\nSignature: `{sig}` (numpy). Use explicit python loops over tiles of at most "
    "128 rows and 512 columns; the final tile along each axis may be smaller. Output exactly one "
    "```python code block.",
    "Using numpy, write `{sig}`. {task} The computation must go tile by tile (tiles up to 128 "
    "rows x 512 columns, smaller at the edges) with explicit python loops. Return only a "
    "```python code block.",
]


def first_prompt(level, terse, wording=0):
    if terse == 0:
        text = FIRST_WORDINGS[wording % len(FIRST_WORDINGS)].format(sig=level.signature, task=level.task)
    else:
        text = (f"Write a Python function `{level.signature}` using numpy. {level.task} "
                f"Reply with only a ```python code block.")
    return [("task", text)]


def repair_prompt(level, code, d, ledger, note=""):
    parts = [("task", f"This numpy function `{level.signature}` should do this: {level.task}"),
             ("code", f"```python\n{code}```"),
             ("feedback", (d.evidence + "\n" if d.evidence else "") + d.instruction)]
    if ledger:
        parts.append(("ledger", "Already tried, and it did not fix this:\n"
                                + "\n".join(f"- {x}" for x in ledger)
                                + "\nDo something different this time."))
    scope = ("Rewrite the loop structure as needed to make this change, and keep what already "
             "works." if d.structural else "Change only that and keep everything else identical.")
    parts.append(("task", (note + " " if note else "") + scope + " Reply with the complete "
                          "function in one ```python code block."))
    return parts


def naive_repair_prompt(level, code, report_text):
    """Ablation baseline: the checker's report verbatim, no translation, no ledger."""
    return [("task", f"This numpy function `{level.signature}` should do this: {level.task}"),
            ("code", f"```python\n{code}```"),
            ("feedback", f"A checker reports:\n{report_text}"),
            ("task", "Fix the kernel. Reply with the complete function in one ```python code block.")]


def render(parts):
    return "\n\n".join(t for _, t in parts)


def account(model, parts):
    """Input tokens per section. 'template' is the chat-template overhead (prompt - sections)."""
    out = dict.fromkeys(SECTIONS, 0)
    for name, text in parts:
        out[name] += model.count(text)
    return out


# ---------------------------------------------------------------- the loop

@dataclass
class Attempt:
    level: int
    run: int
    attempt: int
    prompt_kind: str            # first | repair | reask
    sections: dict
    prompt_tokens: int
    completion_tokens: int
    finish: str
    latency_s: float
    cached: bool
    exact_tokens: bool
    outcome: str                # pass | fail | no-code | truncated | empty
    key: str | None = None      # failure class (taxonomy)
    instruction: str | None = None
    summary: str = ""
    code: str | None = None


@dataclass
class Result:
    level: int
    run: int
    status: str                 # verified | dev-only | failed | stuck
    confidence: float
    attempts: int
    tokens_in: int
    tokens_out: int
    final_key: str | None
    holdout: str = ""
    code: str | None = None
    keys: list = field(default_factory=list)
    mode: str = "ours"
    wording: int = 0
    version: str = VERSION


def _rank(rep):
    crashes = sum(r.status in ("error", "timeout") for r in rep.results)
    return (rep.passed, rep.load_error is None, -crashes, rep.n_pass, -len(rep.violations))


CYCLE_AT = 3  # distinct rule/crash classes in one run before a forced rewrite


def _progress(rep):
    """How far a failing kernel got: fewer crashes, more cases right, fewer rule violations."""
    crashes = sum(r.status in ("error", "timeout") for r in rep.results)
    return (rep.load_error is None, -crashes, rep.n_pass, -len(rep.violations))


def _code_hash(code):
    return hashlib.sha1(code.encode()).hexdigest()[:10]


def solve(level, model, run=0, max_attempts=8, samples=1, patience=3, log=None, say=print,
          mode="ours", wording=0):
    """mode="naive" is the ablation: verdict fed back verbatim, no ledger, no reply gate (a
    no-code reply resends the same prompt). Stopping rule, holdout and scoring are identical."""
    naive = mode == "naive"
    terse, code, directive, note, report_text = 0, None, None, "", ""
    history = []        # (instruction given, failure key it produced)
    seen_hashes = set()
    keys, streak = [], 0
    last_progress = None
    tin = tout = 0

    for att in range(1, max_attempts + 1):
        if code is None:
            parts, kind = first_prompt(level, terse, wording), ("first" if att == 1 else "reask")
        elif naive:
            parts, kind = naive_repair_prompt(level, code, report_text), "repair"
        else:
            # the ledger costs tokens; it is only worth them once a failure class comes back
            ledger = [g for g, k in history if k == directive.key] if directive.key in keys[:-1] else []
            parts, kind = repair_prompt(level, code, directive, ledger[-4:], note), "repair"
        prompt = render(parts)
        sections = account(model, parts)

        with cf.ThreadPoolExecutor(max_workers=samples) as ex:
            replies = list(ex.map(lambda s: model.complete(prompt, f"{run}:{att}:{s}"), range(samples)))
        tin += sum(r.prompt_tokens for r in replies)
        tout += sum(r.completion_tokens for r in replies)

        cands = []
        for r in replies:
            src = extract_code(r.content) if r.finish_reason == "stop" else None
            if src:
                cands.append((r, src, verify(level, src)))
        r0 = replies[0]
        base = dict(level=level.num, run=run, attempt=att, prompt_kind=kind, sections=sections,
                    prompt_tokens=r0.prompt_tokens, completion_tokens=r0.completion_tokens,
                    finish=r0.finish_reason, latency_s=r0.latency_s, cached=r0.cached,
                    exact_tokens=model.exact_tokens)

        if not cands:
            outcome = ("truncated" if r0.finish_reason == "length"
                       else "empty" if not r0.content.strip() else "no-code")
            _log(log, Attempt(**base, outcome=outcome, key=f"reply:{outcome}"))
            say(_line(level, att, sections, r0, f"NO CODE ({outcome})"))
            # greedy-ish decoding: the same prompt gives the same non-answer, so change it
            if not naive:
                if code is None:
                    terse = min(terse + 1, 1)
                note = "Reply with the code block only, no explanation."
            keys.append(f"reply:{outcome}")
            continue

        r, src, rep = max(cands, key=lambda c: _rank(c[2]))
        judge_fail, judged = None, True
        if rep.passed:
            ok, jmsg = judge.check(level.num, src)
            judged = ok is not None
            if ok is False:
                judge_fail = judge_directive(level, jmsg)
        if rep.passed and judge_fail is None:
            hold = verify(level, src, holdout=True, timeout_s=HOLDOUT_TIMEOUT_S)
            status, conf = ("verified", 0.95) if hold.passed else ("dev-only", 0.3)
            if hold.passed and not judged:      # never claim the final check that did not run
                status, conf = "verified-unjudged", 0.6
            _log(log, Attempt(**base, outcome="pass", summary=rep.summary(), code=src))
            say(_line(level, att, sections, r, f"PASS dev; holdout {hold.n_pass}/{len(hold.results)}"))
            res = Result(level.num, run, status, conf, att, tin, tout, None,
                         hold.summary(), src, keys, mode, wording)
            _log(log, res)
            return res

        directive = judge_fail or translate(rep)  # naive mode uses only the key, for the taxonomy
        # Repairs that keep landing on a different rule or crash are going round in circles (live
        # v2/v3 L5: sum-like -> max-like -> layout -> crash -> sum-like until the budget ran out, so
        # 'stuck' never fired). Once per run, after CYCLE_AT distinct such classes, ask for the rewrite.
        if not naive and not judge_fail and directive.key != "rule:rewrite" and "rule:rewrite" not in keys:
            seen = {k for k in keys if k.startswith(("rule:", "crash:"))} | {directive.key}
            if len(seen) >= CYCLE_AT:
                directive = rewrite_directive(rep, cycling=True) or directive
        report_text = rep.text() if judge_fail is None else f"{rep.summary()}\nFinal check: {judge_fail.evidence}"
        repeat_code = _code_hash(src) in seen_hashes
        seen_hashes.add(_code_hash(src))
        note = ("Your last reply repeated an earlier version unchanged, so the change was not "
                "made." if repeat_code and not naive else "")
        if history:
            history[-1] = (history[-1][0], directive.key)
        history.append((directive.short(), None))
        # Same failure class again counts toward "stuck" only when nothing improved. Live v3 L9 run 1
        # went 13 -> 12 -> 10 -> 6 -> 5 violations and L8 runs stopped at 3 repeats of a class while
        # the count was still falling; a repeated class with a falling count is progress, not a loop.
        # Identical code is never progress. Same rule for both arms.
        prog = _progress(rep)
        improved = (last_progress is not None and prog > last_progress and not repeat_code)
        if keys and keys[-1] == directive.key:
            streak = streak if improved else streak + 1
        else:
            streak = 1
        last_progress = prog
        keys.append(directive.key)
        code = src
        _log(log, Attempt(**base, outcome="fail", key=directive.key,
                          instruction=directive.instruction, summary=rep.summary(), code=src))
        say(_line(level, att, sections, r, f"FAIL {directive.key}{' (repeat code)' if repeat_code else ''}"))
        if not naive:
            say(f"      -> {directive.short(160)}")
        if streak >= patience:
            res = Result(level.num, run, "stuck", 0.0, att, tin, tout, directive.key, "", code, keys,
                         mode, wording)
            say(f"   STUCK: '{directive.key}' {streak} attempts in a row without progress; more of the same won't help")
            _log(log, res)
            return res

    res = Result(level.num, run, "failed", 0.0, max_attempts, tin, tout,
                 directive.key if directive else (keys[-1] if keys else None), "", code, keys,
                 mode, wording)
    _log(log, res)
    return res


def _line(level, att, sections, r, verdict):
    sec = " ".join(f"{k[:4]}={v}" for k, v in sections.items() if v)
    c = " cached" if r.cached else ""
    return (f"L{level.num} #{att} in={r.prompt_tokens} [{sec}] out={r.completion_tokens} "
            f"{r.latency_s:.1f}s{c}  {verdict}")


def _log(log, rec):
    if log is not None:
        d = asdict(rec)
        d["type"] = type(rec).__name__.lower()
        log.write(json.dumps(d) + "\n")
        log.flush()


# ---------------------------------------------------------------- cli

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--attempts", type=int, default=12)   # v3: L4 runs ran out at 8 (E21)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--patience", type=int, default=3, help="stop after this many same-class failures in a row")
    ap.add_argument("--repeat", type=int, default=1, help="independent runs per level; report the rate")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--mode", choices=("ours", "naive"), default="ours",
                    help="naive = ablation: checker report verbatim, no ledger, no reply gate")
    ap.add_argument("--same-wording", action="store_true",
                    help="use wording 0 for every run (default: run k uses wording k %% 5)")
    ap.add_argument("--scripted", help="JSON {level: [reply, ...]} to replay instead of a model")
    ap.add_argument("--out", default=None, help="JSONL log (default runs/<time>.jsonl)")
    a = ap.parse_args(argv)

    out = Path(a.out or f"runs/{time.strftime('%Y%m%d-%H%M%S')}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    script = json.loads(Path(a.scripted).read_text()) if a.scripted else None
    results = []
    with out.open("a") as log:
        for run in range(a.repeat):
            for n in a.levels:
                lv = LEVELS[n]
                print(f"\n=== run {run + 1}/{a.repeat}  level {n}: {lv.name}  [{a.mode}, v{VERSION}] ===")
                model = ScriptedModel(script[str(n)]) if script else LiveModel(a.temperature, a.think)
                results.append(solve(lv, model, run, a.attempts, a.samples, a.patience, log,
                                     mode=a.mode, wording=0 if a.same_wording else run))
                r = results[-1]
                print(f"   => {r.status} (confidence {r.confidence:.2f}) after {r.attempts} "
                      f"attempt(s), {r.tokens_in} in / {r.tokens_out} out tokens")

    print(f"\n=== summary over {a.repeat} run(s) ===")
    for n in a.levels:
        rs = [r for r in results if r.level == n]
        ok = [r for r in rs if r.status == "verified"]
        att = sum(r.attempts for r in ok) / len(ok) if ok else float("nan")
        print(f"  L{n} {LEVELS[n].name:12s} verified {len(ok)}/{len(rs)}  "
              f"mean attempts when verified {att:.1f}  statuses {[r.status for r in rs]}")
    print(f"log: {out}  (code version {VERSION})")


if __name__ == "__main__":
    sys.exit(main())
