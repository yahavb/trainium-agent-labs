#!/usr/bin/env python3
"""
agent.py — the v2 loop: the model writes a kernel, the verifier grades it, the INSTRUCTION
goes back, it tries again. Runs anywhere with numpy; on a seat pod it talks to the local
Qwen server (the pod env sets KERNEL_AGENT_BASE_URL / KERNEL_AGENT_MODEL already).

What v2 changes against project 02's agent.py, each mapping to a measured wall there:

  tools            SCRATCH (persistent numpy REPL) + DOCS (retrieval over distilled
                   cards), over a text-marker protocol. The model decides what to check;
                   the REPL checks it. Project 01 measured this split: guesses -> exact.
  rules->verifier  the prompt carries NO rule list (measured twice: rule lists make the
                   model audit itself in its hidden channel and return nothing). The
                   verifier catches, and errors.py turns the verdict into one named change.
  context          a budget manager with auto-compaction (context.py). Budget defaults to
                   the server's own max_model_len minus the answer reserve -- the 8192
                   wall is one config point, not the design.
  memory           a distilled optimization card carried across levels and runs
                   (memory.py; the AccelOpt idea, scaled down).
  calibration      every attempt ends with CONFIDENCE: high|medium|low; the trace records
                   confidence vs verified result. Confidently-wrong is named, because the
                   rubric scores it below honest failure.
  traces           one JSONL per run with per-attempt token spend BY COMPONENT, tool
                   exchanges, compaction events, taxonomy labels. dashboard/ reads it.

    python agent.py --all --rounds 8 --samples 4            # the ladder
    python agent.py --level 5 --repeat 3 --tag softmax      # one level, a rate
    python agent.py --all --holdout                          # include levels 11-13
    python agent.py --offline --level 1                      # no endpoint; exercise the loop

Every attempt is appended to runs/<run>/trace.jsonl. One run is not a result: use --repeat.
"""

import argparse
import concurrent.futures as cf
import datetime
import json
import os
import re
import sys
import time

import numpy as np

import context as ctx_mod
import errors
import ladder
import memory as memory_mod
import tools
import verifier

DOCS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")

# Measured floor on both workshop endpoints: the model writes to a hidden reasoning
# channel first, and a too-small answer budget comes back EMPTY, looking like a refusal.
MIN_ANSWER_TOKENS = 2500
REASONING_KEYS = ("reasoning", "reasoning_content")

# Nudges toward the REPL, per failure mode, offered when the same wall repeats. They aim
# the tool; they do not do the work.
EXPERIMENTS = {
    "ragged-edge": "SCRATCH: x = np.arange(129*513.).reshape(129,513); [(r0, x[r0:r0+128].shape) for r0 in range(0, 129, 128)]",
    "non-finite": "SCRATCH: np.exp(np.array([100., 800., 1000.]))",
    "partial-coverage": "SCRATCH: R, C = 300, 1100; [(r0, c0, x_slice_shape := (min(128, R-r0), min(512, C-c0))) for r0 in range(0, R, 128) for c0 in range(0, C, 512)]",
    "wrong-shape": "SCRATCH: L, K, s, d = 31, 3, 2, 3; (L - d*(K-1) - 1)//s + 1",
    "core-arithmetic": "SCRATCH: t = np.array([[1., 2., 3.], [4., 5., 6.]]); (t.max(axis=1, keepdims=True), t - t.max(axis=1, keepdims=True))",
    "stat-scope": "SCRATCH: t = np.arange(6.).reshape(2, 3); (np.sqrt((t*t).mean(axis=1, keepdims=True)), np.sqrt((t[:, 0:2]*t[:, 0:2]).mean(axis=1, keepdims=True)))",
    "no-tile-loop": "SCRATCH: R = 129; [r0 for r0 in range(0, R, 128)]",
    "banned-call": "SCRATCH: t = np.arange(6.).reshape(2, 3); t.sum(axis=1)",
    "nondeterministic": "SCRATCH: out = np.zeros(4); out",
}


# ---------------------------------------------------------------- prompting

def header(level_n, terse):
    """Everything the model always gets. Deliberately does NOT list the tiling rules --
    the verifier owns those (see module docstring). At terse 0 it carries the tool
    protocol; higher terseness shrinks it, because a long prompt makes these models
    reason instead of answer (project 02, both endpoints)."""
    s = ladder.LEVELS[level_n]
    import inspect
    if terse >= 2:
        return (f"Write a python function `kernel` with the same signature and meaning as "
                f"this reference, tiling the work in <=128x512 slices inside explicit "
                f"loops. Begin the file with `import numpy as np`:\n\n"
                f"{inspect.getsource(s['ref'])}\n"
                f"Reply with one ```python block, then a CONFIDENCE: line.")
    tools_txt = tools.TOOL_INSTRUCTIONS if terse == 0 else (
        "Optional: lines starting `SCRATCH:` run in a persistent numpy session (np "
        "preloaded) and `DOCS: <topic>` searches the local reference cards. Answer with "
        "one ```python block defining `kernel`, then a CONFIDENCE: high|medium|low line.")
    # One concrete, positive requirement -- the measured style that works (a worked
    # example beats a prohibition). The first live run lost three rounds to
    # `NameError: name 'np' is not defined`, so the file's first line is stated here.
    return (f"Write a NumPy kernel -- a function `kernel(...)` with the same signature and "
            f"meaning as this reference. Begin the file with `import numpy as np`. The "
            f"work must be tiled: explicit loops over slices of at most "
            f"{ladder.TILE_ROWS} rows x {ladder.TILE_COLS} columns, with the final "
            f"partial slice handled.\n\n"
            f"{inspect.getsource(s['ref'])}\n\n{tools_txt}")


# ---------------------------------------------------------------- the model call

def ask_model(a, prompt, max_answer_tokens=None):
    import httpx
    budget = max_answer_tokens or a.max_tokens
    est_prompt = len(prompt) // 4
    if a.context and est_prompt + budget + 64 > a.context:
        budget = max(512, a.context - est_prompt - 64)
    body = dict(model=a.model, messages=[{"role": "user", "content": prompt}],
                max_tokens=budget, temperature=a.temperature, top_p=a.top_p,
                chat_template_kwargs={"enable_thinking": a.think})
    r = httpx.post(f"{a.base.rstrip('/')}/chat/completions", json=body,
                   timeout=a.timeout, verify=False)
    if r.status_code != 200:
        raise SystemExit(f"the endpoint returned HTTP {r.status_code}:\n{r.text[:600]}")
    payload = r.json()
    ch = payload["choices"][0]
    msg = ch.get("message", {})
    reasoning = next((msg[k] for k in REASONING_KEYS if msg.get(k)), "")
    content = msg.get("content") or ""
    usage = payload.get("usage") or {}
    return dict(content=content, reasoning=reasoning, finish=ch.get("finish_reason"),
                usage=usage)


def ask_parallel(a, prompt, n):
    if n <= 1:
        return [ask_model(a, prompt)]
    with cf.ThreadPoolExecutor(max_workers=n) as ex:
        return [f.result() for f in [ex.submit(ask_model, a, prompt) for _ in range(n)]]


# ---------------------------------------------------------------- offline replay
#
# No endpoint. Replays canned replies through the identical loop -- prompt build, tool
# parsing, verification, tracing -- so the machinery can be exercised with no model.

OFFLINE_GOOD = {
    1: ("```python\nimport numpy as np\ndef kernel(x, a, b):\n"
        "    R, C = x.shape\n    out = np.zeros((R, C), dtype=np.float64)\n"
        "    for r0 in range(0, R, 128):\n        for c0 in range(0, C, 512):\n"
        "            t = x[r0:r0+128, c0:c0+512]\n            out[r0:r0+128, c0:c0+512] = np.maximum(a * t + b, 0.0)\n"
        "    return out.astype(np.float32)\n```\nCONFIDENCE: medium"),
    2: ("```python\nimport numpy as np\ndef kernel(x):\n"
        "    R, C = x.shape\n    out = np.zeros(R, dtype=np.float64)\n"
        "    for r0 in range(0, R, 128):\n        t = x[r0:r0+128]\n"
        "        out[r0:r0+128] = t.astype(np.float64).sum(axis=1)\n"
        "    return out.astype(np.float32)\n```\nCONFIDENCE: high"),
}
OFFLINE_FIRST = ("Let me check the tiling first.\n"
                 "SCRATCH: x = np.arange(12).reshape(3, 4); x[0:2, 1:3].shape\n"
                 "DOCS: ragged final tile\n")


def offline_reply(level_n, rnd):
    if rnd == 0:
        return OFFLINE_FIRST
    return OFFLINE_GOOD.get(level_n, OFFLINE_GOOD[2])


# ---------------------------------------------------------------- per-level solve

class LevelRun:
    """Everything one level's solving loop needs: context components, REPL, docs, ledger,
    cycle detection, and the trace emitter."""

    def __init__(self, a, level_n, log, docs, mem):
        self.a, self.level_n, self.log, self.docs, self.mem = a, level_n, log, docs, mem
        s = ladder.LEVELS[level_n]
        budget = max(1200, a.context - a.max_tokens - 256) if a.context else 6000
        self.ctx = ctx_mod.Context(budget)
        self.scratch = tools.Scratch()
        self.scratch_txt = ""
        self.docs_txt = ""
        self.ledger = []          # one line per attempt
        self.terse = a.terse
        self.seen = {}            # failure signature -> count
        self.best = (0.0, None, None)
        self.calibration = {"high": [0, 0], "medium": [0, 0], "low": [0, 0]}
        self.taxonomy_counts = {}
        self.latest = ("", None)  # (code, failure) of the LATEST attempt -- repaired, not
                                  # the best (project 02: repairing the best is a fixed point)

    def emit(self, **rec):
        rec["level"] = self.level_n
        self.log.write(json.dumps(rec, default=str) + "\n")
        # flush EVERY line: the dashboard polls this file while the run is live, and an
        # 8KB block buffer hides minutes of attempts (measured: the baseline's trace sat
        # empty for 12 minutes while rounds printed to run.log)
        self.log.flush()

    def log_tool(self, kind, ask, result):
        self.emit(ts=time.time(), type="tool", tool=kind, ask=ask[:300],
                  result=(result or "")[:MAX_TOOL_LOG])
        if kind == "SCRATCH":
            self.scratch_txt += f"SCRATCH: {ask}\n  -> {result}\n"
        else:
            self.docs_txt = f"DOCS: {ask}\n{result}\n"

    def render_prompt(self):
        self.ctx.set("reference", header(self.level_n, self.terse), floor=-1)
        self.ctx.set("memory", ("What worked before on this machine (memory card; "
                                "strategy, not answers):\n" + self.mem.card(self.level_n)
                                ) if not self.terse else "", floor=0)
        self.ctx.set("docs", self.docs_txt, floor=0)
        self.ctx.set("kernel", (f"Your previous kernel:\n\n```python\n{self.latest[0]}\n```"
                                ) if self.latest[0] else "", floor=-1)
        self.ctx.set("feedback", (f"The checker reports:\n{self.latest[1]}"
                                  ) if self.latest[1] else "", floor=-1)
        self.ctx.set("ledger", ("\nAttempts that already failed, do not repeat them:\n"
                                + "\n".join(self.ledger)) if self.ledger else "", floor=0)
        self.ctx.set("scratch", ("\nYour SCRATCH session so far:\n" + self.scratch_txt
                                 ) if self.scratch_txt else "", floor=0)
        prompt = self.ctx.render()
        if self.ctx.events:
            for ev in self.ctx.events:
                self.emit(ts=time.time(), type="compact", **ev)
        return prompt

    def token_report(self, prompt, usage=None):
        # per-component sizes as built -- rebuild the map without recompacting
        parts = {k: v for k, v in self.ctx.texts.items()}
        return ctx_mod.token_report(parts, usage)

    def grade_attempt(self, code, conf, meta):
        r = verifier.check(code, self.level_n)
        tax = r["taxonomy"] or ("solved" if r["solved"] else "unknown")
        if not r["solved"]:
            self.taxonomy_counts[tax] = self.taxonomy_counts.get(tax, 0) + 1
            if conf in self.calibration:
                self.calibration[conf][1] += 1          # confident claims that failed
            if conf == "high" and not r["solved"]:
                self.taxonomy_counts["confidently-wrong"] = \
                    self.taxonomy_counts.get("confidently-wrong", 0) + 1
        if conf in self.calibration:
            self.calibration[conf][0] += 1
        return r, tax

    def solve(self):
        a = self.a
        print(f"\n===== level {self.level_n}: {ladder.LEVELS[self.level_n]['name']}"
              f"{'  [HOLDOUT]' if ladder.LEVELS[self.level_n]['holdout'] else ''} =====")
        solved_round = None
        for rnd in range(a.rounds):
            t0 = time.perf_counter()
            prompt = self.render_prompt()
            attempts_this_round = []
            got_code = False
            for pass_i in range(a.tool_passes + 1):
                if a.offline:
                    raws = [dict(content=offline_reply(self.level_n, rnd), reasoning="",
                                 finish="stop", usage={})] * a.samples
                else:
                    try:
                        raws = ask_parallel(a, prompt, a.samples)
                    except SystemExit as e:
                        print(f"  endpoint: {e}")
                        return self.finish(rnd, solved_round, endpoint_dead=True)
                tool_asks_scratch, tool_asks_docs = [], []
                for raw in raws:
                    code, sc, dc, conf = tools.parse_reply(raw["content"])
                    for q in sc:
                        if q not in tool_asks_scratch:
                            tool_asks_scratch.append(q)
                    for q in dc:
                        if q not in tool_asks_docs:
                            tool_asks_docs.append(q)
                    if code:
                        # A fenced block that does not define `kernel` is the model
                        # sharing a scratch experiment, not an attempt -- grading it
                        # burns a round on a guaranteed parse-error (measured: 242 s to
                        # learn 'no callable kernel defined'). Treat it as a tool pass.
                        if "def kernel" not in code and (tool_asks_scratch or tool_asks_docs):
                            self.log_tool("NOTE", "code block without kernel() seen",
                                          "reply with the kernel in one ```python block")
                            continue
                        got_code = True
                        src = tools.extract_code(raw["content"]) or code
                        r, tax = self.grade_attempt(src, conf, raw)
                        tok = self.token_report(prompt, raw.get("usage"))
                        rec = dict(ts=time.time(), type="attempt", round=rnd, sample=len(attempts_this_round),
                                   reward=r["reward"], parts=r["parts"], solved=r["solved"],
                                   taxonomy=tax, confidence=conf or "none",
                                   tokens=tok, finish=raw["finish"],
                                   reasoning_chars=len(raw.get("reasoning") or ""),
                                   reply_chars=len(raw.get("content") or ""),
                                   verdict=(r["failures"][0]["verdict"] if r["failures"] else ""),
                                   instruction=(r["failures"][0]["instruction"] if r["failures"] else ""),
                                   code=src[:MAX_CODE_LOG])
                        self.emit(**rec)
                        attempts_this_round.append((r, src, tax, conf))
                # tool exchanges first: they do not burn attempts
                for q in tool_asks_scratch:
                    self.log_tool("SCRATCH", q, self.scratch.run(q))
                for q in tool_asks_docs:
                    self.log_tool("DOCS", q, self.docs.search(q))
                if got_code or a.offline:
                    break
                # no code anywhere: run one more tool pass, then escalate terseness
                prompt = self.render_prompt()
            else:
                pass

            dt = time.perf_counter() - t0
            if not attempts_this_round:
                self.terse = min(self.terse + 1, 2)
                self.taxonomy_counts["empty-answer"] = \
                    self.taxonomy_counts.get("empty-answer", 0) + 1
                self.emit(ts=time.time(), type="attempt", round=rnd, reward=0.0,
                          taxonomy="empty-answer", confidence="none", solved=False,
                          tokens=self.token_report(prompt), terse=self.terse,
                          verdict="No code in any sample; escalating prompt terseness.",
                          instruction=errors.instruction_from_failure(
                              dict(taxonomy="empty-answer", verdict=""), ""))
                print(f"round {rnd}: no code from any sample ({dt:.1f}s) -> terseness {self.terse}")
                continue

            attempts_this_round.sort(key=lambda t: t[0]["reward"], reverse=True)
            r, src, tax, conf = attempts_this_round[0]
            if r["reward"] > self.best[0]:
                self.best = (r["reward"], src, r)
            if r["solved"]:
                solved_round = rnd           # 0-indexed; attempts-to-green is rnd+1
                self.latest = (src, None)
                print(f"round {rnd}: SOLVED ({dt:.1f}s)  confidence={conf or 'none'}")
                break

            # repair the LATEST, not the best -- a fixed point otherwise (project 02)
            fail0 = r["failures"][0] if r["failures"] else dict(taxonomy=tax, verdict="", instruction="")
            instr = fail0.get("instruction") or errors.instruction_from_failure(fail0, fail0.get("case", ""))
            self.latest = (src, instr + (f" [case: {fail0.get('case', '')}]"
                                          if fail0.get("case") else ""))
            line = f"- r{rnd} {tax}: {fail0.get('verdict', '')[:110]}"
            if line not in self.ledger:
                self.ledger.append(line)
            sig = tax + "|" + fail0.get("verdict", "")[:120]
            self.seen[sig] = self.seen.get(sig, 0) + 1

            same = max(self.seen.values())
            print(f"round {rnd}: {r['reward']:.2f}  best {self.best[0]:.2f}  {tax}"
                  f"  ({dt:.1f}s)  conf={conf or 'none'}")
            print(f"  {fail0.get('verdict', '')[:200]}")
            if same >= a.give_up_after:
                self.taxonomy_counts["no-improvement"] = 1
                print(f"  STOPPING: the same failure (or a fixed set) {same}x. Cycling, "
                      f"not converging.")
                break
            if same >= 2:
                # A repeated exception means the model is editing blind: make it
                # reproduce the exception on a small array first -- reading the actual
                # message beats a third blind rewrite (measured: level 4 burned four
                # identical rounds on one concat error).
                exp = EXPERIMENTS.get(tax) or (
                    "Reproduce this exact exception on a SMALL array (e.g. a (3, 5) "
                    "tile) with a SCRATCH: line, read the message, then fix the kernel.")
                self.log_tool("SCRATCH-HINT", exp, "(suggested to the model)")
                self.latest = (self.latest[0], self.latest[1] +
                               f"\nBefore answering, do this and read its output:\n  {exp}")
        return self.finish(rnd + 1, solved_round)

    def finish(self, rounds_used, solved_round, endpoint_dead=False):
        solved = solved_round is not None
        if solved:
            self.mem.observe(self.level_n, self.taxonomy_counts, solved=True)
            self.mem.save()
        cal = {k: dict(claims=v[0], wrong=v[1]) for k, v in self.calibration.items() if v[0]}
        self.emit(ts=time.time(), type="level_end", solved=solved,
                  rounds=rounds_used, solved_round=solved_round,
                  best=self.best[0], taxonomy_counts=self.taxonomy_counts,
                  calibration=cal)
        if not solved:
            print(f"  not solved in {rounds_used} round(s); best {self.best[0]:.2f}. "
                  f"Taxonomy: {self.taxonomy_counts or '{}'}")
        # solved_round stays None when unsolved -- round 0 is a real round and must not
        # read as false (measured: the smoke run solved level 1 on round 0 and the
        # summary printed 'not solved')
        return self.best[0], solved_round


MAX_CODE_LOG = 6000
MAX_TOOL_LOG = 900


# ---------------------------------------------------------------- main

def discover_context(a):
    """--context auto: ask the server. The wall is whatever THIS server was started
    with; nothing about the loop hard-codes 8192."""
    if a.context and a.context != "auto":
        a.context = int(a.context)
        return
    try:
        import httpx
        r = httpx.get(f"{a.base.rstrip('/')}/models", timeout=10, verify=False)
        mml = r.json()["data"][0].get("max_model_len")
        a.context = int(mml) if mml else 8192
        print(f"context: server reports max_model_len={a.context}")
    except Exception as e:
        a.context = 8192
        print(f"context: could not query server ({e}); assuming {a.context}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, choices=sorted(ladder.LEVELS))
    ap.add_argument("--levels", default="",
                    help="comma list, e.g. --levels 4,5,6 -- iterate exactly these")
    ap.add_argument("--all", action="store_true", help="the graded ladder, levels 1-10")
    ap.add_argument("--holdout", action="store_true",
                    help="include the self-holdout tier 11-13 (never tuned against)")
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--tool-passes", type=int, default=2,
                    help="SCRATCH/DOCS exchanges allowed before an answer is demanded")
    ap.add_argument("--repeat", type=int, default=1,
                    help="run N times and report a solve RATE (one run is not a result)")
    ap.add_argument("--max-tokens", type=int, default=MIN_ANSWER_TOKENS)
    ap.add_argument("--context", default="auto",
                    help="'auto' asks the server's max_model_len; or an int")
    ap.add_argument("--model", default=os.environ.get("KERNEL_AGENT_MODEL", "Qwen/Qwen3-8B"))
    ap.add_argument("--base", default=os.environ.get("KERNEL_AGENT_BASE_URL",
                                                     os.environ.get("GPTOSS_BASE_URL")))
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--greedy", action="store_true",
                    help="temperature 0 -- for byte-reproducible runs on sampling servers")
    ap.add_argument("--terse", type=int, default=0, choices=(0, 1, 2), help="starting prompt size")
    ap.add_argument("--think", action="store_true",
                    help="hidden reasoning channel ON; measured much worse on both endpoints")
    ap.add_argument("--give-up-after", type=int, default=4,
                    help="stop a level after this many occurrences of one failure signature")
    ap.add_argument("--offline", action="store_true", help="no endpoint; canned replies")
    ap.add_argument("--tag", default="", help="label for the run directory")
    ap.add_argument("--runs-dir", default=os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "runs"))
    a = ap.parse_args()

    a.temperature, a.top_p = (0.0, 1.0) if a.greedy else (0.6, 0.95)

    if not a.offline:
        raw = (a.base or "").strip()
        if not raw:
            sys.exit("KERNEL_AGENT_BASE_URL is empty or unset, and --offline was not passed.\n"
                     "In a seat pod it is already set (http://localhost:8000/v1). Otherwise:\n"
                     "  export KERNEL_AGENT_BASE_URL=http://localhost:8000/v1\n"
                     "  export KERNEL_AGENT_MODEL=Qwen/Qwen3-8B")
        a.base = raw.rstrip("/")
        discover_context(a)
        print(f"endpoint {a.base}  model {a.model}  context {a.context}")
    else:
        a.context = a.context if isinstance(a.context, int) else 8192
        print("*** OFFLINE: canned replies; proves the loop, predicts nothing. ***")

    if a.levels:
        levels = [int(x) for x in a.levels.split(",") if x.strip()]
        bad = [x for x in levels if x not in ladder.LEVELS]
        if bad:
            sys.exit(f"no level(s) {bad}; have {sorted(ladder.LEVELS)}")
    else:
        levels = ([a.level] if a.level else
                  ladder.holdout_levels() + ladder.graded_levels() if (a.all and a.holdout)
                  else ladder.graded_levels() if a.all else [1])

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{stamp}-{(a.tag or ('offline' if a.offline else 'run'))}-{os.getpid()}"
    run_dir = os.path.join(a.runs_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    meta = dict(run_id=run_id, model=a.model, base=a.base if not a.offline else "offline",
                levels=levels, rounds=a.rounds, samples=a.samples, repeat=a.repeat,
                context=a.context, max_tokens=a.max_tokens, greedy=a.greedy,
                terse=a.terse, think=a.think, started=time.time(),
                args=vars(a) | {"base": a.base if not a.offline else "offline"})
    json.dump(meta, open(os.path.join(run_dir, "meta.json"), "w"), indent=1, default=str)
    trace_path = os.path.join(run_dir, "trace.jsonl")

    docs = tools.Docs(DOCS_DIR)
    mem_path = os.path.join(a.runs_dir, "memory.json")
    print(f"run {run_id}  (trace: {trace_path})")

    full = verifier.FULL_REWARD
    with open(trace_path, "a") as log:
        for rep in range(a.repeat):
            if a.repeat > 1:
                print(f"\n############ repeat {rep + 1}/{a.repeat} ############")
            results = []
            for lv in levels:
                mem = memory_mod.Memory(mem_path)
                lr = LevelRun(a, lv, log, docs, mem)
                reward, solved_round = lr.solve()
                results.append((lv, reward, solved_round))
            solved_n = sum(1 for _lv, _r, sr in results if sr is not None)
            print(f"\n--- summary {'rep ' + str(rep + 1) if a.repeat > 1 else ''}: "
                  f"solved {solved_n}/{len(results)} ---")
            for lv, reward, sr in results:
                print(f"  level {lv:>2}: reward {reward:.2f}  "
                      + (f"SOLVED on round {sr + 1}" if sr is not None else "not solved"))
            log.write(json.dumps(dict(ts=time.time(), type="run_end", repeat=rep,
                                      solved=solved_n, of=len(results),
                                      results=[[lv, r, sr] for lv, r, sr in results])) + "\n")
            log.flush()

    meta["done"] = time.time()
    json.dump(meta, open(os.path.join(run_dir, "meta.json"), "w"), indent=1, default=str)
    print(f"\ntrace at {trace_path}\nserve the dashboard:  python dashboard/server.py --runs "
          f"{a.runs_dir}")


if __name__ == "__main__":
    main()
