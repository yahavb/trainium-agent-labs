"""VeriLoop agent: the model writes Verilog, the checker grades it, the feedback goes back, repeat.

    spec.txt --> model --> Verilog --> checker.grade() --> score + feedback (A, B or C) --> model --> ...

    A1  ask the model for a design and pull the Verilog out of its reply
    A2  the loop: up to R rounds x S attempts; stop when solved; log every attempt as one JSON line
    A3  the agent's claim (SOLVED only if the checker passed it, else COULD NOT VERIFY); the model's own
        confidence vs the checker is measured separately by calibrate.py

Usage, on a seat with the model running (see SERVER.md):
    python agent.py --level levels/03_counter --feedback C --log ../results/try.jsonl
    python agent.py --level tests/sim_fixtures/counter8 --feedback B --rounds 4 --samples 2
    python agent.py --level tests/sim_fixtures/counter8 --offline      # no model: canned designs

The experiment changes ONLY the feedback level; the prompt wording, the previous design and the rest
of the loop are identical for A, B and C, so any difference in solve rate is the feedback's.
"""

import argparse
import concurrent.futures as cf
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import checker  # noqa: E402

BASE_URL = os.environ.get("VERILOOP_BASE_URL", "http://localhost:8000/v1")
MODEL = os.environ.get("VERILOOP_MODEL", "Qwen/Qwen3-8B")
CONTEXT = 8192            # the seat's server is started at 8192 tokens
MAX_ANSWER_TOKENS = 2000  # a Verilog module is short; this leaves the rest of the context for the prompt
# Sampling. Measured on seat 7: temperature 0.7 / top_p 0.8 gave 4 IDENTICAL answers out of 4 (even for an
# open-ended prompt), so "4 attempts" were 4 copies; 1.0 / 0.95 gave 3 distinct out of 4. Different
# attempts are the point of sampling, so use the higher setting.
TEMPERATURE, TOP_P = 1.0, 0.95
FORMAT = "Reply with one ```verilog code block containing only the module -- no explanation."
NO_CODE = "No Verilog came back. " + FORMAT


# ---------------------------------------------------------------- A1: prompts, the model, the code

def first_prompt(spec):
    # Short on purpose: the organisers measured that long, rule-heavy prompts make the model reason
    # instead of answer. Rules live in the checker, not here.
    return f"Write a Verilog module for this specification.\n\n{spec.strip()}\n\n{FORMAT}"


REPEAT_NOTE = ("This is the same design you sent before, and it fails in the same way. Repeating it will not "
               "help: rethink the logic and try a different approach.")


def repair_prompt(spec, design, feedback, repeated=False):
    # The repeat note is the same for feedback A, B and C: it says nothing about WHAT is wrong, only that
    # the model is going in circles (measured in the pilot: 6 rounds of the identical design).
    return (f"Specification:\n{spec.strip()}\n\n"
            f"Your previous design:\n```verilog\n{design.strip()}\n```\n\n"
            f"A checker tested it:\n{feedback.strip()}\n\n"
            + (f"{REPEAT_NOTE}\n\n" if repeated else "")
            + f"Fix the design. {FORMAT}")


def variant(prompt, i, session=""):
    """A short tag at the top of every prompt, unique per run and per attempt: `[session 3f2a-2]`.

    Measured on the seats: for code this model is nearly deterministic -- the same prompt gives the same
    design on any chip, even at temperature 1.0. Without the tag, 4 attempts in a round were often
    identical, and runs on different seats followed the very same path (seats 230, 231, 234 produced
    byte-identical designs in rounds 1-2), so "8 runs" were really 2-3. A different first line sends each
    attempt and each run down its own path. It carries no information about the answer, and it is the
    same treatment for feedback A, B and C."""
    return f"[session {session}-{i + 1}]\n{prompt}"


_FENCE = re.compile(r"```[ \t]*(?:verilog|systemverilog|sv|v)?[ \t]*\n(.*?)```", re.S | re.I)
_MODULE = re.compile(r"\bmodule\b.*?\bendmodule\b", re.S)


def extract_verilog(reply):
    """The Verilog in the reply, or "" when there is none -- never prose (prose would reach the compiler
    and be blamed on the model as a syntax error)."""
    reply = reply or ""
    reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.S)
    blocks = [b for b in _FENCE.findall(reply) if "module" in b] or _FENCE.findall(reply)
    if blocks:
        return max(blocks, key=len).strip()
    m = _MODULE.findall(reply)
    return "\n\n".join(m).strip() if m else ""


def ask(prompt, base_url=BASE_URL, model=MODEL, temperature=TEMPERATURE, timeout=300):
    """One chat call to the OpenAI-compatible server. Returns (text, info) where info has token counts,
    finish reason and seconds -- the token counts are the token instrumentation the rubric asks for."""
    est_prompt = len(prompt) // 3
    budget = max(256, min(MAX_ANSWER_TOKENS, CONTEXT - est_prompt - 64))
    # NEVER add `seed` here: on the seats' Neuron vLLM 0.24 a request with `seed` killed the whole model
    # server (measured on seat 7, 2026-10-10). Restart with ./serve.sh if it happens.
    body = dict(model=model, messages=[{"role": "user", "content": prompt}], max_tokens=budget,
                temperature=temperature, top_p=TOP_P,
                # Thinking OFF: measured by the organisers on this model, thinking made rounds 55x slower
                # and the answers came back empty or cut off.
                chat_template_kwargs={"enable_thinking": False})
    req = urllib.request.Request(f"{base_url.rstrip('/')}/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload = json.load(r)
    except urllib.error.URLError as e:
        raise SystemExit(f"cannot reach the model at {base_url} ({e}). Is ./serve.sh running on this seat? "
                         f"See SERVER.md.") from e
    choice = payload["choices"][0]
    usage = payload.get("usage", {})
    info = dict(prompt_tokens=usage.get("prompt_tokens"), completion_tokens=usage.get("completion_tokens"),
                finish=choice.get("finish_reason"), seconds=round(time.time() - t0, 1))
    if info["finish"] == "length":
        print(f"    (an answer was cut off at the token limit -- a compile error on it is the budget, "
              f"not the model)")
    return choice.get("message", {}).get("content") or "", info


# ---------------------------------------------------------------- offline: test the loop without a model

def offline_replies(level_dir, n, rnd):
    """Canned replies from the level folder: broken designs first, then good.v from round 2 on. Only for
    checking the loop works; never report a number from it."""
    bad = sorted(f for f in os.listdir(level_dir) if f.startswith("bad_") and f.endswith(".v"))
    pool = [open(os.path.join(level_dir, f)).read() for f in bad]
    good = open(os.path.join(level_dir, "good.v")).read()
    out = []
    for i in range(n):
        code = good if rnd >= 2 and i == 0 else pool[(rnd * n + i) % len(pool)] if pool else good
        out.append((f"Here you go:\n```verilog\n{code}\n```", dict(prompt_tokens=0, completion_tokens=0,
                                                                   finish="offline", seconds=0.0)))
    return out


# ---------------------------------------------------------------- A2: the loop

def run_level(level_dir, feedback="C", rounds=8, samples=4, log_path=None, offline=False,
              run_id=None, base_url=BASE_URL, model=MODEL, quiet=False):
    """One run on one level. Returns a summary dict; appends every attempt to `log_path` (JSONL)."""
    ref = checker.load_level(level_dir)
    spec = open(os.path.join(level_dir, "spec.txt")).read()
    level = os.path.basename(os.path.normpath(level_dir))
    run_id = run_id or time.strftime("%H%M%S") + f"-{level}-{feedback}"
    say = (lambda *a: None) if quiet else print
    log = open(log_path, "a") if log_path else None

    session = __import__("hashlib").md5(run_id.encode()).hexdigest()[:4]   # unique per run, stable within it
    say(f"=== {level}   feedback {feedback}   {rounds} rounds x {samples} attempts   run {run_id}")
    prev_design, prev_feedback, best = None, None, None
    seen, sent, prev_repeated = set(), set(), False
    totals = dict(attempts=0, prompt_tokens=0, completion_tokens=0, seconds=0.0)
    for rnd in range(rounds):
        prompt = (first_prompt(spec) if prev_design is None
                  else repair_prompt(spec, prev_design, prev_feedback, repeated=prev_repeated))
        if offline:
            replies = offline_replies(level_dir, samples, rnd)
        else:
            with cf.ThreadPoolExecutor(max_workers=samples) as ex:   # the server runs 4 sequences at once
                replies = list(ex.map(lambda i: ask(variant(prompt, i, session), base_url, model), range(samples)))

        graded = []
        totals["seconds"] += max(r[1]["seconds"] for r in replies)   # attempts in a round run in parallel
        for i, (reply, info) in enumerate(replies):
            totals["attempts"] += 1
            totals["prompt_tokens"] += info.get("prompt_tokens") or 0
            totals["completion_tokens"] += info.get("completion_tokens") or 0
            code = extract_verilog(reply)
            if code:
                g = checker.grade(ref, code)
                score, stage, solved, shown = g.score, g.stage, g.solved, g.feedback[feedback]
            else:
                score, stage, solved, shown = 0.0, "no-code", False, NO_CODE
            repeat = code in seen
            seen.add(code)
            graded.append((score, solved, code, shown, stage))
            if log:
                log.write(json.dumps(dict(
                    run=run_id, level=level, feedback=feedback, round=rnd, attempt=i, model=model,
                    offline=offline, score=score, solved=solved, stage=stage, repeat=repeat,
                    feedback_shown=shown, verilog=code, prompt_chars=len(prompt), reply_chars=len(reply),
                    **info)) + "\n")
                log.flush()

        scores = [g[0] for g in graded]
        # Best score; among equals prefer a design not already sent back, so the loop can leave a rut.
        round_best = max(graded, key=lambda g: (g[0], g[2] not in sent))
        if best is None or round_best[0] > best[0]:
            best = round_best
        say(f"round {rnd}: scores {[round(s, 2) for s in scores]}  best so far {best[0]:.2f}  "
            f"({max(r[1]['seconds'] for r in replies):.1f}s)")
        if round_best[1]:
            say(f"SOLVED on round {rnd}")
            break
        # Repair from this round's best, not the all-time best: a single bad round must not freeze the loop
        # (the organisers' kernel agent measured exactly that failure).
        prev_design, prev_feedback = round_best[2] or (prev_design or ""), round_best[3]
        prev_repeated = prev_design in sent
        sent.add(prev_design)
        say("  feedback sent: " + prev_feedback.replace("\n", "\n                 ")[:600])
    else:
        rnd = rounds - 1
        say(f"not solved in {rounds} rounds; best score {best[0]:.2f}")

    if log:
        log.close()
    # The agent's claim comes only from the checker: it cannot report success the checker did not grant.
    solved = bool(best and best[1])
    claim = (f"SOLVED -- verified by the checker on every test" if solved else
             f"COULD NOT VERIFY -- best design passes {best[0]:.2f} of the score; not solved" if best else
             "COULD NOT VERIFY -- no design came back")
    say(claim)
    return dict(run=run_id, level=level, feedback=feedback, solved=solved, claim=claim,
                rounds_used=rnd + 1, best_score=best[0] if best else 0.0, offline=offline,
                distinct_designs=len(seen - {""}), **{k: round(v, 1) for k, v in totals.items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--level", required=True, help="a level folder, e.g. levels/03_counter")
    ap.add_argument("--feedback", choices=["A", "B", "C", "D"], default="C")
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--samples", type=int, default=4, help="attempts per round (the server runs 4 at once)")
    ap.add_argument("--log", help="append every attempt to this JSONL file")
    ap.add_argument("--offline", action="store_true", help="canned designs instead of the model (testing only)")
    ap.add_argument("--base-url", default=BASE_URL)
    ap.add_argument("--model", default=MODEL)
    a = ap.parse_args()
    s = run_level(a.level, a.feedback, a.rounds, a.samples, a.log, a.offline, base_url=a.base_url, model=a.model)
    print(json.dumps(s))
    return 0 if s["solved"] else 1


if __name__ == "__main__":
    sys.exit(main())
