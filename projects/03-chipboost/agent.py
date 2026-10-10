#!/usr/bin/env python3
"""
agent.py -- CHIPBOOST's loop and its three arms, on one budget. Owner: P3.

    --arm referee        Qwen3 + the referee's ONE named change each round        (arm a)
    --arm model_alone    Qwen3, told only "Make it faster." (plus its time, once   (arm b)
                         P1 can time); it never sees the referee's messages
    --arm random_search  no model: candidates from P2's search.py                 (arm c)

    controller  start kernel + budget                    this file
    generator   Qwen3-8B on this seat                    KERNEL_AGENT_BASE_URL (the pod sets it)
    referee     speedcheck.check (P1) if it exists,      dev shapes, then held-out shapes
                else redteam/stage12.check (simulator only, TEMPORARY)
    log         one schema.py line per attempt           attempts.jsonl

    python agent.py --arm referee --budget 8 --repeat 3 --give-up-after 0     # in the seat pod
    python agent.py --offline                                                 # no model; simulator grades
    python agent.py --dry --arm model_alone                                   # no model, rules only, anywhere
    python schema.py --check attempts.jsonl

FAIRNESS. Every arm: same start kernel, same referee, same --budget, where one referee evaluation = one
attempt = one unit whatever its verdict. For a comparison pass --give-up-after 0, so no arm stops early
and every arm spends exactly its budget; the run checks that it did.

"BETTER" UNTIL P1 MERGES. stage12 has no timer, so a correct kernel gets verdict None ("passed,
untimed"). Then the measure is the simulator's HBM traffic over the byte floor on the largest dev shape:
correct on every shape AND fewer bytes than the start kernel. It is printed per arm, labelled sim. With
speedcheck.py present the same runs report chip speedups instead.

Reused from projects/02-kernel-agent/agent.py: extract_code, enrich, API_CARD, and the loop's mechanics
(samples per round, repair the LATEST attempt, a ledger when a failure repeats, give-up, --repeat).
"""

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import random
import socket
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
AGENT02_DIR = os.path.join(os.path.dirname(HERE), "02-kernel-agent")
sys.path.insert(0, AGENT02_DIR)
sys.path.insert(0, os.path.join(HERE, "redteam"))
sys.path.insert(0, HERE)

import diagnose  # noqa: E402
import nkibench  # noqa: E402
import schema    # noqa: E402
import stage12   # noqa: E402


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# 02's agent.py, under another name: `import agent` here would import this file.
agent02 = _load_module("agent02", os.path.join(AGENT02_DIR, "agent.py"))

OP = "matmul"
ENTRY = nkibench.LEVELS[stage12.LEVEL]["entry"]   # nki_matmul_tiled_
P2_START = os.path.join(HERE, "kernels", "matmul_start.py")
P2_SEARCH = os.path.join(HERE, "search.py")
FALLBACK_START = os.path.join(AGENT02_DIR, "reference_level4.py")
MAKE_FASTER = "Make it faster."

# Best first. None = correct on every shape the referee ran, but untimed (stage12 has no timer).
RANK = {"faster": 6, "no_gain": 5, None: 4, "slower": 3, "heldout_fail": 2, "wrong": 1, "rules": 0}
REJECTED = ("rules", "wrong", "heldout_fail")


# ---------------------------------------------------------------- the referee

def pick_referee():
    if os.path.exists(os.path.join(HERE, "speedcheck.py")):
        import speedcheck
        return "speedcheck", speedcheck.check
    return "stage12 (TEMPORARY, simulator only)", stage12.check


def byte_instruction(waste, lbl):
    """ONE named change from the measured HBM traffic / byte floor, naming WHERE the re-reading is.

    Measured on seat-101: the borrowed level-5 text "hoist the operand loads out of the innermost loop"
    sent Qwen3 to the k loop, which has no waste (every pass loads a different, needed tile); it tried to
    load all of K into one 128-row tile and crashed, 8 attempts out of 8. The re-reading in the tiled
    kernel is across the OUTER loops: at K=256 M=512 N=1024 each rhs tile is loaded once per m (4x) and
    each lhsT tile once per n (2x), 2.00x the floor in total. Reusing rhs across m alone reaches 1.14x;
    reusing lhsT alone only 1.86x, which would earn the same hint again. So rhs comes first.
    """
    head = f"CORRECT, BUT MOVING {waste:.2f}x THE MINIMUM HBM BYTES on {lbl} (simulator count)."
    if waste > 1.25:
        return (f"{head} The k loop is not the waste: each of its passes loads a different tile and all are "
                f"needed. The waste is that the same rhs tiles are loaded again for every m. Make n the outer "
                f"loop, load that n's K // 128 rhs tiles into SBUF once before the m loop, and reuse them for "
                f"every m. Keep loading the lhsT tiles inside the m loop as now.")
    if waste > 1.05:
        return (f"{head} What is still re-read is lhsT: each lhsT tile is loaded once per n. Keep the rhs "
                f"reuse you have, and also keep the lhsT tiles of a block of m tiles in SBUF while you sweep "
                f"across n, so each lhsT tile is loaded once.")
    return (f"{head} That is the byte floor: every byte is read once, so no byte reduction is left. Any "
            f"further speedup has to come from the engine schedule.")


def measure_waste(path):
    """Simulator HBM bytes / byte floor on the largest dev shape, where redundant traffic shows."""
    kernel = nkibench.load_kernel(path, ENTRY)
    case = max(stage12.SHAPES["dev"], key=lambda c: c["M"] * c["K"] * c["N"])
    args, _ = nkibench.make_inputs(case, stage12.LEVEL)
    want = nkibench.LEVELS[stage12.LEVEL]["ref"](*args)
    _, counted = nkibench.simulate_and_count(kernel, args)
    return counted["bytes"] / nkibench.minimum_hbm_bytes(args, want), nkibench.label(case, stage12.LEVEL)


def grade(src, referee, dry, path):
    """Dev shapes, then held-out shapes if dev passed. Returns (referee schema fields, waste or None)."""
    with open(path, "w") as f:
        f.write(src)
    if dry:
        return stage12.run(path, "dev", rules_only=True)[0], None
    name, check = referee
    waste = None
    r = check(path, op=OP, shapes="dev")
    if r.get("verdict") not in ("rules", "wrong"):
        h = check(path, op=OP, shapes="heldout")
        if h.get("verdict") in REJECTED:
            r = dict(h, verdict="heldout_fail")
        elif name.startswith("stage12") and r.get("verdict") is None:
            waste, lbl = measure_waste(path)
            r = dict(r, instruction_given=byte_instruction(waste, lbl))
        # else: the dev result carries the timing
    if r.get("verdict") in ("wrong", "heldout_fail"):
        # The referee says what is wrong; name the change when the output matches a known mistake.
        # referee_message keeps the referee's words, so the log shows both.
        named = diagnose.diagnose(path)
        if named:
            r = dict(r, instruction_given=named)
    return r, waste


# ---------------------------------------------------------------- the model

def ask(a, prompt):
    """02's ask(), plus the token count it throws away: prompt_tokens is logged, so it must be the
    endpoint's number, not an estimate. Same request: thinking off, answer budget capped to the context."""
    import httpx
    budget = min(a.max_tokens, max(256, a.context - len(prompt) // 4 - 64))
    body = dict(model=a.model, messages=[{"role": "user", "content": prompt}],
                max_tokens=budget, temperature=0.6, top_p=0.95,
                chat_template_kwargs={"enable_thinking": a.think})
    r = httpx.post(f"{a.base.rstrip('/')}/chat/completions", json=body, timeout=900, verify=False)
    if r.status_code != 200:
        raise SystemExit(f"the endpoint returned HTTP {r.status_code}:\n{r.text[:600]}")
    payload = r.json()
    ch = payload["choices"][0]
    if ch.get("finish_reason") == "length":
        print("    (TRUNCATED: finish_reason=length, so a parse error below is the budget, not the model)")
    return (ch.get("message", {}).get("content") or "",
            (payload.get("usage") or {}).get("prompt_tokens"))


def ask_parallel(a, prompt, n):
    import concurrent.futures as cf
    with cf.ThreadPoolExecutor(max_workers=n) as ex:
        return [f.result() for f in [ex.submit(ask, a, prompt) for _ in range(n)]]


def offline_answers(start_src, n, rnd):
    """No model: a broken start kernel first (no @nki.jit), then the start kernel itself, so the loop,
    the referee path and the log can be exercised. Never report these."""
    src = start_src.replace("@nki.jit", "", 1) if rnd == 0 else start_src
    return [(f"```python\n{src}\n```", None)] * n


def load_search(path):
    """Arm (c) is P2's. The contract asked of P2: search.sample(rng) -> kernel source (str), where rng is a
    random.Random. Without it the arm stops rather than invent candidates."""
    if not os.path.exists(path):
        sys.exit(f"--arm random_search needs P2's search.py at {path}, providing sample(rng) -> source. "
                 f"It is not there yet.")
    mod = _load_module("chipboost_search", path)
    if not callable(getattr(mod, "sample", None)):
        sys.exit(f"{path} has no sample(rng) function. Arm (c) calls search.sample(rng) -> kernel source.")
    return mod


# ---------------------------------------------------------------- prompts

TASK = ("This AWS Neuron NKI kernel computes a matrix multiplication: result = lhsT.T @ rhs, with lhsT of "
        "shape [K, M] and rhs of shape [K, N]. It must stay correct for every K and M that are multiples "
        "of 128 and every N that is a multiple of 512, including sizes you are not shown.")
KEEP = (f"Keep the function name {ENTRY}(lhsT, rhs), the @nki.jit decorator, and the imports nki, "
        f"nki.language as nl and nki.isa as nisa. Reply with ONE python code block.")


def strip_module_docstring(src):
    """The start kernel's module docstring is about Project 2's ladder ("This is the ANSWER ... level 4").
    Measured: Qwen3 copied it back verbatim. It costs tokens and says nothing about this task."""
    try:
        first = ast.parse(src).body[0]
    except (SyntaxError, IndexError):
        return src
    if isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant) \
            and isinstance(first.value.value, str):
        lines = src.splitlines(keepends=True)
        return "".join(lines[:first.lineno - 1] + lines[first.end_lineno:]).lstrip("\n")
    return src


def first_prompt(src, status):
    return (f"{TASK} Make it run faster on the Trainium chip.\n\n```python\n{src}\n```\n\n"
            f"The referee's report on this kernel: {status}\n\n{agent02.API_CARD}\n{KEEP}")


def repair_prompt(src, instruction):
    """The code plus ONE instruction from the referee. Never a corrected kernel."""
    return (f"{TASK}\n\n```python\n{src}\n```\n\nThe referee says: {instruction}\n\n"
            f"Make exactly that change and keep everything else identical. {KEEP}")


def alone_prompt(src, time_us, first):
    """Arm (b): the same task and code, no referee. The API card goes in the first prompt only, as in
    arm (a), so both model arms get the same documentation."""
    t = f"It currently takes {time_us:.1f} microseconds on the chip.\n\n" if time_us else ""
    return (f"{TASK}\n\n```python\n{src}\n```\n\n{t}{MAKE_FASTER}\n\n"
            + (f"{agent02.API_CARD}\n" if first else "") + KEEP)


# ---------------------------------------------------------------- the loop

def better(rec, waste, best):
    """Is this correct attempt better than the best so far? Chip time first; else simulator bytes."""
    if rec.get("time_us_median") is not None and best["time"] is not None:
        return rec["time_us_median"] < best["time"]
    return waste is not None and best["waste"] is not None and waste < best["waste"] - 1e-9


def run_once(a, referee, start_path, rep, log, workdir, search):
    model_arm = a.arm != "random_search"
    tag = "offline-" if (a.offline or a.dry) else ""
    run_id = f"{OP}-{a.arm}-{tag}{time.strftime('%H%M%S')}-{rep}"
    start_src = open(start_path).read()
    start, start_waste = grade(start_src, referee, a.dry, os.path.join(workdir, f"{run_id}_start.py"))
    status = start.get("instruction_given") or start.get("referee_message") or ""
    print(f"\n=========== run {run_id} ===========")
    print(f"start kernel {os.path.relpath(start_path, HERE)}: verdict {start.get('verdict') or 'PASS (untimed)'}"
          + (f", {start_waste:.2f}x the byte floor (sim)" if start_waste else ""))
    out = dict(run_id=run_id, arm=a.arm, attempts=0, verdicts={}, correct=0, start_waste=start_waste,
               best_waste=start_waste, improved=False, best_speedup=None)
    if start.get("verdict") in REJECTED:
        print(f"  THE START KERNEL IS REJECTED BY THE REFEREE, so there is nothing to speed up:\n  {status}")
        return out

    shown = strip_module_docstring(start_src)
    # The best CORRECT kernel so far: model_alone always builds on it (it gets no other signal).
    best = dict(src=shown, time=start.get("time_us_median"), waste=start_waste)
    prompt = (first_prompt(shown, status) if a.arm == "referee"
              else alone_prompt(shown, best["time"], first=True) if a.arm == "model_alone" else None)
    latest = (shown, status)
    rng = random.Random(1000 + rep)
    tried, seen, streak = [], {}, 0
    rounds = a.rounds or -(-a.budget // a.samples)
    for rnd in range(rounds):
        n = min(a.samples, a.budget - out["attempts"])
        if n <= 0:
            break
        t0 = time.perf_counter()
        # Generation finishes for the whole round BEFORE the referee runs, so vLLM is idle while timing.
        if not model_arm:
            replies = [(search.sample(rng), None) for _ in range(n)]
        elif a.offline or a.dry:
            replies = offline_answers(start_src, n, rnd)
        else:
            replies = ask_parallel(a, prompt, n)
        graded = []
        for reply, prompt_tokens in replies:
            src = agent02.extract_code(reply) if model_arm else reply
            out["attempts"] += 1
            r, waste = grade(src, referee, a.dry,
                             os.path.join(workdir, f"{run_id}_{out['attempts']:04d}.py"))
            referee_says = agent02.enrich(r.get("instruction_given") or r.get("referee_message") or "")
            instruction = {"referee": referee_says, "model_alone": MAKE_FASTER}.get(a.arm)
            rec = {k: None for k in schema.ATTEMPT_FIELDS}
            rec.update({k: r.get(k) for k in stage12.REFEREE_FIELDS if k in r})
            rec.update(seat=a.seat, kernel=OP, arm=a.arm, run_id=run_id, attempt_no=out["attempts"],
                       round=rnd, prompt_tokens=prompt_tokens,
                       code_hash=hashlib.sha1(src.encode()).hexdigest(), code=src,
                       prompt=prompt, response=reply if model_arm else None,
                       instruction_given=instruction, timestamp=time.time())
            problems = schema.validate(rec)
            if problems:
                raise SystemExit(f"BUG: this log line breaks the shared schema: {problems}")
            log.write(json.dumps(rec) + "\n")
            out["verdicts"][rec["verdict"]] = out["verdicts"].get(rec["verdict"], 0) + 1
            if rec["verdict"] not in REJECTED:
                out["correct"] += 1
                if better(rec, waste, best):
                    best.update(src=src, time=rec.get("time_us_median"), waste=waste)
                    out["improved"] = True
                    out["best_waste"] = waste if waste is not None else out["best_waste"]
                if rec.get("speedup") and (out["best_speedup"] is None or rec["speedup"] > out["best_speedup"]):
                    out["best_speedup"] = rec["speedup"]
            graded.append((RANK[rec["verdict"]], rec.get("speedup") or 0.0, src, referee_says, rec, waste))
        log.flush()

        graded.sort(key=lambda g: (g[0], g[1]), reverse=True)
        _, speedup, src, referee_says, rec, waste = graded[0]
        print(f"round {rnd}: best this round {rec['verdict'] or 'PASS (untimed)'}"
              + (f", {waste:.2f}x the byte floor (sim)" if waste else "")
              + (f", speedup {speedup:.3f} ({rec['source']})" if rec.get("speedup") else "")
              + f"  [{out['attempts']}/{a.budget} evaluations, {time.perf_counter() - t0:.1f}s]")

        if a.arm == "model_alone":
            prompt = alone_prompt(best["src"], best["time"], first=False)
            continue
        if a.arm == "random_search":
            continue

        # Arm (a): repair the latest attempt with the referee's one instruction; 02's anti-cycling.
        if src.strip():
            latest = (src, referee_says)
        same = referee_says == (tried[-1] if tried else None)
        if not same:
            print(f"  {referee_says[:300]}")
        seen[referee_says] = seen.get(referee_says, 0) + 1
        streak = streak + 1 if same else 1
        if a.give_up_after and seen[referee_says] >= a.give_up_after:
            print(f"  STOPPING: the same instruction {seen[referee_says]} times; more rounds will not help.")
            break
        tried.append(referee_says)
        prompt = repair_prompt(*latest)
        if streak >= 2:
            ledger = "\n".join(f"- {t[:160]}" for t in dict.fromkeys(tried))
            prompt += f"\n\nThese have already been tried and did not work, so do something different:\n{ledger}"
            print(f"  same instruction {streak}x -- adding a ledger of {len(set(tried))} earlier ones")
    return out


def seat_from_hostname():
    h = socket.gethostname()
    return int(h.split("-")[1]) if h.startswith("seat-") and h.split("-")[1].isdigit() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="referee", choices=schema.ARMS)
    ap.add_argument("--budget", type=int, default=8, help="referee evaluations per run, every arm")
    # Measured on seat-101: the samples of one round came back identical every time, so a second sample
    # spent budget on a repeat. One sample per round: every evaluation is a new attempt.
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=None, help="default: enough to spend the budget")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--give-up-after", type=int, default=4,
                    help="arm referee only; 0 = never, which a fair three-arm comparison needs")
    ap.add_argument("--start", default=None, help="start kernel; default P2's kernels/matmul_start.py "
                                                  "if it exists, else reference_level4.py")
    ap.add_argument("--search", default=P2_SEARCH, help="arm random_search: P2's search.py")
    ap.add_argument("--seat", type=int, default=seat_from_hostname())
    ap.add_argument("--max-tokens", type=int, default=agent02.MIN_ANSWER_TOKENS)
    ap.add_argument("--context", type=int, default=8192, help="the server's max-model-len")
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--model", default=agent02.MODEL)
    ap.add_argument("--base", default=os.environ.get("KERNEL_AGENT_BASE_URL"))
    ap.add_argument("--log", default=os.path.join(HERE, "attempts.jsonl"))
    ap.add_argument("--offline", action="store_true", help="no model; the simulator grades (needs nki)")
    ap.add_argument("--dry", action="store_true", help="no model, rules stage only (no nki needed)")
    a = ap.parse_args()

    start_path = a.start or (P2_START if os.path.exists(P2_START) else FALLBACK_START)
    search = load_search(a.search) if a.arm == "random_search" else None
    referee = pick_referee()
    if a.arm == "random_search":
        print(f"arm random_search: candidates from {a.search}")
    elif a.offline or a.dry:
        print("*** OFFLINE: replaying the start kernel, no model. Numbers are meaningless. ***")
    else:
        if not (a.base or "").strip():
            sys.exit("KERNEL_AGENT_BASE_URL is unset. The seat pods set it; or pass --offline / --dry.")
        print(f"endpoint {a.base}  model {a.model}")
    print(f"referee: {'stage12 rules only (--dry)' if a.dry else referee[0]}")
    print(f"start:   {os.path.relpath(start_path, HERE)}   arm: {a.arm}   budget: {a.budget}   "
          f"give-up: {a.give_up_after or 'never'}   log: {a.log}")

    results = []
    workdir = tempfile.mkdtemp(prefix="chipboost_")
    try:
        with open(a.log, "a") as log:
            for rep in range(a.repeat):
                results.append(run_once(a, referee, start_path, rep, log, workdir, search))
    except nkibench.NkiMissing as e:
        sys.exit(f"{e}\nThe simulator needs the seat pod. Use --dry here.")

    print(f"\n=========== summary: arm {a.arm}, budget {a.budget} ===========")
    for r in results:
        counts = ", ".join(f"{k or 'pass-untimed'}={v}" for k, v in sorted(r["verdicts"].items(),
                                                                           key=lambda kv: str(kv[0])))
        gain = (f"best speedup {r['best_speedup']:.3f} (chip)" if r["best_speedup"]
                else f"best {r['best_waste']:.2f}x the byte floor vs start {r['start_waste']:.2f}x (sim)"
                if r["improved"] and r["best_waste"] else "no improvement over the start kernel")
        spent = "" if r["attempts"] == a.budget else f"  (spent {r['attempts']} of {a.budget})"
        print(f"  {r['run_id']}: {r['attempts']} evaluations{spent}; {counts}; {r['correct']} correct; {gain}")
    if a.repeat > 1:
        n_imp = sum(1 for r in results if r["improved"])
        print(f"\n  over {a.repeat} runs: {n_imp}/{a.repeat} found a correct kernel better than the start; "
              f"correct attempts per run {[r['correct'] for r in results]}. Report the rate and the spread.")
    if a.give_up_after == 0 and any(r["attempts"] != a.budget for r in results):
        print("  WARNING: a run did not spend its full budget, so this arm is not comparable as is.")
    print(f"\nattempts logged to {a.log}")


if __name__ == "__main__":
    main()
