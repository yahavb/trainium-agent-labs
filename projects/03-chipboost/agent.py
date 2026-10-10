#!/usr/bin/env python3
"""
agent.py -- CHIPBOOST's loop and its three arms, on one budget. Owner: P3.

    --arm referee        Qwen3 + the referee's ONE named change each round        (arm a)
    --arm model_alone    Qwen3, told only "Make it faster." (plus its time, once   (arm b)
                         P1 can time); it never sees the referee's messages
    arm c (random_search) is not run from here: P2's search.py runs it and logs arm=random_search to the
                         same seat log, e.g.  python search.py --budget 8 --seed 0

    controller  start kernel + budget                    this file
    generator   Qwen3-8B on this seat                    KERNEL_AGENT_BASE_URL (the pod sets it)
    referee     speedcheck.check_isolated (P1)            chip correctness + timing + held-out
                else redteam/stage12.check (simulator only, TEMPORARY)
    log         one schema.py line per attempt           logs/seat-<N>/attempts.jsonl

    python agent.py --arm referee --budget 8 --repeat 3 --give-up-after 0     # in the seat pod
    python agent.py --offline                                                 # no model; simulator grades
    python agent.py --dry --arm model_alone                                   # no model, rules only, anywhere
    python schema.py --check attempts.jsonl

FAIRNESS. The two model arms: same start kernel, same referee, same --budget, where one referee evaluation
= one attempt = one unit whatever its verdict. --give-up-after defaults to 0, so no arm stops early and
every arm spends exactly its budget; the run checks that it did. Arm c (search.py) is NOT like for like: it
tunes the expert kernel's blocking, starts from the expert (already ~2.5x the start kernel), and counts
that as its attempt 0. Say so wherever the three arms are compared.

REFEREE. P1's speedcheck.check_isolated whenever speedcheck.py is present (REFEREE.md): sandboxed, on the
chip, held-out shapes included; a None (referee failure) is retried and never counted. The model is sent
only instruction_given, never referee_message. "Better" = a higher chip speedup; the run's success is a
`faster` verdict. Without speedcheck.py the TEMPORARY stage12 fallback grades in the simulator, untimed,
and "better" = fewer simulator HBM bytes than the start kernel (labelled sim).

Reused from projects/02-kernel-agent/agent.py: extract_code, enrich, API_CARD, and the loop's mechanics
(samples per round, repair the LATEST attempt, a ledger when a failure repeats, give-up, --repeat).
"""

import argparse
import ast
import hashlib
import importlib.util
import json
import os
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
FALLBACK_START = os.path.join(AGENT02_DIR, "reference_level4.py")
MAKE_FASTER = "Make it faster."

# Best first. None = correct on every shape the referee ran, but untimed (stage12 has no timer).
RANK = {"faster": 6, "no_gain": 5, None: 4, "slower": 3, "heldout_fail": 2, "wrong": 1, "rules": 0}
REJECTED = ("rules", "wrong", "heldout_fail")


# ---------------------------------------------------------------- the referee

def pick_referee():
    """(name, speedcheck module or None). Importing speedcheck does not take a core."""
    if os.path.exists(os.path.join(HERE, "speedcheck.py")):
        import speedcheck
        return "speedcheck", speedcheck
    return "stage12 (TEMPORARY, simulator only)", None


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
    """Returns (referee record, waste or None), or (None, None) when the REFEREE failed.

    speedcheck (P1, REFEREE.md): check_isolated, one fresh sandboxed process per candidate; it runs the
    held-out shapes itself. It returns None when the referee failed (no free core, a referee crash): retry,
    and if it stays down, skip -- never log that as the kernel's verdict or count it against the budget.
    Candidates are written OUTSIDE projects/ (path is in /tmp), or the referee reports itself modified.
    diagnose.py is NOT used with speedcheck: it imports and simulates the model's code in this process,
    unsandboxed, and speedcheck's own instruction_given covers every verdict.

    stage12 (fallback): dev shapes, then held-out, then diagnose.py's named mistake for a wrong answer.
    """
    with open(path, "w") as f:
        f.write(src)
    if dry:
        return stage12.run(path, "dev", rules_only=True)[0], None
    name, sc = referee
    if sc is not None:
        for _ in range(3):
            r = sc.check_isolated(path, op=OP)
            if r is not None:
                return r, None
            print("    (the REFEREE failed -- not a verdict on the kernel; retrying in 20 s)")
            time.sleep(20)
        return None, None
    waste = None
    check = stage12.check
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
    # One vLLM hiccup must not end a run that has spent an hour of budget: retry a server error, a 429,
    # a timeout or a dropped connection. A 4xx is the request itself (e.g. the prompt is too long), so stop.
    for attempt, wait in enumerate((10, 30, 60, None)):
        try:
            r = httpx.post(f"{a.base.rstrip('/')}/chat/completions", json=body, timeout=900, verify=False)
        except httpx.HTTPError as e:
            err = f"{type(e).__name__}: {e}"
        else:
            if r.status_code == 200:
                break
            err = f"HTTP {r.status_code}: {r.text[:600]}"
            if r.status_code < 500 and r.status_code != 429:
                raise SystemExit(f"the endpoint rejected the request, {err}")
        if wait is None:
            raise SystemExit(f"the endpoint failed 4 times; last error {err}")
        print(f"    (the model endpoint failed, {err[:200]}; retry {attempt + 1}/3 in {wait} s)")
        time.sleep(wait)
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

def says(r, referee):
    """What the model may be told. speedcheck: instruction_given ONLY -- referee_message can quote the
    kernel's own exception text, which is attacker-controlled, and names held-out shapes (REFEREE.md
    section 5). stage12's messages are our own nkibench text, so they may fall back to the message."""
    if referee[1] is not None:
        return r.get("instruction_given") or ""
    return agent02.enrich(r.get("instruction_given") or r.get("referee_message") or "")


def better(rec, waste, best):
    """Is this correct attempt better than the best so far? On the chip, compare speedup (each one is
    against the start kernel timed in the same session, REFEREE.md); else simulator bytes."""
    if rec.get("speedup") is not None:
        return best["speedup"] is None or rec["speedup"] > best["speedup"]
    return waste is not None and best["waste"] is not None and waste < best["waste"] - 1e-9


def run_once(a, referee, start_path, rep, log, workdir):
    tag = "offline-" if (a.offline or a.dry) else ""
    run_id = f"{OP}-{a.arm}-{tag}{time.strftime('%H%M%S')}-{rep}"
    start_src = open(start_path).read()
    start, start_waste = grade(start_src, referee, a.dry, os.path.join(workdir, f"{run_id}_start.py"))
    if start is None:
        sys.exit("the referee failed 3 times on the START kernel (no free core? see REFEREE.md section 7). "
                 "Nothing was logged.")
    status = says(start, referee)
    print(f"\n=========== run {run_id} ===========")
    print(f"start kernel {os.path.relpath(start_path, HERE)}: verdict {start.get('verdict') or 'PASS (untimed)'}"
          + (f", {start['time_us_median']:.1f} us (chip)" if start.get("time_us_median") else "")
          + (f", {start_waste:.2f}x the byte floor (sim)" if start_waste else ""))
    out = dict(run_id=run_id, arm=a.arm, attempts=0, verdicts={}, correct=0, start_waste=start_waste,
               best_waste=start_waste, improved=False, best_speedup=None)
    if start.get("verdict") in REJECTED:
        print(f"  THE START KERNEL IS REJECTED BY THE REFEREE, so there is nothing to speed up:\n  {status}")
        return out

    shown = strip_module_docstring(start_src)
    # The best CORRECT kernel so far: model_alone always builds on it (it gets no other signal).
    best = dict(src=shown, time=start.get("time_us_median"), speedup=start.get("speedup"), waste=start_waste)
    prompt = (first_prompt(shown, status) if a.arm == "referee"
              else alone_prompt(shown, best["time"], first=True))
    latest = (shown, status)
    tried, seen, streak = [], {}, 0
    rounds = a.rounds or -(-a.budget // a.samples)
    rnd = -1
    # Referee failures are not counted, so allow a few extra rounds to still spend the whole budget.
    while out["attempts"] < a.budget and rnd + 1 < 2 * rounds:
        rnd += 1
        n = min(a.samples, a.budget - out["attempts"])
        t0 = time.perf_counter()
        # Generation finishes for the whole round BEFORE the referee runs, so vLLM is idle while timing.
        if a.offline or a.dry:
            replies = offline_answers(start_src, n, rnd)
        else:
            replies = ask_parallel(a, prompt, n)
        graded = []
        for reply, prompt_tokens in replies:
            src = agent02.extract_code(reply)
            r, waste = grade(src, referee, a.dry,
                             os.path.join(workdir, f"{run_id}_{out['attempts'] + 1:04d}_{rnd}.py"))
            if r is None:
                print("    skipped: the referee stayed down. Not logged, not counted against the budget.")
                continue
            out["attempts"] += 1
            referee_says = says(r, referee)
            instruction = {"referee": referee_says, "model_alone": MAKE_FASTER}.get(a.arm)
            rec = {k: None for k in schema.ATTEMPT_FIELDS}
            rec.update({k: v for k, v in r.items() if k in schema.ATTEMPT_FIELDS})   # the referee's fields
            rec.update(kernel=OP, arm=a.arm, run_id=run_id, attempt_no=out["attempts"],
                       round=rnd, prompt_tokens=prompt_tokens, code=src,
                       prompt=prompt, response=reply, instruction_given=instruction)
            rec["seat"] = a.seat if a.seat is not None else rec["seat"]
            rec["code_hash"] = rec["code_hash"] or hashlib.sha1(src.encode()).hexdigest()[:12]
            rec["timestamp"] = rec["timestamp"] or time.time()
            problems = schema.validate(rec)
            if problems:
                raise SystemExit(f"BUG: this log line breaks the shared schema: {problems}")
            log.write(json.dumps(rec) + "\n")
            out["verdicts"][rec["verdict"]] = out["verdicts"].get(rec["verdict"], 0) + 1
            if rec["verdict"] not in REJECTED:
                out["correct"] += 1
                if better(rec, waste, best):
                    best.update(src=src, time=rec.get("time_us_median"), speedup=rec.get("speedup"),
                                waste=waste)
                    if waste is not None:
                        out["improved"], out["best_waste"] = True, waste
                if rec["verdict"] == "faster":   # beyond the noise threshold, held-out passed (chip)
                    out["improved"] = True
                    if out["best_speedup"] is None or rec["speedup"] > out["best_speedup"]:
                        out["best_speedup"] = rec["speedup"]
            graded.append((RANK[rec["verdict"]], rec.get("speedup") or 0.0, src, referee_says, rec, waste))
        log.flush()
        if not graded:
            continue

        graded.sort(key=lambda g: (g[0], g[1]), reverse=True)
        _, speedup, src, referee_says, rec, waste = graded[0]
        print(f"round {rnd}: best this round {rec['verdict'] or 'PASS (untimed)'}"
              + (f", {waste:.2f}x the byte floor (sim)" if waste else "")
              + (f", speedup {speedup:.3f} ({rec['source']})" if rec.get("speedup") else "")
              + f"  [{out['attempts']}/{a.budget} evaluations, {time.perf_counter() - t0:.1f}s]")

        if a.arm == "model_alone":
            prompt = alone_prompt(best["src"], best["time"], first=False)
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
    ap.add_argument("--arm", default="referee", choices=schema.ARMS,
                    help="referee or model_alone; random_search runs from P2's search.py")
    ap.add_argument("--budget", type=int, default=8, help="referee evaluations per run, every arm")
    # Measured on seat-101: the samples of one round came back identical every time, so a second sample
    # spent budget on a repeat. One sample per round: every evaluation is a new attempt.
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=None, help="default: enough to spend the budget")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--give-up-after", type=int, default=0,
                    help="arm referee only: stop after the same instruction this many times. 0 (default) = "
                         "never, so every arm spends its whole budget, which a fair comparison needs")
    ap.add_argument("--start", default=None, help="start kernel; default P2's kernels/matmul_start.py "
                                                  "if it exists, else reference_level4.py")
    ap.add_argument("--seat", type=int, default=seat_from_hostname())
    ap.add_argument("--max-tokens", type=int, default=agent02.MIN_ANSWER_TOKENS)
    ap.add_argument("--context", type=int, default=8192, help="the server's max-model-len")
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--model", default=agent02.MODEL)
    ap.add_argument("--base", default=os.environ.get("KERNEL_AGENT_BASE_URL"))
    ap.add_argument("--log", default=None, help="default logs/seat-<seat>/attempts.jsonl (P4 collects it)")
    ap.add_argument("--offline", action="store_true", help="no model; the simulator grades (needs nki)")
    ap.add_argument("--dry", action="store_true", help="no model, rules stage only (no nki needed)")
    a = ap.parse_args()
    if a.arm == "random_search":
        sys.exit("arm random_search runs from P2's search.py, which logs arm=random_search to the same seat "
                 "log:\n    python search.py --budget <same budget> --seed <rep>\nagent.py runs the two model "
                 "arms, referee and model_alone.")

    start_path = a.start or (P2_START if os.path.exists(P2_START) else FALLBACK_START)
    # --dry never touches the real referee (importing speedcheck needs the pod's ml_dtypes).
    referee = ("stage12 rules only (--dry)", None) if a.dry else pick_referee()
    # First line on purpose: the team checks `head -1 run.log` says "referee: speedcheck".
    print(f"referee: {referee[0]}")
    if a.seat is not None:
        os.environ["CHIPBOOST_SEAT"] = str(a.seat)   # speedcheck writes it into `seat`
    a.log = a.log or os.path.join(HERE, "logs", f"seat-{a.seat if a.seat is not None else 'unknown'}",
                                  "attempts.jsonl")
    os.makedirs(os.path.dirname(os.path.abspath(a.log)), exist_ok=True)
    if a.offline or a.dry:
        print("*** OFFLINE: replaying the start kernel, no model. Numbers are meaningless. ***")
    else:
        if not (a.base or "").strip():
            sys.exit("KERNEL_AGENT_BASE_URL is unset. The seat pods set it; or pass --offline / --dry.")
        print(f"endpoint {a.base}  model {a.model}")
    print(f"start:   {os.path.relpath(start_path, HERE)}   arm: {a.arm}   budget: {a.budget}   "
          f"give-up: {a.give_up_after or 'never'}   log: {a.log}")

    results = []
    # Outside projects/: a file appearing under the referee's tree mid-check reads as tampering (REFEREE.md).
    workdir = tempfile.mkdtemp(prefix="chipboost_cands_", dir="/tmp" if os.path.isdir("/tmp") else None)
    try:
        with open(a.log, "a") as log:
            for rep in range(a.repeat):
                results.append(run_once(a, referee, start_path, rep, log, workdir))
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
