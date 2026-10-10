"""The orchestrator: plain code, no LLM. Picks the strategy, enforces the budget, keeps the ledger.

DESIGN.md 6.6. One run is one row of this table, and every attempt is one line of runs/<RUN>-<N>.jsonl:

    run  seats  attempts  strategy sequence                         feedback
    A    any    1         S1                                        none
    B    any    up to 6   S1, then S3 x 5                           raw
    C    any    up to 6   S1, then S3 x 5                           located
    D    3      6         round 1: S1 / S2 / S4 on seats A / B / C   located
                          round 2: each seat repairs its own (S3)
    E    gpt-oss up to 6  S1, then S3 x 5                           located
    R    any    up to 6   S1 x 6, fresh each time                   none   (control: retries without feedback)

Seats only change wall-clock time for A, B, C, E and R -- a problem stays on one seat and its
attempts are sequential. Give them several seats and the problems spread across them.

    nohup python agent.py --run C --rep 1 --problems eval/dev.txt --seats "$SEATS" \\
          > runs/C-1.log 2>&1 < /dev/null &
    python agent.py --selftest          # stop rules, ledger and run shapes, with a fake model
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import queue
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import prompts
import problems as P
import translator
import workers

HERE = os.path.dirname(os.path.abspath(__file__))

RUNS = {
    "A": dict(budget=1, mode=None),
    "B": dict(budget=6, mode="raw", cycle_stop=True),
    "C": dict(budget=6, mode="located", cycle_stop=True),
    "D": dict(budget=6, mode="located", breadth=True),
    "E": dict(budget=6, mode="located", cycle_stop=True, model="gpt-oss-20b"),
    "R": dict(budget=6, mode=None),
}
CYCLE_REPEATS = 3
BREADTH_STRATEGIES = ("S1", "S2", "S4")


@dataclass
class Ctx:
    run: str
    rep: int
    budget: int
    mode: str | None
    model: str
    max_tokens: int
    temperature: float
    out: str
    cycle_stop: bool = False
    breadth: bool = False
    code_version: str = ""
    ask: object = None        # workers.ask, or a fake in --selftest
    check: object = None      # checker.check, or a fake in --selftest
    quiet: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


CODE_FILES = ["checker.py", "translator.py", "prompts.py", "agent.py", "workers.py", "problems.py"]


def git_version() -> str:
    """The last commit that changed the code that decides an attempt's outcome, plus "+dirty" if any
    of those files has uncommitted changes.

    Not HEAD: docs, STATUS and committed run files move HEAD without changing behaviour, so two seats
    running identical code would record different versions, and D-015's rule (compare only the same
    code version) would then flag comparable runs.
    """
    try:
        sha = subprocess.run(["git", "log", "-1", "--format=%h", "--", *CODE_FILES], cwd=HERE,
                             capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", *CODE_FILES], cwd=HERE,
                               capture_output=True, text=True).stdout.strip()
        return sha + ("+dirty" if dirty else "") if sha else "unknown"
    except OSError:
        return "unknown"


def write(ctx: Ctx, rec: dict):
    with ctx.lock:
        with open(ctx.out, "a") as f:
            f.write(json.dumps(rec) + "\n")


def say(ctx: Ctx, line: str):
    if not ctx.quiet:
        with ctx.lock:
            print(line, flush=True)


def record(ctx, problem, attempt, rnd, seat, strategy, prompt, note, reply, error, res, fb_sent,
           stop_reason=None, claim=None) -> dict:
    """One attempt, every field in DESIGN 7.1, plus the checker detail the taxonomy needs."""
    return {
        "ts": now(), "run": ctx.run, "rep": ctx.rep, "problem": problem.id, "kind": problem.kind,
        "attempt": attempt, "round": rnd, "seat": seat, "model": ctx.model, "strategy": strategy,
        "prompt_chars": len(prompt), "prompt_note": note,
        "prompt_tokens": reply.prompt_tokens if reply else None,
        "completion_tokens": reply.completion_tokens if reply else None,
        "finish_reason": reply.finish_reason if reply else None,
        "latency_s": reply.latency_s if reply else None,
        "layer_reached": res.layer_reached, "passed": res.passed, "score": res.score,
        "mismatches": res.mismatches, "samples": res.samples, "first_mismatch": res.first_mismatch,
        "formal": res.formal,
        "feedback_mode": ctx.mode, "feedback_sent": fb_sent,
        "code_sha1": hashlib.sha1(res.code.encode()).hexdigest() if res.code else None,
        "code": res.code,
        "stop_reason": stop_reason, "claim": claim,
        # beyond DESIGN 7.1 (1.1.0): what the failure taxonomy and the audit trail need
        "compile_error": res.compile_error, "l0_error": res.l0_error, "tb_hints": res.tb_hints,
        "counterexample": res.counterexample, "timed_out": res.timed_out, "check_s": res.elapsed_s,
        "error": error, "code_version": ctx.code_version,
    }


def progress(ctx, problem, rnd, seat, res, fb):
    layer = f"L{res.layer_reached}" if res.layer_reached >= 0 else "L-"
    note = "PASS" if res.passed else (fb or "")[:70].replace("\n", " ")
    say(ctx, f"{problem.id:34s} {ctx.run}  r{rnd}  {seat:9s} {layer} {res.score:.2f}  "
             f"{note if res.passed else repr(note)}")


def attempt_once(ctx, problem, seat_url, strategy, code=None, fb=None, earlier=None):
    """(prompt, note, reply, error, result). Raises workers.SeatDown; any other model failure is an
    attempt that produced no code, so it costs budget like any other."""
    prompt, note = prompts.build(strategy, problem, code, fb, earlier=earlier)
    error = None
    try:
        reply = ctx.ask(seat_url, prompt, model=ctx.model, max_tokens=ctx.max_tokens,
                        temperature=ctx.temperature)
    except workers.ModelError as e:
        error = str(e)
        reply = workers.Reply("", 0, 0, "error", 0.0, workers.seat_name(seat_url), ctx.model)
    res = ctx.check(problem, reply.text)
    return prompt, note, reply, error, res


def claim_for(res) -> str:
    return "PASS" if res.passed else ("UNVERIFIED" if res.timed_out else "FAIL")


def seat_down_record(ctx, problem, attempt, rnd, seat, strategy):
    from checker import CheckResult
    rec = record(ctx, problem, attempt, rnd, seat, strategy, "", "", None, "seat down",
                 CheckResult(), None, "seat_down", "UNVERIFIED")
    write(ctx, rec)
    say(ctx, f"{problem.id:34s} {ctx.run}  r{rnd}  {seat:9s} SEAT DOWN -> UNVERIFIED")


# ---------------------------------------------------------------- depth: one seat, in sequence

def solve_depth(ctx: Ctx, problem, seat_url: str) -> str:
    seat = workers.seat_name(seat_url)
    code, fb, sent = None, None, []
    copied = False
    seen = []       # D-016 ledger: distinct findings so far, oldest first
    for attempt in range(1, ctx.budget + 1):
        if attempt == 1 or ctx.mode is None:
            strategy = "S1"
        elif copied:
            # D-014: the last repair handed back the code it was asked to fix, byte for byte (71% of
            # repairs on dev C-1). Asking for another edit of the same code is a no-op, so start fresh
            # from the spec with the feedback as a hint. B and C get the same treatment.
            strategy = "S5"
        else:
            strategy = "S3"
        try:
            earlier = [m for m in seen if m != fb] if strategy != "S1" else None
            prompt, note, reply, error, res = attempt_once(ctx, problem, seat_url, strategy, code, fb,
                                                           earlier)
        except workers.SeatDown:
            seat_down_record(ctx, problem, attempt, attempt, seat, strategy)
            raise
        copied = ctx.mode is not None and code is not None and res.code == code
        stop, next_fb = None, None
        if res.passed:
            stop = "passed"
        else:
            if ctx.mode:
                next_fb = translator.feedback(problem, res, ctx.mode)
                # A copy is not a new design, so it does not count towards the cycling stop: three
                # identical messages must come from three attempts that actually tried something.
                if not copied:
                    sent.append(next_fb)
                if ctx.cycle_stop and len(sent) >= CYCLE_REPEATS and len(set(sent[-CYCLE_REPEATS:])) == 1:
                    stop = "cycling"
            if stop is None and attempt == ctx.budget:
                stop = "budget"
            if stop and res.timed_out:
                stop = "checker_timeout"
        claim = claim_for(res) if stop else None
        rec = record(ctx, problem, attempt, attempt, seat, strategy, prompt, note, reply, error,
                     res, None if stop else next_fb, stop, claim)
        rec["ledger_items"] = len(earlier[-prompts.LEDGER_ITEMS:]) if earlier else 0
        write(ctx, rec)
        progress(ctx, problem, attempt, seat, res, next_fb)
        if stop:
            return claim
        if next_fb and next_fb not in seen:
            seen.append(next_fb)
        # Repair the LATEST attempt, not the best: repairing the best froze the event repo's loop.
        code, fb = res.code, next_fb
    return "FAIL"


# ---------------------------------------------------------------- breadth: three seats at once

def solve_breadth(ctx: Ctx, problem, seat_urls: list) -> str:
    seats = [workers.seat_name(u) for u in seat_urls]
    rounds = max(1, ctx.budget // len(seat_urls))
    prev = [None] * len(seat_urls)       # (code, feedback) per seat after its last attempt
    attempt_no = 0
    for rnd in range(1, rounds + 1):
        jobs = []
        with ThreadPoolExecutor(len(seat_urls)) as ex:
            for i, url in enumerate(seat_urls):
                if rnd == 1:
                    jobs.append(ex.submit(attempt_once, ctx, problem, url, BREADTH_STRATEGIES[i % 3]))
                else:
                    code, fb = prev[i]
                    jobs.append(ex.submit(attempt_once, ctx, problem, url, "S3", code, fb))
        outcomes = []
        for i, job in enumerate(jobs):
            try:
                outcomes.append(job.result())
            except workers.SeatDown:
                outcomes.append(None)
        last_round = rnd == rounds
        any_pass = any(o and o[4].passed for o in outcomes)
        down = any(o is None for o in outcomes)
        for i, o in enumerate(outcomes):
            attempt_no += 1
            strategy = BREADTH_STRATEGIES[i % 3] if rnd == 1 else "S3"
            if o is None:
                seat_down_record(ctx, problem, attempt_no, rnd, seats[i], strategy)
                continue
            prompt, note, reply, error, res = o
            fb = None if res.passed else translator.feedback(problem, res, ctx.mode)
            prev[i] = (res.code, fb)
            final = i == len(outcomes) - 1 and (any_pass or last_round or down)
            stop = claim = None
            if final:
                best = next((x[4] for x in outcomes if x and x[4].passed), res)
                stop = "seat_down" if down and not any_pass else ("passed" if any_pass else "budget")
                claim = "UNVERIFIED" if stop == "seat_down" else claim_for(best)
                if claim == "FAIL" and any(x and x[4].timed_out for x in outcomes):
                    stop, claim = "checker_timeout", "UNVERIFIED"
            write(ctx, record(ctx, problem, attempt_no, rnd, seats[i], strategy, prompt, note, reply,
                              error, res, None if (any_pass or last_round) else fb, stop, claim))
            progress(ctx, problem, rnd, seats[i], res, fb)
        if down:
            raise workers.SeatDown(", ".join(s for s, o in zip(seats, outcomes) if o is None))
        if any_pass:
            return "PASS"
    return "FAIL"


# ---------------------------------------------------------------- orchestration

def done_problems(path: str) -> set:
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if rec.get("claim"):
                    done.add(rec["problem"])
    return done


def run_all(ctx: Ctx, todo: list, seat_urls: list, parallel: int) -> dict:
    claims = {}
    if ctx.breadth:
        aborted = threading.Event()

        def one(p):
            if aborted.is_set():
                return
            try:
                claims[p.id] = solve_breadth(ctx, p, seat_urls[:3])
            except workers.SeatDown as e:
                say(ctx, f"seat down ({e}); run D needs all three seats, so it stops here")
                claims[p.id] = "UNVERIFIED"
                aborted.set()

        with ThreadPoolExecutor(parallel) as ex:
            list(ex.map(one, todo))
    else:
        q = queue.Queue()
        for p in todo:
            q.put(p)

        def worker(url):
            while True:
                try:
                    p = q.get_nowait()
                except queue.Empty:
                    return
                try:
                    claims[p.id] = solve_depth(ctx, p, url)
                except workers.SeatDown:
                    claims[p.id] = "UNVERIFIED"
                    say(ctx, f"{workers.seat_name(url)} is down; its other workers stop taking problems")
                    return

        threads = [threading.Thread(target=worker, args=(u,), daemon=True)
                   for u in seat_urls for _ in range(parallel)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    for p in todo:      # every seat died before these were reached: still counted, as UNVERIFIED
        if p.id not in claims:
            seat_down_record(ctx, p, 0, 0, "none", "-")
            claims[p.id] = "UNVERIFIED"
    return claims


def check_heldout_list(problems_path: str):
    """Every held-out run uses the same 40 problems: the committed eval/heldout.txt, unmodified."""
    if os.path.basename(problems_path) != "heldout.txt":
        return
    rel = os.path.relpath(os.path.abspath(problems_path), HERE)
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=HERE, capture_output=True)
    changed = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=HERE)
    if tracked.returncode != 0 or changed.returncode != 0:
        sys.exit("eval/heldout.txt differs from the committed list. Commit or restore it, so every "
                 "held-out run uses the same 40 problems.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", choices=sorted(RUNS))
    ap.add_argument("--rep", type=int, default=1)
    ap.add_argument("--problems", default=os.path.join(HERE, "eval", "dev.txt"))
    ap.add_argument("--seats", default=os.environ.get("SEATS", "http://localhost:8000/v1"),
                    help="comma-separated base URLs; run E defaults to $GPTOSS_BASE_URL/agg/v1")
    ap.add_argument("--out", help="default runs/<RUN>-<REP>.jsonl")
    ap.add_argument("--budget", type=int, default=6, help="attempts per problem (run A is always 1)")
    ap.add_argument("--max-tokens", type=int, default=1500)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--parallel-problems", type=int, default=4, help="problems in flight per seat, at most 4")
    ap.add_argument("--model", help="default Qwen/Qwen3-8B, or gpt-oss-20b for run E")
    ap.add_argument("--only", help="comma-separated problem ids, for a quick try on dev")
    ap.add_argument("--resume", action="store_true", help="append to an existing run file, skipping finished problems")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if not a.run:
        ap.error("--run is required")

    spec = RUNS[a.run]
    model = a.model or spec.get("model") or workers.QWEN
    seats = [s.strip() for s in a.seats.split(",") if s.strip()]
    if a.run == "E" and "--seats" not in " ".join(sys.argv) and os.environ.get("GPTOSS_BASE_URL"):
        seats = [os.environ["GPTOSS_BASE_URL"].rstrip("/") + "/agg/v1"]
    if spec.get("breadth") and len(seats) < 3:
        sys.exit("run D needs three seats: --seats URL_A,URL_B,URL_C")
    check_heldout_list(a.problems)
    probs = P.load_list(a.problems)
    if a.only:
        keep = set(a.only.split(","))
        probs = [p for p in probs if p.id in keep]
    # Held-out runs go in runs/, anything else in runs/<set>/, so a dev C-1 can never be mistaken for
    # (or overwrite) the held-out C-1 that the report and the judges read.
    setname = os.path.splitext(os.path.basename(a.problems))[0]
    outdir = os.path.join(HERE, "runs") if setname == "heldout" else os.path.join(HERE, "runs", setname)
    out = a.out or os.path.join(outdir, f"{a.run}-{a.rep}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    finished = done_problems(out)
    if finished and not a.resume:
        sys.exit(f"{out} already has results. Run files are append-only: pick a new --rep, or "
                 f"--resume to finish this one.")
    todo = [p for p in probs if p.id not in finished]

    from checker import check
    ctx = Ctx(run=a.run, rep=a.rep, budget=1 if a.run == "A" else a.budget, mode=spec["mode"],
              model=model, max_tokens=a.max_tokens, temperature=a.temperature, out=out,
              cycle_stop=spec.get("cycle_stop", False), breadth=spec.get("breadth", False),
              code_version=git_version(), ask=workers.ask, check=check)
    for url in seats:
        if not workers.healthy(url):
            print(f"warning: {url} does not answer /health", flush=True)
    print(f"run {a.run} rep {a.rep}: {len(todo)} problems ({len(finished)} already done), "
          f"budget {ctx.budget}, feedback {ctx.mode or 'none'}, model {model}, "
          f"seats {', '.join(workers.seat_name(s) for s in seats)}, code {ctx.code_version} -> {out}",
          flush=True)
    t0 = time.time()
    claims = run_all(ctx, todo, seats, max(1, min(4, a.parallel_problems)))
    n = len(todo)
    passed = sum(c == "PASS" for c in claims.values())
    unver = sum(c == "UNVERIFIED" for c in claims.values())
    print(f"\n=========== run {a.run} rep {a.rep} ===========\n"
          f"  solved {passed}/{n}   unverified {unver}   wall {time.time() - t0:.0f}s   -> {out}",
          flush=True)


# ---------------------------------------------------------------- self-test (fake model, fake checker)

def selftest() -> int:
    import tempfile
    from checker import CheckResult
    fails = 0
    prob = P.Problem("ProbX_fake", "spec text\n", "/nonexistent_ref.sv", "", "comb",
                     [("input", "a", 1), ("output", "out", 1)])

    def fake_ask_factory(script):
        """script: list of reply texts handed out in order, per seat."""
        it = {}

        def ask(url, prompt, model, max_tokens, temperature):
            if "dead" in url:
                raise workers.SeatDown(workers.seat_name(url))
            seq = it.setdefault(url, iter(script))
            return workers.Reply(next(seq), 100, 50, "stop", 0.1, workers.seat_name(url), model)
        return ask

    def fake_check(problem, text):
        if text == "PASS":
            return CheckResult(layer_reached=2, passed=True, score=1.0, code="module TopModule; endmodule",
                               mismatches=0, samples=10)
        return CheckResult(layer_reached=2, score=0.6, code=f"module TopModule; // {text}\nendmodule",
                           mismatches=4, samples=10, raw="Mismatches: 4 in 10 samples",
                           first_mismatch={"output": "out", "time_ps": 30, "cycle": 3,
                                           "inputs": {"a": "1"}, "dut": "0"})

    def run_case(name, run, script, seats, expect):
        nonlocal fails
        path = os.path.join(tempfile.mkdtemp(prefix="vra-agent-"), f"{run}-1.jsonl")
        spec = RUNS[run]
        ctx = Ctx(run=run, rep=1, budget=1 if run == "A" else 6, mode=spec["mode"], model="fake",
                  max_tokens=100, temperature=0.6, out=path, cycle_stop=spec.get("cycle_stop", False),
                  breadth=spec.get("breadth", False), code_version="test",
                  ask=fake_ask_factory(script), check=fake_check, quiet=True)
        claims = run_all(ctx, [prob], seats, 1)
        recs = [json.loads(l) for l in open(path)]
        problems_seen = expect(claims, recs)
        ok = problems_seen is True
        fails += not ok
        print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + ("" if ok else f"\n        {problems_seen}\n        {claims} "
                                                          f"{[(r['attempt'], r['strategy'], r['stop_reason'], r['claim']) for r in recs]}"))

    def check_fields(recs):
        need = {"ts", "run", "rep", "problem", "kind", "attempt", "round", "seat", "model", "strategy",
                "prompt_chars", "prompt_tokens", "completion_tokens", "finish_reason", "latency_s",
                "layer_reached", "passed", "score", "mismatches", "samples", "first_mismatch", "formal",
                "feedback_mode", "feedback_sent", "code_sha1", "code", "stop_reason", "claim"}
        missing = [need - set(r) for r in recs if need - set(r)]
        return missing or None

    print("agent self-test (fake model, fake checker)")
    one = ["http://localhost:8000/v1"]
    three = ["http://seat-1.seat:8000/v1", "http://seat-2.seat:8000/v1", "http://seat-3.seat:8000/v1"]

    run_case("C: fail, fail, pass -> PASS on attempt 3, S1 then S3, feedback carried",
             "C", ["w1", "w2", "PASS"], one,
             lambda c, r: True if (c == {prob.id: "PASS"} and [x["attempt"] for x in r] == [1, 2, 3]
                                   and [x["strategy"] for x in r] == ["S1", "S3", "S3"]
                                   and r[0]["feedback_sent"] and r[0]["feedback_sent"].startswith("When a=1")
                                   and r[-1]["stop_reason"] == "passed" and r[-1]["claim"] == "PASS"
                                   and r[0]["claim"] is None and not check_fields(r)) else check_fields(r) or "mismatch")
    run_case("B: different code, identical raw feedback 3 times -> cycling, FAIL",
             "B", ["w1", "w2", "w3", "w4", "w5", "w6"], one,
             lambda c, r: True if (c == {prob.id: "FAIL"} and len(r) == 3 and r[-1]["stop_reason"] == "cycling"
                                   and [x["strategy"] for x in r] == ["S1", "S3", "S3"]
                                   and r[0]["feedback_sent"] == "Mismatches: 4 in 10 samples") else "mismatch")
    run_case("C: a repair that copies the code -> next attempt is S5; copies never trigger cycling",
             "C", ["w"] * 6, one,
             lambda c, r: True if (c == {prob.id: "FAIL"} and len(r) == 6
                                   and [x["strategy"] for x in r] == ["S1", "S3", "S5", "S5", "S5", "S5"]
                                   and r[-1]["stop_reason"] == "budget") else "mismatch")
    ledger_prompts = []
    msgs = iter(["v1", "v2", "v3", "v4"])

    def varying_ask(url, prompt, model, max_tokens, temperature):
        ledger_prompts.append(prompt)
        return workers.Reply(next(msgs, "v9"), 100, 50, "stop", 0.1, "seat-1", model)

    def varying_check(problem, text):
        cyc = int(text[1:])
        return CheckResult(layer_reached=2, score=0.6, code=f"module TopModule; // {text}\nendmodule",
                           mismatches=4, samples=10, raw=f"cycle {cyc}",
                           first_mismatch={"output": "out", "time_ps": cyc * 10, "cycle": cyc,
                                           "inputs": {"a": "1"}, "dut": "0"})

    # Sequential, so each finding names a different clock cycle and nothing trips the cycling stop.
    seq = P.Problem("ProbY_fake", "spec text\n", "/nonexistent_ref.sv", "", "seq",
                    [("input", "clk", 1), ("input", "a", 1), ("output", "out", 1)])
    path = os.path.join(tempfile.mkdtemp(prefix="vra-agent-"), "C-1.jsonl")
    ctx = Ctx(run="C", rep=1, budget=4, mode="located", model="fake", max_tokens=100, temperature=0.6,
              out=path, cycle_stop=True, code_version="test", ask=varying_ask, check=varying_check, quiet=True)
    run_all(ctx, [seq], one, 1)
    recs = [json.loads(l) for l in open(path)]
    first_msg = recs[0]["feedback_sent"]
    ok = (len(ledger_prompts) == 4 and "Earlier versions" not in ledger_prompts[0]
          and "Earlier versions" not in ledger_prompts[1]
          and "Earlier versions" in ledger_prompts[2] and first_msg in ledger_prompts[2].split("A checker reports:")[0]
          and [r["ledger_items"] for r in recs] == [0, 0, 1, 2])
    fails += not ok
    print(f"  {'ok  ' if ok else 'FAIL'}  ledger: repair 2 onwards carries the earlier findings (D-016)"
          + ("" if ok else f"  {[r['ledger_items'] for r in recs]}"))
    run_case("A: one attempt only", "A", ["w"] * 6, one,
             lambda c, r: True if (len(r) == 1 and r[0]["stop_reason"] == "budget" and r[0]["claim"] == "FAIL")
             else "mismatch")
    run_case("R: six fresh S1 attempts, no feedback", "R", ["w"] * 6, one,
             lambda c, r: True if (len(r) == 6 and all(x["strategy"] == "S1" for x in r)
                                   and all(x["feedback_sent"] is None for x in r)
                                   and r[-1]["stop_reason"] == "budget") else "mismatch")
    run_case("D: round 1 fails on 3 seats, round 2 passes -> 6 attempts, PASS",
             "D", ["w", "PASS"], three,
             lambda c, r: True if (c == {prob.id: "PASS"} and len(r) == 6
                                   and [x["strategy"] for x in r] == ["S1", "S2", "S4", "S3", "S3", "S3"]
                                   and [x["round"] for x in r] == [1, 1, 1, 2, 2, 2]
                                   and len({x["seat"] for x in r}) == 3
                                   and r[-1]["claim"] == "PASS" and r[2]["feedback_sent"]) else "mismatch")
    run_case("seat down -> UNVERIFIED, never retried", "C", ["w"] * 6, ["http://dead-seat:8000/v1"],
             lambda c, r: True if (c == {prob.id: "UNVERIFIED"} and r[-1]["stop_reason"] == "seat_down")
             else "mismatch")
    print("ALL OK" if not fails else f"{fails} FAILED")
    return 1 if fails else 0


if __name__ == "__main__":
    main()
