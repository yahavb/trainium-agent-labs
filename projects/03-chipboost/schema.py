#!/usr/bin/env python3
"""
schema.py -- the one log format every CHIPBOOST seat writes, and the dashboard reads.

Each line of attempts.jsonl is one candidate kernel through the referee. Changing a field
means telling all four branches; see TEAM.md.

    python schema.py --check attempts.jsonl      # validate a log
    python schema.py --fake > fake_attempts.jsonl  # data for building the dashboard
"""

import argparse
import json
import random
import sys
import time

# no_gain: correct, but the difference from the baseline is inside the measured timing noise.
VERDICTS = ("rules", "wrong", "heldout_fail", "slower", "no_gain", "faster")
SOURCES = ("chip", "sim")
ARMS = ("referee", "model_alone", "random_search")
OPS = ("matmul", "rmsnorm", "swiglu")

# field -> type. None is allowed for every field the referee did not reach
# (a kernel rejected by the rules scan has no timing).
ATTEMPT_FIELDS = {
    "seat": int,                  # 100, 101, ...
    "kernel": str,                # one of OPS
    "arm": str,                   # one of ARMS
    "run_id": str,                # one id per --repeat run
    "attempt_no": int,            # 0, 1, 2 ... within the run; the x axis of the progress curve
    "round": int,                 # agent round (several samples share a round)
    "prompt_tokens": int,         # input tokens sent to the model; None for random_search
    "code_hash": str,             # sha1 of the kernel source
    "code": str,                  # the full kernel source, so the dashboard can show diffs
    "prompt": str,                # what the model was sent; None for random_search
    "response": str,              # what the model returned; None for random_search
    "verdict": str,               # one of VERDICTS
    "referee_message": str,       # what the referee found
    "instruction_given": str,     # the ONE change sent back to the model
    "sim_ok": bool,
    "chip_ok": bool,
    "time_us_median": float,
    "time_us_iqr": float,
    "baseline_us_same_session": float,   # start kernel, timed interleaved in the same session
    "speedup": float,             # baseline / candidate; > 1 is faster
    "source": str,                # one of SOURCES: where the time came from
    "timestamp": float,           # unix seconds
}


def validate(rec):
    """Return a list of problems with one log record; empty means valid."""
    problems = [f"missing field {k}" for k in ATTEMPT_FIELDS if k not in rec]
    problems += [f"unknown field {k}" for k in rec if k not in ATTEMPT_FIELDS]
    for k, typ in ATTEMPT_FIELDS.items():
        v = rec.get(k)
        if v is not None and not isinstance(v, typ) and not (typ is float and isinstance(v, int)):
            problems.append(f"{k} should be {typ.__name__}, got {type(v).__name__}")
    for k, allowed in (("verdict", VERDICTS), ("source", SOURCES), ("arm", ARMS), ("kernel", OPS)):
        if rec.get(k) is not None and rec[k] not in allowed:
            problems.append(f"{k}={rec[k]!r} not in {allowed}")
    return problems


def fake_attempts(n_runs=3, n_attempts=24, seed=0):
    """Plausible-looking records for building the dashboard without a chip. Never report these."""
    r = random.Random(seed)
    out = []
    for kernel, base in (("matmul", 410.0), ("rmsnorm", 38.0)):
        for arm, skill in (("referee", 0.6), ("model_alone", 0.3), ("random_search", 0.45)):
            for run in range(n_runs):
                best = base
                for i in range(n_attempts):
                    verdict = r.choices(VERDICTS, weights=(1, 2, 0.5, 2, 1, 1 + 3 * skill))[0]
                    t = None
                    if verdict in ("slower", "no_gain", "faster"):
                        t = best * {"faster": r.uniform(0.85, 0.97), "no_gain": r.uniform(0.99, 1.01),
                                    "slower": r.uniform(1.03, 1.3)}[verdict]
                        if verdict == "faster":
                            best = t
                    model = arm != "random_search"
                    out.append(dict(
                        seat=100 + ARMS.index(arm), kernel=kernel, arm=arm,
                        run_id=f"{kernel}-{arm}-{run}", attempt_no=i, round=i // 4,
                        prompt_tokens=None if arm == "random_search" else r.randint(900, 3000),
                        code_hash=f"{r.getrandbits(40):010x}",
                        code=f"# (fake) {kernel} kernel, attempt {i}\n",
                        prompt="(fake) prompt" if model else None,
                        response="(fake) response" if model else None,
                        verdict=verdict,
                        referee_message=f"(fake) {verdict}", instruction_given="(fake) one named change",
                        sim_ok=verdict not in ("rules", "wrong"),
                        chip_ok=verdict in ("slower", "no_gain", "faster", "heldout_fail"),
                        time_us_median=t, time_us_iqr=None if t is None else t * 0.02,
                        baseline_us_same_session=base * r.uniform(0.99, 1.01),
                        speedup=None if t is None else base / t, source="chip",
                        timestamp=time.time() + i))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", metavar="JSONL")
    ap.add_argument("--fake", action="store_true")
    a = ap.parse_args()
    if a.fake:
        for rec in fake_attempts():
            print(json.dumps(rec))
    elif a.check:
        bad = 0
        for n, line in enumerate(open(a.check), 1):
            for p in validate(json.loads(line)):
                print(f"line {n}: {p}")
                bad += 1
        print("valid" if not bad else f"{bad} problem(s)")
        sys.exit(1 if bad else 0)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
