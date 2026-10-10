#!/usr/bin/env python3
"""Control run Q: the same 6-attempt budget as R, no feedback, but each attempt uses a different
fresh prompt (S1, S2, S4, S1, S2, S4).

Why: on the seats, identical prompts return identical code (63% of R-1's retries were byte-identical
copies), so R is "the same answer six times". A gain of B or C over R could then be "any change to
the prompt" rather than "the checker's information". Q varies the prompt without giving any
information. B or C beating Q is the feedback effect with that confound removed.

agent.py is untouched: this wrapper adds run Q and dispatches it to its own loop, and every other
run is delegated to the original code. The code files that agent.py hashes into code_version
(checker, translator, prompts, agent, workers, problems) are unchanged, so Q records the same
code_version as the runs it is compared with. Each record also carries `wrapper`, this file's hash.

    python3 controls/prompt_rotation.py --run Q --rep 1 --problems eval/heldout.txt \\
        --seats http://localhost:8000/v1 --parallel-problems 4 > runs/Q-1.log 2>&1
    python3 controls/prompt_rotation.py --q-selftest
"""
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import agent  # noqa: E402
import workers  # noqa: E402

ROTATION = ("S1", "S2", "S4")
with open(os.path.abspath(__file__), "rb") as _f:
    WRAPPER = f"controls/prompt_rotation.py sha256:{hashlib.sha256(_f.read()).hexdigest()[:12]}"

agent.RUNS["Q"] = dict(budget=6, mode=None)

_original_record = agent.record
_original_depth = agent.solve_depth


def record(*args, **kwargs):
    rec = _original_record(*args, **kwargs)
    rec["wrapper"] = WRAPPER
    return rec


def solve_rotate(ctx, problem, seat_url):
    """R's loop, except the strategy rotates. No feedback, no repair, fresh attempts only."""
    seat = workers.seat_name(seat_url)
    for attempt in range(1, ctx.budget + 1):
        strategy = ROTATION[(attempt - 1) % len(ROTATION)]
        try:
            prompt, note, reply, error, res = agent.attempt_once(ctx, problem, seat_url, strategy)
        except workers.SeatDown:
            agent.seat_down_record(ctx, problem, attempt, attempt, seat, strategy)
            raise
        stop = None
        if res.passed:
            stop = "passed"
        elif attempt == ctx.budget:
            stop = "checker_timeout" if res.timed_out else "budget"
        claim = agent.claim_for(res) if stop else None
        agent.write(ctx, agent.record(ctx, problem, attempt, attempt, seat, strategy, prompt, note,
                                      reply, error, res, None, stop, claim))
        agent.progress(ctx, problem, attempt, seat, res, None)
        if stop:
            return claim
    return "FAIL"


def solve_depth(ctx, problem, seat_url):
    if ctx.run == "Q":
        return solve_rotate(ctx, problem, seat_url)
    return _original_depth(ctx, problem, seat_url)


agent.record = record
agent.solve_depth = solve_depth


def selftest() -> int:
    import json
    import tempfile
    import problems as P
    from checker import CheckResult

    prob = P.Problem("ProbX_fake", "spec text\n", "/nonexistent_ref.sv", "", "comb",
                     [("input", "a", 1), ("output", "out", 1)])

    def run_case(script):
        it = iter(script)

        def ask(url, prompt, model, max_tokens, temperature):
            return workers.Reply(next(it), 100, 50, "stop", 0.1, workers.seat_name(url), model)

        def check(problem, text):
            if text == "PASS":
                return CheckResult(layer_reached=2, passed=True, score=1.0, code="m", mismatches=0, samples=10)
            return CheckResult(layer_reached=2, score=0.6, code=f"m {text}", mismatches=4, samples=10)

        path = os.path.join(tempfile.mkdtemp(prefix="vra-q-"), "Q-1.jsonl")
        ctx = agent.Ctx(run="Q", rep=1, budget=6, mode=None, model="fake", max_tokens=100,
                        temperature=0.6, out=path, code_version="test", ask=ask, check=check, quiet=True)
        claims = agent.run_all(ctx, [prob], ["http://localhost:8000/v1"], 1)
        return claims, [json.loads(l) for l in open(path)]

    fails = 0
    claims, recs = run_case(["w"] * 6)
    ok = (claims == {prob.id: "FAIL"} and [r["strategy"] for r in recs] == ["S1", "S2", "S4", "S1", "S2", "S4"]
          and all(r["feedback_sent"] is None and r["feedback_mode"] is None for r in recs)
          and recs[-1]["stop_reason"] == "budget" and recs[-1]["claim"] == "FAIL"
          and all(r.get("wrapper") == WRAPPER for r in recs))
    fails += not ok
    print(f"  {'ok  ' if ok else 'FAIL'}  Q: six fresh attempts rotating S1/S2/S4, no feedback, budget stop")
    claims, recs = run_case(["w", "PASS"])
    ok = claims == {prob.id: "PASS"} and [r["strategy"] for r in recs] == ["S1", "S2"] and recs[-1]["stop_reason"] == "passed"
    fails += not ok
    print(f"  {'ok  ' if ok else 'FAIL'}  Q: stops on the first pass (attempt 2, S2)")
    ok = agent.solve_depth is solve_depth and "Q" in agent.RUNS and agent.RUNS["R"] == dict(budget=6, mode=None)
    fails += not ok
    print(f"  {'ok  ' if ok else 'FAIL'}  other runs are delegated to the original loop; R unchanged")
    print("ALL OK" if not fails else f"{fails} FAILED")
    return 1 if fails else 0


if __name__ == "__main__":
    if "--q-selftest" in sys.argv:
        sys.exit(selftest())
    agent.main()
