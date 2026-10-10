"""verdict_nki.py: after each NKI level, the agent says what it believes about its own kernel, with a
confidence, from evidence it gathers itself. On by setting NKI_VERDICTS=<path> (one JSON line per
level-episode); feedback_v7 then wraps their solve(). Nothing about prompts or grading changes.

The evidence, in order:
  1. the solving kernel, read back from the attempt log (their solve() returns only the score);
  2. forms the CPU simulator accepts but the trn2 compiler rejects. Task 08: 34 of 35 simulator-only
     solves used nl.divide or a (rows, 1) column operand to nisa.tensor_tensor;
  3. if NKI_VERDICT_COMPILE=1 and neuronx-cc is importable: lowering for trn2 on the level's first shape;
  4. extra cases the checker never showed (hidden_eval.py): hostile values on the level's shapes, and
     unseen shapes on levels 1-4, counting only cases the reference kernel passes.
Status: VERIFIED, DOUBTFUL (fails an extra case), SIMULATOR ONLY (uses a form the compiler rejects;
not compiled here), REJECTED BY COMPILER, FAILED (not solved). Confidence: the Wilson 95% lower bound
of the extra-case pass rate; at most 0.3 for SIMULATOR ONLY; 0 for REJECTED and FAILED.
"""
import json
import os
import re
import shutil
import time

import hidden_eval as H
import nkibench

HERE = os.path.dirname(os.path.abspath(__file__))


def wilson_lower(k, n, z=1.96):
    if n == 0:
        return 0.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5
    return max(0.0, (c - r) / d)


def compiler_risks(code):
    """Forms the nki 0.6.0 simulator accepts and neuronx-cc rejects for trn2 (tasks 07-08)."""
    import feedback_v6 as v6
    out = []
    for m in re.finditer(r"^.*\bnl\.divide\(.*$", code, re.M):
        out.append(f"`{m.group(0).strip()}` uses nl.divide, which the trn2 compiler rejects")
        break
    for m in re.finditer(r"^.*\bop[01]?\s*=\s*nl\.divide\b.*$", code, re.M):     # task 12: all level-9 solves
        out.append(f"`{m.group(0).strip()}` uses nl.divide as an operation, which the trn2 compiler rejects")
        break
    for m in re.finditer(r"nisa\.tensor_tensor\(([^)]*)\)", code):
        for arg in ("data1", "data2"):
            a = re.search(rf"\b{arg}\s*=\s*([A-Za-z_]\w*)\b", m.group(1))
            if a and v6.is_column(code, a.group(1)):
                out.append(f"nisa.tensor_tensor gets `{a.group(1)}`, a (rows, 1) column, which the trn2 compiler "
                           f"rejects (use nisa.tensor_scalar with it as operand0)")
    return out


def lowers(kernel, level):
    """None if the compiler isn't available here; else (True, '') or (False, reason)."""
    if os.environ.get("NKI_VERDICT_COMPILE") != "1" or not shutil.which("neuronx-cc"):
        return None                        # no compiler here: say so, never call that a rejection
    try:
        from nkitool import analyze
    except Exception:  # noqa: BLE001
        return None
    s = nkibench.LEVELS[level]["shapes"][0]
    args, _ = nkibench.make_inputs(s, level)
    try:
        analyze(kernel, *args, neff=False)
        return True, ""
    except Exception as e:  # noqa: BLE001
        msg = str(e).replace("\n", " ")
        k = msg.find("error:")
        return False, (msg[k:k + 140] if k >= 0 else msg[:140])


def reference(level):
    for p in (os.path.join(os.getcwd(), f"reference_level{level}.py"), os.path.join(HERE, "answers", f"ans_level{level}.py")):
        if os.path.exists(p):
            return H.load(open(p).read(), level)
    return None


def verdict(level, code, best, rounds):
    if not code:
        return dict(level=level, status="FAILED", confidence=0.0, best=round(best, 3), rounds=rounds,
                    claim=f"not solved: best score {best:.2f} after {rounds} rounds")
    try:
        k = H.load(code, level)
        shown = nkibench.LEVELS[level]["shapes"]
        extra = H.cases_for(level) if level in H.SHAPES else [(s, v) for s in shown for v in H.VALUES if v != "normal"]
        ref = reference(level)
        fair = [c for c in extra if ref is None or not H.run_case(ref, level, *c)]
        bad = [H.case_label(level, *c) for c in fair if H.run_case(k, level, *c)]
        risks = compiler_risks(code)
        comp = lowers(k, level)
    except Exception as e:  # noqa: BLE001
        return dict(level=level, status="UNVERIFIED", confidence=0.0, rounds=rounds,
                    claim=f"solved on the checker's shapes; the extra checks could not run: {type(e).__name__}: {e}"[:300])
    conf = round(wilson_lower(len(fair) - len(bad), len(fair)), 3)
    base = dict(level=level, rounds=rounds, extra_cases=len(fair), extra_failed=len(bad), compiler_risks=risks,
                lowered=None if comp is None else comp[0])
    if comp is not None and not comp[0]:
        return dict(base, status="REJECTED BY COMPILER", confidence=0.0,
                    claim=f"passes the simulator, but the trn2 compiler rejects it: {comp[1]}")
    if bad:
        return dict(base, status="DOUBTFUL", confidence=conf,
                    claim=f"solved on the checker's shapes but fails {len(bad)} of {len(fair)} extra cases, first: {bad[0]}")
    if risks and comp is None:
        return dict(base, status="SIMULATOR ONLY", confidence=min(conf, 0.3),
                    claim=f"passes all {len(fair)} extra cases in the simulator, but {risks[0]}")
    how = ", and lowers for trn2" if comp else " (not compiled on this machine)"
    return dict(base, status="VERIFIED", confidence=conf,
                claim=f"solved, and passes all {len(fair)} extra cases it was never graded on{how}")


def install(agent, path):
    """Wrap agent.solve: after each level, read the solving kernel back from the log and judge it."""
    their_solve = agent.solve
    full = sum(agent.WEIGHTS.values())

    def solve_with_verdict(a, level, log):
        log.flush()
        start = os.path.getsize(log.name)
        out = their_solve(a, level, log)
        log.flush()
        with open(log.name) as f:
            f.seek(start)
            rows = [json.loads(l) for l in f if l.strip()]
        code = next((r["code"] for r in rows if r.get("level") == level and r["reward"] >= full - 1e-9), None)
        gated = next((r["code"] for r in rows if r.get("level") == level and r["reward"] >= full - 0.05 - 1e-9
                      and "trn2 compiler rejects" in (r.get("feedback") or "")), None)
        if code is None and gated:          # GATE held back a simulator solve the compiler rejects
            v = verdict(level, gated, out[0], out[1])
            v.update(status="SIMULATOR ONLY", confidence=min(v["confidence"], 0.3),
                     claim=f"passes the simulator, but the form the trn2 compiler rejects was still there after "
                           f"{out[1]} rounds")
        else:
            v = verdict(level, code, out[0], out[1])
        print(f"  VERDICT level {level}: {v['status']}, confidence {v['confidence']:.2f}: {v['claim']}")
        with open(path, "a") as f:
            f.write(json.dumps(dict(t=round(time.time()), **v)) + "\n")
        return out

    agent.solve = solve_with_verdict
