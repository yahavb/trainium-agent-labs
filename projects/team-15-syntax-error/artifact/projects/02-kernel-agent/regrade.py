#!/usr/bin/env python3
"""
regrade.py -- validity check: re-grade every logged attempt from its own code, compare to the log.

Until 12:50 agent.grade() wrote each candidate to /tmp/_agent_level{N}.py, a path shared by every
process. runq runs several jobs of the SAME level concurrently, so one job could in principle load
another job's file between write and import and log the other kernel's score. Grading is
deterministic, so re-grading each logged `code` in isolation must reproduce the logged reward and
feedback. Any mismatch = a contaminated attempt.

    python regrade.py runs/skel_*.jsonl          (in the pod, where nki exists)
"""

import glob
import json
import sys

import agent
import nkibench
import verdicts


def main(paths):
    # Re-grade with the SAME feedback flags the run used (they change the wording, never the
    # reward): --v2 / --v3 / --directional on the command line, mirroring agent.py's switches.
    verdicts.V2 = "--v2" in sys.argv
    verdicts.V3 = "--v3" in sys.argv
    verdicts.V4 = "--v4" in sys.argv
    verdicts.V5 = "--v5" in sys.argv
    verdicts.V6 = "--v6" in sys.argv
    verdicts.V7 = "--v7" in sys.argv
    verdicts.V8 = "--v8" in sys.argv
    verdicts.V9 = "--v9" in sys.argv
    verdicts.V10 = "--v10" in sys.argv
    nkibench.DIRECTIONAL = "--directional" in sys.argv
    nkibench.DEVICE_RULES = "--device-rules" in sys.argv
    nkibench.PROMPT_FIXES = "--prompt-fixes" in sys.argv
    paths = [p for p in paths if not p.startswith("--")]
    rows, cache = [], {}
    for p in paths:
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            if "claim" not in r and "code" in r:
                rows.append((p, r))
    bad = 0
    for p, r in rows:
        k = (r["level"], r["code"])
        if k not in cache:
            cache[k] = agent.grade(r["code"], r["level"])
        reward, _, feedback = cache[k]
        if abs(reward - r["reward"]) > 1e-9 or feedback != r["feedback"]:
            bad += 1
            if abs(reward - r["reward"]) <= 1e-9:
                print(f"  (reward identical, feedback wording differs: check the flags)")
            print(f"MISMATCH {p} L{r['level']} run {r.get('run')} round {r['round']}: logged "
                  f"{r['reward']:.2f}, regraded {reward:.2f}")
    print(f"regraded {len(rows)} attempts ({len(cache)} distinct kernels): {bad} mismatches")
    return bad


if __name__ == "__main__":
    sys.exit(1 if main([q for a in sys.argv[1:] for q in (glob.glob(a) or [a])]) else 0)
