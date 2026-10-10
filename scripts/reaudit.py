#!/usr/bin/env python3
"""Re-grade every kernel an attempt log scored 1.0, each in its own fresh process.

Why: a run graded in one long process can carry state between candidates. nki 0.6.0 caches by
file path, and before commit c39c0ce agent.py / feedback_v2.py reused one path, so the allocation
audit saw the first candidate's tiles for every later one. A fresh `nkibench.py --check` per kernel
has no such history. Use this before quoting any solve from a run started on older code.

    python scripts/reaudit.py attempts_v3_L4.jsonl [more.jsonl ...]
    (run where nki is importable: a seat pod, or the Docker image in SETUP_PYTHON.md)

Prints one line per distinct solved kernel and a summary; exit status 1 if any of them fails now.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile

BENCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "projects", "02-kernel-agent",
                     "nkibench.py")


def solved(paths):
    seen = {}
    for path in paths:
        for n, line in enumerate(open(path), 1):
            try:
                a = json.loads(line)
            except ValueError:
                continue
            if a.get("reward", 0) >= 1 - 1e-9 and a.get("code"):
                key = (a["level"], hashlib.sha1(a["code"].encode()).hexdigest()[:10])
                seen.setdefault(key, dict(level=a["level"], code=a["code"], where=[]))
                seen[key]["where"].append(f"{os.path.basename(path)}:{n} round {a.get('round')}")
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    a = ap.parse_args()
    kernels = solved(a.logs)
    if not kernels:
        print("no attempt scored 1.0 -- nothing to re-audit")
        return 0
    verdicts = {}
    with tempfile.TemporaryDirectory() as tmp:
        for (level, digest), k in sorted(kernels.items()):
            path = os.path.join(tmp, f"solved_l{level}_{digest}.py")
            with open(path, "w") as f:
                f.write(k["code"])
            out = subprocess.run([sys.executable, BENCH, "--level", str(level), "--check", path],
                                 capture_output=True, text=True)
            text = out.stdout + out.stderr
            if "ILLEGAL ON HARDWARE" in text:
                v = "ILLEGAL"
            elif out.returncode == 0:
                v = "PASS"
            elif out.returncode == 3:
                v = "NO-NKI"
            else:
                v = "FAIL"
            if v == "PASS" and "saw NO allocations" in text:
                v = "PASS-UNAUDITED"
            verdicts[v] = verdicts.get(v, 0) + 1
            print(f"  L{level} {digest}  {v:<14} seen {len(k['where'])}x, first {k['where'][0]}")
            if v not in ("PASS",):
                detail = [l.strip() for l in text.splitlines()
                          if any(s in l for s in ("ILLEGAL", "MISMATCH", "RAISED", "numerics", "audit"))]
                for l in detail[:3]:
                    print(f"      {l[:200]}")
    print("summary: " + ", ".join(f"{v} {n}" for v, n in sorted(verdicts.items())))
    return 0 if set(verdicts) <= {"PASS"} else 1


if __name__ == "__main__":
    sys.exit(main())
