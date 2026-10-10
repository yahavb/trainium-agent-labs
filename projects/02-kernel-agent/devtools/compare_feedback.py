#!/usr/bin/env python3
"""
compare_feedback.py -- does a set of checker flags say the same thing at two commits?

Runs measured at different commits can be pooled only if the checker told the model the same
thing at both. This takes every distinct kernel in the attempt logs, grades it with the given
flags using the code at each commit, and counts the kernels where the feedback differs.

    python devtools/compare_feedback.py OLD_COMMIT [NEW_COMMIT] [--features internal,origin]

NEW_COMMIT defaults to the working tree. A commit of `original` means the organisers' agent.py
(d11ccdf), which has no flags, so it is compared with ours run without any.

It needs a simulator. On a laptop that is the stand-in in devtools/fake_nki, and the header of the
output says so: equal feedback there shows the two versions take the same path through the same
errors, not that those errors are the real simulator's.
"""

import argparse
import glob
import importlib.util
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(os.path.dirname(HERE))
FILES = ["agent.py", "diagnose.py", "nkibench.py", "reference_level1.py", "reference_level2.py",
         "reference_level3.py", "reference_level4.py"]


def checkout(commit):
    """The project's files at `commit`, in a temporary folder. None means the working tree."""
    if commit is None:
        return HERE
    d = tempfile.mkdtemp(prefix=f"feedback-{commit}-")
    for f in FILES:
        out = subprocess.run(["git", "-C", REPO, "show", f"{commit}:projects/02-kernel-agent/{f}"],
                             capture_output=True, text=True)
        if out.returncode == 0:
            open(os.path.join(d, f), "w").write(out.stdout)
    return d


def load(folder, tag):
    for m in ("agent", "nkibench", "diagnose"):
        sys.modules.pop(m, None)
    sys.path.insert(0, folder)
    spec = importlib.util.spec_from_file_location("agent", os.path.join(folder, "agent.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.path.pop(0)
    mod.CANDIDATE_PATH = f"/tmp/_compare_{tag}_level{{level}}.py"
    return mod


_N = [0]


def feedback(mod, code, level, flags):
    # A fresh file for every grade, on both sides. Older commits wrote every candidate to one
    # path, and importlib's bytecode cache then served one kernel's code for the next whenever
    # the two had the same length and were written in the same second.
    _N[0] += 1
    mod.CANDIDATE_PATH = f"/tmp/_compare_{os.getpid()}_{_N[0]}_level{{level}}.py"
    if hasattr(mod, "FEATURES"):
        mod.FEATURES.clear()
        mod.FEATURES.update(flags)
    out = mod.grade(code, level)
    if hasattr(mod, "FEATURES"):
        mod.FEATURES.clear()
    return round(out[0], 6), out[2]


def main():
    # The organisers' agent.py hardcodes its candidate path, so redirecting it is not possible.
    # Without a bytecode cache there is nothing stale to serve: write none, and drop any left over.
    sys.dont_write_bytecode = True
    for stale in glob.glob("/tmp/__pycache__/_agent_level*") + glob.glob("/tmp/__pycache__/_compare_*"):
        try:
            os.remove(stale)
        except OSError:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("old")
    ap.add_argument("new", nargs="?")
    ap.add_argument("--features", default="internal,origin")
    ap.add_argument("--logs", default=os.path.join(REPO, "results", "seat-*", "*.jsonl"))
    a = ap.parse_args()

    original = a.old == "original"
    old_commit = "d11ccdf" if original else a.old
    flags = set() if original else {f for f in a.features.split(",") if f}
    if flags & {"facts", "origin", "internal", "pieces", "ahead"}:
        flags |= {"state", "locate"}
    if "pieces" in flags or "ahead" in flags:
        flags.add("origin")

    kernels = {}
    logs = sorted(glob.glob(a.logs))
    for path in logs:
        for line in open(path):
            if line.strip():
                r = json.loads(line)
                k = "".join((r.get("code") or "").split())
                if k:
                    kernels.setdefault((r["level"], k), r["code"])

    old = load(checkout(old_commit), "old")
    new = load(checkout(a.new), "new")
    import nki
    print(f"Feedback under flags {sorted(flags) or 'none'}")
    print(f"  old: {'organisers original, ' if original else ''}{old_commit}")
    print(f"  new: {a.new or 'working tree at ' + subprocess.run(['git', '-C', REPO, 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()}")
    print(f"  simulator: nki {getattr(nki, '__version__', '?')}"
          + ("  <-- THE STAND-IN, not the real simulator" if "fake" in str(getattr(nki, '__version__', '')) else ""))
    print(f"  kernels: {len(kernels)} distinct, from {len(logs)} logs\n")
    total, differ, examples = {}, {}, []
    for (level, _), code in sorted(kernels.items(), key=lambda kv: kv[0][0]):
        a_, b_ = feedback(old, code, level, flags), feedback(new, code, level, flags)
        total[level] = total.get(level, 0) + 1
        if a_ != b_:
            differ[level] = differ.get(level, 0) + 1
            examples.append((level, a_[1], b_[1]))
    for level in sorted(total):
        print(f"  level {level}: {total[level]:4d} kernels, feedback differs on {differ.get(level, 0)}")
    print(f"\n  TOTAL: {sum(total.values())} kernels, feedback differs on {sum(differ.values())}")
    for level, x, y in examples[:5]:
        i = next((j for j, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))
        print(f"\n  level {level}, first difference at character {i}:\n    old: ...{x[max(0, i - 80):i + 160]}\n    new: ...{y[max(0, i - 80):i + 160]}")
    sys.exit(1 if differ else 0)


if __name__ == "__main__":
    main()
