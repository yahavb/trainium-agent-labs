#!/usr/bin/env python3
"""
ab.py — a matched A/B experiment: baseline and improved runs ALTERNATE (A, B, A, B, ...), with the same
budget and frozen code, so a difference can be attributed to the switch and not to time or server load.

    python ab.py --level 8 --pairs 3 --b "--lint --prompt-portfolio" -- --level-hints --echo-check
                 |                    |                              '-- flags both arms share
                 |                    '-- flags only arm B gets
                 '-- level and how many A/B pairs

Each arm logs to ab-<tag>-A.jsonl / ab-<tag>-B.jsonl; every attempt already records its session, the hash
of the source files, and the flags. The script stops if the source changes mid-experiment.
"""
import argparse, hashlib, json, os, shlex, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))


def source_hash():
    d = hashlib.sha256()
    for name in ("agent.py", "nkibench.py", "lint.py"):
        fp = os.path.join(HERE, name)
        if os.path.exists(fp):
            d.update(open(fp, "rb").read())
    return d.hexdigest()[:12]


def solved_rate(path):
    """Runs solved / runs, from the log's own run ids."""
    runs = {}
    for line in open(path):
        r = json.loads(line)
        key = (r.get("session"), r.get("run", 0))
        runs[key] = runs.get(key, False) or r["reward"] >= 0.999
    return sum(runs.values()), len(runs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, required=True)
    ap.add_argument("--pairs", type=int, default=3)
    ap.add_argument("--b", default="", help="flags only arm B gets")
    ap.add_argument("--tag", default="")
    ap.add_argument("--budget", default="--rounds 8 --samples 4 --context 8192 --model Qwen/Qwen3-8B",
                    help="identical for both arms")
    a, shared = ap.parse_known_args()
    shared = [x for x in shared if x != "--"]
    tag = a.tag or f"l{a.level}"
    frozen = source_hash()
    print(f"matched A/B on level {a.level}: {a.pairs} pairs, source {frozen}\n  A: {' '.join(shared)}\n"
          f"  B: {' '.join(shared)} {a.b}")
    for i in range(a.pairs):
        for arm, extra in (("A", []), ("B", shlex.split(a.b))):
            if source_hash() != frozen:
                sys.exit("the source changed mid-experiment; results would not be comparable -- stopping")
            cmd = [sys.executable, os.path.join(HERE, "agent.py"), "--level", str(a.level), "--repeat", "1",
                   *shlex.split(a.budget), *shared, *extra, "--log", f"ab-{tag}-{arm}.jsonl"]
            print(f"pair {i + 1}/{a.pairs} arm {arm}", flush=True)
            with open(f"ab-{tag}-{arm}.log", "a") as out:
                subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT, cwd=HERE)
    for arm in ("A", "B"):
        path = os.path.join(HERE, f"ab-{tag}-{arm}.jsonl")
        if os.path.exists(path):
            k, n = solved_rate(path)
            print(f"arm {arm}: solved {k}/{n}")


if __name__ == "__main__":
    main()
