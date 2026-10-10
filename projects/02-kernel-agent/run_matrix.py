#!/usr/bin/env python3
"""
run_matrix.py -- run several ARMS over several LEVELS, resumably, and print one comparison.

An "arm" is `mode/hints` plus optional `+flag` ablations:

    plain/none                     agent.py's prompts only (the baseline)
    reflect/none                   rules card + variants + lint/probe/analysis, no API card
    reflect/api                    ... + exact call signatures
    reflect/shape                  ... + "sizes come from the input tensor"  (the v6 default)
    reflect/api+no-analysis        isolates what the computed analysis is worth
    reflect/api+no-variants        isolates what the variant hints are worth
    reflect/shape+no-escalate      a restart keeps the same hints

Why this exists. The first A/B compared `plain --hints none` with `reflect --hints api`: two things
changed at once, five episodes each, and reflect never reached its repair loop. Here:

  * arms are listed explicitly, so every ablation is one flag away;
  * every episode is independent (--independent-episodes, fresh state per arm/level/chunk);
  * arms are INTERLEAVED in chunks (arm A x5, arm B x5, arm A x5, ...), so a slow hour on the
    shared server hits every arm instead of one;
  * it resumes: it counts the episodes already in each log and runs only the shortfall;
  * the summary reports pass@1 / pass@4 on round 0 with intervals, not "verified in k of 5".

Examples (from projects/02-kernel-agent, with KERNEL_AGENT_BASE_URL set):

    # the 2x2 you meant to run, at a sample size that can show a difference
    python run_matrix.py --levels 2 --episodes 20 \\
        --arms plain/none plain/api reflect/none reflect/api

    # does level 1..4 get solved at all, and where does it break?
    python run_matrix.py --levels 1 2 3 4 --episodes 6 --arms reflect/shape

    # see the commands without running anything
    python run_matrix.py --levels 2 --episodes 10 --arms reflect/shape --dry-run

Anything after `--` is passed through to rl_trace_agent.py (e.g. -- --context 8192 --same-temp).
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
FLAG_MAP = {                       # `+name` in an arm  ->  rl_trace_agent.py flags
    "no-analysis": ["--no-analysis"],
    "no-variants": ["--no-variants"],
    "no-escalate": ["--no-escalate"],
    "no-restart": ["--no-plain-restart"],
    "same-temp": ["--same-temp"],
    "seed-refs": ["--seed-references"],
    "trace": ["--trace"],
    "algo-escalation": ["--escalate-algo"],
    "holdout-require": ["--holdout", "require"],
}


def parse_arm(text):
    head, *flags = text.split("+")
    if "/" not in head:
        raise SystemExit(f"arm {text!r}: expected mode/hints, e.g. reflect/shape")
    mode, hints = head.split("/", 1)
    if mode not in ("plain", "reflect", "bandit"):
        raise SystemExit(f"arm {text!r}: mode must be plain, reflect or bandit")
    if hints not in ("none", "api", "shape", "algo"):
        raise SystemExit(f"arm {text!r}: hints must be none, api, shape or algo")
    extra = []
    for f in flags:
        if f not in FLAG_MAP:
            raise SystemExit(f"arm {text!r}: unknown flag +{f}; known: {', '.join(sorted(FLAG_MAP))}")
        extra += FLAG_MAP[f]
    return mode, hints, extra


def slug(arm, level):
    return arm.replace("/", "_").replace("+", "-") + f"_L{level}"


def episodes_done(log_path):
    """Distinct (run_id, episode) already in a log."""
    if not os.path.exists(log_path):
        return 0
    seen = set()
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            seen.add((r.get("run_id"), r.get("episode")))
    return len(seen)


def build_command(args, arm, level, n_eps, chunk_index, passthrough):
    mode, hints, extra = parse_arm(arm)
    name = slug(arm, level)
    cmd = [sys.executable, os.path.join(HERE, "rl_trace_agent.py"),
           "--mode", mode, "--hints", hints, "--level", str(level),
           "--rounds", str(args.rounds), "--samples", str(args.samples),
           "--episodes", str(n_eps), "--patience", str(args.patience),
           "--independent-episodes", "--state", os.path.join(args.out, f"state_{name}.json"),
           "--log", os.path.join(args.out, f"{name}.jsonl"),
           "--seed", str(args.seed + 1000 * chunk_index)]
    if args.export_groups:
        cmd += ["--export-groups", os.path.join(args.out, f"groups_L{level}.jsonl")]
    return cmd + extra + passthrough


def main():
    argv = sys.argv[1:]
    passthrough = []
    if "--" in argv:
        i = argv.index("--")
        argv, passthrough = argv[:i], argv[i + 1:]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--levels", type=int, nargs="+", default=[2])
    p.add_argument("--arms", nargs="+", default=["plain/none", "reflect/none", "reflect/api",
                                                  "reflect/shape"])
    p.add_argument("--episodes", type=int, default=10, help="episodes per arm and level (total)")
    p.add_argument("--chunk", type=int, default=5, help="episodes per call; arms interleave per chunk")
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--samples", type=int, default=4)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default="matrix_runs")
    p.add_argument("--export-groups", action="store_true", help="also write GRPO group rows per level")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-summary", action="store_true")
    args = p.parse_args(argv)

    for arm in args.arms:
        parse_arm(arm)                                        # fail early on a typo
    os.makedirs(args.out, exist_ok=True)

    todo = defaultdict(int)
    for level in args.levels:
        for arm in args.arms:
            have = episodes_done(os.path.join(args.out, f"{slug(arm, level)}.jsonl"))
            todo[(arm, level)] = max(0, args.episodes - have)
            if have:
                print(f"resume: {arm} level {level}: {have} episodes already logged, "
                      f"{todo[(arm, level)]} to go")

    chunk_index = 0
    while any(todo.values()):
        for level in args.levels:
            for arm in args.arms:
                left = todo[(arm, level)]
                if left <= 0:
                    continue
                n = min(args.chunk, left)
                cmd = build_command(args, arm, level, n, chunk_index, passthrough)
                print("\n$ " + " ".join(shlex.quote(c) for c in cmd), flush=True)
                if args.dry_run:
                    todo[(arm, level)] = 0 if left <= n else left - n
                    continue
                # State is fresh per chunk: --independent-episodes clears it per episode anyway, and
                # a chunk must not inherit exemplars or API facts from another arm's run.
                state = cmd[cmd.index("--state") + 1]
                if os.path.exists(state):
                    os.remove(state)
                rc = subprocess.call(cmd, cwd=HERE)
                if rc != 0:
                    print(f"WARNING: {arm} level {level} exited with {rc}; its episodes so far are "
                          f"logged and a re-run will resume.")
                    todo[(arm, level)] = 0
                    continue
                todo[(arm, level)] = left - n
        chunk_index += 1

    if args.dry_run or args.no_summary:
        return
    logs = sorted(os.path.join(args.out, f) for f in os.listdir(args.out)
                  if f.endswith(".jsonl") and not f.startswith("groups_"))
    if logs:
        print("\n" + "=" * 100 + "\nSUMMARY\n" + "=" * 100)
        subprocess.call([sys.executable, os.path.join(HERE, "summarize_rl_trace.py"), *logs], cwd=HERE)


if __name__ == "__main__":
    main()