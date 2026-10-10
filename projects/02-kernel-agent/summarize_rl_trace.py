#!/usr/bin/env python3
"""Summarize rl_trace_agent.py JSONL logs (v3 to v5 rows) without pandas.

  python summarize_rl_trace.py rl_level2_v5.jsonl [more.jsonl ...]

Pass the logs of several arms (--mode reflect / bandit / plain) to get the comparison table at the end.
"""
import argparse
import json
import statistics
from collections import Counter, defaultdict


def mean(xs):
    return statistics.mean(xs) if xs else float("nan")


def load(paths):
    rows = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    print(f"warning: skipped malformed line in {path}")
                    continue
                r["_file"] = path
                rows.append(r)
    return rows


def episodes_of(rows):
    """(file, run_id, level, episode) -> rows. v3 rows have no run_id: one episode per file+level."""
    eps = defaultdict(list)
    for r in rows:
        eps[(r["_file"], r.get("run_id", "v3"), r.get("level"), r.get("episode", 0))].append(r)
    return eps


def first_verified_round(rs):
    solved = [r["round"] for r in rs if float(r["kernel_reward"]) >= .999]
    return min(solved) if solved else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("logs", nargs="+")
    rows = load(p.parse_args().logs)
    if not rows:
        raise SystemExit("No valid JSONL rows found.")

    print(f"attempts: {len(rows)}")
    for key in ("kernel_reward", "progress", "trace_reward", "rl_reward"):
        vals = [float(r[key]) for r in rows if key in r]
        if vals:
            print(f"{key}: mean={mean(vals):.4f} best={max(vals):.4f}")

    print("\nBy level:")
    groups = defaultdict(list)
    for r in rows:
        groups[r.get("level")].append(r)
    for level, rs in sorted(groups.items(), key=lambda x: str(x[0])):
        kr = [float(r["kernel_reward"]) for r in rs]
        print(f"  level {level}: n={len(rs)} mean_kernel_reward={mean(kr):.3f} "
              f"solved_attempts={sum(x >= .999 for x in kr)}/{len(rs)}")

    print("\nEpisodes (attempts to first verified kernel; round is 0-based):")
    eps = episodes_of(rows)
    by_level = defaultdict(list)
    for (f, run, level, ep), rs in sorted(eps.items(), key=lambda x: str(x[0])):
        first = first_verified_round(rs)
        by_level[level].append(first)
        best = max(float(r["kernel_reward"]) for r in rs)
        print(f"  {f} run={run} level={level} ep={ep}: best={best:.2f} "
              f"first_verified_round={first if first is not None else 'never'} "
              f"rounds={1 + max(r['round'] for r in rs)}")
    for level, firsts in sorted(by_level.items(), key=lambda x: str(x[0])):
        got = [x for x in firsts if x is not None]
        print(f"  level {level}: verified in {len(got)}/{len(firsts)} episodes"
              + (f", mean first round {mean(got):.1f}" if got else ""))

    print("\nBy strategy:")
    groups = defaultdict(list)
    for r in rows:
        groups[r.get("action", "unknown")].append(r)
    for action, rs in sorted(groups.items()):
        print(f"  {action:18s} n={len(rs):3d} mean_RL={mean([float(r['rl_reward']) for r in rs]):.3f} "
              f"mean_kernel={mean([float(r['kernel_reward']) for r in rs]):.3f}")

    temps = [r for r in rows if "temperature" in r]
    if temps:
        print("\nBy sampling temperature (v5 rows hold the per-sample value, rounded to 0.1):")
        groups = defaultdict(list)
        for r in temps:
            groups[round(float(r["temperature"]), 1)].append(r)
        for t, rs in sorted(groups.items()):
            print(f"  t={t:<4} n={len(rs):3d} mean_RL={mean([float(r['rl_reward']) for r in rs]):.3f} "
                  f"mean_kernel={mean([float(r['kernel_reward']) for r in rs]):.3f}")

    print("\nFailure categories:")
    cats = Counter(r.get("failure_category", "unknown") for r in rows)
    for c, n in cats.most_common():
        print(f"  {c:14s} {n:3d}  ({100 * n / len(rows):.0f}%)")

    v4 = [r for r in rows if r.get("schema_version", 0) >= 4]
    if v4:
        print("\nHygiene (v4+ rows):")
        print(f"  repeated identical code: {sum(bool(r.get('repeated')) for r in v4)}/{len(v4)}")
        print(f"  token-limit cut-offs:    {sum(bool(r.get('truncated')) for r in v4)}/{len(v4)}")
        print(f"  module-level code stripped: {sum(bool(r.get('sanitizer_dropped')) for r in v4)}/{len(v4)}")
        print(f"  rounds that restarted after identical output: "
              f"{sum(bool(r.get('stuck')) for r in v4 if r.get('sample') == 0)}")
        modes = Counter(r.get("mode", "bandit") for r in v4)
        print("  modes: " + ", ".join(f"{m}={n}" for m, n in modes.items()))

    v5 = [r for r in rows if r.get("schema_version", 0) >= 6]
    first_rows = [r for r in rows if r.get("sample") == 0 and "group_distinct" in r]
    if first_rows:
        print("\nLearning signal (is there anything to learn from?):")
        flat = sum(r["group_distinct"] == 1 for r in first_rows)
        print(f"  rounds where every sample was the same kernel: {flat}/{len(first_rows)}")
        print(f"  mean distinct kernels per round:               "
              f"{mean([r['group_distinct'] for r in first_rows]):.2f}")
        print(f"  rounds re-asked after duplicates:              "
              f"{sum(bool(r.get('resampled')) for r in first_rows)}/{len(first_rows)}")
        adv = [abs(float(r["advantage"])) for r in rows if "advantage" in r]
        print(f"  mean |advantage| within groups:                {mean(adv):.3f}"
              + ("   <- 0 means no group baseline" if adv and max(adv) < 1e-9 else ""))
        print(f"  grades served from cache (no simulator run):   "
              f"{sum(bool(r.get('grade_cached')) for r in v5)}/{len(v5)}")
    if v5:
        src = Counter(r.get("diagnosis_source") or "none" for r in v5 if r.get("sample") == 0
                      and r.get("round", 0) > 0)
        print("\nWhere the repair guidance came from (rounds after the first):")
        for k, n in src.most_common():
            print(f"  {k:9s} {n}")
        lint = [r for r in v5 if r.get("failure_category") != "verified"]
        print(f"  failing samples with a static API finding: "
              f"{sum(bool(r.get('lint')) for r in lint)}/{len(lint)}")
        fp = [r for r in v5 if r.get("failure_category") == "verified" and r.get("lint")]
        if fp:
            print(f"  WARNING: {len(fp)} verified kernels carried a lint finding (false positives)")
        streaks = Counter()
        for (f, run, level, ep), rs in eps.items():
            prev, run_len = None, 0
            for r in sorted((x for x in rs if x.get("sample") == 0), key=lambda x: x["round"]):
                key = (r.get("failure_category"), (r.get("feedback") or "")[:80])
                run_len = run_len + 1 if key == prev else 1
                prev = key
            streaks[run_len] += 1
        print("  longest run of the identical failure, per episode (long = no new information got "
              "through): "
              + ", ".join(f"{k} rounds x{v}" for k, v in sorted(streaks.items())))

    arms = defaultdict(list)
    for key, rs in eps.items():
        arms[(rs[0].get("mode", "bandit"), rs[0].get("hints", "-"))].append(rs)
    if len(arms) > 1:
        print("\nArm comparison (one row per mode/hints; report the rate, not a single run):")
        print(f"  {'arm':22s} {'episodes':>8s} {'verified':>9s} {'mean first round':>17s} "
              f"{'mean best kernel':>17s}")
        for (mode, hints), group in sorted(arms.items()):
            firsts = [first_verified_round(rs) for rs in group]
            got = [x for x in firsts if x is not None]
            bests = [max(float(r["kernel_reward"]) for r in rs) for rs in group]
            print(f"  {mode + '/' + hints:22s} {len(group):8d} {len(got):4d}/{len(group):<4d} "
                  f"{(f'{mean(got):.1f}' if got else '-'):>17s} {mean(bests):17.3f}")


if __name__ == "__main__":
    main()