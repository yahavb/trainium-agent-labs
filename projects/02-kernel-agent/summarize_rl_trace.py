#!/usr/bin/env python3
"""Summarize rl_trace_agent.py JSONL logs (v3 to v6 rows) without pandas.

  python summarize_rl_trace.py logs/*.jsonl
  python summarize_rl_trace.py logs/*.jsonl --level 2          # one level
  python summarize_rl_trace.py logs/*.jsonl --brief            # arm table only

What changed from v5, and why:

  * The headline number is PER SAMPLE, not "verified in k of 5 episodes". Round 0 has no feedback
    and no earlier attempt, so its samples are independent draws of the prompt: pass@1 and pass@k
    are estimated from them with the unbiased estimator, and the confidence interval is a bootstrap
    over EPISODES (samples inside an episode share a prompt, so they are not independent).
  * Episodes solved is reported with a Wilson 95% interval, so 3/5 vs 5/5 is shown for what it is.
  * "solved_attempts" counted cached grades and byte-identical samples as separate wins. It is gone:
    solved counts DISTINCT kernels, and cached/duplicate shares are printed beside it.
  * Arms are labelled by what actually differed (mode, hints, variants, analysis, ...), and the
    comparison prints a warning when two arms differ in more than the factor being compared.
  * trace_reward is only reported when --trace was on (it was a misleading 0.0 otherwise).
"""
import argparse
import json
import math
import random
import statistics
from collections import Counter, defaultdict

SOLVED = 0.999


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
    solved = [r["round"] for r in rs if float(r["kernel_reward"]) >= SOLVED]
    return min(solved) if solved else None


# ---------------------------------------------------------------------------- statistics

def pass_at_k(n, c, k):
    """Unbiased pass@k (Chen et al. 2021): probability that at least one of k samples drawn from n,
    c of which are correct, is correct."""
    if n <= 0 or k <= 0:
        return float("nan")
    k = min(k, n)
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def wilson(k, n, z=1.96):
    """95% Wilson score interval for a binomial proportion. (nan, nan) when n == 0."""
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def cluster_bootstrap(values, reps=2000, seed=0):
    """Percentile 95% interval for the mean of per-episode values, resampling episodes."""
    vals = [v for v in values if not math.isnan(v)]
    if len(vals) < 2:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    means = sorted(mean([vals[rng.randrange(len(vals))] for _ in vals]) for _ in range(reps))
    return means[int(0.025 * reps)], means[int(0.975 * reps) - 1]


# ---------------------------------------------------------------------------- arms

def arm_label(r):
    """mode/hints plus every setting that deviates from that mode's norm."""
    label = f"{r.get('mode', '?')}/{r.get('hints', '?')}"
    cfg = r.get("config") or {}
    mode = r.get("mode")
    extra = []
    if mode != "plain":
        if cfg.get("variants") is False:
            extra.append("no-variants")
        if cfg.get("analysis") is False:
            extra.append("no-analysis")
        if cfg.get("seed_refs"):
            extra.append("seed-refs")
    if cfg.get("same_temp"):
        extra.append("same-temp")
    if cfg.get("escalate") is False:
        extra.append("no-escalate")
    if cfg.get("restart") is False:
        extra.append("no-restart")
    return label + ("".join("+" + e for e in extra))


def arm_description(r):
    cfg = r.get("config") or {}
    mode = r.get("mode")
    if mode == "plain":
        return ("agent.py prompts only; no rules card, no variant hints, no duplicate re-ask, no lint, "
                "no probe, no located analysis")
    parts = ["rules card"]
    h = r.get("hints")
    if h in ("api", "shape", "algo"):
        parts.append("API signatures")
    if h in ("shape", "algo"):
        parts.append("shape rule")
    if h == "algo":
        parts.append("LEVEL-2 ALGORITHM NOTE")
    parts.append("variant hints" if cfg.get("variants", True) else "no variant hints")
    parts.append("lint+probe+located analysis" if cfg.get("analysis", True) else "NO analysis")
    return ", ".join(parts)


def episode_stats(rs):
    """One episode -> dict of the numbers the tables need."""
    r0 = [r for r in rs if r["round"] == 0]
    n0 = len(r0)
    c0 = sum(float(r["kernel_reward"]) >= SOLVED for r in r0)
    solved_round = first_verified_round(rs)
    holds = [r.get("holdout") for r in rs
             if float(r["kernel_reward"]) >= SOLVED and r.get("holdout")
             and r["holdout"].get("available")]
    return {
        "n0": n0, "c0": c0, "solved": solved_round is not None, "solved_round": solved_round,
        "samples": len(rs), "rounds": 1 + max(r["round"] for r in rs),
        "best": max(float(r["kernel_reward"]) for r in rs),
        "holdout_ok": (all(h.get("ok") for h in holds) if holds else None),
        "escalated": any(r.get("escalation", 0) for r in rs),
    }


def fmt_ci(lo, hi):
    return "n/a" if math.isnan(lo) else f"[{lo:.2f},{hi:.2f}]"


def arm_table(rows, ks=(1, 4)):
    """Rows grouped by (arm, level) -> printable lines. Also returns the grouped stats."""
    grouped = defaultdict(list)
    for key, rs in episodes_of(rows).items():
        grouped[(arm_label(rs[0]), key[2])].append((rs, episode_stats(rs)))
    lines = [f"  {'arm':32s} {'lvl':>3s} {'eps':>3s} {'solved':>7s} {'95% CI':>12s} "
             f"{'pass@1 r0':>9s} {'95% CI':>12s} {'pass@4 r0':>9s} {'1st rnd':>7s} {'smp/ep':>6s}"]
    stats = {}
    for (arm, lvl), items in sorted(grouped.items(), key=lambda kv: (kv[0][1] or 0, kv[0][0])):
        st = [s for _, s in items]
        n_eps = len(st)
        solved = sum(s["solved"] for s in st)
        slo, shi = wilson(solved, n_eps)
        p1 = [pass_at_k(s["n0"], s["c0"], 1) for s in st if s["n0"]]
        p4 = [pass_at_k(s["n0"], s["c0"], 4) for s in st if s["n0"]]
        blo, bhi = cluster_bootstrap(p1)
        rounds = [s["solved_round"] for s in st if s["solved"]]
        lines.append(f"  {arm:32s} {lvl!s:>3s} {n_eps:>3d} {solved:>3d}/{n_eps:<3d} "
                     f"{fmt_ci(slo, shi):>12s} {mean(p1):>9.2f} {fmt_ci(blo, bhi):>12s} "
                     f"{mean(p4):>9.2f} {mean(rounds) if rounds else float('nan'):>7.1f} "
                     f"{mean([s['samples'] for s in st]):>6.1f}")
        stats[(arm, lvl)] = dict(episodes=n_eps, solved=solved, pass1=mean(p1), pass4=mean(p4),
                                 ci_solved=(slo, shi), ci_pass1=(blo, bhi))
    return lines, stats


def confound_report(rows):
    arms = {}
    for r in rows:
        arms.setdefault(arm_label(r), r)
    if len(arms) < 2:
        return []
    out = ["Arms and what each one actually contained (so a gap is not attributed to the wrong "
           "factor):"]
    for label, r in sorted(arms.items()):
        out.append(f"  {label:32s} {arm_description(r)}")
    modes = {r.get("mode") for r in arms.values()}
    if "plain" in modes and len(modes) > 1:
        out.append("  WARNING: 'plain' differs from every other mode in several things at once "
                   "(prompt cards, variants, analysis). To isolate one factor, compare arms that "
                   "differ in ONE of them, e.g. reflect/api vs reflect/api+no-analysis.")
    escal = [r for r in rows if r.get("hints_effective") and r.get("hints_effective") != r.get("hints")]
    if escal:
        out.append(f"  NOTE: {len(escal)} rows ran with a hint tier ABOVE the arm's label "
                   f"(restart escalation); read 'hints_effective' before crediting the arm.")
    algo = [r for r in rows if r.get("hints_effective", r.get("hints")) == "algo"]
    if algo:
        out.append(f"  WARNING: {len(algo)} rows were given the level-2 algorithm note, which is "
                   f"close to the answer.")
    return out


# ---------------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser()
    p.add_argument("logs", nargs="+")
    p.add_argument("--level", type=int, default=None, help="only this level")
    p.add_argument("--brief", action="store_true", help="print only the arm table")
    a = p.parse_args()
    rows = load(a.logs)
    if a.level is not None:
        rows = [r for r in rows if r.get("level") == a.level]
    if not rows:
        raise SystemExit("No valid JSONL rows found.")

    distinct_rows = [r for r in rows if not r.get("duplicate")]
    solved_rows = [r for r in rows if float(r["kernel_reward"]) >= SOLVED]
    solved_distinct = {(r["_file"], r.get("run_id"), r.get("level"), r.get("episode"),
                        r.get("fingerprint") or r.get("code")) for r in solved_rows}

    if not a.brief:
        print(f"attempts: {len(rows)}   (distinct within their round: {len(distinct_rows)}, "
              f"grades served from cache: {sum(bool(r.get('grade_cached')) for r in rows)})")
        for key in ("kernel_reward", "progress", "rl_reward"):
            vals = [float(r[key]) for r in rows if key in r]
            if vals:
                print(f"{key}: mean={mean(vals):.4f} best={max(vals):.4f}")
        if any(r.get("trace_enabled") for r in rows):
            vals = [float(r["trace_reward"]) for r in rows if r.get("trace_enabled")]
            print(f"trace_reward (--trace rows only): mean={mean(vals):.4f} best={max(vals):.4f}")
        else:
            print("trace_reward: not measured (--trace was off)")

        print("\nBy level (a solved sample is counted once per distinct kernel and episode):")
        by_level = defaultdict(list)
        for r in rows:
            by_level[r.get("level")].append(r)
        for lvl in sorted(by_level, key=lambda x: (x is None, x)):
            rs = by_level[lvl]
            sd = {k for k in solved_distinct if k[2] == lvl}
            print(f"  level {lvl}: attempts={len(rs)} mean_kernel_reward="
                  f"{mean([float(r['kernel_reward']) for r in rs]):.3f} "
                  f"distinct_solving_kernels={len(sd)} "
                  f"(solving attempts incl. duplicates and cache hits: "
                  f"{sum(float(r['kernel_reward']) >= SOLVED for r in rs)})")

        print("\nEpisodes (attempts to first verified kernel; round is 0-based):")
        per_level = defaultdict(lambda: [0, 0, []])
        for (f, run, lvl, ep), rs in sorted(episodes_of(rows).items(),
                                            key=lambda kv: (str(kv[0][0]), str(kv[0][1]),
                                                            kv[0][2] or 0, kv[0][3])):
            fr = first_verified_round(rs)
            best = max(float(r["kernel_reward"]) for r in rs)
            extra = ""
            hold = [r["holdout"] for r in rs if float(r["kernel_reward"]) >= SOLVED and r.get("holdout")
                    and r["holdout"].get("available")]
            if hold:
                h = hold[0]
                extra = f" holdout={h['passed']}/{h['total']}"
            print(f"  {f} run={run} level={lvl} ep={ep}: best={best:.2f} "
                  f"first_verified_round={'never' if fr is None else fr} "
                  f"rounds={1 + max(r['round'] for r in rs)}{extra}")
            per_level[lvl][0] += 1
            per_level[lvl][1] += fr is not None
            if fr is not None:
                per_level[lvl][2].append(fr)
        for lvl in sorted(per_level, key=lambda x: (x is None, x)):
            n_eps, n_ok, rounds = per_level[lvl]
            print(f"  level {lvl}: verified in {n_ok}/{n_eps} episodes"
                  + (f", mean first round {mean(rounds):.1f}" if rounds else ""))

        print("\nBy strategy (distinct samples):")
        strat = defaultdict(list)
        for r in distinct_rows:
            strat[r.get("action", "?")].append(r)
        for k, rs in sorted(strat.items()):
            print(f"  {k:16s} n={len(rs):3d} mean_RL={mean([float(r['rl_reward']) for r in rs]):.3f} "
                  f"mean_kernel={mean([float(r['kernel_reward']) for r in rs]):.3f}")

        print("\nBy sampling temperature (v5+ rows hold the per-sample value, rounded to 0.1):")
        temps = defaultdict(list)
        for r in rows:
            temps[round(float(r.get("temperature", 0)), 1)].append(r)
        for k, rs in sorted(temps.items()):
            print(f"  t={k:.1f}  n={len(rs):3d} mean_RL={mean([float(r['rl_reward']) for r in rs]):.3f} "
                  f"mean_kernel={mean([float(r['kernel_reward']) for r in rs]):.3f} "
                  f"pass@1={mean([float(r['kernel_reward']) >= SOLVED for r in rs]):.2f}")

        print("\nFailure categories (all attempts / distinct):")
        cat_all = Counter(r.get("failure_category", "?") for r in rows)
        cat_dst = Counter(r.get("failure_category", "?") for r in distinct_rows)
        for k, v in cat_all.most_common():
            print(f"  {k:16s} {v:4d} ({v / len(rows):4.0%})   distinct {cat_dst.get(k, 0):4d}")

        v4 = [r for r in rows if "repeated" in r]
        if v4:
            print("\nHygiene (v4+ rows):")
            print(f"  repeated identical code: {sum(bool(r['repeated']) for r in v4)}/{len(v4)}")
            print(f"  token-limit cut-offs:    {sum(bool(r.get('truncated')) for r in v4)}/{len(v4)}")
            print(f"  module-level code stripped: "
                  f"{sum(bool(r.get('sanitizer_dropped')) for r in v4)}/{len(v4)}")
            restarted = {(r['_file'], r.get('run_id'), r.get('level'), r.get('episode'), r['round'])
                         for r in v4 if r.get('stuck')}
            print(f"  rounds that restarted after identical output: {len(restarted)}")
            print("  modes: " + ", ".join(f"{m}={c}" for m, c in
                                          sorted(Counter(r.get('mode', '?') for r in v4).items())))

        v5 = [r for r in rows if "group_distinct" in r]
        if v5:
            rounds = defaultdict(list)
            for r in v5:
                rounds[(r["_file"], r.get("run_id"), r.get("level"), r.get("episode"),
                        r["round"])].append(r)
            same = sum(1 for g in rounds.values() if g[0]["group_distinct"] == 1 and len(g) > 1)
            multi = sum(1 for g in rounds.values() if len(g) > 1)
            adv = [abs(float(r["advantage"])) for r in v5 if not r.get("duplicate")]
            print("\nLearning signal (is there anything to learn from?):")
            print(f"  rounds where every sample was the same kernel: {same}/{multi}")
            print(f"  mean distinct kernels per round:               "
                  f"{mean([g[0]['group_distinct'] for g in rounds.values()]):.2f}")
            print(f"  mean |advantage| over distinct kernels:        {mean(adv):.3f}")
            print(f"  rounds re-asked after duplicates:              "
                  f"{sum(1 for g in rounds.values() if g[0].get('resampled'))}/{len(rounds)}")
            print(f"  grades served from cache (no simulator run):   "
                  f"{sum(bool(r.get('grade_cached')) for r in v5)}/{len(v5)}")

            later = [r for r in v5 if r["round"] > 0]
            if later:
                src = Counter(r.get("diagnosis_source") or "none" for r in later
                              if r["sample"] == 0)
                print("\nWhere the repair guidance came from (rounds after the first):")
                for k, v in sorted(src.items()):
                    print(f"  {k:9s} {v}")
                fail = [r for r in later if not float(r["kernel_reward"]) >= SOLVED]
                print(f"  failing samples with a static finding (api or hard-coded size): "
                      f"{sum(bool(r.get('lint') or r.get('shape_lint')) for r in fail)}/{len(fail)}")
                print(f"  rows run after a restart: {sum(1 for r in later if r.get('stuck'))}; "
                      f"on an escalated hint tier: "
                      f"{sum(1 for r in later if r.get('hints_effective') not in (None, r.get('hints')))}")

        holds = [r for r in rows if float(r["kernel_reward"]) >= SOLVED and r.get("holdout")
                 and r["holdout"].get("available")]
        if holds:
            full = sum(1 for r in holds if r["holdout"].get("ok"))
            print(f"\nHeld-out shapes: {full}/{len(holds)} solving samples also pass every "
                  f"held-out shape the reference passes.")
            if full < len(holds):
                print("  The rest pass the checker's shapes and are wrong elsewhere: they overfit.")

        lint_on_solved = sum(1 for r in solved_rows if r.get("lint") or r.get("shape_lint"))
        if lint_on_solved:
            print(f"\nWARNING: {lint_on_solved} verified kernels carried a lint finding; the lint "
                  f"flagged code the simulator accepted, so treat that rule as advisory.")
        print()

    lines, _ = arm_table(rows)
    print("Arm comparison. One row per arm and level. 'solved' counts episodes that reached a "
          "verified kernel inside the round budget; 'pass@k r0' uses round-0 samples only (no "
          "feedback, so independent draws); intervals are Wilson for solved and a bootstrap over "
          "episodes for pass@1. With few episodes the intervals are wide: read them.")
    print("\n".join(lines))
    print()
    notes = confound_report(rows)
    if notes:
        print("\n".join(notes))


if __name__ == "__main__":
    main()