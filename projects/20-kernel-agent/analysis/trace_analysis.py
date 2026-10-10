"""Where did the agent's attempts go? Read-only analysis of every attempts*.jsonl we have.

Each file is one experiment (one level, one hint version, 1-3 repeated runs appended in order;
a new run starts when the round number drops back to 0).
"""
import collections, glob, json, os, re, statistics, sys

ROOT = sys.argv[1]


def runs_of(path):
    """Split an attempts file into runs, by (session, run) id when the log has them. Older logs: rows
    of one round are consecutive; a new run starts when
    the round number drops, or when round 0 already holds a full round's worth of samples (a run
    solved on round 0 is followed directly by the next run's round 0)."""
    rows = [json.loads(l) for l in open(path) if l.strip()]
    if rows and all("session" in r for r in rows):
        # Logs with provenance carry their own ids: group by them instead of guessing boundaries
        # (review finding: the guess merges runs when a log is appended to by two processes).
        by = {}
        for r in rows:
            by.setdefault((r["session"], r.get("run", 0)), []).append(r)
        return [sorted(v, key=lambda r: (r["round"], r.get("sample", 0))) for v in by.values()]
    sizes, k = collections.Counter(), 0
    for a, b in zip(rows, rows[1:] + [None]):
        k += 1
        if b is None or b["round"] != a["round"]:
            sizes[k] += 1; k = 0
    per_round = sizes.most_common(1)[0][0] if sizes else 1
    runs, cur, last, n0 = [], [], None, 0
    for r in rows:
        if cur and (r["round"] < last or (r["round"] == 0 and n0 >= per_round)):
            runs.append(cur); cur = []; n0 = 0
        cur.append(r); last = r["round"]
        n0 += r["round"] == 0
    if cur:
        runs.append(cur)
    return runs


def category(fb):
    pats = [("solved", r"^Correct on every shape"), ("traffic bar", r"TOO MUCH HBM TRAFFIC"),
            ("invented kwarg", r"unexpected keyword argument"), ("invented name", r"has no attribute"),
            ("copy size", r"same number of elements"), ("partition>128", r"partition dimension \d+ exceeds"),
            ("out of bound", r"Out-of-bound"), ("broadcast/assign", r"could not be broadcast"),
            ("reshape", r"cannot reshape"), ("1-D tile", r"at least 2 dimensions"),
            ("wrong memory", r"must be in \["), ("python op on tile", r"unsupported operand"),
            ("numerical", r"NUMERICAL MISMATCH"), ("nan", r"NON-FINITE"), ("parse", r"does not parse"),
            ("rule violation", r"Rule violations"), ("no code", r"No code came back"),
            ("transpose engine", r"transpose requires shape"), ("ap pattern", r"ap\(\) pattern"),
            ("arity", r"positional argument")]
    for name, p in pats:
        if re.search(p, fb or ""):
            return name
    return "other"


files = sorted(glob.glob(os.path.join(ROOT, "**", "*.jsonl"), recursive=True))
tot_rounds = tot_attempts = 0
distinct_hist = collections.Counter()
same_as_prev_code = same_as_prev_fb = 0
round_pairs = 0
solve_round = collections.Counter()
solved_runs = unsolved_runs = 0
wasted_after_stall = 0
cat_count = collections.Counter()
transitions_to_solve = collections.Counter()
reply_chars, prompt_chars = [], []
by_level = collections.defaultdict(lambda: dict(runs=0, solved=0, rounds=0))
for path in files:
    for run in runs_of(path):
        lvl = run[0]["level"]
        rounds = collections.OrderedDict()
        for r in run:
            rounds.setdefault(r["round"], []).append(r)
            cat_count[category(r["feedback"])] += 1
            reply_chars.append(r.get("reply_chars") or 0); prompt_chars.append(r.get("prompt_chars") or 0)
        tot_attempts += len(run); tot_rounds += len(rounds)
        best_per_round = []
        for rnd, rs in rounds.items():
            distinct_hist[len({x["code"] for x in rs})] += 1
            best_per_round.append(max(rs, key=lambda x: x["reward"]))
        by_level[lvl]["runs"] += 1; by_level[lvl]["rounds"] += len(rounds)
        solved_at = next((i for i, b in enumerate(best_per_round) if b["reward"] >= 0.999), None)
        if solved_at is not None:
            solved_runs += 1; by_level[lvl]["solved"] += 1; solve_round[solved_at] += 1
            if solved_at > 0:
                transitions_to_solve[category(best_per_round[solved_at - 1]["feedback"])] += 1
        else:
            unsolved_runs += 1
        for a, b in zip(best_per_round, best_per_round[1:]):
            round_pairs += 1
            same_as_prev_code += a["code"] == b["code"]
            same_as_prev_fb += category(a["feedback"]) == category(b["feedback"]) and a["reward"] == b["reward"]
        # rounds spent after the first time the same category+reward repeated twice in a row
        streak = 0
        for a, b in zip(best_per_round, best_per_round[1:]):
            if category(a["feedback"]) == category(b["feedback"]) and a["reward"] == b["reward"]:
                streak += 1
                if streak >= 2:
                    wasted_after_stall += 1
            else:
                streak = 0

print(f"files {len(files)}  runs {solved_runs + unsolved_runs} (solved {solved_runs}, unsolved {unsolved_runs})  "
      f"rounds {tot_rounds}  attempts {tot_attempts}")
print("\n1. SAMPLE DIVERSITY: distinct kernels among the 4 samples of a round")
for k in sorted(distinct_hist):
    print(f"   {k} distinct: {distinct_hist[k]:4d} rounds ({100 * distinct_hist[k] / tot_rounds:.0f}%)")
print("\n2. WHEN SOLVES HAPPEN (round index of first solve, solved runs only)")
for k in sorted(solve_round):
    print(f"   round {k}: {solve_round[k]}")
print("\n3. REPEATS between consecutive rounds (best attempt of each round)")
print(f"   identical kernel to previous round: {same_as_prev_code}/{round_pairs} ({100 * same_as_prev_code / max(1, round_pairs):.0f}%)")
print(f"   same failure category and reward as previous round: {same_as_prev_fb}/{round_pairs} ({100 * same_as_prev_fb / max(1, round_pairs):.0f}%)")
print(f"   rounds spent on the 3rd+ repeat of the same stalled failure: {wasted_after_stall} ({100 * wasted_after_stall / tot_rounds:.0f}% of all rounds)")
print("\n4. WHAT THE ROUND BEFORE A SOLVE LOOKED LIKE (failure category it was repairing)")
for k, v in transitions_to_solve.most_common():
    print(f"   {k}: {v}")
print("\n5. FAILURE CATEGORIES across all attempts")
for k, v in cat_count.most_common(14):
    print(f"   {k:18s} {v:4d} ({100 * v / tot_attempts:.0f}%)")
print("\n6. PER LEVEL: runs, solved, rounds spent")
for lvl in sorted(by_level):
    d = by_level[lvl]
    print(f"   level {lvl}: {d['solved']}/{d['runs']} runs solved, {d['rounds']} rounds")
print(f"\n7. SIZES: prompt chars median {statistics.median(prompt_chars):.0f} (max {max(prompt_chars)}), "
      f"reply chars median {statistics.median(reply_chars):.0f} (max {max(reply_chars)})")


# ---- cycle-aware failure history (selected sample per round) ---------------------------------------
def normalise(fb):
    """Category plus the error text with numbers stripped, so 'index 64' and 'index 32' match."""
    return category(fb) + "|" + re.sub(r"[0-9]+", "N", (fb or "").split(" This is")[0][:160])


def picked(rs):
    """The sample the loop carried forward: the logged 'selected' one, else (old logs) the best reward."""
    sel = [x for x in rs if x.get("selected")]
    return sel[0] if sel else max(rs, key=lambda x: x["reward"])


kinds = collections.Counter()
for path in files:
    for run in runs_of(path):
        rounds = collections.OrderedDict()
        for r in run:
            rounds.setdefault(r["round"], []).append(r)
        history = []
        for rs in rounds.values():
            p = picked(rs)
            if p["reward"] >= 0.999:
                break
            key = normalise(p["feedback"])
            kinds["repeat of last round" if history and history[-1] == key else
                  "RETURN to an earlier failure" if key in history else "new failure"] += 1
            history.append(key)
total = sum(kinds.values())
print("\n8. FAILURE HISTORY per round (did the loop make progress?)")
for k, v in kinds.most_common():
    print(f"   {k:30s} {v:4d} ({100 * v / max(1, total):.0f}%)")
