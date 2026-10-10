"""analyse16.py -- tables for the 16-bit in-model experiment, from the logs alone (no chip, no model).

    python analyse16.py grid16_all.jsonl runs16_checker5.jsonl [runs16_random.jsonl runs16_greedy5.jsonl]
"""
import json, statistics, sys

grid = {}
for l in open(sys.argv[1]):
    if l.strip():
        r = json.loads(l)
        grid.setdefault(tuple(r["knobs"]), []).append(r)
timed = {k: statistics.median(x["time_us"] for x in v if x.get("time_us")) for k, v in grid.items()
         if any(x.get("time_us") for x in v)}
failed = {k: v[0].get("error") for k, v in grid.items() if k not in timed}
best_k = min(timed, key=timed.get); best = timed[best_k]
ts = sorted(timed.values())
print(f"grid: {len(grid)} settings measured, {len(timed)} timed, {len(failed)} failed")
print(f"  best {best:.0f} us at {best_k}; median {statistics.median(ts):.0f}; worst {ts[-1]:.0f} at {max(timed, key=timed.get)} ({ts[-1] / best:.2f}x)")
print(f"  within 2% of best: {sum(t <= best * 1.02 for t in ts)}; within 10%: {sum(t <= best * 1.10 for t in ts)}")
kinds = {}
for k, e in failed.items():
    kind = "accuracy" if "WRONG RESULT" in (e or "") else "buffer" if "buffer" in (e or "") else "other"
    kinds.setdefault(kind, []).append(k)
for kind, ks in kinds.items():
    print(f"  failed ({kind}): {len(ks)}  e.g. {ks[:4]}")
multi = {k: [x["time_us"] for x in v if x.get("time_us")] for k, v in grid.items() if len(v) > 1}
for k, v in multi.items():
    if len(v) > 1:
        print(f"  cross-seat {k}: {[round(t) for t in v]}  spread {100 * (max(v) - min(v)) / min(v):.2f}%")

for path in sys.argv[2:]:
    runs = {}
    for l in open(path):
        if l.strip():
            r = json.loads(l); runs.setdefault((r["arm"], r["run"]), []).append(r)
    arms = {}
    for (arm, run), rows in runs.items():
        rows.sort(key=lambda r: r["attempt"])
        arms.setdefault(arm, []).append(rows)
    for arm, rr in arms.items():
        finals = sorted(rows[-1]["best_time_us_so_far"] for rows in rr)
        starts = [rows[0]["time_us"] for rows in rr]
        nomeas = sum(1 for rows in rr for r in rows[1:] if r["stage_reached"] != "timed")
        att = sum(len(rows) - 1 for rows in rr)
        stages = {}
        for rows in rr:
            for r in rows[1:]:
                stages[r["stage_reached"]] = stages.get(r["stage_reached"], 0) + 1
        print(f"{arm}: {len(rr)} runs | final median {statistics.median(finals):.0f} min {finals[0]:.0f} max {finals[-1]:.0f} us"
              f" | within 2%: {sum(f <= best * 1.02 for f in finals)}/{len(finals)}, within 10%: {sum(f <= best * 1.10 for f in finals)}/{len(finals)}"
              f" | attempts with no measurement {nomeas}/{att} {stages}")
        if len(rr) <= 10:
            for rows in rr:
                print(f"    start {rows[0]['knobs']} {rows[0]['time_us']:.0f} us -> best {rows[-1]['best_time_us_so_far']:.0f} us")
