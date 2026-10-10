"""report.py -- turn the attempt logs into the hand-in numbers.

    python report.py 2048 runs/*_2048_v2.jsonl      # table, plus results/curve_2048.csv / .png / summary

Per arm: runs, final best time (median / min / max), how many runs beat the start, how many
reached within 2% of the best setting the grid found, median attempts to get there, and where
rejected attempts stopped.
"""

import collections
import csv
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def load(paths):
    rows = []
    for p in paths:
        rows += [json.loads(l) for l in open(p) if l.strip()]
    return rows


def main():
    args = sys.argv[1:]
    tag = ""
    if "--tag" in args:                      # e.g. --tag s30 -> results/summary_s30_<shape>.json
        tag = args.pop(args.index("--tag") + 1) + "_"
        args.remove("--tag")
    size = args[0]
    rows = [r for r in load(args[1:]) if str(r.get("size", size)) == size]
    grid = [json.loads(l) for l in open(os.path.join(HERE, "results", f"grid_{size}.jsonl")) if l.strip()]
    timed = [g for g in grid if g.get("time_us")]
    optimum = min(g["time_us"] for g in timed)
    target = optimum * 1.02
    print(f"grid: {len(grid)} settings measured, {len(timed)} ran, {len(grid) - len(timed)} failed to compile; "
          f"best {optimum:.1f} us, within-2% bar {target:.1f} us")

    runs = collections.defaultdict(list)
    for r in rows:
        runs[(r["arm"], r["run"])].append(r)
    arms = collections.defaultdict(list)
    for (arm, _), rs in runs.items():
        rs = sorted(rs, key=lambda r: r["attempt"])
        if rs[-1]["attempt"] == max(r["attempt"] for r in rows):      # an unfinished run is not a result
            arms[arm].append(rs)

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", f"curve_{tag}{size}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "run", "attempt", "best_time_us_so_far"])
        for arm, rr in sorted(arms.items()):
            for rs in rr:
                for r in rs:
                    w.writerow([arm, r["run"], r["attempt"], round(r["best_time_us_so_far"], 1)])

    print(f"\n{'arm':9s} {'runs':>4s} {'final median':>12s} {'min':>8s} {'max':>8s} {'beat start':>10s} "
          f"{'within 2%':>9s} {'median attempts to 2%':>22s}  rejected attempts by stage")
    summary = {}
    for arm, rr in sorted(arms.items()):
        start = rr[0][0]["time_us"]
        finals = [rs[-1]["best_time_us_so_far"] for rs in rr]
        hit = []
        for rs in rr:
            first = next((r["attempt"] for r in rs if r["best_time_us_so_far"] <= target), None)
            hit.append(first)
        reached = [h for h in hit if h is not None]
        after3 = [next(r["best_time_us_so_far"] for r in reversed(rs) if r["attempt"] <= 3) for rs in rr]
        within10 = sum(f <= optimum * 1.10 for f in finals)
        stages = collections.Counter(r["stage_reached"] for rs in rr for r in rs if r["attempt"] > 0 and r["stage_reached"] != "timed")
        summary[arm] = dict(runs=len(rr), start_us=start, final_median_us=statistics.median(finals), final_min_us=min(finals),
                            final_max_us=max(finals), beat_start=sum(f < start for f in finals), within_2pct=len(reached),
                            median_attempts_to_2pct=statistics.median(reached) if reached else None,
                            median_after_3_attempts_us=statistics.median(after3), within_10pct=within10,
                            rejected=dict(stages), finals_us=[round(f, 1) for f in finals], attempts_to_2pct=hit)
        print(f"{arm:9s} {len(rr):4d} {statistics.median(finals):12.1f} {min(finals):8.1f} {max(finals):8.1f} "
              f"{sum(f < start for f in finals):7d}/{len(rr):<2d} {len(reached):6d}/{len(rr):<2d} "
              f"{(statistics.median(reached) if reached else float('nan')):22.1f}  {dict(stages)}")
    json.dump(dict(size=size, grid_best_us=optimum, bar_us=target, arms=summary),
              open(os.path.join(HERE, "results", f"summary_{tag}{size}.json"), "w"), indent=1)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not installed; curve.csv written, no PNG")
        return
    fig, ax = plt.subplots(figsize=(9, 6.2), dpi=150)
    colours = dict(random="#8a8f98", raw="#d97706", template="#b45309", checker="#dc2626", checker2="#7c3aed",
                   checker3="#2563eb", checker4="#059669", checker5="#0891b2", greedy="#111827", greedy5="#6b7280")
    names = dict(random="random choice (no model)", raw="optimiser on raw profile", template="optimiser + fixed sentence",
                 checker="checker v1: raw profile -> model", checker2="checker v2: measured comparisons -> model",
                 checker3="checker v3: model picks from a measured menu", checker4="checker v4: v3 + edit verified",
                 checker5="checker v5: v4 + far moves, names its move", greedy="menu + fixed rule (no model)",
                 greedy5="v5 menu + fixed rule (no model)")
    for arm, rr in sorted(arms.items()):
        n = max(len(rs) for rs in rr)
        med = []
        for i in range(n):
            vals = [rs[min(i, len(rs) - 1)]["best_time_us_so_far"] for rs in rr]
            med.append(statistics.median(vals))
        ax.plot(range(n), med, marker="o", ms=3.5, lw=2, color=colours.get(arm), label=f"{names.get(arm, arm)} (median of {len(rr)} runs)",
                ls="--" if arm in ("random", "greedy", "greedy5") else "-")
    ax.axhline(optimum, color="#111827", lw=1, ls="--")
    ax.text(0.1, optimum, f" best setting on the grid: {optimum:.0f} µs", va="bottom", fontsize=8)
    ax.set_xlabel("attempts measured on the chip")
    ax.set_ylabel("best kernel time so far (µs)")
    ax.set_title(f"Matmul {size} on Trainium2: best time found vs attempts")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    hi = max(statistics.median(rs[min(1, len(rs) - 1)]["best_time_us_so_far"] for rs in rr) for rr in arms.values())
    ax.set_ylim(optimum * 0.97, hi * 1.05)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.32)
    fig.savefig(os.path.join(HERE, "results", f"curve_{tag}{size}.png"))
    print("\nwrote results/curve.csv, results/summary.json, results/curve.png")


if __name__ == "__main__":
    main()
