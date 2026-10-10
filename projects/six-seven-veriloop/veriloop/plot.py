"""M2: the result -- solve rate and best score per level, for feedback A, B and C.

    python veriloop/plot.py                 # reads results/*_summary.csv, writes results/graph.png + results/table.md

Two panels, each with its own axis: (1) how often the agent solved the level, (2) the mean best score per
run (0-1) -- on a level nobody solves, the score is where a difference in feedback can still show.
Every bar is labelled with its counts, so the chart never asks the reader to trust a colour or a height.
Offline test runs are always left out.
"""

import csv
import glob
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")

FEEDBACK = ["A", "B", "C", "D"]
NAMES = {"A": "A  pass/fail", "B": "B  how many wrong", "C": "C  where it goes wrong", "D": "D  why (diagnosis)"}
# Categorical slots 1-3 of the reference palette (validated all-pairs, light mode: CVD dE >= 9.2).
COLORS = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a", "D": "#4a3aa7"}
# Short axis names: folder names ("04_traffic_fsm (12 rounds)") collide on the axis.
SHORT = {"01_mux": "mux", "02_adder": "adder", "03_counter": "counter", "04_traffic_fsm": "traffic light",
         "05_mac": "MAC cell", "06_fifo": "FIFO"}


def short(level):
    base, _, extra = level.partition(" (")
    name = SHORT.get(base, base.split("_", 1)[-1].replace("_", " "))
    return name + ("\n(" + extra if extra else "")
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def load(paths=None):
    paths = paths or sorted(glob.glob(os.path.join(RESULTS, "*_summary.csv")))
    rows = []
    for p in paths:
        with open(p) as f:
            rows += [r for r in csv.DictReader(f) if r["offline"] != "True"]
    for r in rows:
        # Runs with a different round budget are a different setting: keep them apart. Older rows have
        # no max_rounds column; they were all 6 rounds.
        mr = r.get("max_rounds") or "6"
        if mr != "6":
            r["level"] = f"{r['level']} ({mr} rounds)"
    return rows


def cells(rows):
    out = defaultdict(lambda: dict(n=0, solved=0, scores=[], rounds=[]))
    for r in rows:
        c = out[(r["level"], r["feedback"])]
        c["n"] += 1
        c["scores"].append(float(r["best_score"]))
        if r["solved"] == "True":
            c["solved"] += 1
            c["rounds"].append(int(r["rounds_used"]))
    return out


def table_md(levels, cell, rows):
    seats = sorted({r["seat"] for r in rows})
    lines = [f"Real runs only: {len(rows)} runs on {len(seats)} seat(s). Each cell: solved / runs · mean best "
             f"score (min-max) · mean rounds when solved.", "",
             "| level | " + " | ".join(NAMES[f] for f in FEEDBACK) + " |", "|---|" + "---|" * len(FEEDBACK)]
    for lv in levels:
        parts = []
        for fb in FEEDBACK:
            c = cell.get((lv, fb))
            if not c or not c["n"]:
                parts.append("–")
                continue
            sc = c["scores"]
            rounds = f" · {sum(c['rounds']) / len(c['rounds']):.1f} rounds" if c["rounds"] else ""
            parts.append(f"{c['solved']}/{c['n']} · {sum(sc) / len(sc):.2f} ({min(sc):.2f}–{max(sc):.2f}){rounds}")
        lines.append(f"| {lv} | " + " | ".join(parts) + " |")
    return "\n".join(lines) + "\n"


def plot(levels, cell, out_png, n_runs, n_seats):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    fbs = [f for f in FEEDBACK if any(cell.get((lv, f), {}).get("n") for lv in levels)]
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 12, "text.color": INK,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    # Two panels stacked, each full width: every bar label has room (side by side they overlapped).
    fig, axes = plt.subplots(2, 1, figsize=(15, 10.5), facecolor=SURFACE)
    gap = 0.09     # room for each bar's label: with 4 bars per group, 0.03 let '1.00' labels collide
    width = min(0.12, (0.86 - gap * (len(fbs) - 1)) / len(fbs))   # thin bars; the leftover band stays air
    xs = range(len(levels))
    panels = [("Solved — runs that passed every test", lambda c: 100 * c["solved"] / c["n"], 100,
               lambda c: f"{c['solved']}/{c['n']}", "% of runs"),
              ("Mean best score per run (0–1)", lambda c: sum(c["scores"]) / len(c["scores"]), 1.0,
               lambda c: f"{sum(c['scores']) / len(c['scores']):.2f}", "score")]
    for ax, (title, value, top, label, ylab) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        for k, fb in enumerate(fbs):
            for i, lv in enumerate(levels):
                c = cell.get((lv, fb))
                if not c or not c["n"]:
                    continue
                v = value(c)
                x = i + (k - (len(fbs) - 1) / 2) * (width + gap)
                h = max(v, top * 0.006)     # a zero still shows a sliver at the baseline
                ax.add_patch(FancyBboxPatch((x - width / 2, 0), width, h, boxstyle="round,pad=0,rounding_size=0.02",
                                            mutation_aspect=top / 4, linewidth=0, facecolor=COLORS[fb]))
                ax.add_patch(plt.Rectangle((x - width / 2, 0), width, min(h, top * 0.02), linewidth=0,
                                           facecolor=COLORS[fb]))
                ax.text(x, v + top * 0.02, label(c), ha="center", va="bottom", fontsize=9, color=INK2)
        ax.set_xlim(-0.6, len(levels) - 0.4)
        ax.set_ylim(0, top * 1.15)
        ax.set_xticks(list(xs))
        ax.set_xticklabels([short(lv) for lv in levels], fontsize=12)
        ax.set_ylabel(ylab, fontsize=11)
        ax.set_title(title, loc="left", fontsize=14, color=INK, pad=12)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(length=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[f]) for f in fbs]
    fig.legend(handles, [NAMES[f] for f in fbs], loc="upper center", ncol=len(fbs), frameon=False,
               bbox_to_anchor=(0.5, 0.995), title="Feedback the model received", title_fontsize=12, fontsize=12)
    fig.text(0.01, 0.008, f"VeriLoop · Qwen3-8B on AWS Trainium · {n_runs} counted runs · up to 6 rounds × 4 attempts "
             f"per run (12 rounds where marked) · bar labels: solved/runs and mean best score", fontsize=10, color=INK2)
    fig.tight_layout(rect=(0, 0.025, 1, 0.93), h_pad=3)
    fig.savefig(out_png, dpi=150, facecolor=SURFACE)


def main():
    rows = load(sys.argv[1:] or None)
    if not rows:
        sys.exit("No real results in results/*_summary.csv yet (run veriloop/collect.sh first).")
    cell = cells(rows)
    levels = sorted({r["level"] for r in rows})
    os.makedirs(RESULTS, exist_ok=True)
    md = table_md(levels, cell, rows)
    open(os.path.join(RESULTS, "table.md"), "w").write(md)
    print(md)
    png = os.path.join(RESULTS, "graph.png")
    plot(levels, cell, png, len(rows), len({r["seat"] for r in rows}))
    print(f"wrote {os.path.relpath(png, ROOT)} and results/table.md")


if __name__ == "__main__":
    main()
