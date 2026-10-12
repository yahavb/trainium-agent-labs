"""The four figures, drawn from runs/*.jsonl and the replayed days. Re-run after any new run.

    python charts.py            # writes figures/*.png

1. verified level-runs by arm, stacked by level     (does the checker's message matter?)
2. rule-violation rate on real days, C vs C4         (did teaching the checker fix the real market?)
3. profit per episode with 95% bounds, four stocks   (does anything make money on real data?)
4. level-3 fills by side and book imbalance          (why level 3 is hard)
"""
import collections
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

import checker  # noqa: E402
import lobster  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")

# reference palette (dataviz skill), light mode
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"          # categorical slots 1-3 (validated)
DEEMPH = "#b9b7ae"                                    # de-emphasis gray for context marks
RED, MID = "#e34948", "#f0efec"                       # diverging pole and neutral midpoint

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": INK2, "axes.edgecolor": AXIS,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.spines.top": False, "axes.spines.right": False,
})

ARMS = [("A", "A  score only"), ("B", "B  raw metrics"), ("C", "C  names the cause"),
        ("C2", "C2  + no-fills rule"), ("C3", "C3  + exception names its cause"),
        ("C4", "C4  + 1-tick case in the checker")]


def title(fig, head, sub):
    fig.text(0.012, 0.975, head, ha="left", va="top", fontsize=13, fontweight="semibold", color=INK)
    fig.text(0.012, 0.915, sub, ha="left", va="top", fontsize=9.5, color=INK2)


def finals():
    rows = []
    for p in sorted(glob.glob(os.path.join(HERE, "runs", "*-[0-9].jsonl"))):
        rows += [json.loads(line) for line in open(p) if line.strip()]
    return [r for r in rows if r.get("final")]


def distinct_verified(fs, run):
    codes = {}
    for f in fs:
        if f["run"] == run and f["dev_solved"]:
            codes.setdefault(f["code"], f["level"])
    return list(codes)


# ---------------------------------------------------------------- 1

def fig_verified(fs):
    cnt = collections.Counter((f["run"], f["level"]) for f in fs if f["dev_solved"])
    fig, ax = plt.subplots(figsize=(8.6, 3.9), dpi=200)
    fig.subplots_adjust(left=0.30, right=0.97, top=0.74, bottom=0.12)
    title(fig, "Feedback that names the cause is what makes the loop work",
          "Verified level-runs out of 9 (3 levels × 3 repeats) per arm · Qwen3-8B on one Trainium2 "
          "chip · equal attempt budget")
    y = np.arange(len(ARMS))[::-1]
    for (run, label), yy in zip(ARMS, y):
        left = 0
        for lv, col in ((1, S1), (2, S2), (3, S3)):
            v = cnt[(run, lv)]
            if v:
                ax.barh(yy, v, left=left, height=0.56, color=col, edgecolor=SURFACE, linewidth=2)
            left += v
        ax.text(left + 0.12, yy, f"{left}", va="center", ha="left", fontsize=10,
                fontweight="semibold", color=INK)
    ax.set_yticks(y, [label for _, label in ARMS])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 9)
    ax.set_xticks([0, 3, 6, 9])
    ax.xaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (S1, S2, S3)]
    fig.legend(handles, ["level 1 calm", "level 2 trend", "level 3 informed flow"], ncol=3,
               loc="upper left", bbox_to_anchor=(0.30, 0.86), frameon=False, fontsize=9,
               handlelength=1.0, handleheight=1.0)
    fig.savefig(os.path.join(OUT, "fig1_verified_by_arm.png"))
    plt.close(fig)


# ---------------------------------------------------------------- 2

def fig_violations(fs):
    steps = 14 * 1500
    stocks = list(lobster.TICKERS) + list(lobster.UNSEEN)
    rate = {}
    for run in ("C", "C4"):
        codes = distinct_verified(fs, run)
        for t in stocks:
            rs = [lobster.score(c, t) for c in codes]
            rate[(run, t)] = max(100.0 * sum(r["violations"].values()) / steps for r in rs)
    fig, ax = plt.subplots(figsize=(8.6, 3.9), dpi=200)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.72, bottom=0.16)
    title(fig, "Teaching the checker the 1-tick case made the strategies legal on real markets",
          "Steps on which an agent-verified strategy broke a market rule, worst strategy per arm · "
          "LOBSTER, 2012-06-21, 14 × 1,500 s per stock")
    x = np.arange(len(stocks))
    w = 0.34
    for off, run, col, name in ((-w / 2 - 0.01, "C", DEEMPH, "C: verified before the fix (1 strategy)"),
                                (w / 2 + 0.01, "C4", S1, "C4: verified with the 1-tick layer (3 strategies)")):
        vals = [rate[(run, t)] for t in stocks]
        ax.bar(x + off, vals, width=w, color=col, label=name)
        for xi, v in zip(x + off, vals):
            ax.text(xi, v + 1.5, f"{v:.1f}%" if v >= 0.05 else "0%", ha="center", va="bottom",
                    fontsize=9, color=INK)
    ax.set_xticks(x, [f"{t}\n{'seen (spread 1 tick)' if t in lobster.TICKERS else 'unseen (spread 15–25)'}"
                      for t in stocks], color=INK2)
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.yaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.16), ncol=2, frameon=False, fontsize=9,
              handlelength=1.0, handleheight=1.0)
    fig.savefig(os.path.join(OUT, "fig2_real_market_violations.png"))
    plt.close(fig)


# ---------------------------------------------------------------- 3

def fig_profit(fs):
    stocks = list(lobster.TICKERS) + list(lobster.UNSEEN)
    entries = [("naive quoter", open(os.path.join(HERE, "baseline.py")).read(), DEEMPH),
               ("hand-written reference", open(os.path.join(HERE, "reference.py")).read(), INK2)]
    entries += [(f"agent C4, strategy {i + 1}", c, S1)
                for i, c in enumerate(distinct_verified(fs, "C4"))]
    fig, axes = plt.subplots(1, 4, figsize=(10.4, 4.0), dpi=200, sharey=True)
    fig.subplots_adjust(left=0.17, right=0.985, top=0.72, bottom=0.15, wspace=0.22)
    title(fig, "On real order books, no strategy, the agent's or ours, is profitable on every stock",
          "Profit per 1,500 s episode in ticks, mean with 95% interval (±2 standard errors), 14 "
          "episodes · each panel has its own scale")
    ys = np.arange(len(entries))[::-1]
    for ax, t in zip(axes, stocks):
        for (name, code, col), yy in zip(entries, ys):
            r = lobster.score(code, t)
            ax.plot([r["pnl"] - 2 * r["se"], r["pnl"] + 2 * r["se"]], [yy, yy], color=col,
                    linewidth=2, solid_capstyle="round")
            ax.plot(r["pnl"], yy, "o", markersize=7, color=col, markeredgecolor=SURFACE,
                    markeredgewidth=2)
        ax.axvline(0, color=AXIS, linewidth=1)
        ax.xaxis.grid(True, color=GRID, linewidth=1)
        ax.set_axisbelow(True)
        ax.set_title(f"{t}  ·  {'seen' if t in lobster.TICKERS else 'unseen'}", fontsize=10,
                     color=INK, loc="left")
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.locator_params(axis="x", nbins=4)
    axes[0].set_yticks(ys, [e[0] for e in entries])
    fig.text(0.17, 0.035, "Right of 0 with the whole interval = reliably profitable. Wide-spread "
             "stocks (GOOG, AAPL) flatter any quoter: the replay cannot compete for the spread.",
             fontsize=8.5, color=MUTED)
    fig.savefig(os.path.join(OUT, "fig3_real_market_profit.png"))
    plt.close(fig)


# ---------------------------------------------------------------- 4

def fig_table():
    rows = [json.loads(line) for line in open(os.path.join(HERE, "runs", "C4-2.jsonl"))]
    r0 = [r for r in rows if not r.get("final") and r["level"] == 3 and r["tags"] == ["LOSS_ADVERSE"]][0]
    res = checker.check(r0["code"], 3)
    n = np.array(res["metrics"]["table_n"], float)
    mean = np.array(res["metrics"]["table_sum"]) / np.maximum(n, 1)
    cmap = LinearSegmentedColormap.from_list("div", [RED, MID, S1])
    fig, ax = plt.subplots(figsize=(8.6, 3.6), dpi=200)
    fig.subplots_adjust(left=0.16, right=0.97, top=0.70, bottom=0.17)
    title(fig, "Level 3: the model sells into heavy bids and buys into heavy asks",
          "Mid-price move 10 steps after each fill, in ticks (blue = in the model's favour, red = "
          "against) · one run-C4 attempt, 16 dev episodes")
    for i in range(2):
        for j in range(3):
            v = mean[i, j]
            ax.add_patch(plt.Rectangle((j - 0.5 + 0.012, i - 0.5 + 0.02), 0.976, 0.96,
                                       color=cmap((np.clip(v, -2, 2) + 2) / 4), linewidth=0))
            ink = "#ffffff" if abs(v) > 1.1 else INK
            ax.text(j, i - 0.08, f"{v:+.2f}".replace("-", "\u2212"), ha="center", va="center", fontsize=13,
                    fontweight="semibold", color=ink)
            ax.text(j, i + 0.22, f"{int(n[i, j]):,} fills", ha="center", va="center", fontsize=8.5,
                    color=ink)
    ax.set_xticks([0, 1, 2], ["asks heavier\n(imbalance < −0.3)", "balanced",
                              "bids heavier\n(imbalance > 0.3)"])
    ax.set_yticks([0, 1], ["model bought", "model sold"])
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(1.5, -0.5)
    fig.text(0.16, 0.035, "Run C5 showed the model exactly this table as feedback. Level 3 stayed "
             "at 0 of 5: the evidence was clear, and the 8B model could not act on it.", fontsize=8.5,
             color=MUTED)
    fig.savefig(os.path.join(OUT, "fig4_level3_fill_table.png"))
    plt.close(fig)


# ---------------------------------------------------------------- 5

def fig_level3(fs):
    arms = [("C", "C  names the cause"), ("C2", "C2  + no-fills rule"),
            ("C3", "C3  + exception names its cause"), ("C4", "C4  + 1-tick layer (incl. C4b)"),
            ("C5", "C5  + its own fill table"), ("C6", "C6  + published ideas in the prompt"),
            ("C7", "C7  + 1-tick failure diagnosed")]
    rows = []
    for p in sorted(glob.glob(os.path.join(HERE, "runs", "*-[0-9].jsonl"))):
        rows += [json.loads(line) for line in open(p) if line.strip()]
    prof, ver, tot = collections.Counter(), collections.Counter(), collections.Counter()
    runs_prof = set()
    for r in rows:
        if r["level"] != 3:
            continue
        run = "C4" if r["run"] == "C4b" else r["run"]
        if r.get("final"):
            tot[run] += 1
            ver[run] += r["dev_solved"]
        elif r["stage"] in ("robust", "solved"):
            runs_prof.add((r["run"], r["rep"]))
    for run_raw, rep in runs_prof:
        prof["C4" if run_raw == "C4b" else run_raw] += 1
    arms = [a for a in arms if tot[a[0]]]
    fig, ax = plt.subplots(figsize=(8.6, 3.9), dpi=200)
    fig.subplots_adjust(left=0.33, right=0.95, top=0.70, bottom=0.12)
    title(fig, "Level 3: evidence didn't move the model; published ideas did, until the 1-tick check",
          "Share of level-3 repeats per arm · each arm keeps every fix above it")
    y = np.arange(len(arms))[::-1]
    h = 0.32
    for (run, label), yy in zip(arms, y):
        for off, val, col in ((h / 2 + 0.02, prof[run], S2), (-h / 2 - 0.02, ver[run], S1)):
            share = val / tot[run]
            ax.barh(yy + off, max(share, 0.004), height=h, color=col if share else AXIS)
            ax.text(share + 0.012, yy + off, f"{val} of {tot[run]}", va="center", ha="left",
                    fontsize=8.5, color=INK)
    ax.set_yticks(y, [label for _, label in arms])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax.xaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (S2, S1)]
    fig.legend(handles, ["profitable at the 2-tick spread", "verified (also survives 1 tick)"],
               ncol=2, loc="upper left", bbox_to_anchor=(0.33, 0.83), frameon=False, fontsize=9,
               handlelength=1.0, handleheight=1.0)
    fig.savefig(os.path.join(OUT, "fig5_level3_by_arm.png"))
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fs = finals()
    fig_verified(fs)
    fig_violations(fs)
    fig_profit(fs)
    fig_table()
    fig_level3(fs)
    print("wrote", sorted(os.listdir(OUT)))
