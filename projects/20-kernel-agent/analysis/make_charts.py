"""Draws the charts in RESULTS.md from the measured numbers (run logs, held-out check, attempt logs).

    python analysis/make_charts.py        # writes assets/chart_*.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "..", "assets")
os.makedirs(OUT, exist_ok=True)
GREY, GREEN, RED, BLUE = "#b0b0b0", "#2e9e5b", "#d9534f", "#3b7dd8"
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=150, facecolor="white")
    plt.close(fig)


# 1. Before vs after: share of runs solved per level -----------------------------------------
levels = ["1\npooling", "2\ntranspose", "3\nmatmul\n1 tile", "4\nmatmul\ntiled", "5\nless\ntraffic",
          "6\nless\ntraffic", "7\nleast\ntraffic", "8\nattention", "9\ntranspose\n(new)", "10\nsoftmax\n(new)",
          "11\nscores\n(new)"]
before = [0, 3 / 5, 0, 0, 0, 0, 0, 0, 0, 0, 0]   # level 2: the repo measured 2/5 to 4/5 with the original agent
after = [2 / 2, 5 / 6, 2 / 2, 5 / 6, 2 / 2, 2 / 2, 2 / 2, 0, 2 / 2, 2 / 2, 4 / 4]
after_lbl = ["2/2", "5/6", "2/2", "5/6", "2/2", "2/2", "2/2", "0", "2/2", "2/2", "4/4"]
fig, ax = plt.subplots(figsize=(13, 5))
x = range(len(levels))
ax.bar([i - 0.2 for i in x], before, 0.4, color=GREY, label="before our changes")
ax.bar([i + 0.2 for i in x], after, 0.4, color=GREEN, label="after")
for i, t in enumerate(after_lbl):
    ax.text(i + 0.2, after[i] + 0.03, t, ha="center", fontsize=11)
for i in range(len(levels)):
    if i in (4, 5, 6):
        ax.text(i - 0.2, 0.03, "never\nreached", ha="center", fontsize=8, color="#666")
    elif i == 1:
        ax.text(i - 0.2, before[i] + 0.03, "2-4/5", ha="center", fontsize=10, color="#555")
    else:
        ax.text(i - 0.2, 0.02, "0", ha="center", fontsize=10, color="#555")
ax.set_xticks(list(x)); ax.set_xticklabels(levels, fontsize=10)
ax.set_ylim(0, 1.35); ax.set_ylabel("share of runs solved")
ax.set_title("Before vs after: the same model now solves 10 of 11 levels")
ax.legend(loc="upper center", ncol=2, frameon=False)
save(fig, "chart_1_before_after.png")

# 2. How fast: round of the first solve, per solved run ---------------------------------------
solves = {"1": [4, 4], "2": [0, 2, 0, 0, 1], "3": [1, 2], "4": [1, 1], "5": [0, 0], "6": [0, 1], "7": [1, 1],
          "9": [0, 0], "10": [0, 0], "11": [1, 1, 1, 1]}
fig, ax = plt.subplots(figsize=(10, 4.5))
for i, (lv, rounds) in enumerate(solves.items()):
    for j, r in enumerate(rounds):
        ax.scatter(i + (j - (len(rounds) - 1) / 2) * 0.12, r + 1, s=90, color=BLUE)
ax.set_xticks(range(len(solves))); ax.set_xticklabels([f"level {k}" for k in solves])
ax.set_ylabel("attempt number that solved it"); ax.set_ylim(0, 6); ax.set_yticks(range(1, 6))
ax.set_title("How quickly each level was solved (each dot = one run; lower is faster)")
save(fig, "chart_2_speed.png")

# 3. Where the attempts were wasted (from 866 logged attempts) --------------------------------
waste = [("all 4 samples were the\nsame code", 70), ("model sent back the code\nit was given, unchanged", 39),
         ("round repeated a failure\nalready seen twice", 26), ("model invented an API call\nthat doesn't exist", 15)]
fig, ax = plt.subplots(figsize=(10, 4.2))
ax.barh([w[0] for w in waste][::-1], [w[1] for w in waste][::-1], color=RED)
for i, (_, v) in enumerate(waste[::-1]):
    ax.text(v + 1, i, f"{v}%", va="center")
ax.set_xlim(0, 100); ax.set_xlabel("% of rounds / attempts")
ax.set_title("Where the effort went to waste (866 attempts) — and what we fixed")
save(fig, "chart_3_waste.png")

# 4. What went wrong, per level (failure types) ---------------------------------------------
cats = ["wrong tile size", "invented API call", "data in wrong memory", "out of bounds / reshape",
        "wrong numbers", "other"]
colors = ["#f0ad4e", "#d9534f", "#5bc0de", "#9370db", "#5cb85c", "#cccccc"]
per_level = {  # failure counts per level from trace_analysis.py categories, grouped into plain ones
    "1": [44 + 7, 80 + 18, 13, 0, 0, 32], "2": [39, 10 + 1, 0, 29, 16, 5],
    "3": [41 + 48 + 31, 0, 0, 32, 0, 0], "4": [12 + 10 + 66, 2, 1, 16, 1 + 16, 0],
    "5": [32 + 11 + 16, 4, 13, 0, 4, 8 + 2], "6": [13, 0, 0, 0, 0, 5], "7": [12, 0, 0, 0, 0, 0],
    "8": [3, 21 + 16, 33 + 16, 4 + 6, 0, 5 + 3 + 1], "9": [7, 0, 1 + 12, 0, 0, 0]}
fig, ax = plt.subplots(figsize=(11, 5))
bottom = [0] * len(per_level)
for c, (name, col) in enumerate(zip(cats, colors)):
    vals = [v[c] for v in per_level.values()]
    ax.bar([f"level {k}" for k in per_level], vals, bottom=bottom, color=col, label=name)
    bottom = [b + v for b, v in zip(bottom, vals)]
ax.set_ylabel("failed attempts"); ax.legend(frameon=False, fontsize=10)
ax.set_title("What the model got wrong, by level")
save(fig, "chart_4_failures.png")

# 5. Unseen tests: share of held-out cases passed per level -----------------------------------
hold = {"1": (28, 29), "2": (24, 25), "3": (17, 17), "4": (22, 26), "5": (22, 26), "6": (22, 26),
        "7": (22, 26), "9": (23, 23), "10": (30, 30)}
fig, ax = plt.subplots(figsize=(10, 4.5))
names = [f"level {k}" for k in hold]
passed = [p for p, t in hold.values()]; failed = [t - p for p, t in hold.values()]
ax.bar(names, passed, color=GREEN, label="passed")
ax.bar(names, failed, bottom=passed, color=RED, label="failed (shape not supported)")
for i, (p, t) in enumerate(hold.values()):
    ax.text(i, t + 0.5, f"{p}/{t}", ha="center", fontsize=10)
ax.set_ylim(0, max(t for _, t in hold.values()) + 4)
ax.set_ylabel("test cases never seen before"); ax.legend(frameon=True, loc="lower right")
ax.set_title("Tested on 228 cases the agent never saw: 210 pass, none gave wrong numbers")
save(fig, "chart_5_unseen.png")
print("charts written to", os.path.abspath(OUT))
