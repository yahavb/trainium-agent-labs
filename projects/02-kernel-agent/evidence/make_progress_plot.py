#!/usr/bin/env python3
"""Progress figure for the near-miss repair result. Output: progress.png (next to this file).

Data comes from the committed run logs in ./data/ (the unguided queue: pilot_v2 + the five
frozen repeats) and from the repair-round logs in this folder. Reproduce with:
    python3 make_progress_plot.py
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SHAPE4 = "K=256 M=512 N=1024"


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def shape4_waste(rec):
    for c in rec.get("per_case") or []:
        if c.get("case") == SHAPE4:
            return c.get("waste")
    return None


def is_valid(rec):
    if "valid" in rec:
        return bool(rec.get("valid"))
    pcs = rec.get("per_case") or []
    for c in pcs:
        f = c.get("failure") or ""
        if f and "TOO MUCH HBM TRAFFIC" not in f:
            return False
    return bool(pcs)


unguided = load(os.path.join(HERE, "data", "pilot_v2_run1_rounds.jsonl"))
for path in sorted(glob.glob(os.path.join(HERE, "data", "repeats_run*_rounds.jsonl"))):
    unguided += load(path)
repair1 = load(os.path.join(HERE, "agent_kernel_repair_rounds.jsonl"))
repair2 = load(os.path.join(HERE, "agent_kernel_surgical2_rounds.jsonl"))

records = unguided + repair1 + repair2
n_unguided = len(unguided)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 6.2))

# ---------------------------------------------------------------- panel A
labels = ["K128\nM128\nN512", "K256\nM256\nN1024", "K512\nM128\nN512", "K256\nM512\nN1024"]
seed = [1.00, 1.556, 1.00, 2.00]
winner = [1.00, 1.00, 1.00, 1.00]
xs = range(4)
ax1.bar([x - 0.19 for x in xs], seed, width=0.38, color="#9ecae1",
        label="start (correct but slow)")
ax1.bar([x + 0.19 for x in xs], winner, width=0.38, color="#2ca02c",
        label="end (verified winner)")
for x, a, b in zip(xs, seed, winner):
    ax1.text(x - 0.19, a + 0.03, f"{a:.2f}x", ha="center", fontsize=9)
    ax1.text(x + 0.19, b + 0.03, f"{b:.2f}x", ha="center", fontsize=9, color="#1b5e20")
ax1.axhline(1.0, color="black", lw=1.2, ls="--")
ax1.text(3.45, 1.02, "minimum possible (floor)", ha="right", fontsize=9)
ax1.axhline(1.6, color="#e67e22", lw=1.2, ls=":")
ax1.text(3.45, 1.62, "level-5 gate (1.60x)", ha="right", fontsize=9, color="#e67e22")
ax1.set_xticks(list(xs))
ax1.set_xticklabels(labels, fontsize=9)
ax1.set_ylabel("bytes moved / minimum possible  (lower is better)")
ax1.set_title("A. Every benchmark shape ends at the floor")
ax1.set_ylim(0, 2.35)
ax1.legend(loc="upper left", fontsize=9)

# ---------------------------------------------------------------- panel B
gray_x, gray_y = [], []
green_x, green_y = [], []
for i, rec in enumerate(records):
    w = shape4_waste(rec)
    if w is None:
        continue
    if is_valid(rec):
        green_x.append(i)
        green_y.append(w)
    else:
        gray_x.append(i)
        gray_y.append(w)

ax2.scatter(gray_x, gray_y, s=26, facecolors="none", edgecolors="#b0b0b0",
            label="attempts with wrong numbers (invalid)")
ax2.scatter(green_x, green_y, s=90, marker="*", color="#2ca02c",
            label="verified attempts (correct on every shape)")

path_x = [-1.0, min(green_x), max(green_x)]
path_y = [2.00, 1.857, 1.00]
ax2.plot(path_x, path_y, color="#2ca02c", lw=2.2, zorder=5)
ax2.scatter([-1.0], [2.00], marker="*", s=140, color="#2ca02c", zorder=6)

for y, txt, col in ((1.00, "the floor", "black"), (1.05, "level-7 gate 1.05x", "#7f7f7f"),
                    (1.25, "level-6 gate 1.25x", "#7f7f7f"), (1.60, "level-5 gate 1.60x", "#e67e22"),
                    (2.00, "seed 2.00x", "#1f77b4")):
    ax2.axhline(y, color=col, lw=1.1, ls="--" if y in (1.00, 2.00) else ":", alpha=0.8)
    ax2.text(len(records) * 0.995, y + 0.015, txt, ha="right", fontsize=8, color=col)

ax2.axvline(n_unguided - 0.5, color="#888888", lw=1.0, ls="-.")
ax2.text(n_unguided / 2, 0.62, "unguided queue: 90 attempts, 0 verified", ha="center", fontsize=9)
ax2.text((n_unguided + len(records)) / 2 - 2, 0.62, "near-miss repair", ha="center", fontsize=9)
ax2.annotate("model restacked on its own\n(after the behavioral diagnosis)",
             xy=(min(green_x), 1.857), xytext=(n_unguided - 22, 1.30), fontsize=8.5,
             arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.2))
ax2.annotate("final guided repair -> floor", xy=(max(green_x), 1.00),
             xytext=(len(records) - 34, 0.80), fontsize=8.5,
             arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.2))
ax2.set_ylim(0.55, 2.25)
ax2.set_xlabel("attempt number (pilot_v2, then 5 frozen repeats, then repair rounds)")
ax2.set_ylabel("largest shape: bytes moved / minimum possible")
ax2.set_title("B. The path to the floor, with every invalid attempt shown")
ax2.legend(loc="upper right", fontsize=9)

fig.suptitle("Minimum-traffic agent — all runs, simulator-verified; final repairs line-guided "
             "(disclosed)", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.96))
out = os.path.join(HERE, "progress.png")
fig.savefig(out, dpi=150)
print("wrote", out)
