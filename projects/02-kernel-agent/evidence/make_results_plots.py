#!/usr/bin/env python3
"""Results figures for the README (levels_results.png, robust_matrix.png).

Data: the level-kernel evaluations (hardcoded below from the committed evidence) and
robust_l7_winner.json (read directly). Reproduce with:  python3 make_results_plots.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SHAPES = ["K128\nM128\nN512", "K256\nM256\nN1024", "K512\nM128\nN512", "K256\nM512\nN1024"]
SEED = [1.00, 1.5556, 1.00, 2.00]
AGENT = [1.00, 1.00, 1.00, 1.00]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.6))
xs = range(4)
ax1.bar([x - 0.19 for x in xs], SEED, width=0.38, color="#9ecae1",
        label="shipped seed (worst 2.00x)")
ax1.bar([x + 0.19 for x in xs], AGENT, width=0.38, color="#2ca02c",
        label="agent kernel (accepted, levels 5-7)")
for x, a, b in zip(xs, SEED, AGENT):
    ax1.text(x - 0.19, a + 0.03, f"{a:.2f}x", ha="center", fontsize=8.5)
    ax1.text(x + 0.19, b + 0.03, f"{b:.2f}x", ha="center", fontsize=8.5, color="#1b5e20")
for y, lab, col in ((1.00, "byte floor 1.00x", "black"),
                    (1.05, "level-7 gate 1.05x", "#7f7f7f"),
                    (1.25, "level-6 gate 1.25x", "#7f7f7f"),
                    (1.60, "level-5 gate 1.60x", "#e67e22"),
                    (2.00, "seed 2.00x", "#1f77b4")):
    ax1.axhline(y, color=col, lw=1.0, ls="--" if y in (1.00, 2.00) else ":", alpha=0.85)
    yt = y - 0.14 if y == 1.00 else y + 0.02
    ax1.text(3.47, yt, lab, ha="right", fontsize=8, color=col)
ax1.set_xticks(list(xs))
ax1.set_xticklabels(SHAPES, fontsize=8.5)
ax1.set_ylabel("bytes moved / minimum possible  (lower is better)")
ax1.set_ylim(0, 2.3)
ax1.set_title("A. Per-shape waste: seed vs agent kernel")
ax1.legend(loc="upper left", fontsize=8.5)

runs = ["level 5 run\n(near-miss start)", "level 6 run\n(retry from 1.857x)",
        "level 7 run\n(near-miss start)"]
decisions = [3, 1, 2]
matches = [2, 1, 2]
absorbed = [1, 0, 0]
xs2 = range(3)
ax2.bar([x - 0.26 for x in xs2], decisions, width=0.25, color="#6baed6",
        label="classifier decisions")
ax2.bar(list(xs2), matches, width=0.25, color="#2ca02c", label="matched expected guidance")
ax2.bar([x + 0.26 for x in xs2], absorbed, width=0.25, color="#fdae6b",
        label="absorbed misfires (back-out)")
for x, d, m, a in zip(xs2, decisions, matches, absorbed):
    ax2.text(x - 0.26, d + 0.05, str(d), ha="center", fontsize=8.5)
    ax2.text(x, m + 0.05, str(m), ha="center", fontsize=8.5)
    ax2.text(x + 0.26, a + 0.05, str(a), ha="center", fontsize=8.5)
ax2.set_xticks(list(xs2))
ax2.set_xticklabels(runs, fontsize=8.5)
ax2.set_ylim(0, 3.9)
ax2.set_title("B. Classifier behaviour in the three solved runs")
ax2.legend(loc="upper right", fontsize=8.5)
fig.suptitle("Levels 5-7: agent kernels at the byte floor, and how the classifier got there "
             "(simulator-verified)", fontsize=10.5)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(os.path.join(HERE, "levels_results.png"), dpi=150)
print("wrote levels_results.png")

d = json.load(open(os.path.join(HERE, "robust_l7_winner.json")))
fams = [("optimization shapes: level-5 bar", d["opt_l5"]),
        ("optimization shapes: level-6 bar", d["opt_l6"]),
        ("optimization shapes: level-7 bar", d["opt_l7"]),
        ("held-out aligned shapes", d["heldout"]),
        ("hostile value families", d["hostile"]),
        ("ragged correctness shapes", d["ragged"])]
total = sum(f[1][1] for f in fams)
passed = sum(f[1][0] for f in fams)

fig, ax = plt.subplots(figsize=(9.8, 4.8))
ys = list(range(len(fams)))[::-1]
for y, (name, (p, t)) in zip(ys, fams):
    ax.barh(y, p, color="#2ca02c", height=0.55)
    if t - p:
        ax.barh(y, t - p, left=p, color="#d62728", height=0.55)
    ax.text(t + 0.3, y, f"{p}/{t}", va="center", fontsize=9.5)
ax.set_yticks(ys)
ax.set_yticklabels([f[0] for f in fams], fontsize=9)
ax.set_xlim(0, max(f[1][1] for f in fams) + 3.6)
ax.set_xlabel("cases passed (green) / failed (red) -- official checker internals")
ax.set_title(f"Robust check of the level kernels -- {passed}/{total} cases; every failure "
             f"is in the one declared ragged family")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "robust_matrix.png"), dpi=150)
print("wrote robust_matrix.png")
