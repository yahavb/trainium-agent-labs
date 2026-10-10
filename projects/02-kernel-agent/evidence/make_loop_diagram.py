#!/usr/bin/env python3
"""Loop diagram figure for the paper (loop_diagram.png)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))

fig, ax = plt.subplots(figsize=(7.4, 4.4))
ax.set_xlim(0, 10)
ax.set_ylim(0, 7)
ax.axis("off")


def box(x, y, w, h, text, fc="#f2f6fa", ec="#34506b", fs=9.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.12,rounding_size=0.18",
                                fc=fc, ec=ec, lw=1.3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color="#1c2b3a")


def arrow(p, q, rad=0.0, color="#34506b"):
    ax.add_patch(FancyArrowPatch(p, q, connectionstyle=f"arc3,rad={rad}",
                                 arrowstyle="-|>", mutation_scale=13, lw=1.25,
                                 color=color))


box(3.5, 5.75, 3.2, 0.95, "Evaluate kernel\n(checker + diagnosis)")
box(3.9, 4.05, 2.4, 0.85, "Accepted at\nthe level gate?", fc="#fdf3e7", ec="#b76b1f")
box(3.5, 2.20, 3.2, 0.95, "Classifier: choose ONE\nguidance id (no code)",
    fc="#eef7ee", ec="#2c7a2c")
box(3.5, 0.40, 3.2, 0.95, "Applier: apply only that\nguidance's exact text",
    fc="#eef7ee", ec="#2c7a2c")
box(7.3, 4.05, 2.4, 0.85, "Winner: kernel\naccepted", fc="#e9f6e9", ec="#1b5e20")

arrow((5.1, 5.75), (5.1, 4.9))
ax.text(5.22, 5.28, "state", fontsize=8, color="#34506b")
arrow((6.3, 4.475), (7.3, 4.475))
ax.text(6.55, 4.62, "yes", fontsize=8.5, color="#1b5e20")
arrow((5.1, 4.05), (5.1, 3.15))
ax.text(5.22, 3.55, "no", fontsize=8.5, color="#b76b1f")
arrow((5.1, 2.20), (5.1, 1.35))
ax.plot([1.15, 3.5], [0.875, 0.875], color="#34506b", lw=1.25)
ax.plot([1.15, 1.15], [0.875, 6.225], color="#34506b", lw=1.25)
arrow((1.15, 6.225), (3.5, 6.225))
ax.text(1.34, 3.5, "next round (kernel adopted)", rotation=90, fontsize=8,
        color="#34506b", va="center")

fig.tight_layout()
fig.savefig(os.path.join(HERE, "loop_diagram.png"), dpi=200)
print("wrote loop_diagram.png")
