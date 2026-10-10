"""Render the README workflow. Run with a Python environment containing Matplotlib."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon
from matplotlib.path import Path as PlotPath


OUTPUT = Path(__file__).resolve().parent
INK = "#172b45"
MUTED = "#53677f"
LINE = "#8091a6"
BLUE = "#2563eb"
PURPLE = "#7c3aed"
TEAL = "#0f766e"


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(14, 16.5), dpi=160)
    fig.patch.set_facecolor("#f7f9fc")
    ax.set_facecolor("#f7f9fc")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 16.5)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    def text(x, y, value, size=16, color=INK, weight="normal", **kwargs):
        return ax.text(x, y, value, fontsize=size, color=color, weight=weight,
                       ha="center", va="center", linespacing=1.45, **kwargs)

    def card(x, y, title, body, color=BLUE, fill="#eff6ff", width=5, height=1.15):
        ax.add_patch(FancyBboxPatch(
            (x-width/2, y-height/2-0.045), width, height,
            boxstyle="round,pad=0.04,rounding_size=0.16", linewidth=0,
            facecolor="#e4eaf2", zorder=1))
        ax.add_patch(FancyBboxPatch(
            (x-width/2, y-height/2), width, height,
            boxstyle="round,pad=0.04,rounding_size=0.16", linewidth=1.2,
            edgecolor=color, facecolor=fill, zorder=2))
        text(x, y+0.23, title, 20, color, "bold", zorder=3)
        text(x, y-0.23, body, 15, MUTED, zorder=3)

    def diamond(x, y, title):
        ax.add_patch(Polygon([(x, y+0.62), (x+1.65, y), (x, y-0.62),
                              (x-1.65, y)], facecolor="white",
                             edgecolor=LINE, linewidth=1.4, zorder=2))
        text(x, y, title, 18, weight="bold", zorder=3)

    def arrow(points, color=LINE):
        path = PlotPath(points, [PlotPath.MOVETO] + [PlotPath.LINETO]*(len(points)-1))
        ax.add_patch(FancyArrowPatch(path=path, arrowstyle="-|>", mutation_scale=18,
                                    linewidth=1.6, color=color, zorder=1))

    def label(x, y, value, color=MUTED, size=13):
        text(x, y, value, size, color,
             bbox={"facecolor": "#f7f9fc", "edgecolor": "none", "pad": 3})

    text(7, 15.85, "HEAT-ROD CONTROLLER RL", 29, weight="bold")
    text(7, 15.31, "Propose  →  Verify  →  Route  →  Repair", 19, MUTED)

    x, right = 4.8, 10.9
    card(x, 14.25, "1  Read the problem", "Equation · boundaries · initial profile")
    card(x, 12.65, "2  Qwen proposes a solution", "Explicit u(x,t) · model weights frozen")
    card(x, 10.95, "3  Physics checker", "PDE + two boundaries + initial profile",
         TEAL, "#ecfdf5")
    diamond(x, 9.35, "All checks pass?")
    diamond(x, 7.55, "Budget remains?")
    card(right, 9.35, "Solved", "Keep the successful candidate",
         TEAL, "#ecfdf5", width=4.3)
    card(right, 7.55, "Stop", "Keep the best candidate + score",
         MUTED, "#f1f5f9", width=4.3)
    card(x, 5.75, "4  Router selects an action", "Use the current best candidate's feedback",
         PURPLE, "#f5f3ff")
    card(right, 5.75, "USE_CALCULATOR", "Qwen writes 1–8 COMPUTE requests\nSymPy returns the requested results",
         TEAL, "#ecfdf5", width=4.3, height=1.5)
    card(x, 3.8, "5  Qwen repairs the answer", "Selected prompt + feedback + any tool results")
    card(x, 1.85, "6  Check, learn, retain", "Grade the repair · keep the best answer\nTraining only: reward → router table update",
         PURPLE, "#f5f3ff", height=1.5)

    arrow([(x, 13.63), (x, 13.28)])
    arrow([(x, 12.03), (x, 11.58)])
    arrow([(x, 10.33), (x, 10.01)])
    arrow([(x+1.7, 9.35), (right-2.23, 9.35)], TEAL)
    label(7.64, 9.60, "Yes", TEAL)
    arrow([(x, 8.68), (x, 8.22)])
    label(x+0.45, 8.46, "No")
    arrow([(x+1.7, 7.55), (right-2.23, 7.55)])
    label(7.64, 7.80, "No")
    arrow([(x, 6.88), (x, 6.38)], PURPLE)
    label(x+0.45, 6.65, "Yes", PURPLE)
    arrow([(x+2.57, 5.75), (right-2.23, 5.75)], TEAL)
    label(8.05, 6.06, "Tool action", TEAL, 12)
    arrow([(x, 5.13), (x, 4.43)], BLUE)
    label(x+1.08, 4.81, "Other actions", BLUE)
    arrow([(right, 4.94), (right, 3.8), (x+2.57, 3.8)], TEAL)
    label(9.43, 4.07, "Calculator results", TEAL, 12)
    arrow([(x, 3.18), (x, 2.67)], PURPLE)
    arrow([(x-2.57, 1.85), (0.95, 1.85), (0.95, 9.35), (x-1.7, 9.35)])
    text(0.57, 5.75, "Repeat with the best candidate", 14, MUTED, rotation=90)

    text(9.80, 2.02, "Only the router learns", 19, PURPLE, "bold")
    text(9.80, 1.39, "Explore actions → observe repair reward\n→ update the mean value for state + action", 14, MUTED)
    text(7, 0.43, "One initial attempt  ·  Up to three repairs  ·  At most seven model calls", 15, MUTED)
    for extension in ("png", "svg"):
        fig.savefig(OUTPUT / f"agent-workflow.{extension}", facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
