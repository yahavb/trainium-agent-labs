# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Draw the input-cut figures from measurements.json (run from this folder).

    python make_figures.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = json.loads((HERE / "measurements.json").read_text())

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#ffffff"
TINT_BLUE, TINT_ORANGE, TINT_GREY = "#e3eefb", "#fde9e0", "#f3f2ef"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "text.color": INK,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.dpi": 200,
})


def save(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(HERE / f"{name}.{ext}", bbox_inches="tight", facecolor=SURF)
    plt.close(fig)


def hbars(ax, labels, values, colors, fmt, xmax):
    ys = list(range(len(labels)))[::-1]
    ax.barh(ys, values, color=colors, height=0.6, edgecolor=SURF, linewidth=2)
    ax.set_yticks(ys, labels)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, xmax)
    for y, v in zip(ys, values):
        ax.text(v + xmax * 0.01, y, fmt(v), va="center", fontsize=8.5, color=INK)


# Figure 1: full forward pass.
runs = DATA["full_pass"]
order = [("plain", "Plain (one graph)"),
         ("plain_input_cut", "Plain + input cut"),
         ("segments", "--segments 1"),
         ("segments_input_cut", "--segments 1 + input cut")]
base = runs["plain"]["median_ms"]
vals = [runs[k]["median_ms"] for k, _ in order]
fig, ax = plt.subplots(figsize=(6.6, 2.3))
hbars(ax, [l for _, l in order], vals,
      [BLUE, BLUE, BLUE, ORANGE],
      lambda v: f"{v:.1f} ms  ({base / v:.2f}x)", xmax=290)
ax.set_xlabel("ms per 1 degree forward pass (fp32, Trainium2, median of 20; all pass at 1.2e-6)")
save(fig, "fig1_full_pass")

# Figure 2: block 0 variants.
p = DATA["block0_probe"]
labels = ["Concat inside the program\n(as in --segments 1)", "Input already concatenated",
          "+ zero-padded to 256 ch", "+ zero-padded to 192 ch"]
vals = [p["concat_inside_program_ms"], p["input_already_concatenated_ms"],
        p["input_zero_padded_256_ms"], p["input_zero_padded_192_ms"]]
fig, ax = plt.subplots(figsize=(6.6, 2.3))
hbars(ax, labels, vals, [BLUE, ORANGE, MUTED, MUTED], lambda v: f"{v:.1f} ms", xmax=56)
ax.set_xlabel("ms for UNet layer 0 alone (162 -> 280 channels, 180 x 360); rel. error 3e-7 for all")
save(fig, "fig2_block0_variants")

# Figure 3: layer 0 before / after, one small chart per measure (no shared axis).
before, after = DATA["layer0_instructions"]["before"], DATA["layer0_instructions"]["after"]
b_sum = json.loads((HERE / "layer0_before_summary.json").read_text())
a_sum = json.loads((HERE / "layer0_after_summary.json").read_text())
b_s, a_s = next(iter(b_sum.values())), next(iter(a_sum.values()))
panels = [
    ("Chip time (ms)", before["device_ms"], after["device_ms"], "{:.1f}"),
    ("Matmuls (thousands)", before["matmul_count"] / 1e3, after["matmul_count"] / 1e3, "{:.0f}k"),
    ("Weight loads (ms)", before["ldweights_ms_total"], after["ldweights_ms_total"], "{:.1f}"),
    ("Compute util. (%)", 100 * b_s["mfu_estimated_percent"], 100 * a_s["mfu_estimated_percent"], "{:.0f}%"),
    ("Spill (GB)", (b_s["spill_save_bytes"] + b_s["spill_reload_bytes"]) / 1e9,
     (a_s["spill_save_bytes"] + a_s["spill_reload_bytes"]) / 1e9, "{:.2f}"),
]
fig, axes = plt.subplots(1, len(panels), figsize=(10.4, 2.3))
for ax, (title, b, a, fmt) in zip(axes, panels):
    ax.bar([0, 1], [b, a], color=[BLUE, ORANGE], width=0.6, edgecolor=SURF, linewidth=2)
    ax.set_xticks([0, 1], ["before", "after"])
    ax.set_title(title, fontsize=8.5, color=INK2)
    ax.set_ylim(0, max(b, a) * 1.25)
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", length=0)
    for x, v in zip([0, 1], [b, a]):
        ax.text(x, v + max(b, a) * 0.03, fmt.format(v), ha="center", fontsize=8.5)
fig.suptitle("UNet layer 0 on the NeuronCore: before = concat in the same graph, after = input cut",
             fontsize=9.5, y=1.06)
fig.tight_layout()
save(fig, "fig3_layer0_profile")

# Figure 4: where the time goes in the segmented pass, before and after.
layers = DATA["segmented_layer_ms_before"]
names = list(layers)
before_ms = [layers[n] for n in names]
after_ms = [after["device_ms"] if n.startswith("00") else layers[n] for n in names]
colors = [ORANGE, BLUE, AQUA, AQUA, MUTED, "#c3c2b7", GRID]
fig, ax = plt.subplots(figsize=(7.2, 2.0))
for row, ms in ((1, before_ms), (0, after_ms)):
    left = 0.0
    for name, v, c in zip(names, ms, colors):
        ax.barh(row, v, left=left, color=c, height=0.55, edgecolor=SURF, linewidth=2,
                label=name if row == 1 else None)
        if v > 9:
            ax.text(left + v / 2, row, f"{v:.1f}", ha="center", va="center", fontsize=8, color=SURF)
        left += v
    ax.text(left + 2, row, f"{left:.1f} ms", va="center", fontsize=8.5)
ax.set_yticks([1, 0], ["--segments 1\n(profiled)", "+ input cut\n(layer 0 from probe)"])
ax.tick_params(axis="y", length=0)
ax.grid(axis="y", visible=False)
ax.set_xlim(0, 160)
ax.set_xlabel("device ms per pass, by program (sum of per-program Neuron profiles)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.45), ncol=3, frameon=False, fontsize=7.5)
save(fig, "fig4_segmented_breakdown")


# Figure 5: diagram of where the graph boundary goes.
def box(ax, x, y, w, h, text, fill, edge=MUTED, bold=False, fs=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=fill, ec=edge, lw=1.0))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color=INK)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9,
                                 color=INK2, lw=1.0))


fig, ax = plt.subplots(figsize=(8.0, 3.4))
ax.set_xlim(0, 10)
ax.set_ylim(0, 4.6)
ax.axis("off")
ax.text(0.05, 4.35, "Before: concat and the first block share one compiled program (47.4 ms)",
        fontsize=9, fontweight="bold")
ax.add_patch(FancyBboxPatch((1.95, 2.55), 5.3, 1.45, boxstyle="round,pad=0.02,rounding_size=0.08",
                            fc=TINT_BLUE, ec=BLUE, lw=1.2))
ax.text(2.05, 3.82, "program 1", fontsize=7.5, color=BLUE)
box(ax, 0.05, 3.25, 1.6, 0.55, "ocean state\n154 ch", SURF)
box(ax, 0.05, 2.6, 1.6, 0.55, "forcing\n8 ch", SURF)
box(ax, 2.15, 2.75, 1.6, 0.9, "torch.cat\n162 ch", SURF)
box(ax, 4.1, 2.75, 3.0, 0.9, "block 0: ragged 77/81-row\nweight blocks, 77-wide matmuls,\nweights reloaded per matmul", SURF, fs=7.5)
arrow(ax, 1.65, 3.52, 2.15, 3.3)
arrow(ax, 1.65, 2.87, 2.15, 3.05)
arrow(ax, 3.75, 3.2, 4.1, 3.2)
box(ax, 7.6, 2.75, 2.3, 0.9, "next programs\n(layers 1-16, output)", TINT_GREY)
arrow(ax, 7.25, 3.2, 7.6, 3.2)

ax.text(0.05, 1.95, "After: a graph cut right after the concat (15.0 ms for block 0)",
        fontsize=9, fontweight="bold")
ax.add_patch(FancyBboxPatch((1.95, 0.15), 1.95, 1.45, boxstyle="round,pad=0.02,rounding_size=0.08",
                            fc=TINT_GREY, ec=MUTED, lw=1.0))
ax.text(2.05, 1.42, "program 1", fontsize=7.5, color=INK2)
ax.add_patch(FancyBboxPatch((4.25, 0.15), 3.0, 1.45, boxstyle="round,pad=0.02,rounding_size=0.08",
                            fc=TINT_ORANGE, ec=ORANGE, lw=1.2))
ax.text(4.35, 1.42, "program 2", fontsize=7.5, color=ORANGE)
box(ax, 0.05, 0.85, 1.6, 0.55, "ocean state\n154 ch", SURF)
box(ax, 0.05, 0.2, 1.6, 0.55, "forcing\n8 ch", SURF)
box(ax, 2.15, 0.35, 1.6, 0.9, "torch.cat\n162 ch", SURF)
ax.plot([4.07, 4.07], [0.1, 1.65], color=ORANGE, lw=2, ls=(0, (3, 2)))
ax.text(4.07, 1.72, "torch_xla.sync()", ha="center", fontsize=7.5, color=ORANGE)
box(ax, 4.4, 0.35, 2.7, 0.9, "block 0: one 162-channel input,\n128 + 34 row weight blocks,\n128-wide matmuls", SURF, fs=7.5)
arrow(ax, 1.65, 1.12, 2.15, 0.9)
arrow(ax, 1.65, 0.47, 2.15, 0.65)
arrow(ax, 3.75, 0.8, 4.4, 0.8)
box(ax, 7.6, 0.35, 2.3, 0.9, "next programs\n(layers 1-16, output)", TINT_GREY)
arrow(ax, 7.1, 0.8, 7.6, 0.8)
save(fig, "fig5_graph_cut_diagram")
print("wrote figures to", HERE)
