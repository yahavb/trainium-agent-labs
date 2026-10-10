"""Numerical-analysis figure: how much of the proven float32 error bound each hand kernel uses,
and why a constant tolerance is the wrong test (notes E2, E7). Writes notes/fig-bounds.png."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from kagent.harness import verify  # noqa: E402
from kagent.levels import LEVELS  # noqa: E402

F64, F32 = "#2a78d6", "#eb6834"          # reference palette slots 1-2
INK, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e6e4df"
KERNELS = [(1, "l1_relu_affine", "L1 relu"), (2, "l2_row_sum", "L2 sum, f64 acc"),
           (2, "l2_row_sum_f32acc", "L2 sum, f32 acc"), (4, "l4_rmsnorm", "L4 RMSNorm"),
           (5, "l5_softmax", "L5 softmax"), (7, "l7_matmul", "L7 matmul"),
           (8, "l8_layernorm", "L8 layernorm, f64"), (8, "l8_layernorm_f32_two_pass", "L8 layernorm, f32 two-pass"),
           (9, "l9_band_attention", "L9 band attention"), (10, "l10_conv1d", "L10 conv1d")]
rows = []
for n, f, label in KERNELS:
    r = verify(LEVELS[n], (ROOT / f"kernels/hand/{f}.py").read_text())
    rows.append((label, r.bound_use(), "f32" in label))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE,
                               gridspec_kw={"width_ratios": [1.6, 1]})
for ax in (ax1, ax2):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)
y = range(len(rows))[::-1]
ax1.barh(list(y), [r[1] for r in rows], color=[F32 if r[2] else F64 for r in rows], height=0.6,
         edgecolor=SURFACE, linewidth=1.5)
for yi, (label, use, _) in zip(y, rows):
    ax1.text(use + 0.01, yi, f"{use:.0%}", va="center", fontsize=8, color=INK)
ax1.set_yticks(list(y), [r[0] for r in rows], fontsize=8, color=MUTED)
ax1.axvline(1.0, color=MUTED, linestyle=(0, (4, 3)), linewidth=1)
ax1.text(0.99, len(rows) - 0.4, "proven float32 bound  γ_k · scale", ha="right", fontsize=8, color=MUTED)
ax1.set_xlim(0, 1.08)
ax1.set_xlabel("max |error| / bound   (all pass; 1.0 = at the bound)", fontsize=9, color=MUTED)
ax1.set_title("Correct kernels use 0–47% of their proven error bound", loc="left", fontsize=10, color=INK)
ax1.grid(axis="x", color=GRID, linewidth=0.8)
ax1.set_axisbelow(True)

# E2 -> E7: float32 row sum, N = 1031, constant rows
vals = [("error of a correct\nfloat32 row sum", 1.47e-5, F32), ("old constant\ntolerance", 1e-5, MUTED),
        ("proven bound\nγ_1031", 6.1e-5, F64)]
ax2.bar(range(3), [v[1] for v in vals], color=[v[2] for v in vals], width=0.6, edgecolor=SURFACE)
for i, (_, v, _) in enumerate(vals):
    ax2.text(i, v, f"{v:.2g}", ha="center", va="bottom", fontsize=8, color=INK)
ax2.set_xticks(range(3), [v[0] for v in vals], fontsize=8, color=MUTED)
ax2.set_ylabel("error relative to Σ|x|", fontsize=9, color=MUTED)
ax2.set_title("A constant tolerance rejected a correct algorithm", loc="left", fontsize=10, color=INK)
ax2.grid(axis="y", color=GRID, linewidth=0.8)
ax2.set_axisbelow(True)
fig.tight_layout()
fig.savefig(ROOT / "notes/fig-bounds.png", dpi=160, facecolor=SURFACE)
print("wrote notes/fig-bounds.png")
