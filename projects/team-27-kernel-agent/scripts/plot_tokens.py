"""Input tokens per attempt, split by what they were spent on (task / code / feedback / ledger,
plus chat-template overhead). Writes <log>.tokens.png and prints the same numbers as a table.

    uv run python scripts/plot_tokens.py runs/20261010-160000.jsonl [RUN]   # RUN is 1-based
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# reference categorical palette, slots 1-4 in fixed order (validated: CVD dE >= 9.1); overhead
# is not a category of spend, so it is a recessive neutral
SERIES = [("task", "#2a78d6"), ("code", "#eb6834"), ("feedback", "#1baf7a"), ("ledger", "#eda100")]
OVERHEAD = ("template", "#c9c7c0")
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def load(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    att = [r for r in rows if r.get("type") == "attempt"]
    for r in att:
        r["sections"]["template"] = max(0, r["prompt_tokens"] - sum(r["sections"].values()))
    return att


def main(path, run=None):
    att = load(path)
    if run is not None:
        att = [r for r in att if r["run"] == run]
    if not att:
        sys.exit("no attempts in log")
    exact = all(r["exact_tokens"] for r in att)
    labels = [f"L{r['level']} r{r['run'] + 1} #{r['attempt']}" for r in att]
    names = [n for n, _ in SERIES] + [OVERHEAD[0]]

    print(f"{'attempt':14s}" + "".join(f"{n:>10s}" for n in names) + f"{'total':>8s}  outcome")
    for lab, r in zip(labels, att):
        s = r["sections"]
        print(f"{lab:14s}" + "".join(f"{s.get(n, 0):>10d}" for n in names)
              + f"{r['prompt_tokens']:>8d}  {r['outcome']} {r.get('key') or ''}")

    fig, ax = plt.subplots(figsize=(max(6, 0.42 * len(att) + 2), 4.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    bottom = [0] * len(att)
    for name, color in SERIES + [OVERHEAD]:
        vals = [r["sections"].get(name, 0) for r in att]
        ax.bar(range(len(att)), vals, bottom=bottom, width=0.62, color=color, label=name,
               edgecolor=SURFACE, linewidth=1.5)
        bottom = [b + v for b, v in zip(bottom, vals)]
    for i, (tot, r) in enumerate(zip(bottom, att)):
        mark = {"pass": "pass", "fail": "", "truncated": "trunc", "empty": "empty",
                "no-code": "no code"}.get(r["outcome"], "")
        if mark:
            ax.text(i, tot, mark, ha="center", va="bottom", fontsize=7, color=MUTED)

    ax.set_xticks(range(len(att)), labels, rotation=60, ha="right", fontsize=7, color=MUTED)
    ax.tick_params(axis="y", labelsize=8, colors=MUTED)
    ax.set_ylabel("input tokens" + ("" if exact else " (estimated)"), color=MUTED, fontsize=9)
    ax.set_title("Where each attempt's input tokens went", loc="left", fontsize=11, color=INK)
    ax.grid(axis="y", color="#e6e4df", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#d0cec8")
    ax.legend(ncol=5, fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.32))
    fig.tight_layout()
    out = Path(path).with_suffix(f".tokens{'' if run is None else f'-run{run + 1}'}.png")
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) - 1 if len(sys.argv) > 2 else None)   # optional 1-based run
