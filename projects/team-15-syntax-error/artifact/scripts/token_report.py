#!/usr/bin/env python3
"""
token_report.py -- where the 8192-token budget went, per attempt.

    python scripts/token_report.py runs/*/*.jsonl > TOKENS.md      # tables
    python scripts/token_report.py --png token_budget.png --tag skel_L4 runs/*/*.jsonl

Every attempt in the log carries `prompt_sections` (chars/4 per section: task, api, skeleton, docs,
code, error, ledger) and the server's own `usage.prompt_tokens`. The report
  1. calibrates the chars/4 estimate against the server's count (ratio printed, not assumed);
  2. tabulates mean input tokens per section by tag and level, and the share of the window;
  3. optionally draws one run as a stacked bar per attempt -- the "tokens spent on docs vs error
     vs code, per attempt" graph the challenge asks for.
Identical replies within a round share one prompt, so prompts are counted once per round.
"""

import argparse
import collections
import glob
import json
import sys

SECTIONS = ["task", "api", "skeleton", "docs", "code", "error", "ledger"]
# dataviz reference palette, categorical slots 1-7 in fixed order (validated: light mode passes;
# contrast WARN on 3 slots is relieved by the legend + the tables this script also prints)
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
SURFACE, INK, MUTED = "#fcfcfb", "#1f1f1e", "#6b6a64"
WINDOW = 8192


def load(paths):
    rows = []
    for p in paths:
        for line in open(p, encoding="utf-8"):
            if line.strip():
                rows.append(json.loads(line))
    return rows


def prompts(rows):
    """One record per distinct prompt (tag, run, level, round)."""
    out = {}
    for r in rows:
        if "prompt_sections" not in r:
            continue
        k = (r.get("tag"), r.get("run"), r["level"], r["round"])
        out.setdefault(k, r)
    return [out[k] for k in sorted(out, key=lambda k: (str(k[0]), k[1] or 0, k[2], k[3]))]


def table(ps):
    est = [sum(p["prompt_sections"].values()) for p in ps]
    srv = [(p.get("usage") or {}).get("prompt_tokens") for p in ps]
    pairs = [(e, s) for e, s in zip(est, srv) if s]
    print("# TOKENS — input-token instrumentation\n")
    if pairs:
        ratio = sum(s for _, s in pairs) / max(1, sum(e for e, _ in pairs))
        print(f"Calibration: the server counted **{ratio:.2f}x** the chars/4 estimate over "
              f"{len(pairs)} prompts (sections below are scaled by this factor). The chat template "
              f"adds a fixed overhead the sections cannot see.\n")
    else:
        ratio = 1.0
    groups = collections.defaultdict(list)
    for p in ps:
        groups[((p.get("tag") or "?").split("_L")[0], p["level"])].append(p)
    print("| experiment | level | prompts | " + " | ".join(SECTIONS) + " | total | % of 8192 | max |")
    print("|---|---|---|" + "---|" * (len(SECTIONS) + 3))
    for (tag, lv), g in sorted(groups.items()):
        means = [sum(p["prompt_sections"].get(s, 0) for p in g) / len(g) * ratio for s in SECTIONS]
        tot = [sum(p["prompt_sections"].values()) * ratio for p in g]
        print(f"| {tag} | {lv} | {len(g)} | " + " | ".join(f"{m:.0f}" for m in means)
              + f" | {sum(tot) / len(tot):.0f} | {sum(tot) / len(tot) / WINDOW:.0%} | {max(tot):.0f} |")
    kinds = collections.Counter((p.get("prompt_kind") or {}).get("kind", "?") for p in ps)
    print(f"\nPrompt kinds: {dict(kinds)}.")
    comp = [(p.get("usage") or {}).get("completion_tokens") for p in ps]
    comp = [c for c in comp if c]
    if comp:
        print(f"Completion tokens: mean {sum(comp) / len(comp):.0f}, max {max(comp)} "
              f"(n={len(comp)}). Truncated answers (finish=length): "
              f"{sum(1 for p in ps if p.get('finish') == 'length')}.")
    return ratio


def png(ps, path, tag, run, ratio):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sel = [p for p in ps if (tag is None or p.get("tag") == tag) and p.get("run", 0) == run]
    if not sel:
        raise SystemExit(f"no prompts for tag={tag} run={run}")
    fig, ax = plt.subplots(figsize=(max(6, 0.55 * len(sel) + 2.5), 4.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    x = list(range(len(sel)))
    bottom = [0.0] * len(sel)
    for s, c in zip(SECTIONS, COLORS):
        h = [p["prompt_sections"].get(s, 0) * ratio for p in sel]
        if not any(h):
            continue
        # 2px surface gap between stacked segments (edgecolor = surface)
        ax.bar(x, h, bottom=bottom, width=0.6, color=c, edgecolor=SURFACE, linewidth=1.5, label=s)
        bottom = [b + v for b, v in zip(bottom, h)]
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{p['level']}\nr{p['round']}" for p in sel], fontsize=8, color=MUTED)
    ax.set_ylabel("input tokens (server-calibrated)", color=MUTED, fontsize=9)
    ax.set_title(f"Where the input budget went, per attempt (window 8192 tokens) — {tag or 'all'} run {run} [sim]",
                 color=INK, fontsize=10, loc="left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#d6d5ce")
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.yaxis.grid(True, color="#ecebe5", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, ncol=len(SECTIONS), loc="upper left",
              bbox_to_anchor=(0, -0.18), labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    print(f"\nwrote {path} ({len(sel)} attempts)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--png")
    ap.add_argument("--tag")
    ap.add_argument("--run", type=int, default=0)
    a = ap.parse_args()
    paths = [p for g in a.logs for p in glob.glob(g)]
    ps = prompts(load(paths))
    r = table(ps)
    if a.png:
        png(ps, a.png, a.tag, a.run, r)
