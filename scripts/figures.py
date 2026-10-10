#!/usr/bin/env python3
"""The write-up's three figures, from attempts logs.

    python scripts/figures.py OUT_DIR VERSION=GLOB[,GLOB...] [VERSION=...] [--final VERSION] [--levels 1,2,3,4]

  solved_by_version.png         share of runs solved per level, one bar per version, labelled x/n
  attempts_to_solve.png         for --final (default: the last version): the round of the first 1.0 of every
                                solved run, per level, with the median
  failure_modes_by_version.png  the 8 most common failure modes (taxonomy.py's names) as a share of each
                                version's failed attempts
and figures.json with every number drawn. A run is one (file, run, level) episode as taxonomy.py splits
them. A run that is the last one in its file and neither solved nor reached round 7 may still have been
running when the log was pulled: it is left out and listed (unless its version is in --complete).
"""
import argparse
import collections
import glob
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import taxonomy  # noqa: E402

COLORS = ["#9e9e9e", "#7fa7d6", "#4c78a8", "#f58518", "#54a24b", "#e45756"]


def episodes_of(spec, complete=False):
    files = sorted({f for g in spec.split(",") for f in glob.glob(g)
                    if not any(s in os.path.basename(f) for s in ("usage", "verdicts"))})
    eps = taxonomy.load_attempts(files)
    last = {}
    for (fi, run, lv) in eps:
        last[fi] = max(last.get(fi, -1), run)
    kept, dropped = [], []
    for (fi, run, lv), rows in eps.items():
        solved = any(r["reward"] >= 1 - 1e-9 for r in rows)
        done = complete or solved or run < last[fi] or max(r["round"] for r in rows) >= 7
        (kept if done else dropped).append((files[fi], run, lv, rows))
    return files, kept, dropped


def first_solve(rows):
    for i, r in enumerate(rows):
        if r["reward"] >= 1 - 1e-9:
            return r["round"], i + 1
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("versions", nargs="+", help="VERSION=GLOB[,GLOB...]")
    ap.add_argument("--final", default=None)
    ap.add_argument("--levels", default="1,2,3,4")
    ap.add_argument("--complete", default="", help="comma-separated versions whose logs are known finished "
                    "(e.g. the baseline, whose --all runs end early on repeated failures): count every run")
    ap.add_argument("--note", default="", help="appended to the figures' subtitles, e.g. the log cut-off")
    a = ap.parse_args()
    levels = [int(x) for x in a.levels.split(",")]
    os.makedirs(a.out, exist_ok=True)
    data = collections.OrderedDict()
    for v in a.versions:
        name, _, spec = v.partition("=")
        files, kept, dropped = episodes_of(spec, name in a.complete.split(","))
        data[name] = dict(files=files, kept=kept, dropped=dropped)
        print(f"{name}: {len(files)} files, {len(kept)} runs" + (f"; left out as unfinished: "
              + ", ".join(f"{os.path.basename(f)} run {r} L{lv}" for f, r, lv, _ in dropped) if dropped else ""))
    final = a.final or list(data)[-1]
    out = dict(note=a.note, solved={}, first_solve={}, modes={})

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # (a) solved by version
    fig, ax = plt.subplots(figsize=(9, 4.8))
    w = 0.8 / len(data)
    for k, (name, d) in enumerate(data.items()):
        out["solved"][name] = {}
        for j, lv in enumerate(levels):
            runs = [rows for _, _, l, rows in d["kept"] if l == lv]
            n, s = len(runs), sum(first_solve(rows) is not None for rows in runs)
            out["solved"][name][lv] = [s, n]
            x = j + (k - (len(data) - 1) / 2) * w
            ax.bar(x, s / n if n else 0, w * 0.92, color=COLORS[k % len(COLORS)], label=name if j == 0 else None)
            ax.text(x, (s / n if n else 0) + 0.015, f"{s}/{n}" if n else "not run", ha="center", va="bottom",
                    fontsize=7.5, rotation=90 if len(data) > 4 else 0)
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels([f"level {lv}" for lv in levels])
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("share of runs solved")
    ax.set_title("Runs solved per level, by agent version" + (f"\n{a.note}" if a.note else ""), fontsize=11)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "solved_by_version.png"), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # (b) round of the first 1.0, final version
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    d = data[final]
    for j, lv in enumerate(levels):
        firsts = [first_solve(rows) for _, _, l, rows in d["kept"] if l == lv]
        rounds = sorted(f[0] for f in firsts if f)
        attempts = sorted(f[1] for f in firsts if f)
        out["first_solve"][lv] = dict(rounds=rounds, attempts=attempts, runs=len(firsts))
        if rounds:
            cnt = collections.Counter()
            for r in rounds:
                off = (cnt[r] - (rounds.count(r) - 1) / 2) * 0.09
                cnt[r] += 1
                ax.plot(j + off, r, "o", color=COLORS[2], ms=7, alpha=0.85)
            med = statistics.median(rounds)
            ax.hlines(med, j - 0.3, j + 0.3, color=COLORS[3], lw=2.2, label="median" if j == 0 else None)
            ax.text(j + 0.32, med, f"median {med:g}", va="center", fontsize=8, color=COLORS[3])
        ax.text(j, -0.9, f"{len(rounds)}/{len(firsts)} solved", ha="center", fontsize=8)
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels([f"level {lv}" for lv in levels])
    ax.set_ylim(-1.3, 8)
    ax.set_yticks(range(0, 8))
    ax.set_ylabel("round of the first 1.0 (0 = first prompt)")
    ax.set_title(f"When each solved run first scored 1.0 ({final})" + (f"\n{a.note}" if a.note else ""),
                 fontsize=11)
    ax.grid(axis="y", color="#eeeeee")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "attempts_to_solve.png"), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # (c) failure modes by version
    counts = {}
    for name, d in data.items():
        c = collections.Counter(taxonomy.mode_of(r) for _, _, l, rows in d["kept"] if l in levels
                                for r in rows if r["reward"] < 1 - 1e-9)
        counts[name] = c
    total = collections.Counter()
    for c in counts.values():
        total.update({m: n for m, n in c.items() if m != "other"})
    top = [m for m, _ in sorted(total.items(), key=lambda kv: (-kv[1], kv[0]))[:8]]
    import numpy as np
    share = np.array([[counts[v][m] / max(sum(counts[v].values()), 1) for v in data] for m in top])
    fig, ax = plt.subplots(figsize=(1.6 * len(data) + 4, 0.5 * len(top) + 2))
    im = ax.imshow(share, cmap="Blues", vmin=0, vmax=max(share.max(), 0.01), aspect="auto")
    for i, m in enumerate(top):
        for k, v in enumerate(data):
            n = counts[v][m]
            ax.text(k, i, f"{share[i, k]:.0%}\n({n})", ha="center", va="center", fontsize=7.5,
                    color="white" if share[i, k] > 0.6 * share.max() else "black")
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels([f"{v}\n{sum(counts[v].values())} failed" for v in data], fontsize=8)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top, fontsize=8)
    ax.set_title("Failure modes: share of each version's failed attempts (count)"
                 + (f"\n{a.note}" if a.note else ""), fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.03, format=matplotlib.ticker.PercentFormatter(1.0))
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "failure_modes_by_version.png"), dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    out["modes"] = {v: {m: counts[v][m] for m in top} | {"failed attempts": sum(counts[v].values())} for v in data}
    out["files"] = {v: [os.path.relpath(f) for f in d["files"]] for v, d in data.items()}
    out["left_out"] = {v: [f"{os.path.basename(f)} run {r} L{lv}" for f, r, lv, _ in d["dropped"]]
                       for v, d in data.items()}
    json.dump(out, open(os.path.join(a.out, "figures.json"), "w"), indent=1)
    for v in data:
        print(f"  {v}: " + "  ".join(f"L{lv} {s}/{n}" for lv, (s, n) in out["solved"][v].items()))
    print(f"  first 1.0 ({final}): " + "  ".join(f"L{lv} rounds {d['rounds']}" for lv, d in out["first_solve"].items()))


if __name__ == "__main__":
    main()
