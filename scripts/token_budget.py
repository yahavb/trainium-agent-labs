#!/usr/bin/env python3
"""Where each attempt's tokens went: the chart and the table the challenge asks for.

    python scripts/token_budget.py runs/seat-116/latest/projects/02-kernel-agent/attempts.jsonl \
        -o analysis/token_budget

Writes <out>.png (one bar per round: the prompt split by segment, the answer on top, the context
limit as a line) and <out>.csv (mean tokens per segment per level). Reads the prompt_split field
agent.py logs per attempt; older lines without it are skipped and counted.
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import usage  # noqa: E402

SEGMENTS = ("instructions", "reference", "api_card", "prev_code", "feedback", "ledger",
            "chat_template")
COLORS = {"instructions": "#8c8c8c", "reference": "#4e79a7", "api_card": "#76b7b2",
          "prev_code": "#f28e2b", "feedback": "#e15759", "ledger": "#b07aa1",
          "chat_template": "#d4d4d4", "answer": "#59a14f"}


def load(paths, queues=None):
    rounds, skipped, unmatched = {}, 0, 0
    for path in paths:
        for line in open(path):
            r = json.loads(line)
            if queues is not None and "prompt_split" in r:
                unmatched += usage.apply([r], queues)[1]
            if "prompt_split" not in r:
                skipped += 1
                continue
            key = (path, r.get("run", 0), r["level"], r["round"])
            g = rounds.setdefault(key, dict(level=r["level"], run=r.get("run", 0),
                                            round=r["round"], split=r["prompt_split"],
                                            prompt=r["prompt_tokens"], answers=[], rewards=[],
                                            method=r.get("count_method", "?")))
            g["answers"].append(r["completion_tokens"])
            g["rewards"].append(r["reward"])
            g["unmatched"] = g.get("unmatched", 0) + (r.get("usage_matched") is False)
    if queues is not None:
        return list(rounds.values()), skipped, unmatched
    return list(rounds.values()), skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("-o", "--out", default="analysis/token_budget")
    ap.add_argument("--context", type=int, default=8192)
    ap.add_argument("--usage", nargs="*", default=None,
                    help="USAGE_LOG files (feedback_v5+): exact server token counts")
    a = ap.parse_args()

    if a.usage is not None:
        rounds, skipped, unmatched = load(a.logs, usage.load(a.usage))
    else:
        rounds, skipped = load(a.logs)
    if not rounds:
        raise SystemExit(f"no attempts with token accounting ({skipped} older lines skipped)")
    methods = sorted({g["method"] for g in rounds})

    per_level = defaultdict(lambda: defaultdict(float))
    count = defaultdict(int)
    for g in rounds:
        count[g["level"]] += 1
        for s in SEGMENTS:
            per_level[g["level"]][s] += g["split"].get(s, 0)
        per_level[g["level"]]["prompt"] += g["prompt"]
        per_level[g["level"]]["answer"] += sum(g["answers"]) / len(g["answers"])
    with open(a.out + ".csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["level", "rounds"] + [f"mean {s}" for s in SEGMENTS]
                   + ["mean prompt", "mean answer", "share: prev_code+feedback+ledger"])
        for lv in sorted(per_level):
            n = count[lv]
            m = {k: v / n for k, v in per_level[lv].items()}
            repair = m["prev_code"] + m["feedback"] + m["ledger"]
            w.writerow([lv, n] + [round(m[s]) for s in SEGMENTS]
                       + [round(m["prompt"]), round(m["answer"]),
                          f"{repair / max(m['prompt'], 1):.0%}"])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(max(8, len(rounds) * 0.22), 5))
    x = range(len(rounds))
    bottom = [0] * len(rounds)
    for s in SEGMENTS + ("answer",):
        h = [(sum(g["answers"]) / len(g["answers"])) if s == "answer" else g["split"].get(s, 0)
             for g in rounds]
        if any(h):
            ax.bar(x, h, bottom=bottom, color=COLORS[s], width=0.85,
                   label="answer (mean of samples)" if s == "answer" else s)
            bottom = [b + v for b, v in zip(bottom, h)]
    ax.axhline(a.context, color="black", lw=1, ls="--")
    ax.text(0, a.context, f" context {a.context:,}", va="bottom", fontsize=8)
    edges = [i for i in range(1, len(rounds))
             if (rounds[i]["level"], rounds[i]["run"]) != (rounds[i - 1]["level"], rounds[i - 1]["run"])]
    for e in edges:
        ax.axvline(e - 0.5, color="#cccccc", lw=0.8)
    starts = [0] + edges
    ax.set_xticks(starts)
    ax.set_xticklabels([f"L{rounds[i]['level']} r{rounds[i]['run'] + 1}" for i in starts],
                       fontsize=7, rotation=90)
    ax.set_ylabel("tokens")
    ax.set_title(f"Tokens per round, by what they were spent on\n({len(rounds)} rounds; "
                 f"split counted by {', '.join(methods)})"
                 + (f"\n{unmatched} attempt(s) not in the usage log kept chars/4 estimates"
                    if a.usage is not None and unmatched else ""), fontsize=10)
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(a.out + ".png", dpi=150, bbox_inches="tight")
    print(f"{len(rounds)} rounds ({skipped} older lines without token data skipped) -> "
          f"{a.out}.png, {a.out}.csv")
    if a.usage is not None:
        n = sum(len(g["answers"]) for g in rounds)
        print(f"usage log: {n - unmatched} of {n} attempts matched to server counts; {unmatched} "
              f"unmatched kept their estimates"
              + (": " + ", ".join(f"L{g['level']} run {g['run'] + 1} round {g['round']} "
                                  f"({g['unmatched']})" for g in rounds if g.get("unmatched"))
                 if unmatched else ""))


if __name__ == "__main__":
    main()
