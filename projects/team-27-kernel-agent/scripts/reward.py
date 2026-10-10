"""Per-run reward in the upstream NKI agent's format (upstream-nki/FEEDBACK-EXPERIMENTS.md):
0.1 parses + 0.2 obeys the rules + 0.2 runs without crashing + 0.5 x (dev cases correct / cases),
best attempt of each run; "solved" = verified (dev and holdout). Comparable across Stage A and B.

    uv run python scripts/reward.py results/seat-130/ours-L1234-seat-130.jsonl
"""
import json
import re
import sys
from collections import defaultdict


def reward(a):
    if not a.get("summary"):
        return 0.0
    m = re.search(r"(\d+) rule violation\(s\), (\d+)/(\d+) cases correct", a["summary"])
    viol, ok, n = map(int, m.groups())
    crashed = str(a.get("key")).startswith("crash")
    return 0.1 + 0.2 * (viol == 0) + 0.2 * (not crashed) + 0.5 * ok / n


for path in sys.argv[1:]:
    best, solved = defaultdict(dict), defaultdict(dict)
    for line in open(path):
        r = json.loads(line)
        key = (r["level"], r["run"])
        if r.get("type") == "attempt":
            best[r["level"]][r["run"]] = max(best[r["level"]].get(r["run"], 0.0), reward(r))
        elif r.get("type") == "result":
            solved[r["level"]][r["run"]] = r["status"] == "verified"
    print(path)
    for lv in sorted(best):
        runs = sorted(best[lv])
        print(f"level {lv}: solved {sum(solved[lv].values())}/{len(runs)}   all = "
              f"[{', '.join(f'{best[lv][k]:.2f}' for k in runs)}]")
