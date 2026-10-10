import csv
import json
from pathlib import Path

RESULTS = Path("experiment-results")

FILES = [
    "with-tools-seed7.jsonl",
    "with-tools-seed42.jsonl",
    "original-full-seed21.jsonl",
    "original-level1-1-seed7.jsonl",
    "original-level1-2-seed7.jsonl",
    "verified-full-seed7.jsonl",
    "verified-full-seed42.jsonl",
    "verified-full-seed21.jsonl",
    "verified-level1-1-seed7.jsonl",
    "verified-level1-2-seed7.jsonl",
    "independent-level1-3-seed7.jsonl",
    "independent-level1-2-seed7.jsonl",
]

rows = []

for filename in FILES:
    path = RESULTS / filename

    if not path.exists():
        print(f"Skipping missing file: {filename}")
        continue

    with path.open() as f:
        attempts = [
            json.loads(line)
            for line in f
            if line.strip()
        ]

    if not attempts:
        continue

    best_reward = max(float(a["reward"]) for a in attempts)
    rounds = len(set(a["round"] for a in attempts))
    tool_calls = sum(int(a.get("tool_calls", 0)) for a in attempts)

    rows.append({
        "experiment": filename,
        "problem": attempts[0]["problem"],
        "seed": attempts[0]["seed"],
        "best_reward": best_reward,
        "solved": best_reward >= 1.0,
        "rounds": rounds,
        "candidate_attempts": len(attempts),
        "calculator_calls": tool_calls,
    })

output = RESULTS / "benchmark-analysis.csv"

with output.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "experiment",
        "problem",
        "seed",
        "best_reward",
        "solved",
        "rounds",
        "candidate_attempts",
        "calculator_calls",
    ])
    writer.writeheader()
    writer.writerows(rows)

print(f"\nSaved: {output}\n")

for row in rows:
    status = "SOLVED" if row["solved"] else "FAILED"
    print(
        f"{row['experiment']:<38} "
        f"reward={row['best_reward']:.1f} "
        f"rounds={row['rounds']} "
        f"attempts={row['candidate_attempts']} "
        f"tools={row['calculator_calls']} "
        f"{status}"
    )
