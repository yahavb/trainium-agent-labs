import argparse
import csv
import json
import statistics
from pathlib import Path

from agent import load_json, run_question

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "neuron_docs.json"
QUESTIONS_FILE = ROOT / "questions.json"
DATA_DIR = ROOT / "data"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args()

    if args.runs < 1 or args.max_attempts < 1:
        parser.error("--runs and --max-attempts must be positive")

    docs = load_json(DATA_FILE)
    questions = load_json(QUESTIONS_FILE)
    results = []

    for run_id in range(1, args.runs + 1):
        print(f"\n===== EXPERIMENT RUN {run_id}/{args.runs} =====")

        for question in questions:
            result = run_question(
                question,
                docs,
                max_attempts=args.max_attempts,
                run_id=run_id
            )
            results.append(result)

    csv_path = DATA_DIR / "experiment_results.csv"
    fields = [
        "run_id", "question_id", "question", "score",
        "passed", "attempts_used", "missing_facts", "answer"
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for result in results:
            row = result.copy()
            row["missing_facts"] = ", ".join(row["missing_facts"])
            writer.writerow(row)

    scores = [r["score"] for r in results]
    pass_rate = (
        sum(r["passed"] for r in results) / len(results)
        if results else 0
    )

    summary = {
        "runs": args.runs,
        "questions_per_run": len(questions),
        "total_evaluations": len(results),
        "pass_rate": round(pass_rate, 4),
        "mean_score": round(statistics.mean(scores), 2) if scores else 0,
        "score_std_dev": round(statistics.pstdev(scores), 2) if len(scores) > 1 else 0,
        "mean_attempts": round(
            statistics.mean(r["attempts_used"] for r in results), 2
        ) if results else 0,
        "per_question": {}
    }

    for question in questions:
        subset = [
            r for r in results
            if r["question_id"] == question["id"]
        ]
        summary["per_question"][question["id"]] = {
            "pass_rate": round(
                sum(r["passed"] for r in subset) / len(subset), 4
            ) if subset else 0,
            "mean_score": round(
                statistics.mean(r["score"] for r in subset), 2
            ) if subset else 0
        }

    summary_path = DATA_DIR / "experiment_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n===== SUMMARY =====")
    print(json.dumps(summary, indent=2))
    print("\nSaved:", csv_path)
    print("Saved:", summary_path)


if __name__ == "__main__":
    main()