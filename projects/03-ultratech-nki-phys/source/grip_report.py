"""Export measured per-case deviations without changing physics acceptance gates."""

import argparse
import json
from pathlib import Path
import uuid


def metrics(check):
    force = check["force_check"]
    return dict(residual=force["numerical"]["residual"],
                objective_error=force["numerical"]["objective_error"],
                force_error=force["force_error"],
                acceleration_error=force["acceleration_error"],
                wrench_error=check["wrench_error"], cone_violation=check["cone_violation"])


def case_deviations(summary):
    limits = dict(residual=1e-4, objective_error=1e-5, force_error=1e-4,
                  acceleration_error=1e-4, wrench_error=1e-4, cone_violation=1e-6)
    cases = {}
    for attempt in summary["scored_attempts"]:
        for case in attempt["cases"]:
            row = cases.setdefault(case["case_id"], dict(case_id=case["case_id"], passed=True,
                                                        worst_metrics={}))
            row["passed"] = row["passed"] and case["passed"]
            for phase in ("before", "after"):
                for name, value in metrics(case[phase]).items():
                    row["worst_metrics"][name] = max(value, row["worst_metrics"].get(name, 0))
    for row in cases.values():
        row["limit_multiples"] = {name: value / limits[name] for name, value in row["worst_metrics"].items()}
        row["failed_metrics"] = [name for name, ratio in row["limit_multiples"].items() if ratio > 1]
    return limits, sorted(cases.values(), key=lambda row: row["case_id"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grading", type=Path, required=True, help="Trusted grading results.json")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    summary = json.loads(args.grading.read_text())
    limits, cases = case_deviations(summary)
    out = args.out or args.grading.parent / f"deviations-{uuid.uuid4().hex}"
    out.mkdir(parents=True, exist_ok=False)
    report = dict(grading=str(args.grading.resolve()), context=summary["context"], limits=limits,
                  cases=cases, throughput_eligible=summary["throughput_eligible"])
    (out / "deviations.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# Gripping Deviations", "",
             f"Accepted evaluations: {summary['passing_case_evaluations']}/{summary['expected_case_evaluations']}. "
             f"Backend: {summary['context']['backend']}; steps: {summary['context']['steps']}.", "",
             "Worst measured error across phases/repeats; numbers in parentheses are multiples of the gate limit.",
             "Force, acceleration and wrench errors are normalized errors, not universal percentage errors.", "",
             "| Case | Status | Residual | Force Error | Acceleration Error | Wrench Error |",
             "|---|---|---:|---:|---:|---:|"]
    for case in cases:
        values = [f"{case['worst_metrics'][name]:.6g} ({case['limit_multiples'][name]:.2f}x)"
                  for name in ("residual", "force_error", "acceleration_error", "wrench_error")]
        lines.append(f"| {case['case_id']} | {'PASS' if case['passed'] else 'FAIL'} | " + " | ".join(values) + " |")
    lines += ["", "Limits: residual, force, acceleration and wrench = 0.0001; "
              "objective = 0.00001; cone violation = 0.000001.",
              "Complete metrics, including objective and friction feasibility, are in deviations.json.",
              "A case can also fail feasibility or harness checks; this table does not replace the trusted checker.",
              "Simulator results contain no device timing. Timing of a failing device run is diagnostic, "
              "not eligible accepted-solution throughput."]
    (out / "deviations.md").write_text("\n".join(lines) + "\n")
    print(out / "deviations.md")


if __name__ == "__main__":
    main()
