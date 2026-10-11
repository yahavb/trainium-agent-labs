#!/usr/bin/env python3
"""
Compare an original Verilog design and an optimized candidate using a
user-provided testbench.

The exact same testbench is run against both designs. The optimized design is
selected only if BOTH designs pass the testbench.

Example:
  python compare_verilog.py --original original.v --optimized optimized.v --tb tb.v

Requirements:
  - vagent.py must be importable from this script's working directory/PYTHONPATH.
  - Icarus Verilog (iverilog and vvp) must be installed and on PATH.
"""
import argparse
import json
import sys
import time
from pathlib import Path

try:
    import vagent
except ImportError as exc:
    raise SystemExit(
        "Could not import vagent.py. Put this script beside vagent.py "
        "or add its directory to PYTHONPATH."
    ) from exc


def read_text(path, label):
    try:
        content = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise SystemExit(f"Cannot read {label} file {path!r}: {exc}") from exc
    if not content.strip():
        raise SystemExit(f"The {label} file {path!r} is empty.")
    return content


def evaluate(code, tb):
    """Call vagent.simulate with Verilog source strings, not file paths."""
    score, detailed, basic = vagent.simulate(code, tb)
    score = float(score)
    return {
        "score": score,
        "passed": score == 1.0,
        "detailed_feedback": detailed,
        "basic_feedback": basic,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", required=True, help="Path to original .v source")
    parser.add_argument("--optimized", required=True, help="Path to optimized candidate .v source")
    parser.add_argument("--tb", required=True, help="Path to the existing self-checking Verilog testbench")
    parser.add_argument("--result", default="verification_result.json", help="Output JSON report path")
    parser.add_argument("--selected", default="selected.v", help="Output path for the selected Verilog design")
    args = parser.parse_args()

    original_path = Path(args.original)
    optimized_path = Path(args.optimized)
    tb_path = Path(args.tb)
    result_path = Path(args.result)
    selected_path = Path(args.selected)

    original = read_text(original_path, "original design")
    optimized = read_text(optimized_path, "optimized design")
    tb = read_text(tb_path, "testbench")

    started = time.time()

    # Both designs are evaluated with precisely the same provided testbench.
    original_result = evaluate(original, tb)
    optimized_result = evaluate(optimized, tb)

    if original_result["passed"] and optimized_result["passed"]:
        selected_source = "optimized"
        selected_code = optimized
        decision = "optimized_candidate_passed_shared_testbench"
    else:
        selected_source = "original"
        selected_code = original
        decision = "kept_original_because_both_designs_did_not_pass"

    selected_path.parent.mkdir(parents=True, exist_ok=True)
    selected_path.write_text(selected_code, encoding="utf-8")

    result = {
        "schema_version": 1,
        "status": "completed",
        "decision": decision,
        "selected_source": selected_source,
        "selected_design": selected_code,
        "selected_file": str(selected_path),
        "original_file": str(original_path),
        "optimized_file": str(optimized_path),
        "testbench_file": str(tb_path),
        "testbench": tb,
        "testbench_generated": False,
        "original": original_result,
        "optimized": optimized_result,
        "optimization_accepted": selected_source == "optimized",
        "elapsed_seconds": round(time.time() - started, 3),
        "verification_note": (
            "Passing the same testbench means both designs passed its checks; "
            "it does not prove formal equivalence for every possible input."
        ),
    }

    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"Original score:  {original_result['score']}")
    print(f"Optimized score: {optimized_result['score']}")
    print(f"Decision:        {selected_source} ({decision})")
    print(f"Selected source: {selected_path}")
    print(f"JSON result:     {result_path}")

    # Nonzero exit code when the optimized candidate is not accepted.
    return 0 if selected_source == "optimized" else 2


if __name__ == "__main__":
    sys.exit(main())

