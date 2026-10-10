#!/usr/bin/env python3
"""
Iterative Verilog simplification checker.

Flow:
  current design -> Icarus functional checker -> Yosys synthesis metrics
  -> separate simplifier_agent.py -> check the returned design again.
Accept a candidate ONLY if it passes the same testbench and has fewer
Yosys generic cells. Stop at the first rejected/non-smaller candidate and
print/save the best passing design found.

Place this beside vagent.py and simplifier_agent.py.
Requires: Python, Icarus Verilog (iverilog + vvp), Yosys, and the model
service configured for vagent.py.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import vagent as base


def module_name(code):
    match = re.search(r"\bmodule\s+([A-Za-z_$][\w$]*)", code)
    return match.group(1) if match else None


def synthesize_metrics(code, timeout=90):
    """Return generic synthesized cell count and cell-type counts."""
    if not shutil.which("yosys"):
        raise RuntimeError("Yosys was not found on PATH.")
    top = module_name(code)
    if not top:
        raise RuntimeError("Could not find a Verilog module declaration.")

    with tempfile.TemporaryDirectory(prefix="verilog_check_") as tmp:
        src = Path(tmp) / "design.v"
        out = Path(tmp) / "netlist.json"
        src.write_text(code, encoding="utf-8")

        # Reject unexpected module-name characters before inserting into script.
        if not re.fullmatch(r"[A-Za-z_$][\w$]*", top):
            raise RuntimeError("Invalid top module name.")

        script = (
            f"read_verilog -sv {src}; "
            f"hierarchy -check -top {top}; "
            f"synth -top {top}; "
            f"write_json {out}"
        )
        run = subprocess.run(
            ["yosys", "-p", script],
            capture_output=True, text=True, timeout=timeout
        )
        if run.returncode != 0 or not out.exists():
            detail = (run.stderr + "\n" + run.stdout).strip()
            raise RuntimeError("Yosys synthesis failed:\n" + detail[-1800:])

        design = json.loads(out.read_text(encoding="utf-8"))
        cells = design["modules"][top].get("cells", {})
        types = {}
        for cell in cells.values():
            typ = cell.get("type", "unknown")
            types[typ] = types.get(typ, 0) + 1
        return {"top": top, "cell_count": len(cells), "cell_types": types}


def ask_simplifier(agent_path, spec, code, metrics, round_no, feedback, timeout=900):
    """Call the separate simplifier program and read its candidate output."""
    with tempfile.TemporaryDirectory(prefix="simplifier_request_") as tmp:
        tmp = Path(tmp)
        request_path = tmp / "request.json"
        output_path = tmp / "candidate.v"
        request_path.write_text(json.dumps({
            "spec": spec,
            "code": code,
            "metrics": metrics,
            "round": round_no,
            "feedback": feedback,
            "output_path": str(output_path),
        }), encoding="utf-8")

        run = subprocess.run(
            [sys.executable, str(agent_path), "--request", str(request_path)],
            capture_output=True, text=True, timeout=timeout
        )
        if run.returncode != 0 or not output_path.exists():
            detail = (run.stderr + "\n" + run.stdout).strip()
            raise RuntimeError(
                "Simplifier agent failed to return a candidate:\n" + detail[-1800:]
            )
        candidate = output_path.read_text(encoding="utf-8").strip()
        if not candidate or not module_name(candidate):
            raise RuntimeError("Simplifier agent returned no recognizable Verilog module.")
        return candidate


def check_functionality(code, testbench):
    score, detail, basic = base.simulate(code, testbench)
    return score == 1.0, detail, score


def main():
    parser = argparse.ArgumentParser(description="Iteratively simplify a Verilog circuit.")
    parser.add_argument("--code", help="Path to starting Verilog source. If omitted, paste it interactively.")
    parser.add_argument("--spec", help="Short but precise description of required circuit behavior.")
    parser.add_argument("--testbench", help="Path to a self-checking Verilog testbench.")
    parser.add_argument("--rounds", type=int, default=10, help="Maximum simplification attempts (default: 10).")
    parser.add_argument("--output", default="most_simplified.v", help="Output file (default: most_simplified.v)")
    parser.add_argument("--agent", default=str(Path(__file__).with_name("simplifier_agent.py")),
                        help="Path to the separate simplifier agent.")
    args = parser.parse_args()

    if args.rounds < 1:
        parser.error("--rounds must be at least 1")

    spec = args.spec or input("Describe required behavior and keep the module name/ports unchanged:\n> ").strip()
    if not spec:
        parser.error("A circuit specification is required.")

    if args.code:
        current = Path(args.code).read_text(encoding="utf-8")
    else:
        print("Paste the starting Verilog. Enter a line containing END when finished.")
        lines = []
        while True:
            try:
                line = input()
            except EOFError:
                break
            if line.strip() == "END":
                break
            lines.append(line)
        current = "\n".join(lines) + "\n"

    if args.testbench:
        tb = Path(args.testbench).read_text(encoding="utf-8")
    else:
        print("\nGenerating a self-checking testbench from your specification...")
        with open("simplification_attempts.jsonl", "a", encoding="utf-8") as log:
            tb = base.auto_tb(spec, current, log, "simplification")
        if not tb:
            print("Could not create a trustworthy testbench. Stopping rather than optimize unchecked code.")
            sys.exit(2)

    if not shutil.which("iverilog") or not shutil.which("vvp"):
        sys.exit("Icarus Verilog executables iverilog and vvp must be installed and on PATH.")

    try:
        ok, detail, score = check_functionality(current, tb)
        if not ok:
            print("Starting circuit failed functional verification; it will not be optimized.")
            print(detail)
            sys.exit(2)
        best_metrics = synthesize_metrics(current)
    except Exception as ex:
        sys.exit(f"Initial checking failed: {ex}")

    best_code = current
    print(f"Initial circuit: PASS; {best_metrics['cell_count']} synthesized generic cells.")

    agent_path = Path(args.agent).resolve()
    if not agent_path.exists():
        sys.exit(f"Simplifier agent not found: {agent_path}")

    for round_no in range(1, args.rounds + 1):
        feedback = (
            f"Current circuit passes the functional testbench and uses "
            f"{best_metrics['cell_count']} generic synthesized cells. "
            "Try to reduce the synthesized cell count. Preserve the exact specified "
            "behavior, module name, ports, clock/reset semantics, and synthesizability. "
            "Do not remove required behavior. Return a complete module only."
        )
        print(f"\nRound {round_no}: asking the separate simplifier agent for a smaller design...")
        try:
            candidate = ask_simplifier(
                agent_path, spec, best_code, best_metrics, round_no, feedback
            )
            candidate_top = module_name(candidate)
            if candidate_top != best_metrics["top"]:
                print(f"Rejected: top module changed from {best_metrics['top']} to {candidate_top}.")
                break

            ok, detail, score = check_functionality(candidate, tb)
            if not ok:
                print("Rejected: candidate failed functional verification.")
                print(detail)
                break

            candidate_metrics = synthesize_metrics(candidate)
            old_count = best_metrics["cell_count"]
            new_count = candidate_metrics["cell_count"]
            print(f"Candidate passes tests: {new_count} generic cells (previous best: {old_count}).")

            if new_count >= old_count:
                print("No strict reduction in cell count. Keeping the previous best and stopping.")
                break

            best_code = candidate
            best_metrics = candidate_metrics
            print(f"Accepted: improved from {old_count} to {new_count} cells.")

            with open("simplification_attempts.jsonl", "a", encoding="utf-8") as log:
                log.write(json.dumps({
                    "round": round_no,
                    "accepted": True,
                    "cell_count": new_count,
                    "cell_types": candidate_metrics["cell_types"],
                    "code": candidate,
                }) + "\n")

        except Exception as ex:
            print(f"Could not complete round {round_no}: {ex}")
            print("Keeping the best verified design found so far.")
            break

    Path(args.output).write_text(best_code.rstrip() + "\n", encoding="utf-8")
    print("\n=== MOST SIMPLIFIED DESIGN FOUND ===")
    print(f"Synthesis metric: {best_metrics['cell_count']} generic cells")
    print(f"Saved to: {Path(args.output).resolve()}")
    print("\n" + best_code)
    print(
        "Note: this is the smallest passing design found in this search, not a proof "
        "of global optimality. Generic cell count is only a rough hardware-cost proxy; "
        "target-specific area, timing, and power may differ."
    )


if __name__ == "__main__":
    main()
