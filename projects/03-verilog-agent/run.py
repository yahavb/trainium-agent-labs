#!/usr/bin/env python3
"""The whole loop in one command (prompt -> verify -> fix -> verify ... -> pass).

  prompt.py  (prompt step)   request -> AI -> designs/<module>/design.v + spec.txt
  verify.py  (testbench/DV)  testbench + check; FAIL -> for_feedback.json, PASS -> for_optimize.json
  prompt.py --fix            for_feedback.json -> AI -> new design.v
  ... repeat until it passes, the testbench is suspect (for_review.json), or the rounds run out.

  python run.py "8-bit ALU with add, sub, and, or, xor"
  python run.py "4-bit up counter with clk, synchronous reset and enable" --rounds 3
  python run.py "add an overflow flag" --code alu.v --spec "8-bit ALU ... with an overflow flag"
"""
import argparse, json, os, sys, time
import vagent
import prompt as P
import verify as V

def main():
    ap = argparse.ArgumentParser(description="Describe a circuit; get a verified Verilog design.")
    ap.add_argument("request", nargs="*")
    ap.add_argument("--code", help="existing Verilog to modify")
    ap.add_argument("--spec", help="full behaviour (default: the request)")
    ap.add_argument("--rounds", type=int, default=3, help="fix rounds after the first check")
    ap.add_argument("--designs", default="designs")
    ap.add_argument("--log", default="attempts.jsonl")
    ap.add_argument("--samples", type=int, default=1)
    a = ap.parse_args()
    vagent.SAMPLES = max(1, a.samples)
    t0 = time.time()

    print("\n[prompt] AI writes the circuit", flush=True)
    try:
        d = generate_and_show(argparse.Namespace(request=a.request, code=a.code, spec=a.spec,
                                                 designs=a.designs, log=a.log, fix=None))
    except SystemExit as e:
        sys.exit(f"prompt step stopped: {e}")

    outcome, history = check_and_fix(d, a.rounds, a.log)
    summary(d, outcome, history, t0)

def generate_and_show(ns):
    """Run the prompt step, then show the circuit it wrote as one clear block before testing starts."""
    import io, contextlib
    print("  the AI is writing the circuit...", flush=True)
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            d = P.generate(ns)
    except SystemExit:
        print(buf.getvalue(), end="")
        raise
    code = open(os.path.join(d, "design.v")).read().rstrip()
    print(f"\n===== AI-generated circuit ({code.count(chr(10)) + 1} lines) -> {d}/design.v =====")
    print(code)
    print("=" * 60, flush=True)
    return d

def check_and_fix(d, rounds=3, log="attempts.jsonl", fix=True):
    """verify -> (fix -> verify) ... until PASS, review, untestable, or out of rounds."""
    history, outcome = [], None
    for rnd in range(rounds + 1):
        print(f"\n[verify] round {rnd}", flush=True)
        r = V.verify(d, print_tb=(rnd == 0), do_mutation=True)
        if r is None:
            outcome = "no design.v"
            break
        history.append(round(r["score"], 3))
        if r["passed"]:
            outcome = "PASS"
            break
        if r.get("testbench_suspect"):
            outcome = "REVIEW (the golden model looks wrong; a human should decide)"
            break
        if not r["behaviour_tested"]:
            outcome = "NOT TESTED (no valid testbench; only syntax was checked)"
            break
        if not fix:
            outcome = "FAIL (fixing was not requested; see for_feedback.json)"
            break
        if rnd == rounds:
            outcome = f"FAIL after {rounds} fix round(s)"
            break
        print(f"\n[fix] AI fixes the design from for_feedback.json (round {rnd + 1})", flush=True)
        try:
            P.fix(argparse.Namespace(fix=d, log=log))
        except SystemExit as e:
            outcome = f"fix step stopped: {e}"
            break

    return outcome, history

def summary(d, outcome, history, t0):
    rep = json.load(open(os.path.join(d, "report.json"))) if os.path.exists(os.path.join(d, "report.json")) else {}
    mut = rep.get("mutation")
    dp = os.path.join(d, "design.v")
    if os.path.exists(dp):
        code = open(dp).read().rstrip()
        print(f"\n----- final circuit: {dp} ({code.count(chr(10)) + 1} lines) -----")
        print(code)
        print("-" * 60)
    print(f"\n===== RESULT: {os.path.basename(d)} =====")
    print(f"outcome    : {outcome}")
    print(f"scores     : {' -> '.join(str(s) for s in history)}")
    if mut:
        print(f"testbench  : caught {mut['killed']}/{mut['total']} planted bugs")
    print(f"time       : {time.time() - t0:.0f}s")
    print(f"folder     : {d}/")
    if outcome == "PASS":
        print("next       : optimize step takes design.v (listed in designs/PASSED.txt)")
    elif outcome and outcome.startswith("REVIEW"):
        print(f"next       : read {d}/for_review.json")

if __name__ == "__main__":
    main()
