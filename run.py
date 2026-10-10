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
    ap.add_argument("--no-optimize", action="store_true", help="skip the gate-count optimize step")
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
    opt = optimize(d) if outcome == "PASS" and not a.no_optimize else None
    summary(d, outcome, history, t0, opt)

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
        if r.get("testbench_error"):
            outcome = "TESTBENCH ERROR (the testbench does not compile; the design was not blamed)"
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

def optimize(d, rounds=8):
    """Optimize step (teammate's optimize_agent.py): AI shrinks the circuit, Yosys counts cells, and every
    smaller version must pass the SAME verified testbench (verify.optimize_check) or it is thrown away."""
    import shutil
    try:
        import optimize_agent as OA
        from yosys_checker import synthesize
    except ImportError as e:
        print(f"  [optimize] optimize step not available ({e})")
        return None
    if not shutil.which("yosys"):
        print("  [optimize] yosys is not installed. Run: apt-get install -y yosys")
        return None
    design, tb = os.path.join(d, "design.v"), os.path.join(d, "tb.v")
    if not (os.path.exists(design) and os.path.exists(tb)):
        print("  [optimize] needs a verified design.v and tb.v")
        return None
    before, top = synthesize(os.path.abspath(design))
    req = os.path.join(d, "optimize_request.json")
    json.dump({"verilog": os.path.abspath(design), "testbench": os.path.abspath(tb)}, open(req, "w"), indent=2)
    last_ok = {"path": None}
    try:
        OA._find_compare_script()                     # the comparator (teammate's comparator.py) is the judge
        compare = OA.member3_check
        print("  [optimize] checker: comparator.py (same testbench on the original and the smaller version)")
    except FileNotFoundError:
        compare = V.optimize_check                    # fallback: the verified testbench must still pass
        print("  [optimize] checker: verify.optimize_check (comparator.py not found)")
    def checker(vprev, vnew, testbench):
        last_ok["path"] = vprev                       # vprev is always the last VERIFIED version
        return compare(vprev, vnew, testbench)
    try:
        final = OA.run(req, checker=checker, max_rounds=rounds, work_dir=os.path.join(d, "opt_work"))
    except Exception as e:
        print(f"  [optimize] stopped: {e}")
        return None
    final = final or last_ok["path"] or design
    after = synthesize(os.path.abspath(final), top)[0] if before is not None else None
    out = os.path.join(d, "optimized.v")
    shutil.copy(final, out)
    rep = dict(cells_before=before, cells_after=after, top=top, optimized=out,
               reduction=(round(1 - after / before, 3) if before and after is not None else None))
    json.dump(rep, open(os.path.join(d, "optimize_report.json"), "w"), indent=2)
    code = open(out).read().rstrip()
    print(f"\n----- optimized circuit: {out} ({before} -> {after} cells) -----")
    print(code)
    print("-" * 60)
    return rep

def summary(d, outcome, history, t0, opt=None):
    rep = json.load(open(os.path.join(d, "report.json"))) if os.path.exists(os.path.join(d, "report.json")) else {}
    mut = rep.get("mutation")
    dp = os.path.join(d, "design.v")
    if opt and opt.get("reduction") and os.path.exists(opt.get("optimized", "")):
        dp = opt["optimized"]                        # the shrunk, verified version is the final circuit
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
    if opt and opt.get("cells_before") is not None:
        print(f"optimize   : {opt['cells_before']} -> {opt['cells_after']} cells"
              + (f" ({opt['reduction']:.0%} smaller)" if opt.get("reduction") else " (no smaller verified version)"))
    print(f"time       : {time.time() - t0:.0f}s")
    print(f"folder     : {d}/")
    if outcome == "PASS":
        if opt and opt.get("reduction"):
            print(f"next       : original = {d}/design.v, smaller verified version = {opt['optimized']}")
        else:
            print("next       : optimize step takes design.v (listed in designs/PASSED.txt)")
    elif outcome and outcome.startswith("REVIEW"):
        print(f"next       : read {d}/for_review.json")

if __name__ == "__main__":
    main()
