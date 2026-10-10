#!/usr/bin/env python3
"""Verification step (testbench + DV). Works on a design folder:

  designs/<name>/
    spec.txt      <- from the prompt->code step: module header + behaviour
    design.v      <- from the prompt->code step (or a human)
    tb.v          -> written here (or reused if it already exists)
    report.json   -> written here: pass/fail, mismatches, coverage, mutation score

  python verify.py designs/adder4            # verify one design
  python verify.py --all                     # every folder in designs/
  python verify.py designs/adder4 --regen    # throw away tb.v and make a new one
  python verify.py --examples                # create sample folders from the built-in problems
  python verify.py --design my.v --spec "4-bit gray code converter"   # one file, no folder needed
  python verify.py --from-json from_prompt.json                        # the prompt step's JSON

The testbench is printed. If the design FAILS, designs/<name>/for_feedback.json is written:
the raw results (inputs, expected, got) for the feedback step to turn into a hint for the AI.
If it PASSES, designs/<name>/for_optimize.json is written (design + the testbench that proves it)
and the name is added to designs/PASSED.txt, for the optimize step.
"""
import os, sys, json, glob, argparse
import vagent
JUDGE = True   # --no-judge to switch the design-vs-golden-model judge off
from vagent import P, simulate, auto_tb, header
import mutate

def show(title, text, limit=80):
    lines = text.rstrip().splitlines()
    print(f"\n----- {title} ({len(lines)} lines) -----")
    print("\n".join(lines[:limit]) + (f"\n... ({len(lines) - limit} more lines)" if len(lines) > limit else ""))
    print("-" * (len(title) + 20))

def verify(folder, regen=False, do_mutation=True, print_tb=True):
    name = os.path.basename(os.path.normpath(folder))
    spec_p, design_p, tb_p = (os.path.join(folder, f) for f in ("spec.txt", "design.v", "tb.v"))
    if not os.path.exists(design_p):
        print(f"[{name}] no design.v, skipping")
        return None
    design = open(design_p).read()
    spec = open(spec_p).read().strip() if os.path.exists(spec_p) else ""
    report = dict(name=name, spec=spec)
    print(f"\n=== verify {name} ===", flush=True)

    # 1. testbench: reuse a fixed one (so the target never moves), or have the AI write one FROM THE SPEC
    if os.path.exists(tb_p) and not regen:
        tb, report["testbench"] = open(tb_p).read(), "existing tb.v"
        print("  using existing tb.v", flush=True)
    elif not spec:
        tb, report["testbench"] = None, "none (no spec.txt, so no testbench can be written)"
        print("  no spec.txt -> cannot write a testbench; syntax check only", flush=True)
    else:
        # the header the testbench must match: from the spec if it has one, else from the design
        dh = vagent.ansi_header(design)
        hdr_src = design if dh and vagent.ports(dh) else (spec if header(spec) else design)
        with open(os.path.join(folder, "tb_attempts.jsonl"), "a") as log:
            tb = auto_tb(spec, hdr_src, log, name)
        report["testbench"] = "AI-written" if tb else "none (AI could not write a valid one)"
        if tb:
            open(tb_p, "w").write(tb)
    try:
        report["tb_attempts"] = [json.loads(l) for l in open(os.path.join(folder, "tb_attempts.jsonl"))
                                 if '"kind": "testbench"' in l][-3:]
        for a in report["tb_attempts"]:
            a.pop("tb", None)
    except FileNotFoundError:
        pass

    if tb and print_tb:
        show(f"testbench for {name}", tb)

    # 2. run the design through it
    score, detailed, _ = simulate(design, tb)

    # 2b. the judge: a design failing a JUST-BUILT testbench may mean the golden model misread the spec.
    #     (never for an existing tb.v: that one was already trusted, e.g. when re-checking a shrunk design)
    report["judge"] = None
    fresh = report.get("testbench") == "AI-written"
    mism = [l for l in detailed.splitlines() if l.startswith("MISMATCH")]
    if tb and fresh and score < 1.0 and mism and JUDGE:
        verdict, details = vagent.judge(spec, vagent.ansi_header(hdr_src), tb, mism)
        print(f"  [judge] design or golden model? -> {verdict}", flush=True)
        for d_ in details:
            print(f"    {d_}", flush=True)
        report["judge"] = dict(verdict=verdict, details=details)
        if verdict == "design":
            print("  [judge] the golden model looks wrong -> rebuilding the testbench once", flush=True)
            cases = "; ".join(d_.split(" -> ")[0].split(":")[0] for d_ in details)
            with open(os.path.join(folder, "tb_attempts.jsonl"), "a") as log:
                tb2 = auto_tb(spec, hdr_src, log, name, hint=(
                    f"a reviewer found that a previous reference model gave WRONG outputs for these inputs: {cases}. "
                    "Re-read the spec word by word (bit positions, which bit wins, what reset clears) and compute "
                    "those cases very carefully."))
            if tb2:
                tb = tb2
                open(tb_p, "w").write(tb)
                score, detailed, _ = simulate(design, tb)
                print(f"  rebuilt testbench -> design score={score:.3f}", flush=True)
                mism2 = [l for l in detailed.splitlines() if l.startswith("MISMATCH")]
                if score < 1.0 and mism2:
                    verdict2, details2 = vagent.judge(spec, vagent.ansi_header(hdr_src), tb, mism2)
                    print(f"  [judge] after rebuild: design or golden model? -> {verdict2}", flush=True)
                    for d_ in details2:
                        print(f"    {d_}", flush=True)
                    report["judge"]["after_rebuild"] = dict(verdict=verdict2, details=details2)
                    if verdict2 == "design":
                        report["testbench_suspect"] = True
    report.update(passed=(score == 1.0 and tb is not None), behaviour_tested=tb is not None,
                  score=round(score, 4), result=detailed,
                  mismatches=[l for l in detailed.splitlines() if l.startswith("MISMATCH")])
    status = "PASS" if report["passed"] else ("COMPILES, NOT TESTED" if tb is None and score == 1 else "FAIL")
    print(f"  design: {status}  score={score:.3f}  {detailed.splitlines()[0]}", flush=True)
    for m in report["mismatches"][:5]:
        print(f"    {m}")

    # 3. how strong is the testbench? (only meaningful when the design is correct)
    report["mutation"] = None
    if do_mutation and report["passed"]:
        r = mutate.score(design, tb, show=3)
        if r.get("valid") and r["total"]:
            report["mutation"] = dict(killed=r["killed"], total=r["total"],
                                      score=round(r["killed"] / r["total"], 3), survivors=r["survivors"])

    # 4. hand-off to the feedback step: raw facts only (what was tested, what was expected, what came out)
    hand = os.path.join(folder, "for_feedback.json")
    review = os.path.join(folder, "for_review.json")
    if os.path.exists(review):
        os.remove(review)
    if report.get("testbench_suspect"):
        json.dump(dict(name=name, spec=spec, design=design, testbench=tb, result=detailed, judge=report["judge"],
                       why="the design and the golden model disagree and the judge sided with the design twice; "
                           "a human should decide before anyone 'fixes' this design"),
                  open(review, "w"), indent=2)
        print(f"  ⚠ testbench suspect -> NOT sent to feedback; human review: {review}", flush=True)
        if os.path.exists(hand):
            os.remove(hand)
    elif tb and not report["passed"]:
        json.dump(dict(name=name, spec=spec, design=design, testbench=tb, score=report["score"],
                       result=detailed, mismatches=report["mismatches"]), open(hand, "w"), indent=2)
        print(f"  ✗ FAILED -> handed to feedback: {hand}", flush=True)
    elif os.path.exists(hand):
        os.remove(hand)

    # 5. hand-off to the optimize (shrink) step: a verified design + the testbench that proves it
    opt = os.path.join(folder, "for_optimize.json")
    index = os.path.join(os.path.dirname(os.path.normpath(folder)) or ".", "PASSED.txt")
    listed = open(index).read().split() if os.path.exists(index) else []
    if report["passed"]:
        json.dump(dict(name=name, spec=spec, design=design, testbench=tb,
                       mutation=report["mutation"]), open(opt, "w"), indent=2)
        if name not in listed:
            open(index, "a").write(name + "\n")
        print(f"  ✓ PASSED -> ready for optimize: {opt}", flush=True)
    else:
        if os.path.exists(opt):
            os.remove(opt)
        if name in listed:
            open(index, "w").write("".join(n + "\n" for n in listed if n != name))

    json.dump(report, open(os.path.join(folder, "report.json"), "w"), indent=2)
    print(f"  -> {os.path.join(folder, 'report.json')}", flush=True)
    return report

CODE_KEYS = ("verilog", "design", "code", "verilog_code", "module", "verilog_file", "file", "path", "design_file", "out")
SPEC_KEYS = ("spec", "prompt", "request", "description", "question", "task")

def from_prompt_json(path):
    """Read the prompt step's JSON. Accepts the Verilog as code or as a path to a .v file,
    and the prompt under any common key name. Returns (verilog_code, spec_text)."""
    j = json.load(open(path))
    if isinstance(j, list):          # a list of attempts: take the last one
        j = j[-1]
    code = spec = ""
    for k in CODE_KEYS:
        v = j.get(k)
        if isinstance(v, str) and v.strip():
            p = v.strip()
            for cand in (p, os.path.join(os.path.dirname(path), p)):
                if p.endswith(".v") and os.path.exists(cand):
                    code = open(cand).read()
                    break
            else:
                if "module" in v:
                    code = v
            if code:
                break
    for k in SPEC_KEYS:
        v = j.get(k)
        if isinstance(v, str) and v.strip():
            spec = v.strip()
            break
    if not code:
        sys.exit(f"{path}: no Verilog found. Expected one of {CODE_KEYS} holding the code or a .v path")
    return code, spec

def examples(root="designs"):
    """Sample folders so the verification step can be tested without the prompt->code step."""
    for n in P:
        for kind, code in (("buggy", P[n]["buggy"]), ("ok", P[n]["ref"])):
            d = os.path.join(root, f"{n}_{kind}")
            os.makedirs(d, exist_ok=True)
            open(os.path.join(d, "spec.txt"), "w").write(P[n]["spec"] + "\n")
            open(os.path.join(d, "design.v"), "w").write(code)
            print("created", d)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="*")
    ap.add_argument("--all", action="store_true", help="every folder in designs/")
    ap.add_argument("--regen", action="store_true", help="ignore an existing tb.v and write a new one")
    ap.add_argument("--no-mutation", action="store_true")
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--examples", action="store_true", help="create sample design folders")
    ap.add_argument("--design", help="a Verilog file to verify (makes designs/<module>/ for you)")
    ap.add_argument("--from-json", dest="from_json", nargs="+",
                    help="JSON file(s) from the prompt step: the Verilog (a .v path or the code) + the prompt")
    ap.add_argument("--spec", help="what the design should do (with --design)")
    ap.add_argument("--no-print", action="store_true", help="don't print the testbench")
    ap.add_argument("--no-judge", action="store_true", help="always blame the design when it fails")
    a = ap.parse_args()
    vagent.SAMPLES = max(1, a.samples)
    global JUDGE
    JUDGE = not a.no_judge
    if a.examples:
        return examples()
    if a.from_json:
        a.folders = []
        for jp in a.from_json:
            code, spec = from_prompt_json(jp)
            m = vagent.re.search(r"\bmodule\s+(\w+)", code)
            d = os.path.join("designs", m[1] if m else os.path.splitext(os.path.basename(jp))[0])
            os.makedirs(d, exist_ok=True)
            open(os.path.join(d, "design.v"), "w").write(code)
            if spec:
                open(os.path.join(d, "spec.txt"), "w").write(spec + "\n")
            print(f"[{jp}] -> {d}  (prompt: {spec[:60]!r})", flush=True)
            a.folders.append(d)
    elif a.design:
        code = open(a.design).read()
        m = vagent.re.search(r"\bmodule\s+(\w+)", code)
        d = os.path.join("designs", m[1] if m else "circuit")
        os.makedirs(d, exist_ok=True)
        open(os.path.join(d, "design.v"), "w").write(code)
        if a.spec:
            open(os.path.join(d, "spec.txt"), "w").write(a.spec + "\n")
        a.folders = [d]
    folders = sorted(glob.glob("designs/*/")) if a.all else a.folders
    if not folders:
        return ap.print_help()
    reports = [r for f in folders if (r := verify(f, a.regen, not a.no_mutation, not a.no_print))]
    print("\n===== VERIFY SUMMARY =====")
    for r in reports:
        mut = f"{r['mutation']['killed']}/{r['mutation']['total']}" if r["mutation"] else "-"
        st = "PASS" if r["passed"] else ("NOT TESTED" if not r["behaviour_tested"] else "FAIL")
        print(f"{r['name']:16s} {st:10s} score={r['score']:.3f}  testbench={r['testbench']:14.14s}  mutants killed={mut}")

if __name__ == "__main__":
    main()
