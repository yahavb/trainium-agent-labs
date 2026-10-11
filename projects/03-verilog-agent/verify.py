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

def optimize_check(vprev, vnew, testbench):
    """The optimize step's checker: a smaller circuit is only accepted if it passes the SAME verified
    testbench the original passed (exhaustive for <= 12 input bits, else 4000 vectors / 300 cycles)."""
    tb = open(testbench).read()
    score, det, _ = simulate(open(vnew).read(), tb)
    ok = score == 1.0
    print(f"  [check] smaller circuit vs the verified testbench: "
          f"{'PASS -> accepted' if ok else 'FAIL -> rejected (' + det.splitlines()[0] + ')'}", flush=True)
    if not ok:
        for l in [l for l in det.splitlines() if l.startswith("MISMATCH")][:3]:
            print(f"    {l}", flush=True)
    return ok

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
    user_tb = os.path.exists(os.path.join(folder, "tb_source.txt"))
    if os.path.exists(os.path.join(folder, "syntax_only.txt")):
        tb, report["testbench"] = None, "none (user asked for a syntax check only)"
        print("  syntax check only (user's choice)", flush=True)
    elif os.path.exists(tb_p) and (not regen or user_tb):
        tb = open(tb_p).read()
        report["testbench"] = "user-provided" if user_tb else "existing tb.v"
        print(f"  using {'the user-provided testbench' if user_tb else 'existing tb.v'}", flush=True)
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

    # 2a. a testbench that doesn't compile is the TESTBENCH's fault, not the design's
    report["testbench_error"] = False
    if tb and detailed.startswith("COMPILE ERROR") and simulate(design, None)[0] > 0:
        report["testbench_error"] = True
        src = tb.splitlines()
        nums = sorted({int(x) for x in vagent.re.findall(r"tb\.v:(\d+)", detailed)})[:3]
        near = sorted({m for n in nums for m in (n - 1, n) if 0 < m <= len(src) and src[m-1].strip()})
        detailed = ("TESTBENCH ERROR: the testbench does not compile (the design itself compiles fine).\n" +
                    "\n".join(l for l in detailed.splitlines()[1:] if "tb.v" in l)[:400] +
                    ("\nThe mistake is on or just before:\n" + "\n".join(f"  tb.v line {m}: {src[m-1].strip()}"
                                                                         for m in near) if near else ""))
        print("  ✗ the TESTBENCH has an error (not the design):", flush=True)
        for l in detailed.splitlines()[1:]:
            print("    " + l, flush=True)

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
    elif report.get("testbench_error"):
        if os.path.exists(hand):
            os.remove(hand)
        print("  -> not sent to feedback: fix the testbench (or press Enter next time to let the AI write one)",
              flush=True)
    elif tb and not report["passed"]:
        json.dump(dict(name=name, spec=spec, design=design, testbench=tb, score=report["score"],
                       result=detailed, mismatches=report["mismatches"]), open(hand, "w"), indent=2)
        print(f"  ✗ FAILED -> handed to feedback: {hand}", flush=True)
        # the feedback step: turn the raw failure into a diagnosis + hints (teammate's feedback.py)
        try:
            import feedback as FB
            fbr = FB.generate_feedback_from_file(hand, os.path.join(folder, "feedback_result.json"))
            print(f"  [feedback] {fbr.get('error_type')} -> fix the {fbr.get('retry_target') or 'design'}", flush=True)
            for s_ in (fbr.get("suggestions") or [])[:3]:
                print(f"    - {s_[:170]}", flush=True)
        except ImportError:
            pass
        except Exception as ex:
            print(f"  [feedback] could not analyse the failure: {ex}", flush=True)
    elif os.path.exists(hand):
        os.remove(hand)

    # 5. hand-off to the optimize (shrink) step: a verified design + the testbench that proves it
    opt = os.path.join(folder, "for_optimize.json")
    index = os.path.join(os.path.dirname(os.path.normpath(folder)) or ".", "PASSED.txt")
    listed = open(index).read().split() if os.path.exists(index) else []
    if report["passed"] and os.path.exists(os.path.join(folder, "feedback_result.json")):
        os.remove(os.path.join(folder, "feedback_result.json"))
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
    # every check is kept (report.json only holds the latest), so the attempt log shows the whole loop
    import hashlib, time as _t
    with open(os.path.join(folder, "history.jsonl"), "a") as h:
        h.write(json.dumps(dict(time=round(_t.time(), 3), score=report["score"], passed=report["passed"],
                                testbench=report.get("testbench"), judge=(report["judge"] or {}).get("verdict"),
                                mismatches=len(report["mismatches"]),
                                outcome=("PASS" if report["passed"] else "REVIEW" if report.get("testbench_suspect")
                                         else "TESTBENCH ERROR" if report.get("testbench_error")
                                         else "NOT TESTED" if not report["behaviour_tested"] else "FAIL"),
                                design_sha=hashlib.sha1(design.encode()).hexdigest()[:8],
                                first=(report["result"].splitlines() or [""])[0][:120])) + "\n")
    print(f"  -> {os.path.join(folder, 'report.json')}", flush=True)
    return report

CODE_KEYS = ("verilog", "design", "code", "verilog_code", "module", "verilog_file", "file", "path", "design_file", "out")
SPEC_KEYS = ("spec", "prompt", "request", "description", "question", "task")

TB_KEYS = ("testbench", "tb", "testbench_file", "tb_file", "testbench_code")

def _code_or_file(v, base, must=None):
    """A field holding either Verilog text or a path to a .v file."""
    if not (isinstance(v, str) and v.strip()):
        return ""
    p = v.strip()
    for cand in (p, os.path.join(base, p)):
        if p.endswith(".v") and os.path.exists(cand):
            return open(cand).read()
    return v if (must or "module") in v else ""

def from_request_json(path):
    """The hand-off JSON (from the menu, the prompt step, or a teammate):
         {"prompt": "what it should do", "verilog": "<code>" or "verilog_file": "x.v",
          "testbench": "<code>" or "testbench_file": "tb.v"   <- optional: if given, it is used as-is}
    Returns (verilog_code, spec_text, testbench_or_None)."""
    code, spec = from_prompt_json(path)
    j = json.load(open(path))
    j = j[-1] if isinstance(j, list) else j
    tb = next((c for k in TB_KEYS if (c := _code_or_file(j.get(k), os.path.dirname(path), "module"))), "")
    return code, spec, (tb or None)

def json_to_folder(path):
    """Unpack a hand-off JSON into designs/<module>/ (design.v, spec.txt, and tb.v if the user gave one)."""
    code, spec, tb = from_request_json(path)
    m = vagent.re.search(r"\bmodule\s+(\w+)", code)
    if tb and m and not vagent.re.search(rf"\b{m[1]}\s+(?:#\s*\([^;]*?\)\s*)?\w+\s*\(", tb):
        print(f"  ⚠ the testbench you gave never uses module {m[1]} (it is not a testbench for this design) "
              f"-> ignoring it; the AI will write one", flush=True)
        tb = None
    d = os.path.join("designs", m[1] if m else os.path.splitext(os.path.basename(path))[0])
    os.makedirs(d, exist_ok=True)
    sp, tp, src = (os.path.join(d, f) for f in ("spec.txt", "tb.v", "tb_source.txt"))
    old_spec = open(sp).read().strip() if os.path.exists(sp) else None
    open(os.path.join(d, "design.v"), "w").write(code)
    if spec:
        open(sp, "w").write(spec + "\n")
    so = os.path.join(d, "syntax_only.txt")
    if os.path.exists(so):
        os.remove(so)
    j = json.load(open(path))
    if isinstance(j, dict) and j.get("syntax_only") and not tb:   # the user chose "no testbench"
        open(so, "w").write("user asked for a syntax check only\n")
        for f in (tp, src):
            if os.path.exists(f):
                os.remove(f)
    elif tb:                                   # the user's own testbench: use it as-is
        open(tp, "w").write(tb)
        open(src, "w").write("user\n")
    elif os.path.exists(src):                  # no testbench this time: drop the previous user's one
        for f in (tp, src):
            if os.path.exists(f):
                os.remove(f)
    elif old_spec is not None and spec and old_spec != spec.strip():
        for f in (tp, src):                    # a different circuit under the same name: old testbench is stale
            if os.path.exists(f):
                os.remove(f)
    print(f"[{path}] -> {d}  (prompt: {spec[:60]!r}, testbench: {'user-provided' if tb else 'AI will write one'})",
          flush=True)
    return d

def verify_json(path, **kw):
    d = json_to_folder(path)
    return d, verify(d, **kw)

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
            a.folders.append(json_to_folder(jp))
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
