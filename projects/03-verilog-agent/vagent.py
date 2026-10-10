#!/usr/bin/env python3
"""Verilog debugging agent.
The AI (Qwen on Trainium) writes/fixes Verilog. The CHECKER (this Python + Icarus Verilog)
compiles and simulates it and turns the result into feedback for the next attempt.

  python vagent.py                 -> menu (debug your code / write from a prompt / experiment)
  python vagent.py --selftest      -> prove the checker on the built-in problems
  python vagent.py --mode both     -> experiment: basic vs detailed feedback
  python vagent.py --tbtest        -> experiment: do AI-written testbenches catch the bugs?
"""
import os, re, sys, json, time, argparse, tempfile, subprocess, urllib.request

URL = os.environ.get("VAGENT_URL", "http://localhost:8000/v1/chat/completions")
MODEL = os.environ.get("VAGENT_MODEL", "Qwen/Qwen3-8B")

# ---------------- built-in problems: spec, buggy code, reference fix, testbench ----------------
P = {}

P["adder4"] = dict(
spec="4-bit adder with carry. module adder4(input [3:0] a, input [3:0] b, input cin, output [3:0] sum, output cout); {cout,sum} = a + b + cin.",
buggy="""module adder4(input [3:0] a, input [3:0] b, input cin, output [3:0] sum, output cout);
  assign {cout, sum} = a + b;
endmodule
""",
ref="""module adder4(input [3:0] a, input [3:0] b, input cin, output [3:0] sum, output cout);
  assign {cout, sum} = a + b + cin;
endmodule
""",
tb="""`timescale 1ns/1ps
module tb;
  reg [3:0] a, b; reg cin; wire [3:0] sum; wire cout;
  integer i, j, k, errs; reg [4:0] ex;
  adder4 dut(.a(a), .b(b), .cin(cin), .sum(sum), .cout(cout));
  initial begin
    errs = 0;
    for (i = 0; i < 16; i = i + 1) for (j = 0; j < 16; j = j + 1) for (k = 0; k < 2; k = k + 1) begin
      a = i; b = j; cin = k; #1;
      ex = i + j + k;
      if ({cout, sum} !== ex) begin
        errs = errs + 1;
        if (errs <= 5) $display("MISMATCH a=%0d b=%0d cin=%0d expected sum=%0d cout=%0d got sum=%0d cout=%0d", a, b, cin, ex[3:0], ex[4], sum, cout);
      end
    end
    $display("ERRORS=%0d TOTAL=512", errs);
    $finish;
  end
endmodule
""")

P["counter4"] = dict(
spec="4-bit up counter. module counter4(input clk, input rst, input en, output reg [3:0] q); synchronous active-high reset sets q to 0; when en=1 q increments on posedge clk; wraps 15->0.",
buggy="""module counter4(input clk, input rst, input en, output reg [3:0] q);
  always @(posedge clk) begin
    if (rst) q <= 4'd1;
    else if (en) q <= q + 1;
  end
endmodule
""",
ref="""module counter4(input clk, input rst, input en, output reg [3:0] q);
  always @(posedge clk) begin
    if (rst) q <= 4'd0;
    else if (en) q <= q + 1;
  end
endmodule
""",
tb="""`timescale 1ns/1ps
module tb;
  reg clk = 0, rst = 1, en = 0; wire [3:0] q;
  reg [3:0] model; integer cyc, errs;
  counter4 dut(.clk(clk), .rst(rst), .en(en), .q(q));
  always #5 clk = ~clk;
  initial begin
    errs = 0; model = 0;
    for (cyc = 0; cyc < 40; cyc = cyc + 1) begin
      rst = (cyc < 2) || (cyc == 25);
      en = (cyc % 7 != 3);
      @(posedge clk); #1;
      if (rst) model = 0; else if (en) model = model + 1;
      if (q !== model) begin
        errs = errs + 1;
        if (errs <= 5) $display("MISMATCH cycle=%0d rst=%0d en=%0d expected q=%0d got q=%0d", cyc, rst, en, model, q);
      end
    end
    $display("ERRORS=%0d TOTAL=40", errs);
    $finish;
  end
endmodule
""")

P["alu4"] = dict(
spec="4-bit ALU. module alu4(input [3:0] a, input [3:0] b, input [1:0] op, output reg [3:0] y); op 0: a+b, 1: a-b, 2: a&b, 3: a|b (results truncated to 4 bits).",
buggy="""module alu4(input [3:0] a, input [3:0] b, input [1:0] op, output reg [3:0] y);
  always @(*) begin
    case (op)
      2'b00: y = a + b;
      2'b01: y = b - a;
      2'b10: y = a & b;
      2'b11: y = a | b;
    endcase
  end
endmodule
""",
ref="""module alu4(input [3:0] a, input [3:0] b, input [1:0] op, output reg [3:0] y);
  always @(*) begin
    case (op)
      2'b00: y = a + b;
      2'b01: y = a - b;
      2'b10: y = a & b;
      2'b11: y = a | b;
    endcase
  end
endmodule
""",
tb="""`timescale 1ns/1ps
module tb;
  reg [3:0] a, b; reg [1:0] op; wire [3:0] y; reg [3:0] ex;
  integer i, j, k, errs;
  alu4 dut(.a(a), .b(b), .op(op), .y(y));
  initial begin
    errs = 0;
    for (k = 0; k < 4; k = k + 1) for (i = 0; i < 16; i = i + 1) for (j = 0; j < 16; j = j + 1) begin
      a = i; b = j; op = k; #1;
      case (k) 0: ex = i + j; 1: ex = i - j; 2: ex = i & j; default: ex = i | j; endcase
      if (y !== ex) begin
        errs = errs + 1;
        if (errs <= 5) $display("MISMATCH op=%0d a=%0d b=%0d expected y=%0d got y=%0d", op, a, b, ex, y);
      end
    end
    $display("ERRORS=%0d TOTAL=1024", errs);
    $finish;
  end
endmodule
""")

# ---------------- checker ----------------
NO_TB_NOTE = "(no testbench: only checked that it compiles cleanly - behaviour NOT verified)"

def simulate(code, tb=None):
    """The checker. Returns (score 0..1, detailed feedback, basic feedback).
    With a testbench: compile + simulate. The testbench should print MISMATCH lines and
    'ERRORS=<n> TOTAL=<m>'; otherwise any line containing FAIL/MISMATCH/ERROR counts as a failure.
    Without a testbench: compile with all warnings on (syntax / lint check only)."""
    d = tempfile.mkdtemp()
    files = [f"{d}/dut.v"]
    open(files[0], "w").write(code)
    if tb:
        files.append(f"{d}/tb.v")
        open(files[1], "w").write(tb)
    cmd = ["iverilog", "-g2012", "-o", f"{d}/sim"] + ([] if tb else ["-Wall"]) + files
    c = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    msgs = (c.stderr + c.stdout).replace(d + "/", "").strip()
    if c.returncode:
        return 0.0, "COMPILE ERROR:\n" + msgs[-800:], "FAIL (does not compile)"
    if not tb:
        if msgs:
            return 0.5, "Compiles, but with warnings:\n" + msgs[-800:], "FAIL (warnings)"
        return 1.0, "PASS " + NO_TB_NOTE, "PASS"
    try:
        s = subprocess.run(["vvp", f"{d}/sim"], capture_output=True, text=True, timeout=10)
    except subprocess.TimeoutExpired:
        return 0.0, "SIMULATION TIMEOUT (likely an infinite loop, or the testbench never reaches $finish)", "FAIL"
    out = s.stdout
    m = re.search(r"ERRORS=(\d+) TOTAL=(\d+)", out)
    if m:
        e, t = int(m[1]), int(m[2])
        if e == 0:
            return 1.0, "PASS", "PASS"
        mism = "\n".join(l for l in out.splitlines() if l.startswith("MISMATCH"))
        return 1 - e / t, f"{e} of {t} checks failed. First mismatches:\n{mism}", "FAIL"
    bad = [l for l in out.splitlines() if re.search(r"FAIL|MISMATCH|ERROR", l, re.I)]
    if bad:
        return 0.0, "Testbench reported failures:\n" + "\n".join(bad[:8]), "FAIL"
    return 1.0, "PASS (testbench printed no failures)", "PASS"

def check(name, code):
    return simulate(code, P[name]["tb"])

# ---------------- model ----------------
SYSTEM = "You are a Verilog engineer. Reply with ONLY the complete module in one ```verilog code block. No explanation."
TB_SYSTEM = "You are a Verilog verification engineer. Reply with ONLY one ```verilog code block containing a self-checking testbench module named tb. No explanation."

SAMPLES = 1   # answers per request (--samples); the server generates them in parallel

def ask_many(user, system=SYSTEM, n=None):
    """One request, n different answers (temperature 0.7 makes them differ)."""
    body = {"model": MODEL, "max_tokens": 2000, "temperature": 0.7, "n": n or SAMPLES,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return [c["message"]["content"] or "" for c in json.load(r)["choices"]]

def ask(user, system=SYSTEM):
    return ask_many(user, system, 1)[0]

MOD = re.compile(r"\bmodule\s+(\w+).*?\bendmodule\b", re.S)

def _blocks(txt):
    txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.S)
    b = re.findall(r"```(?:verilog|systemverilog|sv|v)?\s*\n(.*?)```", txt, re.S)
    return [re.sub(r"^\s*`include.*$", "", x, flags=re.M) for x in (b or [txt])]

def extract(txt):
    """Design code only: every module except a testbench called tb."""
    for b in reversed(_blocks(txt)):
        mods = [m[0] for m in MOD.finditer(b) if m[1] != "tb"]
        if mods:
            return "\n\n".join(mods) + "\n"
    return txt

def extract_tb(txt):
    """Testbench only: module tb (+ `timescale). Drops any copy of the design the AI pasted in,
    which would otherwise be defined twice and fail to compile."""
    for b in reversed(_blocks(txt)):
        m = re.search(r"\bmodule\s+tb\b.*?\bendmodule\b", b, re.S)
        if m:
            ts = re.search(r"`timescale[^\n]*", b)
            return (ts[0] + "\n" if ts else "") + m[0] + "\n"
    return txt

# ---------------- automatic testbench ----------------
TB_PROMPT = """Write a self-checking Verilog testbench (module tb) for this design.
Spec: {spec}
Design header (instantiate it exactly as written): {hdr}
Rules:
- Compute the EXPECTED output yourself from the spec (a simple reference model in the testbench).
- Test many input combinations (all of them if the inputs are small). For clocked designs, drive clk and reset.
- Compare with !== so undriven/x outputs count as wrong.
- On each wrong output print one line: MISMATCH <inputs> expected=<value> got=<value>
- At the very end print exactly: ERRORS=<n> TOTAL=<m>   and then call $finish.
- Plain Verilog-2005 only: declare EVERY variable at the top of module tb (never inside begin/end or for),
  no int/logic/bit, no i++ or +=, write i = i + 1. Inputs are reg, outputs are wire.
- Clocked designs: always #5 clk = ~clk;  then for each step: set inputs, @(posedge clk); #1; check.

Follow this structure exactly (example for a different design, an AND gate):
```verilog
`timescale 1ns/1ps
module tb;
  reg [3:0] a, b;
  wire [3:0] y;
  reg [3:0] expected;
  integer i, j, errs;
  and4 dut(.a(a), .b(b), .y(y));
  initial begin
    errs = 0;
    for (i = 0; i < 16; i = i + 1)
      for (j = 0; j < 16; j = j + 1) begin
        a = i; b = j; #1;
        expected = i & j;
        if (y !== expected) begin
          errs = errs + 1;
          $display("MISMATCH a=%0d b=%0d expected=%0d got=%0d", a, b, expected, y);
        end
      end
    $display("ERRORS=%0d TOTAL=%0d", errs, 256);
    $finish;
  end
endmodule
```"""

def header(code):
    """Pull 'module name(...ports...);' out of the code, even if the body is broken."""
    m = re.search(r"module\s+\w+\s*(?:#\s*\([^;]*?\)\s*)?\([^)]*\)", code, re.S)
    return (m[0] + ";") if m else ""

def ports(hdr):
    """[(direction, width, name)] from an ANSI header, or None if it can't be parsed."""
    try:
        inner = hdr[hdr.index("(") + 1: hdr.rindex(")")]
    except ValueError:
        return None
    out, d, w = [], None, 1
    for part in inner.split(","):
        p = part.strip()
        m = re.match(r"(input|output|inout)\b\s*(?:wire|reg|logic)?\s*(?:signed\s*)?"
                     r"(\[\s*(\d+)\s*:\s*(\d+)\s*\])?\s*(\w+)$", p)
        if m:
            d, w, name = m[1], (abs(int(m[3]) - int(m[4])) + 1 if m[2] else 1), m[5]
        elif re.match(r"\w+$", p) and d:
            name = p
        else:
            return None
        out.append((d, w, name))
    return out

def coverage_stub(hdr):
    """Empty design that records every value each input bit was driven to."""
    ps = ports(hdr)
    if not ps:
        return None
    L = [hdr]
    for d, w, n in ps:
        if d != "input":
            continue
        L.append(f"  reg [{w-1}:0] cv0_{n} = 0, cv1_{n} = 0;")
        L.append(f"  always @({n}) if (^{n} !== 1'bx) begin cv1_{n} = cv1_{n} | {n}; cv0_{n} = cv0_{n} | ~{n}; end")
        L.append(f'  final $display("COVER {n} %0d %b %b", {w}, cv0_{n}, cv1_{n});')
    return "\n".join(L) + "\nendmodule\n"

def uncovered(out):
    """From COVER lines: which input bits never became 0 or never became 1."""
    miss = []
    for n, w, s0, s1 in re.findall(r"COVER (\w+) (\d+) ([01xz]+) ([01xz]+)", out):
        w = int(w)
        for val, seen in (("1", s1), ("0", s0)):
            bits = [w - 1 - i for i, c in enumerate(seen) if c != "1"]
            if bits:
                miss.append(f"{n} to {val}" if w == 1 else f"bit(s) {sorted(bits)} of {n} to {val}")
    return miss

def validate_tb(tb, stub):
    """A testbench is only trusted if it compiles, finishes, reports a result,
    and FAILS an empty dummy design (proves it really checks the outputs)."""
    if not re.search(r"module\s+tb\b", tb):
        return False, "the testbench module must be named tb"
    d = tempfile.mkdtemp()
    cov = coverage_stub(header(stub) or "")
    open(f"{d}/stub.v", "w").write(cov or stub)
    open(f"{d}/tb.v", "w").write(tb)
    c = subprocess.run(["iverilog", "-g2012", "-o", f"{d}/sim", f"{d}/stub.v", f"{d}/tb.v"],
                       capture_output=True, text=True, timeout=30)
    if c.returncode:
        errs = [l for l in (c.stderr + c.stdout).replace(d + "/", "").splitlines() if l.strip()]
        src = tb.splitlines()
        bad = sorted({int(x) for x in re.findall(r"tb\.v:(\d+)", " ".join(errs))})[:3]
        shown = "; ".join(f"line {n}: `{src[n-1].strip()}`" for n in bad if 0 < n <= len(src))
        return False, ("it does not compile: " + " | ".join(errs[:3])[:300]
                       + (f"  -> broken code: {shown}" if shown else ""))
    try:
        s = subprocess.run(["vvp", f"{d}/sim"], capture_output=True, text=True, timeout=10)
    except subprocess.TimeoutExpired:
        return False, "the simulation never finishes ($finish missing, or an infinite loop)"
    m = re.search(r"ERRORS=(\d+) TOTAL=(\d+)", s.stdout)
    if not m:
        return False, "it never prints the final line ERRORS=<n> TOTAL=<m>"
    e, t = int(m[1]), int(m[2])
    if t == 0:
        return False, "TOTAL is 0, so it tests nothing"
    if e == 0:
        return False, ("it PASSES an empty design whose outputs are never driven, so it is not really "
                       "checking outputs. Compare every output with !== against an expected value")
    miss = uncovered(s.stdout)
    if miss:
        return False, ("it never drives " + ", ".join(miss[:6]) + ". Every input must be tested at every "
                       "value (loop over ALL values of small inputs, e.g. cin = 0 and cin = 1)")
    return True, ""

def auto_tb(spec, code, log, tag, tries=3):
    hdr = header(code)
    if not hdr:
        print("  [tb] couldn't find the module header -> syntax check only", flush=True)
        return None
    stub = hdr + "\nendmodule\n"
    why = ""
    for i in range(1, tries + 1):
        prompt = TB_PROMPT.format(spec=spec, hdr=hdr)
        if why:
            prompt += f"\n\nYour previous testbench was rejected because {why}\nWrite a corrected one."
        t0 = time.time()
        try:
            cands = [extract_tb(x) for x in ask_many(prompt, TB_SYSTEM)]
        except Exception as ex:
            print(f"  [tb] request failed: {ex}", flush=True)
            return None
        for tb in cands:
            ok, why = validate_tb(tb, stub)
            if ok:
                break
        print(f"  [tb] attempt {i}: {'accepted' if ok else 'rejected - ' + why[:200]} "
              f"({time.time() - t0:.0f}s, {len(cands)} candidate(s))", flush=True)
        log.write(json.dumps(dict(problem=tag, kind="testbench", attempt=i, accepted=ok,
                                  reason=why, tb=tb)) + "\n")
        log.flush()
        if ok:
            return tb
    print("  [tb] no valid testbench after retries -> syntax check only", flush=True)
    return None

# ---------------- agent loop ----------------
def loop(spec, code, tb, rounds, mode, log, tag, rep=1, verbose=False):
    """code=None means: write from scratch. Returns (round solved or None, final code)."""
    if code is not None:
        score, det, basic = simulate(code, tb)
        print(f"  [{mode}] {tag} rep{rep} round0 (starting code): score={score:.3f} {det.splitlines()[0]}", flush=True)
        if verbose and score < 1.0:
            print("    " + "\n    ".join(det.splitlines()[1:6]), flush=True)
        if score == 1.0:
            return 0, code
    for rnd in range(1, rounds + 1):
        if code is None:
            prompt = f"Write a Verilog module.\nSpec: {spec}"
        else:
            fb = det if mode == "detailed" else basic
            prompt = (f"Spec: {spec}\n\nCurrent code:\n```verilog\n{code}```\n\n"
                      f"Checker result: {fb}\n\nFix the module so it meets the spec and passes the checker.")
        t0 = time.time()
        try:
            replies = ask_many(prompt)
        except Exception as ex:
            replies = [f"(request failed: {ex})"]
        best = None
        for rep_txt in replies:
            cand = extract(rep_txt)
            res = simulate(cand, tb)
            if best is None or res[0] > best[0][0]:
                best = (res, cand)
            if res[0] == 1.0:
                break
        (score, det, basic), code = best
        dt = time.time() - t0
        print(f"  [{mode}] {tag} rep{rep} round{rnd}: score={score:.3f} ({dt:.0f}s) {det.splitlines()[0]}", flush=True)
        if verbose and score < 1.0:
            print("    " + "\n    ".join(det.splitlines()[1:6]), flush=True)
        log.write(json.dumps(dict(problem=tag, mode=mode, rep=rep, round=rnd, score=score,
                                  feedback=det, code=code, seconds=round(dt, 1))) + "\n")
        log.flush()
        if score == 1.0:
            return rnd, code
    return None, code

# ---------------- experiments ----------------
def experiment(names, modes, rounds, repeat, log):
    """Built-in buggy modules with hand-written testbenches: basic vs detailed feedback."""
    res = {}
    for rep in range(1, repeat + 1):
        for mode in modes:
            for n in names:
                print(f"=== {n} | {mode} | rep {rep}/{repeat} ===", flush=True)
                r, _ = loop(P[n]["spec"], P[n]["buggy"], P[n]["tb"], rounds, mode, log, n, rep)
                res.setdefault((mode, n), []).append(r)
    print("\n===== SUMMARY =====")
    for mode in modes:
        tot = 0
        for n in names:
            r = res[(mode, n)]
            solved = [x for x in r if x]
            tot += len(solved)
            avg = f"{sum(solved)/len(solved):.1f}" if solved else "-"
            print(f"{mode:9s} {n:9s} solved {len(solved)}/{len(r)}  avg rounds {avg}  runs={r}")
        print(f"{mode:9s} TOTAL     solved {tot}/{len(names)*repeat}")

def tbtest(names, repeat, log):
    """Can the AI-written testbench be trusted? It must fail the buggy code and pass the correct code."""
    tally = {}
    for rep in range(1, repeat + 1):
        for n in names:
            print(f"=== AI testbench for {n} | rep {rep}/{repeat} ===", flush=True)
            tb = auto_tb(P[n]["spec"], P[n]["ref"], log, n)
            if not tb:
                v = "NO VALID TESTBENCH"
            else:
                sb, _, _ = simulate(P[n]["buggy"], tb)
                sr, _, _ = simulate(P[n]["ref"], tb)
                v = ("CATCHES BUG" if sb < 1 and sr == 1 else
                     "REJECTS CORRECT CODE" if sr < 1 else "MISSES BUG")
                print(f"  buggy={sb:.3f} correct={sr:.3f}", flush=True)
            print(f"  -> {v}", flush=True)
            tally.setdefault(n, []).append(v)
    print("\n===== AI TESTBENCH SUMMARY =====")
    for n, vs in tally.items():
        print(f"{n:9s} catches bug {vs.count('CATCHES BUG')}/{len(vs)}   {vs}")

# ---------------- interactive menu ----------------
def read_paste(first):
    lines = [first]
    while True:
        try:
            l = input()
        except EOFError:
            break
        if l.strip() == "END":
            break
        lines.append(l)
    return "\n".join(lines) + "\n"

def read_code(label):
    print(f"\n{label}")
    print("  Type a file path, OR paste the code and finish with a line that says END.")
    first = input("> ").rstrip("\n")
    if not first.strip():
        return None
    if os.path.isfile(first.strip()):
        return open(first.strip()).read()
    return read_paste(first)

def read_tb():
    print("\nTestbench:")
    print("  Enter = AI writes one automatically (recommended)")
    print("  none  = only check that it compiles")
    print("  or type a file path / paste your own, ending with a line END")
    first = input("> ").rstrip("\n")
    s = first.strip()
    if not s:
        return "auto"
    if s.lower() == "none":
        return None
    if os.path.isfile(s):
        return open(s).read()
    return read_paste(first)

def finish(r, code, tb, n):
    path = f"result_{n}.v"
    open(path, "w").write(code)
    if tb:
        open(f"result_{n}_tb.v", "w").write(tb)
    if r is None:
        print(f"\n✗ Not solved. Last attempt saved to {path}")
        if tb:
            print(f"  The testbench could be wrong too - check result_{n}_tb.v")
    elif not tb:
        print(f"\n⚠ It compiles, but its behaviour was NOT tested (no valid testbench). Saved to {path}")
    else:
        print(f"\n✓ Passed {'(no fix needed)' if r == 0 else f'after {r} round(s)'}. Saved to {path}"
              f" (testbench: result_{n}_tb.v)")
    print("-" * 40 + "\n" + code.strip() + "\n" + "-" * 40)

def menu(a):
    n = 0
    with open(a.log, "a") as log:
        while True:
            print("\n========== Verilog Agent ==========")
            print(" 1) Debug my Verilog    (you give code -> AI writes a testbench -> checker runs it -> AI fixes)")
            print(" 2) Write from a prompt (you describe it -> AI writes code + testbench -> checker -> AI fixes)")
            print(" 3) Experiment: basic vs detailed feedback (built-in buggy modules)")
            print(" 4) Experiment: can the AI-written testbench catch real bugs?")
            print(" q) Quit")
            ch = input("Choose: ").strip().lower()
            if ch in ("q", "quit", "exit"):
                break
            if ch in ("1", "2"):
                code = None
                if ch == "1":
                    code = read_code("Your Verilog module:")
                    if not code:
                        continue
                    spec = input("\nWhat should it do? (needed for the auto testbench):\n> ").strip()
                else:
                    spec = input("\nDescribe the module (include the module name and ports):\n> ").strip()
                    if not spec:
                        continue
                tb = read_tb()
                n += 1
                tag = f"user{n}"
                if tb == "auto":
                    if not spec:
                        print("  No description given -> can't write a testbench; syntax check only")
                        tb = None
                    else:
                        if code is None:
                            print("  [ai] writing the first version of the module...", flush=True)
                            try:
                                code = extract(ask(f"Write a Verilog module.\nSpec: {spec}"))
                            except Exception as ex:
                                print(f"  request failed: {ex}")
                                continue
                        print("  [ai] writing a testbench from the spec...", flush=True)
                        tb = auto_tb(spec, code, log, tag)
                if not tb:
                    print("  " + NO_TB_NOTE)
                r, code = loop(spec or "Fix every bug so the module compiles cleanly and behaves correctly.",
                               code, tb, a.rounds, "detailed", log, tag, verbose=True)
                finish(r, code, tb, n)
            elif ch == "3":
                experiment(list(P), ["basic", "detailed"], a.rounds, a.repeat, log)
            elif ch == "4":
                tbtest(list(P), a.repeat, log)
            else:
                print("Pick 1, 2, 3, 4 or q.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["basic", "detailed", "both"])
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--samples", type=int, default=1, help="answers per request, generated in parallel; keeps the best")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--problems", default="all")
    ap.add_argument("--log", default="attempts.jsonl")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--tbtest", action="store_true", help="measure whether AI testbenches catch the built-in bugs")
    a = ap.parse_args()
    names = list(P) if a.problems == "all" else a.problems.split(",")
    global SAMPLES
    SAMPLES = max(1, a.samples)

    if a.selftest:  # prove the checker: buggy must fail, reference must pass
        ok = True
        for n in names:
            sb, db, _ = check(n, P[n]["buggy"])
            sr, dr, _ = check(n, P[n]["ref"])
            good = sb < 1.0 and sr == 1.0
            ok &= good
            print(f"{n}: buggy={sb:.3f} ref={sr:.3f} {'OK' if good else 'BROKEN'}\n    {db.splitlines()[0]}")
        print("SELFTEST", "PASSED" if ok else "FAILED")
        return

    if a.tbtest:
        with open(a.log, "a") as log:
            tbtest(names, a.repeat, log)
        return

    if a.mode is None:
        return menu(a)

    modes = ["basic", "detailed"] if a.mode == "both" else [a.mode]
    with open(a.log, "a") as log:
        experiment(names, modes, a.rounds, a.repeat, log)

if __name__ == "__main__":
    main()
