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
        if tb:   # report the design's OWN errors, not the knock-on errors in the testbench
            c1 = subprocess.run(["iverilog", "-g2012", "-o", f"{d}/sim1", files[0]], capture_output=True, text=True, timeout=30)
            if c1.returncode:
                msgs = (c1.stderr + c1.stdout).replace(d + "/", "").strip()
        src = code.splitlines()
        nums = sorted({int(x) for x in re.findall(r"dut\.v:(\d+)", msgs)})[:3]
        near = sorted({m for n in nums for m in (n - 1, n) if 0 < m <= len(src) and src[m-1].strip()})
        shown = "\n".join(f"  line {m}: {src[m-1].strip()}" for m in near)   # compilers often notice one line late
        return (0.0, "COMPILE ERROR:\n" + msgs[-800:] + (f"\nThe mistake is on or just before:\n{shown}" if shown else ""),
                "FAIL (does not compile)")
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
VOTES = 3     # independent reference models per testbench attempt; two must agree (--votes)
PY_MODEL = True   # golden model written in Python first (--no-python to use only Verilog reference models)

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
  no int/logic/bit, no i++ or +=, no break/continue/return (not supported), write i = i + 1. Inputs are reg, outputs are wire.
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

def signed_names(hdr):
    """Port names declared signed in the header (the testbench must declare them signed too)."""
    out = set()
    for part in hdr[hdr.find("(") + 1: hdr.rfind(")")].split(","):
        if re.search(r"\bsigned\b", part):
            m = re.search(r"(\w+)\s*$", part.strip())
            if m:
                out.add(m[1])
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

def ansi_header(code):
    """Header in ANSI style. Old style 'module m(a, y); input [3:0] a; ...' is rebuilt as
    'module m(input [3:0] a, ...);' so stubs, coverage and templates all work."""
    h = header(code)
    if not h or re.search(r"\b(input|output|inout)\b", h):
        return h
    name = re.match(r"module\s+(\w+)", h)[1]
    order = [p.strip() for p in h[h.index("(") + 1: h.rindex(")")].split(",") if p.strip()]
    decl = {}
    for m in re.finditer(r"\b(input|output|inout)\s+(?:(?:wire|reg)\s+)?(?:signed\s+)?(\[[^\]]+\])?\s*([\w\s,]+?);", code):
        for n in m[3].split(","):
            if n.strip():
                decl[n.strip()] = f"{m[1]} {m[2] + ' ' if m[2] else ''}{n.strip()}"
    if not order or not all(p in decl for p in order):
        return h
    return f"module {name}(" + ", ".join(decl[p] for p in order) + ");"

# ---- template testbench: Python writes the testbench, the AI only writes the reference model ----
REF_SYSTEM = ("You are a Verilog verification engineer. Reply with ONLY one ```verilog code block containing "
              "reference-model statements. No module, no declarations, no explanation.")
REF_PROMPT = """Design spec: {spec}
Design ports: {hdr}
Write ONLY procedural Verilog statements that compute the CORRECT expected outputs from the inputs, exactly as the spec says:
{todo}
Rules: no module, no declarations, no assign keyword, no #delays, no initial/always blocks, no break/continue/return.
You may use the scratch variables t and k2 (integers) and if/case/for.
Verilog width trap: an expression is cut to the width of what you assign it to BEFORE shifts are applied.
For a carry/overflow bit write {{exp_cout, exp_sum}} = a + b + cin; never exp_cout = (a + b + cin) >> 16;
Ports declared signed are signed here too; for unsigned ports use $signed(x) only if the spec says two's complement.
Example for a different design (and4 with y = a AND b):
```verilog
exp_y = a & b;
```"""

def is_comb(hdr, code):
    ps = ports(hdr) or []
    clocked = any(d == "input" and re.match(r"(clk|clock)", n, re.I) for d, _, n in ps)
    return bool(ps) and not clocked and not re.search(r"\b(posedge|negedge)\b", code)

def extract_ref(txt, outs=()):
    """The AI's reference statements. Writes to an output name (valid = ...) are renamed to the
    expected-value register (exp_valid = ...), a mistake small models make a lot."""
    b = _blocks(txt)[-1]
    keep = []
    for l in b.splitlines():
        if re.match(r"\s*(module|endmodule|reg|wire|integer|logic|input|output|`timescale)\b", l):
            continue
        if re.match(r"\s*\$(display|finish|monitor|write)\b", l):
            continue
        l = re.sub(r"^(\s*)assign\s+", r"\1", l)
        # 'initial begin' / 'always @(*) begin' around the answer -> a plain begin...end block
        l = re.sub(r"^(\s*)(?:initial|always\s*@\s*\([^)]*\)|always\s*@\*|always)\s*(begin\b)?", r"\1\2", l)
        l = re.sub(r"@\s*\([^)]*\)\s*;?", "", l)
        l = re.sub(r"#\s*\d+\s*;?", "", l)
        keep.append(l)
    ref = "\n".join(keep).strip()
    for n in outs:
        # every use of an output name means the MODEL's value: valid = ..., {cout, sum} = ..., q[6:0]
        ref = re.sub(rf"(?<![\w.$]){n}\b", f"exp_{n}", ref)
    return ref

def refs_agree(hdr, r1, r2, builder=None):
    """Run two reference models side by side on the same inputs. (True, '') or (False, first difference)."""
    tb = (builder or comb_tb)(hdr, r1, other=r2)
    d = tempfile.mkdtemp()
    open(f"{d}/tb.v", "w").write(tb)
    c = subprocess.run(["iverilog", "-g2012", "-o", f"{d}/sim", f"{d}/tb.v"], capture_output=True, text=True, timeout=30)
    if c.returncode:
        return False, "does not compile"
    try:
        out = subprocess.run(["vvp", f"{d}/sim"], capture_output=True, text=True, timeout=10).stdout
    except subprocess.TimeoutExpired:
        return False, "timeout"
    m = re.search(r"ERRORS=(\d+)", out)
    if m and m[1] == "0":
        return True, ""
    first = next((l for l in out.splitlines() if l.startswith("MISMATCH")), "they disagree")
    return False, first.replace("MISMATCH", "").replace("got", "other model says").strip()

def comb_tb(hdr, ref, other=None):
    """Exhaustive (inputs <= 12 bits) or 4000 random vectors; compares every output with !==.
    With other=..., no design is used: the two reference models are compared with each other."""
    ps = ports(hdr)
    name = re.match(r"module\s+(\w+)", hdr)[1]
    ins = [(w, n) for d, w, n in ps if d == "input"]
    outs = [(w, n) for d, w, n in ps if d == "output"]
    W = sum(w for w, _ in ins)
    exhaustive = W <= 12
    N = 2 ** W if exhaustive else 4000
    L = ["`timescale 1ns/1ps", "module tb;"]
    sg = signed_names(hdr)
    S = lambda n: "signed " if n in sg else ""
    L += [f"  reg {S(n)}[{w-1}:0] {n};" for w, n in ins]
    for w, n in outs:
        L += [f"  wire {S(n)}[{w-1}:0] {n};", f"  reg {S(n)}[{w-1}:0] exp_{n};"]
        if other is not None:
            L.append(f"  reg {S(n)}[{w-1}:0] exp2_{n};")
    L += ["  integer k, errs, t, k2;"]
    if other is None:
        L.append(f"  {name} dut(" + ", ".join(f".{n}({n})" for _, _, n in ps) + ");")
    L += [
          "  initial begin", "    errs = 0;", f"    for (k = 0; k < {N}; k = k + 1) begin"]
    if exhaustive:
        L.append("      {" + ", ".join(n for _, n in ins) + "} = k;")
    else:
        L += [f"      {n} = {{{', '.join(['$random'] * ((w + 31) // 32))}}};" for w, n in ins]
    L.append("      #1;")
    L += ["      " + l for l in ref.splitlines()]
    if other is not None:
        L += ["      " + re.sub(r"\bexp_", "exp2_", l) for l in other.splitlines()]
    fmt = " ".join(f"{n}=%0d" for _, n in ins)
    args = ", ".join(n for _, n in ins)
    for _, n in outs:
        got = f"exp2_{n}" if other is not None else n
        L += [f"      if ({got} !== exp_{n}) begin",
              "        errs = errs + 1;",
              f'        if (errs <= 20) $display("MISMATCH {fmt} expected {n}=%0d got {n}=%0d", {args}, exp_{n}, {got});',
              "      end"]
    L += ["    end", f'    $display("ERRORS=%0d TOTAL=%0d", errs, {N * len(outs)});', "    $finish;", "  end", "endmodule", ""]
    return "\n".join(L)

SEQ_PROMPT = """Design spec: {spec}
Design ports: {hdr}
The testbench runs your statements ONCE AT EVERY RISING EDGE of {clk}, after that edge's inputs are set.
Write ONLY procedural Verilog statements that update the expected outputs exactly as the design should on that edge:
{todo}
- Each exp_ output still holds its value from the previous edge (that IS the current state).
- {rst_text}
- For extra state (history bits, FSM state, counters) use the scratch registers s1, s2, s3, s4 (64-bit, start at 0).
  Compute the outputs from the OLD state first, then update the state.
Rules: no module, no declarations, no assign, no #delays, no @(...), no initial/always blocks, no break.
Example for a different design (8-bit register q with synchronous reset rst and load enable ld, data d):
```verilog
if (rst) exp_q = 0;
else if (ld) exp_q = d;
```"""

def clock_reset(ps):
    """(clock name, reset name or None, reset is active low)."""
    one = [n for d, w, n in ps if d == "input" and w == 1]
    clk = next((n for n in one if re.fullmatch(r"(i_)?(clk|clock)(_i)?", n, re.I)), None)
    rst = next((n for n in one if re.search(r"rst|reset|clr|clear", n, re.I)), None)
    low = bool(rst and re.search(r"(^n|_n$|n$|_b$)", rst, re.I))
    return clk, rst, low

def is_seq(hdr):
    ps = ports(hdr) or []
    return bool(ps) and clock_reset(ps)[0] is not None

def seq_tb(hdr, ref, other=None, cycles=300):
    """Clocked testbench: Python drives clock, reset and random inputs; the AI's statements model
    one rising edge. With other=..., two models are compared with each other (no design)."""
    ps = ports(hdr)
    name = re.match(r"module\s+(\w+)", hdr)[1]
    clk, rst, low = clock_reset(ps)
    sg = signed_names(hdr)
    S = lambda n: "signed " if n in sg else ""
    ins = [(w, n) for d, w, n in ps if d == "input" and n != clk]
    data = [(w, n) for w, n in ins if n != rst]
    outs = [(w, n) for d, w, n in ps if d == "output"]
    on, off = ("1'b0", "1'b1") if low else ("1'b1", "1'b0")
    L = ["`timescale 1ns/1ps", "module tb;", f"  reg {clk} = 0;", f"  always #5 {clk} = ~{clk};"]
    L += [f"  reg {S(n)}[{w-1}:0] {n};" for w, n in ins]
    for w, n in outs:
        L += [f"  wire {S(n)}[{w-1}:0] {n};", f"  reg {S(n)}[{w-1}:0] exp_{n};"]
        if other is not None:
            L.append(f"  reg {S(n)}[{w-1}:0] exp2_{n};")
    L.append("  reg [63:0] s1, s2, s3, s4" + (", s1b, s2b, s3b, s4b;" if other is not None else ";"))
    L.append("  integer k, errs, t, k2;")
    if other is None:
        L.append(f"  {name} dut(" + ", ".join(f".{n}({n})" for _, _, n in ps) + ");")
    init = ["s1 = 0; s2 = 0; s3 = 0; s4 = 0;"] + [f"exp_{n} = 0;" for _, n in outs]
    if other is not None:
        init += ["s1b = 0; s2b = 0; s3b = 0; s4b = 0;"] + [f"exp2_{n} = 0;" for _, n in outs]
    L += ["  initial begin", "    errs = 0;", "    " + " ".join(init),
          f"    for (k = 0; k < {cycles}; k = k + 1) begin"]
    if rst:
        L.append(f"      {rst} = (k < 2 || k == {cycles // 2} || ($random % 29) == 0) ? {on} : {off};")
    L += [f"      {n} = {{{', '.join(['$random'] * ((w + 31) // 32))}}};" for w, n in data]
    L.append(f"      @(posedge {clk});")
    L += ["      " + l for l in ref.splitlines()]
    if other is not None:
        o = re.sub(r"\bexp_", "exp2_", other)
        o = re.sub(r"\bs([1-4])\b", r"s\1b", o)
        L += ["      " + l for l in o.splitlines()]
    L.append("      #1;")
    fmt = " ".join(f"{n}=%0d" for _, n in ins) + " cycle=%0d"
    args = ", ".join([n for _, n in ins] + ["k"])
    for _, n in outs:
        got = f"exp2_{n}" if other is not None else n
        L += [f"      if ({got} !== exp_{n}) begin",
              "        errs = errs + 1;",
              f'        if (errs <= 20) $display("MISMATCH {fmt} expected {n}=%0d got {n}=%0d", {args}, exp_{n}, {got});',
              "      end"]
    L += ["    end", f'    $display("ERRORS=%0d TOTAL=%0d", errs, {cycles * len(outs)});', "    $finish;", "  end",
          "endmodule", ""]
    return "\n".join(L)

# ---- Python golden model: the AI writes the reference in Python, Python computes every expected value,
# ---- and the values are baked into a self-contained Verilog testbench (no Verilog width/sign traps).
import keyword, random as _random

PY_SYSTEM = ("You are a hardware verification engineer. Reply with ONLY one ```python code block containing the "
             "requested function. No explanation, no imports, no printing.")
PY_COMB = """Design spec: {spec}
Design ports: {hdr}
Write a Python function that returns the CORRECT outputs for one set of inputs, exactly as the spec says:

def model({sig}):
    ...
    return {{{ret}}}

- Inputs are Python ints. {sign_note}
- Return every output as an int; it is cut to its bit width automatically (negative values are fine).
- Plain Python: no imports, no printing.
Example for a different design (and4: y = a AND b):
```python
def model(a, b):
    return {{"y": a & b}}
```"""
PY_SEQ = """Design spec: {spec}
Design ports: {hdr}
The design is clocked by {clk}. Write a Python function for ONE rising edge of {clk}:

def step(state, {sig}):
    ...
    return {{{ret}}}

- state is a dict you own (it starts empty, {{}}) - keep registers, counters, history, FSM state in it.
- The inputs are the values present at this edge. Return what the outputs show right AFTER this edge.
- {rst_text}
- Inputs are Python ints. {sign_note} Return ints (cut to bit width automatically).
- Plain Python: no imports, no printing.
Example for a different design (8-bit register q, synchronous reset rst, load enable ld, data d):
```python
def step(state, rst, ld, d):
    if rst:
        state["q"] = 0
    elif ld:
        state["q"] = d
    return {{"q": state.get("q", 0)}}
```"""

_PY_RUNNER = r'''
import json, sys
job = json.load(open(sys.argv[1]))
SAFE = {k: __builtins__.__dict__[k] if hasattr(__builtins__, "__dict__") else __builtins__[k]
        for k in ("range", "len", "abs", "min", "max", "int", "bool", "sum", "any", "all", "enumerate", "zip",
                  "list", "dict", "tuple", "set", "bin", "hex", "pow", "divmod", "sorted", "reversed", "isinstance",
                  "str", "format", "round", "map", "filter", "ValueError", "KeyError", "Exception", "True", "False", "None")
        if (hasattr(__builtins__, "__dict__") and k in __builtins__.__dict__) or (isinstance(__builtins__, dict) and k in __builtins__)}
ns = {"__builtins__": SAFE}
try:
    exec(job["src"], ns)          # one namespace, so tables/helpers defined next to the function are visible
    if job["fn"] not in ns:
        raise NameError("define a function called " + job["fn"])
    fn = ns[job["fn"]]
    state, out = {}, []
    for vec in job["vectors"]:
        args = {}
        for (name, py, w, sg), v in zip(job["ins"], vec):
            args[py] = v - (1 << w) if sg and v >= (1 << (w - 1)) else v
        r = fn(state, **args) if job["fn"] == "step" else fn(**args)
        if not isinstance(r, dict):
            raise ValueError("the function must return a dict of outputs")
        row = []
        for name, w in job["outs"]:
            if name not in r:
                raise KeyError("missing output " + repr(name) + " in the returned dict")
            row.append(int(r[name]) & ((1 << w) - 1))
        out.append(row)
    print(json.dumps({"ok": out}))
except Exception as e:
    print(json.dumps({"error": type(e).__name__ + ": " + str(e)}))
'''

def _pyname(n):
    return n + "_" if keyword.iskeyword(n) else n

def py_vectors(hdr, kind, cycles=300):
    """Input vectors (one list per test, in input-port order; clock excluded)."""
    ps = ports(hdr)
    rng = _random.Random(1)
    if kind == "comb":
        ins = [(w, n) for d, w, n in ps if d == "input"]
        W = sum(w for w, _ in ins)
        if W <= 12:
            vecs = []
            for k in range(2 ** W):
                row, sh = [], W
                for w, _ in ins:
                    sh -= w
                    row.append((k >> sh) & ((1 << w) - 1))
                vecs.append(row)
            return ins, vecs
        return ins, [[rng.getrandbits(w) for w, _ in ins] for _ in range(4000)]
    clk, rst, low = clock_reset(ps)
    ins = [(w, n) for d, w, n in ps if d == "input" and n != clk]
    vecs = []
    for k in range(cycles):
        row = []
        for w, n in ins:
            if n == rst:
                on = (k < 2 or k == cycles // 2 or rng.randrange(29) == 0)
                row.append((0 if on else 1) if low else (1 if on else 0))
            else:
                row.append(rng.getrandbits(w))
        vecs.append(row)
    return ins, vecs

def run_py_model(src, hdr, kind, ins, vecs):
    """Run the AI's Python model on every vector. (expected rows, None) or (None, error text)."""
    sg = signed_names(hdr)
    outs = [(n, w) for d, w, n in ports(hdr) if d == "output"]
    job = dict(src=src, fn="model" if kind == "comb" else "step", vectors=vecs, outs=outs,
               ins=[(n, _pyname(n), w, n in sg) for w, n in ins])
    d = tempfile.mkdtemp()
    open(f"{d}/job.json", "w").write(json.dumps(job))
    open(f"{d}/run.py", "w").write(_PY_RUNNER)
    try:
        r = subprocess.run([sys.executable, "-I", f"{d}/run.py", f"{d}/job.json"],
                           capture_output=True, text=True, timeout=10, cwd=d)
    except subprocess.TimeoutExpired:
        return None, "the Python model never finished (infinite loop?)"
    try:
        res = json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        return None, "the Python model crashed: " + (r.stderr.strip().splitlines() or ["no output"])[-1]
    return (res["ok"], None) if "ok" in res else (None, "the Python model raised " + res["error"])

def extract_py(txt):
    txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.S)
    b = re.findall(r"```(?:python|py)?\s*\n(.*?)```", txt, re.S)
    src = b[-1] if b else txt
    return "\n".join(l for l in src.splitlines() if not re.match(r"\s*(import|from)\s", l))

def vector_tb(hdr, kind, ins, vecs, expected):
    """Self-contained testbench with every input vector and expected output baked in."""
    ps = ports(hdr)
    name = re.match(r"module\s+(\w+)", hdr)[1]
    sg = signed_names(hdr)
    S = lambda n: "signed " if n in sg else ""
    outs = [(w, n) for d, w, n in ps if d == "output"]
    WI, WO, N = sum(w for w, _ in ins), sum(w for w, _ in outs), len(vecs)
    clk = clock_reset(ps)[0] if kind == "seq" else None
    L = ["`timescale 1ns/1ps", "module tb;"]
    if clk:
        L += [f"  reg {clk} = 0;", f"  always #5 {clk} = ~{clk};"]
    L += [f"  reg {S(n)}[{w-1}:0] {n};" for w, n in ins]
    for w, n in outs:
        L += [f"  wire {S(n)}[{w-1}:0] {n};", f"  reg {S(n)}[{w-1}:0] exp_{n};"]
    L += [f"  reg [{WI-1}:0] vin [0:{N-1}];", f"  reg [{WO-1}:0] vexp [0:{N-1}];", "  integer k, errs;",
          f"  {name} dut(" + ", ".join(f".{n}({n})" for _, _, n in ps) + ");", "  initial begin"]
    def pack(vals, widths):
        x = 0
        for v, w in zip(vals, widths):
            x = (x << w) | (v & ((1 << w) - 1))
        return x
    wi, wo = [w for w, _ in ins], [w for w, _ in outs]
    for k in range(N):
        L.append(f"    vin[{k}] = {WI}'h{pack(vecs[k], wi):x}; vexp[{k}] = {WO}'h{pack(expected[k], wo):x};")
    L += ["    errs = 0;", f"    for (k = 0; k < {N}; k = k + 1) begin",
          "      {" + ", ".join(n for _, n in ins) + "} = vin[k];"]
    if clk:
        L.append(f"      @(posedge {clk});")
    L += ["      {" + ", ".join(f"exp_{n}" for _, n in outs) + "} = vexp[k];", "      #1;"]
    fmt = " ".join(f"{n}=%0d" for _, n in ins) + (" cycle=%0d" if clk else "")
    args = ", ".join([n for _, n in ins] + (["k"] if clk else []))
    for _, n in outs:
        L += [f"      if ({n} !== exp_{n}) begin", "        errs = errs + 1;",
              f'        if (errs <= 20) $display("MISMATCH {fmt} expected {n}=%0d got {n}=%0d", {args}, exp_{n}, {n});',
              "      end"]
    L += ["    end", f'    $display("ERRORS=%0d TOTAL=%0d", errs, {N * len(outs)});', "    $finish;", "  end",
          "endmodule", ""]
    return "\n".join(L)

def python_route(spec, hdr, kind, stub, log, tag, tries, hint=""):
    ps = ports(hdr)
    sg = signed_names(hdr)
    ins, vecs = py_vectors(hdr, kind)
    sig = ", ".join(_pyname(n) for _, n in ins)
    ret = ", ".join(f'"{n}": ...' for d, _, n in ps if d == "output")
    sign_note = ("Unsigned ports are 0..2^width-1." + (f" Signed ports ({', '.join(sorted(sg))}) are already "
                 "converted to negative/positive values." if sg else ""))
    if kind == "comb":
        base = PY_COMB.format(spec=spec, hdr=hdr, sig=sig, ret=ret, sign_note=sign_note)
    else:
        clk, rst, low = clock_reset(ps)
        rst_text = (f"Handle reset first: when {rst} is {'0 (active low)' if low else '1 (active high)'}, put the "
                    f"outputs and state back to their reset values (synchronous reset)." if rst else "There is no reset.")
        base = PY_SEQ.format(spec=spec, hdr=hdr, clk=clk, sig=sig, ret=ret, sign_note=sign_note, rst_text=rst_text)
    base += hint
    why = ""
    for i in range(1, tries + 1):
        prompt = base + (f"\n\nYour previous answer was rejected because {why}\nWrite a corrected function." if why else "")
        t0 = time.time()
        try:
            srcs = [extract_py(x) for x in ask_many(prompt, PY_SYSTEM, max(SAMPLES, VOTES))]
        except Exception as ex:
            print(f"  [tb] request failed: {ex}", flush=True)
            return None
        results, why = [], ""
        for src in srcs:
            exp, err = run_py_model(src, hdr, kind, ins, vecs)
            if err:
                why = why or err
            else:
                results.append(exp)
        ok, tb, note = False, None, ""
        if len(results) == 1 and VOTES <= 1:
            ok, chosen = True, results[0]
        else:
            chosen = None
            for a in range(len(results)):
                for b in range(a + 1, len(results)):
                    if results[a] == results[b]:
                        chosen, note = results[a], f"2 of {len(srcs)} Python models agree"
                        break
                if chosen is not None:
                    break
            if chosen is None and len(results) >= 2:
                k = next(j for j in range(len(vecs)) if results[0][j] != results[1][j])
                shown = ", ".join(f"{n}={v}" for (_, n), v in zip(ins, vecs[k]))
                outn = [n for d, _, n in ps if d == "output"]
                why = (f"independent models DISAGREE for inputs {shown}" + (f" (cycle {k})" if kind == "seq" else "") +
                       f": one says {dict(zip(outn, results[0][k]))}, another says {dict(zip(outn, results[1][k]))}. "
                       "Re-read the spec carefully and compute exactly what it says")
            elif chosen is None:
                why = why or "only one answer worked, so there was no second opinion to confirm it"
            ok = chosen is not None
        if ok:
            tb = vector_tb(hdr, kind, ins, vecs, chosen)
            ok, w = validate_tb(tb, stub)
            why = "" if ok else w
        print(f"  [tb] python {kind} model attempt {i}: {'accepted (' + note + ')' if ok else 'rejected - ' + why[:200]} "
              f"({time.time() - t0:.0f}s, {len(results)}/{len(srcs)} ran)", flush=True)
        log.write(json.dumps(dict(problem=tag, kind="testbench", route="python-" + kind, attempt=i,
                                  accepted=ok, reason=why, tb=tb if ok else None)) + "\n")
        log.flush()
        if ok:
            return tb
    print("  [tb] Python model route failed -> trying the Verilog reference template", flush=True)
    return None

# ---- the judge: when a design fails a freshly built testbench, who is wrong, the design or the golden model? ----
JUDGE_SYSTEM = "You are a careful digital design reviewer. Decide which answer follows the spec. Be brief."
JUDGE_PROMPT = """Spec: {spec}
Ports: {hdr}
{history}{where}the inputs are: {inputs}
Two answers disagree about output {out}:
  A) {out} = {a}
  B) {out} = {b}
Work it out from the spec in a few short steps. Then on the LAST line write only the letter A or B."""

def parse_mismatch(line):
    """'MISMATCH rst=0 x=1 cycle=16 expected y=1 got y=0' -> (inputs dict, cycle or None, out, expected, got)."""
    m = re.match(r"MISMATCH (.*?) expected (\w+)=(-?\w+) got \w+=(-?\w+)", line.strip())
    if not m:
        return None
    ins = dict(re.findall(r"(\w+)=(-?\w+)", m[1]))
    cyc = ins.pop("cycle", None)
    return ins, cyc, m[2], m[3], m[4]

def tb_history(tb, hdr, cycle, back=6):
    """For a vector testbench: the inputs of the cycles before 'cycle' (so the judge sees resets and history)."""
    order = re.search(r"\{([^}]*)\} = vin\[k\];", tb)
    if not order or cycle is None:
        return ""
    names = [n.strip() for n in order[1].split(",")]
    widths = {n: w for d, w, n in (ports(hdr) or []) if d == "input"}
    vals = dict((int(k), int(v, 16)) for k, v in re.findall(r"vin\[(\d+)\] = \d+'h([0-9a-f]+);", tb))
    rows = []
    for c in range(max(0, int(cycle) - back), int(cycle) + 1):
        if c not in vals:
            continue
        x, parts, sh = vals[c], [], sum(widths.get(n, 1) for n in names)
        for n in names:
            w = widths.get(n, 1)
            sh -= w
            parts.append(f"{n}={(x >> sh) & ((1 << w) - 1)}")
        rows.append(f"  rising edge {c}: " + ", ".join(parts))
    return ("Inputs at the last rising edges (oldest first; the design starts in its reset state):\n" +
            "\n".join(rows) + "\n") if rows else ""

def judge(spec, hdr, tb, mismatches, votes=3):
    """Returns (verdict, details). verdict: 'design' (golden model looks wrong), 'model' (design is wrong), or None."""
    picked = [p for p in (parse_mismatch(l) for l in mismatches) if p][:3]
    if not picked:
        return None, []
    details, design_wins = [], 0
    for ins, cyc, out, exp, got in picked:
        q = JUDGE_PROMPT.format(spec=spec, hdr=hdr, history=tb_history(tb, hdr, cyc),
                                where=(f"At rising edge {cyc} " if cyc is not None else "When "),
                                inputs=", ".join(f"{k}={v}" for k, v in ins.items()), out=out, a=got, b=exp)
        try:
            answers = ask_many(q, JUDGE_SYSTEM, votes)
        except Exception as ex:
            return None, [f"judge request failed: {ex}"]
        picks = []
        for ans in answers:
            last = [l for l in ans.strip().splitlines() if l.strip()]
            m = re.search(r"\b([AB])\b", last[-1]) if last else None
            picks.append(m[1] if m else "?")
        side = "design" if picks.count("A") > votes // 2 else ("model" if picks.count("B") > votes // 2 else "unsure")
        design_wins += side == "design"
        details.append(f"{', '.join(f'{k}={v}' for k, v in ins.items())}" + (f" @edge {cyc}" if cyc else "") +
                       f": design {out}={got} vs model {out}={exp} -> votes {''.join(picks)} -> {side}")
    verdict = "design" if design_wins * 2 > len(picked) else "model"
    return verdict, details

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

def auto_tb(spec, code, log, tag, tries=3, hint=""):
    """hint: extra warning appended to every prompt (e.g. from the judge: 'your last model was wrong for ...')."""
    hint = f"\n\nIMPORTANT: {hint}" if hint else ""
    hdr = ansi_header(code)
    if not hdr:
        print("  [tb] couldn't find the module header -> syntax check only", flush=True)
        return None
    stub = hdr + "\nendmodule\n"
    why = ""
    kind = "comb" if is_comb(hdr, code) else ("seq" if is_seq(hdr) else None)
    if kind and PY_MODEL:
        tb = python_route(spec, hdr, kind, stub, log, tag, tries, hint)
        if tb:
            return tb
    if kind:
        ps = ports(hdr)
        outs = [n for d, _, n in ps if d == "output"]
        todo = "\n".join(f"- set exp_{n} ({w} bit{'s' if w > 1 else ''})" for d, w, n in ps if d == "output")
        builder = comb_tb if kind == "comb" else seq_tb
        if kind == "comb":
            base = REF_PROMPT.format(spec=spec, hdr=hdr, todo=todo)
        else:
            clk, rst, low = clock_reset(ps)
            rst_text = (f"Handle reset first: when {rst} is {'0 (active low)' if low else '1 (active high)'}, "
                        f"set the outputs and state to their reset values (synchronous reset)."
                        if rst else "There is no reset input.")
            base = SEQ_PROMPT.format(spec=spec, hdr=hdr, clk=clk, todo=todo, rst_text=rst_text)
        base += hint
        for i in range(1, tries + 1):
            prompt = base + (f"\n\nYour previous answer was rejected because {why}\nWrite corrected statements."
                             if why else "")
            t0 = time.time()
            try:
                refs = [extract_ref(x, outs) for x in ask_many(prompt, REF_SYSTEM, max(SAMPLES, VOTES))]
            except Exception as ex:
                print(f"  [tb] request failed: {ex}", flush=True)
                break
            good, why = [], ""
            for r in refs:                                  # 1. each reference model must make a valid testbench
                okr, w = validate_tb(builder(hdr, r), stub)
                if okr:
                    good.append(r)
                else:
                    why = why or w
            ok, tb, note = False, None, ""
            if len(good) == 1 and VOTES <= 1:
                ok, tb = True, builder(hdr, good[0])
            elif len(good) >= 2:                            # 2. consensus: two independent models must agree
                for a in range(len(good)):
                    for b in range(a + 1, len(good)):
                        same, diff = refs_agree(hdr, good[a], good[b], builder)
                        if same:
                            ok, tb, note = True, builder(hdr, good[a]), f"2 of {len(refs)} reference models agree"
                            break
                        why = why if why.startswith("the independent") else (
                            f"the independent reference models DISAGREE ({diff}). Re-read the spec carefully "
                            f"(directions, signedness, bit order, reset) and compute exactly what it says")
                    if ok:
                        break
            elif len(good) == 1:
                why = why or "only one answer was usable, so there was no second opinion to confirm it"
            print(f"  [tb] {kind} template attempt {i}: {'accepted (' + note + ')' if ok else 'rejected - ' + why[:200]} "
                  f"({time.time() - t0:.0f}s, {len(good)}/{len(refs)} usable)", flush=True)
            log.write(json.dumps(dict(problem=tag, kind="testbench", route=kind + "-template", attempt=i,
                                      accepted=ok, reason=why, tb=tb)) + "\n")
            log.flush()
            if ok:
                return tb
        print("  [tb] template route failed -> asking the AI for a whole testbench", flush=True)
        why = ""
    for i in range(1, tries + 1):
        prompt = TB_PROMPT.format(spec=spec, hdr=hdr) + hint
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

def looks_like_verilog(line):
    return bool(re.match(r"\s*(module\b|`timescale|`include|//|/\*)", line))

def read_code(label):
    while True:
        print(f"\n{label}")
        print("  Type a file path, OR paste the code and finish with a line that says END.")
        print("  (Just press Enter to go back.)")
        first = input("> ").rstrip("\n")
        s = first.strip()
        if not s:
            return None
        if os.path.isfile(s):
            return open(s).read()
        if looks_like_verilog(first):
            return read_paste(first)
        print(f"  ✗ '{s[:40]}' is not a file and doesn't look like Verilog (it should start with 'module'). Try again.")

def ask_text(question, min_words=1):
    """Ask until the answer is real text, not a stray menu number."""
    while True:
        a = input(question).strip()
        if not a:
            return ""
        if re.fullmatch(r"[0-9q]", a, re.I) or len(a.split()) < min_words:
            print(f"  ✗ '{a}' looks like a menu choice, not a description. Describe the circuit in words "
                  f"(or press Enter to go back).")
            continue
        return a

def read_tb():
    while True:
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
        if looks_like_verilog(first):
            return read_paste(first)
        print(f"  ✗ '{s[:40]}...' is not a testbench. Press Enter for an automatic one, type none, "
              f"or paste Verilog starting with 'module' or '`timescale'.")

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
    """Front door. Both options end in the SAME hand-off: a JSON with the Verilog, what it should do,
    and (optionally) the user's own testbench. The testbench step reads only that JSON."""
    import prompt as P, verify as V, run as R          # imported here: they import this file too
    os.makedirs("requests", exist_ok=True)
    while True:
        print("\n========== Verilog Agent ==========")
        print(" 1) Describe a circuit -> the AI writes it")
        print(" 2) Paste my own Verilog (+ what it should do, + my testbench if I have one)")
        print(" 3) Run the 9-circuit testbench exam")
        print(" q) Quit")
        ch = input("Choose: ").strip().lower()
        if ch in ("q", "quit", "exit"):
            return
        t0 = time.time()
        if ch == "1":
            spec = ask_text("\nDescribe the circuit in one line (what it does; name and ports if you know them):\n> ",
                            min_words=3)
            if not spec:
                continue
            print("\n[prompt] the AI writes the circuit", flush=True)
            try:
                d = P.generate(argparse.Namespace(request=[spec], code=None, spec=None, designs="designs",
                                                  log=a.log, fix=None))
            except SystemExit as e:
                print(f"  prompt step stopped: {e}")
                continue
            req = dict(prompt=open(os.path.join(d, "spec.txt")).read().strip(),
                       verilog_file=os.path.abspath(os.path.join(d, "design.v")))
            fix = True
        elif ch == "2":
            code = read_code("Your Verilog module:")
            if not code:
                continue
            spec = ask_text("\nWhat should it do? (the testbench checks against this):\n> ", min_words=3)
            if not spec:
                print("  a description is needed to test behaviour")
                continue
            tb = read_tb()
            if tb is None:
                print("  (no testbench: only syntax will be checked)")
            req = dict(prompt=spec, verilog=code)
            if tb not in (None, "auto"):
                req["testbench"] = tb
            if tb is None:
                req["syntax_only"] = True
            fix = input("\nIf it fails, let the AI fix your code? [Y/n] ").strip().lower() not in ("n", "no")
        elif ch == "3":
            import make_tests
            make_tests.run(list(make_tests.T))
            continue
        else:
            print("Pick 1, 2, 3 or q.")
            continue
        m = re.search(r"\bmodule\s+(\w+)", req.get("verilog") or open(req["verilog_file"]).read())
        path = os.path.join("requests", f"{m[1] if m else 'circuit'}.json")
        json.dump(req, open(path, "w"), indent=2)
        print(f"\n[hand-off] {path}", flush=True)
        d = V.json_to_folder(path)
        outcome, history = R.check_and_fix(d, a.rounds, a.log, fix=fix)
        R.summary(d, outcome, history, t0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["basic", "detailed", "both"])
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--samples", type=int, default=1, help="answers per request, generated in parallel; keeps the best")
    ap.add_argument("--votes", type=int, default=3, help="reference models per testbench; 2 must agree (1 = no voting)")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--problems", default="all")
    ap.add_argument("--log", default="attempts.jsonl")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--tbtest", action="store_true", help="measure whether AI testbenches catch the built-in bugs")
    a = ap.parse_args()
    names = list(P) if a.problems == "all" else a.problems.split(",")
    global SAMPLES
    SAMPLES = max(1, a.samples)
    global VOTES
    VOTES = max(1, a.votes)

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
