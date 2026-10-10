"""The checker: four deterministic layers, no model anywhere.

DESIGN.md 6.2. Each layer stops at the first failure and the result says how far the candidate got:

    L0 static    pull the code out of the reply; module TopModule; every port present
    L1 compile   iverilog, candidate + VerilogEval testbench + reference
    L2 simulate  vvp; the testbench's own Mismatches / Hint lines, and wave.vcd for the inputs at
                 the first mismatch
    L3 formal    combinational only, and only when L2 fails (or for the audit): a yosys equivalence
                 miter against RefModule, whose counterexample is a concrete input that breaks it

    python checker.py --selftest                       # six planted bugs; must all be caught
    python checker.py --check Prob050_kmap1 cand.sv    # grade one file
    python checker.py --refs eval/dev.txt              # every reference must score 1.0
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field

import problems as P

IVERILOG = os.environ.get("IVERILOG", "iverilog")
VVP = os.environ.get("VVP", "vvp")
YOSYS = os.environ.get("YOSYS", "yosys")
TIMEOUT_S = 60
PERIOD_PS = 10          # clk toggles every 5 ps in every VerilogEval testbench


@dataclass
class CheckResult:
    layer_reached: int = -1          # highest layer completed: 0,1,2,3 ; -1 = no code
    passed: bool = False             # True iff L2 reports 0 mismatches
    score: float = 0.0
    code: str = ""                   # the extracted candidate
    compile_error: str | None = None # first error line from iverilog, verbatim
    mismatches: int | None = None
    samples: int | None = None
    per_output: dict = field(default_factory=dict)   # {"q": {"mismatches": 12, "first_time_ps": 210}}
    first_mismatch: dict | None = None  # {"output", "time_ps", "cycle", "inputs", "dut"}
    tb_hints: list = field(default_factory=list)     # every other testbench "Hint:" line
    formal: str = "not_run"          # equivalent | not_equivalent | unsupported | not_run
    counterexample: dict | None = None  # {"inputs": {...}, "gold": {...}, "gate": {...}}  (comb)
    raw: str = ""                    # raw tool text, for translator mode "raw"
    elapsed_s: float = 0.0
    l0_error: str | None = None      # why L0 failed: "no_code" | "no_topmodule" | "missing_port:<name>"
    timed_out: str | None = None     # the layer whose tool ran past TIMEOUT_S, if any

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------- L0: extraction

FENCE = re.compile(r"```[ \t]*([A-Za-z]*)[^\n]*\n(.*?)(?:```|\Z)", re.S)
CODE_TAGS = {"verilog", "systemverilog", "sv", ""}


def extract_code(reply: str) -> str:
    """The longest verilog-ish fenced block, else `module` ... last `endmodule`, else ''.

    An unterminated fence still counts -- a reply cut off at max_tokens leaves one open, and the
    compiler should be the thing that says the module is incomplete, not the extractor.
    """
    text = re.sub(r"<think>.*?</think>", "", reply or "", flags=re.S)
    blocks = [body for tag, body in FENCE.findall(text) if tag.lower() in CODE_TAGS]
    blocks = [b for b in blocks if "module" in b] or blocks
    if blocks:
        code = max(blocks, key=len)
    else:
        start = re.search(r"^\s*module\b", text, re.M)
        if not start:
            return ""
        end = text.rfind("endmodule")
        code = text[start.start(): end + len("endmodule")] if end > start.start() else text[start.start():]
    code = "\n".join(ln for ln in code.splitlines() if not ln.strip().startswith("```"))
    return code.strip() + "\n" if code.strip() else ""


def topmodule_text(code: str) -> str | None:
    m = re.search(r"\bmodule\s+TopModule\b", code)
    if not m:
        return None
    end = code.find("endmodule", m.end())
    return code[m.start(): end if end >= 0 else len(code)]


def l0(problem, reply: str, res: CheckResult) -> bool:
    res.code = extract_code(reply)
    if not res.code:
        res.l0_error, res.raw = "no_code", "No Verilog code was found in the reply."
        return False
    top = topmodule_text(res.code)
    if top is None:
        res.l0_error, res.raw = "no_topmodule", "The code does not define `module TopModule`."
        return False
    for _, name, _ in problem.ports:
        if not re.search(rf"\b{re.escape(name)}\b", top):
            res.l0_error = f"missing_port:{name}"
            res.raw = f"module TopModule has no port named `{name}`."
            return False
    res.layer_reached = 0
    res.score = 0.1
    return True


# ---------------------------------------------------------------- tools

def run(cmd, cwd, timeout=TIMEOUT_S):
    """(returncode, combined output, timed_out). Never raises for a tool failure."""
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout,
                           errors="replace")
        return p.returncode, (p.stdout or "") + (p.stderr or ""), False
    except subprocess.TimeoutExpired as e:
        out = e.stdout or ""
        out = out.decode(errors="replace") if isinstance(out, bytes) else out
        return -1, out, True
    except FileNotFoundError:
        raise SystemExit(f"`{cmd[0]}` is not installed. Run ./setup.sh first.")


def tidy_paths(text: str, problem) -> str:
    """Testbench and reference paths shortened to their file names, so messages stay short."""
    for path in (problem.test_path, problem.ref_path):
        text = text.replace(path, os.path.basename(path))
    return text


# ---------------------------------------------------------------- L1: compile

ERROR_LINE = re.compile(r"^[^\s:][^:\n]*:\d+:\s*(?:error|syntax error|sorry)", re.I | re.M)


def first_error(output: str) -> str:
    for line in output.splitlines():
        if ERROR_LINE.match(line):
            return line.strip()
    for line in output.splitlines():
        if "error" in line.lower() or "give up" in line.lower():
            return line.strip()
    lines = [ln.strip() for ln in output.splitlines() if ln.strip() and "warning" not in ln.lower()]
    return lines[0] if lines else "iverilog failed with no message"


def l1(problem, workdir: str, res: CheckResult) -> bool:
    cmd = [IVERILOG, "-Wall", "-Winfloop", "-Wno-timescale", "-g2012", "-s", "tb", "-o", "t.vvp",
           "cand.sv", problem.test_path, problem.ref_path]
    rc, out, timed_out = run(cmd, workdir)
    out = tidy_paths(out, problem)
    if timed_out:
        res.timed_out, res.raw = "compile", "iverilog did not finish within 60 s."
        return False
    if rc != 0 or not os.path.exists(os.path.join(workdir, "t.vvp")):
        res.compile_error = first_error(out)
        res.raw = out.strip()
        return False
    res.layer_reached = 1
    res.score += 0.2
    return True


# ---------------------------------------------------------------- L2: simulate

MISMATCHES = re.compile(r"^Mismatches:\s*(\d+)\s+in\s+(\d+)\s+samples", re.M)
OUT_BAD = re.compile(r"^Hint: Output '(\w+)' has (\d+) mismatches\. First mismatch occurred at time (\d+)\.", re.M)
OUT_OK = re.compile(r"^Hint: Output '(\w+)' has no mismatches\.", re.M)


def parse_sim(output: str) -> dict:
    """Everything the testbench prints, as data. Pure, so it can be tested without iverilog."""
    sim = dict(mismatches=None, samples=None, per_output={}, tb_hints=[])
    m = MISMATCHES.search(output)
    if m:
        sim["mismatches"], sim["samples"] = int(m.group(1)), int(m.group(2))
    for name, n, t in OUT_BAD.findall(output):
        sim["per_output"][name] = {"mismatches": int(n), "first_time_ps": int(t)}
    for name in OUT_OK.findall(output):
        sim["per_output"][name] = {"mismatches": 0, "first_time_ps": None}
    for line in output.splitlines():
        s = line.strip()
        if not s.startswith("Hint:") or OUT_BAD.match(s) or OUT_OK.match(s):
            continue
        if s.startswith("Hint: Total mismatched samples") or s == "Hint:":
            continue
        sim["tb_hints"].append(s)
    if re.search(r"^TIMEOUT\s*$", output, re.M):
        sim["tb_hints"].append("TIMEOUT")
    return sim


def l2(problem, workdir: str, res: CheckResult) -> bool:
    rc, out, timed_out = run([VVP, "-n", "t.vvp"], workdir)
    out = tidy_paths(out, problem)
    res.raw = "\n".join(ln for ln in out.splitlines() if not ln.startswith("VCD info:")).strip()
    if timed_out:
        res.timed_out = "simulate"
        res.raw = (res.raw + "\n" if res.raw else "") + "vvp did not finish within 60 s."
        return False
    sim = parse_sim(out)
    res.per_output, res.tb_hints = sim["per_output"], sim["tb_hints"]
    if sim["mismatches"] is None or not sim["samples"]:
        return False
    res.mismatches, res.samples = sim["mismatches"], sim["samples"]
    res.layer_reached = 2
    res.score += 0.2 + 0.5 * (1 - res.mismatches / res.samples)
    res.passed = res.mismatches == 0
    if res.passed:
        res.score = 1.0
    else:
        res.first_mismatch = first_mismatch(problem, res.per_output, os.path.join(workdir, "wave.vcd"))
    res.score = round(res.score, 4)
    return True


# ---------------------------------------------------------------- VCD

def read_vcd(path: str, scope: str = "tb") -> dict:
    """{signal name: [(time, value-string), ...]} for the signals declared directly in `scope`.

    Deliberately minimal (DESIGN 6.2 says no dependency): $var declarations, #time markers,
    scalar `0!` and vector `b0101 !` changes. Real values are skipped.
    """
    with open(path, errors="replace") as f:
        text = f.read()
    head, _, body = text.partition("$enddefinitions")
    ids, stack = {}, []
    for tok in re.finditer(r"\$(scope|upscope|var)\b(.*?)\$end", head, re.S):
        kind, rest = tok.group(1), tok.group(2).split()
        if kind == "scope":
            stack.append(rest[1] if len(rest) > 1 else "")
        elif kind == "upscope":
            stack and stack.pop()
        elif len(rest) >= 4 and stack and stack[-1] == scope:
            ids.setdefault(rest[2], []).append(rest[3])
    changes = {name: [] for names in ids.values() for name in names}
    t = 0
    toks = body.split()
    i = 0
    while i < len(toks):
        tk = toks[i]
        if tk.startswith("#"):
            t = int(tk[1:]) if tk[1:].isdigit() else t
        elif tk[0] in "bB" and i + 1 < len(toks):
            for name in ids.get(toks[i + 1], ()):
                changes[name].append((t, tk[1:].lower()))
            i += 1
        elif tk[0] in "rR":
            i += 1
        elif tk[0] in "01xXzZ" and len(tk) > 1:
            for name in ids.get(tk[1:], ()):
                changes[name].append((t, tk[0].lower()))
        i += 1
    return changes


def value_at(changes: list, t: int, strictly_before: bool):
    v = None
    for when, val in changes:
        if when < t or (when == t and not strictly_before):
            v = val
        else:
            break
    return v


def literal(bits: str | None, width: int) -> str:
    """A value the model can read: 1 bit as 0/1, narrow vectors in binary, wide ones in hex."""
    if bits is None:
        return "x"
    bits = bits.rjust(width, "0" if bits[0] in "01" else bits[0])[-width:] if width else bits
    if width <= 1:
        return bits
    if width <= 8 or any(c not in "01" for c in bits):
        return f"{width}'b{bits}"
    return f"{width}'h{int(bits, 2):0{(width + 3) // 4}x}"


def first_mismatch(problem, per_output: dict, vcd_path: str) -> dict | None:
    bad = [(v["first_time_ps"], name) for name, v in per_output.items()
           if v.get("mismatches") and v.get("first_time_ps") is not None]
    if not bad:
        return None
    time_ps, out = min(bad)
    fm = {"output": out, "time_ps": time_ps, "cycle": time_ps // PERIOD_PS, "inputs": {}, "dut": None}
    if not os.path.exists(vcd_path):
        return fm
    try:
        ch = read_vcd(vcd_path)
    except Exception:
        return fm
    widths = {n: w for _, n, w in problem.ports}
    # The testbench checks on a clock edge, before that edge's non-blocking updates land, so the
    # values it compared are the ones from just before time_ps. If ref and dut agree there, the
    # stimulus used blocking assignments and the values AT time_ps are the compared ones.
    instant = True
    ref, dut = ch.get(f"{out}_ref", []), ch.get(f"{out}_dut", [])
    if ref and dut and value_at(ref, time_ps, True) == value_at(dut, time_ps, True):
        instant = False
    for d, name, w in problem.ports:
        if d == "input" and name in ch and name != "clk":
            fm["inputs"][name] = literal(value_at(ch[name], time_ps, instant), w)
    if dut:
        fm["dut"] = literal(value_at(dut, time_ps, instant), widths.get(out, 1))
    return fm


# ---------------------------------------------------------------- L3: formal

def formal_script(ref_path: str, cand_path: str) -> str:
    return (f"read_verilog -sv {ref_path}; read_verilog -sv {cand_path}; prep; "
            f"miter -equiv -flatten -make_outputs RefModule TopModule miter; hierarchy -top miter; "
            f"sat -prove trigger 0 -show-inputs -show-outputs miter")


SAT_ROW = re.compile(r"^\s*\\?(in|gold|gate)_(\w+)\s+.*?\s([01xXzZ]+)\s*$")


def parse_formal(output: str, problem) -> tuple:
    """(formal, counterexample, table_text). Pure, so it can be tested without yosys."""
    if re.search(r"^ERROR:", output, re.M):
        return "unsupported", None, ""
    if "model found: FAIL" in output:
        widths = {n: w for _, n, w in problem.ports}
        cex = {"inputs": {}, "gold": {}, "gate": {}}
        rows = []
        for line in output.splitlines():
            m = SAT_ROW.match(line)
            if m:
                rows.append(line.rstrip())
                side, name, bits = m.groups()
                cex["inputs" if side == "in" else side][name] = literal(bits.lower(), widths.get(name, len(bits)))
        return "not_equivalent", cex, "\n".join(rows)
    if "SUCCESS" in output or re.search(r"\bQED\b", output):
        return "equivalent", None, ""
    return "unsupported", None, ""


def formal_check(problem, code: str, workdir: str | None = None) -> tuple:
    """(formal, counterexample, table_text, timed_out) for one candidate. Used by L3 and audit.py."""
    own = workdir is None
    workdir = workdir or tempfile.mkdtemp(prefix="vra-formal-")
    try:
        cand = os.path.join(workdir, "cand.sv")
        if not os.path.exists(cand):
            with open(cand, "w") as f:
                f.write(code)
        rc, out, timed_out = run([YOSYS, "-p", formal_script(problem.ref_path, "cand.sv")], workdir)
        if timed_out:
            return "unsupported", None, "", True
        formal, cex, table = parse_formal(out, problem)
        return formal, cex, table, False
    finally:
        if own:
            shutil.rmtree(workdir, ignore_errors=True)


# ---------------------------------------------------------------- check

def check(problem, reply_text: str, formal: bool = True) -> CheckResult:
    t0 = time.time()
    res = CheckResult()
    workdir = tempfile.mkdtemp(prefix="vra-")
    try:
        if not l0(problem, reply_text, res):
            return res
        with open(os.path.join(workdir, "cand.sv"), "w") as f:
            f.write(res.code)
        if not l1(problem, workdir, res) or not l2(problem, workdir, res):
            return res
        if formal and not res.passed and problem.kind == "comb":
            # A formal timeout is not the candidate's fault and L2 has already failed it, so it
            # only means "no counterexample", never an UNVERIFIED claim.
            res.formal, res.counterexample, table, _ = formal_check(problem, res.code, workdir)
            if table:
                # Run B gets the same information as run C, only untranslated: the claim is about
                # the translation, so the raw text must not be missing the counterexample.
                res.raw += "\nyosys equivalence check, counterexample:\n" + table
            if res.formal in ("equivalent", "not_equivalent"):
                res.layer_reached = 3
        return res
    finally:
        res.elapsed_s = round(time.time() - t0, 3)
        shutil.rmtree(workdir, ignore_errors=True)


# ---------------------------------------------------------------- self-test

KMAP1_WRONG = """```verilog
module TopModule (input a, input b, input c, output out);
  assign out = a | b;
endmodule
```"""
KMAP1_WIRE = """```verilog
module TopModule (input a, input b, input c, output out);
  always @(*) out = a | b | c;
endmodule
```"""
COUNT_NO_RESET = """```verilog
module TopModule (input clk, input reset, output reg [3:0] q);
  always @(posedge clk)
    if (q == 10) q <= 1;
    else q <= q + 1;
endmodule
```"""
COUNT_OFF_BY_ONE = """```verilog
module TopModule (input clk, input reset, output reg [3:0] q);
  always @(posedge clk)
    if (reset || q == 11) q <= 1;
    else q <= q + 1;
endmodule
```"""


def reference_as_candidate(problem) -> str:
    with open(problem.ref_path) as f:
        return "```verilog\n" + re.sub(r"\bRefModule\b", "TopModule", f.read()) + "```"


def selftest(parsers_only: bool = False) -> int:
    kmap = P.load("Prob050_kmap1")
    count = P.load("Prob035_count1to10")
    fails = 0

    def report(name, ok, detail=""):
        nonlocal fails
        fails += not ok
        print(f"  {'ok  ' if ok else 'FAIL'}  {name}{('' if ok else '   ' + detail) if detail else ''}")

    print("parsers (no tools needed)")
    parser_selftest(report, kmap)
    if parsers_only:
        print("ALL OK (parsers only -- the six planted bugs need iverilog and yosys)" if not fails
              else f"{fails} FAILED")
        return 1 if fails else 0

    print("checker self-test (DESIGN 6.2): every planted bug must be caught")
    r = check(kmap, reference_as_candidate(kmap))
    report("1 reference renamed to TopModule passes, score 1.0", r.passed and r.score == 1.0,
           f"layer={r.layer_reached} score={r.score} {r.compile_error or ''}")
    r = check(kmap, KMAP1_WRONG)
    cex_c = (r.counterexample or {}).get("inputs", {}).get("c")
    report("2 kmap1 `a | b` fails, L3 counterexample has c=1",
           not r.passed and r.formal == "not_equivalent" and cex_c == "1",
           f"formal={r.formal} cex={r.counterexample}")
    r = check(kmap, KMAP1_WIRE)
    report("3 wire driven from always fails L1 with `is not a valid l-value`",
           r.layer_reached == 0 and "is not a valid l-value" in (r.compile_error or ""),
           f"error={r.compile_error!r}")
    r = check(count, COUNT_NO_RESET)
    report("4 counter without reset fails L2 with a reset hint",
           r.layer_reached >= 2 and not r.passed and any("reset" in h.lower() for h in r.tb_hints),
           f"hints={r.tb_hints}")
    r = check(count, COUNT_OFF_BY_ONE)
    report("5 off-by-one limit fails L2 with first_mismatch.cycle",
           r.layer_reached >= 2 and not r.passed and isinstance((r.first_mismatch or {}).get("cycle"), int),
           f"first_mismatch={r.first_mismatch}")
    r = check(kmap, "I am not able to write this module.")
    report("6 a reply with no code returns layer_reached = -1", r.layer_reached == -1 and not r.code)
    print("ALL OK" if not fails else f"{fails} FAILED")
    return 1 if fails else 0


def parser_selftest(report, kmap):
    sim = parse_sim("VCD info: dumpfile wave.vcd opened for output.\n"
                    "Hint: Your reset should be synchronous, but doesn't appear to be.\n"
                    "Hint: Output 'q' has 12 mismatches. First mismatch occurred at time 210.\n"
                    "Hint: Output 'r' has no mismatches.\n"
                    "Hint: Total mismatched samples is 12 out of 446 samples\n\n"
                    "Simulation finished at 4460 ps\nMismatches: 12 in 446 samples\n")
    report("parse_sim", sim["mismatches"] == 12 and sim["samples"] == 446
           and sim["per_output"] == {"q": {"mismatches": 12, "first_time_ps": 210},
                                     "r": {"mismatches": 0, "first_time_ps": None}}
           and sim["tb_hints"] == ["Hint: Your reset should be synchronous, but doesn't appear to be."],
           str(sim))
    out = ("SAT proof finished - model found: FAIL!\n\n"
           "        Signal Name                 Dec        Hex             Bin\n"
           "  -------------------- ---------- ---------- ---------------\n"
           "  \\gate_out                      0          0               0\n"
           "  \\gold_out                      1          1               1\n"
           "  \\in_a                          0          0               0\n"
           "  \\in_b                          0          0               0\n"
           "  \\in_c                          1          1               1\n")
    formal, cex, _ = parse_formal(out, kmap)
    report("parse_formal", formal == "not_equivalent" and cex["inputs"] == {"a": "0", "b": "0", "c": "1"}
           and cex["gold"] == {"out": "1"} and cex["gate"] == {"out": "0"}, str(cex))
    report("extract_code: fenced, think-tags, unterminated",
           extract_code("<think>\n</think>\nHere:\n```verilog\nmodule TopModule(); endmodule\n```\n")
           == "module TopModule(); endmodule\n"
           and extract_code("```systemverilog\nmodule TopModule(input a);\n  assign")
           .startswith("module TopModule")
           and extract_code("no code here") == "")
    report("literal", literal("1", 1) == "1" and literal("101", 4) == "4'b0101"
           and literal("1" * 16, 16) == "16'hffff" and literal("x", 4) == "4'bxxxx")
    vcd = ("$timescale 1ps $end\n$scope module tb $end\n$var reg 1 ! a $end\n"
           "$var reg 4 \" q_dut [3:0] $end\n$var reg 4 # q_ref [3:0] $end\n$upscope $end\n"
           "$scope module stim1 $end\n$var wire 1 $ clk $end\n$upscope $end\n$enddefinitions $end\n"
           "#0\n$dumpvars\n0!\nb0 \"\nb0 #\n0$\n$end\n#10\n1!\nb11 \"\nb11 #\n#20\n0!\nb100 \"\nb1010 #\n")
    path = os.path.join(tempfile.mkdtemp(prefix="vra-vcd-"), "wave.vcd")
    with open(path, "w") as f:
        f.write(vcd)
    ch = read_vcd(path)
    report("read_vcd + value_at", set(ch) == {"a", "q_dut", "q_ref"}
           and value_at(ch["a"], 20, True) == "1" and value_at(ch["a"], 20, False) == "0"
           and value_at(ch["q_dut"], 20, False) == "100", str(ch))
    fm = first_mismatch(P.Problem("t", "", "", "", "seq", [("input", "clk", 1), ("input", "a", 1),
                                                           ("output", "q", 4)]),
                        {"q": {"mismatches": 3, "first_time_ps": 20}}, path)
    report("first_mismatch", fm == {"output": "q", "time_ps": 20, "cycle": 2,
                                    "inputs": {"a": "0"}, "dut": "4'b0100"}, str(fm))
    shutil.rmtree(os.path.dirname(path), ignore_errors=True)


# ---------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--parsers-only", action="store_true",
                    help="with --selftest: only the tool-free parser tests (for a laptop)")
    ap.add_argument("--check", nargs=2, metavar=("PROBLEM", "FILE"), help="grade a reply or .sv file")
    ap.add_argument("--refs", metavar="LIST", help="check every reference in a problem list")
    ap.add_argument("--no-formal", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest(a.parsers_only))
    if a.check:
        prob = P.load(a.check[0])
        with open(a.check[1]) as f:
            r = check(prob, f.read(), formal=not a.no_formal)
        d = r.to_dict()
        d.pop("code")
        print(json.dumps(d, indent=2))
        sys.exit(0 if r.passed else 1)
    if a.refs:
        bad = 0
        for prob in P.load_list(a.refs):
            r = check(prob, reference_as_candidate(prob), formal=False)
            ok = r.passed and r.score == 1.0
            bad += not ok
            print(f"  {'ok  ' if ok else 'FAIL'}  {prob.id:40s} {prob.kind}  layer={r.layer_reached} "
                  f"score={r.score} {r.elapsed_s:.1f}s {r.compile_error or ''}")
        print(f"{bad} of the references did not score 1.0" if bad else "every reference scores 1.0")
        sys.exit(1 if bad else 0)
    ap.print_help()


if __name__ == "__main__":
    main()
