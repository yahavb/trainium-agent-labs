"""Turn a CheckResult into the feedback for the next attempt.

DESIGN.md 6.3. Two modes, and the comparison between them is the project's primary claim:

    raw       the tool's own text, cut to 1,500 characters        (run B)
    located   ONE instruction, at most 400 characters, naming      (runs C, D, E)
              where the design first fails and what to change

The event repo's lesson, learned eight times: a verdict ("off by 341 percent") does not move a
model, an instruction that names the change does. And never print the answer -- the model copies
it and derives nothing. So a located message carries the candidate's own wrong value and the
inputs that expose it, never the reference's code, the testbench's code, or an expected multi-bit
value.

    python translator.py --selftest
"""
from __future__ import annotations

import argparse
import os
import re
import sys

RAW_LIMIT = 1500
LOCATED_LIMIT = 400
LITERAL_LIMIT = 40

# Every rule came from a dev-set failure and says which (DESIGN 6.3). New rules: dev only.
RULES = [
    {"id": "R1", "pattern": r"is not a valid l-value",
     "instruction": "`{signal}` is assigned inside an always block, so declare it as {decl}, not a wire.",
     "observed_in": "Prob054_edgedetect, Prob109_fsm1, Prob100_fsm3comb (internal signal)"},
    {"id": "R2", "pattern": r"always process does not have any delay",
     "instruction": "This always block has no trigger. Use `always @(*)` (or `always_comb`) for "
                    "combinational logic, or `assign` for a constant.",
     "observed_in": "Prob001_zero"},
    {"id": "R3", "pattern": r"break statements not supported",
     "instruction": "Icarus does not support `break`. Replace it with a flag variable or restructure "
                    "the loop.",
     "observed_in": "Prob071_always_casez"},
    {"id": "R4", "pattern": r"requires an explicit cast",
     "instruction": "Assigning to an enum needs a cast, e.g. `state <= state_t'(next);`, or declare "
                    "the state as `logic [N:0]`.",
     "observed_in": "Prob142_lemmings2"},
    {"id": "R5", "pattern": r"is not allowed in a constant expression",
     "instruction": "A part-select `[hi:lo]` needs constant bounds. To take W bits starting at a "
                    "variable position, use the indexed part-select `vector[start +: W]`.",
     "observed_in": "Prob021_mux256to1v"},
    {"id": "R6", "pattern": r"can not select part of scalar",
     "instruction": "`{signal}` is declared as a single bit, but the code selects bits of it. Declare it "
                    "with its full width: {decl}.",
     "observed_in": "Prob073_dff16e"},
    {"id": "R8", "pattern": r"has already been declared in this scope",
     "instruction": "`{signal}` is declared twice. {keep}",
     "observed_in": "Prob109_fsm1 (after R1, the model added `reg out;` next to the `output out` port)"},
    # R7 matches on the offending LINE as well as the message: a bare "syntax error" on a declaration.
    {"id": "R7", "pattern": r"syntax error", "line_pattern": r"^(integer|reg|logic|wire|int|bit|genvar)\b",
     "instruction": "Icarus rejects the declaration `{line}` at this point. Declare `{signal}` once at module "
                    "level, above the always block, and only use it inside the block.",
     "observed_in": "Prob071_always_casez"},
]

COMPILE_LOC = re.compile(r"^(?P<file>[^:\s]+):(?P<line>\d+):\s*(?P<msg>.*)$")


def short(value: str) -> str:
    """Wide literals abbreviated, so one 1024-bit input cannot eat the whole message."""
    if len(value) <= LITERAL_LIMIT:
        return value
    head, _, digits = value.partition("'")
    return f"{head}'{digits[:9]}...{digits[-8:]}"


def assignment_list(values: dict) -> str:
    return ", ".join(f"{k}={short(v)}" for k, v in values.items())


def cap(msg: str, limit: int = LOCATED_LIMIT) -> str:
    msg = re.sub(r"\s+", " ", msg).strip()
    return msg if len(msg) <= limit else msg[: limit - 3].rstrip() + "..."


def reset_input(problem) -> str | None:
    names = [n for d, n, _ in problem.ports if d == "input" and "reset" in n.lower()]
    return names[0] if names else None


def matching_rule(error: str, line: str = ""):
    for rule in RULES:
        if not re.search(rule["pattern"], error or ""):
            continue
        if "line_pattern" in rule and not re.search(rule["line_pattern"], line or ""):
            continue
        return rule
    return None


def compile_message(problem, result) -> str:
    error = result.compile_error or "iverilog failed"
    loc = COMPILE_LOC.match(error)
    msg = re.sub(r"^(?:error|sorry):\s*", "", loc.group("msg").strip()) if loc else error
    in_candidate = bool(loc) and loc.group("file") == "cand.sv"
    where, text = "", ""
    if in_candidate:
        n = int(loc.group("line"))
        lines = result.code.splitlines()
        text = lines[n - 1].strip() if 0 < n <= len(lines) else ""
        where = f" (line {n}: `{text}`)" if text else f" (line {n})"
    rule = matching_rule(error, text)
    if rule and rule["id"] == "R7":
        names = re.findall(r"([A-Za-z_]\w*)\s*(?:\[[^\]]*\])?\s*[,;=]", re.sub(r"\[[^\]]*\]", "", text))
        return rule["instruction"].format(line=text, signal=names[0] if names else "it") + f" (line {n})"
    if rule:
        sig = (re.search(r"(?:^|\s)([\w.]+) is not a valid l-value", msg)
               or re.search(r"can not select part of scalar: (\w+)", msg)
               or re.search(r"'(\w+)' has already been declared", msg))
        signal = sig.group(1).split(".")[-1] if sig else "the signal"
        is_port = any(n == signal for _, n, _ in problem.ports)
        keep = (f"Keep one declaration, {declaration(problem, signal, 'R1')} in the port list, and delete "
                f"the other." if is_port else "Keep one declaration and delete the other.")
        text_out = rule["instruction"].format(signal=signal, decl=declaration(problem, signal, rule["id"]),
                                              keep=keep)
        if rule["id"] == "R1" and is_port:
            # Prob109_fsm1 (dev C-3): told to "declare it as `output reg out`", the model ADDED
            # `reg out;` beside the port and hit R8 instead. Say where the change goes.
            text_out += " Change it in the port list itself; do not add a second declaration."
        return text_out + where
    if not in_candidate:
        # The error surfaced in the testbench or reference, which means TopModule's interface does
        # not match what they instantiate. Quote the message, never their code.
        return (f"The testbench cannot use TopModule: `{msg}`. Use exactly these ports: "
                f"{problem.port_list()}.")
    # Observed on dev (Prob092_gatesv100): a bare `syntax error` with only a line number was copied
    # back unchanged, so the message now quotes the line itself.
    shown = f" The line is `{text}`." if text else ""
    return f"Line {loc.group('line')} does not compile: `{msg}`.{shown} Fix that line."


def declaration(problem, signal: str, rule_id: str) -> str:
    """How `signal` should be declared: from the specification's port list when it is a port (the
    spec already states it, so this reveals nothing), else as an internal variable."""
    port = next(((d, w) for d, n, w in problem.ports if n == signal), None)
    if port:
        d, w = port
        rng = f"[{w - 1}:0] " if w > 1 else ""
        return f"`{d} {'reg ' if d == 'output' and rule_id == 'R1' else ''}{rng}{signal}`"
    return f"`reg` (or `logic`) with its width, e.g. `logic [N-1:0] {signal}`" if rule_id == "R6" else "`reg` (or `logic`)"


def active_low(name: str) -> bool:
    """resetn, reset_n, rst_n, aresetn, nreset: the circuit resets when the signal is LOW.
    Observed in Prob073_dff16e (dev): telling the model `resetn` "is high" pointed it the wrong way."""
    n = name.lower()
    return bool(re.search(r"(reset|rst)_?n$|_n$|_b$|^n_?(reset|rst)", n))


def reset_message(problem, hints: list) -> str | None:
    r = reset_input(problem) or "reset"
    low = active_low(r)
    level, edge = ("low", "negedge") if low else ("high", "posedge")
    for h in hints:
        if "reset should be synchronous" in h:
            return (f"Make the reset synchronous: check `{r}` only inside `always @(posedge clk)`, "
                    f"not in the sensitivity list.")
        if "reset should be asynchronous" in h:
            return (f"Make the reset asynchronous: use `always @(posedge clk, {edge} {r})` and test "
                    f"`{r}` first in that block{' (it is active low)' if low else ''}.")
        if "reset doesn't seem to be working" in h:
            return (f"When `{r}` is {level}, the outputs do not take their reset values. Test `{r}` "
                    f"first and assign every register the reset value the specification gives.")
    return None


def mismatch_message(problem, result) -> str | None:
    cex = result.counterexample
    if problem.kind == "comb" and cex and cex.get("gate"):
        wrong = [o for o in cex.get("gate", {}) if cex["gate"][o] != cex.get("gold", {}).get(o)]
        out = wrong[0] if wrong else next(iter(cex.get("gate", {})), None)
        if out and not cex.get("inputs"):
            # A module with no inputs (Prob001_zero, dev): "for that input combination" means nothing,
            # and the old fallback, "wrong for some inputs", was repeated back three times.
            return (f"Your `{out}` is {short(cex['gate'][out])}, which is wrong. Re-read what the "
                    f"specification says `{out}` must be.")
        if out:
            return (f"When {assignment_list(cex['inputs'])}, your `{out}` is "
                    f"{short(cex['gate'][out])}, which is wrong. Re-derive `{out}` for that input "
                    f"combination.")
    fm = result.first_mismatch
    if not fm:
        return None
    out, dut, inputs = fm["output"], fm.get("dut"), fm.get("inputs") or {}
    if problem.kind == "comb":
        if inputs and dut is not None:
            return (f"When {assignment_list(inputs)}, your `{out}` is {short(dut)}, which is wrong. "
                    f"Re-derive `{out}` for that input combination.")
        return f"Output `{out}` is wrong for some inputs. Re-derive `{out}` from the specification."
    seen = f" At that cycle the inputs were {assignment_list(inputs)}" if inputs else ""
    val = f" and your `{out}` was {short(dut)}" if dut is not None else ""
    lead = (seen + val + ".") if seen else (f" Your `{out}` was {short(dut)}." if dut is not None else "")
    return (f"Output `{out}` first goes wrong at clock cycle {fm['cycle']}.{lead} Check the logic "
            f"that updates `{out}` on that cycle.")


def has_x(value) -> bool:
    """True when a value from the simulation has an X or Z bit (`x`, `4'bx0x1`, `8'bzzzz0000`)."""
    if not value:
        return False
    bits = value.split("'", 1)[1][1:] if "'" in value else value
    return any(c in "xz" for c in bits.lower())


def x_message(problem, result) -> str | None:
    """The testbench saw X or Z: no value at all. Observed on dev (TAXONOMY.md, finding 3): on
    Prob093_ece241_2014_q3 the simulator saw `4'bxxxx` but the message quoted Yosys's counterexample,
    "your `mux_in` is 4'b1100", a defined value the testbench never saw. Neither problem recovered."""
    fm = result.first_mismatch
    if not fm or not has_x(fm.get("dut")):
        return None
    out, dut, inputs = fm["output"], short(fm["dut"]), fm.get("inputs") or {}
    if problem.kind == "comb":
        when = f"when {assignment_list(inputs)}, " if inputs else ""
        return (f"In simulation, {when}your `{out}` is {dut}: X means it has no value. Some bit of `{out}` "
                f"is never assigned on that path, or is driven from two places. Assign every bit of "
                f"`{out}` on every path, in one place.")
    return (f"Output `{out}` is {dut} at clock cycle {fm['cycle']}: X means it has no value. A register "
            f"behind it is never given a value, or is assigned in two places. Give every register a value "
            f"on reset and assign each one in only one always block.")


def located(problem, result) -> str:
    if result.passed:
        return ""
    if result.layer_reached < 0:
        if result.l0_error and result.l0_error.startswith("missing_port:"):
            name = result.l0_error.split(":", 1)[1]
            d, w = next(((d, w) for d, n, w in problem.ports if n == name), ("input", 1))
            return cap(f"TopModule is missing port `{name}` (`{d}`, {w} bits). Use exactly the ports "
                       f"in the specification.")
        return cap(f"Reply with one ```verilog block containing `module TopModule` with exactly these "
                   f"ports: {problem.port_list()}.")
    if result.layer_reached == 0:
        if result.timed_out:
            return cap("iverilog did not finish compiling. Remove any loop or generate construct "
                       "whose bound is not a constant.")
        return cap(compile_message(problem, result))
    if result.mismatches is None:
        return cap("The simulation never finished. Look for a combinational loop, or an always block "
                   "with neither a clock edge nor `@(*)`.")
    return cap(reset_message(problem, result.tb_hints) or x_message(problem, result)
               or mismatch_message(problem, result)
               or f"{result.mismatches} of {result.samples} samples are wrong. Re-read the "
                  f"specification and check every output.")


def feedback(problem, result, mode: str) -> str:
    if mode == "raw":
        return (result.raw or "")[:RAW_LIMIT]
    if mode == "located":
        return located(problem, result)
    raise ValueError(f"unknown feedback mode {mode!r}")


# ---------------------------------------------------------------- leak check (TRN-5)

def ref_lines(problem) -> list:
    """Reference lines distinctive enough that seeing one in a message would be a leak."""
    with open(problem.ref_path) as f:
        lines = [ln.strip() for ln in f]
    boring = {"endmodule", "end", "begin", "else", "endcase", "default:", ");", "end else begin"}
    return [ln for ln in lines if len(ln) >= 12 and ln not in boring and not ln.startswith("//")
            and not re.match(r"^(input|output)\b", ln)]


def leaks(problem, message: str, candidate_code: str = "") -> list:
    """Reference lines that appear in the message and did not come from the candidate itself."""
    return [ln for ln in ref_lines(problem) if ln in message and ln not in candidate_code]


# ---------------------------------------------------------------- self-test

def selftest() -> int:
    import problems as P
    from checker import CheckResult
    kmap, count = P.load("Prob050_kmap1"), P.load("Prob035_count1to10")
    fails = 0

    def expect(name, problem, result, pattern, mode="located"):
        nonlocal fails
        msg = feedback(problem, result, mode)
        leaked = leaks(problem, msg, result.code)
        ok = re.search(pattern, msg) is not None and len(msg) <= (LOCATED_LIMIT if mode == "located"
                                                                  else RAW_LIMIT) and not leaked
        fails += not ok
        print(f"  {'ok  ' if ok else 'FAIL'}  {name}\n        {msg[:160]}" + (f"\n        LEAK: {leaked}" if leaked else ""))

    print("translator self-test (DESIGN 6.3)")
    expect("no code", kmap, CheckResult(layer_reached=-1, l0_error="no_code"),
           r"^Reply with one ```verilog block containing `module TopModule`.*input a, input b, input c, output out")
    expect("missing port", kmap, CheckResult(layer_reached=-1, l0_error="missing_port:c"),
           r"missing port `c` \(`input`, 1 bits\)")
    code = "module TopModule (input a, input b, input c, output out);\n  always @(*) out = a | b | c;\nendmodule\n"
    expect("R1 wire l-value, with the candidate's line", kmap,
           CheckResult(layer_reached=0, code=code,
                       compile_error="cand.sv:2: error: out is not a valid l-value in tb.top_module1."),
           r"^`out` is assigned inside an always block, so declare it as `output reg out`, not a wire\. Change it in the port list itself; do not add a second declaration\. \(line 2: `always @\(\*\) out = a \| b \| c;`\)")
    expect("R1 on an internal signal says reg, not output", kmap,
           CheckResult(layer_reached=0, code=code,
                       compile_error="cand.sv:2: error: next_state is not a valid l-value in tb.top_module1."),
           r"^`next_state` is assigned inside an always block, so declare it as `reg` \(or `logic`\), not a wire")
    mux = P.Problem("Prob021_mux256to1v", "", kmap.ref_path, "", "comb",
                    [("input", "in", 1024), ("input", "sel", 8), ("output", "out", 4)])
    expect("R5 variable part-select", mux,
           CheckResult(layer_reached=0, code="module TopModule(input [1023:0] in, input [7:0] sel, output [3:0] out);\n"
                       "  wire [9:0] index = sel * 4;\n  assign out = in[index + 3 : index];\nendmodule\n",
                       compile_error="cand.sv:3: error: A reference to a wire or reg (`index') is not allowed in a constant expression."),
           r"^A part-select `\[hi:lo\]` needs constant bounds\..*`vector\[start \+: W\]`\. \(line 3: `assign out = in\[index \+ 3 : index\];`\)")
    dff = P.Problem("Prob073_dff16e", "", kmap.ref_path, "", "seq",
                    [("input", "clk", 1), ("input", "resetn", 1), ("input", "byteena", 2), ("input", "d", 16), ("output", "q", 16)])
    expect("R6 bit-select of a 1-bit port gives the spec's width", dff,
           CheckResult(layer_reached=0, code="module TopModule(input clk, input byteena);\n  always @(posedge clk)\n    if (byteena[1]) ;\nendmodule\n",
                       compile_error="cand.sv:3: error: can not select part of scalar: byteena"),
           r"^`byteena` is declared as a single bit.*Declare it with its full width: `input \[1:0\] byteena`\. \(line 3: `if \(byteena\[1\]\) ;`\)")
    expect("compile error with no rule", kmap,
           CheckResult(layer_reached=0, code="module TopModule(\n input a\n", compile_error="cand.sv:2: syntax error"),
           r"^Line 2 does not compile: `syntax error`\. The line is `input a`\. Fix that line\.$")
    expect("error inside the testbench", kmap,
           CheckResult(layer_reached=0, code=code,
                       compile_error="Prob050_kmap1_test.sv:93: error: port ``c'' is not a port of top_module1."),
           r"^The testbench cannot use TopModule: .*Use exactly these ports")
    expect("comb counterexample", kmap,
           CheckResult(layer_reached=3, code=code, mismatches=40, samples=430, formal="not_equivalent",
                       counterexample={"inputs": {"a": "0", "b": "0", "c": "1"}, "gold": {"out": "1"},
                                       "gate": {"out": "0"}}),
           r"^When a=0, b=0, c=1, your `out` is 0, which is wrong\.")
    zero = P.load("Prob001_zero")
    expect("no-input module names its value, not 'some inputs'", zero,
           CheckResult(layer_reached=3, code="module TopModule(output zero); assign zero = 1; endmodule",
                       mismatches=5, samples=5, formal="not_equivalent",
                       counterexample={"inputs": {}, "gold": {"zero": "0"}, "gate": {"zero": "1"}}),
           r"^Your `zero` is 1, which is wrong\. Re-read what the specification says `zero` must be\.$")
    dff = P.Problem("Prob073_dff16e", "", kmap.ref_path, "", "seq",
                    [("input", "clk", 1), ("input", "resetn", 1), ("output", "q", 16)])
    expect("active-low reset says LOW", dff,
           CheckResult(layer_reached=2, code=code, mismatches=40, samples=400,
                       tb_hints=["Hint: Your reset doesn't seem to be working."]),
           r"^When `resetn` is low, the outputs do not take their reset values")
    casez = P.Problem("Prob071_always_casez", "", kmap.ref_path, "", "comb", [("input", "in", 8), ("output", "pos", 3)])
    expect("R7 declaration inside a block", casez,
           CheckResult(layer_reached=0, code="module TopModule(input [7:0] in, output reg [2:0] pos);\n"
                       "  always @(*) begin\n    pos = 0;\n    integer i;\n  end\nendmodule\n",
                       compile_error="cand.sv:4: syntax error"),
           r"^Icarus rejects the declaration `integer i;` at this point\. Declare `i` once at module level.*\(line 4\)$")
    expect("a syntax error on a normal line is not R7", casez,
           CheckResult(layer_reached=0, code="module TopModule(\n pos = 0\n", compile_error="cand.sv:2: syntax error"),
           r"^Line 2 does not compile: `syntax error`\. The line is `pos = 0`")
    fsm = P.Problem("Prob109_fsm1", "", kmap.ref_path, "", "seq",
                    [("input", "clk", 1), ("input", "in", 1), ("input", "areset", 1), ("output", "out", 1)])
    expect("R8 a port declared twice", fsm,
           CheckResult(layer_reached=0, code="module TopModule(input clk, input in, input areset, output out);\n"
                       "  reg out;\nendmodule\n",
                       compile_error="cand.sv:2: error: 'out' has already been declared in this scope."),
           r"^`out` is declared twice\. Keep one declaration, `output reg out` in the port list, and delete the other\. \(line 2: `reg out;`\)$")
    q3 = P.Problem("Prob093_ece241_2014_q3", "", kmap.ref_path, "", "comb",
                   [("input", "c", 1), ("input", "d", 1), ("output", "mux_in", 4)])
    expect("X in simulation beats the formal counterexample (comb)", q3,
           CheckResult(layer_reached=3, code=code, mismatches=100, samples=200, formal="not_equivalent",
                       counterexample={"inputs": {"c": "0", "d": "0"}, "gold": {"mux_in": "4'b0100"},
                                       "gate": {"mux_in": "4'b1100"}},
                       first_mismatch={"output": "mux_in", "time_ps": 20, "cycle": 2,
                                       "inputs": {"c": "0", "d": "1"}, "dut": "4'bxxxx"}),
           r"^In simulation, when c=0, d=1, your `mux_in` is 4'bxxxx: X means it has no value\.")
    expect("X in a sequential output", count,
           CheckResult(layer_reached=2, code=code, mismatches=40, samples=400,
                       first_mismatch={"output": "q", "time_ps": 30, "cycle": 3, "inputs": {"reset": "0"},
                                       "dut": "4'bxxxx"}),
           r"^Output `q` is 4'bxxxx at clock cycle 3: X means it has no value\. A register")
    expect("a defined wrong value still gets the counterexample", q3,
           CheckResult(layer_reached=3, code=code, mismatches=100, samples=200, formal="not_equivalent",
                       counterexample={"inputs": {"c": "0", "d": "0"}, "gold": {"mux_in": "4'b0100"},
                                       "gate": {"mux_in": "4'b1100"}},
                       first_mismatch={"output": "mux_in", "time_ps": 20, "cycle": 2,
                                       "inputs": {"c": "0", "d": "1"}, "dut": "4'b1100"}),
           r"^When c=0, d=0, your `mux_in` is 4'b1100, which is wrong\.")
    expect("reset hint", count,
           CheckResult(layer_reached=2, code=code, mismatches=400, samples=446,
                       tb_hints=["Hint: Your reset doesn't seem to be working."]),
           r"^When `reset` is high")
    expect("seq first mismatch, no expected value", count,
           CheckResult(layer_reached=2, code=code, mismatches=20, samples=446,
                       first_mismatch={"output": "q", "time_ps": 210, "cycle": 21,
                                       "inputs": {"reset": "0"}, "dut": "4'b1011"}),
           r"^Output `q` first goes wrong at clock cycle 21\. At that cycle the inputs were reset=0 and your `q` was 4'b1011\.")
    expect("raw is the tool text, capped", kmap, CheckResult(layer_reached=2, raw="x" * 3000), r"^x{1500}$", "raw")
    wide = {"in": "1024'h" + "f" * 256, "sel": "8'b00000011"}
    msg = mismatch_message(kmap, CheckResult(first_mismatch={"output": "out", "time_ps": 5, "cycle": 0,
                                                             "inputs": wide, "dut": "4'b0000"}))
    ok = len(cap(msg)) <= LOCATED_LIMIT and "sel=8'b00000011" in msg
    fails += not ok
    print(f"  {'ok  ' if ok else 'FAIL'}  wide inputs abbreviated\n        {cap(msg)}")
    ok = bool(leaks(kmap, "the answer is assign out = (a | b | c);"))
    fails += not ok
    print(f"  {'ok  ' if ok else 'FAIL'}  leak check catches a reference line")
    print("ALL OK" if not fails else f"{fails} FAILED")
    return 1 if fails else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    sys.exit(selftest() if a.selftest else ap.print_help())
