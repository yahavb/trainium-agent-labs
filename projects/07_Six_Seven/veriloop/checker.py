"""VeriLoop checker.

Grades a Verilog design the model wrote. The checker is the authority: the model never overrides it.

    C1 (this part)  compile the design and explain any error in plain words, with the line
    C2              generate a test bench from the level's reference.py, simulate, compare
    C3              score 0-1 and feedback at three levels: A pass/fail, B how many wrong, C the first wrong step

Usage:
    python checker.py levels/03_counter design.v      # score it and print feedback A, B and C
    python checker.py levels/03_counter design.v --feedback C
    python checker.py design.v --module counter8      # compile only, no port check

Needs Icarus Verilog (`iverilog`) on the PATH -- see SERVER.md.
"""

import argparse
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field

IVERILOG = os.environ.get("IVERILOG", "iverilog")
COMPILE_TIMEOUT_S = 20
MAX_MESSAGES = 3          # more than this and the model fixes nothing; the first errors matter most
PROBE = "__veriloop_probe"


# ---------------------------------------------------------------- levels

def load_level(level_dir):
    """Load a level folder's reference.py (see levels/README.md for the contract)."""
    path = os.path.join(level_dir, "reference.py")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{level_dir} has no reference.py -- see levels/README.md")
    spec = importlib.util.spec_from_file_location("reference", path)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    for name in ("MODULE", "INPUTS", "OUTPUTS", "CLOCKED"):
        if not hasattr(ref, name):
            raise ValueError(f"{path} is missing `{name}` -- see levels/README.md")
    return ref


def ports_of(ref):
    """The ports the spec promises: {name: (direction, width)}. A clocked level also gets `clk`."""
    ports = {}
    if ref.CLOCKED:
        ports["clk"] = ("input", 1)
    ports.update({n: ("input", w) for n, w in ref.INPUTS.items()})
    ports.update({n: ("output", w) for n, w in ref.OUTPUTS.items()})
    return ports


# ---------------------------------------------------------------- C1: compile

@dataclass
class CompileResult:
    ok: bool
    messages: list = field(default_factory=list)   # plain-words problems, for the model and for people
    raw: str = ""                                   # iverilog's own output, for debugging


def _decl(name, width):
    return f"[{width - 1}:0] {name}" if width > 1 else name


def _probe_source(module, ports):
    """A tiny wrapper that instantiates the design with the spec's port names and widths.

    Compiling it is how a wrong port name (an error) or a wrong width (only a warning in iverilog, which
    silently drops bits) gets caught before any simulation.
    """
    lines = [f"module {PROBE};"]
    for name, (direction, width) in ports.items():
        kind = "reg" if direction == "input" else "wire"
        lines.append(f"  {kind} {_decl(name, width)};")
    conns = ", ".join(f".{n}({n})" for n in ports)
    lines.append(f"  {module} dut({conns});")
    lines.append("endmodule")
    return "\n".join(lines) + "\n"


_LOC = re.compile(r"^(?P<file>[^:\n]+):(?P<line>\d+):\s*(?P<rest>.*)$")


def explain(raw, design_lines, module, ports):
    """Turn iverilog's output into at most MAX_MESSAGES plain-words messages, each naming the line.

    A verdict ("syntax error") is not an instruction. Each message says what is wrong AND what to change,
    without writing the design for the model.
    """
    out = []

    def src(n):
        return design_lines[n - 1].strip() if 0 < n <= len(design_lines) else ""

    def add(msg):
        if msg not in out:
            out.append(msg)

    port_list = ", ".join(f"{d} {_decl(n, w)}" for n, (d, w) in ports.items()) if ports else ""

    for line in raw.splitlines():
        m = _LOC.match(line)
        if not m:
            root = re.search(r'Unable to find the root module "([^"]+)"', line)
            if root:
                add(f"There is no module named `{module}`. The spec asks for exactly that name: "
                    f"`module {module}(...)`.")
            continue
        fname, n, rest = os.path.basename(m["file"]), int(m["line"]), m["rest"]
        in_design = fname == "design.v"

        if rest.startswith(":") or "Unable to elaborate" in rest:
            continue                       # continuation lines and knock-on errors add nothing

        if in_design and rest.startswith("syntax error"):
            text = src(n)
            if n == 1 and not re.match(r"\s*(module|`|//|/\*)", text):
                add(f"Line 1 is not Verilog: `{text[:80]}`. Send only the Verilog module -- no words "
                    f"before or after it.")
                continue
            hint = "The mistake is often at the end of the line just before it -- a missing `;`."
            src_all = "\n".join(design_lines)
            begins = len(re.findall(r"\bbegin\b", src_all))
            ends = len(re.findall(r"\bend\b", src_all))
            if begins > ends:
                hint = (f"There are {begins} `begin` but only {ends} `end`: a block is not closed "
                        f"before this line.")
            add(f"Syntax error at line {n}: `{text[:80]}`. {hint}")
            continue

        x = re.search(r"Unable to bind wire/reg/memory `([^']+)'", rest)
        if x:
            add(f"Line {n}: `{src(n)[:80]}` uses `{x[1]}`, which is never declared. Declare it, or use "
                f"a port name from the spec.")
            continue
        # iverilog 13: "'count' is not a valid l-value for a procedural assignment"
        # iverilog 12: "count is not a valid l-value in top.dut."
        x = re.search(r"'?([A-Za-z_]\w*)'? is not a valid l-value", rest)
        if x:
            add(f"Line {n}: `{x[1]}` is assigned inside an `always` block but declared as a wire. "
                f"Declare it as `reg` -- for an output port, `output reg {x[1]}` with its width.")
            continue
        x = re.search(r"'([^']+)' has already been declared", rest)
        if x:
            add(f"Line {n}: `{x[1]}` is declared twice. Keep only one declaration.")
            continue
        x = re.search(r"Unknown module type: (\S+)", rest)
        if x and not in_design and x[1] == module:
            # Raised by the port probe: the design never defines the module the spec names.
            add(f"There is no module named `{module}`. The spec asks for exactly that name: "
                f"`module {module}(...)`.")
            continue
        if x:
            add(f"Line {n}: the design uses a module `{x[1]}` that is not defined. Put everything inside "
                f"the one module `{module}`, or define `{x[1]}` in the same file.")
            continue
        x = re.search(r"port ``([^']+)'' is not a port of", rest)
        if x:
            add(f"The module has no port named `{x[1]}`. The spec's ports are exactly: {port_list}. "
                f"Use those names.")
            continue
        # iverilog 13: "Port 1 (a) of module add4 expects 3 bit(s), given 4."
        # iverilog 12: "Port 1 (a) of add4 expects 3 bits, got 4."
        x = re.search(r"Port \d+ \(([^)]+)\) of (?:module )?\S+ expects (\d+) bits?(?:\(s\))?, "
                      r"(?:given|got) (\d+)", rest)
        if x:
            name, got, want = x[1], int(x[2]), int(x[3])
            add(f"Port `{name}` is {got} bit(s) wide, but the spec says {want}. Declare it as "
                f"`{_decl(name, want)}`.")
            continue
        if "error" in rest:
            where = f"Line {n}: `{src(n)[:80]}` -- " if in_design else ""
            add(f"{where}{rest.replace('error: ', '')}")

        if len(out) >= MAX_MESSAGES:
            break

    if not out and raw.strip():
        out.append("The design does not compile: " + raw.strip().splitlines()[0])
    return out[:MAX_MESSAGES]


def compile_verilog(design_src, module, ports=None, workdir=None):
    """Compile `design_src` (Verilog text). With `ports`, also check the module has exactly the spec's
    port names and widths. Returns a CompileResult; on success `workdir` holds the compiled `sim`."""
    if not shutil.which(IVERILOG):
        raise RuntimeError("iverilog is not installed here -- see SERVER.md, 'Install the simulator'.")
    workdir = workdir or tempfile.mkdtemp(prefix="veriloop_")
    design = os.path.join(workdir, "design.v")
    with open(design, "w") as f:
        f.write(design_src)

    files, top = [design], module
    if ports:
        probe = os.path.join(workdir, "probe.v")
        with open(probe, "w") as f:
            f.write(_probe_source(module, ports))
        files, top = [probe, design], PROBE

    # -g2012 accepts the SystemVerilog style the model often writes (logic, always_ff, '0).
    cmd = [IVERILOG, "-g2012", "-s", top, "-o", os.path.join(workdir, "sim")] + files
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=COMPILE_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return CompileResult(False, [f"Compiling took over {COMPILE_TIMEOUT_S}s -- the design is "
                                     f"probably far too large or loops forever at compile time."])
    raw = (p.stdout + p.stderr).strip()
    messages = explain(raw, design_src.splitlines(), module, ports)
    width_problem = any(m.startswith("Port `") and "bit(s) wide" in m for m in messages)  # a warning in iverilog
    ok = p.returncode == 0 and not width_problem
    return CompileResult(ok, [] if ok else messages, raw)


# ---------------------------------------------------------------- C2: simulate and compare

SIM_TIMEOUT_S = 10
TB = "__veriloop_tb"


@dataclass
class SimResult:
    compiled: CompileResult
    ran: bool = False            # the simulation finished and printed every step
    problem: str = ""            # why it did not run, in plain words (empty when it ran)
    rows: list = field(default_factory=list)        # per step: (inputs, expected, got)
    mismatches: list = field(default_factory=list)  # per wrong signal: dict(step, signal, expected, got)

    @property
    def passed(self):
        return self.compiled.ok and self.ran and not self.mismatches


def _mask(value, width):
    return int(value) & ((1 << width) - 1)


def testbench_source(ref, vecs):
    """A Verilog test bench that drives `vecs` into the design and prints every output after each step.

    Combinational: set the inputs, wait 1 ns, print. Clocked: set the inputs while clk is low, raise clk,
    print 1 ns after the rising edge (the contract: "the output just after that cycle's rising edge"),
    then lower clk. Outputs are printed in binary so an undefined value (x or z) shows up as such.
    """
    ins, outs = ref.INPUTS, ref.OUTPUTS
    L = ["`timescale 1ns/1ps", f"module {TB};"]
    if ref.CLOCKED:
        L.append("  reg clk;")
    L += [f"  reg {_decl(n, w)};" for n, w in ins.items()]
    L += [f"  wire {_decl(n, w)};" for n, w in outs.items()]
    conns = (["clk"] if ref.CLOCKED else []) + list(ins) + list(outs)
    L.append(f"  {ref.MODULE} dut(" + ", ".join(f".{n}({n})" for n in conns) + ");")
    fmt = " ".join(f"{n}=%b" for n in outs)
    show = f'$display("@@ %0d {fmt}", step, ' + ", ".join(outs) + ");"
    L += ["  integer step;", "  initial begin"]
    if ref.CLOCKED:
        L.append("    clk = 0;")
    for i, v in enumerate(vecs):
        sets = " ".join(f"{n} = {w}'d{_mask(v[n], w)};" for n, w in ins.items())
        if ref.CLOCKED:
            L.append(f"    step = {i}; {sets} #4 clk = 1; #1 {show} #5 clk = 0;")
        else:
            L.append(f"    step = {i}; {sets} #1 {show}")
    L += ["    $finish;", "  end", "endmodule"]
    return "\n".join(L) + "\n"


def _parse_bits(bits):
    return None if re.search(r"[xzXZ]", bits) else int(bits, 2)


def simulate(ref, design_src, workdir=None):
    """Compile the design against the spec, simulate it on ref.vectors(), and compare every output with
    ref.reference(). Returns a SimResult; C3 turns it into a score and feedback."""
    workdir = workdir or tempfile.mkdtemp(prefix="veriloop_")
    compiled = compile_verilog(design_src, ref.MODULE, ports_of(ref), workdir)
    result = SimResult(compiled)
    if not compiled.ok:
        return result

    vecs = ref.vectors()
    want = ref.reference(vecs)
    if len(want) != len(vecs):
        raise ValueError(f"reference() returned {len(want)} steps for {len(vecs)} vectors")

    tb = os.path.join(workdir, "tb.v")
    with open(tb, "w") as f:
        f.write(testbench_source(ref, vecs))
    sim = os.path.join(workdir, "tb_sim")
    p = subprocess.run([IVERILOG, "-g2012", "-s", TB, "-o", sim, tb, os.path.join(workdir, "design.v")],
                       capture_output=True, text=True, timeout=COMPILE_TIMEOUT_S)
    if p.returncode != 0:
        # The port probe passed, so this is rare; report it rather than guess.
        result.problem = "The design compiles alone but not inside the test bench: " + \
            (p.stdout + p.stderr).strip().splitlines()[0]
        return result

    try:
        run = subprocess.run(["vvp", "-n", sim], capture_output=True, text=True, timeout=SIM_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        result.problem = (f"The simulation did not finish within {SIM_TIMEOUT_S}s. Usually a loop that never "
                          f"ends: a `while` or `for` inside an `always` block whose condition never becomes "
                          f"false, or logic whose output feeds back into its own input.")
        return result

    got_by_step = {}
    for line in run.stdout.splitlines():
        if line.startswith("@@ "):
            parts = line.split()
            got_by_step[int(parts[1])] = {k: _parse_bits(v) for k, v in (x.split("=", 1) for x in parts[2:])}
    if len(got_by_step) < len(vecs):
        result.problem = (f"The simulation stopped after {len(got_by_step)} of {len(vecs)} steps. Remove any "
                          f"`$finish`, `$stop` or `initial` block from the design -- the test bench drives it.")
        return result

    result.ran = True
    for i, (v, w) in enumerate(zip(vecs, want)):
        got = got_by_step[i]
        exp = {n: _mask(w[n], width) for n, width in ref.OUTPUTS.items()}
        result.rows.append((v, exp, got))
        for n in ref.OUTPUTS:
            if got.get(n) != exp[n]:
                result.mismatches.append(dict(step=i, signal=n, expected=exp[n], got=got.get(n)))
    return result


# ---------------------------------------------------------------- C3: score and feedback A / B / C

# Score: 0 does not compile, 0.2 compiles (with the spec's ports) but does not run to the end,
# 0.3 + 0.7 x (fraction of steps fully correct) once it runs. Solved means 1.0: every step right.
W_COMPILES, W_RUNS, W_CORRECT = 0.2, 0.1, 0.7
MAX_EXAMPLES = 3     # never show more wrong rows than this: the model must not be handed the whole answer
CONTEXT = 2          # clocked levels: how many cycles before the first wrong one to show


@dataclass
class Grade:
    score: float
    solved: bool
    stage: str           # "compile" | "run" | "behaviour" | "solved" -- where it stopped
    feedback: dict       # {"A": ..., "B": ..., "C": ...}, all from the same simulation
    sim: SimResult


def _show(value, name=None, labels=None):
    """A value as feedback C shows it. With LABELS in reference.py (e.g. light: 0 -> RED), the name of the
    value is shown too -- "YELLOW (2)" says more to the model than "2". `labels` may also carry
    "__signed__": {name: width} (from SIGNED in reference.py), so -5 is shown as -5, not 1048571."""
    if value is None:
        return "x (undefined)"
    labels = labels or {}
    width = labels.get("__signed__", {}).get(name)
    if width and value >= 1 << (width - 1):
        value -= 1 << width
    label = labels.get(name, {}).get(value)
    return f"{label} ({value})" if label else str(value)


def _vals(d, labels=None):
    return " ".join(f"{k}={_show(v, k, labels)}" for k, v in d.items())


def _labels(ref):
    """LABELS from reference.py, plus the widths of any SIGNED outputs, for _show."""
    lab = dict(getattr(ref, "LABELS", None) or {})
    signed = getattr(ref, "SIGNED", None)
    if signed:
        lab["__signed__"] = {n: ref.OUTPUTS[n] for n in signed if n in ref.OUTPUTS}
    return lab


def score_of(sim):
    if not sim.compiled.ok:
        return 0.0
    if not sim.ran:
        return W_COMPILES
    steps = len(sim.rows)
    wrong = len({m["step"] for m in sim.mismatches})
    return round(W_COMPILES + W_RUNS + W_CORRECT * (steps - wrong) / steps, 3) if steps else 0.0


def _first_wrong_c(ref, sim):
    """Feedback C for a design that runs but is wrong: WHERE it first goes wrong, as an engineer would
    read it off a waveform -- the inputs, the signal, what it gave, what was expected, and for a clocked
    design the cycles just before. A few examples at most, never the whole table."""
    unit = "cycle" if ref.CLOCKED else "test"
    lab = _labels(ref)
    first = min(m["step"] for m in sim.mismatches)
    lines = []
    if ref.CLOCKED:
        for i in range(max(0, first - CONTEXT), first + 1):
            v, exp, got = sim.rows[i]
            wrong = [n for n in ref.OUTPUTS if got.get(n) != exp[n]]
            if not wrong:
                lines.append(f"  {unit} {i}: inputs {_vals(v, lab)} -> {_vals(got, lab)} (correct)")
            else:
                outs = ", ".join(f"`{n}` = {_show(got.get(n), n, lab)}, expected {_show(exp[n], n, lab)}" for n in wrong)
                lines.append(f"  {unit} {i}: inputs {_vals(v, lab)} -> {outs}   <-- first wrong {unit}")
    else:
        # One example per wrong signal first (so no broken output goes unmentioned), then more rows.
        firsts = {}
        for m in sim.mismatches:
            firsts.setdefault(m["signal"], m)
        picks = list(firsts.values())
        picks += [m for m in sim.mismatches if m not in picks]
        for m in picks[:MAX_EXAMPLES]:
            v, exp, _ = sim.rows[m["step"]]
            lines.append(f"  inputs {_vals(v, lab)} -> `{m['signal']}` = {_show(m['got'], m['signal'], lab)}, "
                         f"expected {_show(exp[m['signal']], m['signal'], lab)}")
    undefined = any(m["got"] is None for m in sim.mismatches)
    note = ("\nAn output is x (undefined): it is never given a value on some path -- check that every "
            "output is assigned, and that reset sets every register." if undefined else "")
    return "\n".join(lines) + note


def _with_d(fb, ref, sim):
    """Feedback D = C plus a diagnosis of the CAUSE, from the level's optional diagnose() in reference.py.

    Added after the main experiment showed that C (where it first goes wrong) did not help on the hard levels:
    the model saw the symptom but not the cause (e.g. every traffic-light phase one cycle too long). D names
    the cause in terms of the spec -- never the code to write. A level without diagnose() gets D = C."""
    d = fb["C"]
    diag = getattr(ref, "diagnose", None)
    if diag and sim.ran and sim.mismatches:
        try:
            text = diag(sim.rows)
        except Exception:
            text = None
        if text:
            d += "\nDiagnosis: " + text.strip()
    fb["D"] = d
    return fb


def feedback_of(ref, sim):
    """The feedback levels for the experiment, from one simulation. Each one adds information:
    A only the verdict; B how much is wrong; C where and how it first goes wrong; D also why."""
    steps = len(sim.rows)
    if sim.passed:
        msg = f"PASS: correct on all {steps} tests."
        return {"A": "PASS.", "B": msg, "C": msg, "D": msg}
    a = "FAIL: the design is not correct."
    if not sim.compiled.ok:
        return _with_d({"A": a,
                "B": f"FAIL: the design does not compile ({len(sim.compiled.messages)} problem(s)).",
                "C": "FAIL: the design does not compile.\n" + "\n".join(f"- {m}" for m in sim.compiled.messages)},
                       ref, sim)
    if not sim.ran:
        return _with_d({"A": a, "B": "FAIL: the design compiles, but the simulation did not run to the end.",
                "C": f"FAIL: the design compiles, but the simulation did not run to the end. {sim.problem}"}, ref, sim)
    wrong = len({m["step"] for m in sim.mismatches})
    signals = sorted({m["signal"] for m in sim.mismatches})
    b = f"FAIL: {wrong} of {steps} tests give a wrong output."
    c = (f"FAIL: {wrong} of {steps} tests give a wrong output, on "
         + ", ".join(f"`{s}`" for s in signals) + ".\n" + _first_wrong_c(ref, sim))
    return _with_d({"A": a, "B": b, "C": c}, ref, sim)


def grade(ref, design_src):
    """Compile, simulate, score, and write all three feedback levels. The one call the agent needs."""
    sim = simulate(ref, design_src)
    stage = ("compile" if not sim.compiled.ok else "run" if not sim.ran
             else "solved" if sim.passed else "behaviour")
    return Grade(score_of(sim), sim.passed, stage, feedback_of(ref, sim), sim)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="[LEVEL_DIR] DESIGN.v")
    ap.add_argument("--module", help="module name, when no level folder is given")
    ap.add_argument("--feedback", choices=["A", "B", "C", "D", "all"], default="all",
                    help="which feedback level to print (default: all three)")
    a = ap.parse_args()

    if len(a.paths) == 2:
        ref = load_level(a.paths[0])
        if hasattr(ref, "vectors") and hasattr(ref, "reference"):
            g = grade(ref, open(a.paths[1]).read())
            print(f"score {g.score:.2f}   stage: {g.stage}   {'SOLVED' if g.solved else 'not solved'}")
            for lvl in ("A", "B", "C", "D") if a.feedback == "all" else (a.feedback,):
                print(f"\n--- feedback {lvl} ---\n{g.feedback[lvl]}")
            return 0 if g.solved else 1
        module, ports, design_path = ref.MODULE, ports_of(ref), a.paths[1]
    else:
        if not a.module:
            ap.error("give a level folder, or --module NAME")
        module, ports, design_path = a.module, None, a.paths[0]

    r = compile_verilog(open(design_path).read(), module, ports)
    if r.ok:
        print(f"COMPILES: `{module}`" + (" with the spec's ports" if ports else ""))
        return 0
    print("DOES NOT COMPILE:")
    for m in r.messages:
        print(f"  - {m}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
