#!/usr/bin/env python3
"""
prompt.py - the prompt -> code step: model takes in a request to generate
applicable Verilog code to a given question or given verilog code, and fixes it
from the verify step's feedback

    prompt.py   request -> model -> designs/<module>/design.v + spec.txt
    verify.py   writes tb.v + report.json; FAIL -> for_feedback.json, PASS -> for_optimize.json
    prompt.py   --fix reads for_feedback.json -> model -> new design.v -> run verify.py again

example prompts:
  python prompt.py "8-bit ALU with add, sub, and, or, xor"
  python prompt.py "add an overflow flag" --code alu.v --spec "8-bit ALU ... with an overflow flag"
  python prompt.py                                   -> asks interactively
  python verify.py designs/alu8                      -> the testbench step
  python prompt.py --fix designs/alu8                -> fix from designs/alu8/for_feedback.json

Every model answer is appended to a JSONL file (prompt, answer, code, folder) for an RL trainer;
its reward is the score verify.py writes to <folder>/report.json.
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.request

URL = os.environ.get("VAGENT_URL", "http://localhost:8000/v1/chat/completions")
MODEL = os.environ.get("VAGENT_MODEL", "Qwen/Qwen3-8B")

SYSTEM = """You are an expert digital circuit designer. The user tells you what circuit they want,
sometimes in a few words ("an ALU", etc.), sometimes as a precise spec, sometimes more open-ended and conceptual,
and sometimes with existing Verilog they want changed. You reply with one correct Verilog module that does what they asked.

When the request is vague, choose sensible, conventional defaults (e.g. 8-bit datapath,
clk + synchronous active-high rst for sequential logic, common operations for an ALU) rather
than asking questions. When the user names a module, ports, widths, opcodes or reset behaviour,
use exactly what they said.

Start the module with a comment block that states:
  - what the circuit does
  - every port: name, direction, width, meaning
  - any encoding (opcodes, states) and every assumption you made that the user did not specify

Verilog rules (the code is compiled with Icarus Verilog, iverilog -g2012, and synthesized with Yosys):
- Plain synthesizable Verilog-2005: no SystemVerilog classes, interfaces, packages or assertions.
- Sequential logic: always @(posedge clk) with nonblocking <= only.
- Combinational logic: always @(*) with blocking = only, or assign. Give every output a value
  on every path (a default before the case/if, or a default: branch) so no latch is inferred.
- Declare widths explicitly and size constants (4'd0, not 0). Keep carry/overflow bits:
  extend operands before adding when the result needs the extra bit.
- An output assigned inside an always block is declared reg; one driven by assign is a wire.
- No initial blocks, # delays, $display or $finish in the design.
- Reply with exactly one module
"""


# ---------------- model ----------------
def ask(user, system=SYSTEM):
    """One chat request to the model server (vLLM in the seat pod). Returns the reply text."""
    body = {"model": MODEL, "max_tokens": 2000, "temperature": 0.7,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)["choices"][0]["message"]["content"] or ""


def extract(reply):
    """The Verilog in the reply: the last ```verilog block (or the raw text), minus any
    testbench (module tb) the model added, which verify.py would see as a second design."""
    txt = re.sub(r"<think>.*?</think>", "", reply, flags=re.S)
    blocks = re.findall(r"```(?:verilog|systemverilog|sv|v)?\s*\n(.*?)```", txt, re.S)
    code = blocks[-1] if blocks else txt
    code = re.sub(r"^\s*module\s+tb\b.*?\bendmodule\b", "", code, flags=re.S | re.M)
    return code.strip() + "\n"


def module_name(code):
    """Name of the module; only a 'module' that starts a line, so comments don't count."""
    m = re.search(r"^\s*module\s+([A-Za-z_]\w*)", code, re.M)
    return m[1] if m else None


def header(code):
    """'module name(...ports...);' - verify.py builds its testbench against this."""
    m = re.search(r"^\s*(module\s+\w+\s*(?:#\s*\([^;]*?\)\s*)?\([^)]*\))", code, re.S | re.M)
    return (m[1] + ";") if m else ""


# ---------------- prompts ----------------
def build_prompt(request, code=None):
    """The user message: a new circuit from a description, or a change to existing code."""
    if code is None:
        return f"""Design this circuit:

{request}"""
    return f"""Here is my current Verilog:
```verilog
{code.rstrip()}
```

{request}

Keep the module name and existing ports unless my request says to change them.
Return the complete updated module."""


def fix_prompt(spec, code, fb, hints=None):
    """From verify.py's for_feedback.json: the spec, the failing design, and what the test saw.
    hints: the feedback step's suggestions (feedback_result.json), if it ran."""
    lines = fb.get("mismatches") or fb.get("result", "").splitlines()
    shown = "\n".join(lines[:10])
    if hints:
        shown += "\n\nDiagnosis from the feedback step:\n" + "\n".join(f"- {h}" for h in hints[:5])
    return build_prompt(f"""It must do this:
{spec}

A testbench checked it and it FAILED (score {fb.get('score', 0):.3f}). What the test saw:
{shown}

Find the cause in the code and fix it. Do not special-case the tested input values.""", code)


# ---------------- hand-off folder ----------------
def folder_for(root, name, spec):
    """designs/<module>/. If that folder already holds a DIFFERENT spec, use <module>_2, _3...
    so verify.py doesn't reuse a tb.v written for another circuit."""
    n = 1
    while True:
        d = os.path.join(root, name if n == 1 else f"{name}_{n}")
        sp = os.path.join(d, "spec.txt")
        if not os.path.exists(sp) or open(sp).read().strip() == spec.strip():
            return d
        n += 1


def log_attempt(path, **rec):
    with open(path, "a") as f:
        f.write(json.dumps(dict(time=int(time.time()), system=SYSTEM, **rec)) + "\n")


def generate(a):
    """New design (or a change to --code) -> designs/<module>/design.v + spec.txt."""
    request = " ".join(a.request).strip() or input("What circuit do you want?\n> ").strip()
    if not request:
        sys.exit("nothing requested")
    code = open(a.code).read() if a.code else None

    prompt = build_prompt(request, code)
    print("solving...", flush=True)
    t0 = time.time()
    try:
        reply = ask(prompt)
    except Exception as ex:
        sys.exit(f"request failed: {ex}")
    design = extract(reply)
    name = module_name(design)
    if not name:
        log_attempt(a.log, kind="design", folder=None, request=request, prompt=prompt,
                    answer=reply, code=design, seconds=round(time.time() - t0, 1))
        sys.exit("the model's reply has no Verilog module; nothing written (attempt logged)")

    # spec.txt is what verify.py writes the testbench from: the behaviour + the exact header
    spec = (a.spec or request) + "\n\nModule header (instantiate exactly): " + header(design)
    d = folder_for(a.designs, name, spec)
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "design.v"), "w").write(design)
    open(os.path.join(d, "spec.txt"), "w").write(spec + "\n")
    log_attempt(a.log, kind="design", folder=d, request=request, prompt=prompt,
                answer=reply, code=design, seconds=round(time.time() - t0, 1))

    print("-" * 40 + "\n" + design + "-" * 40)
    print(f"wrote {d}/design.v and spec.txt")
    print(f"next: python verify.py {d}")
    return d


def fix(a):
    """--fix designs/<name>: read verify.py's for_feedback.json, ask for a fix, overwrite design.v."""
    d = a.fix.rstrip("/\\")
    p = lambda f: os.path.join(d, f)
    if os.path.exists(p("for_review.json")):
        sys.exit(f"{d}: verify.py suspects its own testbench (for_review.json) - "
                 "a human should decide before the design is 'fixed'")
    if not os.path.exists(p("for_feedback.json")):
        rep = json.load(open(p("report.json"))) if os.path.exists(p("report.json")) else None
        if rep and rep.get("passed"):
            sys.exit(f"{d}: already PASSES (report.json) - nothing to fix")
        sys.exit(f"{d}: no for_feedback.json - run: python verify.py {d}")

    fb = json.load(open(p("for_feedback.json")))
    spec = fb.get("spec") or open(p("spec.txt")).read()
    old = fb.get("design") or open(p("design.v")).read()
    hints = []
    if os.path.exists(p("feedback_result.json")):
        try:
            hints = json.load(open(p("feedback_result.json"))).get("suggestions") or []
        except Exception:
            hints = []
    prompt = fix_prompt(spec, old, fb, hints)
    print(f"fixing {d} (score {fb.get('score', 0):.3f})...", flush=True)
    t0 = time.time()
    try:
        reply = ask(prompt)
    except Exception as ex:
        sys.exit(f"request failed: {ex}")
    design = extract(reply)
    log_attempt(a.log, kind="fix", folder=d, request=spec, prompt=prompt, answer=reply,
                code=design, previous_score=fb.get("score"), seconds=round(time.time() - t0, 1))
    if module_name(design) != module_name(old):
        sys.exit(f"the fix renamed the module ({module_name(old)} -> {module_name(design)}); "
                 "design.v left unchanged (attempt logged)")

    open(p("design_prev.v"), "w").write(old)
    open(p("design.v"), "w").write(design)
    os.remove(p("for_feedback.json"))     # used up; verify.py writes a fresh one if it still fails
    if os.path.exists(p("feedback_result.json")):
        os.remove(p("feedback_result.json"))
    print("-" * 40 + "\n" + design + "-" * 40)
    print(f"wrote {d}/design.v (previous kept as design_prev.v)")
    print(f"next: python verify.py {d}")
    return d


def main():
    ap = argparse.ArgumentParser(description="Describe a circuit, get Verilog for the verify step.")
    ap.add_argument("request", nargs="*", help="what you want, e.g. '4-bit ALU with add/sub/and/or'")
    ap.add_argument("--code", help="existing Verilog file to modify instead of starting fresh")
    ap.add_argument("--spec", help="full behaviour for spec.txt (default: the request). "
                                   "Use it with --code, where the request is only the change")
    ap.add_argument("--fix", metavar="FOLDER", help="fix designs/<name> from its for_feedback.json")
    ap.add_argument("--designs", default="designs", help="hand-off folder root (default: designs)")
    ap.add_argument("--log", default="attempts.jsonl")
    a = ap.parse_args()
    fix(a) if a.fix else generate(a)


if __name__ == "__main__":
    main()