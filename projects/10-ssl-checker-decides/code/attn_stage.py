"""Level 8 (single-head attention) in three frozen stages.

Qwen writes the kernel body in three pieces, each checked on its own and frozen once it passes:
  (A) scores = q @ k.T into a PSUM tile named `scores`
  (B) a numerically stable softmax of scores / sqrt(dim) over the last axis, into SBUF `probs`
  (C) probs @ v, written into the HBM output `result`
The harness writes only the skeleton (imports, def line, `seq, dim = q.shape`, `return result`) and,
for checking A and B, a test tail that copies the intermediate out so it can be compared to NumPy.
The final kernel (skeleton + A + B + C, all the model's lines) is graded by the stock agent.grade.

Stock-grader note: on level 8 both agent.grade and nkibench.verify read case["M"], case["K"],
case["N"] for the roofline after a correct case, and level 8's shapes only carry seq/dim, so a
CORRECT kernel raises KeyError. We add M=seq, K=dim, N=seq to the shape dicts before grading; the
inputs are built from seq/dim alone, so the kernel sees exactly the same data.

  python attn_stage.py --rounds 6 --runs 1
"""
import argparse
import json
import os
import re
import textwrap
import time
import types

import numpy as np

import agent
import nkibench

LEVEL = 8
ENTRY = nkibench.LEVELS[LEVEL]["entry"]

SKEL = """import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def nki_attention_(q, k, v):
    seq, dim = q.shape
    # (A)
    # (B)
    # (C)
    return result
"""

# Harness-only tails that expose an intermediate for checking. Never part of the final kernel.
TAIL = {
    "A": ("_chk = nl.ndarray((seq, seq), dtype=nl.float32, buffer=nl.sbuf)\n"
          "nisa.tensor_copy(dst=_chk, src=scores)\n"
          "result = nl.ndarray((seq, seq), dtype=nl.float32, buffer=nl.shared_hbm)\n"
          "nisa.dma_copy(dst=result, src=_chk)"),
    "B": ("result = nl.ndarray((seq, seq), dtype=nl.float32, buffer=nl.shared_hbm)\n"
          "nisa.dma_copy(dst=result, src=probs)"),
}


def want_A(q, k, v):
    return q.astype(np.float32) @ k.astype(np.float32).T


def want_B(q, k, v):
    s = want_A(q, k, v) / np.sqrt(q.shape[1])
    s = s - s.max(axis=-1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(axis=-1, keepdims=True)


WANT = {"A": want_A, "B": want_B}

API = """Real NKI signatures (keyword arguments):
  nl.ndarray(shape, dtype=nl.float32, buffer=nl.sbuf)       buffer is nl.sbuf, nl.psum or nl.shared_hbm
  nisa.dma_copy(dst=, src=)                                 HBM <-> SBUF only (never PSUM)
  nisa.tensor_copy(dst=, src=)                              e.g. PSUM -> SBUF
  nisa.nc_transpose(dst=, data=)                            data in SBUF, dst a PSUM tile with the two axes swapped
  nisa.nc_matmul(dst=, stationary=, moving=)                stationary [K, M] and moving [K, N] both in SBUF,
                                                            dst [M, N] in PSUM; it contracts over the FIRST
                                                            (partition) axis K: dst = stationary.T @ moving
  nisa.tensor_scalar(dst=, data=, op0=nl.multiply, operand0=)   operand0 a Python float or a (rows, 1) tile
  nisa.tensor_reduce(dst=, op=nl.maximum, data=, axis=1, negate=True)   dst shape (rows, 1)
  nisa.activation(dst=, op=nl.exp, data=, bias=, reduce_op=nl.add, reduce_res=, reduce_cmd=nisa.reduce_cmd.reset_reduce)
  nisa.reciprocal(dst=, data=)"""

STEPS = {
    "A": ("compute the raw scores q @ k.T (no scaling yet) into a PSUM tile named scores with shape "
          "(seq, seq) and dtype nl.float32. Steps: allocate an SBUF tile with q's own shape and dma_copy "
          "q into it; do the same for k. nc_matmul contracts over the partition (first) axis, so both "
          "operands need dim on the partition axis: transpose the q tile with nisa.nc_transpose into a "
          "PSUM tile of shape (dim, seq) and dtype nl.float32, then nisa.tensor_copy it into an SBUF "
          "tile of shape (dim, seq), because nc_matmul operands must be in SBUF. Do the same for the k "
          "tile. Then allocate scores in nl.psum and nisa.nc_matmul into it with the transposed q SBUF "
          "tile as stationary and the transposed k SBUF tile as moving. Each allocation goes directly "
          "above its first use."),
    "B": ("turn scores into softmax probabilities over the last axis of scores / sqrt(dim), in an SBUF "
          "tile named probs with shape (seq, seq) and dtype nl.float32. It must be numerically stable: "
          "subtract the row maximum before exp. Steps: nisa.tensor_scalar with op0=nl.multiply and "
          "operand0 the Python float 1.0 / dim ** 0.5 reads scores and writes an SBUF tile of shape "
          "(seq, seq). nisa.tensor_reduce with op=nl.maximum, axis=1, negate=True writes MINUS the row "
          "max of that scaled tile into an SBUF tile of shape (seq, 1). One nisa.activation with "
          "op=nl.exp, data the scaled tile, bias the negated row max, reduce_op=nl.add, reduce_res an "
          "SBUF tile of shape (seq, 1) and reduce_cmd=nisa.reduce_cmd.reset_reduce writes exp(x - max) "
          "into an SBUF tile of shape (seq, seq) and the row sums into reduce_res. nisa.reciprocal of "
          "the row sums into an SBUF tile of shape (seq, 1), then nisa.tensor_scalar with "
          "op0=nl.multiply and that reciprocal tile as operand0 multiplies the exp tile into probs. "
          "Each allocation goes directly above its first use."),
    "C": ("compute result = probs @ v and write it to the HBM output named result. nc_matmul contracts "
          "over the partition (first) axis, so the keys must be on the partition axis of probs: "
          "transpose probs with nisa.nc_transpose into a PSUM tile of shape (seq, seq) and dtype "
          "nl.float32, then nisa.tensor_copy it into an SBUF tile of shape (seq, seq). Allocate an SBUF "
          "tile with v's own shape and dma_copy v into it. Allocate a PSUM tile of shape (seq, dim) and "
          "dtype nl.float32 and nisa.nc_matmul into it with the transposed probs SBUF tile as "
          "stationary and the v SBUF tile as moving. nisa.tensor_copy that PSUM tile into an SBUF tile "
          "of shape (seq, dim) with dtype q.dtype, allocate result with shape (seq, dim), dtype "
          "q.dtype and buffer=nl.shared_hbm, and nisa.dma_copy the SBUF tile into result. The line "
          "return result is already written. Each allocation goes directly above its first use."),
}

GOAL = ("Write part of an AWS Neuron NKI kernel for single-head attention: "
        "result = softmax(q @ k.T / sqrt(dim)) @ v, with q, k, v all [seq, dim], seq <= 128 and "
        "dim <= 128, so every input is exactly ONE tile: no loops, no reshape, every tile 2-D.")


def clean(block):
    """A stage's lines, dedented. If the model sent a whole kernel, keep only the body, drop the
    imports, the def, the `seq, dim = q.shape` line and `return`."""
    lines = block.splitlines()
    if any(re.match(r"^\s*def ", ln) for ln in lines):
        i = next(i for i, ln in enumerate(lines) if re.match(r"^\s*def ", ln))
        lines = lines[i + 1:]
    lines = [ln for ln in lines if ln.strip()
             and not re.match(r"^\s*(import |from |@nki|return\b|#)", ln)
             and not re.match(r"^\s*seq\s*,\s*dim\s*=\s*q\.shape\s*$", ln)]
    if len(lines) > 1 and not lines[0][:1].isspace() and not lines[0].rstrip().endswith(":"):
        lines = [lines[0]] + textwrap.dedent("\n".join(lines[1:])).splitlines()
    return textwrap.dedent("\n".join(lines)).strip() or "pass"


HARNESS = "v2-shape-hints"

# Elementwise ISA calls whose output tile must have the same shape as the tile they read.
ELEMENTWISE = {"reciprocal": "data", "tensor_scalar": "data", "activation": "data",
               "tensor_copy": "src"}


def shape_hints(code):
    """Name shape mistakes in the stage's own code, using its own tile names.

    Measured on run v1: the model passed reduce_res=nl.ndarray(...) inline (so the row sums had no
    name) and then called reciprocal on the (seq, seq) exp tile into a (seq, 1) tile. The simulator
    says 'could not be broadcast', and the stock message for that is about loop slicing, so the model
    resent the same code. This says which call and which tiles.
    """
    import ast
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ""
    shapes, hints = {}, []

    def nd_shape(call):
        if (isinstance(call, ast.Call) and ast.unparse(call.func) == "nl.ndarray"):
            if call.args:
                return ast.unparse(call.args[0])
            for kw in call.keywords:
                if kw.arg == "shape":
                    return ast.unparse(kw.value)
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            s = nd_shape(node.value)
            if s:
                shapes[node.targets[0].id] = s.replace(" ", "")
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = ast.unparse(node.func)
        kws = {kw.arg: kw.value for kw in node.keywords if kw.arg}
        for arg, val in kws.items():
            if nd_shape(val):
                hints.append(f"In {fn}, {arg}= is a fresh nl.ndarray written inline, so nothing can "
                             f"read it afterwards. Allocate it on its own line above the call with a "
                             f"name, for example exp_sum = {ast.unparse(val)}, pass {arg}=exp_sum, and "
                             f"use exp_sum wherever those values are needed next.")
        leaf = fn.split(".")[-1]
        if fn.startswith("nisa.") and leaf in ELEMENTWISE and "dst" in kws:
            src = kws.get(ELEMENTWISE[leaf])
            d = kws["dst"]
            if isinstance(src, ast.Name) and isinstance(d, ast.Name):
                sd, ss = shapes.get(d.id), shapes.get(src.id)
                if sd and ss and sd != ss:
                    extra = ""
                    if leaf == "reciprocal":
                        extra = (" The reciprocal must read the (seq, 1) row sums that nisa.activation "
                                 "wrote into its reduce_res tile, not the (seq, seq) exp tile.")
                    hints.append(f"{fn}(dst={d.id}, {ELEMENTWISE[leaf]}={src.id}) writes a tile of "
                                 f"shape {sd} from a tile of shape {ss}; an elementwise call needs "
                                 f"both the same shape.{extra}")
    return " ".join(hints)


def build(parts):
    """parts: dict stage -> code. Missing stages are dropped from the skeleton."""
    out = []
    for line in SKEL.splitlines():
        m = re.match(r"^(\s*)# \((A|B|C|T)\)$", line)
        if m:
            if m.group(2) in parts:
                out.append(textwrap.indent(parts[m.group(2)], m.group(1)))
            continue
        out.append(line)
    return "\n".join(out) + "\n"


def stage_source(frozen, stage, code):
    parts = dict(frozen)
    parts[stage] = code
    if stage in TAIL:
        # The tail goes where (C) would go.
        parts["C"] = TAIL[stage]
    return build(parts)


def check_stage(stage, src):
    """Simulate a stage test kernel on all level-8 shapes and compare its intermediate to NumPy."""
    v = nkibench.check_rules(src, LEVEL)
    if v:
        return False, "Rule violations: " + " ".join(v)
    try:
        compile(src, "<stage>", "exec")
    except SyntaxError as e:
        return False, f"The code does not parse: {e.msg} on line {e.lineno}."
    path = f"/tmp/_attn_stage_{stage}.py"
    with open(path, "w") as f:
        f.write(src)
    try:
        kernel = nkibench.load_kernel(path, ENTRY)
    except Exception as e:
        return False, agent.enrich(f"The file could not be loaded: {type(e).__name__}: {e}", src)
    for case in nkibench.LEVELS[LEVEL]["shapes"]:
        args, _ = nkibench.make_inputs(case, LEVEL)
        want = WANT[stage](*args)
        lbl = nkibench.label(case, LEVEL)
        try:
            got, _ = nkibench.simulate_and_count(kernel, args)
        except Exception as e:
            return False, f"On {lbl}: " + agent.enrich(f"raised {type(e).__name__}: {e}", src)
        m = nkibench.describe_mismatch(got, want)
        if m:
            what = "scores (q @ k.T)" if stage == "A" else "probs (softmax of scores / sqrt(dim))"
            return False, f"On {lbl}, {what} is wrong: {m}"
    return True, "correct on every shape"


def patch_shapes():
    for c in nkibench.LEVELS[LEVEL]["shapes"]:
        c.setdefault("M", c["seq"])
        c.setdefault("K", c["dim"])
        c.setdefault("N", c["seq"])


def prompt_for(frozen, stage):
    shown = dict(frozen)
    for s in "ABC":
        if s not in shown:
            shown[s] = f"# ({s})" + ("  <-- write this part" if s == stage else "  -- a later step")
    done = ", ".join(f"({s})" for s in frozen)
    note = (f"{done} {'is' if len(frozen) == 1 else 'are'} done, checked and frozen: do not repeat "
            f"{'it' if len(frozen) == 1 else 'them'}. " if frozen else "")
    return (f"{GOAL}\n\n```python\n{build(shown)}```\n\n{note}Write ONLY the code for ({stage}): "
            f"{STEPS[stage]}\n\n{API}\n\nReply with ONE python code block containing only the lines "
            f"for ({stage}): no imports, no def, no return.")


def repair_for(frozen, stage, code, feedback):
    return (f"{GOAL}\n\nYour code for ({stage}) is not right yet:\n\n```python\n{code}\n```\n\n"
            f"The task for ({stage}) was: {STEPS[stage]}\n\nA checker reports:\n{feedback}\n\n"
            f"{API}\n\nChange what the checker names and keep the rest. Reply with ONE python code "
            f"block: only the lines for ({stage}), no imports, no def, no return.")


def run_once(a, run, log):
    frozen, history = {}, []
    for stage in "ABC":
        prompt, solved = prompt_for(frozen, stage), False
        for rnd in range(1, a.rounds + 1):
            t0 = time.perf_counter()
            reply = agent.ask_parallel(a, prompt, 1)[0]
            code = clean(agent.extract_code(reply))
            if stage == "C":
                src = build({**frozen, "C": code})
                reward, parts, feedback = agent.grade(src, LEVEL)
                ok = bool(parts.get("correct"))
            else:
                src = stage_source(frozen, stage, code)
                ok, feedback = check_stage(stage, src)
                reward = None
            if not ok:
                h = shape_hints(code)
                if h:
                    feedback = "Fix first: " + h + "\n(simulator said: " + feedback + ")"
            dt = time.perf_counter() - t0
            print(f"run {run} stage {stage} round {rnd}: {'PASS' if ok else 'fail'}"
                  f"{'' if reward is None else f'  reward={reward:.2f}'}  ({dt:.1f}s)")
            if not ok:
                print(textwrap.indent(feedback[:600], "    "))
            log.write(json.dumps(dict(level=LEVEL, harness=HARNESS, run=run, stage=stage, round=rnd, ok=ok,
                                      reward=reward, feedback=feedback[:2000], code=code,
                                      source=src, seconds=round(dt, 1))) + "\n")
            log.flush()
            if ok:
                frozen[stage] = code
                history.append((stage, rnd))
                solved = True
                if stage == "C":
                    with open(a.kernel_out, "w") as f:
                        f.write(src)
                    print(f"  LEVEL 8 SOLVED (stock agent.grade reward {reward:.2f}); kernel -> "
                          f"{a.kernel_out}")
                break
            prompt = repair_for(frozen, stage, code, feedback)
        if not solved:
            print(f"run {run}: stuck at stage {stage} after {a.rounds} rounds")
            return dict(run=run, solved=False, stuck=stage, passed=history)
    return dict(run=run, solved=True, passed=history)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--log", default="logs/attempts-rishabh-attention.jsonl")
    p.add_argument("--kernel-out", default="logs/rishabh-l8-kernel.py")
    p.add_argument("--max-tokens", type=int, default=2500)
    args = p.parse_args()
    a = types.SimpleNamespace(model=agent.MODEL,
                              base=os.environ.get("KERNEL_AGENT_BASE_URL", "http://localhost:8000/v1"),
                              max_tokens=args.max_tokens, context=8192, think=False,
                              rounds=args.rounds, kernel_out=args.kernel_out)
    patch_shapes()
    os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
    results = []
    with open(args.log, "a") as log:
        for run in range(1, args.runs + 1):
            results.append(run_once(a, run, log))
    print("\nSUMMARY:", json.dumps(results))


if __name__ == "__main__":
    main()
