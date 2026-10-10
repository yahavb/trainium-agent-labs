#!/usr/bin/env python3
"""
agent.py — the loop: a model writes an NKI kernel, the harness grades it, the reason goes back,
it tries again.

    controller  picks a level                          nkibench.py
    generator   the model writes a kernel             the shared gpt-oss endpoint
    checker     rules, then simulate, then roofline   nkibench.py
    loop        the failure becomes the next prompt   this file

Everything here runs on the CPU, so it does NOT need the Trainium device -- which is the point,
because the device is what Project 2 needs later for real timings, and a Neuron device cannot be
shared by two processes. The model therefore comes from the shared endpoint, not from a server on
your own chip.

    export GPTOSS_BASE_URL="https://..."
    python agent.py --level 1
    python agent.py --level 4 --rounds 6 --samples 4
    python agent.py --all
    python agent.py --offline --level 4        # no model; replays the reference kernel

Every attempt is appended to a JSONL file with its reward, so the log is the deliverable.
"""

import argparse
import ast
import hashlib
import inspect
import json
import os
import re
import sys
import tempfile
import textwrap
import time

import numpy as np

import nkibench

MODEL = os.environ.get("KERNEL_AGENT_MODEL", "Qwen/Qwen3-8B")

# The model writes to a hidden reasoning channel before it writes any answer. Measured on this
# endpoint: a coding task burned 900 tokens thinking and returned EMPTY content. See gptoss/README.
MIN_ANSWER_TOKENS = 2500

REASONING_KEYS = ("reasoning", "reasoning_content")


# ---------------------------------------------------------------- reward
#
# Graded, not pass/fail, so a near miss is distinguishable from nonsense and the loop has
# something to climb. Matches Project 1's shape: correctness dominates, and nothing else counts
# until the kernel is right.

WEIGHTS = dict(parses=0.1, rules=0.2, runs=0.2, correct=0.5)


def grade(source, level):
    """Returns (reward, parts, feedback, progress). Feedback is an INSTRUCTION, never just a verdict.

    progress breaks ties between samples with the same reward, so the loop repairs the attempt that
    got furthest. Measured on level 3: all four samples scored 0.30 every round, and the repair
    always went to whichever came back first -- once a kernel failing on its first line, while
    another sample had loaded both operands correctly. 0..1 is how far into the kernel the simulator
    got before raising; 1..2 means it ran, plus the fraction of the output that is right.
    """
    parts = dict(parses=False, rules=False, runs=False, correct=False)

    if not source.strip():
        return 0.0, parts, ("No code came back. Reply with one python code block containing the "
                            "kernel and nothing else."), 0.0
    try:
        compile(source, "<candidate>", "exec")
        parts["parses"] = True
    except SyntaxError as e:
        return (WEIGHTS["parses"] * 0, parts,
                f"The code does not parse: {e.msg} on line {e.lineno}. Send one complete python "
                f"code block.", 0.0)

    violations = nkibench.check_rules(source, level)
    if violations:
        extra = ""
        if any("no function named" in v for v in violations):
            # Measured: this repeated 15 rounds running, because "there is no function named X"
            # never said what the function should look like. Hand over the exact line.
            ref = nkibench.LEVELS[level]["ref"]
            args = ", ".join(inspect.signature(ref).parameters)
            extra = (f" Start the function with exactly this line:  "
                     f"def {nkibench.LEVELS[level]['entry']}({args}):  "
                     f"and put @nki.jit on the line above it.")
        return (sum(WEIGHTS[k] for k, v in parts.items() if v), parts,
                "Rule violations, which score zero however fast the kernel is. Fix exactly "
                "these: " + " ".join(violations) + extra, 0.0)
    parts["rules"] = True

    spec = nkibench.LEVELS[level]
    # One file per distinct kernel. Every candidate used to go to the same /tmp path, and Python
    # reuses a cached .pyc when the source has the same size and the same mtime second -- so a
    # same-length repair graded within a second of the previous sample (swapping stationary= and
    # moving=, or (N, M) for (M, N)) was silently graded as the OLD code. Two agents running at
    # once also overwrote each other's file.
    path = os.path.join(_scratch_dir(), f"level{level}_"
                        f"{hashlib.sha1(source.encode()).hexdigest()[:16]}.py")
    with open(path, "w") as f:
        f.write(source)
    try:
        kernel = nkibench.load_kernel(path, spec["entry"])
    except ModuleNotFoundError as e:
        # Two very different causes, and conflating them aborted a whole run: either the SDK is
        # absent (an environment problem the model cannot fix), or the model imported a module it
        # invented -- observed: `import nki.nl`. Tell them apart by asking whether nki itself is
        # importable.
        try:
            import nki  # noqa: F401
            sdk_present = True
        except ImportError:
            sdk_present = False
        if not sdk_present:
            raise SystemExit(
                f"cannot import {e.name!r}, so no NKI kernel can be loaded here. Run this where "
                f"the Neuron SDK exists -- k8s/kernel-agent-job.yaml does that -- or use "
                f"--offline to exercise the loop without a kernel ever running.") from e
        return (sum(WEIGHTS[k] for k, v in parts.items() if v), parts,
                f"There is no module named {e.name!r}. The only imports that exist are: "
                f"`import nki`, `import nki.language as nl`, and `import nki.isa as nisa`. "
                f"Use exactly those three.", 0.0)
    except Exception as e:
        return (sum(WEIGHTS[k] for k, v in parts.items() if v), parts,
                f"The file imports but {spec['entry']} could not be loaded: "
                f"{type(e).__name__}: {e}", 0.0)

    failures, passed, intensity, progress = [], 0, None, []
    for case in spec["shapes"]:
        args, _ = nkibench.make_inputs(case, level)
        before = [x.copy() if isinstance(x, np.ndarray) else x for x in args]
        want = spec["ref"](*args)
        try:
            got, counted = nkibench.simulate_and_count(kernel, args)
        except nkibench.NkiMissing as e:
            return (sum(WEIGHTS[k] for k, v in parts.items() if v), parts,
                    f"CANNOT SIMULATE: {e}", 0.0)
        except Exception as e:
            msg, got_to = explain_exception(e, path, source, level, np.shape(want))
            failures.append((nkibench.label(case, level), msg))
            progress.append(got_to)
            continue
        parts["runs"] = True
        m = (nkibench.check_inputs_untouched(before, args)
             or nkibench.describe_mismatch(got, want)
             or nkibench.check_traffic_bar(level, counted, args, want))
        # A simulator warning about a hardware-correctness hazard counts as a failure even when the
        # numbers happen to match on CPU: the kernel would be wrong on the device.
        hazards = [w for w in counted.get("warnings", [])
                   if "incorrect results on hardware" in w]
        if hazards and not m:
            m = ("CORRECT ON CPU BUT WRONG ON HARDWARE: " + hazards[0]
                 + ". Fix that before anything else -- the simulator agrees with the reference here "
                   "and the device would not.")
        if m:
            failures.append((nkibench.label(case, level), m))
            progress.append(1.0 + fraction_right(got, want))
            continue
        progress.append(2.0)
        passed += 1
        if level >= 3 and counted["bytes"]:
            intensity = nkibench.roofline(
                nkibench.matmul_flops(case["M"], case["K"], case["N"]), counted["bytes"])

    if failures:
        lbl, first = failures[0]
        return (sum(WEIGHTS[k] for k, v in parts.items() if v)
                + WEIGHTS["correct"] * passed / len(spec["shapes"]), parts,
                f"{passed} of {len(spec['shapes'])} shapes passed. On {lbl}: {first}",
                sum(progress) / len(progress))

    parts["correct"] = True
    reward = sum(WEIGHTS.values())
    note = "Correct on every shape."
    if intensity:
        note += " " + nkibench.explain_roofline(intensity)
    return reward, parts, note, 2.0


_SCRATCH = []


def _scratch_dir():
    if not _SCRATCH:
        _SCRATCH.append(tempfile.mkdtemp(prefix="kernel_agent_"))
    return _SCRATCH[0]


def fraction_right(got, want, tol=2e-2):
    """Share of output elements within tolerance; 0 when the shape is wrong."""
    got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
    if got.shape != want.shape:
        return 0.0
    scale = float(np.sqrt((want ** 2).mean())) or 1.0
    with np.errstate(invalid="ignore"):
        return float((np.abs(got - want) / scale <= tol).mean())


# ---------------------------------------------------------------- prompting

# Every name here appears in the three shipped tutorial kernels, so none of it is invented. The
# model does not know this API and guesses plausible names -- nl.scalar, nl.value, nl.dot,
# tile.mean -- none of which exist. A short card of what IS real costs ~200 tokens and is
# documentation rather than the answer.
API_CARD = """Available NKI functions:

  @nki.jit                                  decorate the entry point
  nl.ndarray(shape, dtype=..., buffer=b)    allocate; b is nl.sbuf, nl.psum or nl.shared_hbm
  nl.affine_range(n)                        the loop
  nl.sum(view, axis=[i, j])                 reduce; axis is a list
  nl.float32, nl.bfloat16                   dtypes
  nisa.dma_copy(dst=, src=)                 move data between HBM and SBUF
  nisa.nc_matmul(dst=, stationary=, moving=)   matmul into a PSUM tile
  nisa.tensor_copy(dst=, src=)              copy, e.g. PSUM to SBUF
  nisa.tensor_scalar(dst=, data=, op0=nl.multiply, operand0=0.5)   scale by a constant
  tile.ap([[stride, count], ...])           a strided view, for reductions

nisa.nc_matmul has strict memory rules: dst must live in nl.psum, while stationary and moving must
both live in nl.sbuf. So the pattern is: dma_copy both operands from HBM into sbuf tiles, allocate a
psum tile for the result, call nc_matmul(dst=psum_tile, stationary=..., moving=...), then
tensor_copy from psum into an sbuf tile, then dma_copy that out to the shared_hbm output.
The left operand arrives already transposed, with K on the partition axis.

Slice tiles with ranges, e.g. a[0:128, 0:64]. A complete kernel looks like this:

import nki
import nki.isa as nisa
import nki.language as nl

@nki.jit
def copy_kernel(a):
    out = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.shared_hbm)
    tile = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=tile, src=a)
    nisa.dma_copy(dst=out, src=tile)
    return out

"""


# What nc_matmul DOES, as shapes. Only the matmul levels get this, so levels 1 and 2 see exactly the
# prompt they were measured with. Measured on level 3 before it existed: the memory rules above were
# followed and the shapes were not. Samples sized the output as lhsT.shape[1:] -- a 1-D (64,) -- or
# poured both operands into one (128, 512) tile, and never once got past the first copy. The
# reference's docstring said "-> [M, N]" and that was not enough: it never said which operand
# supplies M and which N.
MATMUL_CARD = """How nc_matmul uses shapes: stationary is [K, M] and moving is [K, N], two sbuf tiles that share
the partition axis K (K <= 128, M <= 128, N <= 512). dst is [M, N], a float32 psum tile, and
receives stationary.T @ moving. lhsT already is [K, M] and rhs is [K, N], so load each into its OWN
sbuf tile of exactly its own shape. The result, and the shared_hbm output you return, is
(M, N) = (lhsT.shape[1], rhs.shape[1]). PSUM never goes straight to HBM: tensor_copy the psum tile
into a new (M, N) sbuf tile, then dma_copy that sbuf tile to the output."""


def is_matmul(level):
    return nkibench.LEVELS[level]["ref"] is nkibench.ref_matmul


def matmul_card(level):
    shapes = ", ".join(nkibench.label(c, level) for c in nkibench.LEVELS[level]["shapes"])
    return f"{MATMUL_CARD}\nIt is tested on: {shapes}.\n\n"


def available_names(dotted):
    """Turn 'no attribute X' into 'here are the real ones'.

    Measured: told only `module 'nki.language' has no attribute 'value'`, the model guessed
    another invented name every round -- value, scalar, sbuf_scalar, dot. A verdict names the
    mistake and never the fix, so list what actually exists and let it choose.
    """
    import difflib
    import importlib
    mod_name, _, attr = dotted.rpartition(".")
    try:
        mod = importlib.import_module(mod_name)
    except Exception:
        return ""
    names = [n for n in dir(mod) if not n.startswith("_")]
    close = difflib.get_close_matches(attr, names, n=6, cutoff=0.4)
    if close:
        return (f" `{mod_name}` has no `{attr}`. The closest real names are: "
                f"{', '.join(close)}. Pick one of those or use a different approach.")
    return (f" `{mod_name}` has no `{attr}`, and nothing similar exists. Its real names include: "
            f"{', '.join(sorted(names)[:25])}.")


def real_signature(func_name):
    """The actual signature of an NKI function, for when the model invents arguments."""
    for mod_name in ("nki.language", "nki.isa", "nki"):
        try:
            mod = __import__(mod_name, fromlist=["x"])
        except Exception:
            continue
        fn = getattr(mod, func_name, None)
        if fn is None:
            continue
        try:
            return f"{mod_name.split('.')[-1]}.{func_name}{inspect.signature(fn)}"
        except (TypeError, ValueError):
            return f"{mod_name.split('.')[-1]}.{func_name}"
    return ""


def enrich(error_text):
    """Add the real names when the failure is an invented API call."""
    if "'MemoryRegion' object is not callable" in error_text:
        return (error_text + " nl.sbuf, nl.psum and nl.shared_hbm are memory regions, not "
                "functions. Do not call them. Allocate with "
                "nl.ndarray(shape, dtype=nl.float32, buffer=nl.sbuf) and pass the region as the "
                "buffer= argument.")
    m = re.search(r"(\w+)\(\) got an unexpected keyword argument '(\w+)'", error_text)
    if m:
        sig = real_signature(m.group(1))
        return (error_text + f" Remove the `{m.group(2)}=` argument."
                + (f" The real signature is {sig}." if sig else ""))
    if "unsupported operand type(s) for" in error_text and "NkiTensor" in error_text:
        return (error_text + " A tile is not a number, so Python operators like += do not work on "
                "one. Accumulate by allocating a PSUM tile with "
                "nl.ndarray(shape, nl.float32, buffer=nl.psum) and letting nisa.nc_matmul add into "
                "it across the loop, or combine two tiles with a nisa op rather than a Python "
                "operator.")
    m = re.search(r"dma_copy requires src and dst to have the same number of elements, "
                  r"got src=(\d+), dst=(\d+)", error_text)
    if m:
        src, dst = int(m.group(1)), int(m.group(2))
        return (error_text + f" The tile you allocated holds {dst} elements but you copied {src} "
                f"into it. nisa.dma_copy does not slice or broadcast: allocate the destination with "
                f"EXACTLY the shape of the slice you are moving. If you want a 128x512 piece of a "
                f"bigger tensor, write "
                f"t = nl.ndarray((128, 512), dtype=a.dtype, buffer=nl.sbuf) and then "
                f"nisa.dma_copy(dst=t, src=a[0:128, 0:512]) -- the slice on the right must have the "
                f"same shape as the tile on the left.")
    m = re.search(r"dma_copy (\w+) partition dimension (\d+) exceeds maximum (\d+)", error_text)
    if m:
        which, got, mx = m.group(1), int(m.group(2)), int(m.group(3))
        return (error_text + f" A tile may have at most {mx} rows, and you asked for {got}. Do not "
                f"allocate one tile for the whole tensor: loop over the partition dimension in "
                f"chunks of at most {mx} with nl.affine_range, allocate the tile inside the loop with "
                f"the chunk's own size, and copy one chunk at a time, e.g. "
                f"src=a[i*{mx}:(i+1)*{mx}, :]. If a dimension is already {mx} or smaller, use it "
                f"whole -- do NOT pad it up to {mx}, that reads past the end of the tensor. The same "
                f"applies to where you write the result back.")
    m = re.search(r"value array of shape \((\d+),?\) could not be broadcast to "
                  r"indexing result of shape \((\d+),?\)", error_text)
    if m:
        val, dst = int(m.group(1)), int(m.group(2))
        return (error_text + f" You assigned {val} elements into a slice that holds {dst}. Assignment "
                f"does not reshape or broadcast either: the slice on the left and the value on the "
                f"right must have the SAME shape. If the value is bigger, you are writing a whole tile "
                f"where a slice belongs -- index the destination to match, e.g. "
                f"out[i*128:(i+1)*128, :] = tile. If it is smaller, you are looping over the wrong "
                f"dimension.")
    m = re.search(r"Out-of-bound access for tensor .*? on dimension (\d+): "
                  r"index range \[(\d+), (\d+)\] exceed dimension size of (\d+)",
                  error_text)
    if m:
        dim, hi, size = m.group(1), int(m.group(3)), int(m.group(4))
        return (error_text + f" You indexed up to {hi} on dimension {dim}, which is only {size} "
                f"long. Tile limits are a MAXIMUM, not a target. Derive every bound from the tensor's "
                f"own shape -- use min(limit, size) and let the final chunk be partial -- rather than "
                f"writing a fixed number. Note the two limits differ: the partition dimension (first) "
                f"allows at most 128, the free dimension allows more.")
    m = re.search(r"Matmul contraction dimension (\d+) exceeds pmax=(\d+)", error_text)
    if m:
        k, mx = int(m.group(1)), int(m.group(2))
        return (error_text + f" The contraction dimension K is {k} and one nc_matmul can only "
                f"contract {mx}. Split K into chunks of {mx} and accumulate: allocate ONE psum tile "
                f"OUTSIDE the K loop, call nisa.nc_matmul into that same psum tile once per chunk so "
                f"the partial products add up there, and only after the loop copy it out with "
                f"nisa.tensor_copy. Do not allocate a new psum tile per chunk and do not write partial "
                f"results to HBM.")
    m = re.search(r"(\w+) (?:dst|src)? ?must be in \['sbuf', 'psum'\], got shared_hbm", error_text)
    if m:
        return (error_text + f" `nisa.{m.group(1)}` only moves data between on-chip buffers, sbuf and "
                f"psum. To reach HBM -- the tensor you allocated with buffer=nl.shared_hbm and will "
                f"return -- use nisa.dma_copy instead. The usual sequence is nc_matmul into psum, "
                f"tensor_copy psum to sbuf, then dma_copy sbuf to the shared_hbm output.")
    m = re.search(r"(\w+) must be in \['(\w+)'\], got (\w+)", error_text)
    if m:
        which, needed, got = m.groups()
        place = {"psum": "nl.psum", "sbuf": "nl.sbuf"}.get(needed, needed)
        return (error_text + f" Allocate the `{which}` tile with buffer={place} instead of "
                f"nl.{got}. For nisa.nc_matmul: dst must be in nl.psum, and stationary and moving "
                f"must both be in nl.sbuf. Copy between them with nisa.tensor_copy.")
    if "got multiple values for argument" in error_text:
        return (error_text + " Pass every argument by keyword, e.g. "
                "nisa.nc_matmul(dst=..., stationary=..., moving=...), so none is bound twice.")
    if "must have at least 2 dimensions" in error_text:
        return (error_text + " Every SBUF and PSUM tile needs two dimensions: a partition dimension "
                "first, then a free dimension. A 1-D tile is not allowed, so write "
                "nl.ndarray((rows, cols), ...) and give a length-N vector the shape (1, N) or "
                "(N, 1) depending on which axis you are reducing over.")
    if "cannot reshape array of size" in error_text:
        return (error_text + " Do not reshape. Work with the shapes you were given and slice "
                "them into tiles, e.g. src=a[0:128, 0:64].")
    m = re.search(r"module '([\w.]+)' has no attribute '(\w+)'", error_text)
    if m:
        return error_text + available_names(f"{m.group(1)}.{m.group(2)}")
    m = re.search(r"'(\w+)' object has no attribute '(\w+)'", error_text)
    if m:
        return (error_text + f" A {m.group(1)} is not a numpy array, so it has no "
                f"`{m.group(2)}`. Use the nl/nisa functions instead.")
    return error_text

# ---------------------------------------------------------------- locating a failure
#
# Measured on level 3, four runs out of four: every repair round sent the kernel back BYTE-FOR-BYTE
# unchanged, all four samples. The feedback named the error -- "dma_copy requires src and dst to have
# the same number of elements, got src=8192, dst=65536" -- but not the line, nor which tile was the
# wrong shape, so "change exactly what the checker names" was satisfied by changing nothing. Worse,
# two of the canned hints answered the wrong question: "cannot reshape array of size 32768" came from
# a PSUM tile shaped (128, 512) for a (64, 512) result, and the hint said "do not reshape" to a kernel
# that never called reshape.
#
# The simulator runs the kernel as plain Python, so the traceback still holds the failing line and
# the live tiles. Read them, and name the change in the model's own variable names.

def _shape_of(t):
    try:
        return tuple(int(s) for s in t.shape)
    except Exception:
        return None


def _region_name(region):
    return str(region).rsplit(".", 1)[-1] or "?"      # MemoryRegion.sbuf -> sbuf


def _buffer_of(t):
    return _region_name(getattr(t, "buffer", ""))


def _is_tile(v):
    return hasattr(v, "shape") and hasattr(v, "buffer") and not isinstance(v, type)


def _entry_span(tree, entry):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == entry:
            return node.lineno, node.end_lineno
    return None


def locate_failure(exc, path, source, entry):
    """The line of the candidate that raised, the call on it, and that call's real operands.

    Returns None when the exception never passed through the candidate's own code.
    """
    tb, frame, lineno, wrapper = exc.__traceback__, None, None, None
    while tb is not None:
        f = tb.tb_frame
        if f.f_code.co_filename == path:
            frame, lineno, wrapper = f, tb.tb_lineno, None
        elif frame is not None and wrapper is None and {"func", "args", "kwargs"} <= set(f.f_locals):
            # nki routes every public call through one context wrapper; its locals are the call.
            wrapper = dict(f.f_locals)
        tb = tb.tb_next
    if frame is None:
        return None
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    stmt = None
    for node in ast.walk(tree):
        if isinstance(node, ast.stmt) and not isinstance(node, (ast.FunctionDef, ast.For, ast.If,
                                                                ast.While, ast.With)) \
                and node.lineno <= lineno <= (node.end_lineno or node.lineno):
            if stmt is None or node.lineno >= stmt.lineno:
                stmt = node
    code = " ".join((ast.get_source_segment(source, stmt) or "").split()) if stmt else ""
    local_vars = dict(frame.f_locals)

    op, args, exprs = None, {}, {}
    if wrapper is not None:
        func = wrapper["func"]
        op = getattr(func, "__name__", None)
        try:
            params = list(inspect.signature(func).parameters)
            args = dict(inspect.signature(func).bind_partial(*wrapper["args"],
                                                             **wrapper["kwargs"]).arguments)
        except (TypeError, ValueError):
            params, args = [], dict(wrapper["kwargs"])
        for node in ast.walk(stmt or tree):
            if isinstance(node, ast.Call) and nkibench._dotted(node.func).split(".")[-1] == op:
                for i, arg in enumerate(node.args):
                    if i < len(params):
                        exprs[params[i]] = ast.get_source_segment(source, arg)
                for kw in node.keywords:
                    if kw.arg:
                        exprs[kw.arg] = ast.get_source_segment(source, kw.value)
                break

    names = []
    for node in ast.walk(stmt) if stmt else []:
        if isinstance(node, ast.Name) and node.id not in names and _is_tile(local_vars.get(node.id)):
            names.append(node.id)
    tiles = {n: local_vars[n] for n in names}

    span = _entry_span(tree, entry)
    progress = 0.0
    if span and span[1] > span[0]:
        progress = min(1.0, max(0.0, (lineno - span[0]) / (span[1] - span[0])))
    return dict(line=lineno, code=code, op=op, args=args, exprs=exprs, tiles=tiles,
                progress=progress)


def shape_advice(loc, want_shape, matmul):
    """One named change for the shape mistakes the simulator reports obliquely, or None."""
    op, args, exprs = loc["op"], loc["args"], loc["exprs"]
    T = {k: v for k, v in args.items() if _is_tile(v)}

    def nm(k):
        return f"`{exprs.get(k) or k}`"

    if op == "ndarray":
        shape = args.get("shape")
        buf = _region_name(args.get("buffer"))
        try:
            shape = tuple(int(s) for s in shape)
        except Exception:
            return None
        if len(shape) < 2 and buf in ("sbuf", "psum"):
            msg = (f"This allocates a {buf} tile of shape {shape}, which is 1-D. SBUF and PSUM tiles "
                   f"are 2-D: (partition, free).")
            if matmul and want_shape:
                msg += (f" For this matmul the result is (M, N) = (lhsT.shape[1], rhs.shape[1]) = "
                        f"{want_shape}, so a tile or output meant to hold it needs that 2-D shape.")
            return msg
        return None

    if op == "nc_matmul" and {"dst", "stationary", "moving"} <= set(T):
        d, s, m = (_shape_of(T[k]) for k in ("dst", "stationary", "moving"))
        if not (d and s and m and len(s) == 2 and len(m) == 2):
            return None
        if s[0] != m[0]:
            return (f"nc_matmul contracts over the FIRST axis of both operands, so they must share it: "
                    f"stationary {nm('stationary')} is {s} and moving {nm('moving')} is {m}. "
                    f"stationary is [K, M] (the left matrix, already transposed: lhsT) and moving is "
                    f"[K, N] (rhs), with K on the partition axis of both.")
        if s[1] > nkibench.GEMM_STATIONARY_FMAX:
            return (f"stationary {nm('stationary')} is {s}, and its free dimension {s[1]} exceeds "
                    f"{nkibench.GEMM_STATIONARY_FMAX}. stationary must be the [K, M] tile loaded from "
                    f"lhsT and moving the [K, N] tile loaded from rhs -- check they are not swapped.")
        if m[1] > nkibench.GEMM_MOVING_FMAX:
            return (f"moving {nm('moving')} is {m}, and its free dimension {m[1]} exceeds "
                    f"{nkibench.GEMM_MOVING_FMAX}. Split N into chunks of at most "
                    f"{nkibench.GEMM_MOVING_FMAX}.")
        if d != (s[1], m[1]):
            return (f"nc_matmul writes stationary.T @ moving into dst. stationary {nm('stationary')} "
                    f"is {s} = [K, M] and moving {nm('moving')} is {m} = [K, N], so dst must be "
                    f"[M, N] = {(s[1], m[1])}, but dst {nm('dst')} is {d}. Allocate dst as "
                    f"nl.ndarray({(s[1], m[1])}, dtype=nl.float32, buffer=nl.psum).")
        return None

    if op in ("dma_copy", "tensor_copy") and {"dst", "src"} <= set(T):
        d, s = _shape_of(T["dst"]), _shape_of(T["src"])
        db, sb = _buffer_of(T["dst"]), _buffer_of(T["src"])
        if sb == "psum" and "hbm" in db:
            # Measured on level 3: four fresh samples of four got the loads, the PSUM tile and the
            # matmul right, then copied PSUM straight to the output -- with tensor_copy in some,
            # dma_copy in the repairs. Each copy's own error says only half of it. Name both hops.
            dst, src = exprs.get("dst") or "out", exprs.get("src") or "psum_tile"
            dtype = f"{dst}.dtype" if dst.isidentifier() else "nl.float32"
            return (f"PSUM cannot be copied straight to HBM by either copy. Replace this line with "
                    f"three: `res_sb = nl.ndarray({s}, dtype={dtype}, buffer=nl.sbuf)`, then "
                    f"`nisa.tensor_copy(dst=res_sb, src={src})`, then "
                    f"`nisa.dma_copy(dst={dst}, src=res_sb)`.")
        if op == "dma_copy" and "psum" in (db, sb):
            return (f"dma_copy cannot read or write PSUM ({nm('src')} is in {sb}, {nm('dst')} is in "
                    f"{db}). First nisa.tensor_copy the PSUM tile into an sbuf tile of the same shape, "
                    f"then dma_copy that sbuf tile to the shared_hbm output.")
        if d == s or d is None or s is None:
            return None
        if "hbm" in db and want_shape and d != want_shape and s == want_shape:
            return (f"The output {nm('dst')} is {d}, but the result is {want_shape}. Allocate the "
                    f"output you return as nl.ndarray({want_shape}, dtype=..., buffer=nl.shared_hbm).")
        if "hbm" in db and want_shape and s != want_shape:
            return (f"You are writing {nm('src')}, shape {s}, to the output {nm('dst')}, shape {d}. "
                    f"The result is {want_shape}: write the sbuf tile that holds the result (copied "
                    f"out of PSUM with nisa.tensor_copy), and make the output that shape too.")
        if "hbm" in sb:
            return (f"{nm('dst')} is {d} but {nm('src')} is {s}; a copy needs the same shape on both "
                    f"sides. Give the tensor you load its own sbuf tile of exactly {s}, e.g. "
                    f"nl.ndarray({s}, dtype=nl.float32, buffer=nl.sbuf), instead of one shared tile.")
        return (f"{nm('dst')} is {d} but {nm('src')} is {s}; a copy needs the same shape on both "
                f"sides. Copy into a new tile allocated with exactly {s}, e.g. "
                f"nl.ndarray({s}, dtype=nl.float32, buffer=nl.sbuf) -- do not reuse an input tile to "
                f"hold a result of a different shape.")
    return None


def explain_exception(exc, path, source, level, want_shape):
    """(feedback, progress) for a kernel that raised in the simulator."""
    entry = nkibench.LEVELS[level]["entry"]
    raw = f"raised {type(exc).__name__}: {exc}"
    loc = locate_failure(exc, path, source, entry)
    if loc is None:
        return enrich(raw), 0.0
    where = f"\n  The failing line is {loc['line']}: `{loc['code'][:200]}`"
    if loc["tiles"]:
        where += "\n  where " + "; ".join(f"`{n}` is {_shape_of(t)} in {_buffer_of(t)}"
                                          for n, t in loc["tiles"].items())
    advice = shape_advice(loc, want_shape, is_matmul(level))
    if advice:
        # The specific diagnosis replaces the generic hint, which for these errors points the wrong way.
        return f"{raw}{where}\n  FIX: {advice}", loc["progress"]
    return enrich(raw) + where, loc["progress"]


def first_prompt(level, terse=0):
    """Deliberately short, and it does NOT list the rules.

    Measured twice in this repo: hand a model an enumerated list of prohibitions and it audits
    itself against each one and returns nothing, while a bigger budget only buys more thinking.
    So the rules live in the checker. Generate freely, let the checker object, then send back one
    named change.
    """
    s = nkibench.LEVELS[level]
    if terse >= 2:
        # Last resort. Measured on this endpoint: one-sentence prompts answered in 300-700
        # tokens while every structured, rule-carrying prompt spiralled.
        return (f"Write a Python function `{s['entry']}` decorated with @nki.jit that computes "
                f"the same thing as this, using nki.language as nl and nki.isa as nisa:\n\n"
                f"{inspect.getsource(s['ref'])}\n"
                f"Reply with one python code block.")
    if terse >= 1:
        # The matmul memory rules are the substance of levels 3 and 4, and the short prompt has to
        # carry them: measured, the agent cycled between "dst must be in ['psum']" and "moving must
        # be in ['sbuf']" because nothing told it where the operands live.
        mm = ("nisa.nc_matmul(dst=, stationary=, moving=) needs dst in nl.psum and both operands "
              "in nl.sbuf. So: dma_copy the operands HBM->sbuf, allocate a psum tile, nc_matmul "
              "into it, tensor_copy psum->sbuf, then dma_copy sbuf->the shared_hbm output you "
              "return. The left operand is already transposed, with K on the partition axis.\n"
              if level >= 3 else "")
        return (f"Write an AWS Neuron NKI kernel: a function `{s['entry']}` decorated with "
                f"@nki.jit that computes what this reference computes.\n\n"
                f"{inspect.getsource(s['ref'])}\n"
                f"Allocate with nl.ndarray(shape, dtype=..., buffer=nl.sbuf), move data with "
                f"nisa.dma_copy(dst=, src=), loop with nl.affine_range(n). A tile's partition "
                f"dimension is at most {nkibench.PMAX}.\n{mm}\n"
                f"Reply with one python code block.")
    return (
        f"Write an AWS Neuron NKI kernel.\n\n"
        f"Operation: {s['op']}\n"
        f"Entry point: a function named `{s['entry']}`, decorated with `@nki.jit`.\n"
        f"It must compute exactly what this NumPy reference computes:\n\n"
        f"{inspect.getsource(s['ref'])}\n"
        f"Hardware limits: a tile's partition dimension is at most {nkibench.PMAX}. For matmul, "
        f"the stationary free dimension is at most {nkibench.GEMM_STATIONARY_FMAX} and the "
        f"moving free dimension at most {nkibench.GEMM_MOVING_FMAX}.\n\n"
        f"Import nki, nki.language as nl, and nki.isa as nisa.\n\n{API_CARD}\n\n"
        f"{matmul_card(level) if is_matmul(level) else ''}"
        f"Reply with ONE python code block containing the imports and the function. No prose.")


def repair_prompt(level, source, feedback, echoed=False):
    """One named change, and the previous code. No rules list, no reference re-sent.

    The lesson this whole repo keeps re-learning: feeding a verifier's report back verbatim
    reproduces the same mistake, because a report says what is wrong and never what to do.

    Measured on level 3: "change exactly what the checker names and keep everything else identical"
    got the kernel back byte-for-byte, 16 samples of 16 across four runs -- keeping everything
    identical is the easiest way to obey it. So the prompt now says the code as written fails, and
    says so twice as loudly when the last answer was an echo.
    """
    echo_note = ("You already sent this exact code back once, unchanged, and it failed the same way. "
                 "It has to change, starting with the failing line.\n\n" if echoed else "")
    return (
        f"This NKI kernel for {nkibench.LEVELS[level]['op']} fails.\n\n"
        f"```python\n{source}\n```\n\n"
        f"A checker ran it and reports:\n{feedback}\n\n"
        f"{MATMUL_CARD + chr(10) + chr(10) if is_matmul(level) else ''}"
        f"{echo_note}"
        f"Make the change the checker names, then reply with the complete corrected kernel in ONE "
        f"python code block. Sending the code back unchanged fails again.")


CODE_BLOCK = re.compile(r"```(?:python)?\s*(.*?)```", re.S)


def extract_code(text):
    """Pull out the code, and return NOTHING rather than prose.

    The old fallback returned the whole reply whenever it contained "def ", so a numbered list
    or a sentence reached the compiler and produced "invalid decimal literal on line 2" -- a
    parse error that blamed the model for the extractor's mistake.
    """
    text = text or ""
    blocks = CODE_BLOCK.findall(text)
    if blocks:
        return max(blocks, key=len).strip()
    # No fence: start at the first line that can legally begin a module and keep the rest.
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if re.match(r"^\s*(import |from |@nki|def )", line):
            return "\n".join(lines[i:]).strip()
    return ""


# ---------------------------------------------------------------- the model

def ask(a, prompt):
    import httpx
    # enable_thinking=False matters. Qwen3 reasons before answering, and with thinking on it
    # spent the whole budget there: the first cluster run returned "No code came back" at 54.7s
    # over and over, plus truncated fragments (invalid decimal literal, unterminated string).
    # Keep prompt + answer inside the server's context, or the answer is silently cut off and
    # every parse error below is really a budget error. Repair prompts grow with the kernel.
    est_prompt = len(prompt) // 4
    budget = min(a.max_tokens, max(256, a.context - est_prompt - 64))
    if budget < a.max_tokens:
        print(f"    (prompt is ~{est_prompt} tokens, so the answer budget is capped at {budget} "
              f"to stay inside the {a.context}-token context)")
    body = dict(model=a.model, messages=[{"role": "user", "content": prompt}],
                max_tokens=budget, temperature=0.6, top_p=0.95,
                chat_template_kwargs={"enable_thinking": a.think})
    r = httpx.post(f"{a.base.rstrip('/')}/chat/completions", json=body,
                   timeout=900, verify=False)
    if r.status_code != 200:
        raise SystemExit(f"the endpoint returned HTTP {r.status_code}:\n{r.text[:600]}")
    payload = r.json()
    ch = payload["choices"][0]
    msg = ch.get("message", {})
    reasoning = next((msg[k] for k in REASONING_KEYS if msg.get(k)), "")
    content = msg.get("content") or ""
    finish = ch.get("finish_reason")
    if finish == "length":
        # Do not let a budget problem look like a model failure.
        print(f"    (TRUNCATED: finish_reason=length after {len(content)} chars. The answer was "
              f"cut off, so any parse error below is the budget, not the model. Prompt is "
              f"{len(prompt)} chars; server context is the ceiling.)")
    if not content.strip() and reasoning:
        # The single most common surprise on this endpoint, so name it rather than reporting
        # an empty answer as a model failure.
        print(f"    (empty answer, {len(reasoning)} chars of hidden reasoning, "
              f"finish={ch.get('finish_reason')} — shorten the prompt rather than raising the "
              f"budget)")
    return content


def ask_parallel(a, prompts):
    """One request per prompt, all at once; replies come back in the same order."""
    import concurrent.futures as cf
    with cf.ThreadPoolExecutor(max_workers=len(prompts)) as ex:
        return [f.result() for f in [ex.submit(ask, a, p) for p in prompts]]


def offline_answers(level, n, rnd):
    """No model. Replays the shipped reference, preceded by a deliberately broken version, so the
    loop and the feedback path can be exercised with no endpoint. Never report a number."""
    ref = open(f"reference_level{level}.py").read()
    if rnd == 0:
        broken = ref.replace("@nki.jit", "", 1)
        return [f"```python\n{broken}\n```"] * n
    return [f"```python\n{ref}\n```"] * n


# ---------------------------------------------------------------- the loop

def _normalized(src):
    return "\n".join(line.rstrip() for line in (src or "").strip().splitlines())


def solve(a, level, log):
    print(f"\n=========== level {level}: {nkibench.LEVELS[level]['op']} ===========")
    terse = a.terse
    n = a.samples
    prompts = [first_prompt(level, terse)] * n
    strategies = ["fresh"] * n
    best = (0.0, None, "")
    tried, streak, seen = [], 0, {}
    graded_before = {}      # normalized code -> (reward, parts, feedback, progress)
    latest = None           # the attempt the next repair starts from
    for rnd in range(a.rounds):
        t0 = time.perf_counter()
        replies = (offline_answers(level, n, rnd) if a.offline else ask_parallel(a, prompts))
        earlier = set(graded_before)
        graded = []
        for prompt, strategy, reply in zip(prompts, strategies, replies):
            src = extract_code(reply)
            key = _normalized(src)
            # An echo is code this level has already graded. Measured on level 3: every repair
            # sample of every round was one, so the loop spent three rounds of four regrading a
            # kernel it already knew failed. Reuse the grade, and say so in the next prompt.
            # Two samples of one round agreeing is not an echo, so only earlier rounds count.
            echo = bool(key) and key in earlier
            if key in graded_before:
                reward, parts, feedback, progress = graded_before[key]
            else:
                reward, parts, feedback, progress = grade(src, level)
                graded_before[key] = (reward, parts, feedback, progress)
            graded.append(dict(reward=reward, src=src, feedback=feedback, parts=parts,
                               progress=progress, echo=echo))
            log.write(json.dumps(dict(level=level, round=rnd, reward=reward, parts=parts,
                                      progress=round(progress, 3), echo=echo, strategy=strategy,
                                      prompt_chars=len(prompt), reply_chars=len(reply),
                                      code=src, feedback=feedback)) + "\n")
        log.flush()
        # Rank by reward, then by how far the kernel got, then prefer new code over an echo.
        # Reward alone ties constantly below 0.5 -- every level-3 sample scored 0.30 -- and the
        # tie went to whichever reply came back first.
        graded.sort(key=lambda g: (g["reward"], g["progress"], not g["echo"]), reverse=True)
        top = graded[0]
        if top["reward"] > best[0]:
            best = (top["reward"], top["src"], top["feedback"])
        # Repair the LATEST round's best attempt, not the all-time best. Rebuilding from the best
        # attempt with the best attempt's feedback is a fixed point: once a round scores worse, the
        # prompt stops changing, and a greedy model then returns the same answer forever.
        # Measured: level 2 stuck at 0.10 for four rounds while the prompt still carried the 0.50
        # code.
        if top["src"].strip():
            latest = top
        echoes = sum(g["echo"] for g in graded)
        same = top["feedback"] == (tried[-1] if tried else None)
        if same:
            # Collapse. Fifteen identical multi-line blocks is noise, not information.
            print(f"round {rnd}: same failure again ({top['reward']:.2f}, best so far "
                  f"{best[0]:.2f})" + (f"  [{echoes}/{n} echoed earlier code]" if echoes else ""))
        else:
            print(f"round {rnd}: this round {top['reward']:.2f}  best so far {best[0]:.2f}  "
                  f"({time.perf_counter() - t0:.1f}s)"
                  + (f"  [{echoes}/{n} echoed earlier code]" if echoes else ""))
            print(f"  {top['feedback'][:700]}")
        if top["reward"] >= sum(WEIGHTS.values()) - 1e-9:
            print(f"  SOLVED on round {rnd}. {top['feedback']}")
            print("  ---------------- the kernel ----------------")
            print(textwrap.indent(top["src"], "  "))
            print("  -------------------------------------------")
            return top["reward"], rnd + 1
        seen[top["feedback"]] = seen.get(top["feedback"], 0) + 1
        streak = streak + 1 if same else 1
        if seen[top["feedback"]] >= a.give_up_after:
            how = ("the identical failure %d rounds running" % streak if streak >= a.give_up_after
                   else "this failure for the %dth time, alternating with %d other(s)"
                        % (seen[top["feedback"]], len(seen) - 1))
            print(f"  STOPPING this level: {how}. The agent is cycling between a fixed set of "
                  f"mistakes rather than converging, so more rounds will not help. Failures seen:")
            for f, k in sorted(seen.items(), key=lambda kv: -kv[1]):
                print(f"    {k}x  {f[:110]}")
            return best[0], rnd + 1
        tried.append(top["feedback"])

        if latest is None:
            # Nothing came back to repair. Asking it to "fix" an empty code block produced a
            # 202-character prompt and, under greedy sampling, the identical non-answer six
            # rounds running. Shorten and re-ask instead.
            terse = min(terse + 1, 2)
            prompts, strategies = [first_prompt(level, terse)] * n, ["fresh"] * n
            print(f"  no code yet, so re-asking with a shorter prompt (terseness {terse})")
            continue
        repair = repair_prompt(level, latest["src"], latest["feedback"], echoed=latest["echo"])
        if streak >= 2:
            # Sampling on the gpt-oss endpoint is greedy, so an unchanged prompt returns an
            # unchanged answer. Measured: the same TypeError 19 rounds running. Changing the
            # prompt is the only thing that can change the answer, so say what has already failed.
            ledger = "\n".join(f"- {t[:160]}" for t in dict.fromkeys(tried))
            repair += (f"\n\nThese approaches have already failed, so do something different:\n"
                       f"{ledger}")
            print(f"  same failure {streak}x — adding a ledger of {len(set(tried))} failed "
                  f"attempts to break the repeat")
        # Half the samples repair, half start over from the first prompt. Repairs converge on one
        # kernel; fresh samples are how a round escapes a kernel whose structure is wrong. Round 0
        # on level 3 produced up to four DISTINCT kernels, while repairs produced one. With one
        # sample (the greedy endpoint, where a fresh prompt would replay round 0) it only repairs.
        n_repair = max(1, (n + 1) // 2)
        prompts = [repair] * n_repair + [first_prompt(level, terse)] * (n - n_repair)
        strategies = ["repair"] * n_repair + ["fresh"] * (n - n_repair)
    print(f"  not solved in {a.rounds} rounds; best reward {best[0]:.2f}")
    return best[0], a.rounds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, choices=sorted(nkibench.LEVELS))
    ap.add_argument("--all", action="store_true", help="levels 1 to 4")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=MIN_ANSWER_TOKENS)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--base", default=os.environ.get("KERNEL_AGENT_BASE_URL")
                    or os.environ.get("GPTOSS_BASE_URL"))
    ap.add_argument("--path", default="", help="path to append, e.g. /agg/v1 for gpt-oss")
    ap.add_argument("--repeat", type=int, default=1,
                    help="run the whole thing N times and report a solve RATE. One run is not a "
                         "result: measured, the same config scored 1.00, 1.00 and 0.50 on level 2 "
                         "across three runs with no code change.")
    ap.add_argument("--log", default="attempts.jsonl")
    ap.add_argument("--give-up-after", type=int, default=4,
                    help="stop a level after this many identical failures in a row. Measured: 15 "
                         "was pure waste, because the prompt had stopped changing.")
    ap.add_argument("--terse", type=int, default=0, choices=(0, 1, 2),
                    help="starting prompt length. Measured on gpt-oss-20b: 0 produced 13,245 "
                         "chars of hidden reasoning and no answer, while 1 answered with code. "
                         "Qwen3-8B is fine at 0.")
    ap.add_argument("--context", type=int, default=4096,
                    help="the server's max-model-len; prompt + answer must fit inside it")
    ap.add_argument("--think", action="store_true",
                    help="let the model reason first; costs budget, and it ran out")
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args()

    if not a.offline:
        # Validate before the first request. An empty or scheme-less value produces a hostname
        # starting with "." and 30 lines of httpx/idna traceback that say nothing about the cause.
        raw = (a.base or "").strip()
        if not raw:
            sys.exit("KERNEL_AGENT_BASE_URL is empty or unset. Point it at a model:\n"
                     "  export KERNEL_AGENT_BASE_URL=http://localhost:8000/v1\n"
                     "  export KERNEL_AGENT_MODEL=Qwen/Qwen3-8B\n"
                     "That is the server ./serve.sh started. The seat pods set these for you; if they are "
                     "missing, you are not in a seat pod. Or pass --offline to run with no model.\n"
                     "(Separately: the shared gpt-oss endpoint is reached by setting "
                     "GPTOSS_BASE_URL and passing --path /agg/v1 -- not needed for the local run.)")
        from urllib.parse import urlparse
        u = urlparse(raw)
        if u.scheme not in ("http", "https") or not u.netloc:
            sys.exit(f"base URL {raw!r} is not usable. It needs a scheme and a host, e.g. "
                     f"http://qwen3-8b:8000/v1.")
        a.base = raw.rstrip("/") + a.path
        print(f"endpoint {a.base}  model {a.model}")
    else:
        print("*** OFFLINE: replaying the reference kernel. Numbers are meaningless. ***")

    levels = sorted(nkibench.LEVELS)[:4] if a.all else [a.level or 1]
    full = sum(WEIGHTS.values())
    history = {lv: [] for lv in levels}

    with open(a.log, "a") as log:
        for rep in range(a.repeat):
            if a.repeat > 1:
                print(f"\n################ run {rep + 1} of {a.repeat} ################")
            results = []
            for level in levels:
                results.append((level,) + solve(a, level, log))
                history[level].append(results[-1][1])

            print("\n=========== summary ===========")
            for level, reward, rounds in results:
                print(f"  level {level}  reward {reward:.2f} after {rounds} round(s)"
                      + ("  SOLVED" if reward >= full - 1e-9 else ""))
            print(f"  solved {sum(1 for _, r, _ in results if r >= full - 1e-9)}/{len(results)}")

    if a.repeat > 1:
        # The number that actually means something. A solve rate over N runs survives the variance
        # that makes any single run uninterpretable.
        print(f"\n=========== over {a.repeat} runs ===========")
        for lv in levels:
            got = history[lv]
            solves = sum(1 for r in got if r >= full - 1e-9)
            print(f"  level {lv}: solved {solves}/{len(got)}  "
                  f"best {max(got):.2f}  worst {min(got):.2f}  "
                  f"mean {sum(got) / len(got):.2f}  all={[round(r, 2) for r in got]}")
        print("\n  Report the rate, not your best run. A level that solves 1 in 3 times is not solved.")
    print(f"\nattempts logged to {a.log}")


if __name__ == "__main__":
    main()
