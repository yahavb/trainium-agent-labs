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
import hashlib
import json
import os
import re
import sys
import tempfile
import textwrap
import time

import numpy as np

import nkibench

MODEL = os.environ.get("KERNEL_AGENT_MODEL", "Qwen/Qwen3-32B")

# The model writes to a hidden reasoning channel before it writes any answer. Measured on this
# endpoint: a coding task burned 900 tokens thinking and returned EMPTY content. See gptoss/README.
MIN_ANSWER_TOKENS = 2500

# Sampling settings. Every attempt in the log records them, so a run made with different ones is
# never mistaken for the same experiment.
TEMPERATURE = 0.6
TOP_P = 0.95
EXPLORE_TEMPERATURES = [0.6, 0.8, 1.0, 1.2]     # one per sample under --explore

REASONING_KEYS = ("reasoning", "reasoning_content")


# ---------------------------------------------------------------- reward
#
# Graded, not pass/fail, so a near miss is distinguishable from nonsense and the loop has
# something to climb. Matches Project 1's shape: correctness dominates, and nothing else counts
# until the kernel is right.

WEIGHTS = dict(parses=0.1, rules=0.2, runs=0.2, correct=0.5)


def grade(source, level, fragments=False):
    """Returns (reward, parts, feedback). Feedback is an INSTRUCTION, never just a verdict.

    fragments=True appends a checked example kernel to the feedback for the errors that have one.
    """
    parts = dict(parses=False, rules=False, runs=False, correct=False)

    if not source.strip():
        return 0.0, parts, ("No code came back. Reply with one python code block containing the "
                            "kernel and nothing else.")
    try:
        compile(source, "<candidate>", "exec")
        parts["parses"] = True
    except SyntaxError as e:
        return (WEIGHTS["parses"] * 0, parts,
                f"The code does not parse: {e.msg} on line {e.lineno}. Send one complete python "
                f"code block.")

    violations = nkibench.check_rules(source, level)
    if violations:
        extra = ""
        if any("no function named" in v for v in violations):
            # Measured: this repeated 15 rounds running, because "there is no function named X"
            # never said what the function should look like. Hand over the exact line.
            import inspect
            ref = nkibench.LEVELS[level]["ref"]
            args = ", ".join(inspect.signature(ref).parameters)
            extra = (f" Start the function with exactly this line:  "
                     f"def {nkibench.LEVELS[level]['entry']}({args}):  "
                     f"and put @nki.jit on the line above it.")
        return (sum(WEIGHTS[k] for k, v in parts.items() if v), parts,
                "Rule violations, which score zero however fast the kernel is. Fix exactly "
                "these: " + " ".join(violations) + extra)
    parts["rules"] = True

    # Every attempt gets its own file. One shared /tmp path per level let two agents in a pod grade
    # each other's kernels, and fed the bytecode-cache mix-up that load_kernel now avoids.
    fd, path = tempfile.mkstemp(prefix=f"_agent_level{level}_", suffix=".py")
    with os.fdopen(fd, "w") as f:
        f.write(source)
    try:
        return _grade_file(path, level, parts, fragments)
    finally:
        os.unlink(path)


def _grade_file(path, level, parts, fragments=False):
    """The part of grade() that needs the kernel on disk: import it, then simulate every shape."""
    spec = nkibench.LEVELS[level]
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
                f"Use exactly those three.")
    except Exception as e:
        return (sum(WEIGHTS[k] for k, v in parts.items() if v), parts,
                f"The file imports but {spec['entry']} could not be loaded: "
                f"{type(e).__name__}: {e}")

    failures, passed, intensity = [], 0, None
    for case in spec["shapes"]:
        args, _ = nkibench.make_inputs(case, level)
        before = [x.copy() if isinstance(x, np.ndarray) else x for x in args]
        want = spec["ref"](*args)
        try:
            got, counted = nkibench.simulate_and_count(kernel, args)
        except nkibench.NkiMissing as e:
            return (sum(WEIGHTS[k] for k, v in parts.items() if v), parts,
                    f"CANNOT SIMULATE: {e}")
        except Exception as e:
            raised = f"raised {type(e).__name__}: {e}"
            failures.append((nkibench.label(case, level),
                             enrich(raised) + (fragment_note(raised, level) if fragments else "")))
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
            continue
        passed += 1
        # Matmul arithmetic, so only shapes given as M, K, N. Level 8's are seq x dim, and asking
        # for case["M"] ended the whole run with a KeyError as soon as one of its shapes passed.
        if {"M", "K", "N"} <= case.keys() and counted["bytes"]:
            intensity = nkibench.roofline(
                nkibench.matmul_flops(case["M"], case["K"], case["N"]), counted["bytes"])

    if failures:
        lbl, first = failures[0]
        return (sum(WEIGHTS[k] for k, v in parts.items() if v)
                + WEIGHTS["correct"] * passed / len(spec["shapes"]), parts,
                f"{passed} of {len(spec['shapes'])} shapes passed. On {lbl}: {first}")

    parts["correct"] = True
    reward = sum(WEIGHTS.values())
    note = "Correct on every shape."
    if intensity:
        note += " " + nkibench.explain_roofline(intensity)
    return reward, parts, note


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


def api_card(level):
    """The card trimmed to what the level uses (--level-cards).

    Matmul is a third of the card, and it leaks: on seat-38, 12 of 84 level-1 pooling attempts called
    nisa.nc_matmul. Levels 1 and 2 get the card without it; matmul levels get all of it.
    """
    if level >= 3:
        return API_CARD
    text = "\n".join(l for l in API_CARD.split("\n") if "nisa.nc_matmul(dst=" not in l)
    return text[:text.index("nisa.nc_matmul has strict")] + text[text.index("Slice tiles with"):]


# Code shown with a failure, only when that failure comes back (--fragments). A chunked-copy example
# placed in EVERY prompt once made every level worse (STATE.md: level 2 fell from 2 of 5 to 0 of 5),
# so each fragment answers one specific error, and each is a smaller kernel than the level's own: it
# shows the pattern, not the answer. `python agent.py --check-fragments` runs every one through
# nki.simulate, so the prompt never teaches NKI that does not work.
FRAGMENTS = [
    dict(name="scale",
         when=r"module 'nki\.(?:isa|language)' has no attribute "
              r"'(?:multiply|mul|scale|divide|div|mean|average)'",
         note="Arithmetic on a whole tile is nisa.tensor_scalar with an op from nki.language, "
              "for example scaling a tile by 0.25:",
         entry="scale_kernel", shape=(32, 16), expect=lambda a: a * 0.25,
         code="""\
@nki.jit
def scale_kernel(a):
    out = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.shared_hbm)
    t = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=t, src=a)
    s = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=s, data=t, op0=nl.multiply, operand0=0.25)
    nisa.dma_copy(dst=out, src=s)
    return out"""),
    dict(name="block",
         when=r"dma_copy requires src and dst to have the same number of elements"
              r"|cannot reshape array of size",
         note="To work on part of a tensor, allocate a tile with exactly the part's shape and slice "
              "the source to match. Never reshape. For example, copying rows 128-255 and columns 32-95:",
         entry="block_copy_kernel", shape=(256, 96), expect=lambda a: a[128:256, 32:96],
         code="""\
@nki.jit
def block_copy_kernel(a):
    out = nl.ndarray((128, 64), dtype=a.dtype, buffer=nl.shared_hbm)
    t = nl.ndarray((128, 64), dtype=a.dtype, buffer=nl.sbuf)   # exactly the slice's shape
    nisa.dma_copy(dst=t, src=a[128:256, 32:96])
    nisa.dma_copy(dst=out, src=t)
    return out"""),
    dict(name="chunks",
         when=r"partition dimension \d+ exceeds maximum",
         note="A tile holds at most 128 rows, so a taller tensor is moved in 128-row chunks, one tile "
              "per chunk, for example copying a 384-row tensor:",
         entry="chunked_copy_kernel", shape=(384, 64), expect=lambda a: a,
         code="""\
@nki.jit
def chunked_copy_kernel(a):
    rows, cols = a.shape
    out = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.shared_hbm)
    for i in nl.affine_range(rows // 128):
        t = nl.ndarray((128, cols), dtype=a.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=t, src=a[i * 128:(i + 1) * 128, 0:cols])
        nisa.dma_copy(dst=out[i * 128:(i + 1) * 128, 0:cols], src=t)
    return out"""),
    dict(name="rows",
         when=r"must have at least 2 dimensions",
         note="Every tile is 2-D, rows first. A result with one value per row has shape (rows, 1), "
              "for example summing each row:",
         entry="row_sum_kernel", shape=(64, 32), expect=lambda a: a.sum(axis=1, keepdims=True),
         code="""\
@nki.jit
def row_sum_kernel(a):
    rows, cols = a.shape
    out = nl.ndarray((rows, 1), dtype=a.dtype, buffer=nl.shared_hbm)
    t = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=t, src=a)
    s = nl.sum(t, axis=[1], keepdims=True)
    o = nl.ndarray((rows, 1), dtype=a.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=o, src=s)
    nisa.dma_copy(dst=out, src=o)
    return out"""),
    dict(name="columns", level=2,
         when=r"Out-of-bound access for tensor .* on dimension 0",
         note="shape2D = (F1, F2) describes the F1*F2 values inside each row. The first axis (rows) is "
              "never indexed with F1 or F2: keep it whole with `:` and move along the second axis "
              "with nl.ds(start, size). For example, swapping the two columns of every row:",
         entry="column_swap_kernel", shape=(32, 2), expect=lambda a: a[:, ::-1],
         code="""\
@nki.jit
def column_swap_kernel(a):
    out = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.shared_hbm)
    t = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=t, src=a)
    s = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=s[:, nl.ds(0, 1)], src=t[:, nl.ds(1, 1)])
    nisa.tensor_copy(dst=s[:, nl.ds(1, 1)], src=t[:, nl.ds(0, 1)])
    nisa.dma_copy(dst=out, src=s)
    return out"""),
]

FRAGMENT_IMPORTS = "import nki\nimport nki.isa as nisa\nimport nki.language as nl\n\n"


def fragment_note(error_text, level):
    """The checked example for this error, if there is one."""
    for f in FRAGMENTS:
        if f.get("level", level) == level and re.search(f["when"], error_text):
            return f"\n{f['note']}\n```python\n{f['code']}\n```"
    return ""


def check_fragments():
    """Run every fragment through nki.simulate against NumPy. Returns 0 if all of them are right."""
    import nki
    rc = 0
    for f in FRAGMENTS:
        ns = {}
        exec(compile(FRAGMENT_IMPORTS + f["code"], f"<fragment {f['name']}>", "exec"), ns)
        a = np.random.default_rng(0).standard_normal(f["shape"]).astype(np.float32)
        try:
            run, _ = nkibench._simulator(nki, ns[f["entry"]])
            got = np.asarray(run(a.copy()))
            ok = got.shape == f["expect"](a).shape and np.allclose(got, f["expect"](a), atol=1e-4)
            why = "" if ok else f"got shape {got.shape}, values off by {np.abs(got - f['expect'](a)).max():.3g}"
        except Exception as e:
            ok, why = False, f"raised {type(e).__name__}: {str(e)[:120]}"
        rc |= 0 if ok else 1
        print(f"  fragment {f['name']:<8} {'ok' if ok else 'FAIL ' + why}")
    return rc


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
    import inspect
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

def first_prompt(level, terse=0, cards=False):
    """Deliberately short, and it does NOT list the rules.

    Measured twice in this repo: hand a model an enumerated list of prohibitions and it audits
    itself against each one and returns nothing, while a bigger budget only buys more thinking.
    So the rules live in the checker. Generate freely, let the checker object, then send back one
    named change.
    """
    s = nkibench.LEVELS[level]
    import inspect
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
    mm_limits = ("" if cards and level < 3 else
                 f" For matmul, the stationary free dimension is at most "
                 f"{nkibench.GEMM_STATIONARY_FMAX} and the moving free dimension at most "
                 f"{nkibench.GEMM_MOVING_FMAX}.")
    return (
        f"Write an AWS Neuron NKI kernel.\n\n"
        f"Operation: {s['op']}\n"
        f"Entry point: a function named `{s['entry']}`, decorated with `@nki.jit`.\n"
        f"It must compute exactly what this NumPy reference computes:\n\n"
        f"{inspect.getsource(s['ref'])}\n"
        f"Hardware limits: a tile's partition dimension is at most {nkibench.PMAX}.{mm_limits}\n\n"
        f"Import nki, nki.language as nl, and nki.isa as nisa.\n\n"
        f"{api_card(level) if cards else API_CARD}\n\n"
        f"Reply with ONE python code block containing the imports and the function. No prose.")


def repair_prompt(level, source, feedback):
    """One named change, and the previous code. No rules list, no reference re-sent.

    The lesson this whole repo keeps re-learning: feeding a verifier's report back verbatim
    reproduces the same mistake, because a report says what is wrong and never what to do.
    """
    return (
        f"This NKI kernel for {nkibench.LEVELS[level]['op']} is not right yet.\n\n"
        f"```python\n{source}\n```\n\n"
        f"A checker reports:\n{feedback}\n\n"
        f"Change exactly what the checker names and keep everything else identical. Reply with "
        f"ONE python code block.")


def rethink_prompt(level, source, feedback):
    """For when every sample hands back the kernel it was given (--echo-break, first time).

    Measured on the 8K baseline: level 4's best kernel came back byte-identical three rounds
    running. "Keep everything else identical" plus unchanged feedback leaves the model nothing to
    change, so this drops that line, says the approach is what fails, and asks for a different one.
    """
    return (
        f"This NKI kernel for {nkibench.LEVELS[level]['op']} fails, and your last answer returned "
        f"it unchanged:\n\n```python\n{source}\n```\n\n"
        f"A checker reports:\n{feedback}\n\n"
        f"Repeating it cannot pass. Write the kernel a different way, so that this failure cannot "
        f"happen. Reply with ONE python code block.")


def fresh_prompt(level, terse, tried, cards=False):
    """For a second echo in a row: start over from the task, without the old code to anchor on,
    and list what has already failed."""
    ledger = "\n".join(f"- {t[:160]}" for t in dict.fromkeys(tried))
    return (first_prompt(level, terse, cards)
            + f"\n\nEarlier attempts at this failed like this; do not repeat them:\n{ledger}")


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

def ask(a, prompt, temperature=None):
    import httpx
    temperature = a.temperature if temperature is None else temperature
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
                max_tokens=budget, temperature=temperature, top_p=a.top_p,
                chat_template_kwargs={"enable_thinking": a.think})
    meta = dict(max_tokens=budget, temperature=temperature, top_p=a.top_p, error=None)
    t0 = time.perf_counter()
    # One failed request used to end the whole run, losing every later level and the --repeat
    # summary. Server faults and timeouts are retried after a pause; a 4xx is this request's own
    # fault and would fail identically, so it is recorded instead. Either way the round carries on
    # and the failure is logged as an attempt that returned no code.
    for attempt in range(a.retries + 1):
        try:
            r = httpx.post(f"{a.base.rstrip('/')}/chat/completions", json=body,
                           timeout=900, verify=False)
        except httpx.HTTPError as e:
            meta["error"] = f"{type(e).__name__}: {e}"
        else:
            if r.status_code == 200:
                meta["error"] = None
                break
            meta["error"] = f"HTTP {r.status_code}: {r.text[:300]}"
            if r.status_code < 500:
                break
        if attempt < a.retries:
            wait = 5 * 2 ** attempt
            print(f"    (request failed: {meta['error'][:160]}; retrying in {wait}s)")
            time.sleep(wait)
    meta["gen_seconds"] = round(time.perf_counter() - t0, 2)
    if meta["error"]:
        print(f"    (request failed for good: {meta['error'][:200]})")
        return "", meta
    payload = r.json()
    usage = payload.get("usage") or {}
    ch = payload["choices"][0]
    msg = ch.get("message", {})
    reasoning = next((msg[k] for k in REASONING_KEYS if msg.get(k)), "")
    content = msg.get("content") or ""
    finish = ch.get("finish_reason")
    meta.update(finish_reason=finish, prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"))
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
    return content, meta


def ask_parallel(a, prompt, n):
    """n samples of one prompt, at once. With --temperatures each sample gets its own temperature.

    At 0.6 / top_p 0.95 the samples were byte-identical in most rounds of the 8K baseline, and its
    runs replayed each other, so --samples 4 paid four times for one answer. A spread of
    temperatures is the cheapest way to make the samples differ.
    """
    import concurrent.futures as cf
    temps = a.temperatures or [a.temperature]
    with cf.ThreadPoolExecutor(max_workers=n) as ex:
        futures = [ex.submit(ask, a, prompt, temps[i % len(temps)]) for i in range(n)]
        return [f.result() for f in futures]


def server_context(a):
    """Ask the server what it serves, before the first round.

    This catches a wrong endpoint or model name with one clear line instead of a failed round, and
    reads the context length so --context cannot quietly disagree with the server: the seat
    servers run 8192 or 32768, while the old default here was 4096.
    """
    import httpx
    try:
        r = httpx.get(f"{a.base.rstrip('/')}/models", timeout=30, verify=False)
        r.raise_for_status()
        served = {m["id"]: m for m in r.json().get("data", [])}
    except Exception as e:
        sys.exit(f"cannot list the models at {a.base}/models: {type(e).__name__}: {e}")
    if a.model not in served:
        sys.exit(f"the server does not serve {a.model!r}; it serves {sorted(served)}. "
                 f"Set KERNEL_AGENT_MODEL or pass --model.")
    limit = served[a.model].get("max_model_len")
    if a.context is None:
        if not limit:
            print("the server did not report max_model_len; assuming a 4096-token context")
        return limit or 4096
    if limit and a.context > limit:
        print(f"--context {a.context} is more than the server's {limit}; using {limit}")
        return limit
    return a.context


def offline_answers(level, n, rnd):
    """No model. Replays the shipped reference, preceded by a deliberately broken version, so the
    loop and the feedback path can be exercised with no endpoint. Never report a number."""
    ref = open(f"reference_level{level}.py").read()
    if rnd == 0:
        broken = ref.replace("@nki.jit", "", 1)
        return [f"```python\n{broken}\n```"] * n
    return [f"```python\n{ref}\n```"] * n


# ---------------------------------------------------------------- the loop

def solve(a, level, log, run=0):
    print(f"\n=========== level {level}: {nkibench.LEVELS[level]['op']} ===========")
    terse = a.terse
    prompt = first_prompt(level, terse, a.level_cards)
    best = (0.0, None, "")
    tried, streak, seen = [], 0, {}
    latest = ("", "")
    sent_code, echoes = None, 0      # the kernel inside the current prompt; echo rounds in a row
    for rnd in range(a.rounds):
        t0 = time.perf_counter()
        replies = ([(r, {}) for r in offline_answers(level, a.samples, rnd)] if a.offline
                   else ask_parallel(a, prompt, a.samples))
        graded = []
        for sample, (reply, meta) in enumerate(replies):
            src = extract_code(reply)
            t1 = time.perf_counter()
            reward, parts, feedback = grade(src, level, a.fragments)
            grade_seconds = round(time.perf_counter() - t1, 2)
            graded.append((reward, src, feedback, parts))
            # Enough to rebuild any table in the write-up without guessing: which run and sample,
            # what the sampler was set to, where the time went (generation vs grading), and a
            # fingerprint of the code so identical samples are visible at a glance.
            log.write(json.dumps(dict(
                run=run, level=level, round=rnd, sample=sample, reward=reward, parts=parts,
                model=a.model, context=a.context, echo_break=a.echo_break,
                fragments=a.fragments, level_cards=a.level_cards,
                prompt_chars=len(prompt), reply_chars=len(reply), **meta,
                grade_seconds=grade_seconds, code_sha=hashlib.sha1(src.encode()).hexdigest()[:12],
                code=src, feedback=feedback)) + "\n")
        log.flush()
        if replies and all(m.get("error") for _, m in replies):
            # The server, not the model, failed this round. Asking again with a shorter prompt
            # (what an empty answer otherwise triggers) would change the experiment for nothing.
            print(f"round {rnd}: every request failed ({replies[0][1]['error'][:120]}); "
                  f"sending the same prompt next round")
            continue
        graded.sort(key=lambda g: g[0], reverse=True)
        top = graded[0]
        if top[0] > best[0]:
            best = (top[0], top[1], top[2])
        # Repair the LATEST attempt, not the best one. Rebuilding from the best attempt with the
        # best attempt's feedback is a fixed point: once a round scores worse, the prompt stops
        # changing, and a greedy model then returns the same answer forever. Measured: level 2
        # stuck at 0.10 for four rounds while the prompt still carried the 0.50 code.
        if (top[1] or "").strip():
            latest = (top[1], top[2])
        same = top[2] == (tried[-1] if tried else None)
        if same:
            # Collapse. Fifteen identical multi-line blocks is noise, not information.
            print(f"round {rnd}: same failure again ({top[0]:.2f}, best so far {best[0]:.2f})")
        else:
            print(f"round {rnd}: this round {top[0]:.2f}  best so far {best[0]:.2f}  "
                  f"({time.perf_counter() - t0:.1f}s)")
            print(f"  {top[2][:400]}")
        if top[0] >= sum(WEIGHTS.values()) - 1e-9:
            print(f"  SOLVED on round {rnd}. {top[2]}")
            print("  ---------------- the kernel ----------------")
            print(textwrap.indent(top[1], "  "))
            print("  -------------------------------------------")
            return top[0], rnd + 1
        seen[top[2]] = seen.get(top[2], 0) + 1
        streak = streak + 1 if same else 1
        if seen[top[2]] >= a.give_up_after:
            how = ("the identical failure %d rounds running" % streak if streak >= a.give_up_after
                   else "this failure for the %dth time, alternating with %d other(s)"
                        % (seen[top[2]], len(seen) - 1))
            print(f"  STOPPING this level: {how}. The agent is cycling between a fixed set of "
                  f"mistakes rather than converging, so more rounds will not help. Failures seen:")
            for f, n in sorted(seen.items(), key=lambda kv: -kv[1]):
                print(f"    {n}x  {f[:110]}")
            return best[0], rnd + 1
        tried.append(top[2])
        returned = {g[1].strip() for g in graded if g[1].strip()}
        if a.echo_break and sent_code is not None and returned == {sent_code.strip()}:
            # Every sample handed back the kernel it was given. Resending a prompt that asks for
            # one named change would get the same kernel again, so change what is asked.
            echoes += 1
            if echoes == 1:
                prompt, sent_code = rethink_prompt(level, latest[0], latest[1]), latest[0]
                print("  every sample returned the kernel it was given; asking for a different "
                      "approach")
            else:
                prompt, sent_code = fresh_prompt(level, terse, tried, a.level_cards), None
                print("  echoed again; starting over from the task, with the failures listed and "
                      "without the old kernel")
            continue
        echoes = 0
        repeats = streak
        if repeats >= 2 and (best[1] or "").strip():
            # Sampling on this endpoint is greedy, so an unchanged prompt returns an unchanged
            # answer. Measured: the same TypeError 19 rounds running. Changing the prompt is the
            # only thing that can change the answer, so say what has already been tried.
            ledger = "\n".join(f"- {t[:160]}" for t in dict.fromkeys(tried))
            prompt = (repair_prompt(level, latest[0], latest[1])
                      + f"\n\nThese approaches have already failed, so do something different:\n"
                        f"{ledger}")
            sent_code = latest[0]
            print(f"  same failure {repeats}x — adding a ledger of {len(set(tried))} failed "
                  f"attempts to break the repeat")
            continue
        if not (latest[0] or "").strip():
            # Nothing came back to repair. Asking it to "fix" an empty code block produced a
            # 202-character prompt and, under greedy sampling, the identical non-answer six
            # rounds running. Shorten and re-ask instead.
            terse = min(terse + 1, 2)
            prompt, sent_code = first_prompt(level, terse, a.level_cards), None
            print(f"  no code yet, so re-asking with a shorter prompt (terseness {terse})")
        else:
            prompt, sent_code = repair_prompt(level, latest[0], latest[1]), latest[0]
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
    ap.add_argument("--context", type=int, default=None,
                    help="the server's max-model-len; prompt + answer must fit inside it. "
                         "Default: read from the server")
    ap.add_argument("--retries", type=int, default=3,
                    help="retries for a failed or timed-out request before it is logged as an "
                         "attempt that returned no code")
    # Step 2 levers. All off by default, so the default run stays the baseline and each lever can
    # be measured on its own. --explore turns on all four.
    ap.add_argument("--temperature", type=float, default=TEMPERATURE)
    ap.add_argument("--top-p", type=float, default=TOP_P)
    ap.add_argument("--temperatures", type=lambda s: [float(x) for x in s.split(",")],
                    help="one temperature per sample, e.g. 0.6,0.8,1.0,1.2 (overrides --temperature)")
    ap.add_argument("--echo-break", action="store_true",
                    help="when every sample returns the kernel it was given, ask for a different "
                         "approach, then start over")
    ap.add_argument("--fragments", action="store_true",
                    help="show a checked example kernel with the errors that have one")
    ap.add_argument("--level-cards", action="store_true",
                    help="trim the API card to what the level uses (no matmul for levels 1-2)")
    ap.add_argument("--explore", action="store_true",
                    help=f"all four: --temperatures {','.join(map(str, EXPLORE_TEMPERATURES))} "
                         f"--echo-break --fragments --level-cards")
    ap.add_argument("--check-fragments", action="store_true",
                    help="run every example kernel through nki.simulate, then exit")
    ap.add_argument("--think", action="store_true",
                    help="let the model reason first; costs budget, and it ran out")
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args()
    if a.check_fragments:
        sys.exit(check_fragments())
    if a.explore:
        a.temperatures = a.temperatures or EXPLORE_TEMPERATURES
        a.echo_break = a.fragments = a.level_cards = True

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
        a.context = server_context(a)
        print(f"endpoint {a.base}  model {a.model}  context {a.context}")
    else:
        print("*** OFFLINE: replaying the reference kernel. Numbers are meaningless. ***")
    print(f"sampling: temperature {a.temperatures or a.temperature}, top_p {a.top_p}.  "
          f"echo-break {a.echo_break}, fragments {a.fragments}, level cards {a.level_cards}")

    levels = sorted(nkibench.LEVELS)[:4] if a.all else [a.level or 1]
    full = sum(WEIGHTS.values())
    history = {lv: [] for lv in levels}

    with open(a.log, "a") as log:
        for rep in range(a.repeat):
            if a.repeat > 1:
                print(f"\n################ run {rep + 1} of {a.repeat} ################")
            results = []
            for level in levels:
                results.append((level,) + solve(a, level, log, rep))
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
