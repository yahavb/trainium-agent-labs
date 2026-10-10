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
import json
import os
import re
import sys
import textwrap
import time

import numpy as np

import nkibench

MODEL = os.environ.get("KERNEL_AGENT_MODEL", "Qwen/Qwen3-8B")

# The model writes to a hidden reasoning channel before it writes any answer. Measured on this
# endpoint: a coding task burned 900 tokens thinking and returned EMPTY content. See gptoss/README.
MIN_ANSWER_TOKENS = 2500

REASONING_KEYS = ("reasoning", "reasoning_content")

CORE_CARD = (
    "Tiny NKI card: import nki, nki.language as nl, nki.isa as nisa. Decorate the entry point "
    "with @nki.jit. Allocate with nl.ndarray(..., buffer=nl.sbuf/nl.psum/nl.shared_hbm). "
    "Use nisa.dma_copy for HBM<->SBUF and return the shared_hbm output."
)

CONTEXT_CARDS = {
    "api_core": (
        "Core NKI API: decorate the entry point with @nki.jit. Allocate tiles with "
        "nl.ndarray(shape, dtype=..., buffer=nl.sbuf/nl.psum/nl.shared_hbm). Loop with "
        "nl.affine_range(n). Move HBM<->SBUF with nisa.dma_copy. Copy PSUM<->SBUF with "
        "nisa.tensor_copy. Scale tensors with nisa.tensor_scalar."
    ),
    "dma_copy_shape": (
        "DMA shape rule: nisa.dma_copy does not reshape, slice, pad, or broadcast. The dst tile "
        "and src slice must have exactly the same number of elements and compatible shape. Allocate "
        "tiles to match the slice being moved, not the full operation unless the full tensor is the "
        "slice."
    ),
    "tile_limits": (
        "Tile limits are maximums, not target sizes. The partition axis is at most 128 rows. If a "
        "dimension is larger, loop over chunks. If the final chunk is smaller, allocate/copy the "
        "smaller final slice rather than padding or reading past the tensor."
    ),
    "tile_rank": (
        "SBUF and PSUM tiles must be at least 2D: partition dimension first, free dimension second. "
        "Do not allocate a 1D on-chip tile. Represent vectors as (1, N) or (N, 1), whichever matches "
        "the operation and copy slice."
    ),
    "reductions": (
        "Reduction pattern: nl.sum returns a tile; it is called as nl.sum(view, axis=[...]) with no "
        "dst argument. To divide by a constant, write the sum into another tile with "
        "nisa.tensor_scalar(dst=..., data=sum_tile, op0=nl.multiply, operand0=scale)."
    ),
    "avgpool_reduction": (
        "Avgpool reduction pattern: copy the input block into SBUF, make an access-pattern view "
        "whose axes separate output rows, output columns, and the pool rows/columns, reduce only "
        "the pool axes with nl.sum(..., axis=[...]), then multiply by "
        "1.0 / (pool_size * pool_size). Do not call .mean() on an NKI tensor."
    ),
    "reduction_axis": (
        "NKI reduction axis rule: nl.sum can reduce only the last contiguous dimensions of a tile. "
        "For a 5D avgpool access-pattern view shaped like output_h, pool_h, output_w, pool_w, the "
        "pool axes are not both trailing. Build/reorder the access pattern so the two pool "
        "dimensions are the final axes, then reduce axis=[3, 4]."
    ),
    "access_patterns": (
        "Access-pattern views use strides and counts over an existing SBUF tile. The first stride "
        "for the partition axis must match the size of the free dimensions behind one partition "
        "row. Do not invent zero strides; derive strides from the tensor layout."
    ),
    "matmul_psum": (
        "Matmul buffer rule: nisa.nc_matmul(dst=..., stationary=..., moving=...) writes into a "
        "PSUM dst tile. Both operands must be SBUF tiles. After matmul, tensor_copy PSUM to SBUF, "
        "then dma_copy SBUF to the returned shared_hbm output."
    ),
    "matmul_tiling": (
        "Tiled matmul: produce output blocks by looping over M and N chunks, and loop over K chunks "
        "for the contraction. Allocate one PSUM tile for an output block outside the K loop and "
        "accumulate all K chunks into it before copying the final block out."
    ),
    "signatures": (
        "Use real signatures only. nl.sum(x, axis, dtype=None, keepdims=False). "
        "nisa.nc_matmul(dst=, stationary=, moving=, ...). Do not add guessed keyword arguments."
    ),
}

REPAIR_CARD_NAMES = {
    "scale": ["reductions", "signatures"],
    "reduction_api": ["reductions", "signatures"],
    "reduction_axis": ["reduction_axis", "avgpool_reduction", "reductions"],
    "nonfinite": ["dma_copy_shape", "matmul_psum"],
    "rule": ["api_core", "signatures"],
    "dma_shape": ["dma_copy_shape", "tile_limits"],
    "tile_rank": ["tile_rank", "dma_copy_shape"],
    "ragged": ["tile_limits", "dma_copy_shape"],
    "traffic": ["matmul_psum", "matmul_tiling"],
    "signature": ["signatures"],
    "generic": ["api_core"],
}

LEVEL_BASE_CARDS = {
    1: ["api_core", "dma_copy_shape", "tile_rank", "reductions"],
    2: ["api_core", "dma_copy_shape", "tile_limits"],
    3: ["api_core", "dma_copy_shape", "matmul_psum", "signatures"],
    4: ["api_core", "dma_copy_shape", "tile_limits", "matmul_psum", "matmul_tiling"],
}


# ---------------------------------------------------------------- reward
#
# Graded, not pass/fail, so a near miss is distinguishable from nonsense and the loop has
# something to climb. Matches Project 1's shape: correctness dominates, and nothing else counts
# until the kernel is right.

WEIGHTS = dict(parses=0.1, rules=0.2, runs=0.2, correct=0.5)


def grade(source, level):
    """Returns (reward, parts, feedback). Feedback is an INSTRUCTION, never just a verdict."""
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

    spec = nkibench.LEVELS[level]
    path = f"/tmp/_agent_level{level}.py"
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
            failures.append((nkibench.label(case, level),
                             enrich(f"raised {type(e).__name__}: {e}")))
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
        if level >= 3 and counted["bytes"]:
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
    if "ap() pattern has invalid partition stride" in error_text:
        return (error_text + " For level 1 avgpool, use the tutorial access pattern exactly: "
                "pool_view = input_tile.ap([[sz_hin*sz_win, sz_cin], "
                "[pool_size*sz_win, sz_hin//pool_size], "
                "[pool_size, sz_win//pool_size], [sz_win, pool_size], [1, pool_size]]). "
                "The first partition stride must be H*W, not 0 or 1.")
    m = re.search(r"module '([\w.]+)' has no attribute '(\w+)'", error_text)
    if m:
        return error_text + available_names(f"{m.group(1)}.{m.group(2)}")
    m = re.search(r"'(\w+)' object has no attribute '(\w+)'", error_text)
    if m:
        return (error_text + f" A {m.group(1)} is not a numpy array, so it has no "
                f"`{m.group(2)}`. Use the nl/nisa functions instead.")
    return error_text


def select_context_cards(level, feedback="", source="", max_cards=4):
    """Pick general context cards by failure category, not by visible test shape."""
    text = f"{feedback}\n{source}".lower()
    chosen = []

    def add(name):
        if name in CONTEXT_CARDS and name not in chosen:
            chosen.append(name)

    for name in LEVEL_BASE_CARDS.get(level, ["api_core", "dma_copy_shape"]):
        add(name)

    triggers = [
        ("dma_copy", "dma_copy_shape"),
        ("same number of elements", "dma_copy_shape"),
        ("partition dimension", "tile_limits"),
        ("exceeds maximum", "tile_limits"),
        ("out-of-bound", "tile_limits"),
        ("ragged", "tile_limits"),
        ("at least 2 dimensions", "tile_rank"),
        ("1-d tile", "tile_rank"),
        ("nl.sum", "reductions"),
        ("sum()", "reductions"),
        ("unexpected keyword argument 'dst'", "reductions"),
        ("ap() pattern", "access_patterns"),
        ("access pattern", "access_patterns"),
        ("nc_matmul", "matmul_psum"),
        ("psum", "matmul_psum"),
        ("stationary", "matmul_psum"),
        ("moving", "matmul_psum"),
        ("contraction", "matmul_tiling"),
        ("k loop", "matmul_tiling"),
        ("unexpected keyword argument", "signatures"),
        ("no attribute", "signatures"),
    ]
    for needle, card in triggers:
        if needle in text:
            add(card)

    return chosen[:max_cards]


def render_context_cards(names):
    if not names:
        return ""
    lines = ["Relevant context cards:"]
    for name in names:
        lines.append(f"- {name}: {CONTEXT_CARDS[name]}")
    return "\n".join(lines)


def repair_card_names(level, category):
    names = list(REPAIR_CARD_NAMES.get(category, ["api_core"]))
    if level == 1 and category in {"scale", "reduction_api", "reduction_axis"}:
        names.insert(0, "avgpool_reduction")
    deduped = []
    for name in names:
        if name not in deduped:
            deduped.append(name)
    return deduped


def compact_ledger(failures, limit=4):
    seen = []
    for failure in failures:
        short = re.sub(r"\s+", " ", failure).strip()[:180]
        if short and short not in seen:
            seen.append(short)
    return "\n".join(f"- {x}" for x in seen[-limit:])


def compact_feedback(feedback):
    fn = getattr(nkibench, "compact_message", None)
    return fn(feedback) if fn else " ".join(str(feedback).split())[:360]


def distill_failure(feedback):
    text = compact_feedback(feedback)
    low = text.lower()
    if "scale:" in low or "consistent scale" in low:
        return ("scale",
                "Fix only the scalar scaling after the reduction. Divide by the full reduction "
                "count/area and keep the tiling/copy structure unchanged.")
    if "nonfinite:" in low or "nan" in low or "shared_hbm output" in low:
        return ("nonfinite",
                "Fix only the final output path. Ensure every computed tile is written into the "
                "returned shared_hbm output with nisa.dma_copy to the exact output slice.")
    if "fail rules" in low or "rule violations" in low or "calls `np." in low:
        return ("rule",
                "Fix only the rule violation. Replace framework/host operations with explicit NKI "
                "tile operations; keep the entry point and operation unchanged.")
    if "has no `mean`" in low or "object has no attribute 'mean'" in low:
        return ("reduction_api",
                "Fix only the reduction. NKI tensors do not have .mean(); use nl.sum over the "
                "reduction axes, then nisa.tensor_scalar to divide by the full reduction area.")
    if "tensor_reduce axis" in low or "last contiguous" in low:
        return ("reduction_axis",
                "Fix only the reduction view axes. nl.sum can reduce only trailing contiguous "
                "dimensions, so make the pool dimensions the final axes of the access-pattern view "
                "and reduce axis=[3, 4].")
    if "same number of elements" in low or "dma_copy requires" in low:
        return ("dma_shape",
                "Fix only the dma_copy shape mismatch. Allocate the destination tile to exactly "
                "match the source slice being copied; do not pad, reshape, or copy a whole tensor "
                "into a smaller/larger tile.")
    if "at least 2 dimensions" in low or "1d" in low or "1-d" in low:
        return ("tile_rank",
                "Fix only the tile rank. SBUF/PSUM tiles must be 2D, with partition dimension "
                "first and free dimension second.")
    if "ragged edge" in low or "final partial" in low:
        return ("ragged",
                "Fix only the final partial tile bounds. Use the remaining size for the last "
                "chunk and copy/write only the valid output slice.")
    if "byte floor" in low or "hbm" in low or "traffic" in low:
        return ("traffic",
                "Fix only HBM traffic. Reuse tiles, keep one PSUM tile across the K loop, and copy "
                "the final block out once.")
    if "missing 1 required positional argument" in low and "dtype" in low:
        return ("signature",
                "Fix only the nl.ndarray allocation calls. Every nl.ndarray needs an explicit "
                "dtype argument, usually dtype=input_tensor.dtype, plus the intended buffer.")
    if "unexpected keyword argument" in low or "no attribute" in low:
        return ("signature",
                "Fix only the invalid API call. Use the real NKI signature named by the checker; "
                "do not invent keyword arguments or helper functions.")
    return ("generic", "Fix exactly the checker-reported issue and keep unrelated code unchanged.")


def estimate_tokens(text):
    return max(1, len(text) // 4)


def prompt_accounting(reference="", code="", feedback="", cards="", ledger="", instruction="", core=""):
    sections = {
        "core": estimate_tokens(core),
        "reference": estimate_tokens(reference),
        "code": estimate_tokens(code),
        "feedback": estimate_tokens(feedback),
        "cards": estimate_tokens(cards),
        "ledger": estimate_tokens(ledger),
        "instruction": estimate_tokens(instruction),
    }
    sections["total"] = sum(sections.values())
    return sections

def first_prompt(level, terse=0):
    """Deliberately short, and it does NOT list the rules.

    Measured twice in this repo: hand a model an enumerated list of prohibitions and it audits
    itself against each one and returns nothing, while a bigger budget only buys more thinking.
    So the rules live in the checker. Generate freely, let the checker object, then send back one
    named change.
    """
    s = nkibench.LEVELS[level]
    import inspect
    ref = inspect.getsource(s['ref'])
    if terse >= 2:
        # Last resort. Measured on this endpoint: one-sentence prompts answered in 300-700
        # tokens while every structured, rule-carrying prompt spiralled.
        return (f"Write a Python function `{s['entry']}` decorated with @nki.jit that computes "
                f"the same thing as this, using nki.language as nl and nki.isa as nisa:\n\n"
                f"{ref}\n"
                f"{CORE_CARD}\nReply with one python code block.")
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
                f"{ref}\n"
                f"{CORE_CARD}\n{mm}\n"
                f"Reply with one python code block.")
    return (
        f"Write an AWS Neuron NKI kernel.\n\n"
        f"Operation: {s['op']}\n"
        f"Entry point: a function named `{s['entry']}`, decorated with `@nki.jit`.\n"
        f"It must compute exactly what this NumPy reference computes:\n\n"
        f"{ref}\n\n"
        f"{CORE_CARD}\n\n"
        f"Reply with ONE python code block containing the imports and the function. No prose.")


def repair_prompt(level, source, feedback, tried=None):
    """One named change, and the previous code. No rules list, no reference re-sent.

    The lesson this whole repo keeps re-learning: feeding a verifier's report back verbatim
    reproduces the same mistake, because a report says what is wrong and never what to do.
    """
    compact = compact_feedback(feedback)
    category, instruction = distill_failure(compact)
    card_names = repair_card_names(level, category)
    cards = render_context_cards(card_names)
    ledger = compact_ledger(tried or [])
    ledger_text = f"\n\nAlready tried; avoid repeating these failures:\n{ledger}" if ledger else ""
    return (
        f"This NKI kernel for {nkibench.LEVELS[level]['op']} is not right yet.\n\n"
        f"Repair instruction: {instruction}\n\n"
        f"```python\n{source}\n```\n\n"
        f"{cards}\n\n"
        f"Compact checker report:\n{compact}\n\n"
        f"Make only the repair instruction above. Keep everything else identical.{ledger_text} "
        f"Reply with "
        f"ONE python code block.")


AUDIT_FAILURES = {
    "dtype": "raised TypeError: ndarray() missing 1 required positional argument: 'dtype'",
    "mean": "raised AttributeError: 'NkiTensor' object has no attribute 'mean'",
    "dma": "nisa.dma_copy requires src and dst to have the same number of elements",
    "scale": "CONSISTENT SCALE ERROR: output is about 2x the reference across most elements.",
    "axis": "raised AssertionError: tensor_reduce axis must be the last contiguous dim(s) of the tile. Got axis=(2, 4)",
    "nonfinite": "NON-FINITE OUTPUT: output contains NaN/Inf. Check the final copy path.",
    "traffic": "traffic K=256 M=256 N=1024: 2.0x byte floor.",
}


def print_budget(name, budget):
    parts = ", ".join(f"{k}={v}" for k, v in budget.items() if k != "total")
    print(f"  {name}: total~{budget['total']} tokens ({parts})")


def audit_context(level):
    import inspect
    ref = inspect.getsource(nkibench.LEVELS[level]["ref"])
    first = first_prompt(level)
    first_budget = prompt_accounting(reference=ref, core=CORE_CARD)
    print(f"level {level}: {nkibench.LEVELS[level]['op']}")
    print_budget("first prompt", first_budget)
    print(f"    chars={len(first)} cards=core_minimal")

    source = (
        "import nki\nimport nki.language as nl\nimport nki.isa as nisa\n\n"
        f"@nki.jit\ndef {nkibench.LEVELS[level]['entry']}(*args):\n"
        "    out = nl.ndarray(args[0].shape, dtype=args[0].dtype, buffer=nl.shared_hbm)\n"
        "    return out\n"
    )
    for name, feedback in AUDIT_FAILURES.items():
        cat, inst = distill_failure(feedback)
        cards = repair_card_names(level, cat)
        rendered = render_context_cards(cards)
        prompt = repair_prompt(level, source, feedback, tried=[feedback])
        budget = prompt_accounting(code=source, feedback=compact_feedback(feedback),
                                   cards=rendered, ledger=compact_ledger([feedback]),
                                   instruction=inst)
        print(f"\n  case={name} category={cat}")
        print(f"    cards={', '.join(cards)}")
        print(f"    instruction={inst}")
        print_budget("repair prompt", budget)
        print(f"    chars={len(prompt)} compact_feedback={compact_feedback(feedback)}")


def summarize_log(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        print(f"{path}: no attempts")
        return

    print(f"{path}: {len(rows)} attempt(s)")
    by_cat = {}
    for r in rows:
        cat = r.get("failure_category", "unknown")
        by_cat[cat] = by_cat.get(cat, 0) + 1
    print("failure categories:")
    for cat, n in sorted(by_cat.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {cat}: {n}")

    print("\nattempts:")
    for r in rows:
        b = r.get("prompt_budget") or {}
        cards = ",".join(r.get("context_cards") or [])
        print(
            f"  L{r.get('level')} R{r.get('round')}: reward={r.get('reward'):.2f} "
            f"cat={r.get('failure_category', 'unknown')} "
            f"prompt~{b.get('total', len(str(r.get('prompt_chars', ''))) // 4)}tok "
            f"cards={cards or 'none'}"
        )


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


def ask_parallel(a, prompt, n):
    import concurrent.futures as cf
    with cf.ThreadPoolExecutor(max_workers=n) as ex:
        return [f.result() for f in [ex.submit(ask, a, prompt) for _ in range(n)]]


def offline_answers(level, n, rnd):
    """No model. Replays the shipped reference, preceded by a deliberately broken version, so the
    loop and the feedback path can be exercised with no endpoint. Never report a number."""
    ref = open(f"reference_level{level}.py").read()
    if rnd == 0:
        broken = ref.replace("@nki.jit", "", 1)
        return [f"```python\n{broken}\n```"] * n
    return [f"```python\n{ref}\n```"] * n


# ---------------------------------------------------------------- the loop

def solve(a, level, log):
    print(f"\n=========== level {level}: {nkibench.LEVELS[level]['op']} ===========")
    terse = a.terse
    prompt = first_prompt(level, terse)
    prompt_cards = ["core_minimal"]
    prompt_budget = prompt_accounting(
        reference=__import__("inspect").getsource(nkibench.LEVELS[level]["ref"]),
        core=CORE_CARD)
    best = (0.0, None, "")
    tried, streak, seen = [], 0, {}
    latest = ("", "")
    for rnd in range(a.rounds):
        t0 = time.perf_counter()
        replies = (offline_answers(level, a.samples, rnd) if a.offline
                   else ask_parallel(a, prompt, a.samples))
        graded = []
        for reply in replies:
            src = extract_code(reply)
            reward, parts, feedback = grade(src, level)
            failure_category, repair_instruction = distill_failure(feedback)
            graded.append((reward, src, feedback, parts))
            log.write(json.dumps(dict(level=level, round=rnd, reward=reward, parts=parts,
                                      prompt_chars=len(prompt), reply_chars=len(reply),
                                      context_cards=prompt_cards,
                                      prompt_budget=prompt_budget,
                                      failure_category=failure_category,
                                      repair_instruction=repair_instruction,
                                      compact_feedback=compact_feedback(feedback),
                                      code=src, feedback=feedback)) + "\n")
        log.flush()
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
            print(f"  context cards: {', '.join(prompt_cards) if prompt_cards else 'none'}")
            cat, inst = distill_failure(top[2])
            print(f"  failure: {cat}; {inst[:260]}")
            print(f"  checker: {compact_feedback(top[2])[:300]}")
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
        repeats = streak
        if repeats >= 2 and (best[1] or "").strip():
            # Sampling on this endpoint is greedy, so an unchanged prompt returns an unchanged
            # answer. Measured: the same TypeError 19 rounds running. Changing the prompt is the
            # only thing that can change the answer, so say what has already been tried.
            ledger = "\n".join(f"- {t[:160]}" for t in dict.fromkeys(tried))
            prompt = (repair_prompt(level, latest[0], latest[1], tried)
                      + f"\n\nThese approaches have already failed, so do something different:\n"
                        f"{ledger}")
            cat, inst = distill_failure(latest[1])
            prompt_cards = repair_card_names(level, cat)
            prompt_budget = prompt_accounting(
                code=latest[0], feedback=compact_feedback(latest[1]),
                cards=render_context_cards(prompt_cards), ledger=compact_ledger(tried),
                instruction=inst)
            print(f"  same failure {repeats}x — adding a ledger of {len(set(tried))} failed "
                  f"attempts to break the repeat")
            continue
        if not (latest[0] or "").strip():
            # Nothing came back to repair. Asking it to "fix" an empty code block produced a
            # 202-character prompt and, under greedy sampling, the identical non-answer six
            # rounds running. Shorten and re-ask instead.
            terse = min(terse + 1, 2)
            prompt = first_prompt(level, terse)
            prompt_cards = ["core_minimal"]
            prompt_budget = prompt_accounting(
                reference=__import__("inspect").getsource(nkibench.LEVELS[level]["ref"]),
                core=CORE_CARD)
            print(f"  no code yet, so re-asking with a shorter prompt (terseness {terse})")
        else:
            prompt = repair_prompt(level, latest[0], latest[1], tried)
            cat, inst = distill_failure(latest[1])
            prompt_cards = repair_card_names(level, cat)
            prompt_budget = prompt_accounting(
                code=latest[0], feedback=compact_feedback(latest[1]),
                cards=render_context_cards(prompt_cards), ledger=compact_ledger(tried),
                instruction=inst)
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
    ap.add_argument("--audit-context", action="store_true",
                    help="print prompt cards and token estimates for representative failures, "
                         "without calling the model")
    ap.add_argument("--summarize-log", metavar="PATH",
                    help="summarize an attempts JSONL log without calling the model")
    a = ap.parse_args()

    if a.summarize_log:
        summarize_log(a.summarize_log)
        return

    if a.audit_context:
        levels = sorted(nkibench.LEVELS)[:4] if a.all else [a.level or 1]
        for level in levels:
            audit_context(level)
        return

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
