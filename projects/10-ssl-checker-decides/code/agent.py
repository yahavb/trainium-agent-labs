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
                             enrich(f"raised {type(e).__name__}: {e}", source)))
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


# Names the model invents, mapped to what it was reaching for. Measured: baseline level 1 called
# nisa.multiply 36 times and nisa.scalar_mul 12; the plan-given runs invented nl.tile_index,
# nl.tile_offset, nisa.fill, acc.fill and nisa.psum_init*. A list of real names sorted alphabetically
# ("NkiInstruction, NkiValidationError, VirtualRegister, ...") did not move any of them.
INTENT = [
    (("multiply", "mul", "scalar_mul", "scale", "tensor_mul", "scalar_multiply"),
     "To multiply a tile by a number or a [P, 1] column, call nisa.tensor_scalar with dst= the "
     "sbuf tile you allocated for the result, data= the tile you are scaling, op0=nl.multiply and "
     "operand0= the number. nl.multiply is the operation, not a function."),
    (("fill", "zeros", "zero", "memset", "psum_init", "init", "initialize", "clear", "reset"),
     "PSUM needs no zeroing or filling: nisa.nc_matmul writes into a fresh "
     "nl.ndarray(shape, nl.float32, buffer=nl.psum) and accumulates across the K loop on its own. "
     "Remove the fill/zero step."),
    (("tile_index", "tile_offset", "tile_range", "tile_id", "get_tile", "tile_slice"),
     "There is no tile-index helper. Slice with the loop variable directly, e.g. "
     "lhsT[k*128:(k+1)*128, m*128:(m+1)*128] inside `for k in nl.affine_range(K // 128)`."),
]


def kernel_args(source):
    """The @nki.jit entry point's parameter names, e.g. ['lhsT', 'rhs'].

    Measured (loop-tool rerun, seat 49): a hint written with a placeholder -- "t = nl.ndarray(...)",
    "add a loop, say `i`" -- is copied literally, so 6 attempts wrote dma_copy(dst=t, ...) and hit a
    NameError. Hints are built from the kernel's real names instead.
    """
    m = re.search(r"@nki\.jit\s*\ndef\s+\w+\(([^)]*)\)", source or "")
    if not m:
        return []
    return [a.split(":")[0].split("=")[0].strip() for a in m.group(1).split(",") if a.strip()]


def alloc_hint(source):
    """A copyable allocation built from what the kernel itself loads -- never fixed indices.

    v2 used src=lhsT[0:128, 0:128]; the loop-tool session pointed out that, copied literally, that
    loads the first tile every iteration: right only on single-tile shapes, i.e. the 0.62 trap.
    So: reuse the kernel's own slice variable (e.g. lhsT_slice) or its own slice expression, and
    only fall back to words when it has neither. Checked in nki.simulate:
    nl.ndarray(x_slice.shape, ...) allocates a tile of that slice's shape.
    """
    args = kernel_args(source)
    names = [n for n in dict.fromkeys(re.findall(r"\b(\w+_slice)\b", source or ""))]
    if names:
        sl = names[0]
        base = sl[: -len("_slice")]
        dtype = f"{base}.dtype" if base in args else f"{sl}.dtype"
        more = (f" Do the same for {', '.join(names[1:])}." if len(names) > 1 else "")
        return (f"{base}_tile = nl.ndarray({sl}.shape, dtype={dtype}, buffer=nl.sbuf)\n"
                f"nisa.dma_copy(dst={base}_tile, src={sl})\n"
                f"then pass {base}_tile wherever you used the buffer.{more}")
    for a in args:
        m = re.search(rf"\b{re.escape(a)}\[[^\]\n]+\]", source or "")
        if m:
            expr = m.group(0)
            return (f"{a}_tile = nl.ndarray({expr}.shape, dtype={a}.dtype, buffer=nl.sbuf)\n"
                    f"nisa.dma_copy(dst={a}_tile, src={expr})\n"
                    f"then pass {a}_tile wherever you used the buffer. Do the same for each "
                    f"operand you load.")
    a = args[0] if args else "the input"
    return (f"allocate {a}_tile with nl.ndarray, giving it the shape of the slice of {a} you are "
            f"loading and buffer=nl.sbuf, then nisa.dma_copy that slice -- written with your loop "
            f"variables -- into {a}_tile, and use {a}_tile wherever you used the buffer.")


def intent_hint(name):
    n = name.lower()
    for keys, hint in INTENT:
        if any(n == k or n.startswith(k) for k in keys):
            return " " + hint
    return ""


def enrich(error_text, source=""):
    """Add the real names when the failure is an invented API call. `source`, when given, is the
    kernel, so hints can use its own argument names (placeholders get copied literally)."""
    if ("'MemoryRegion' object is not subscriptable" in error_text
            or re.search(r"'MemoryRegion' object has no attribute", error_text)
            or "'MemoryRegion' object is not callable" in error_text):
        # nl.sbuf[0], nl.sbuf.buf, nl.sbuf(0): all the same mistake, so one fix.
        return (error_text + " nl.sbuf, nl.psum and nl.shared_hbm are buffer KINDS, not memory: you "
                "cannot index, call or read attributes from them. Allocate a tile, copy into it, "
                "and use the tile:\n" + alloc_hint(source))
    m = re.search(r"ap\(\) pattern has invalid partition stride\. Partition step (\d+) must equal tensor "
                  r"free dimension size (\d+)", error_text)
    if m:
        # Level-1 step-list run: 8 of 8 attempts passed all-zero strides, e.g. [[0, 32], [0, 32], [0, 32]].
        return (error_text + f" In tile.ap([[stride, count], ...]) each pair steps `stride` ELEMENTS, "
                f"`count` times; a stride of 0 never moves. The first pair walks the partitions: its "
                f"stride must be the number of elements in one partition ({m.group(2)} here, H*W) and its "
                f"count the number of partitions. The pairs after it walk inside one partition, in "
                f"elements: one image row is W elements, one column is 1.")
    m = re.search(r"(\w+)\(\) takes (\d+) positional arguments? but (\d+) were given", error_text)
    if m:
        # Level-2 step-list run: 12 of 12 attempts declared the entry point with one parameter and then
        # used shape2D anyway. The harness calls it with the reference's arguments.
        ref_args = ""
        for lv in nkibench.LEVELS.values():
            if lv["entry"] == m.group(1):
                import inspect
                ref_args = ", ".join(inspect.signature(lv["ref"]).parameters)
        return (error_text + (f" The checker calls {m.group(1)} with the same arguments as the reference, "
                              f"so declare it as: def {m.group(1)}({ref_args}): and use those names."
                              if ref_args else " Declare the entry point with every argument the "
                              "reference takes."))
    if re.search(r"dma_copy requires HBM or SBUF tensors, got src=MemoryRegion\.psum", error_text):
        # Level 3 step-list run: 12 of 15 attempts DMA'd the PSUM accumulator straight to HBM.
        m = re.search(r"(\w+)\s*=\s*nl\.ndarray\([^\n]*buffer\s*=\s*nl\.psum", source or "")
        acc = m.group(1) if m else "the PSUM tile"
        sb = f"{acc}_sbuf" if m else "an SBUF tile"
        r = re.search(r"(\w+)\s*=\s*nl\.ndarray\([^\n]*buffer\s*=\s*nl\.shared_hbm", source or "")
        res = r.group(1) if r else None
        # Real names only -- placeholders like <the result> get copied literally.
        return (error_text + f" dma_copy cannot read PSUM. Copy {acc} into SBUF first, then DMA that:\n"
                + (f"{sb} = nl.ndarray({acc}.shape, dtype={res}.dtype, buffer=nl.sbuf)\n"
                   f"nisa.tensor_copy(dst={sb}, src={acc})\n"
                   f"nisa.dma_copy(dst={res}, src={sb})" if m and res else
                   "allocate an SBUF tile with the PSUM tile's shape, nisa.tensor_copy the PSUM tile into "
                   "it, then nisa.dma_copy that SBUF tile to the result."))
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
        hint = intent_hint(m.group(2))
        return error_text + (f" `{m.group(1)}` has no `{m.group(2)}`." + hint if hint
                             else available_names(f"{m.group(1)}.{m.group(2)}"))
    m = re.search(r"'(\w+)' object has no attribute '(\w+)'", error_text)
    if m:
        return (error_text + f" A {m.group(1)} is not a numpy array, so it has no "
                f"`{m.group(2)}`." + (intent_hint(m.group(2)) or " Use the nl/nisa functions instead."))
    return error_text

STEP_A = ("allocate SBUF tiles for lhsT_slice and rhs_slice, dma_copy each slice into its tile, then "
          "accumulate their product into acc with nisa.nc_matmul. Each allocation goes directly "
          "above its load.")
STEP_B = "move acc out to out_slice (acc is in PSUM; out_slice is in HBM)."


def step_list_note():
    """H2 (--step-list). STEP_A and STEP_B are loop_tool.WORDING["steps"] verbatim -- the wording that
    gave 2/5 unstaged and 5/5 staged with the loop tool. The first sentence is the one step the tool's
    skeleton used to do itself (acc, and the per-tile slices); it is the only added text."""
    return ("\n\nWrite it in these steps. For each output tile, allocate one PSUM tile `acc` before the k "
            "loop; inside the k loop, lhsT_slice and rhs_slice are this step's slices of lhsT and rhs, "
            "and out_slice is this output tile's slice of the result. Inside the k loop: "
            + STEP_A + " After the k loop: " + STEP_B)


def step_list_note_l3():
    """--step-list for level 3: one tile, no loops. Same style as STEP_A/STEP_B (each allocation
    directly above its use), aimed at level 3's measured walls: a 1-D tile (27 of 48 baseline
    attempts) and a reshape instead of using the given shapes (16 of 48)."""
    return ("\n\nWrite it in these steps. Each input is already exactly one tile, so use it whole: no "
            "loops, no reshape, every tile 2-D. Allocate an SBUF tile with lhsT's own shape and "
            "dma_copy lhsT into it; allocate an SBUF tile with rhs's own shape and dma_copy rhs into it. "
            "Allocate one PSUM tile acc of shape (M, N) with dtype nl.float32 and accumulate the product "
            "into acc with nisa.nc_matmul. Then move acc out to the result you return (acc is in PSUM; "
            "the result is in HBM, allocated with shape (M, N) and buffer=nl.shared_hbm). Each "
            "allocation goes directly above its first use.")


# Method-level step lists for levels 1 and 2. They name the method and the API, never the strides or
# index arithmetic (that IS the answer). Each targets the level's top measured failure: "dma_copy
# requires src and dst to have the same number of elements" (L1 36/96, L2 29/48 attempts).
STEPS_L12 = {
    1: ("\n\nWrite it in these steps. The whole input fits in one tile (C is at most 128 partitions): "
        "allocate one SBUF tile with the input's own shape (C, H, W) and dma_copy the WHOLE input into "
        "it -- the tile and the source of a dma_copy must have the same shape. Make a strided view of "
        "that tile with tile.ap(...) that puts each pool x pool window in the last two axes, sum those "
        "two axes with nl.sum, scale by 1/(pool_size*pool_size) with nisa.tensor_scalar into an SBUF "
        "tile you allocate with the sum's shape, then dma_copy that tile into the output allocated with "
        "shape (C, H // pool_size, W // pool_size) and buffer=nl.shared_hbm. Each allocation goes "
        "directly above its first use."),
    2: ("\n\nWrite it in these steps. The whole input fits in one tile (P is at most 128 partitions): "
        "allocate one SBUF tile with the input's own shape and dma_copy the WHOLE input into it -- the "
        "tile and the source of a dma_copy must have the same shape. Allocate a second SBUF tile with "
        "the same shape for the result. Within each partition row the input is an F1 x F2 matrix laid "
        "out row-major; build the transposed F2 x F1 layout by copying single columns between the two "
        "SBUF tiles with nisa.tensor_copy, looping over the matrix positions with nl.affine_range. Then "
        "dma_copy the result tile into the output allocated with the input's shape and "
        "buffer=nl.shared_hbm. Each allocation goes directly above its first use."),
}


def first_prompt(level, terse=0):
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


# ---------------------------------------------------------------- plan first (--plan-first)
#
# PLAN.md: level 4 scores 0.62 every run because the model picks the tiling and writes NKI in one go,
# and gets the tiling wrong. Gate 1 asks for the tiling alone, as a JSON plan graded by plan_check.py
# in milliseconds; gate 2 asks for the kernel only once a plan has passed, graded exactly as before.
#
# The format below uses placeholders on purpose. The example at the top of plan_check.py IS the
# correct level-4 plan, and showing the answer makes the model copy instead of reason (README).

PLAN_FORMAT = """{"loops": {"<loop name>": "<how many times it runs, a formula in M, K, N>", ...},
 "accumulate_over": ["<the loops whose iterations sum into the same output tile>"],
 "reads":  {"lhsT": [["<start>", "<stop>"], ["<start>", "<stop>"]],
            "rhs":  [["<start>", "<stop>"], ["<start>", "<stop>"]]},
 "writes": {"out":  [["<start>", "<stop>"], ["<start>", "<stop>"]]}}"""


def plan_prompt(level):
    s = nkibench.LEVELS[level]
    return (
        f"Before writing any NKI code, plan the tiling for {s['op']}: out = lhsT.T @ rhs, where lhsT "
        f"is [K, M], rhs is [K, N] and out is [M, N]. K and M are multiples of 128, N a multiple of "
        f"512.\n\n"
        f"Hardware limits: every tile's first (partition) axis is at most {nkibench.PMAX}; the lhsT "
        f"tile's second axis is at most {nkibench.GEMM_STATIONARY_FMAX}; the rhs and out tiles' second "
        f"axis is at most {nkibench.GEMM_MOVING_FMAX}.\n\n"
        f"Reply with ONE JSON object in this format and nothing else:\n\n{PLAN_FORMAT}\n\n"
        f"Each read or write gives one pair per axis of that tensor, first axis first: a JSON list of "
        f"two strings, the start formula and the stop formula, where the stop is exclusive. Close "
        f"every pair with ], as in the format. Formulas may use + - * / // %, ceil, min, max, numbers, "
        f"M, K, N and your loop names.")


def plan_repair_prompt(plan_text, feedback):
    return (f"This tiling plan is not right yet:\n\n```json\n{plan_text}\n```\n\n"
            f"A checker that tries it on many shapes reports:\n{feedback}\n\n"
            f"Change what the checker names and keep everything else identical. Reply with ONE JSON "
            f"object and nothing else.")


def extract_json(text):
    """The plan object from a reply: a ```json block if there is one, else the outermost {...}."""
    text = text or ""
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if m:
        return m.group(1).strip()
    i, j = text.find("{"), text.rfind("}")
    return text[i:j + 1].strip() if 0 <= i < j else ""


def grade_plan(text):
    """(rank, ok, message, parsed-or-None). rank orders attempts: 2 passes, 1 parses, 0 neither."""
    import plan_check
    if not text:
        return 0, False, "No JSON object came back. Reply with ONE JSON object and nothing else.", None
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        msg = f"The plan is not valid JSON: {e.msg} at line {e.lineno} column {e.colno}."
        if '")' in text:
            # Measured on seat 49: the prompt's "[start, stop)" notation made Qwen close pairs with ")",
            # and json.loads then pointed at the wrong line.
            msg += ' Fix: close every pair with ], not ) -- write ["0", "K"], never ["0", "K").'
        return 0, False, msg, None
    try:
        p = plan_check.parse_plan(obj)
    except plan_check.PlanError as e:
        return 0, False, f"The plan cannot be read: {e}", None
    r = plan_check.search(p, 300)
    return (2, True, r["message"], obj) if r["ok"] else (1, False, r["message"], obj)


def offline_plans(n, rnd):
    import plan_check
    if rnd == 0:
        return [json.dumps(plan_check.MUTANTS[0][1])] * n
    return [json.dumps(plan_check.REFERENCE)] * n


def gate1(a, level, log):
    """Returns (passing plan as text or None, rounds used, last checker message)."""
    import copy
    print(f"\n----------- gate 1: tiling plan for level {level} -----------")
    # A plan is ~400 characters. Measured on the first screen run: at the default 2500-token budget
    # all 4 replies ran to finish_reason=length (5359 chars) and every JSON object was cut off.
    ap = copy.copy(a)
    ap.max_tokens = a.plan_max_tokens
    prompt, latest, msg = plan_prompt(level), None, ""
    for rnd in range(a.plan_rounds):
        t0 = time.perf_counter()
        replies = offline_plans(a.samples, rnd) if a.offline else ask_parallel(ap, prompt, a.samples)
        graded = []
        for reply in replies:
            text = extract_json(reply)
            rank, ok, msg_i, obj = grade_plan(text)
            if getattr(a, "plan_feedback", "prescriptive") == "diagnostic" and not ok:
                # The honest "can Qwen plan?" test: the checker says what is wrong, never the fix.
                # Measured on seat 49: with Fix lines Qwen applied only the quoted fix text, so a
                # passing plan was mostly the checker's work.
                msg_i = msg_i.split(" Fix:")[0].strip()
            if rank == 0 and len(reply) > 4 * len(text) + 200:
                msg_i = (f"{msg_i} Your reply was {len(reply)} characters long but the plan is only "
                         f"about 400: reply with the JSON object alone, with no explanation and no "
                         f"code, so it is not cut off.")
            graded.append((rank, text, msg_i, obj))
            log.write(json.dumps(dict(level=level, gate=1, round=rnd, ok=ok, plan=text,
                                      feedback=msg_i, prompt_chars=len(prompt),
                                      reply_chars=len(reply), reply=reply[:4000])) + "\n")
        log.flush()
        graded.sort(key=lambda g: g[0], reverse=True)
        rank, text, msg, obj = graded[0]
        print(f"plan round {rnd}: {sum(g[0] == 2 for g in graded)}/{len(graded)} plans pass  "
              f"({time.perf_counter() - t0:.1f}s)")
        if rank == 2:
            plan_text = json.dumps(obj, indent=1)
            print(f"  PLAN PASSES on round {rnd}:\n{textwrap.indent(plan_text, '    ')}")
            return plan_text, rnd + 1, msg
        print(f"  {msg[:400]}")
        if rank >= 1:
            latest = text
        prompt = (plan_repair_prompt(latest, msg) if latest
                  else plan_prompt(level) + f"\n\nYour last reply could not be used: {msg}")
    print(f"  NO PASSING PLAN after {a.plan_rounds} rounds. Stuck at gate 1 on: {msg[:300]}")
    return None, a.plan_rounds, msg


def plan_first(a, level, log):
    """Gate 1, then gate 2 (the normal loop, prompted with the plan). Returns solve()'s tuple plus
    gate stats."""
    if getattr(a, "plan_file", None):
        import plan_check
        obj = (plan_check.REFERENCE if a.plan_file == "reference"
               else json.load(open(a.plan_file)))
        rank, ok, msg, _ = grade_plan(json.dumps(obj))
        if not ok:
            sys.exit(f"--plan-file does not pass plan_check, so it cannot stand in for gate 1: {msg}")
        plan_text, plan_rounds = json.dumps(obj, indent=1), 0
        print(f"\n----------- plan given (--plan-file {a.plan_file}); gate 1 skipped -----------")
    else:
        plan_text, plan_rounds, msg = gate1(a, level, log)
    if plan_text is None:
        return 0.0, 0, dict(gate1_rounds=plan_rounds, plan_passed=False, stuck_at=1, stuck_on=msg)
    steps = step_list_note() if getattr(a, "step_list", False) else ""
    note = (f"\n\nImplement this tiling plan exactly (loop counts, and the start and stop, exclusive, of each "
            f"tile reads and writes):\n```json\n{plan_text}\n```") + steps
    print(f"\n----------- gate 2: kernel for level {level}, following the plan -----------")
    reward, rounds = solve(a, level, log, initial_prompt=first_prompt(level, a.terse) + note,
                           plan_note=note, gate=2)
    return reward, rounds, dict(gate1_rounds=plan_rounds, plan_passed=True,
                                stuck_at=None if reward >= sum(WEIGHTS.values()) - 1e-9 else 2)


# ---------------------------------------------------------------- the loop

def _err_key(feedback):
    """The raw error a round ended on, without the hint: what a 'keep it fixed' line names."""
    m = re.search(r"raised (\w+: [^\n]{0,110})", feedback or "")
    if m:
        return m.group(1).split(" Fix")[0].strip()
    return (feedback or "").split(". On ")[-1][:110].strip()


def keep_note(fixed):
    """H3 (--keep-fixes). Measured on seat 49: Qwen applies the LATEST named fix and drops earlier
    ones, so it cycles. The existing ledger lists what failed; this lists what was fixed, to keep."""
    if not fixed:
        return ""
    return ("\n\nAlready fixed in earlier rounds -- keep these fixed, do not bring them back:\n"
            + "\n".join(f"- {f}" for f in fixed[-5:]))


def solve(a, level, log, initial_prompt=None, plan_note="", gate=None):
    print(f"\n=========== level {level}: {nkibench.LEVELS[level]['op']} ===========")
    terse = a.terse
    prompt = initial_prompt or first_prompt(level, terse)
    best = (0.0, None, "")
    tried, streak, seen = [], 0, {}
    latest = ("", "")
    keep = getattr(a, "keep_fixes", False)
    fixed, prev_key, prev_reward = [], None, -1.0
    for rnd in range(a.rounds):
        t0 = time.perf_counter()
        replies = (offline_answers(level, a.samples, rnd) if a.offline
                   else ask_parallel(a, prompt, a.samples))
        graded = []
        for reply in replies:
            src = extract_code(reply)
            reward, parts, feedback = grade(src, level)
            graded.append((reward, src, feedback, parts))
            entry = dict(level=level, round=rnd, reward=reward, parts=parts,
                         prompt_chars=len(prompt), reply_chars=len(reply), code=src,
                         feedback=feedback)
            if gate is not None:
                entry["gate"] = gate
            log.write(json.dumps(entry) + "\n")
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
        if keep:
            cur = _err_key(top[2])
            if prev_key and cur != prev_key and top[0] >= prev_reward and prev_key not in fixed:
                fixed.append(prev_key)
            if cur in fixed:
                fixed.remove(cur)        # it came back, so it is not fixed
            prev_key, prev_reward = cur, top[0]
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
        repeats = streak
        if repeats >= 2 and (best[1] or "").strip():
            # Sampling on this endpoint is greedy, so an unchanged prompt returns an unchanged
            # answer. Measured: the same TypeError 19 rounds running. Changing the prompt is the
            # only thing that can change the answer, so say what has already been tried.
            ledger = "\n".join(f"- {t[:160]}" for t in dict.fromkeys(tried))
            prompt = (repair_prompt(level, latest[0], latest[1])
                      + f"\n\nThese approaches have already failed, so do something different:\n"
                        f"{ledger}" + plan_note + keep_note(fixed))
            print(f"  same failure {repeats}x — adding a ledger of {len(set(tried))} failed "
                  f"attempts to break the repeat")
            continue
        if not (latest[0] or "").strip():
            # Nothing came back to repair. Asking it to "fix" an empty code block produced a
            # 202-character prompt and, under greedy sampling, the identical non-answer six
            # rounds running. Shorten and re-ask instead.
            terse = min(terse + 1, 2)
            prompt = first_prompt(level, terse) + plan_note + keep_note(fixed)
            print(f"  no code yet, so re-asking with a shorter prompt (terseness {terse})")
        else:
            prompt = repair_prompt(level, latest[0], latest[1]) + plan_note + keep_note(fixed)
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
    ap.add_argument("--plan-first", action="store_true",
                    help="gate 1: a JSON tiling plan graded by plan_check.py; gate 2: the kernel, "
                         "prompted with the passing plan. Level 4 only (PLAN.md).")
    ap.add_argument("--plan-file", metavar="PATH|reference",
                    help="skip gate 1 and give gate 2 this plan; 'reference' is plan_check.REFERENCE. "
                         "Tests plan -> code on its own (implies --plan-first)")
    ap.add_argument("--step-list", action="store_true",
                    help="H2: add a step list to the prompt (level 4: loop_tool's wording; level 3: "
                         "the single-tile version)")
    ap.add_argument("--keep-fixes", action="store_true",
                    help="H3: repair prompts also list errors fixed in earlier rounds, to keep fixed")
    ap.add_argument("--plan-feedback", choices=("prescriptive", "diagnostic"), default="prescriptive",
                    help="gate 1: prescriptive names the fix; diagnostic says only what is wrong")
    ap.add_argument("--plan-max-tokens", type=int, default=800,
                    help="gate-1 answer budget; a plan is ~400 characters")
    ap.add_argument("--plan-rounds", type=int, default=4,
                    help="gate-1 rounds; gate 2 keeps the full --rounds, so its code budget matches "
                         "the baseline's")
    a = ap.parse_args()
    if a.plan_file:
        a.plan_first = True
    if a.plan_first and (a.all or (a.level or 1) != 4):
        sys.exit("--plan-first is built for level 4 only: pass --level 4 (plan_check.py knows matmul).")

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
    gates = {lv: [] for lv in levels}

    with open(a.log, "a") as log:
        for rep in range(a.repeat):
            if a.repeat > 1:
                print(f"\n################ run {rep + 1} of {a.repeat} ################")
            results = []
            for level in levels:
                if a.step_list and level in (1, 2, 3, 4) and not a.plan_first:
                    note = (step_list_note() if level == 4 else step_list_note_l3() if level == 3
                            else STEPS_L12[level])
                    results.append((level,) + solve(a, level, log,
                                                    initial_prompt=first_prompt(level, a.terse) + note,
                                                    plan_note=note))
                    history[level].append(results[-1][1])
                    continue
                if a.plan_first:
                    reward, rounds, g = plan_first(a, level, log)
                    gates[level].append(g)
                    results.append((level, reward, rounds))
                else:
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
    if a.plan_first:
        print("\n=========== plan-first gates ===========")
        for lv in levels:
            for i, g in enumerate(gates[lv]):
                where = {None: "solved", 1: "stuck at gate 1 (plan)", 2: "stuck at gate 2 (code)"}
                print(f"  level {lv} run {i + 1}: plan {'passed' if g['plan_passed'] else 'never passed'}"
                      f" after {g['gate1_rounds']} gate-1 round(s); {where[g['stuck_at']]}"
                      + (f"; last plan message: {g['stuck_on'][:160]}" if g.get("stuck_on") else ""))
    print(f"\nattempts logged to {a.log}")


if __name__ == "__main__":
    main()
