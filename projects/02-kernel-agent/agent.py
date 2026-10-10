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
import skills
import tools as agent_tools

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
        if level == 4 and isinstance(e, NameError) and "nki" in str(e):
            return (sum(WEIGHTS[k] for k, v in parts.items() if v), parts,
                "The replacement file is missing the required NKI imports. A complete "
                "file must begin with these exact lines, before @nki.jit:\n"
                "import nki\n"
                "import nki.language as nl\n"
                "import nki.isa as nisa\n"
                "Then define the decorated kernel. Return the imports and function together.")
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
                             enrich(f"raised {type(e).__name__}: {e}", level)))
            continue
        parts["runs"] = True
        m = (nkibench.check_inputs_untouched(before, args)
             or nkibench.describe_mismatch(got, want, op=spec["op"])
             or nkibench.check_traffic_bar(level, counted, args, want))
        # A simulator warning about a hardware-correctness hazard counts as a failure even when the
        # numbers happen to match on CPU: the kernel would be wrong on the device.
        hazards = [w for w in counted.get("warnings", [])
                   if "incorrect results on hardware" in w]
        if m and level == 2:
            m = transpose_result_hint(m)
        if hazards and not m:
            m = ("CORRECT ON CPU BUT WRONG ON HARDWARE: " + hazards[0]
                 + ". Fix that before anything else -- the simulator agrees with the reference here "
                   "and the device would not.")
        if m:
            failures.append((nkibench.label(case, level),
                             enrich(m, level=level) if level == 4 else m))
            continue
        passed += 1
        if level >= 3 and counted["bytes"] and {"M", "K", "N"} <= set(case):
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

  This is NKI 0.6 (import nki). There is no neuronxcc, no nl.mgrid or nl.arange, and nisa.* calls
  take dst= and return nothing (do not assign their result).

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


# Keep pooling guidance separate from the matmul card. In the seat-21 baseline,
# level-1 samples used nc_matmul for a pooling window, then spent their repair
# rounds fixing that operation's API instead of computing a window average.
POOL_METHOD = """Average pooling is a window reduction: sum each pool_size by pool_size window,
then scale by 1.0 / (pool_size * pool_size). Keep channels on the partition axis.
Use nl.sum for the reduction and nisa.tensor_scalar with op0=nl.multiply to scale.
"""

POOL_API_CARD = """NKI pooling primitives:
  nl.ndarray(shape, dtype=..., buffer=nl.sbuf) allocates an on-chip tile.
  nl.ndarray(shape, dtype=..., buffer=nl.shared_hbm) allocates the returned output.
  nisa.dma_copy(dst=tile, src=input_slice) loads a tile; both sides must have the
    same element count. Derive tile sizes from the input, with at most 128 channels.
    Preserve the channel axis when slicing, even for a single channel.
  tile.ap([[stride, count], ...]) creates a view, with strides in ELEMENTS.
    For pooling, group the tile as [channels, output_rows, output_cols, pool_rows,
    pool_cols]. Derive strides from the original row width and pool_size.
  sums = nl.sum(window_view, axis=[3, 4]) reduces the two window axes.
  scaled = nl.ndarray(sums.shape, dtype=sums.dtype, buffer=nl.sbuf)
  nisa.tensor_scalar(dst=scaled, data=sums, op0=nl.multiply,
                     operand0=1.0 / (pool_size * pool_size)) scales the sums.
  nisa.dma_copy(dst=output_slice, src=scaled) stores the matching output tile.
"""


MATMUL_SHAPES = """This is a single-tile matmul. Read K, M = lhsT.shape and K_rhs, N = rhs.shape.
The input shapes are (K, M) and (K, N); the output shape is (M, N).
lhsT.shape[1:] is only (M,), so it is not a valid result shape.
The left input already has the layout nc_matmul needs: K is the partition axis
of BOTH inputs. The supported level-3 inputs fit one hardware tile.
"""

MATMUL_API_CARD = """Allocate separate tiles for these roles:
  left:   shape (K, M), dtype lhsT.dtype, buffer nl.sbuf
  right:  shape (K, N), dtype rhs.dtype,  buffer nl.sbuf
  accum:  shape (M, N), dtype nl.float32, buffer nl.psum
  result: shape (M, N), dtype lhsT.dtype, buffer nl.sbuf
  output: shape (M, N), dtype lhsT.dtype, buffer nl.shared_hbm
Use nl.ndarray(shape, dtype=..., buffer=...) for allocations.
Load each input into its own matching SBUF tile with nisa.dma_copy(dst=, src=).
Keep both input tiles alive until matmul: loading rhs into the left tile overwrites
lhsT. Allocate from each input shape, not fixed hardware maxima.
Call nisa.nc_matmul(dst=accum, stationary=left, moving=right).
Copy accum to the separate result SBUF tile with nisa.tensor_copy(dst=, src=),
then store result to output with nisa.dma_copy(dst=, src=) and return output.
Both result copies preserve the (M, N) shape; an input tile has a different role
and shape, so using it as the result buffer loses the required dimensions.
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


def enrich(error_text, level=None):
    """Add the real names when the failure is an invented API call.

    `level` is optional so existing callers keep working; it lets a level-specific message fire
    (used for the level-1 pooling load/reduce idiom below).
    """
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
        if level == 1 and src > dst:
            # Measured on level 1: the model sized the SBUF tile to the OUTPUT (pooled) shape and
            # then copied the whole INPUT into it (src=32768 into dst=16384). Pooling does not shrink
            # on load -- it loads everything and reduces afterwards. Name that idiom.
            return (error_text + f" You copied {src} input elements into a tile that holds only "
                    f"{dst}, which is the OUTPUT (pooled) size. Average pooling does not shrink the "
                    f"data when it loads it -- it loads the FULL input, then reduces. So: allocate "
                    f"the SBUF tile with the INPUT's own shape, "
                    f"in_tile = nl.ndarray(in_tensor.shape, dtype=in_tensor.dtype, buffer=nl.sbuf), "
                    f"and nisa.dma_copy(dst=in_tile, src=in_tensor). The averaging happens AFTER the "
                    f"load: build a strided view of the pool windows with in_tile.ap([...]) that "
                    f"groups each pool_size x pool_size window onto the last axes, then "
                    f"nl.sum(view, axis=[...]) over those axes and scale by 1/(pool_size*pool_size). "
                    f"The separate, smaller OUTPUT tile is what you write the reduced result into "
                    f"before dma_copy'ing it out.")
        if level == 3:
            return (error_text + f" The source has {src} elements but the destination has {dst}. "
                    "For this single-tile matmul, load lhsT into a (K, M) SBUF tile and rhs "
                    "into a different (K, N) SBUF tile. The PSUM result, result SBUF tile "
                    "and returned HBM output must all be (M, N). Derive K and M from "
                    "lhsT.shape and N from rhs.shape[1]; match each copy to its own tensor. "
                    "Keep the two input buffers distinct: loading rhs into the same "
                    "buffer or an overlapping view destroys lhsT before matmul.")
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
        if level == 4:
            return (error_text + f" The copied value has {val} elements but the destination "
                    f"has {dst}. Check the PSUM-to-SBUF tensor_copy: its result needs "
                    "a separate SBUF tile with the SAME shape as the PSUM tile, "
                    "(tile_m, tile_n). An input tile has shape (tile_k, tile_m) or "
                    "(tile_k, tile_n); do not reuse it for this result. Store the "
                    "result SBUF into the matching M and N output slice after the K loop.")
        if level == 2:
            return (error_text + " A partition row holds F1*F2 elements; copying it into "
                    "a 128-row tile does not broadcast it. Keep both transpose SBUF "
                    "tiles shaped (rows, F1*F2), with rows=min(128, P-start). "
                    "Copy input_tile[:, nl.ds(i*F2+j, 1)] to "
                    "output_tile[:, nl.ds(j*F1+i, 1)] using nisa.tensor_copy. "
                    "Both views have shape (rows, 1). Then DMA the output tile "
                    "to out[start:start+rows, :]. PSUM is unnecessary here.")
        return (error_text + f" You assigned {val} elements into a slice that holds {dst}. Assignment "
                f"does not reshape or broadcast either: the slice on the left and the value on the "
                f"right must have the SAME shape. If the value is bigger, you are writing a whole tile "
                f"where a slice belongs -- index the destination to match, e.g. "
                f"out[i*128:(i+1)*128, :] = tile. If it is smaller, you are looping over the wrong "
                f"dimension.")
    if level == 2:
        scalar_bound = re.search(
            r"Out-of-bound access for tensor .*? on dimension (\d+): "
            r"index (\d+) exceed dimension size of (\d+)", error_text)
        if scalar_bound:
            dim, index, size = scalar_bound.groups()
            return (error_text + f" Index {index} is outside dimension {dim}, whose "
                    f"valid indices are 0 through {int(size)-1}. Swapping indices "
                    "in tile[i,j] = tile[j,i] does not transpose a rectangular "
                    "tile safely and also overwrites source values. Keep P as "
                    "the partition axis and use separate (rows, F1*F2) input "
                    "and output SBUF tiles. For i in range F1 and j in range F2, "
                    "tensor_copy input[:, nl.ds(i*F2+j, 1)] into "
                    "output[:, nl.ds(j*F1+i, 1)]. Store every output row via DMA.")
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
    if m and level == 2:
        return (error_text + f" `nisa.{m.group(1)}` needs on-chip tiles, but the tile you passed "
                f"was allocated with buffer=nl.shared_hbm. Change the INPUT and OUTPUT WORK tiles "
                f"to buffer=nl.sbuf: nl.ndarray((rows, F), dtype=x.dtype, buffer=nl.sbuf). Keep "
                f"shared_hbm only for the single `out` tensor you return. Do not replace "
                f"tensor_copy with dma_copy for the element moves.")
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
        if level == 3:
            return (error_text + " The matmul result has TWO dimensions: "
                    "(lhsT.shape[1], rhs.shape[1]), or (M, N). lhsT.shape[1:] is only "
                    "(M,). Correct the output allocation AND any PSUM/SBUF allocation "
                    "derived from output.shape. Use a separate (M, N) result SBUF tile "
                    "for the PSUM-to-SBUF copy; keep the input tiles (K, M) and (K, N).")
        return (error_text + " Every SBUF and PSUM tile needs two dimensions: a partition dimension "
                "first, then a free dimension. A 1-D tile is not allowed, so write "
                "nl.ndarray((rows, cols), ...) and give a length-N vector the shape (1, N) or "
                "(N, 1) depending on which axis you are reducing over.")
    if "cannot reshape array of size" in error_text:
        return (error_text + " Do NOT reshape. nc_matmul consumes its operands along the "
                "PARTITION axis, so the layout is already what you need and reshaping fights it. "
                "lhsT arrives as [K, M] (K on partitions, the left operand pre-transposed) and rhs "
                "as [K, N] (K on partitions too); the result is [M, N]. For a single tile the whole "
                "kernel is: allocate two sbuf tiles with the operands' own shapes and dma_copy each "
                "operand straight in (no slicing, no reshaping -- they already fit); allocate a psum "
                "tile of shape (M, N) with dtype=nl.float32; call "
                "nisa.nc_matmul(dst=psum, stationary=lhs_tile, moving=rhs_tile), which contracts K "
                "and gives [M, N]; tensor_copy the psum tile into an sbuf tile; dma_copy that out to "
                "the shared_hbm result. Read M and N from the shapes (M=lhsT.shape[1], "
                "N=rhs.shape[1]), never with reshape.")
    m = re.search(r"module '([\w.]+)' has no attribute '(\w+)'", error_text)
    if m:
        return error_text + available_names(f"{m.group(1)}.{m.group(2)}")
    m = re.search(r"'(\w+)' object has no attribute '(\w+)'", error_text)
    if m:
        if level == 2 and m.groups() == ("int", "dtype"):
            return (error_text + " Shape dimensions and offsets are Python integers. "
                    "For a shape bound use Python min(128, P-start), not nl.min: "
                    "nl.min reduces a tensor and tries to read its dtype. Allocate "
                    "both SBUF tiles as (rows, F), load x[start:start+rows, :], and "
                    "take the element dtype from x.dtype, not from a shape integer.")
        return (error_text + f" A {m.group(1)} is not a numpy array, so it has no "
                f"`{m.group(2)}`. Use the nl/nisa functions instead.")
    return error_text

def _assemble(a, segments):
    """Join named prompt segments, dropping empties, and record their sizes on `a` for logging.

    The challenge doc calls a plot of tokens-per-segment "the single most interesting artifact you
    can bring", so every prompt is built from named parts and their char counts are logged.
    """
    a._last_seg = {k: len(v) for k, v in segments.items() if v}
    return "\n\n".join(v for v in segments.values() if v)


def _extras(a, level):
    """Optional prompt segments from the flag-gated tools: the tiling helper and the skill example.
    Both are offered as utilities/templates, never as the answer."""
    tiles_seg = ""
    if getattr(a, "tiles", False) and level >= 3:
        tiles_seg = (
            "You MAY use a tiling helper: `from nkitile import tiles`. `tiles(n, size)` returns a "
            "list of (start, length) pairs covering 0..n, the last one partial when n is not a "
            "multiple of size, e.g. tiles(300, 128) == [(0,128),(128,128),(256,44)]. Loop over it "
            "to cover a dimension in tiles and get the ragged final tile right automatically:\n"
            "  for m0, mm in tiles(M, 128):\n"
            "      ...  # allocate a tile of the chunk's own size mm and slice [m0:m0+mm]\n"
            "You still choose the loops, the layout, the PSUM accumulation and the copy-out.")
    skill_seg = skills.example_block(level) if getattr(a, "skills", False) else ""
    return tiles_seg, skill_seg


def _tools_preamble(a):
    """Instruction telling the model it may call the enabled tools before writing code."""
    bits = []
    if "doc" in a.tools:
        bits.append("To check an API, write a line `DOC: <name>` (e.g. `DOC: nisa.nc_matmul`), up "
                    "to 3, and you will be given the real signature and docstring.")
    if "probe" in a.tools:
        bits.append("To test an idea, put a tiny @nki.jit snippet in a ```probe``` block (up to 2); "
                    "it is run under nki.simulate and you get its printed shapes/values/errors.")
    if not bits:
        return ""
    return ("Before writing the kernel you may use these tools; ask only what you are unsure of, "
            "then write the kernel.\n" + "\n".join("- " + b for b in bits))


def first_prompt(level, terse=0, a=None):
    """Deliberately short, and it does NOT list the rules.

    Measured twice in this repo: hand a model an enumerated list of prohibitions and it audits
    itself against each one and returns nothing, while a bigger budget only buys more thinking.
    So the rules live in the checker. Generate freely, let the checker object, then send back one
    named change.
    """
    s = nkibench.LEVELS[level]
    import inspect
    # Optional, flag-gated prompt segments. Assigned once here so every branch below (including
    # the level-3 matmul card and the generic default) can reference them without a NameError.
    tiles_seg, skill_seg = _extras(a, level) if a is not None else ("", "")
    tools_seg = _tools_preamble(a) if a is not None and getattr(a, "tools", None) else ""
    if level == 1:
        # A task-specific card avoids teaching matmul's PSUM workflow to a
        # reduction. Retain the operation and scalar API when shortening retries.
        card = POOL_API_CARD if terse == 0 else (
            "Allocate with nl.ndarray(shape, dtype=..., buffer=nl.sbuf); return an "
            "nl.shared_hbm output. Move matching slices with nisa.dma_copy(dst=, src=). "
            "Use tile.ap([[stride, count], ...]) to group windows, nl.sum(view, axis=[3, 4]), "
            "then nisa.tensor_scalar(dst=, data=, op0=nl.multiply, operand0=).\n")
        if terse >= 2:
            card = "Scale with nisa.tensor_scalar(dst=, data=, op0=nl.multiply, operand0=).\n"
        return (
            f"Write an AWS Neuron NKI kernel named `{s['entry']}`, decorated with @nki.jit.\n"
            f"Compute exactly what this NumPy reference computes:\n\n"
            f"{inspect.getsource(s['ref'])}\n{POOL_METHOD}\n{card}\n"
            f"Import nki, nki.language as nl, and nki.isa as nisa. "
            f"Reply with ONE python code block containing the imports and function.")
    if level == 3:
        # A dedicated matmul card, mirroring level 1's dedicated pooling card. The seat-21
        # baseline cycled on a one-dimensional (M,) output allocation; the shape contract and
        # per-role allocation card name the fix instead of restating the verdict.
        card = MATMUL_API_CARD if terse == 0 else (
            "Use separate input SBUF tiles, a float32 (M, N) PSUM tile, and separate "
            "(M, N) result SBUF and HBM tiles. dma_copy loads inputs; "
            "nc_matmul(dst=, stationary=, moving=) writes PSUM; tensor_copy moves "
            "PSUM to result SBUF; dma_copy stores the result in HBM.\n")
        if terse >= 2:
            card = ("Use nc_matmul into float32 PSUM, tensor_copy into a separate (M, N) SBUF "
                    "tile, then dma_copy to HBM.\n")
        return _assemble(a or _Dummy(), dict(
            task=(f"Write an NKI kernel `{s['entry']}` decorated with @nki.jit.\n"
                  f"Match this NumPy reference:\n\n{inspect.getsource(s['ref'])}"),
            shapes=MATMUL_SHAPES, card=card, tiles=tiles_seg, skill=skill_seg, tools=tools_seg,
            reply=("Import nki, nki.language as nl, and nki.isa as nisa. "
                   "Reply with ONE complete python code block.")))
    if terse >= 2:
        # Last resort. Measured on this endpoint: one-sentence prompts answered in 300-700
        # tokens while every structured, rule-carrying prompt spiralled.
        pool_hint = (" The input is (C,H,W); the output is (C,H//p,W//p); load the FULL input then "
                     "reduce each pxp window, never size the loaded tile to the output."
                     if level == 1 else "")
        return _assemble(a or _Dummy(), dict(
            task=(f"Write a Python function `{s['entry']}` decorated with @nki.jit that computes "
                  f"the same thing as this, using nki.language as nl and nki.isa as nisa:\n\n"
                  f"{inspect.getsource(s['ref'])}{pool_hint}"),
            reply="Reply with one python code block."))
    if terse >= 1:
        # The matmul memory rules are the substance of levels 3 and 4, and the short prompt has to
        # carry them: measured, the agent cycled between "dst must be in ['psum']" and "moving must
        # be in ['sbuf']" because nothing told it where the operands live.
        mm = ("nisa.nc_matmul(dst=, stationary=, moving=) needs dst in nl.psum and both operands "
              "in nl.sbuf. So: dma_copy the operands HBM->sbuf, allocate a psum tile, nc_matmul "
              "into it, tensor_copy psum->sbuf, then dma_copy sbuf->the shared_hbm output you "
              "return. The left operand is already transposed, with K on the partition axis."
              if level >= 3 else "")
        # Level 1's substance is the pooling access pattern: load the whole input, reduce after.
        pool = (POOL_SHAPES + " Load the full input into an in_tensor.shape SBUF tile, then build "
                "an in_tile.ap([...]) view that groups each pool window onto the last axes and "
                "nl.sum over them, scaling by 1/(pool_size*pool_size). Write that into a SEPARATE "
                "smaller output tile." if level == 1 else "")
        return _assemble(a or _Dummy(), dict(
            task=(f"Write an AWS Neuron NKI kernel: a function `{s['entry']}` decorated with "
                  f"@nki.jit that computes what this reference computes.\n\n"
                  f"{inspect.getsource(s['ref'])}"),
            rules=(f"Allocate with nl.ndarray(shape, dtype=..., buffer=nl.sbuf), move data with "
                   f"nisa.dma_copy(dst=, src=), loop with nl.affine_range(n). A tile's partition "
                   f"dimension is at most {nkibench.PMAX}."),
            matmul=mm, pool=pool, tiles=tiles_seg, skill=skill_seg, tools=tools_seg,
            reply="Reply with one python code block."))
    # Level 1 gets the dedicated pooling card instead of the generic API card, mirroring the
    # matmul levels' dedicated card -- same pattern, different operation.
    pool_card = (POOL_SHAPES + "\n\n" + POOL_API_CARD) if level == 1 else ""
    return _assemble(a or _Dummy(), dict(
        task=(f"Write an AWS Neuron NKI kernel.\n\n"
              f"Operation: {s['op']}\n"
              f"Entry point: a function named `{s['entry']}`, decorated with `@nki.jit`.\n"
              f"It must compute exactly what this NumPy reference computes:\n\n"
              f"{inspect.getsource(s['ref'])}"),
        rules=(f"Hardware limits: a tile's partition dimension is at most {nkibench.PMAX}. For "
               f"matmul, the stationary free dimension is at most {nkibench.GEMM_STATIONARY_FMAX} "
               f"and the moving free dimension at most {nkibench.GEMM_MOVING_FMAX}.\n\n"
               f"Import nki, nki.language as nl, and nki.isa as nisa."),
        api_card=pool_card or API_CARD, tiles=tiles_seg, skill=skill_seg, tools=tools_seg,
        reply="Reply with ONE python code block containing the imports and the function. No prose."))


class _Dummy:
    """Lets first_prompt record segment sizes even when no args object is passed (e.g. a test)."""
    _last_seg = {}


def repair_prompt(level, source, feedback, a=None, ledger=""):
    """One named change, and the previous code. No rules list, no reference re-sent.

    The lesson this whole repo keeps re-learning: feeding a verifier's report back verbatim
    reproduces the same mistake, because a report says what is wrong and never what to do.
    """
    if level == 3:
        return (
            f"Repair this single-tile NKI matmul:\n\n```python\n{source}\n```\n\n"
            f"A checker reports:\n{feedback}\n\n"
            f"{MATMUL_SHAPES}\n{MATMUL_API_CARD}\n"
            f"Fix the reported allocation and any dependent buffer shapes or copies. "
            f"Preserve the entry point, arguments and required output dtype. "
            f"Reply with ONE complete python code block.")
    if level == 1:
        # A local API repair cannot rescue an algorithm that multiplies a window
        # by itself. Let the model replace that computation while preserving the
        # entry point and the original operation's shape contract.
        return (
            f"Repair this NKI average-pooling kernel:\n\n```python\n{source}\n```\n\n"
            f"A checker reports:\n{feedback}\n\n"
            f"Required output: [C, H//pool_size, W//pool_size]; use complete windows "
            f"of the [C, H, W] input and return the input dtype.\n"
            f"{POOL_METHOD}\n{POOL_API_CARD}\n"
            f"Fix the reported error and any computation that does not average a window. "
            f"You may replace the algorithm; preserve the entry point and arguments. "
            f"Reply with ONE complete python code block.")
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
                max_tokens=budget, temperature=a.temperature, top_p=a.top_p,
                chat_template_kwargs={"enable_thinking": a.think})
    if a.top_k is not None:
        body["top_k"] = a.top_k
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


def ask_with_tools(a, prompt, n):
    """One tool round-trip, then the kernel. If no tools are enabled this is just ask_parallel.

    The model may answer the tool-preamble with DOC:/```probe``` requests; we resolve them against
    the installed SDK and append the results to the prompt before asking for the kernel. The tool
    output is tracked as its own prompt segment so the token-breakdown plot shows its cost.
    """
    if not a.tools:
        return ask_parallel(a, prompt, n)
    # A single cheap probe of what the model wants to look up (1 sample; the asks are usually the
    # same regardless of sampling, and this keeps the round-trip from doubling cost per sample).
    first = ask(a, prompt)
    tool_out = agent_tools.gather_tool_output(first, a.tools)
    if tool_out:
        print(f"  tools: resolved {len(tool_out)} chars of DOC/PROBE output into the prompt")
        prompt = _assemble(a, dict(base=prompt, tool_output=tool_out,
                                   go="Now write the kernel. Reply with ONE python code block."))
    return ask_parallel(a, prompt, n)


def offline_answers(level, n, rnd):
    """No model. Replays the shipped reference, preceded by a deliberately broken version, so the
    loop and the feedback path can be exercised with no endpoint. Never report a number."""
    ref = open(f"reference_level{level}.py").read()
    if rnd == 0:
        broken = ref.replace("@nki.jit", "", 1)
        return [f"```python\n{broken}\n```"] * n
    return [f"```python\n{ref}\n```"] * n


# ---------------------------------------------------------------- the loop

def solve(a, level, log, rep=0):
    print(f"\n=========== level {level}: {nkibench.LEVELS[level]['op']} ===========")
    terse = a.terse
    prompt = first_prompt(level, terse, a)
    best = (0.0, None, "")
    tried, streak, seen = [], 0, {}
    latest = ("", "")
    beam = []            # top-K distinct (reward, src, feedback) kept across rounds for --beam
    for rnd in range(a.rounds):
        t0 = time.perf_counter()
        if a.offline:
            replies = offline_answers(level, a.samples, rnd)
        elif a.beam > 1 and beam:
            # Beam search: repair every kept candidate, not just the latest, with samples each.
            replies = []
            for _, bsrc, bfb in beam:
                replies += ask_parallel(a, repair_prompt(level, bsrc, bfb, a), a.samples)
        else:
            replies = ask_with_tools(a, prompt, a.samples)
        graded = []
        for s_i, reply in enumerate(replies):
            src = extract_code(reply)
            reward, parts, feedback = grade(src, level)
            graded.append((reward, src, feedback, parts))
            # exp/run/sample/t let a --repeat log be split back into runs, which is what makes a
            # solve RATE computable from the log (see analyze.py). Without them a repeated run is
            # one undifferentiated stream and the rate -- the only honest number -- is lost.
            log.write(json.dumps(dict(exp=a.exp, run=rep, level=level, round=rnd, sample=s_i,
                                      reward=reward, parts=parts, t=round(time.time(), 1),
                                      prompt_chars=len(prompt), reply_chars=len(reply),
                                      seg_chars=getattr(a, "_last_seg", {}),
                                      code=src, feedback=feedback)) + "\n")
        log.flush()
        graded.sort(key=lambda g: g[0], reverse=True)
        top = graded[0]
        if top[0] > best[0]:
            best = (top[0], top[1], top[2])
        # Beam: keep the top-K DISTINCT candidates (by code) for next round's repairs.
        if a.beam > 1:
            seen_src, beam = set(), []
            for reward, src, feedback, _ in graded:
                if (src or "").strip() and src not in seen_src:
                    seen_src.add(src)
                    beam.append((reward, src, feedback))
                if len(beam) >= a.beam:
                    break
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
            if getattr(a, "skills", False) or getattr(a, "calibration", False):
                skills.save(level, top[1])      # curriculum reuse and/or calibration input
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
        if level == 4 and not scaffolded and (latest[0] or "").strip():
            # Level 4's first useful failure often names a tile-dimension mistake. Do not
            # spend another full model round repeating prose guidance: put the explicit
            # M/N/K loop structure into the very first repair prompt.
            scaffolded = True
            ledger = "\n".join(f"- {t[:160]}" for t in dict.fromkeys(tried))
            prompt = (repair_prompt(level, latest[0], latest[1], scaffold=True)
                      + f"\n\nThis candidate failed; repair it using the scaffold.\n{ledger}")
            print("  level 4 escalation: adding the 3D tile-loop scaffold to the first repair")
            continue
        repeated_level4_failure = level == 4 and seen[top[2]] >= 2
        if (repeats >= 2 or repeated_level4_failure) and (best[1] or "").strip():
            # Sampling on this endpoint is greedy, so an unchanged prompt returns an unchanged
            # answer. Measured: the same TypeError 19 rounds running. Changing the prompt is the
            # only thing that can change the answer, so say what has already been tried.
            ledger = ("These approaches have already failed, so do something different:\n"
                      + "\n".join(f"- {t[:160]}" for t in dict.fromkeys(tried)))
            prompt = repair_prompt(level, latest[0], latest[1], a, ledger)
            print(f"  same failure {repeats}x — adding a ledger of {len(set(tried))} failed "
                  f"attempts to break the repeat")
            continue
        if not (latest[0] or "").strip():
            # Nothing came back to repair. Asking it to "fix" an empty code block produced a
            # 202-character prompt and, under greedy sampling, the identical non-answer six
            # rounds running. Shorten and re-ask instead.
            terse = min(terse + 1, 2)
            prompt = first_prompt(level, terse, a)
            print(f"  no code yet, so re-asking with a shorter prompt (terseness {terse})")
        else:
            prompt = repair_prompt(level, latest[0], latest[1], a)
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
    ap.add_argument("--exp", default="baseline",
                    help="experiment name, written on every log line so runs can be compared")
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
    # --- plan backlog, all OFF by default so the baseline stays runnable (protocol rule 1) ---
    ap.add_argument("--tools", default="",
                    help="comma-separated tools the model aims itself: doc,probe. DOC resolves "
                         "real NKI signatures/docstrings; PROBE runs a snippet under nki.simulate. "
                         "A pre-generation round lets the model ask before it writes the kernel.")
    ap.add_argument("--tiles", action="store_true",
                    help="offer the nkitile.tiles(n,size) helper in the first prompt (a tool, not "
                         "a worked example). Targets the level-4 'one tile for the whole tensor' wall.")
    ap.add_argument("--skills", action="store_true",
                    help="curriculum: save solved kernels and seed later levels with the nearest "
                         "OTHER solved kernel. Never seeds a reference or same-level tutorial kernel.")
    ap.add_argument("--beam", type=int, default=1,
                    help="keep the top-K distinct candidates and repair all of them each round, "
                         "instead of only the latest. Same wall-clock when K*samples == server seqs.")
    ap.add_argument("--heldout", action="store_true",
                    help="run the held-out levels (ragged matmul + added ops) instead of 1-4, to "
                         "test generalisation on operations never tuned on.")
    ap.add_argument("--calibration", action="store_true",
                    help="print a calibration label per level (VERIFIED-COMPILED / SIMULATED-ONLY "
                         "/ FAILED) derived from evidence, and re-check solved kernels on fresh seeds.")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=None,
                    help="pass top_k to the server (Qwen non-thinking recommends 20). Only honoured "
                         "if the server respects per-request sampling params -- check first.")
    a = ap.parse_args()
    a.tools = {t.strip() for t in a.tools.split(",") if t.strip()}
    a._last_seg = {}

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

    if a.heldout:
        levels = [n for n in sorted(nkibench.LEVELS) if n >= 10]
        if not levels:
            sys.exit("no held-out levels (>=10) are registered. They are added in nkibench.py "
                     "under 'held-out and hostile evaluation set'.")
    elif a.all:
        levels = sorted(nkibench.LEVELS)[:4]
    else:
        levels = [a.level or 1]
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

    if getattr(a, "calibration", False):
        # Say how sure we are, from evidence: fresh-seed re-check and a compile gate on each solved
        # kernel. This is the honesty deliverable -- a SIMULATED-ONLY kernel is reported as such.
        import calibrate
        print(f"\n=========== calibration ladder ===========")
        for lv in levels:
            path = os.path.join(skills.SKILL_DIR, f"level{lv}.py")
            if not os.path.exists(path):
                print(f"  level {lv}: FAILED (no kernel solved this run)")
                continue
            try:
                label, note = calibrate.label_for(path, lv)
            except Exception as e:
                label, note = "UNKNOWN", f"calibration error: {type(e).__name__}: {e}"
            print(f"  level {lv}: {label}  ({note})")

    print(f"\nattempts logged to {a.log}")


if __name__ == "__main__":
    main()
