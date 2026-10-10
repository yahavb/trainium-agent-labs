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
                             enrich(f"raised {type(e).__name__}: {e}", level=level,
                                    shape=case)))
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
            failure = enrich(m, level=level, shape=case) if level == 4 else m
            if (level == 4 and "NUMERICAL MISMATCH" in m
                    and (case["M"] > nkibench.GEMM_STATIONARY_FMAX
                         or case["N"] > nkibench.GEMM_MOVING_FMAX)):
                failure += (
                    " This case needs multiple output tiles, while the smaller one-tile cases "
                    "passed. Focus on the outer M/N tile coordinates and slice origins: use "
                    "m0=mi*TILE_M, m1=min(m0+TILE_M, M), n0=ni*TILE_N, and "
                    "n1=min(n0+TILE_N, N). For each K tile use "
                    "lhsT[k0:k1, m0:m1] and rhs[k0:k1, n0:n1], then store the completed "
                    "accumulator to out[m0:m1, n0:n1]. Do not use mi/ni directly as element "
                    "offsets or swap the M and N offsets. lhsT is already [K,M]; do not "
                    "transpose it. Allocate one (tile_m,tile_n) PSUM per output tile before "
                    "the K loop, accumulate every K tile into it, and write that output tile "
                    "once after the K loop."
                )
                index = re.search(r"at index \((\d+),\s*(\d+)\)", m)
                if index:
                    row, col = map(int, index.groups())
                    tile_m = nkibench.GEMM_STATIONARY_FMAX
                    tile_n = nkibench.GEMM_MOVING_FMAX
                    m0 = (row // tile_m) * tile_m
                    n0 = (col // tile_n) * tile_n
                    failure += (
                        f" The reported wrong element ({row},{col}) belongs to output tile "
                        f"mi={m0 // tile_m}, ni={n0 // tile_n}, whose correct origins are "
                        f"m0={m0}, n0={n0}. For this K={case['K']}, M={case['M']}, "
                        f"N={case['N']} shape, load lhsT[k0:k1, {m0}:{min(m0 + tile_m, case['M'])}] "
                        f"and rhs[k0:k1, {n0}:{min(n0 + tile_n, case['N'])}], then store that "
                        f"tile to out[{m0}:{min(m0 + tile_m, case['M'])}, "
                        f"{n0}:{min(n0 + tile_n, case['N'])}]. Do not change the K "
                        "accumulation to fix this coordinate mismatch."
                    )
            failures.append((nkibench.label(case, level), failure))
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


TILED_MATMUL_METHOD = """Read K, M = lhsT.shape and K_rhs, N = rhs.shape; return an (M, N) output.
Tile all three dimensions: K at most 128, M at most 128, N at most 512 per tile.
For each (M, N) output tile, allocate ONE float32 PSUM tile outside its K loop.
Accumulate all K tiles into it, then copy and store the completed output tile.
Use ceiling division (size+tile_size-1)//tile_size and Python min for edge sizes.
M=128 must execute an M tile; M//512 is zero and leaves output unwritten.
Slice lhsT as [k_start:k_end, m_start:m_end], since its axes are (K, M).
"""

TILED_MATMUL_API_CARD = """Use separate buffers with these tile shapes:
  stationary input SBUF: (tile_k, tile_m), loaded from lhsT's K and M ranges
  moving input SBUF:     (tile_k, tile_n), loaded from rhs's K and N ranges
  accumulator PSUM:     (tile_m, tile_n), dtype nl.float32
  result SBUF:          (tile_m, tile_n), dtype lhsT.dtype
  returned shared HBM:  (M, N), dtype lhsT.dtype
Allocate with nl.ndarray(shape, dtype=..., buffer=region), choosing nl.sbuf,
nl.psum or nl.shared_hbm for the region according to the buffer's role.
Use nl.affine_range for tile loops; derive bounds and partial-tile sizes from
the input dimensions. Every DMA source slice must match its destination tile.
The copy and matmul functions are in nki.isa, imported as nisa. Call them only
as nisa.dma_copy(...), nisa.nc_matmul(...), and nisa.tensor_copy(...).
There is no nl.dma_copy, nl.nc_matmul or nl.tensor_copy. Inside the K loop,
call nisa.dma_copy(dst=left_tile, src=lhsT[k0:k1, m0:m1]) and
nisa.dma_copy(dst=right_tile, src=rhs[k0:k1, n0:n1]), then call
nisa.nc_matmul(dst=accumulator, stationary=left_tile, moving=right_tile).
After that loop, call nisa.tensor_copy(dst=result_sbuf, src=accumulator).
The result SBUF must match the accumulator's shape, not an input tile's shape.
Store it with nisa.dma_copy into the matching M and N slice of the HBM output.
"""


# Escalation aid for repeated level-4 failures. Give the model the complete tile-indexed
# implementation pattern; its returned candidate still goes through the normal rules,
# simulator and numeric checks.
TILED_MATMUL_SKELETON = """Repair this kernel by returning this complete tiled matmul structure.
Preserve the index calculations exactly: mi and ni are tile indices, while m0 and n0 are
element offsets. Do not transpose lhsT; it is already [K,M]. The level's shapes are divisible
by these hardware tile limits.

```python
import nki
import nki.language as nl
import nki.isa as nisa

@nki.jit
def nki_matmul_tiled_(lhsT, rhs):
    K, M = lhsT.shape
    K_rhs, N = rhs.shape
    assert K == K_rhs
    TILE_M = nl.tile_size.gemm_stationary_fmax
    TILE_K = nl.tile_size.pmax
    TILE_N = nl.tile_size.gemm_moving_fmax
    assert M % TILE_M == 0 and K % TILE_K == 0 and N % TILE_N == 0
    out = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

    for mi in nl.affine_range(M // TILE_M):
        m0 = mi * TILE_M
        m1 = m0 + TILE_M
        for ni in nl.affine_range(N // TILE_N):
            n0 = ni * TILE_N
            n1 = n0 + TILE_N
            acc = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
            for ki in nl.affine_range(K // TILE_K):
                k0 = ki * TILE_K
                k1 = k0 + TILE_K
                left = nl.ndarray((TILE_K, TILE_M), dtype=lhsT.dtype, buffer=nl.sbuf)
                right = nl.ndarray((TILE_K, TILE_N), dtype=rhs.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=left, src=lhsT[k0:k1, m0:m1])
                nisa.dma_copy(dst=right, src=rhs[k0:k1, n0:n1])
                nisa.nc_matmul(dst=acc, stationary=left, moving=right)
            result = nl.ndarray((TILE_M, TILE_N), dtype=lhsT.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=result, src=acc)
            nisa.dma_copy(dst=out[m0:m1, n0:n1], src=result)
    return out
```
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


def enrich(error_text, level=None, shape=None):
    """Add the real names when the failure is an invented API call."""
    if level == 4 and "NON-FINITE OUTPUT" in error_text:
        detail = (
            " Check output-tile coverage: allocate one PSUM for each (M,N) output tile "
            "before its K loop, accumulate every K tile, copy PSUM to a same-shaped SBUF, "
            "then DMA that SBUF to the matching HBM output slice. Every output tile must "
            "be written exactly once; do not read an uninitialized PSUM/SBUF or rely on "
            "a fresh shared-HBM output being zeroed."
        )
        index = re.search(r"first at \((\d+),\s*(\d+)\)", error_text)
        if shape and index:
            row, col = map(int, index.groups())
            tile_m = nkibench.GEMM_STATIONARY_FMAX
            tile_n = nkibench.GEMM_MOVING_FMAX
            m0 = (row // tile_m) * tile_m
            n0 = (col // tile_n) * tile_n
            detail += (
                f" The failing shape is M={shape['M']}, N={shape['N']}; the first NaN at "
                f"({row},{col}) lies in output tile starting at m0={m0}, n0={n0} "
                f"(tile indices mi={m0 // tile_m}, ni={n0 // tile_n}). Check that this "
                f"tile's operands load lhsT[k0:k1, m0:m1] and rhs[k0:k1, n0:n1], and "
                f"that its result is stored to out[m0:m1, n0:n1]. Compute element offsets "
                f"as mi*TILE_M and ni*TILE_N, not directly from mi/ni; derive tile counts "
                f"with ceiling division and bound ends with min(start+tile_size, dimension)."
            )
        return error_text + detail
    if level == 4 and (
            "module 'nki.isa' has no attribute 'fill'" in error_text
            or "module 'nki.language' has no attribute 'temporary'" in error_text):
        return (error_text + " Check whether the output-writing loops execute: "
                "Use tile_m<=128 and ceiling division for tile counts, with bounded "
                "edge slices. Allocate one PSUM per (M, N) tile outside its K loop; "
                "load both operands, perform nc_matmul for every K tile, then "
                "tensor_copy to a matching SBUF and dma_copy to the output slice. "
                "The shipped matmul pattern needs no manual fill. nisa.fill and "
                "nl.temporary do not exist, and copying a fresh uninitialized "
                "ndarray does not initialize anything to zero. Ensure every "
                "output element is written before returning.")
    if level == 4:
        m = re.search(r"module 'nki\.language' has no attribute '(dma_copy|nc_matmul|tensor_copy)'",
                      error_text)
        if m:
            name = m.group(1)
            return (error_text + f" This operation belongs to nki.isa, not nki.language. "
                    f"Replace every nl.{name}(...) with nisa.{name}(...), after "
                    f"`import nki.isa as nisa`. The valid level-4 calls are "
                    "nisa.dma_copy(dst=..., src=...), "
                    "nisa.nc_matmul(dst=..., stationary=..., moving=...), and "
                    "nisa.tensor_copy(dst=..., src=...). Keep both input tiles in nl.sbuf, "
                    "the accumulator in nl.psum, and copy the completed output tile to HBM "
                    "only after the K loop.")
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
        if level == 4:
            return (error_text + f" This copy moves {src} elements into a {dst}-element tile. "
                    "For tiled matmul, match lhsT slices to (tile_k, tile_m) SBUF tiles "
                    "and rhs slices to (tile_k, tile_n) SBUF tiles. Match each result "
                    "tile to an output slice of (tile_m, tile_n). Tile M and N as well "
                    "as K; copying a whole input only works when that input fits one tile.")
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
        return (error_text + f" You assigned {val} elements into a slice that holds {dst}. Assignment "
                f"does not reshape or broadcast either: the slice on the left and the value on the "
                f"right must have the SAME shape. If the value is bigger, you are writing a whole tile "
                f"where a slice belongs -- index the destination to match, e.g. "
                f"out[i*128:(i+1)*128, :] = tile. If it is smaller, you are looping over the wrong "
                f"dimension.")
    m = re.search(r"Out-of-bound access for tensor .*? on dimension (\d+): "
                  r"(?:index range \[(\d+), (\d+)\]|start index (\d+)) "
                  r"exceed dimension size of (\d+)", error_text)
    if m:
        dim = m.group(1)
        hi = int(m.group(3) or m.group(4))
        size = int(m.group(5))
        if level == 4:
            return (error_text + f" The requested start {hi} is not a valid tile origin for "
                    f"dimension {dim} of length {size}; this level's tile origins are computed "
                    "once from tile indices, never from an already-scaled offset: "
                    "m0=mi*TILE_M, n0=ni*TILE_N, k0=ki*TILE_K. Use only those offsets in "
                    "the slices lhsT[k0:k1, m0:m1], rhs[k0:k1, n0:n1], and "
                    "out[m0:m1, n0:n1], with m1=min(m0+TILE_M,M), "
                    "n1=min(n0+TILE_N,N), and k1=min(k0+TILE_K,K). Do not multiply "
                    "an offset by a dimension or tile size a second time, swap the axes, or "
                    "use mi/ni/ki directly as element offsets. Check that every loop has "
                    "the correct bound: M tiles for mi, N tiles for ni, and K tiles for ki.")
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

def first_prompt(level, terse=0):
    """Deliberately short, and it does NOT list the rules.

    Measured twice in this repo: hand a model an enumerated list of prohibitions and it audits
    itself against each one and returns nothing, while a bigger budget only buys more thinking.
    So the rules live in the checker. Generate freely, let the checker object, then send back one
    named change.
    """
    s = nkibench.LEVELS[level]
    import inspect
    if level == 4:
        card = TILED_MATMUL_API_CARD if terse == 0 else (
            "Use distinct (tile_k, tile_m) and (tile_k, tile_n) input SBUF tiles, "
            "a (tile_m, tile_n) float32 PSUM accumulator and a separate matching "
            "result SBUF tile. Use ONLY nisa.dma_copy for transfers, "
            "nisa.nc_matmul for accumulation, and nisa.tensor_copy for PSUM-to-SBUF. "
            "Never call these as nl.* functions.\n")
        if terse >= 2:
            card = ("Use only nisa.dma_copy, nisa.nc_matmul, and nisa.tensor_copy; "
                    "never call these as nl.*. Keep one (tile_m, tile_n) PSUM accumulator "
                    "outside the K loop, copy it to a separate same-shaped SBUF tile after "
                    "the loop, then use nisa.dma_copy to store the output slice.\n")
        return (
            f"Write an NKI kernel `{s['entry']}` decorated with @nki.jit.\n"
            f"Match this NumPy reference:\n\n{inspect.getsource(s['ref'])}\n"
            f"{TILED_MATMUL_METHOD}\n{card}\n"
            f"Import nki, nki.language as nl, and nki.isa as nisa. "
            f"Reply with ONE complete python code block.")
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


def repair_prompt(level, source, feedback, scaffold=False):
    """One named change, and the previous code. No rules list, no reference re-sent.

    The lesson this whole repo keeps re-learning: feeding a verifier's report back verbatim
    reproduces the same mistake, because a report says what is wrong and never what to do.
    """
    if level == 4:
        return (
            f"Repair this tiled NKI matmul:\n\n```python\n{source}\n```\n\n"
            f"A checker reports:\n{feedback}\n\n"
            f"{TILED_MATMUL_METHOD}\n{TILED_MATMUL_API_CARD}\n"
            f"Return a COMPLETE REPLACEMENT FILE, not a function fragment. It MUST begin "
            f"with these exact imports, in this order, before any decorator or function:\n"
            f"import nki\n"
            f"import nki.language as nl\n"
            f"import nki.isa as nisa\n\n"
            f"Then include @nki.jit followed by the complete {nkibench.LEVELS[level]['entry']} function.\n"
            f"{TILED_MATMUL_SKELETON if scaffold else ''}\n"
            f"Fix the reported failure, buffer allocations and missing or zero-iteration tile loops. "
            f"Preserve the entry point, arguments and required output dtype. "
            f"Reply with ONE complete Python code block containing the whole replacement file, "
            f"including the three exact imports listed above.")
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


def ensure_level4_imports(source):
    """Restore the three required module aliases when a level-4 repair omits imports.

    The model sometimes returns only the decorated function on a repair turn. The checker then
    reports NameError for `nki` before it can inspect the kernel. Add missing module-scope imports
    deterministically and record them in the attempt log so this repair stays visible.
    """
    if not source.strip():
        return source, []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source, []

    imports = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.update((alias.name, alias.asname or alias.name.split(".")[0])
                           for alias in node.names)

    required = [
        (("nki", "nki"), "import nki"),
        (("nki.language", "nl"), "import nki.language as nl"),
        (("nki.isa", "nisa"), "import nki.isa as nisa"),
    ]
    missing = [line for alias, line in required if alias not in imports]
    if not missing:
        return source, []

    # Keep a shebang first, then a module docstring and any __future__ imports ahead of the
    # injected imports, as required by Python's module syntax.
    insertion_line = 1 if source.startswith("#!") else 0
    for index, node in enumerate(tree.body):
        is_docstring = (
            index == 0
            and isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        )
        is_future = isinstance(node, ast.ImportFrom) and node.module == "__future__"
        if is_docstring or is_future:
            insertion_line = max(insertion_line, node.end_lineno)
        else:
            break

    lines = source.splitlines(keepends=True)
    lines[insertion_line:insertion_line] = [line + "\n" for line in missing]
    return "".join(lines), missing


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
    best = (0.0, None, "")
    tried, streak, seen = [], 0, {}
    latest = ("", "")
    scaffolded = False
    for rnd in range(a.rounds):
        t0 = time.perf_counter()
        replies = (offline_answers(level, a.samples, rnd) if a.offline
                   else ask_parallel(a, prompt, a.samples))
        graded = []
        for reply in replies:
            src = extract_code(reply)
            autofixes = []
            if level == 4:
                src, autofixes = ensure_level4_imports(src)
            reward, parts, feedback = grade(src, level)
            graded.append((reward, src, feedback, parts))
            log.write(json.dumps(dict(level=level, round=rnd, reward=reward, parts=parts,
                                      prompt_chars=len(prompt), reply_chars=len(reply),
                                      scaffolded=scaffolded,
                                      autofixes=autofixes,
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
            ledger = "\n".join(f"- {t[:160]}" for t in dict.fromkeys(tried))
            add_scaffold = level == 4 and not scaffolded
            scaffolded = scaffolded or add_scaffold
            prompt = (repair_prompt(level, latest[0], latest[1], scaffold=scaffolded)
                      + f"\n\nThese approaches have already failed, so do something different:\n"
                        f"{ledger}")
            if repeats >= 2:
                print(f"  same failure {repeats}x — adding a ledger of {len(set(tried))} failed "
                      f"attempts to break the repeat")
            else:
                print(f"  recurring level 4 failure seen {seen[top[2]]}x — adding the failure "
                      "ledger and scaffold escalation")
            if add_scaffold:
                print("  level 4 escalation: adding an explicitly scaffolded 3D tile loop")
            continue
        if not (latest[0] or "").strip():
            # Nothing came back to repair. Asking it to "fix" an empty code block produced a
            # 202-character prompt and, under greedy sampling, the identical non-answer six
            # rounds running. Shorten and re-ask instead.
            terse = min(terse + 1, 2)
            prompt = first_prompt(level, terse)
            print(f"  no code yet, so re-asking with a shorter prompt (terseness {terse})")
        else:
            prompt = repair_prompt(level, latest[0], latest[1], scaffold=scaffolded)
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
