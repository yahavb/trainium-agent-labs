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
                             enrich(f"raised {type(e).__name__}: {e}", level=level,
                                    shape=case)))
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
        if m and level in (5, 6, 7):
            m = hoist_result_hint(m)
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
For the one .ap() on the ORIGINAL full input tile (C,H,W), use exactly this layout; every pair is
[element_stride, count], and the first stride MUST be H*W:
  view = in_tile.ap([[H*W, C], [pool_size*W, H//pool_size],
                     [pool_size, W//pool_size], [W, pool_size], [1, pool_size]])
Then use nl.sum(view, axis=[3, 4]) and nisa.tensor_scalar with op0=nl.multiply to scale.
Do not use 0 or 1 as the first stride, and do not call .ap() on an already-created view.
"""

POOL_API_CARD = """NKI pooling primitives:
  nl.ndarray(shape, dtype=..., buffer=nl.sbuf) allocates an on-chip tile.
  nl.ndarray(shape, dtype=..., buffer=nl.shared_hbm) allocates the returned output.
  nisa.dma_copy(dst=tile, src=input_slice) loads a tile; both sides must have the
    same element count. Derive tile sizes from the input, with at most 128 channels.
    Preserve the channel axis when slicing, even for a single channel.
  tile.ap([[stride, count], ...]) creates a view, with strides in ELEMENTS. Each pair
    is [element_stride, count]. For the original in_tile of shape (C,H,W), the exact
    pooling view is:
    in_tile.ap([[H*W, C], [pool_size*W, H//pool_size],
                [pool_size, W//pool_size], [W, pool_size], [1, pool_size]])
    The first stride H*W is mandatory (it is the original tile's free-dimension size);
    do not use 0 or 1 there. Call .ap() once on in_tile, not on another view.
    This produces [channels, output_rows, output_cols, pool_rows, pool_cols].
  sums = nl.sum(window_view, axis=[3, 4]) reduces the two window axes.
  scaled = nl.ndarray(sums.shape, dtype=sums.dtype, buffer=nl.sbuf)
  nisa.tensor_scalar(dst=scaled, data=sums, op0=nl.multiply,
                     operand0=1.0 / (pool_size * pool_size)) scales the sums.
  nisa.dma_copy(dst=output_slice, src=scaled) stores the matching output tile.
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


HOIST_SHAPES = ("lhsT is (K, M) -- the left operand arrives TRANSPOSED, K on the partition axis -- "
                "rhs is (K, N), and the result is (M, N). This level is scored on HBM BYTES as "
                "well as correctness: a correct kernel that re-reads rhs for every output row "
                "tile is level 4 and fails here. Read each rhs tile ONCE per column slab and "
                "reuse it across every M tile.")

HOIST_API_CARD = """Tile sizes come from the hardware, so read them off nl.tile_size:
  tile_k = nl.tile_size.pmax                   # 128, the partition-axis maximum
  tile_m = nl.tile_size.gemm_stationary_fmax   # 128
  tile_n = nl.tile_size.gemm_moving_fmax       # 512
A tile may never have more than tile_k partition rows. To hold several K tiles at once, DO NOT
grow the partition axis -- pack them along the FREE axis and slice the one you need:
  rhs_cache = nl.ndarray((tile_k, (K // tile_k) * tile_n), dtype=rhs.dtype, buffer=nl.sbuf)
  ...
  rhs_cache[:, k * tile_n:(k + 1) * tile_n]    # the k-th K tile, still 2-D, still tile_k rows
The returned tensor is allocated ONCE, before every loop, and it is the only shared_hbm tile:
  result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)
Allocating it inside a loop gives each iteration a fresh uninitialised tile, so every tile but the
last reads back as NaN. It must not live in nl.sbuf either: an (M, N) SBUF tile breaks the
partition limit as soon as M > tile_k, and an SBUF tile is not what the caller receives.
Never write into lhsT or rhs; they belong to the caller.
The loop order is what buys the reuse -- N outermost, then the cache, then M, then K:
  for n in nl.affine_range(N // tile_n):
      rhs_cache = nl.ndarray((tile_k, (K // tile_k) * tile_n), dtype=rhs.dtype, buffer=nl.sbuf)
      for k in nl.affine_range(K // tile_k):          # fill the cache ONCE for this slab
          nisa.dma_copy(dst=rhs_cache[:, k * tile_n:(k + 1) * tile_n],
                        src=rhs[k * tile_k:(k + 1) * tile_k, n * tile_n:(n + 1) * tile_n])
      for m in nl.affine_range(M // tile_m):
          accum = nl.ndarray((tile_m, tile_n), dtype=nl.float32, buffer=nl.psum)
          for k in nl.affine_range(K // tile_k):      # accumulate into the SAME psum tile
              left = nl.ndarray((tile_k, tile_m), dtype=lhsT.dtype, buffer=nl.sbuf)
              nisa.dma_copy(dst=left, src=lhsT[k * tile_k:(k + 1) * tile_k,
                                               m * tile_m:(m + 1) * tile_m])
              nisa.nc_matmul(dst=accum, stationary=left,
                             moving=rhs_cache[:, k * tile_n:(k + 1) * tile_n])
          # Copy out through SBUF: psum cannot reach HBM directly, and `result` is the tile
          # allocated once above -- do NOT allocate it here.
          out_tile = nl.ndarray((tile_m, tile_n), dtype=lhsT.dtype, buffer=nl.sbuf)
          nisa.tensor_copy(dst=out_tile, src=accum)
          nisa.dma_copy(dst=result[m * tile_m:(m + 1) * tile_m,
                                   n * tile_n:(n + 1) * tile_n], src=out_tile)
  return result
Allocate `accum` per output tile OUTSIDE the K loop, never inside it: a fresh psum tile per K step
writes partial products out and reads them back, which is the traffic this level measures.
Derive K, M and N from lhsT.shape and rhs.shape; do not hard-code the test sizes.
"""

BLOCK_SHAPES = ("lhsT is (K, M) -- the left operand arrives TRANSPOSED, K on the partition axis "
                "-- rhs is (K, N), and the result is (M, N). This level is scored on HBM BYTES as "
                "well as correctness, and more tightly than the level below it. Caching ONE rhs "
                "column slab is not enough here: that still reloads each lhsT tile once per slab. "
                "Hold SEVERAL slabs resident at once and load each lhsT tile once per BLOCK of "
                "slabs, so rhs is read once, lhsT is read once per block, and the output is "
                "written once.")

BLOCK_API_CARD = """Tile sizes come from the hardware, so read them off nl.tile_size:
  tile_k = nl.tile_size.pmax                   # 128, the partition-axis maximum
  tile_m = nl.tile_size.gemm_stationary_fmax   # 128
  tile_n = nl.tile_size.gemm_moving_fmax       # 512
A tile may never have more than tile_k partition rows, so a cache holding several tiles packs
them along the FREE axis and slices the one it needs. Never grow the partition axis to fit more.
Choose the block size under a CAPACITY BOUND you state yourself, not from the shape -- that
choice is the substance of this level. Keep it a divisor of the tile count so every loop bound
stays an exact division and there is no ragged final block:
  k_tiles, n_tiles, m_tiles = K // tile_k, N // tile_n, M // tile_m
  BUDGET = 16384                                     # floats per partition you allow the cache
  affordable = max(1, BUDGET // (k_tiles * tile_n))  # slabs that fit the budget
  block_n = largest divisor of n_tiles that is <= affordable
  n_blocks = n_tiles // block_n
The returned tensor is allocated ONCE, before every loop, and is the only shared_hbm tile:
  result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)
Allocating it inside a loop hands each iteration a fresh uninitialised tile, so every tile but
the last reads back as NaN. Never write into lhsT or rhs; they belong to the caller.
The structure -- note that the lhsT load sits OUTSIDE the slab loop, which is the whole point:
  for nb in nl.affine_range(n_blocks):
      rhs_block = nl.ndarray((tile_k, block_n * k_tiles * tile_n),
                             dtype=rhs.dtype, buffer=nl.sbuf)
      for j in nl.affine_range(block_n):              # fill the block ONCE
          for k in nl.affine_range(k_tiles):
              col = (j * k_tiles + k) * tile_n
              n0 = (nb * block_n + j) * tile_n
              nisa.dma_copy(dst=rhs_block[:, col:col + tile_n],
                            src=rhs[k * tile_k:(k + 1) * tile_k, n0:n0 + tile_n])
      for m in nl.affine_range(m_tiles):
          left_block = nl.ndarray((tile_k, k_tiles * tile_m),
                                  dtype=lhsT.dtype, buffer=nl.sbuf)
          for k in nl.affine_range(k_tiles):          # ONCE per block, NOT once per slab
              nisa.dma_copy(dst=left_block[:, k * tile_m:(k + 1) * tile_m],
                            src=lhsT[k * tile_k:(k + 1) * tile_k,
                                     m * tile_m:(m + 1) * tile_m])
          for j in nl.affine_range(block_n):
              accum = nl.ndarray((tile_m, tile_n), dtype=nl.float32, buffer=nl.psum)
              for k in nl.affine_range(k_tiles):      # accumulate into the SAME psum tile
                  col = (j * k_tiles + k) * tile_n
                  nisa.nc_matmul(dst=accum,
                                 stationary=left_block[:, k * tile_m:(k + 1) * tile_m],
                                 moving=rhs_block[:, col:col + tile_n])
              out_tile = nl.ndarray((tile_m, tile_n), dtype=lhsT.dtype, buffer=nl.sbuf)
              nisa.tensor_copy(dst=out_tile, src=accum)
              n0 = (nb * block_n + j) * tile_n
              nisa.dma_copy(dst=result[m * tile_m:(m + 1) * tile_m, n0:n0 + tile_n],
                            src=out_tile)
  return result
Allocate `accum` per output tile OUTSIDE the K loop: a fresh psum tile per K step writes partial
products out and reads them back, which is the traffic this level measures.
Derive K, M and N from lhsT.shape and rhs.shape; do not hard-code the test sizes.
"""

FULL_SHAPES = ("lhsT is (K, M) -- the left operand arrives TRANSPOSED, K on the partition axis -- "
               "rhs is (K, N), and the result is (M, N). This is the fully blocked level: block "
               "K, N AND M. Reading each input once is necessary but no longer sufficient to make "
               "this level interesting -- what is left is issuing FEWER, LARGER transfers. "
               "Because lhsT is (K, M) its M axis is contiguous, so a whole block of M tiles is "
               "one rectangular slice and moves in a SINGLE dma_copy per K tile, instead of one "
               "per M tile. Same bytes, fewer transfers.")

FULL_API_CARD = """Tile sizes come from the hardware, so read them off nl.tile_size:
  tile_k = nl.tile_size.pmax                   # 128, the partition-axis maximum
  tile_m = nl.tile_size.gemm_stationary_fmax   # 128
  tile_n = nl.tile_size.gemm_moving_fmax       # 512
Caches pack their tiles along the FREE axis and keep the partition axis at tile_k; never grow the
partition axis to fit more. Size BOTH blocks under one capacity bound you state yourself, taking
the rhs block first and giving the lhsT block what is left, and keep each a divisor of its tile
count so no loop bound is ragged:
  k_tiles, n_tiles, m_tiles = K // tile_k, N // tile_n, M // tile_m
  BUDGET = 16384                                          # floats per partition, your choice
  block_n = largest divisor of n_tiles <= BUDGET // (k_tiles * tile_n)
  remaining = BUDGET - block_n * k_tiles * tile_n
  block_m = largest divisor of m_tiles <= remaining // (k_tiles * tile_m)
  n_blocks, m_blocks = n_tiles // block_n, m_tiles // block_m
The returned tensor is allocated ONCE, before every loop, and is the only shared_hbm tile:
  result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)
Allocating it inside a loop hands each iteration a fresh uninitialised tile, so every tile but
the last reads back as NaN. Never write into lhsT or rhs; they belong to the caller.
The structure -- note the single lhsT transfer per K tile covering `span` columns:
  for nb in nl.affine_range(n_blocks):
      rhs_block = nl.ndarray((tile_k, block_n * k_tiles * tile_n),
                             dtype=rhs.dtype, buffer=nl.sbuf)
      for j in nl.affine_range(block_n):
          for k in nl.affine_range(k_tiles):
              col = (j * k_tiles + k) * tile_n
              n0 = (nb * block_n + j) * tile_n
              nisa.dma_copy(dst=rhs_block[:, col:col + tile_n],
                            src=rhs[k * tile_k:(k + 1) * tile_k, n0:n0 + tile_n])
      for mb in nl.affine_range(m_blocks):
          m0 = mb * block_m * tile_m
          span = block_m * tile_m
          left_block = nl.ndarray((tile_k, k_tiles * span), dtype=lhsT.dtype, buffer=nl.sbuf)
          for k in nl.affine_range(k_tiles):          # ONE transfer per K tile, whole M block
              nisa.dma_copy(dst=left_block[:, k * span:(k + 1) * span],
                            src=lhsT[k * tile_k:(k + 1) * tile_k, m0:m0 + span])
          for i in nl.affine_range(block_m):          # the M tiles inside this block
              for j in nl.affine_range(block_n):
                  accum = nl.ndarray((tile_m, tile_n), dtype=nl.float32, buffer=nl.psum)
                  for k in nl.affine_range(k_tiles):  # accumulate into the SAME psum tile
                      left_col = k * span + i * tile_m
                      rhs_col = (j * k_tiles + k) * tile_n
                      nisa.nc_matmul(dst=accum,
                                     stationary=left_block[:, left_col:left_col + tile_m],
                                     moving=rhs_block[:, rhs_col:rhs_col + tile_n])
                  out_tile = nl.ndarray((tile_m, tile_n), dtype=lhsT.dtype, buffer=nl.sbuf)
                  nisa.tensor_copy(dst=out_tile, src=accum)
                  row, n0 = m0 + i * tile_m, (nb * block_n + j) * tile_n
                  nisa.dma_copy(dst=result[row:row + tile_m, n0:n0 + tile_n], src=out_tile)
  return result
The STORE cannot be merged the same way: an output tile is tile_m partition rows, and a block of
them would exceed the partition maximum, so store one tile at a time. Only the lhsT load merges.
Allocate `accum` per output tile OUTSIDE the K loop, never inside it.
Derive K, M and N from lhsT.shape and rhs.shape; do not hard-code the test sizes.
"""


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


def hoist_result_hint(msg):
    """Level 5 only: name the output-buffer mistake behind a non-finite or zeroed result.

    Measured on seat 21: once the rhs cache is right, the remaining failure is the copy-out. The
    model allocated `result = nl.ndarray((M, N), ..., buffer=nl.sbuf)` INSIDE the m loop, so every
    output tile but the last read back as NaN (3 of 4 tiles on K=256 M=256 N=1024), and returned an
    SBUF tile that never reached HBM. The generic verdict blames an uninitialised PSUM tile, which
    sends the model to re-check accumulation that was already correct.
    """
    if "NON-FINITE OUTPUT" in msg or "OUTPUT IS" in msg or "of the output is zero" in msg:
        return (msg + " For this level the accumulation is usually NOT the problem -- the output "
                "buffer is. Allocate the returned tensor exactly once, BEFORE every loop, as "
                "result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm). Allocating "
                "it inside the n or m loop hands each iteration a fresh uninitialised tile, so "
                "only the last tile written is real and the rest come back NaN. Then copy each "
                "finished tile out in two steps, because psum cannot reach HBM: "
                "nisa.tensor_copy(dst=out_tile, src=accum) into a (tile_m, tile_n) nl.sbuf tile, "
                "then nisa.dma_copy(dst=result[m*tile_m:(m+1)*tile_m, n*tile_n:(n+1)*tile_n], "
                "src=out_tile). Return `result`, and never write into lhsT or rhs.")
    return msg


def transpose_result_hint(msg):
    """Level 2 only: say WHY the checker's generic verdict happened for a transpose kernel.

    Measured on seat 21: kernels wrote the result back into x with dma_copy(dst=x[...]) and
    returned x or nothing ("MODIFIED ITS INPUT"), and kernels that copied only partition 0 with
    tile[nl.ds(0, 1), c] left every other row uninitialised ("NON-FINITE", first NaN at row 1).
    The generic messages blame PSUM or give no cause, so the model repeated both.
    """
    if "THE KERNEL MODIFIED ITS INPUT" in msg:
        return (msg + " For this level: do not use x as a destination anywhere (no "
                "dma_copy(dst=x[...]), no x[...] = ...). Load x into an SBUF tile, build the "
                "transposed row layout in a SECOND SBUF tile, then "
                "out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm); "
                "nisa.dma_copy(dst=out, src=output_tile); return out.")
    if "NON-FINITE OUTPUT" in msg:
        return (msg + " For this level the cause is output memory that was never written, not "
                "PSUM. Two common ways: (1) a view like tile[nl.ds(0, 1), c] or "
                "output_tile[i, c] touches one partition only; (2) the copies write into the "
                "INPUT tile (e.g. input_tile[...] = input_tile[...]) so the output tile that is "
                "stored stays empty. Every copy must read from the input tile and write the "
                "output tile, on whole-partition column views: "
                "nisa.tensor_copy(dst=output_tile[:, nl.ds(j*F1+i, 1)], "
                "src=input_tile[:, nl.ds(i*F2+j, 1)]) for each (i, j). Then "
                "nisa.dma_copy(dst=out, src=output_tile) with the full (rows, F) tile.")
    return msg


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
    """Add the real names when the failure is an invented API call.

    `level` is optional so existing callers keep working; it lets a level-specific message fire
    (used for the level-1 pooling load/reduce idiom below).
    """
    if (level == 2 and "positional argument" in error_text.lower()
            and "were given" in error_text.lower()):
        import inspect
        spec = nkibench.LEVELS[2]
        params = ", ".join(inspect.signature(spec["ref"]).parameters)
        return (error_text + f" The harness calls `{spec['entry']}` with exactly two arguments: "
                f"({params}). Define the entry point as `def {spec['entry']}({params}):` "
                "with `@nki.jit` above it. Keep `shape2D` as a runtime argument; do not drop it "
                "or hard-code the current test shape. Return the transposed data with the same "
                "shape and dtype as `x`.")
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
    if level == 1 and "ap() pattern has invalid partition stride" in error_text.lower():
        detail = (
            " The access pattern must be applied ONCE to the original full-input SBUF tile "
            "with shape (C,H,W), not to a pooled/output tile or another view. Replace the "
            "entire pattern with exactly `in_tile.ap([[H*W, C], "
            "[pool_size*W, H//pool_size], [pool_size, W//pool_size], "
            "[W, pool_size], [1, pool_size]])`. Each pair is [element_stride, count]. "
            "The first stride must be H*W because that is the original tile's free-dimension "
            "size; never use 0 or 1 there. This pattern gives "
            "[C, H//pool_size, W//pool_size, pool_size, pool_size]; reduce axes [3,4] and "
            "scale by 1/(pool_size*pool_size). Keep the full input tile shape for DMA."
        )
        if shape:
            dims = shape.get("shape")
            pool_size = shape.get("pool_size")
            if dims and pool_size:
                channels, height, width = dims
                p = pool_size
                exact = [
                    [height * width, channels],
                    [p * width, height // p],
                    [p, width // p],
                    [width, p],
                    [1, p],
                ]
                detail += (
                    f" For this failing case C,H,W={dims}, pool_size={p}, the exact "
                    f"numeric pattern is `{exact}`. In the general kernel, spell those "
                    "five pairs with the input shape variables as shown above; do not "
                    "copy the numeric test values as hard-coded constants."
                )
        return error_text + detail
    if level == 4 and (
            "module 'nki.isa' has no attribute 'fill'" in error_text
            or "module 'nki.language' has no attribute 'temporary'" in error_text):
        return (error_text + " Check whether output-writing loops execute: use tile_m<=128 and "
                "ceiling division for tile counts, with bounded edge slices. Allocate one PSUM "
                "per (M,N) tile outside its K loop, load both operands, accumulate every K tile, "
                "then copy to a matching SBUF and DMA to the output slice. The shipped pattern "
                "needs no manual fill; a new ndarray is not initialized to zero.")
    if level == 4:
        missing = re.search(
            r"module 'nki\.language' has no attribute '(dma_copy|nc_matmul|tensor_copy)'",
            error_text)
        if missing:
            name = missing.group(1)
            return (error_text + f" {name} belongs to nki.isa, not nki.language. "
                    f"Use nisa.{name}(...) after `import nki.isa as nisa`. "
                    "The Level 4 calls are nisa.dma_copy(dst=..., src=...), "
                    "nisa.nc_matmul(dst=..., stationary=..., moving=...), and "
                    "nisa.tensor_copy(dst=..., src=...). Keep operands in SBUF, "
                    "the accumulator in PSUM, and store each completed output tile once.")
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
    if m and level in (5, 6, 7):
        # The generic advice below -- "loop over the partition dimension in chunks" -- is actively
        # wrong on these levels: chunking the cache means re-reading it, which the byte bar
        # penalises. Measured on seat 21: the level-5 and level-6 baselines both cycled four
        # rounds on this message before this branch existed.
        which, got, mx = m.group(1), int(m.group(2)), int(m.group(3))
        return (error_text + f" A tile may have at most {mx} partition rows and you asked for "
                f"{got}, so the cache cannot hold the K axis on the partition axis. Do NOT fix "
                f"this by looping over the partition dimension in {mx}-row chunks -- re-reading "
                f"is exactly what this level's byte budget penalises. Keep the partition axis at "
                f"ONE K tile and pack the K tiles along the FREE axis instead: allocate "
                f"nl.ndarray(({mx}, (K // {mx}) * tile_n), dtype=rhs.dtype, buffer=nl.sbuf) and "
                f"address the k-th tile as cache[:, k * tile_n:(k + 1) * tile_n], which is still "
                f"2-D and still {mx} rows. Fill that cache once in the outer loop, then reuse it "
                f"for every M tile.")
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
                  r"(?:index range \[(\d+), (\d+)\]|start index (\d+)) "
                  r"exceed dimension size of (\d+)",
                  error_text)
    if m:
        dim = m.group(1)
        hi = int(m.group(3) or m.group(4))
        size = int(m.group(5))
        if level == 4:
            return (error_text + f" The requested start {hi} is not a valid tile origin for "
                    f"dimension {dim} of length {size}; compute origins only once: "
                    "m0=mi*TILE_M, n0=ni*TILE_N, k0=ki*TILE_K. Use these offsets in "
                    "lhsT[k0:k1,m0:m1], rhs[k0:k1,n0:n1], and out[m0:m1,n0:n1]. "
                    "Use min(start+tile_size, dimension) for bounds; do not scale an offset "
                    "twice, swap axes, or use tile indices directly as element offsets.")
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
    # Optional prompt segments are shared by every level-specific prompt branch.
    tiles_seg, skill_seg = _extras(a, level) if a is not None else ("", "")
    tools_seg = _tools_preamble(a) if a is not None and getattr(a, "tools", None) else ""
    if level == 2:
        # The harness supplies both parameters from ref_transpose2d. Spell out the exact public
        # entry signature so the model cannot emit a one-argument kernel that fails before NKI
        # simulation begins.
        return _assemble(a or _Dummy(), dict(
            task=(f"Write an AWS Neuron NKI kernel `{s['entry']}` decorated with @nki.jit that "
                  "transposes each row's free dimensions according to the given shape2D."),
            signature=(f"Required function signature, exactly: `def {s['entry']}(x, shape2D):`. "
                       "The harness calls this entry point with TWO positional arguments in this "
                       "order: x, then shape2D. Keep both parameters; shape2D is runtime input, "
                       "not a constant to hard-code."),
            shapes=("x has shape (P, F), shape2D is (F1, F2), and F=F1*F2. Transpose only the "
                    "two free dimensions within each partition row; return shape (P, F) and x.dtype."),
            method=("Keep the partition axis P intact. Use separate SBUF input and output tiles "
                    "with shape (rows, F), where rows=min(128, P-start). For each i in F1 and j "
                    "in F2, copy input[:, nl.ds(i*F2+j, 1)] to output[:, "
                    "nl.ds(j*F1+i, 1)] with nisa.tensor_copy. DMA each completed output tile "
                    "to the returned shared-HBM tensor."),
            imports="Import nki, nki.language as nl, and nki.isa as nisa.",
            tiles=tiles_seg, skills=skill_seg, tools=tools_seg,
            reply="Return one concise, complete Python code block with imports and the kernel."))
    if level == 4:
        card = TILED_MATMUL_API_CARD if terse == 0 else (
            "Use separate (tile_k,tile_m) and (tile_k,tile_n) SBUF operands, one "
            "(tile_m,tile_n) float32 PSUM per output tile outside the K loop, then "
            "copy PSUM to a same-shaped SBUF and store it. Call transfers and matmul "
            "only as nisa.dma_copy, nisa.nc_matmul, and nisa.tensor_copy; never nl.*.\n")
        if terse >= 2:
            card = ("Use nisa.dma_copy for transfers, nisa.nc_matmul for accumulation, "
                    "and nisa.tensor_copy to move the completed PSUM to SBUF.\n")
        return _assemble(a or _Dummy(), dict(
            task=(f"Write an AWS Neuron NKI kernel `{s['entry']}` decorated with @nki.jit. "
                  f"Compute exactly what this NumPy reference computes:\n\n"
                  f"{inspect.getsource(s['ref'])}"),
            rules=(f"Hardware limits: partition dimension <= {nkibench.PMAX}; tile M and N "
                   "as well as K. Include import nki, import nki.language as nl, and "
                   "import nki.isa as nisa."),
            matmul=TILED_MATMUL_METHOD, api_card=card, tiles=tiles_seg,
            skill=skill_seg, tools=tools_seg,
            reply="Reply with one complete Python code block containing imports and function."))
    if level == 1:
        # A task-specific card avoids teaching matmul's PSUM workflow to a
        # reduction. Retain the operation and scalar API when shortening retries.
        card = POOL_API_CARD if terse == 0 else (
            "Allocate with nl.ndarray(shape, dtype=..., buffer=nl.sbuf); return an "
            "nl.shared_hbm output. Move matching slices with nisa.dma_copy(dst=, src=). "
            "On the original (C,H,W) input tile use ap([[H*W,C],[p*W,H//p], "
            "[p,W//p],[W,p],[1,p]]) (pairs are [stride,count]; first stride MUST be H*W). "
            "Then nl.sum(view, axis=[3, 4]), "
            "then nisa.tensor_scalar(dst=, data=, op0=nl.multiply, operand0=).\n")
        if terse >= 2:
            card = ("On the original input tile (C,H,W), use ap([[H*W,C],[p*W,H//p], "
                    "[p,W//p],[W,p],[1,p]]); first stride MUST be H*W. Sum axes [3,4], "
                    "then scale with nisa.tensor_scalar(op0=nl.multiply).\n")
        return (
            f"Write an AWS Neuron NKI kernel named `{s['entry']}`, decorated with @nki.jit.\n"
            f"Compute exactly what this NumPy reference computes:\n\n"
            f"{inspect.getsource(s['ref'])}\n{POOL_METHOD}\n{card}\n"
            f"Import nki, nki.language as nl, and nki.isa as nisa. "
            f"Reply with ONE python code block containing the imports and function.")
    if level in (5, 6, 7):
        # Dedicated cards for the byte-budget levels, mirroring level 1's and level 3's. Each
        # level's substance is one specific idiom that the generic API card does not contain.
        shapes, full_card = {5: (HOIST_SHAPES, HOIST_API_CARD),
                             6: (BLOCK_SHAPES, BLOCK_API_CARD),
                             7: (FULL_SHAPES, FULL_API_CARD)}[level]
        card = full_card if terse == 0 else (
            "Pack several tiles along the FREE axis of one SBUF cache and keep the partition "
            "axis at nl.tile_size.pmax; never grow it. Allocate the returned (M, N) tensor ONCE "
            "before every loop with buffer=nl.shared_hbm, allocate the float32 PSUM accumulator "
            "per output tile OUTSIDE the K loop, then tensor_copy it into an SBUF tile and "
            "dma_copy that into the result slice.\n")
        if terse >= 2:
            card = ("Cache tiles along the free axis, keep the partition axis at 128, and "
                    "allocate the shared_hbm result once before all loops.\n")
        return (
            f"Write an AWS Neuron NKI kernel named `{s['entry']}`, decorated with @nki.jit.\n"
            f"Compute exactly what this NumPy reference computes:\n\n"
            f"{inspect.getsource(s['ref'])}\n{shapes}\n{card}\n"
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
        pool = (POOL_METHOD + " Load the full input into an in_tensor.shape SBUF tile, then build "
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
    pool_card = (POOL_METHOD + "\n\n" + POOL_API_CARD) if level == 1 else ""
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


def repair_prompt(level, source, feedback, a=None, ledger="", scaffold=False):
    """One named change, and the previous code. No rules list, no reference re-sent.

    The lesson this whole repo keeps re-learning: feeding a verifier's report back verbatim
    reproduces the same mistake, because a report says what is wrong and never what to do.
    """
    if (level == 2 and "positional argument" in feedback.lower()
            and "were given" in feedback.lower()):
        import inspect
        spec = nkibench.LEVELS[2]
        params = ", ".join(inspect.signature(spec["ref"]).parameters)
        return (
            f"Repair this NKI transpose kernel. The current code is:\n\n```python\n{source}\n```\n\n"
            f"Checker error: {feedback}\n\n"
            f"The harness calls `{spec['entry']}` with exactly two positional arguments: "
            f"({params}). The decorated entry-point signature must be exactly:\n"
            f"@nki.jit\ndef {spec['entry']}({params}):\n"
            "Preserve both parameters. Use `shape2D` at runtime; "
            "do not hard-code the current test's (3, 4). Keep the transpose implementation and "
            "return the output with x.shape and x.dtype. Reply with one concise, complete Python "
            "code block including imports.")
    if level in (5, 6, 7):
        # Without the contract on repair, a fix to one allocation drifts back into chunking,
        # which re-reads and still fails the bar.
        shapes, card = {5: (HOIST_SHAPES, HOIST_API_CARD),
                        6: (BLOCK_SHAPES, BLOCK_API_CARD),
                        7: (FULL_SHAPES, FULL_API_CARD)}[level]
        return (
            f"Repair this NKI matmul, which is scored on HBM traffic as well as correctness:"
            f"\n\n```python\n{source}\n```\n\n"
            f"A checker reports:\n{feedback}\n\n"
            f"{shapes}\n{card}\n"
            f"{ledger}"
            f"Fix the reported failure and any allocation or loop bound it depends on. Preserve "
            f"the entry point, arguments and required output dtype. "
            f"Reply with ONE complete python code block.")
    if level == 3:
        return (
            f"Repair this single-tile NKI matmul:\n\n```python\n{source}\n```\n\n"
            f"A checker reports:\n{feedback}\n\n"
            f"{MATMUL_SHAPES}\n{MATMUL_API_CARD}\n"
            f"Fix the reported allocation and any dependent buffer shapes or copies. "
            f"Preserve the entry point, arguments and required output dtype. "
            f"Reply with ONE complete python code block.")
    if level == 1:
        if "ap() pattern has invalid partition stride" in feedback.lower():
            return (
                f"Repair this NKI average-pooling kernel; preserve its entry point and arguments.\n\n"
                f"```python\n{source}\n```\n\n"
                f"The checker reports:\n{feedback}\n\n"
                "Change the access pattern on the EXISTING full-input SBUF tile only. "
                "Replace its current `.ap(...)` pattern with exactly:\n"
                "```python\n"
                "view = in_tile.ap([[H*W, C], [pool_size*W, H//pool_size], "
                "[pool_size, W//pool_size], [W, pool_size], [1, pool_size]])\n"
                "```\n"
                "Use the names that your kernel already uses for C, H, W, and the full-input "
                "tile. For the reported case C=32,H=32,W=32,pool_size=2, the concrete pairs "
                "are [[1024,32],[64,16],[2,16],[32,2],[1,2]]. Every pair is "
                "[element_stride,count]. The leading stride is H*W, never 0 or 1. Call `.ap()` "
                "once on the original (C,H,W) tile. Do not rewrite other parts of the kernel. "
                "Reply with one complete Python code block."
            )
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
    if level == 4:
        return _assemble(a or _Dummy(), dict(
            task=f"Repair this tiled NKI matmul kernel `{nkibench.LEVELS[level]['entry']}`.",
            previous=f"```python\n{source}\n```",
            feedback=f"A checker reports:\n{feedback}",
            method=TILED_MATMUL_METHOD,
            api_card=TILED_MATMUL_API_CARD,
            imports=("Return a COMPLETE replacement file, not a function fragment. Include these "
                     "exact imports before the decorator:\nimport nki\nimport nki.language as nl\n"
                     "import nki.isa as nisa"),
            scaffold=TILED_MATMUL_SKELETON if scaffold else "",
            ledger=ledger,
            reply=("Fix the reported failure while preserving the entry point, arguments, and "
                   "output dtype. Reply with one complete Python code block containing imports "
                   "and the function.")))
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
    """Insert missing NKI module imports in a Level 4 model response."""
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

    insertion_line = 1 if source.startswith("#!") else 0
    for index, node in enumerate(tree.body):
        is_docstring = (index == 0 and isinstance(node, ast.Expr)
                        and isinstance(node.value, ast.Constant)
                        and isinstance(node.value.value, str))
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
    scaffolded = False
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
            autofixes = []
            if level == 4:
                src, autofixes = ensure_level4_imports(src)
            reward, parts, feedback = grade(src, level)
            graded.append((reward, src, feedback, parts))
            # exp/run/sample/t let a --repeat log be split back into runs, which is what makes a
            # solve RATE computable from the log (see analyze.py). Without them a repeated run is
            # one undifferentiated stream and the rate -- the only honest number -- is lost.
            log.write(json.dumps(dict(exp=a.exp, run=rep, level=level, round=rnd, sample=s_i,
                                      reward=reward, parts=parts, t=round(time.time(), 1),
                                      prompt_chars=len(prompt), reply_chars=len(reply),
                                      seg_chars=getattr(a, "_last_seg", {}),
                                      scaffolded=scaffolded, autofixes=autofixes,
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
