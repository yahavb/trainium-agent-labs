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

CORE_CARD = (
    "NKI base card: import nki, nki.language as nl, nki.isa as nisa. Decorate the entry point "
    "with @nki.jit. Allocate with nl.ndarray(shape, dtype=..., buffer=nl.sbuf/nl.psum/"
    "nl.shared_hbm). Use nisa.dma_copy(dst=, src=) for HBM<->SBUF and return the shared_hbm "
    "output. NKI tensors are not NumPy arrays: do not use .reshape(), .mean(), .copy_from(), "
    "Python arithmetic like tile / scalar, or .shape on instruction results. Use explicit tiles, "
    "nl.sum(..., axis=[...]), nisa.tensor_scalar, and nisa.dma_copy."
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
    "reduction_patterns": (
        "Reduction pattern: copy or construct a tile in SBUF, form any needed strided view, reduce "
        "with nl.sum(..., axis=[...]), then write scaled results into a separate SBUF/output tile "
        "with nisa.tensor_scalar. Do not call .mean() on an NKI tensor, and do not use direct "
        "Python arithmetic on tile values."
    ),
    "reduction_axis": (
        "NKI reduction axis rule: nl.sum can reduce only the last contiguous dimensions of a tile. "
        "When a reduction spans non-adjacent logical axes, build or reorder the SBUF/access-pattern "
        "view so the reduced dimensions are trailing and contiguous, then reduce those trailing axes."
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
        "nisa.nc_matmul(dst=, stationary=, moving=, ...). Do not add guessed keyword arguments. "
        "Memory regions are values, not functions: write buffer=nl.sbuf, buffer=nl.psum, or "
        "buffer=nl.shared_hbm, never nl.sbuf() or nl.shared_hbm(). Use NKI dtypes such as "
        "nl.float32/nl.bfloat16 or an input tensor's dtype; do not pass np.float32."
    ),
}

REPAIR_CARD_NAMES = {
    "scale": ["reductions", "signatures"],
    "reduction_api": ["reductions", "signatures"],
    "reduction_axis": ["reduction_axis", "reduction_patterns", "reductions"],
    "nonfinite": ["dma_copy_shape", "matmul_psum"],
    "rule": ["api_core", "signatures"],
    "dma_shape": ["dma_copy_shape", "tile_limits"],
    "tile_rank": ["tile_rank", "dma_copy_shape"],
    "ragged": ["tile_limits", "dma_copy_shape"],
    "traffic": ["matmul_psum", "matmul_tiling"],
    "signature": ["signatures"],
    "generic": ["api_core"],
}

GENERIC_BASE_CARDS = ["api_core", "dma_copy_shape", "signatures"]

GENERIC_DOCS_CARDS = [
    "api_core",
    "dma_copy_shape",
    "tile_rank",
    "tile_limits",
    "access_patterns",
    "reduction_patterns",
    "reduction_axis",
    "matmul_psum",
    "matmul_tiling",
    "signatures",
]

FULL_GENERIC_GUIDE = """
Generic NKI kernel-writing guide.

Mental model:
- HBM tensors are the function inputs and returned output. SBUF/PSUM are on-chip tiles.
- Load from HBM into SBUF with nisa.dma_copy, compute on SBUF/PSUM tiles, then copy final SBUF
  tiles back to the returned shared_hbm output.
- Memory regions are constants, not constructors: use buffer=nl.sbuf, buffer=nl.psum,
  buffer=nl.shared_hbm. Never call nl.sbuf() or nl.shared_hbm().
- Prefer tile operations and nl.affine_range loops. Avoid Python scalar loops that assign one
  element at a time unless the verifier proves it is legal.

Allocation and dtype:
- nl.ndarray(shape, dtype=..., buffer=...) always needs dtype and buffer.
- Use input.dtype when matching an input tensor. Use nl.float32 or nl.bfloat16 for explicit NKI
  dtypes. Do not pass NumPy dtypes such as np.float32.
- SBUF and PSUM tiles must have at least two dimensions: partition dimension first, free dimension
  second or later. Represent a vector as (1, N) or (N, 1), not (N,).
- Tile-size limits are maximums. If a dimension exceeds a limit, loop over chunks; final chunks
  must use the remaining valid size.

DMA and shape rules:
- nisa.dma_copy(dst=tile_or_output_slice, src=tensor_or_slice) requires src and dst to have the
  same number of elements and compatible shape. It does not reshape, broadcast, pad, or slice for
  you.
- Allocate an SBUF tile to match the exact HBM slice being copied. Copy final computed SBUF tiles
  into the exact returned output slice.
- If the verifier says src/dst element counts differ, fix the tile/slice shape first.

Views, reductions, and scaling:
- NKI tensors are not NumPy arrays: do not use .reshape(), .mean(), .copy_from(), Python / on
  tiles, or .shape on instruction results.
- Use access-pattern/strided views when logical grouping is needed without reshaping the tile.
  Derive strides from the physical layout; do not invent zero strides.
- Use nl.sum(view, axis=[...]) for reductions. It returns a tile/instruction; it does not take a
  dst argument.
- nl.sum can reduce only trailing contiguous dimensions of the tile/view. If logical reduction axes
  are not trailing, form a view where the reduced dimensions are trailing and contiguous.
- Scale or multiply a tile with nisa.tensor_scalar(dst=..., data=..., op0=nl.multiply,
  operand0=constant). Write into a destination tile; do not rely on Python arithmetic.

Matmul rules:
- nisa.nc_matmul(dst=..., stationary=..., moving=...) writes into a PSUM dst tile.
- stationary and moving operands must be SBUF tiles. The left/stationary operand often has the K
  contraction dimension on the partition axis.
- For tiled matmul, keep one PSUM tile for an output block across the K loop, accumulate all K
  chunks into it, tensor_copy PSUM to SBUF, then dma_copy the final block to shared_hbm.

Verifier-driven repair:
- Treat the observed verifier result as evidence. The hint and docs may be imperfect.
- Fix the first concrete API/shape/rule failure before optimizing.
- Preserve fixes from previous failures: do not reintroduce missing dtype, called memory regions,
  1D SBUF/PSUM tiles, nisa.sum, .mean(), Python tile arithmetic, or wrong dma_copy shapes.
- If a new attempt lowers reward, return to the best-scoring kernel and make a smaller change.

Common legal API surface:
- Imports: import nki; import nki.language as nl; import nki.isa as nisa.
- Entry point: decorate the submitted function with @nki.jit and return an nl.shared_hbm tensor.
- Allocation: nl.ndarray(shape, dtype=..., buffer=nl.sbuf/nl.psum/nl.shared_hbm).
- Loops: use nl.affine_range for static kernel loops where possible.
- HBM movement: nisa.dma_copy(dst=..., src=...) between shared_hbm tensors/slices and SBUF tiles.
- On-chip copy: nisa.tensor_copy(dst=..., src=...) between SBUF/PSUM-compatible tiles.
- Reductions: nl.sum(tile_or_view, axis=[...], dtype=None, keepdims=False).
- Scaling: nisa.tensor_scalar(dst=..., data=..., op0=nl.multiply, operand0=constant).
- Matmul: nisa.nc_matmul(dst=psum_tile, stationary=sbuf_tile, moving=sbuf_tile).
- Dtypes: use input.dtype to preserve input type, or nl.float32/nl.bfloat16 for explicit NKI dtype.

Generic implementation checklist:
- Allocate the returned output first in shared HBM with the exact output shape.
- For each output tile, copy the needed input slice into SBUF, compute using NKI operations, then copy
  the computed SBUF tile into the matching output slice.
- Keep all SBUF/PSUM allocations at least 2D and within tile limits. If a logical value is scalar or
  vector-like, still represent it as a 2D tile.
- Preserve shape compatibility across copy calls. The source slice and destination tile should have
  the same element count and compatible layout.
- For reductions, arrange the tile/view so reduced axes are trailing and contiguous before nl.sum.
- Use explicit destination tiles for operations that write results; avoid expressions that assume NKI
  tiles behave like NumPy arrays.
- Handle partial final tiles by allocating/copying only the valid remaining shape.
"""

DOC_CHUNKS = {
    "generic_mental_model": """
Generic accelerator kernel mental model:
- The submitted function is not normal NumPy code. It is a device kernel with explicit memory.
- Inputs live in HBM. The output you return should be allocated in shared HBM.
- SBUF and PSUM are on-chip memories. You explicitly move data into them, operate there, then copy
  results back out.
- A correct kernel usually has this structure: allocate output in shared HBM; loop over output tiles;
  copy the needed input slice into SBUF; compute with NKI primitives; copy the computed tile into the
  matching output slice.
- Shape compatibility matters more than surface syntax. Every copy and compute op should have a
  source tile/view and destination tile with compatible element counts and layout.
""",
    "api_signatures": """
Core API signatures and legal names:
- import nki
- import nki.language as nl
- import nki.isa as nisa
- @nki.jit decorates the entry function.
- nl.ndarray(shape, dtype=..., buffer=...) allocates a tensor/tile. The buffer is one of nl.sbuf,
  nl.psum, or nl.shared_hbm. These are values, not functions.
- nl.affine_range(n) is the standard loop form inside kernels.
- nl.sum(x, axis, dtype=None, keepdims=False) reduces a tile/view and returns a tile-like value.
- nisa.dma_copy(dst=..., src=...) moves between HBM/shared_hbm and SBUF-compatible storage.
- nisa.tensor_copy(dst=..., src=...) copies between on-chip tile buffers such as PSUM and SBUF.
- nisa.tensor_scalar(dst=..., data=..., op0=nl.multiply, operand0=...) writes a scaled tile.
- nisa.nc_matmul(dst=..., stationary=..., moving=...) writes matmul results into a PSUM tile.
Do not invent helpers such as nl.value, nl.scalar, nl.sbuf_scalar, nisa.sum, tile.mean, tile.reshape,
or copy_from. If an operation is not listed here, assume it is unavailable.
""",
    "memory_and_dtype": """
Allocation, buffers, and dtype:
- nl.ndarray always needs a shape, dtype, and buffer.
- For output tensors, allocate with buffer=nl.shared_hbm and return that object.
- For input tiles, allocate with buffer=nl.sbuf and copy input slices into them.
- For matrix accumulation, allocate accumulator tiles with buffer=nl.psum and dtype=nl.float32 when
  accumulation precision matters.
- Use input_tensor.dtype when output should match the input dtype.
- Use nl.float32 or nl.bfloat16 for explicit NKI dtypes. Do not pass np.float32 or NumPy dtype objects.
- SBUF/PSUM tiles must be at least 2D. Use (1, N), (N, 1), or another 2D layout for vector/scalar-like
  intermediates.
- The partition dimension is the first tile dimension and has a hardware limit. If it exceeds the
  limit, split the work into chunks and handle the final partial chunk explicitly.
""",
    "copy_and_shape": """
DMA/copy shape rules:
- nisa.dma_copy does not reshape, broadcast, pad, or slice implicitly.
- The destination tile/slice and source tile/slice must have the same number of elements and a
  compatible shape.
- Allocate SBUF tiles to match the exact HBM slice you are copying.
- Copy final computed tiles into the exact output slice they represent.
- If the verifier reports different src/dst element counts, fix the slice/tile shape first rather
  than changing arithmetic.
- Avoid assigning individual Python scalar elements. Prefer tile-sized copies and tile operations.
- For ragged final chunks, use the remaining valid rows/columns instead of reading or writing past the
  logical tensor boundary.
""",
    "reductions_and_views": """
Reductions, views, and scaling:
- NKI tensors are not NumPy arrays. Do not use .reshape(), .mean(), direct Python division on tiles,
  or Python arithmetic expecting NumPy broadcasting.
- Use access-pattern/strided views when logical grouping is needed without physically reshaping.
- Derive access-pattern strides from the physical layout in SBUF; do not guess zero strides.
- nl.sum can reduce only trailing contiguous dimensions of the tile/view. If the logical reduction
  axes are not trailing, build a view where they become trailing and contiguous.
- nl.sum does not take dst. It returns a value/tile-like result that should be copied or scaled into
  a destination tile.
- To divide a reduced result by a constant, use nisa.tensor_scalar with an explicit destination tile
  and op0=nl.multiply, operand0=reciprocal.
""",
    "matmul_rules": """
Matmul-specific generic rules:
- nisa.nc_matmul writes into a PSUM destination tile.
- stationary and moving operands must be SBUF tiles.
- After matmul, copy PSUM to SBUF with nisa.tensor_copy, then copy SBUF to shared HBM output with
  nisa.dma_copy.
- For tiled matmul, keep one PSUM tile for an output block across the contraction loop and accumulate
  all contraction chunks before copying the final block out.
- Do not allocate a fresh PSUM for every contraction chunk unless the intended semantics reset the
  accumulator.
""",
    "repair_strategy": """
Verifier-driven repair strategy:
- Treat the observed verifier result as primary evidence. Hints and docs may be incomplete.
- First fix parse/rule/API failures, then simulation failures, then numerical mismatches, then traffic.
- Preserve all previous fixes. Do not reintroduce called memory regions, missing dtype, nisa.sum,
  NumPy dtype objects, 1D SBUF/PSUM tiles, .mean(), .reshape(), Python tile arithmetic, or bad copy
  shapes.
- If a candidate regresses to an earlier verifier phase, repair from the strongest archived candidate,
  not the latest broken text.
- Make one minimal change that directly explains the current failure.
""",
    "patch_format": """
Patch response format:
- Return a valid unified diff only.
- Include file headers exactly like --- a/kernel.py and +++ b/kernel.py.
- Include at least one @@ hunk header.
- Keep context lines in the hunk so the patch can be applied.
- Do not return prose, markdown explanation, or a full rewritten file when patch mode is requested.
""",
    "hardware_constraints": """
Generic hardware and tiling constraints:
- Kernels are explicit-memory programs. Whole-array host operations are not acceptable substitutes for
  tiled device work.
- Work should be expressed as loops over tiles. A tile is the unit of movement/computation, not the
  whole tensor unless the verifier shapes genuinely fit.
- Partition dimension comes first. Many failures come from treating a logical vector as a 1D tile,
  but SBUF/PSUM require two dimensions.
- Tile limits are upper bounds, not target shapes. If an input/output dimension is larger than the
  hardware tile limit, split it; if the last block is smaller, make the final tile/slice smaller.
- Reads and writes must stay inside the logical tensor bounds. Padding or reading past the valid
  shape can accidentally pass friendly tests and fail ragged/hostile tests.
- HBM traffic matters only after correctness. First produce correct output for all tested shapes,
  then reduce redundant copies and transfers.
- Do not hide work in framework calls such as NumPy, PyTorch, JAX, @ matrix multiply, input.T, or
  reshape/mean methods. The verifier treats those as rule violations because they bypass explicit
  kernel logic.
""",
    "verifier_interpretation": """
How to interpret verifier phases:
- Parse failure: the response is not valid Python code or a patch was malformed. Fix formatting first.
- Rule failure: the code uses a prohibited host/framework shortcut, wrong entry point, missing
  @nki.jit, over-large tile, or other statically visible violation. Fix rules before semantics.
- Load/import failure: the file imports but the expected function or legal modules are wrong. Keep
  imports limited to nki, nki.language as nl, and nki.isa as nisa.
- Simulation failure: the NKI simulator raised. This usually means wrong API signature, wrong buffer
  location, invalid tile shape/rank, invalid dtype, or incompatible copy shape.
- Numerical failure: the code ran but output differs. Use the named case/index/shape to decide whether
  the bug is output path, scale, reduction axis/order, missing initialization, ragged edge, or core
  arithmetic.
- Traffic/issue-bound warning: numerics are correct but movement is inefficient. Do not optimize
  traffic before correctness.
""",
    "common_failures": """
Common generic failure patterns and likely repairs:
- MemoryRegion object is not callable: change nl.sbuf(), nl.psum(), nl.shared_hbm() to nl.sbuf,
  nl.psum, nl.shared_hbm as buffer values.
- ndarray missing dtype: add dtype=input.dtype when matching an input/output, or dtype=nl.float32 for
  accumulation/intermediate math where explicit precision is needed.
- unknown dtype np.float32: use nl.float32 or an input tensor's dtype.
- module nki.isa has no sum: reductions are nl.sum, not nisa.sum.
- NKI tensor has no mean/reshape/copy_from: use explicit SBUF tiles, access-pattern views, nl.sum,
  nisa.tensor_scalar, and nisa.dma_copy/tensor_copy.
- SBUF/PSUM must have at least 2 dimensions: represent scalar/vector intermediates as 2D tiles.
- dma_copy element-count mismatch: make the destination tile shape match the source slice exactly.
- tensor_reduce axis must be trailing contiguous: build a tile/view where reduction axes are the final
  contiguous axes, then call nl.sum over those axes.
- unsupported Python operator on tile: use an explicit NKI instruction with a destination tile.
- output all zeros or nonfinite: check that every computed tile is copied into the returned
  shared_hbm output and that accumulators are initialized/written before reading.
""",
    "planning_checklist": """
Before writing or patching code, make this internal checklist concrete:
1. What is the exact input shape and output shape relationship from the reference?
2. What output tile is being produced by the current loop iteration?
3. Which input slice is needed for that output tile?
4. What SBUF/PSUM tiles are allocated, with what shape, dtype, and buffer?
5. Which copy moves HBM input into SBUF, and do src/dst element counts match?
6. Which NKI instruction computes the result, and where is its destination tile?
7. If reducing, are the reduction axes trailing and contiguous in the tile/view?
8. If scaling, is the result written into a destination tile rather than using Python arithmetic?
9. Which copy writes the computed SBUF tile to the exact shared_hbm output slice?
10. What previous failure must not be reintroduced?
""",
    "anti_patterns": """
Generic anti-patterns to avoid:
- Do not translate the NumPy reference literally. NumPy references often use reshape, transpose,
  vectorized broadcasting, mean, sum over arbitrary axes, slicing conveniences, or implicit temporary
  arrays. In NKI these usually need explicit tiles, loops, copies, views, and destination buffers.
- Do not allocate a single SBUF tile for an arbitrarily large tensor. Tile limits are real hardware
  constraints; loop over chunks when needed.
- Do not use Python range for kernel-space iteration when an NKI loop is expected; prefer
  nl.affine_range for static device loops.
- Do not create 1D SBUF/PSUM intermediates. Even if the logical value is one-dimensional, allocate a
  2D tile representation.
- Do not perform tile arithmetic with Python operators such as tile / scalar, tile + tile, or +=
  unless the API explicitly supports it. Use NKI instructions with destination tiles.
- Do not return an SBUF/PSUM tile. Return the shared_hbm output allocation.
- Do not forget the final write-back. A kernel can compute the right intermediate and still return
  zeros if it never copies into the returned output.
- Do not fix a numerical mismatch by changing the reference operation. The function must match the
  reference semantics; only implementation strategy should change.
- Do not optimize traffic before correctness. A fast wrong kernel is still wrong.
""",
    "debugging_playbook": """
Generic debugging playbook:
- If parsing fails, ignore kernel semantics and return syntactically complete Python only.
- If the entry point is missing or undecorated, fix the function name/import/decorator first.
- If a rule violation mentions a host shortcut, replace that shortcut with explicit tile movement and
  NKI operations.
- If an API name is missing, check namespace: reductions are in nl, movement/compute instructions are
  commonly in nisa, memory regions are nl values.
- If a buffer-location error says an operand must be in SBUF/PSUM, allocate/copy that operand into the
  required on-chip buffer before calling the instruction.
- If a copy shape mismatch occurs, write down source slice shape, destination tile shape, and output
  slice shape. Make them agree before touching arithmetic.
- If a reduction axis error occurs, inspect the physical tile/view layout. The reduced axes must be
  trailing contiguous axes in that tile/view.
- If a scale error occurs, preserve the reduction and copy structure, then fix only the scalar factor
  or denominator.
- If a ragged-edge mismatch occurs, handle final partial tile bounds separately from full tiles.
- If output is all zero or unchanged, inspect the output write-back path before compute logic.
- If a patch is requested, change only the smallest region needed; broad rewrites often lose working
  imports, decorators, dtype, buffers, or output copies.
""",
    "shape_reasoning": """
Shape reasoning rules:
- Keep separate the logical tensor shape, the HBM slice shape, the SBUF tile shape, the PSUM tile
  shape, and the output slice shape.
- The output shape comes from the reference. Do not infer output shape from a convenient tile shape.
- A loop index usually identifies a logical tile in the output. Convert that to exact input slice
  bounds and output slice bounds.
- For each copy, compare the source and destination element counts. If they differ, the copy is wrong
  even if the code parses.
- For reductions, distinguish the dimensions being preserved from the dimensions being reduced. The
  reduced dimensions disappear or become size one depending on API behavior; allocate the destination
  tile accordingly.
- When flattening logical groups into a free dimension, preserve the partition dimension as the first
  tile dimension and ensure access-pattern strides reflect the actual layout.
- For partial tiles, compute remaining rows/columns from tensor bounds rather than assuming full tile
  size.
""",
    "namespace_reference": """
Namespace reference and decision rules:
- Use nl for language-level constructs, tensor allocation, dtypes, memory-region constants, loops,
  arithmetic operators/constants exposed by the language, and reductions such as nl.sum.
- Use nisa for instruction-level operations that move/copy/compute into explicit destination tiles,
  such as dma_copy, tensor_copy, tensor_scalar, and nc_matmul.
- Memory regions are not allocators. The allocator is nl.ndarray; the region is passed through the
  buffer keyword.
- If an operation needs a destination, allocate that destination explicitly before the instruction.
- If an instruction error says a parameter is bound twice, use keyword arguments only and match the
  documented signature.
- If an instruction says a buffer must be sbuf or psum, first copy/allocate data into that buffer;
  do not pass shared_hbm directly to on-chip-only instructions.
- If a language object lacks a NumPy method, do not search for the same method in another namespace;
  reformulate the operation with the available NKI primitives.
- If a dtype or scalar constant is needed inside the kernel, prefer NKI language constants and
  instruction operands over host NumPy objects.
- If in doubt between changing API syntax and changing algorithm semantics, fix API syntax first and
  preserve the intended operation.
""",
}

DOC_PRIORITIES = {
    "signature": ["api_signatures", "memory_and_dtype", "repair_strategy"],
    "rule": ["api_signatures", "memory_and_dtype", "copy_and_shape", "repair_strategy"],
    "dma_shape": ["copy_and_shape", "memory_and_dtype", "repair_strategy"],
    "tile_rank": ["memory_and_dtype", "copy_and_shape", "repair_strategy"],
    "reduction_api": ["reductions_and_views", "api_signatures", "memory_and_dtype", "repair_strategy"],
    "reduction_axis": ["reductions_and_views", "copy_and_shape", "repair_strategy"],
    "scale": ["reductions_and_views", "api_signatures", "repair_strategy"],
    "ragged": ["copy_and_shape", "memory_and_dtype", "repair_strategy"],
    "traffic": ["copy_and_shape", "matmul_rules", "repair_strategy"],
    "nonfinite": ["copy_and_shape", "memory_and_dtype", "repair_strategy"],
    "generic": ["generic_mental_model", "hardware_constraints", "api_signatures", "memory_and_dtype",
                "copy_and_shape", "reductions_and_views", "verifier_interpretation",
                "common_failures", "namespace_reference", "debugging_playbook", "shape_reasoning",
                "anti_patterns", "planning_checklist", "repair_strategy"],
}


# ---------------------------------------------------------------- reward
#
# Graded, not pass/fail, so a near miss is distinguishable from nonsense and the loop has
# something to climb. Matches Project 1's shape: correctness dominates, and nothing else counts
# until the kernel is right.

WEIGHTS = dict(parses=0.1, rules=0.2, runs=0.2, correct=0.5)
PHASE_ORDER = {"empty": 0, "parse": 1, "rules": 2, "load": 3, "simulate": 4,
               "numerics": 5, "traffic": 6, "correct": 7}


class _PreflightFixer(ast.NodeTransformer):
    def __init__(self):
        self.fixes = []

    def visit_Call(self, node):
        self.generic_visit(node)
        if not (isinstance(node.func, ast.Attribute) and node.func.attr == "ndarray"):
            return node
        if not (isinstance(node.func.value, ast.Name) and node.func.value.id == "nl"):
            return node
        has_dtype = any(kw.arg == "dtype" for kw in node.keywords)
        if has_dtype or len(node.args) >= 2 or not node.args:
            return node
        shape = node.args[0]
        dtype = None
        if isinstance(shape, ast.Attribute) and shape.attr == "shape" and isinstance(shape.value, ast.Name):
            dtype = ast.Attribute(value=ast.Name(id=shape.value.id, ctx=ast.Load()),
                                  attr="dtype", ctx=ast.Load())
        elif isinstance(shape, ast.Subscript) and isinstance(shape.value, ast.Attribute) \
                and shape.value.attr == "shape" and isinstance(shape.value.value, ast.Name):
            dtype = ast.Attribute(value=ast.Name(id=shape.value.value.id, ctx=ast.Load()),
                                  attr="dtype", ctx=ast.Load())
        else:
            dtype = ast.Attribute(value=ast.Name(id="nl", ctx=ast.Load()),
                                  attr="float32", ctx=ast.Load())
        node.keywords.insert(0, ast.keyword(arg="dtype", value=dtype))
        self.fixes.append("added missing dtype to nl.ndarray")
        return node


def static_preflight_fix(source):
    """Patch obvious global NKI API typos before the verifier sees the candidate."""
    fixed = source
    replacements = [
        ("nl.sbuf()", "nl.sbuf"),
        ("nl.psum()", "nl.psum"),
        ("nl.shared_hbm()", "nl.shared_hbm"),
        ("nisa.sum", "nl.sum"),
        ("np.float32", "nl.float32"),
    ]
    fixes = []
    for old, new in replacements:
        if old in fixed:
            fixed = fixed.replace(old, new)
            fixes.append(f"{old} -> {new}")

    try:
        tree = ast.parse(fixed)
    except SyntaxError:
        return fixed, fixes
    fixer = _PreflightFixer()
    tree = fixer.visit(tree)
    ast.fix_missing_locations(tree)
    if fixer.fixes:
        fixed = ast.unparse(tree)
        fixes.extend(fixer.fixes)
    return fixed, fixes


def grade(source, level):
    """Returns (reward, parts, feedback). Feedback is an INSTRUCTION, never just a verdict."""
    parts = dict(parses=False, rules=False, runs=False, correct=False)

    if not source.strip():
        return 0.0, parts, ("No code came back. Reply with one python code block containing the "
                            "kernel and nothing else.")
    source, _ = static_preflight_fix(source)
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


def failure_phase(parts, feedback):
    text = compact_feedback(feedback).lower()
    if parts.get("correct"):
        return "correct"
    if not any(parts.values()):
        return "empty" if "no code came back" in text else "parse"
    if not parts.get("rules"):
        return "rules"
    if "could not be loaded" in text or "there is no module named" in text:
        return "load"
    if not parts.get("runs") or "raised " in text or "cannot simulate" in text:
        return "simulate"
    if "traffic " in text or "byte floor" in text or "issue-bound" in text:
        return "traffic"
    return "numerics"


def structured_failure(parts, feedback):
    phase = failure_phase(parts, feedback)
    key = failure_key(feedback)
    return {
        "phase": phase,
        "key": key,
        "summary": ledger_line(feedback),
        "compact": compact_feedback(feedback),
    }


def phase_score(parts, feedback):
    return PHASE_ORDER.get(failure_phase(parts, feedback), 0)


def update_candidate_archive(archive, candidate):
    reward, src, feedback, parts = candidate
    if not (src or "").strip():
        return
    phase = failure_phase(parts, feedback)
    old = archive.get(phase)
    if old is None or reward > old[0]:
        archive[phase] = candidate


def strongest_archived_candidate(archive):
    if not archive:
        return None
    return max(archive.values(), key=lambda c: (phase_score(c[3], c[2]), c[0]))


def choose_repair_base(top, best, latest, repeats, archive=None):
    """Choose which candidate to repair using verifier outcomes, not prompt wording."""
    top_reward, top_src, top_feedback, top_parts = top
    best_reward, best_src, best_feedback, best_parts = best
    archived = strongest_archived_candidate(archive or {})
    if archived and (archived[1] or "").strip():
        return (archived[1], archived[2]), "candidate_archive"
    if not (top_src or "").strip():
        return latest, "latest_empty"
    if (best_src or "").strip():
        return (best_src, best_feedback), "best"
    return (top_src, top_feedback), "latest_no_best"


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
        return (error_text + " Fix the access-pattern strides from the actual SBUF layout. The "
                "partition-axis stride should span the free dimensions behind one partition row; "
                "do not use zero or guessed strides. Keep reduced logical dimensions trailing if "
                "the view will feed nl.sum.")
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

    for name in GENERIC_BASE_CARDS:
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
    deduped = []
    for name in names:
        if name not in deduped:
            deduped.append(name)
    return deduped


def failure_key(feedback):
    text = compact_feedback(feedback)
    low = text.lower()
    if "memoryregion" in low or "not callable" in low:
        return "signature.memory_region_called"
    if "unexpected keyword argument" in low and "src_slice" in low:
        return "signature.unexpected_keyword_src_slice"
    if "unexpected keyword argument" in low:
        return "signature.unexpected_keyword"
    if "missing 1 required positional argument" in low and "dtype" in low:
        return "signature.ndarray_missing_dtype"
    if "unknown dtype" in low:
        return "signature.unknown_dtype"
    if "has no `mean`" in low or "attribute 'mean'" in low:
        return "reduction.mean_not_supported"
    if "nki.isa" in low and ("has no `sum`" in low or "has no attribute 'sum'" in low):
        return "signature.nisa_sum_not_found"
    if "tensor_reduce axis" in low or "last contiguous" in low:
        return "reduction.axes_not_trailing"
    if "same number of elements" in low or "dma_copy requires" in low:
        return "dma.shape_mismatch"
    if "at least 2 dimensions" in low or "1-d tile" in low or "1d" in low:
        return "tile.rank_1d"
    if "unsupported operand type" in low or "python operators" in low:
        return "api.python_operator_on_tile"
    if "has no `shape`" in low or "attribute 'shape'" in low:
        return "api.instruction_shape"
    if "nonfinite:" in low or "nan" in low:
        return "numeric.nonfinite"
    if "scale:" in low or "consistent scale" in low:
        return "numeric.scale"
    if "byte floor" in low or "traffic " in low or " transfers" in low or "issue-bound" in low:
        return "traffic.hbm"
    return "generic." + re.sub(r"[^a-z0-9]+", "_", low[:60]).strip("_")


def ledger_line(feedback):
    key = failure_key(feedback)
    compact = compact_feedback(feedback)
    evidence = error_context(feedback)
    summary = {
        "signature.memory_region_called": "memory regions called as functions; use buffer=nl.sbuf",
        "signature.unexpected_keyword_src_slice": "dma_copy does not take src_slice; slice the source tensor directly",
        "signature.unexpected_keyword": "invalid keyword argument; use the real NKI signature",
        "signature.ndarray_missing_dtype": "nl.ndarray missing dtype",
        "signature.unknown_dtype": "used non-NKI dtype; use nl.float32 or input.dtype, not np.float32",
        "reduction.mean_not_supported": ".mean() used on NKI tensor",
        "signature.nisa_sum_not_found": "used nisa.sum; use nl.sum",
        "reduction.axes_not_trailing": "nl.sum axes not trailing contiguous",
        "dma.shape_mismatch": "dma_copy src/dst shapes differ",
        "tile.rank_1d": "created 1D SBUF/PSUM tile",
        "api.python_operator_on_tile": "used Python operator on tile",
        "api.instruction_shape": "read .shape from instruction result",
        "numeric.nonfinite": "nonfinite output",
        "numeric.scale": "consistent scale error",
        "traffic.hbm": "excess HBM traffic",
    }.get(key, compact[:120])
    return f"{key}: {summary}"


def compact_ledger(failures, limit=5):
    by_key = {}
    for failure in failures:
        by_key[failure_key(failure)] = ledger_line(failure)
    return "\n".join(f"- {line}" for line in list(by_key.values())[-limit:])


def retrieved_card_names(level, category, feedback="", source="", max_cards=5):
    names = repair_card_names(level, category)
    names += select_context_cards(level, feedback=feedback, source=source, max_cards=max_cards)
    deduped = []
    for name in names:
        if name not in deduped:
            deduped.append(name)
    return deduped[:max_cards]


def known_invalid_patterns(failures):
    text = "\n".join(compact_feedback(f) for f in failures or []).lower()
    patterns = []
    checks = [
        ("memoryregion" in text or "not callable" in text,
         "do not call nl.sbuf/nl.psum/nl.shared_hbm; pass them as buffer=nl.sbuf"),
        ("unexpected keyword argument" in text and "src_slice" in text,
         "do not pass src_slice= to dma_copy; slice the source tensor directly in src=..."),
        ("unexpected keyword argument" in text,
         "remove unsupported keyword arguments and use the real NKI function signature"),
        ("missing 1 required positional argument" in text and "dtype" in text,
         "add explicit dtype=... to every nl.ndarray allocation"),
        ("unknown dtype" in text,
         "use NKI dtype constants such as nl.float32 or input.dtype; do not use np.float32"),
        ("has no `mean`" in text or "attribute 'mean'" in text,
         "replace .mean() with nl.sum plus nisa.tensor_scalar"),
        ("reshape" in text,
         "do not call .reshape() on NKI tensors; allocate/copy the intended tile shape directly"),
        ("copy_from" in text,
         "do not call .copy_from(); use nisa.dma_copy(dst=..., src=...)"),
        ("unsupported operand type" in text or "python operators" in text,
         "do not use Python arithmetic on tiles; use nisa.tensor_scalar or NKI instructions"),
        ("has no `shape`" in text or "attribute 'shape'" in text,
         "do not read .shape from NKI instruction results; keep explicit tile shapes"),
        ("tensor_reduce axis" in text or "last contiguous" in text,
         "make reduction axes trailing contiguous before calling nl.sum"),
    ]
    for ok, desc in checks:
        if ok and desc not in patterns:
            patterns.append(desc)
    return patterns


def invalid_patterns_text(failures):
    patterns = known_invalid_patterns(failures)
    if not patterns:
        return ""
    return "Known invalid patterns from earlier attempts; fix these too:\n" + "\n".join(
        f"- {p}" for p in patterns)


def history_text(failures, limit=8):
    if not failures:
        return ""
    lines = ["Repair history, newest last:"]
    seen = set()
    for i, failure in enumerate(failures[-limit:], 1):
        key = failure_key(failure)
        line = ledger_line(failure)
        marker = "repeat" if key in seen else "new"
        seen.add(key)
        lines.append(f"- attempt {i}: {marker} {line}")
    return "\n".join(lines)


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
    if "memoryregion" in low or "not callable" in low:
        return ("signature",
                "Fix only the memory-region allocation calls. nl.sbuf, nl.psum and nl.shared_hbm "
                "are values, not functions: use buffer=nl.sbuf or buffer=nl.shared_hbm, without "
                "parentheses.")
    if "unknown dtype" in low:
        return ("signature",
                "Fix only the dtype arguments. Use NKI dtype constants like nl.float32 or reuse an "
                "input tensor dtype; do not pass NumPy dtype objects such as np.float32.")
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
    if "byte floor" in low or "traffic " in low or " transfers" in low or "issue-bound" in low:
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


def dedupe(names):
    out = []
    for name in names:
        if name not in out:
            out.append(name)
    return out


def render_doc_chunks(names, token_budget):
    lines, used = [], 0
    for name in dedupe(names):
        text = DOC_CHUNKS.get(name, "").strip()
        if not text:
            continue
        cost = estimate_tokens(text) + 8
        if lines and used + cost > token_budget:
            continue
        lines.append(f"## {name}\n{text}")
        used += cost
    return "\n\n".join(lines), used


def initial_doc_names(style):
    if style == "minimal":
        return []
    names = ["generic_mental_model", "hardware_constraints", "api_signatures", "memory_and_dtype",
             "copy_and_shape", "reductions_and_views", "verifier_interpretation",
             "common_failures", "namespace_reference", "debugging_playbook", "shape_reasoning",
             "anti_patterns", "planning_checklist", "repair_strategy"]
    if style == "full-docs":
        names += ["matmul_rules", "patch_format"]
    return names


def repair_doc_names(category, feedback="", source="", patch_mode=False):
    key = failure_key(feedback)
    names = list(DOC_PRIORITIES.get(category, DOC_PRIORITIES["generic"]))
    if key in {"signature.memory_region_called", "signature.ndarray_missing_dtype",
               "signature.unknown_dtype", "signature.nisa_sum_not_found"}:
        names = ["api_signatures", "memory_and_dtype"] + names
    if key in {"tile.rank_1d"}:
        names = ["memory_and_dtype", "copy_and_shape"] + names
    if key in {"dma.shape_mismatch"}:
        names = ["copy_and_shape", "memory_and_dtype"] + names
    if key in {"reduction.mean_not_supported", "reduction.axes_not_trailing"}:
        names = ["reductions_and_views", "api_signatures"] + names
    if "nc_matmul" in source or "matmul" in feedback.lower() or "psum" in feedback.lower():
        names += ["matmul_rules"]
    if patch_mode:
        names = ["patch_format"] + names
    names += ["verifier_interpretation", "common_failures", "namespace_reference", "debugging_playbook",
              "shape_reasoning", "planning_checklist", "repair_strategy"]
    return dedupe(names)


def error_context(feedback, limit=1800):
    text = str(feedback or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit - 3].rstrip() + "..."


def doc_budget(context_budget, answer_budget, fixed_text="", minimum=800):
    target_prompt = max(1000, context_budget - answer_budget - 128)
    return max(minimum, target_prompt - estimate_tokens(fixed_text))


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


def start_card_names(level, style="minimal"):
    names = []
    if style in {"docs", "full-docs"}:
        names += GENERIC_DOCS_CARDS
    deduped = []
    for name in names:
        if name not in deduped:
            deduped.append(name)
    return deduped


def start_context(level, style="minimal", context_budget=8192, answer_budget=MIN_ANSWER_TOKENS,
                  fixed_text=""):
    text = render_context_cards(start_card_names(level, style))
    names = initial_doc_names(style)
    if names:
        chunks, _ = render_doc_chunks(names, doc_budget(context_budget, answer_budget, fixed_text))
        text = "\n\n".join(t for t in (text, chunks) if t)
    return text


def first_prompt(level, terse=0, style="minimal", context_budget=8192,
                 answer_budget=MIN_ANSWER_TOKENS):
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
        fixed = (f"Write a Python function `{s['entry']}` decorated with @nki.jit that computes "
                 f"the same thing as this, using nki.language as nl and nki.isa as nisa:\n\n"
                 f"{ref}\n")
        start_cards = start_context(level, style, context_budget, answer_budget, fixed)
        start_text = f"\n\n{start_cards}" if start_cards else ""
        # Last resort. Measured on this endpoint: one-sentence prompts answered in 300-700
        # tokens while every structured, rule-carrying prompt spiralled.
        return fixed + f"{CORE_CARD}{start_text}\nReply with one python code block."
    if terse >= 1:
        fixed = (f"Write an AWS Neuron NKI kernel: a function `{s['entry']}` decorated with "
                 f"@nki.jit that computes what this reference computes.\n\n"
                 f"{ref}\n")
        start_cards = start_context(level, style, context_budget, answer_budget, fixed)
        start_text = f"\n\n{start_cards}" if start_cards else ""
        return fixed + f"{CORE_CARD}{start_text}\nReply with one python code block."
    fixed = (
        f"Write an AWS Neuron NKI kernel.\n\n"
        f"Operation: {s['op']}\n"
        f"Entry point: a function named `{s['entry']}`, decorated with `@nki.jit`.\n"
        f"It must compute exactly what this NumPy reference computes:\n\n"
        f"{ref}\n\n")
    start_cards = start_context(level, style, context_budget, answer_budget, fixed)
    start_text = f"\n\n{start_cards}" if start_cards else ""
    return (fixed + f"{CORE_CARD}{start_text}\n\n"
            f"Reply with ONE python code block containing the imports and the function. No prose.")


def repair_prompt(level, source, feedback, tried=None, best_reward=None, current_reward=None,
                  context_budget=8192, answer_budget=MIN_ANSWER_TOKENS):
    """Evidence-weighted repair prompt.

    The checker report is the primary evidence. The category only retrieves compact docs and helps
    logs; it must not replace the observed failure, because a wrong category can make a good
    checker message worse.
    """
    compact = compact_feedback(feedback)
    evidence = error_context(feedback)
    category, instruction = distill_failure(compact)
    card_names = retrieved_card_names(level, category, feedback=compact, source=source, max_cards=10)
    ledger = compact_ledger(tried or [])
    ledger_text = f"\n\nPrevious unique failures to avoid repeating:\n{ledger}" if ledger else ""
    hist = history_text(tried or [])
    history = f"\n\n{hist}" if hist else ""
    invalid_text = invalid_patterns_text(tried or [])
    invalid_text = f"\n\n{invalid_text}" if invalid_text else ""
    score_text = ""
    if best_reward is not None or current_reward is not None:
        score_text = (f"\n\nScore context: best_reward={best_reward if best_reward is not None else '?'}; "
                      f"current_reward={current_reward if current_reward is not None else '?'}. "
                      "Preserve changes that improved reward and avoid regressions.")
    fixed = (
        f"This NKI kernel for {nkibench.LEVELS[level]['op']} is not right yet.\n\n"
        f"Candidate code:\n"
        f"```python\n{source}\n```\n\n"
        f"Observed verifier result, primary evidence:\n{evidence}\n\n"
        f"Verifier hint, may be imperfect:\n{instruction}\n\n"
    )
    doc_names = repair_doc_names(category, feedback=compact, source=source, patch_mode=False)
    docs, _ = render_doc_chunks(doc_names, doc_budget(
        context_budget, answer_budget,
        fixed + score_text + invalid_text + ledger_text + history))
    cards = "\n\n".join(t for t in (render_context_cards(card_names), docs) if t)
    return (
        fixed +
        f"{cards}\n\n"
        f"Use the observed verifier result as the main evidence. The hint and docs are supporting "
        f"context, not commands to follow blindly. Make the smallest code change that best explains "
        f"and fixes the observed failure. Also fix known invalid NKI API patterns from earlier "
        f"attempts. Keep unrelated logic unchanged.{score_text}{invalid_text}{ledger_text}{history}\n\n"
        f"Reply with ONE python code block.")


def patch_repair_prompt(level, source, feedback, tried=None, best_reward=None, current_reward=None,
                        context_budget=8192, answer_budget=MIN_ANSWER_TOKENS):
    compact = compact_feedback(feedback)
    evidence = error_context(feedback)
    category, instruction = distill_failure(compact)
    card_names = retrieved_card_names(level, category, feedback=compact, source=source, max_cards=10)
    ledger = compact_ledger(tried or [])
    ledger_text = f"\n\nPrevious unique failures to avoid repeating:\n{ledger}" if ledger else ""
    invalid_text = invalid_patterns_text(tried or [])
    invalid_text = f"\n\n{invalid_text}" if invalid_text else ""
    score_text = ""
    if best_reward is not None or current_reward is not None:
        score_text = (f"\n\nScore context: best_reward={best_reward if best_reward is not None else '?'}; "
                      f"current_reward={current_reward if current_reward is not None else '?'}.")
    fixed = (
        f"This NKI kernel for {nkibench.LEVELS[level]['op']} needs one small repair.\n\n"
        f"Current best candidate:\n"
        f"```python\n{source}\n```\n\n"
        f"Observed verifier result, primary evidence:\n{evidence}\n\n"
        f"Verifier hint, may be imperfect:\n{instruction}\n\n"
    )
    doc_names = repair_doc_names(category, feedback=compact, source=source, patch_mode=True)
    docs, _ = render_doc_chunks(doc_names, doc_budget(
        context_budget, answer_budget,
        fixed + score_text + invalid_text + ledger_text))
    cards = "\n\n".join(t for t in (render_context_cards(card_names), docs) if t)
    return (
        fixed +
        f"{cards}\n\n"
        f"Return a minimal unified diff patch against the current best candidate. The patch must "
        f"include --- a/kernel.py, +++ b/kernel.py, and at least one @@ hunk header. Do not rewrite "
        f"the whole file. Keep unrelated code unchanged."
        f"{score_text}{invalid_text}{ledger_text}\n\n"
        f"Reply with ONE ```diff code block and no prose.")


DIFF_BLOCK = re.compile(r"```(?:diff|patch)?\s*(.*?)```", re.S)


def extract_patch(text):
    text = text or ""
    blocks = DIFF_BLOCK.findall(text)
    for block in blocks:
        if "@@" in block and ("--- " in block or "+++ " in block):
            return block.strip()
    return text.strip() if "@@" in text and ("--- " in text or "+++ " in text) else ""


def _parse_hunk_header(line):
    m = re.match(r"@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@", line)
    if not m:
        raise ValueError(f"bad hunk header: {line}")
    return int(m.group(1))


def apply_unified_patch(source, patch):
    """Apply a simple unified diff to one in-memory source string."""
    if not patch.strip():
        raise ValueError("empty patch")
    old = source.splitlines()
    out, i, pos = [], 0, 0
    lines = patch.splitlines()
    saw_hunk = False
    while i < len(lines):
        line = lines[i]
        if line.startswith(("--- ", "+++ ", "diff ", "index ")):
            i += 1
            continue
        if not line.startswith("@@"):
            i += 1
            continue
        saw_hunk = True
        start = _parse_hunk_header(line) - 1
        if start < pos:
            raise ValueError("overlapping hunks")
        out.extend(old[pos:start])
        pos = start
        i += 1
        while i < len(lines) and not lines[i].startswith("@@"):
            h = lines[i]
            if h == r"\ No newline at end of file":
                i += 1
                continue
            if not h:
                raise ValueError("empty diff line without prefix")
            tag, text = h[0], h[1:]
            if tag == " ":
                if pos >= len(old) or old[pos] != text:
                    raise ValueError("patch context does not match source")
                out.append(old[pos])
                pos += 1
            elif tag == "-":
                if pos >= len(old) or old[pos] != text:
                    raise ValueError("patch removal does not match source")
                pos += 1
            elif tag == "+":
                out.append(text)
            else:
                raise ValueError(f"bad diff line: {h}")
            i += 1
    if not saw_hunk:
        raise ValueError("patch has no hunk header")
    out.extend(old[pos:])
    return "\n".join(out).rstrip() + "\n"


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
    print(f"level {level}: {nkibench.LEVELS[level]['op']}")
    for style in ("minimal", "docs", "full-docs"):
        first = first_prompt(level, style=style)
        first_budget = prompt_accounting(
            reference=ref, core=CORE_CARD, cards=start_context(level, style))
        print_budget(f"first prompt [{style}]", first_budget)
        first_cards = ["core_minimal"] + start_card_names(level, style)
        print(f"    chars={len(first)} cards={', '.join(first_cards)}")

    source = (
        "import nki\nimport nki.language as nl\nimport nki.isa as nisa\n\n"
        f"@nki.jit\ndef {nkibench.LEVELS[level]['entry']}(*args):\n"
        "    out = nl.ndarray(args[0].shape, dtype=args[0].dtype, buffer=nl.shared_hbm)\n"
        "    return out\n"
    )
    for name, feedback in AUDIT_FAILURES.items():
        cat, inst = distill_failure(feedback)
        cards = retrieved_card_names(level, cat, feedback=feedback, source=source)
        rendered = render_context_cards(cards)
        prompt = repair_prompt(level, source, feedback, tried=[feedback])
        budget = prompt_accounting(code=source, feedback=compact_feedback(feedback),
                                   cards=rendered,
                                   ledger=(compact_ledger([feedback]) + "\n"
                                           + invalid_patterns_text([feedback])),
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
    by_key = {}
    for r in rows:
        key = r.get("failure_key") or failure_key(r.get("feedback", r.get("compact_feedback", "")))
        by_key[key] = by_key.get(key, 0) + 1
    print("failure keys:")
    for key, n in sorted(by_key.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {key}: {n}")

    print("\nattempts:")
    for r in rows:
        b = r.get("prompt_budget") or {}
        cards = ",".join(r.get("context_cards") or [])
        docs = b.get("cards", 0) + b.get("core", 0) + b.get("reference", 0)
        evidence = b.get("feedback", 0) + b.get("ledger", 0)
        print(
            f"  L{r.get('level')} R{r.get('round')}: reward={r.get('reward'):.2f} "
            f"cat={r.get('failure_category', 'unknown')} "
            f"key={r.get('failure_key', '?')} "
            f"prompt~{b.get('total', len(str(r.get('prompt_chars', ''))) // 4)}tok "
            f"docs~{docs} evidence~{evidence} code~{b.get('code', 0)} "
            f"cards={cards or 'none'}"
        )


def show_attempt(path, round_n=None, index=None):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        print(f"{path}: no attempts")
        return
    if index is not None:
        row = rows[index]
    elif round_n is not None:
        matches = [r for r in rows if r.get("round") == round_n]
        if not matches:
            raise SystemExit(f"no attempt with round={round_n}")
        row = matches[0]
    else:
        row = rows[-1]

    print(f"level={row.get('level')} round={row.get('round')} reward={row.get('reward')}")
    print(f"category={row.get('failure_category')} mode={row.get('repair_mode', 'full')} "
          f"cards={','.join(row.get('context_cards') or [])}")
    print(f"budget={row.get('prompt_budget')}")
    print("\n========== PROMPT ==========")
    print(row.get("prompt", "<prompt was not logged; rerun with --dump-prompts>"))
    print("\n========== RAW REPLY ==========")
    print(row.get("raw_reply", "<reply was not logged; rerun with --dump-prompts>"))
    if row.get("patch") or row.get("patch_error"):
        print("\n========== PATCH ==========")
        print(row.get("patch", ""))
        if row.get("patch_error"):
            print(f"\nPATCH ERROR: {row.get('patch_error')}")
    print("\n========== EXTRACTED CODE ==========")
    print(row.get("code", ""))
    print("\n========== CHECKER ==========")
    print(row.get("feedback", ""))


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
    prompt = first_prompt(level, terse, a.prompt_style, a.context, a.max_tokens)
    prompt_cards = ["core_minimal"] + start_card_names(level, a.prompt_style)
    prompt_budget = prompt_accounting(
        reference=__import__("inspect").getsource(nkibench.LEVELS[level]["ref"]),
        core=CORE_CARD,
        cards=start_context(level, a.prompt_style, a.context, a.max_tokens))
    best = (0.0, None, "", dict(parses=False, rules=False, runs=False, correct=False))
    candidate_archive = {}
    tried, streak, seen = [], 0, {}
    latest = ("", "")
    patch_base = None
    for rnd in range(a.rounds):
        t0 = time.perf_counter()
        replies = (offline_answers(level, a.samples, rnd) if a.offline
                   else ask_parallel(a, prompt, a.samples))
        graded = []
        for reply in replies:
            patch_text, patch_error = "", ""
            if patch_base and a.repair_mode == "patch":
                patch_text = extract_patch(reply)
                try:
                    src = apply_unified_patch(patch_base[0], patch_text)
                except Exception as e:
                    patch_error = f"{type(e).__name__}: {e}"
                    fallback_src = extract_code(reply)
                    src = fallback_src if fallback_src.strip() else patch_base[0]
            else:
                src = extract_code(reply)
            src, preflight_fixes = static_preflight_fix(src)
            reward, parts, feedback = grade(src, level)
            if patch_error and src == patch_base[0]:
                reward = 0.0
                parts = dict(parses=False, rules=False, runs=False, correct=False)
                feedback = f"Patch could not be applied: {patch_error}. Return a valid unified diff."
            failure_category, repair_instruction = distill_failure(feedback)
            failure = structured_failure(parts, feedback)
            graded.append((reward, src, feedback, parts))
            if not patch_error:
                update_candidate_archive(candidate_archive, graded[-1])
            row = dict(level=level, round=rnd, reward=reward, parts=parts,
                       prompt_chars=len(prompt), reply_chars=len(reply),
                       context_cards=prompt_cards,
                       prompt_budget=prompt_budget,
                       failure_category=failure_category,
                       failure_key=failure_key(feedback),
                       failure=failure,
                       repair_instruction=repair_instruction,
                       compact_feedback=compact_feedback(feedback),
                       preflight_fixes=preflight_fixes,
                       repair_mode=("patch" if patch_base and a.repair_mode == "patch" else "full"),
                       patch=patch_text,
                       patch_error=patch_error,
                       code=src, feedback=feedback)
            if a.dump_prompts:
                row["prompt"] = prompt
                row["raw_reply"] = reply
            log.write(json.dumps(row) + "\n")
        log.flush()
        graded.sort(key=lambda g: g[0], reverse=True)
        top = graded[0]
        if top[0] > best[0]:
            best = (top[0], top[1], top[2], top[3])
        # Default to repairing the latest attempt so the prompt keeps changing. Fall back to the
        # best candidate only when verifier outcomes show a regression or a repeat loop.
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
        repair_base, repair_base_reason = choose_repair_base(top, best, latest, repeats,
                                                             candidate_archive)
        if repeats >= 2 and (repair_base[0] or "").strip():
            # Sampling on this endpoint is greedy, so an unchanged prompt returns an unchanged
            # answer. Measured: the same TypeError 19 rounds running. Changing the prompt is the
            # only thing that can change the answer, so say what has already been tried.
            ledger = "\n".join(f"- {t[:160]}" for t in dict.fromkeys(tried))
            make_prompt = patch_repair_prompt if a.repair_mode == "patch" else repair_prompt
            prompt = (make_prompt(level, repair_base[0], repair_base[1], tried,
                                  best_reward=best[0], current_reward=top[0],
                                  context_budget=a.context, answer_budget=a.max_tokens)
                      + f"\n\nThese approaches have already failed, so do something different:\n"
                        f"{ledger}")
            patch_base = repair_base if a.repair_mode == "patch" else None
            cat, inst = distill_failure(repair_base[1])
            prompt_cards = retrieved_card_names(level, cat, feedback=repair_base[1],
                                                source=repair_base[0])
            prompt_budget = prompt_accounting(
                code=repair_base[0], feedback=compact_feedback(repair_base[1]),
                cards=render_context_cards(prompt_cards),
                ledger=(compact_ledger(tried) + "\n" + invalid_patterns_text(tried)),
                instruction=inst)
            print(f"  same failure {repeats}x — adding a ledger of {len(set(tried))} failed "
                  f"attempts to break the repeat; repairing {repair_base_reason}")
            continue
        if not (repair_base[0] or "").strip():
            # Nothing came back to repair. Asking it to "fix" an empty code block produced a
            # 202-character prompt and, under greedy sampling, the identical non-answer six
            # rounds running. Shorten and re-ask instead.
            terse = min(terse + 1, 2)
            prompt = first_prompt(level, terse, a.prompt_style, a.context, a.max_tokens)
            prompt_cards = ["core_minimal"] + start_card_names(level, a.prompt_style)
            prompt_budget = prompt_accounting(
                reference=__import__("inspect").getsource(nkibench.LEVELS[level]["ref"]),
                core=CORE_CARD,
                cards=start_context(level, a.prompt_style, a.context, a.max_tokens))
            print(f"  no code yet, so re-asking with a shorter prompt (terseness {terse})")
        else:
            make_prompt = patch_repair_prompt if a.repair_mode == "patch" else repair_prompt
            prompt = make_prompt(level, repair_base[0], repair_base[1], tried,
                                 best_reward=best[0], current_reward=top[0],
                                 context_budget=a.context, answer_budget=a.max_tokens)
            patch_base = repair_base if a.repair_mode == "patch" else None
            cat, inst = distill_failure(repair_base[1])
            prompt_cards = retrieved_card_names(level, cat, feedback=repair_base[1],
                                                source=repair_base[0])
            prompt_budget = prompt_accounting(
                code=repair_base[0], feedback=compact_feedback(repair_base[1]),
                cards=render_context_cards(prompt_cards),
                ledger=(compact_ledger(tried) + "\n" + invalid_patterns_text(tried)),
                instruction=inst)
            if repair_base_reason != "latest":
                print(f"  repairing {repair_base_reason}")
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
    ap.add_argument("--repair-mode", choices=("patch", "full"), default="full",
                    help="repair by asking for a full rewritten code block, or experimental minimal unified diff")
    ap.add_argument("--give-up-after", type=int, default=4,
                    help="stop a level after this many identical failures in a row. Measured: 15 "
                         "was pure waste, because the prompt had stopped changing.")
    ap.add_argument("--terse", type=int, default=0, choices=(0, 1, 2),
                    help="starting prompt length. Measured on gpt-oss-20b: 0 produced 13,245 "
                         "chars of hidden reasoning and no answer, while 1 answered with code. "
                         "Qwen3-8B is fine at 0.")
    ap.add_argument("--context", type=int, default=8192,
                    help="the server's max-model-len; prompt + answer must fit inside it")
    ap.add_argument("--prompt-style", choices=("minimal", "docs", "full-docs"), default="full-docs",
                    help="initial prompt context: minimal cards, generic API docs, or expanded "
                         "generic API/error-repair docs")
    ap.add_argument("--think", action="store_true",
                    help="let the model reason first; costs budget, and it ran out")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--audit-context", action="store_true",
                    help="print prompt cards and token estimates for representative failures, "
                         "without calling the model")
    ap.add_argument("--summarize-log", metavar="PATH",
                    help="summarize an attempts JSONL log without calling the model")
    ap.add_argument("--dump-prompts", action="store_true",
                    help="store full prompt and raw model reply in the JSONL log")
    ap.add_argument("--show-attempt", metavar="PATH",
                    help="print one logged attempt, including prompt/reply if dumped")
    ap.add_argument("--show-round", type=int,
                    help="with --show-attempt, choose this round; defaults to last attempt")
    ap.add_argument("--show-index", type=int,
                    help="with --show-attempt, choose this zero-based JSONL row")
    a = ap.parse_args()

    if a.show_attempt:
        show_attempt(a.show_attempt, round_n=a.show_round, index=a.show_index)
        return

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
