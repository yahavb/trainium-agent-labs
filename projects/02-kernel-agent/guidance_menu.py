"""guidance_menu.py — the option set for the guidance classifier.

One entry per repair action known from the day's runs. The classifier reads only
`id`, `name`, and `when` (the selection signal); the applier receives `exact` (the
complete change) and nothing more. Written the way a Jev Choice wants its criteria:
each option describes the situation that should trigger it, not just a label.
"""

MENU = [
    {
        "id": "stack_stationary_operand",
        "name": "Stack the stationary operand into a reused cache",
        "when": (
            "The kernel runs but the numbers are wrong, and the checker diagnosed the "
            "STATIONARY (left, lhsT) operand as stuck on its first k-chunk: the numbers equal "
            "lhsT chunk 0 multiplied against every rhs chunk. Also use this when the lhs tile "
            "is reloaded inside the k loop even though it does not depend on the inner loop. "
            "Typical evidence: correct bytes on some shapes but wrong numbers on others."
        ),
        "exact": (
            "Replace the stationary operand load with a stacked SBUF cache that is loaded once "
            "and reused across n:\n"
            "1) Before the m/n loops, allocate:\n"
            "   cache = nl.ndarray((TILE_K, k_tiles * M), dtype=lhsT.dtype, buffer=nl.sbuf)\n"
            "   where k_tiles = K // TILE_K.\n"
            "2) Load every k-chunk exactly once, before the loops:\n"
            "   for kk in nl.affine_range(k_tiles):\n"
            "       nisa.dma_copy(dst=cache[:, kk * M:(kk + 1) * M],\n"
            "                     src=lhsT[kk * TILE_K:(kk + 1) * TILE_K, :])\n"
            "   The source slice is [TILE_K, M] -- the FULL M width -- so the destination "
            "slice must also have M columns. The cache width is k_tiles * M, NOT k_tiles * "
            "TILE_K, and the chunk offset is kk * M, not kk * TILE_K.\n"
            "3) Inside the k loop, use the cache slice as the stationary operand:\n"
            "   nisa.nc_matmul(dst=res_psum,\n"
            "                  stationary=cache[:, k * M + m * TILE_M : k * M + (m + 1) * TILE_M],\n"
            "                  moving=rhs_tile)\n"
            "   The slice stays [TILE_K, TILE_M] with K on the partition axis.\n"
            "4) Delete the old per-iteration lhsT load. Change nothing else."
        ),
    },
    {
        "id": "fix_cache_stride_and_width",
        "name": "Fix a stacked cache's width and chunk stride",
        "when": (
            "A dma_copy fails with \"requires src and dst to have the same number of elements\" "
            "after a stacked cache was introduced. The cache's column width or the destination "
            "offset does not match the full-width source chunk -- the classic form is a cache "
            "sized by k_tiles * TILE_K (or similar) when the source chunk is the full width of "
            "the operand, so src and dst element counts differ."
        ),
        "exact": (
            "Fix the cache geometry so every dma_copy moves exactly the full-width chunk:\n"
            "1) Cache column width = k_tiles * W, where W is the operand's full width (M for "
            "lhsT, N for rhs).\n"
            "2) Each chunk's destination slice is cache[:, kk * W:(kk + 1) * W] -- length W.\n"
            "3) Each chunk's source is the full-width slice (e.g. lhsT[kk * TILE_K:(kk + 1) * "
            "TILE_K, :] with ':' for the full second dimension).\n"
            "4) Operand views inside the k loop keep the same offset arithmetic: "
            "cache[:, k * W + offset : k * W + offset + TILE], with TILE the tile size "
            "(TILE_M for lhsT's consumer, TILE_N for rhs).\n"
            "Change only the cache geometry and the slices that reference it."
        ),
    },
    {
        "id": "stack_moving_operand_symmetrically",
        "name": "Give the moving operand the same cached treatment",
        "when": (
            "The kernel is correct on every shape (no numerical failures, valid=True) but still "
            "moves more than the byte floor: the rhs tile is re-read from HBM for every m block "
            "(M // TILE_M times), while the lhs is already cached. The only failure text, if "
            "any, is the traffic-gate message about moving too many bytes."
        ),
        "exact": (
            "Apply to the rhs (moving) operand exactly the treatment lhsT already received:\n"
            "1) Allocate: rhs_cache = nl.ndarray((TILE_K, k_tiles * N), dtype=rhs.dtype, "
            "buffer=nl.sbuf)\n"
            "2) Load once before the loops:\n"
            "   for kk in nl.affine_range(k_tiles):\n"
            "       nisa.dma_copy(dst=rhs_cache[:, kk * N:(kk + 1) * N],\n"
            "                     src=rhs[kk * TILE_K:(kk + 1) * TILE_K, :])\n"
            "3) In the k loop, use as the moving operand:\n"
            "   nisa.nc_matmul(dst=res_psum,\n"
            "                  stationary=cache[:, k * M + m * TILE_M : k * M + (m + 1) * TILE_M],\n"
            "                  moving=rhs_cache[:, k * N + n * TILE_N : k * N + (n + 1) * TILE_N])\n"
            "4) Delete the per-iteration rhs dma_copy. Change nothing else."
        ),
    },
    {
        "id": "keep_k_on_partition_axis",
        "name": "Keep K on the partition axis in every slice",
        "when": (
            "nc_matmul or a load fails with a complaint about the partition dimension or a "
            "shape mismatch after operands were restacked: an operand slice has K on the wrong "
            "axis (sliced along K, or a transposed view). The partition axis must carry K."
        ),
        "exact": (
            "Restore the [TILE_K, TILE_out] orientation with K on axis 0 for both matmul "
            "operands:\n"
            "1) Slice caches only along the second axis: cache[:, offset : offset + TILE].\n"
            "2) Never slice or transpose along the K axis; the first dimension of both operands "
            "is TILE_K.\n"
            "3) Allocate caches as (TILE_K, ...) and partial-sum tiles as (TILE_M, TILE_N).\n"
            "Change only the offending slices and allocations."
        ),
    },
    {
        "id": "single_psum_accumulation",
        "name": "Accumulate all k-chunks into one PSUM tile",
        "when": (
            "The output contains non-finite values, or the checker diagnosed that only one "
            "k-chunk contributed (or a partial-sum tile is read before being written). The "
            "PSUM tile is allocated inside the k loop or reallocated per chunk instead of once "
            "per output tile."
        ),
        "exact": (
            "Allocate ONE PSUM tile per output tile before the k loop and accumulate every "
            "chunk into it:\n"
            "1) res_psum = nl.ndarray((TILE_M, TILE_N), nl.float32, buffer=nl.psum) -- placed "
            "once per (m, n) tile, BEFORE the k loop.\n"
            "2) Every nisa.nc_matmul in the k loop writes into that same res_psum so the "
            "partial sums accumulate.\n"
            "3) Copy res_psum out once, after the k loop. Change nothing else."
        ),
    },
    {
        "id": "write_output_tile_once",
        "name": "Write each output tile exactly once, after accumulation",
        "when": (
            "The output is written inside the k loop, written more than once per tile, or a "
            "store fails with a shape or dtype error. Each output tile must be produced once, "
            "after the k loop finishes."
        ),
        "exact": (
            "Move the store out of the k loop and write each output tile exactly once:\n"
            "1) After the k loop: res_sb = nl.ndarray(res_psum.shape, dtype=result.dtype, "
            "buffer=nl.sbuf); nisa.tensor_copy(dst=res_sb, src=res_psum).\n"
            "2) One dma_copy to the output tile: "
            "result[m * TILE_M:(m + 1) * TILE_M, n * TILE_N:(n + 1) * TILE_N].\n"
            "Change nothing else."
        ),
    },
    {
        "id": "fix_slice_bounds",
        "name": "Make loop bounds and slice ends exact",
        "when": (
            "A failure mentions an index out of bounds (OOB): a loop bound or slice end "
            "exceeds the operand extents, or a tile count is not exact (e.g. a runtime "
            "expression in the bound arithmetic instead of K // TILE_K)."
        ),
        "exact": (
            "Make every loop bound and slice end exact and within the operand extents:\n"
            "1) Use integer tile counts derived from the operands: k_tiles = K // TILE_K, "
            "M // TILE_M, N // TILE_N.\n"
            "2) Slice ends are computed as offset + tile size and never exceed the operand's "
            "dimension.\n"
            "3) Do not use dynamic or runtime expressions in slice bounds. Change only the "
            "bounds."
        ),
    },
    {
        "id": "hoist_invariant_load",
        "name": "Hoist a loop-invariant load out of the loop",
        "when": (
            "A dma_copy sits inside a loop but its source indices do not depend on the loop "
            "index, and the rest of the kernel is fine. Hoisting the copy out removes repeated "
            "HBM traffic without changing the math."
        ),
        "exact": (
            "Hoist the loop-invariant dma_copy out of the loop:\n"
            "1) If a copy's source indices use only outer-loop variables or constants, move it "
            "outside the innermost loop that does not affect it.\n"
            "2) Keep the same destination tile shape and orientation.\n"
            "Change nothing else."
        ),
    },
    {
        "id": "continue_from_best",
        "name": "No match -- back out and try a different option",
        "when": (
            "No other option matches the evidence, or the current kernel is worse than the "
            "best seen so far (invalid with an unrecognized failure, or a regression against a "
            "previous kernel). Use this to back out; the loop reloads the best-scoring kernel "
            "and the next round picks a different guidance."
        ),
        "exact": (
            "Do not edit the kernel. Reply with the SAME kernel unchanged. The loop will "
            "reload the best-scoring kernel so far and pick a different guidance next round."
        ),
    },
]
