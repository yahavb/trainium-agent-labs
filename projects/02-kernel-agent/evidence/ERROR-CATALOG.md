# Error catalog — every failure the loop can hit, and its fix

Sources: the checker/simulator's own message families, plus every failure logged across
the day's runs on both seats and the local evidence logs.

**Curated families: 23. Observed distinct signatures: 11.**

## Observed failures (ranked by count)

| count | signature | example |
|---:|---|---|
| 141 | NUMERICAL MISMATCH: worst error N of the output's RMS (N), tolerance N | NUMERICAL MISMATCH: worst error 4.73 of the output's RMS (15.98), tolerance 0.02.   at index (222, 687): expected +35.26 |
| 131 | raised AssertionError: dma_copy requires src and dst to have the same number of elements, got src=N, dst=N | raised AssertionError: dma_copy requires src and dst to have the same number of elements, got src=32768, dst=16384 |
| 131 | raised AssertionError: Out-of-bound access for tensor `unnamed` on dimension N: index range [N, N] exceed dime | raised AssertionError: Out-of-bound access for tensor `unnamed` on dimension 0: index range [0, 511] exceed dimension si |
| 81 | raised AssertionError: dma_copy dst partition dimension N exceeds maximum N | raised AssertionError: dma_copy dst partition dimension 256 exceeds maximum 128 |
| 33 | raised UnboundLocalError: cannot access local variable 'k' where it is not associated with a value | raised UnboundLocalError: cannot access local variable 'k' where it is not associated with a value |
| 29 | CORRECT, BUT TOO MUCH HBM TRAFFIC FOR THIS LEVEL: moving Nx the byte floor, and level N requires Nx or better | CORRECT, BUT TOO MUCH HBM TRAFFIC FOR THIS LEVEL: moving 1.86x the byte floor, and level 5 requires 1.60x or better. Cor |
| 16 | raised AssertionError: Matmul stationary free dimension N exceeds gemm_stationary_fmax=N | raised AssertionError: Matmul stationary free dimension 256 exceeds gemm_stationary_fmax=128 |
| 6 | raised ValueError: cannot reshape array of size N into shape (N,N) | raised ValueError: cannot reshape array of size 16384 into shape (128,512) |
| 4 | NON-FINITE OUTPUT: N NaN and N Inf, first at (N, N) | NON-FINITE OUTPUT: 131072 NaN and 0 Inf, first at (0, 0). Usually an uninitialised PSUM or SBUF tile being read before a |
| 3 | raised AssertionError: Matmul contraction dimension N exceeds pmax=N | raised AssertionError: Matmul contraction dimension 256 exceeds pmax=128 |
| 2 | raised AssertionError: dma_copy requires HBM or SBUF tensors, got src=MemoryRegion | raised AssertionError: dma_copy requires HBM or SBUF tensors, got src=MemoryRegion.psum, dst=MemoryRegion.shared_hbm |

## Curated families (message -> cause -> fix)

| signature | cause | fix |
|---|---|---|
| dma_copy requires src and dst to have the same number of elements, got src=N, dst=N | the destination tile does not hold exactly the elements of the source slice (a cache sized by a tile where the full-width chunk is required) | size the cache by k_tiles * W (the operand's full width) and copy full-width chunks; operand views then slide inside that cache |
| dma_copy <dst\|src> partition dimension N exceeds maximum M | one tile was allocated for more rows than the partition axis allows (max 128) | loop over the partition dimension in <=128-row chunks, allocate the tile per chunk; never pad a smaller dimension up to 128 |
| Out-of-bound access ... index range [a, b] exceeds dimension size N | a bound or slice end was computed past the tensor's extent | derive bounds from the tensor's own shape; the final chunk may be partial; partition <= 128 |
| Matmul contraction dimension N exceeds pmax=M | one nc_matmul was given a K chunk larger than the hardware contraction limit | split K into pmax-sized chunks and accumulate all chunks into ONE psum tile |
| Matmul stationary free dimension N exceeds gemm_stationary_fmax=M | the stationary operand slice is wider than the stationary free limit (128); typical overshoot: slicing the full operand width instead of a TILE_M window | slice the stationary operand in TILE_M-wide windows (<=128); keep K on the partition axis |
| <op> must be in ['sbuf', 'psum'], got shared_hbm | an on-chip operation was pointed at HBM | dma_copy is the only HBM bridge; go psum -> sbuf with tensor_copy, then dma_copy sbuf -> shared_hbm |
| <op> must be in ['psum'], got <region> / must be in ['sbuf'], got <region> | a tile sits in the wrong memory region for its operation | nc_matmul: dst in nl.psum, stationary and moving in nl.sbuf; move tiles with tensor_copy |
| value array of shape (N, N) could not be broadcast to indexing result of shape (N, N) | an assignment wrote a value whose shape differs from the destination slice | match both sides exactly; nothing broadcasts or reshapes |
| module 'nki.language' has no attribute 'X' | the model invented an API name (nl.value, nl.scalar, nl.dot, tile.mean) | use only real names: nl.ndarray, nl.affine_range, nl.sum(view, axis=[...]), nisa.dma_copy, nisa.nc_matmul, nisa.tensor_copy, nisa.tensor_scalar, tile.ap |
| 'X' object has no attribute 'Y' | a numpy-ism was applied to a tile | use nl/nisa operations instead of numpy attributes |
| got multiple values for argument | one argument was passed twice (positionally and by keyword) | pass every argument exactly once, by keyword |
| must have at least 2 dimensions | a 1-D tile was allocated | every tile is (rows, cols); a length-N vector is (1, N) or (N, 1) |
| cannot reshape array of size | the kernel attempted a reshape | do not reshape; slice the given shapes into tiles |
| 'MemoryRegion' object is not callable | nl.sbuf / nl.psum / nl.shared_hbm were called like functions | pass the region as buffer= to nl.ndarray |
| unsupported operand type(s) for <op>: 'NkiTensor' and ... | a python operator was used on a tile | accumulate in a psum tile via nc_matmul, or use nisa ops |
| UnboundLocalError: cannot access local variable 'k' where it is not associated with a value | a loop variable (or any local) is used outside the loop or on a path where it was never assigned -- usually a restructure moved the code but not the binding | initialize the variable before the loop, or restructure so every path assigns it before use; keep loop bodies self-contained |
| NUMERICAL MISMATCH: worst error X of the output's RMS (Y), tolerance Z | the kernel runs but computes wrong numbers | run the behavioral diagnosis: stationary operand stuck / only one k-chunk / dtype or layout fault; apply the matching menu guidance |
| CORRECT, BUT TOO MUCH HBM TRAFFIC FOR THIS LEVEL | the numbers are right but the kernel moves more bytes than the level gate allows (1.60x / 1.25x / 1.05x) | find the operand still re-read per outer-loop pass and give it the cached treatment (stack_stationary_operand / stack_moving_operand_symmetrically) |
| TRAFFIC UNMEASURED / bytes below the byte floor | the measurement is incomplete, or the kernel moves fewer bytes than any correct kernel can (it skips required reads -- the computation is broken) | fail closed: restore the full computation, then re-measure; never accept a below-floor count |
| THE KERNEL MODIFIED ITS INPUT (argument N) | the kernel wrote into one of its input tensors | allocate a new output with buffer=nl.shared_hbm and return that |
| CORRECT ON CPU BUT WRONG ON HARDWARE: <simulator warning> | the simulator flagged a pattern that is numerically correct on CPU and incorrect on the device | restructure the flagged pattern (most often the accumulation/psum scheme) |
| RULE VIOLATIONS (banned framework call / entry point missing / @nki.jit missing) | a level rule was broken; this scores zero however fast the kernel is | remove the framework call (matmul/dot/einsum/@/.T), define the level's entry point, keep it decorated with @nki.jit |
| the file imports but <entry> could not be loaded / defines no `<entry>` | the kernel does not define the entry name this level checks through | keep the level's entry-point name exactly as the checker expects it |
