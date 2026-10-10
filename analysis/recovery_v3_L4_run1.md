# One failure and the recovery: feedback v3, level 4 (tiled matmul), run 1

Source: `runs/seat-117/Ev3_L4/attempts_v3_L4.jsonl` (seat-117, pulled 14:06), **lines 29-52**, and its console log `runs/seat-117/Ev3_L4/run_v3_L4.log` lines 4-19. (Lines 1-28 of the attempts file are an earlier invocation that is not in this console log.) Configuration: `feedback_v3.py`, `MESSAGES=v3 REPAIR_PROMPT=restructure`, Qwen3-8B, 4 samples per round. Rounds count from 0; round 5 is the sixth.

Each round below shows the attempt the agent carried forward (the round's best, first on ties), the simulator's own error, and the message the next round's prompt carried, **verbatim from the log**. Every next prompt was rebuilt from the logged code and feedback with the agent's own repair template; all five match the logged `prompt_chars` exactly (2532, 2953, 3210, 3161, 3860).

| round | best of 4 | sample scores | the wall | prompt chars | answer chars (4 samples) |
|---|---|---|---|---|---|
| 0 | 0.62 | 0.62 0.30 0.62 0.30 | dma partition 256 > 128 | 2,471 | 1,577, 1,032, 1,154, 1,052 |
| 1 | 0.62 | 0.30 0.30 0.62 0.30 | stationary free 256 > 128 | 2,532 | 1,450, 1,450, 1,675, 1,450 |
| 2 | 0.62 | 0.62 0.62 0.62 0.62 | moving free 1024 > 512 | 2,953 | 1,908, 2,000, 2,088, 2,088 |
| 3 | 0.62 | 0.62 0.62 0.62 0.62 | illegal SBUF allocation | 3,210 | 2,266, 2,249, 2,249, 2,249 |
| 4 | 0.62 | 0.62 0.62 0.62 0.62 | same, again (ledger added) | 3,161 | 2,247, 2,247, 2,247, 2,247 |
| 5 | 1.00 | 1.00 1.00 1.00 1.00 | **solved** | 3,860 | 2,260, 2,260, 2,260, 2,260 |

Tokens: this run predates exact token logging on that seat, so only character counts exist. At ~4 characters a token, a round cost about 600-1,000 prompt tokens per sample and 250-570 answer tokens per sample. These are estimates; the attempts file has no server counts.

## Round 0: 0.62

`runs/seat-117/Ev3_L4/attempts_v3_L4.jsonl` line 29; console log line 7.

The whole operand is copied into one SBUF tile, and the K loop slices it. Passes only the single-tile shape (K=128 M=128 N=512).

Key lines:

```python
    out = nl.ndarray(shape=(lhsT.shape[1], rhs.shape[1]), dtype=lhsT.dtype, buffer=nl.shared_hbm)
    sbuf_lhsT = nl.ndarray(shape=lhsT.shape, dtype=lhsT.dtype, buffer=nl.sbuf)
    sbuf_rhs = nl.ndarray(shape=rhs.shape, dtype=rhs.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=sbuf_lhsT, src=lhsT)
    nisa.dma_copy(dst=sbuf_rhs, src=rhs)
    psum = nl.ndarray(shape=(lhsT.shape[1], rhs.shape[1]), dtype=nl.float32, buffer=nl.psum)
    for tile_idx in nl.affine_range(num_tiles):
        nisa.nc_matmul(dst=psum, stationary=lhsT_tile, moving=rhs_tile)
    sbuf_out = nl.ndarray(shape=out.shape, dtype=out.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=out, src=sbuf_out)
```

**Simulator / checker says:** `dma_copy dst partition dimension 256 exceeds maximum 128`

**Sent in round 1's prompt** (under "A checker reports:", after the code):

```text
1 of 4 shapes passed. On K=256 M=256 N=1024: Line 15: `nisa.dma_copy(dst=sbuf_lhsT, src=lhsT)` raised AssertionError: dma_copy dst partition dimension 256 exceeds maximum 128 A tile may have at most 128 rows, and you asked for 256. Do not allocate one tile for the whole tensor: loop over the partition dimension in chunks of at most 128 with nl.affine_range, allocate the tile inside the loop with the chunk's own size, and copy one chunk at a time, e.g. src=a[i*128:(i+1)*128, :]. If a dimension is already 128 or smaller, use it whole -- do NOT pad it up to 128, that reads past the end of the tensor. The same applies to where you write the result back.
```

followed by the repair instruction: "Make the change the checker describes. If it gives code, use that code. If the change needs new loops or new tiles, restructure around them; otherwise keep the rest of the kernel as it is. Reply with ONE python code block."

## Round 1: 0.62

`runs/seat-117/Ev3_L4/attempts_v3_L4.jsonl` line 35; console log line 9.

It moved the copies into the K loop, but still allocates the whole operand and multiplies the full width: the K limit is fixed and the stationary limit is next.

Changed from the round before:

```diff
-    # Copy lhsT and rhs from HBM to SBUF
-    nisa.dma_copy(dst=sbuf_lhsT, src=lhsT)
-    nisa.dma_copy(dst=sbuf_rhs, src=rhs)
+        # Copy lhsT and rhs from HBM to SBUF in chunks
+        nisa.dma_copy(dst=sbuf_lhsT[start_k:end_k, :], src=lhsT[start_k:end_k, :])
+        nisa.dma_copy(dst=sbuf_rhs[start_k:end_k, :], src=rhs[start_k:end_k, :])
```

**Simulator / checker says:** `Matmul stationary free dimension 256 exceeds gemm_stationary_fmax=128`

**Sent in round 2's prompt** (under "A checker reports:", after the code):

```text
1 of 4 shapes passed. On K=256 M=256 N=1024: Line 40: `nisa.nc_matmul(dst=psum, stationary=lhsT_tile, moving=rhs_tile)` raised AssertionError: Matmul stationary free dimension 256 exceeds gemm_stationary_fmax=128 `lhsT_tile` is 256 columns wide, but nc_matmul takes at most 128 there. Compute the output 128 rows at a time: put your K loop inside a loop over m, give each m its own `psum`, take a 128-column slice of `lhsT_tile`, and write that block of rows out. The code has this shape:

    for m in nl.affine_range(lhsT.shape[1] // 128):
        psum = nl.ndarray((128, rhs.shape[1]), dtype=nl.float32, buffer=nl.psum)
        for k in ...:   # your K loop, with only the nc_matmul changed to:
            nisa.nc_matmul(dst=psum, stationary=lhsT_tile[:, m * 128:(m + 1) * 128], moving=rhs_tile)
        res = nl.ndarray(psum.shape, dtype=out.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=res, src=psum)
        nisa.dma_copy(dst=out[m * 128:(m + 1) * 128, :], src=res)

```

followed by the repair instruction: "Make the change the checker describes. If it gives code, use that code. If the change needs new loops or new tiles, restructure around them; otherwise keep the rest of the kernel as it is. Reply with ONE python code block."

## Round 2: 0.62

`runs/seat-117/Ev3_L4/attempts_v3_L4.jsonl` line 37; console log line 11.

It took v3's code nearly verbatim: an m loop with its own PSUM tile, a 128-column stationary slice and a 128-row output block. The moving operand is now the limit.

Changed from the round before:

```diff
-    # Allocate PSUM for result
-    psum = nl.ndarray(shape=(lhsT.shape[1], rhs.shape[1]), dtype=nl.float32, buffer=nl.psum)
-    # Loop over tiles
-    for tile_idx in nl.affine_range(num_tiles):
-        # Compute tile start and end indices
-        start_k = tile_idx * 128
-        end_k = min(start_k + 128, K)
+    # Loop over rows of output in chunks of 128
+    for m in nl.affine_range(M // 128):
+        # Allocate PSUM for this chunk of output
+        psum = nl.ndarray(shape=(128, N), dtype=nl.float32, buffer=nl.psum)
-        # Copy lhsT and rhs from HBM to SBUF in chunks
-        nisa.dma_copy(dst=sbuf_lhsT[start_k:end_k, :], src=lhsT[start_k:end_k, :])
-        nisa.dma_copy(dst=sbuf_rhs[start_k:end_k, :], src=rhs[start_k:end_k, :])
+        # Compute tile start and end indices along K
+        for tile_idx in nl.affine_range(num_tiles):
+            start_k = tile_idx * 128
+            end_k = min(start_k + 128, K)
+            # Copy lhsT and rhs from HBM to SBUF in chunks
+            nisa.dma_copy(dst=sbuf_lhsT[start_k:end_k, :], src=lhsT[start_k:end_k, :])
+            nisa.dma_copy(dst=sbuf_rhs[start_k:end_k, :], src=rhs[start_k:end_k, :])
+            # Slice lhsT and rhs
+            lhsT_tile = sbuf_lhsT[start_k:end_k, :]
+            rhs_tile = sbuf_rhs[start_k:end_k, :]
+            # Perform NC matmul for this row chunk
+            nisa.nc_matmul(dst=psum, stationary=lhsT_tile[:, m * 128:(m + 1) * 128], moving=rhs_tile)
-        # Slice lhsT and rhs
-        lhsT_tile = sbuf_lhsT[start_k:end_k, :]
-        rhs_tile = sbuf_rhs[start_k:end_k, :]
+        # Copy result from PSUM to SBUF
+        sbuf_out = nl.ndarray(shape=(128, N), dtype=out.dtype, buffer=nl.sbuf)
+        nisa.tensor_copy(dst=sbuf_out, src=psum)
-        # Perform NC matmul
-        nisa.nc_matmul(dst=psum, stationary=lhsT_tile, moving=rhs_tile)
-    # Copy result from PSUM to SBUF
-    sbuf_out = nl.ndarray(shape=out.shape, dtype=out.dtype, buffer=nl.sbuf)
-    nisa.tensor_copy(dst=sbuf_out, src=psum)
-    # Copy result from SBUF to HBM
-    nisa.dma_copy(dst=out, src=sbuf_out)
+        # Copy result from SBUF to HBM
+        nisa.dma_copy(dst=out[m * 128:(m + 1) * 128, :], src=sbuf_out)
```

**Simulator / checker says:** `Matmul moving free dimension 1024 exceeds max 512 for nc_version=nc_version.gen3`

**Sent in round 3's prompt** (under "A checker reports:", after the code):

```text
1 of 4 shapes passed. On K=256 M=256 N=1024: Line 41: `nisa.nc_matmul(dst=psum, stationary=lhsT_tile[:, m * 128:(m + 1) * 128], moving=rhs_tile)` raised AssertionError: Matmul moving free dimension 1024 exceeds max 512 for nc_version=nc_version.gen3 `rhs_tile` is 1024 columns wide, but nc_matmul takes at most 512 there. Split the output columns the same way as the rows: inside the loop that owns `psum`, loop n over rhs.shape[1] // 512, allocate `psum` with 512 columns, pass a 512-column slice of `rhs_tile`, and write to columns [n * 512:(n + 1) * 512] of the output:

    for n in nl.affine_range(rhs.shape[1] // 512):
        psum = nl.ndarray((<rows of this block>, 512), dtype=nl.float32, buffer=nl.psum)
        for k in ...:   # your K loop, with only the nc_matmul changed to:
            nisa.nc_matmul(dst=psum, stationary=lhsT_tile[:, m * 128:(m + 1) * 128], moving=rhs_tile[:, n * 512:(n + 1) * 512])
        ...   # then copy psum out to out[<rows of this block>, n * 512:(n + 1) * 512]

```

followed by the repair instruction: "Make the change the checker describes. If it gives code, use that code. If the change needs new loops or new tiles, restructure around them; otherwise keep the rest of the kernel as it is. Reply with ONE python code block."

## Round 3: 0.62

`runs/seat-117/Ev3_L4/attempts_v3_L4.jsonl` line 41; console log line 13.

It took the second code message too: an n loop, a (128, 512) PSUM tile, a 512-column moving slice. The simulator now runs every shape, but the allocation audit stops it: the SBUF staging tiles are still whole operands, (256, 256) and (256, 1024). A chip cannot hold them.

Changed from the round before:

```diff
+    # Compute number of tiles along N (columns of rhs)
+    num_n_tiles = (N + 511) // 512
-        # Allocate PSUM for this chunk of output
-        psum = nl.ndarray(shape=(128, N), dtype=nl.float32, buffer=nl.psum)
-        # Compute tile start and end indices along K
-        for tile_idx in nl.affine_range(num_tiles):
-            start_k = tile_idx * 128
-            end_k = min(start_k + 128, K)
+        # Loop over columns of output in chunks of 512
+        for n in nl.affine_range(num_n_tiles):
+            # Allocate PSUM for this chunk of output
+            psum = nl.ndarray(shape=(128, 512), dtype=nl.float32, buffer=nl.psum)
-            # Copy lhsT and rhs from HBM to SBUF in chunks
-            nisa.dma_copy(dst=sbuf_lhsT[start_k:end_k, :], src=lhsT[start_k:end_k, :])
-            nisa.dma_copy(dst=sbuf_rhs[start_k:end_k, :], src=rhs[start_k:end_k, :])
+            # Compute tile start and end indices along K
+            for tile_idx in nl.affine_range(num_tiles):
+                start_k = tile_idx * 128
+                end_k = min(start_k + 128, K)
+                # Copy lhsT and rhs from HBM to SBUF in chunks
+                nisa.dma_copy(dst=sbuf_lhsT[start_k:end_k, :], src=lhsT[start_k:end_k, :])
+                nisa.dma_copy(dst=sbuf_rhs[start_k:end_k, :], src=rhs[start_k:end_k, :])
+                # Slice lhsT and rhs
+                lhsT_tile = sbuf_lhsT[start_k:end_k, :]
+                rhs_tile = sbuf_rhs[start_k:end_k, :]
+                # Perform NC matmul for this row and column chunk
+                nisa.nc_matmul(dst=psum, stationary=lhsT_tile[:, m * 128:(m + 1) * 128], moving=rhs_tile[:, n * 512:(n + 1) * 512])
-            # Slice lhsT and rhs
-            lhsT_tile = sbuf_lhsT[start_k:end_k, :]
-            rhs_tile = sbuf_rhs[start_k:end_k, :]
+            # Copy result from PSUM to SBUF
+            sbuf_out = nl.ndarray(shape=(128, 512), dtype=out.dtype, buffer=nl.sbuf)
+            nisa.tensor_copy(dst=sbuf_out, src=psum)
-            # Perform NC matmul for this row chunk
-            nisa.nc_matmul(dst=psum, stationary=lhsT_tile[:, m * 128:(m + 1) * 128], moving=rhs_tile)
-        # Copy result from PSUM to SBUF
-        sbuf_out = nl.ndarray(shape=(128, N), dtype=out.dtype, buffer=nl.sbuf)
-        nisa.tensor_copy(dst=sbuf_out, src=psum)
-        # Copy result from SBUF to HBM
-        nisa.dma_copy(dst=out[m * 128:(m + 1) * 128, :], src=sbuf_out)
+            # Copy result from SBUF to HBM
+            nisa.dma_copy(dst=out[m * 128:(m + 1) * 128, n * 512:(n + 1) * 512], src=sbuf_out)
```

**Simulator / checker says:** `ILLEGAL ON HARDWARE: the CPU simulator ran this kernel, but it does not check tile limits when a tile is allocated and the chip does.`

**Sent in round 4's prompt** (under "A checker reports:", after the code):

```text
1 of 4 shapes passed. On K=256 M=256 N=1024: ILLEGAL ON HARDWARE: the CPU simulator ran this kernel, but it does not check tile limits when a tile is allocated and the chip does. The kernel allocated a sbuf tile of shape (256, 256), whose partition dimension (the first) is 256 where the maximum is 128; and a sbuf tile of shape (256, 1024), whose partition dimension (the first) is 256 where the maximum is 128. Do not allocate one on-chip tile for a whole operand or the whole result: loop over the rows in chunks of at most 128 and allocate each tile inside the loop with the chunk's own shape.
```

followed by the repair instruction: "Make the change the checker describes. If it gives code, use that code. If the change needs new loops or new tiles, restructure around them; otherwise keep the rest of the kernel as it is. Reply with ONE python code block."

## Round 4: 0.62

`runs/seat-117/Ev3_L4/attempts_v3_L4.jsonl` line 45; console log line 15.

The same kernel with one comment changed. Same failure twice, so the agent adds the ledger of four failed approaches to the next prompt.

Changed from the round before:

```diff
-            # Compute tile start and end indices along K
+            # Loop over tiles along K
```

**Simulator / checker says:** `ILLEGAL ON HARDWARE: the CPU simulator ran this kernel, but it does not check tile limits when a tile is allocated and the chip does.`

**Sent in round 5's prompt** (under "A checker reports:", after the code):

```text
1 of 4 shapes passed. On K=256 M=256 N=1024: ILLEGAL ON HARDWARE: the CPU simulator ran this kernel, but it does not check tile limits when a tile is allocated and the chip does. The kernel allocated a sbuf tile of shape (256, 256), whose partition dimension (the first) is 256 where the maximum is 128; and a sbuf tile of shape (256, 1024), whose partition dimension (the first) is 256 where the maximum is 128. Do not allocate one on-chip tile for a whole operand or the whole result: loop over the rows in chunks of at most 128 and allocate each tile inside the loop with the chunk's own shape.
```

The prompt then ends with the repair instruction and, because the same failure came twice, the ledger:

```text
Make the change the checker describes. If it gives code, use that code. If the change needs new loops or new tiles, restructure around them; otherwise keep the rest of the kernel as it is. Reply with ONE python code block.

These approaches have already failed, so do something different:
- 1 of 4 shapes passed. On K=256 M=256 N=1024: Line 15: `nisa.dma_copy(dst=sbuf_lhsT, src=lhsT)` raised AssertionError: dma_copy dst partition dimension 256 excee
- 1 of 4 shapes passed. On K=256 M=256 N=1024: Line 40: `nisa.nc_matmul(dst=psum, stationary=lhsT_tile, moving=rhs_tile)` raised AssertionError: Matmul stationary
- 1 of 4 shapes passed. On K=256 M=256 N=1024: Line 41: `nisa.nc_matmul(dst=psum, stationary=lhsT_tile[:, m * 128:(m + 1) * 128], moving=rhs_tile)` raised Asserti
- 1 of 4 shapes passed. On K=256 M=256 N=1024: ILLEGAL ON HARDWARE: the CPU simulator ran this kernel, but it does not check tile limits when a tile is allocated 
```

## Round 5: 1.00

`runs/seat-117/Ev3_L4/attempts_v3_L4.jsonl` line 49; console log line 17.

With the ledger, it rewrote the two allocations as (128, 128) and (128, 512) tiles and loaded each block straight from HBM. Correct on all four shapes.

Changed from the round before:

```diff
-    sbuf_lhsT = nl.ndarray(shape=lhsT.shape, dtype=lhsT.dtype, buffer=nl.sbuf)
-    sbuf_rhs = nl.ndarray(shape=rhs.shape, dtype=rhs.dtype, buffer=nl.sbuf)
+    sbuf_lhsT = nl.ndarray(shape=(128, 128), dtype=lhsT.dtype, buffer=nl.sbuf)
+    sbuf_rhs = nl.ndarray(shape=(128, 512), dtype=rhs.dtype, buffer=nl.sbuf)
+    # Compute number of tiles along M (rows of lhsT)
+    num_m_tiles = (M + 127) // 128
-    for m in nl.affine_range(M // 128):
+    for m in nl.affine_range(num_m_tiles):
-                nisa.dma_copy(dst=sbuf_lhsT[start_k:end_k, :], src=lhsT[start_k:end_k, :])
-                nisa.dma_copy(dst=sbuf_rhs[start_k:end_k, :], src=rhs[start_k:end_k, :])
+                nisa.dma_copy(dst=sbuf_lhsT, src=lhsT[start_k:end_k, m * 128:(m + 1) * 128])
+                nisa.dma_copy(dst=sbuf_rhs, src=rhs[start_k:end_k, n * 512:(n + 1) * 512])
-                lhsT_tile = sbuf_lhsT[start_k:end_k, :]
-                rhs_tile = sbuf_rhs[start_k:end_k, :]
+                lhsT_tile = sbuf_lhsT
+                rhs_tile = sbuf_rhs
-                nisa.nc_matmul(dst=psum, stationary=lhsT_tile[:, m * 128:(m + 1) * 128], moving=rhs_tile[:, n * 512:(n + 1) * 512])
+                nisa.nc_matmul(dst=psum, stationary=lhsT_tile, moving=rhs_tile)
```

**Checker:** `Correct on every shape. MEMORY BOUND: 36.6 Flops/Byte against a ridge of 222, so 16% of what the engine could sustain. The engine is idle waiting for data. Find reuse -- make the same bytes do more work -- rather than tuning the arithmetic. 6.1x more reuse needed.`

## The solving kernel, checked again

`runs/seat-117/Ev3_L4/attempts_v3_L4.jsonl` line 49 (all four samples of round 5 returned this kernel). Checked on this branch with `NEURON_PLATFORM_TARGET_OVERRIDE=trn2`, NKI 0.6.0:

| check | result |
|---|---|
| re-grade of all 24 attempts of the run with feedback_v3's `grade()` (fresh path each, current checker, allocation audit on) | rewards 24/24 and feedback text 24/24 identical to the log |
| `scripts/reaudit.py` (fresh process) | **PASS** (kernel c733c15749, seen 4x, first at line 49) |
| held-out set, `nkibench.py --level 4 --eval` (4 new shapes x 4 value kinds, never shown to the agent) | **16/16** |
| `agent.confidence()` before the held-out check | 0.90 ("passed every loop shape and nothing in the code is shape-specific") |
| verdict | **VERIFIED**: the confidence was right |

So the agent's claim holds. It reached it after 21 attempts (line 29 to line 49), and the checker message that broke the last wall was the allocation audit (26c43ed), carried with the ledger. The kernel is still memory bound (36.6 Flops/Byte, 6.1x short of the ridge): correct, not fast.

Reproduce:

```bash
sed -n 49p runs/seat-117/Ev3_L4/attempts_v3_L4.jsonl | python -c 'import json,sys; print(json.loads(sys.stdin.read())["code"])' > /tmp/solve.py
export NEURON_PLATFORM_TARGET_OVERRIDE=trn2
python projects/02-kernel-agent/nkibench.py --level 4 --check /tmp/solve.py
python projects/02-kernel-agent/nkibench.py --level 4 --eval /tmp/solve.py
python scripts/reaudit.py runs/seat-117/Ev3_L4/attempts_v3_L4.jsonl
```
