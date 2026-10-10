# Levels 5–7: reuse and transfer reduction in matmul

This file documents the level-5, level-6, and level-7 references, prompts, and checker behavior
in this repository. It separates code behavior, recorded measurements, and results that still
need a fresh run.

## 1. Operation and registered shapes

All three levels compute the same reference operation as level 4:

```text
lhsT:  (K, M)     # left input already transposed into NKI's matmul layout
rhs:   (K, N)
out:   (M, N)     # mathematically lhsT.T @ rhs
```

The four registered cases are tile-aligned:

| K | M | N | K tiles | M tiles | N tiles |
|---:|---:|---:|---:|---:|---:|
| 128 | 128 | 512 | 1 | 1 | 1 |
| 256 | 256 | 1024 | 2 | 2 | 2 |
| 512 | 128 | 512 | 4 | 1 | 1 |
| 256 | 512 | 1024 | 2 | 4 | 2 |

Tile sizes come from NKI: `tile_k = nl.tile_size.pmax` (128),
`tile_m = nl.tile_size.gemm_stationary_fmax` (128), and
`tile_n = nl.tile_size.gemm_moving_fmax` (512). These references require positive K, M, and N
that divide evenly by their tile sizes. They do not handle ragged edges.

## 2. What each level changes

| Level | Entry point / file | Main change |
|---|---|---|
| 5 | `nki_matmul_hoist_load_` / `reference_level5.py` | Cache one RHS N slab across all M output tiles in that slab. RHS tiles are reused; LHS tiles are still loaded once per M tile and N slab. |
| 6 | `nki_matmul_block_free_dimension_` / `reference_level6.py` | Cache a block of N slabs and reuse each LHS tile across all slabs in that block. |
| 7 | `nki_matmul_fully_optimized_` / `reference_level7.py` | Also load a block of contiguous M columns from `lhsT` in one transfer per K tile, reducing LHS transfer calls without reducing the bytes. |

All levels keep the result in shared HBM, inputs and intermediate operand caches in SBUF, and the
accumulator in PSUM. For each output tile, the kernel accumulates K tiles into one float32 PSUM,
copies that PSUM to a same-shaped SBUF tile with `nisa.tensor_copy`, then stores it with
`nisa.dma_copy`. It allocates the output once and must not overwrite either input.

### Level 5 loop and tile layout

The loop order is `N slab → preload K tiles → M tile → accumulate K tiles`:

| Buffer | Shape | Role |
|---|---|---|
| `rhs_cache` | `(tile_k, k_tiles * tile_n)` | Holds one N slab's RHS K tiles, packed along the free axis. |
| `left` | `(tile_k, tile_m)` | One LHS tile loaded for the current M tile and K step. |
| `accum` | `(tile_m, tile_n)` float32 PSUM | Accumulates all K steps for one output tile. |
| `output_tile` | `(tile_m, tile_n)` SBUF | Receives the completed PSUM before its HBM store. |

For N tile index `n`, K tile index `k`, and M tile index `m`, the RHS cache stores the source
slice `rhs[k*tile_k:(k+1)*tile_k, n*tile_n:(n+1)*tile_n]` in
`rhs_cache[:, k*tile_n:(k+1)*tile_n]`. Each RHS tile is loaded once for the N slab and reused
for every M tile. The cache has `K * tile_n` elements; at the largest registered K=512 and
float32, that is 1 MiB.

### Level 6 loop and tile layout

The loop order is `N block → fill RHS block → M tile → load its LHS K tiles → N slab → K`:

| Buffer | Shape | Role |
|---|---|---|
| `rhs_block` | `(tile_k, block_n * k_tiles * tile_n)` | Holds all RHS K tiles for `block_n` N slabs. |
| `left_block` | `(tile_k, k_tiles * tile_m)` | Holds the current M tile's LHS K tiles. |
| `accum` | `(tile_m, tile_n)` float32 PSUM | Accumulates one M/N output tile over K. |

For N slab `j` inside N block `nb`, K tile `k` is stored at RHS cache columns
`(j * k_tiles + k) * tile_n : (j * k_tiles + k + 1) * tile_n`. The left block is loaded once
per M tile and N block, then reused for every `j` in that block.

The code sets `BUDGET_FLOATS_PER_PARTITION = 16384`. For each candidate N block it calculates
`affordable = max(1, budget // (k_tiles * tile_n))`, then chooses the largest divisor of
`n_tiles` no larger than `affordable`. Using a divisor keeps all loop bounds exact for these
aligned shapes. This budget is a choice in the reference code, not a published hardware SBUF
capacity.

### Level 7 loop and tile layout

The loop order is `N block → fill RHS block → M block → load its LHS K tiles → M tile → N slab → K`:

| Buffer | Shape | Role |
|---|---|---|
| `rhs_block` | `(tile_k, block_n * k_tiles * tile_n)` | Holds RHS K tiles for the current N block. |
| `left_block` | `(tile_k, k_tiles * span)`, `span = block_m * tile_m` | Holds all LHS K tiles for the current M block. |
| `accum` | `(tile_m, tile_n)` float32 PSUM | Accumulates one M/N output tile over K. |

Because `lhsT` has shape `(K, M)`, its M axis is contiguous. For each K tile, one DMA loads
`lhsT[k0:k1, m0:m0+span]` into `left_block[:, k*span:(k+1)*span]`. M tile `i` then uses the
subview `left_block[:, k*span + i*tile_m : k*span + (i+1)*tile_m]`. The output is still stored
one `(tile_m, tile_n)` tile at a time; merging output stores would exceed the partition limit.

Level 7 selects `block_n` using the same budget approach as level 6. It subtracts the RHS block
size from the budget, then chooses a divisor for `block_m` using the remaining allowance. The
code keeps at least enough allowance for one M tile. This is a block-size heuristic for the
registered shapes, not a general capacity proof.

## 3. Bytes and DMA transfers

The byte floor is one read of each input and one write of the output. For the registered
same-dtype inputs, with element size `b` bytes:

```text
floor = b * (K*M + K*N + M*N)
```

The bytes moved by the three loop structures can be expressed using tile counts
`Kt = K/tile_k`, `Mt = M/tile_m`, `Nt = N/tile_n`, and `Nb = number of N blocks`:

| Level | Approximate byte formula | Why bytes are reread |
|---|---|---|
| 5 | `b * (K*M*Nt + K*N + M*N)` | LHS is read once for every N slab. |
| 6 | `b * (K*M*Nb + K*N + M*N)` | LHS is read once for every N block. |
| 7 | Same bytes as level 6 | M blocking combines LHS transfers; it does not reduce the number of LHS elements read. |

The source loop bounds also give these DMA call counts over all output tiles for one simulator
invocation, assuming each `nisa.dma_copy` call in the loop executes once per iteration:

| Level | DMA calls over the registered cases | Formula per case |
|---|---:|---|
| 5 | 56 | `Nt*Kt + Nt*Mt*Kt + Mt*Nt` |
| 6 | 44 | `Nt*Kt + Nb*Mt*Kt + Mt*Nt` |
| 7 | 36 | `Nt*Kt + Nb*Mb*Kt + Mt*Nt`, where `Mb = number of M blocks` |

These call totals are derived from the loop structure, not reported here as a fresh validator run.
The validators count calls at runtime and should be used to confirm them in the pod.

The agent checker instruments `nisa.dma_copy` and rejects an unmeasured kernel if it counts zero
bytes or zero transfers. The counter cannot see data movement through APIs it does not instrument.
It measures simulator-requested DMA bytes, not physical-device latency or actual HBM timing.

## 4. Configured traffic limits and their overlap

`nkibench.py` sets these maximum byte ratios:

| Level | `max_waste` | What the bar enforces |
|---|---:|---|
| 5 | 1.60× floor | Kernel must be correct and stay at or under 1.60 times the minimum bytes. |
| 6 | 1.25× floor | Same type of byte limit, but the level-5 reference is documented at a worst case of 1.143× and already passes it. |
| 7 | 1.05× floor | The level-6 reference is designed to reach 1.00×, so this byte limit does not distinguish levels 6 and 7. |

The thresholds alone do not make levels 5, 6, and 7 progressively harder. The actual level-7
goal is to reduce DMA call count while keeping the byte count unchanged. The grader gates on bytes,
not the transfer count, so fewer transfers are measured by validators but are not required for
reward 1.00.

## 5. Recorded level-5 simulator bytes

The README records the following level-4 and level-5 measurements from the Seat 21 NKI CPU
simulator with float32 inputs. These are simulated DMA byte counts, not hardware timing results.

| K | M | N | Byte floor | Level 4 bytes | Level 5 bytes | Level 5 / floor |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 128 | 512 | 589,824 | 589,824 | 589,824 | 1.000× |
| 256 | 256 | 1024 | 2,359,296 | 3,670,016 | 2,621,440 | 1.111× |
| 512 | 128 | 512 | 1,572,864 | 1,572,864 | 1,572,864 | 1.000× |
| 256 | 512 | 1024 | 3,670,016 | 7,340,032 | 4,194,304 | 1.143× |

The repository has level-6 and level-7 validators that compare the references and report byte
ratios and transfer counts. No fresh level-6/7 validator output is included here. Likewise, the
presence of references and agent prompts does not establish a current model solve rate. A
comment in `agent.py` refers to earlier 3/3 prompt runs, but the corresponding logs and full run
settings are not checked in, so treat that as historical source-comment context rather than a
reproducible result.

## 6. What the validators check

Each validator uses the four registered shapes and seeds `0`, `1`, and `2`. They require the NKI
simulator, call no model, and make no claim about hardware latency.

| Command | References compared | Checks |
|---|---|---|
| `python validate_level5.py` | Levels 4 and 5 | Static rules; numerical match; inputs unchanged; selected hardware-hazard warnings; level-5 byte gate; exact level-5 bytes and transfer formula. Also confirms level 4 fails the level-5 gate on at least one case. |
| `python validate_level6.py` | Levels 4, 5, and 6 | Static rules; numerical match; inputs unchanged; selected warnings; level-6 byte gate; level 6 does not exceed level 5 bytes. Reports that level 5 already passes the level-6 threshold. |
| `python validate_level7.py` | Levels 5, 6, and 7 | Static rules; numerical match; inputs unchanged; selected warnings; level-7 byte gate; level 7 does not exceed level 6 in bytes or transfer calls. Reports totals and the threshold overlap. |

Run them from the kernel-agent directory on a pod with NKI installed:

```bash
cd /workspace/projects/02-kernel-agent
python validate_level5.py
python validate_level6.py
python validate_level7.py
```

`python nkibench.py --level N --check reference_levelN.py` is useful for checking a reference,
but it does not enforce the level-5 to level-7 traffic gate. The dedicated validators and
`agent.py` grading path do.

## 7. Running and measuring the model

`agent.py --all` runs levels 1–4. Select levels 5, 6, and 7 explicitly. This example runs five
repeats for each level and writes a separate JSONL log:

```bash
cd /workspace/projects/02-kernel-agent
python -u agent.py --level 5 --rounds 8 --samples 4 --context 8192 --repeat 5 --log attempts-level5.jsonl
python -u agent.py --level 6 --rounds 8 --samples 4 --context 8192 --repeat 5 --log attempts-level6.jsonl
python -u agent.py --level 7 --rounds 8 --samples 4 --context 8192 --repeat 5 --log attempts-level7.jsonl
```

Run one agent command at a time. The grader writes a candidate to a shared path such as
`/tmp/_agent_level5.py`, so simultaneous runs of the same level can interfere even from separate
checkouts.

Summarize each log with:

```bash
python analyze.py attempts-level5.jsonl
python analyze.py attempts-level6.jsonl
python analyze.py attempts-level7.jsonl
```

The grader's reward weights are 0.10 for parsing, 0.20 for static rules, 0.20 for running, and
0.50 for correctness. A run reaches reward 1.00 only if it passes all checks on all four shapes,
including that level's byte gate. Report model solve rate separately from reference-validator
results.

## 8. Current files and stale documentation

| File | Contents |
|---|---|
| `reference_level5.py` – `reference_level7.py` | NKI kernels for RHS reuse, N blocking, and M/N blocking. |
| `validate_level5.py` – `validate_level7.py` | Simulator-based reference and traffic checks. |
| `agent.py` | Level-specific initial/repair cards and candidate grading. |
| `nkibench.py` | Registered shapes, `max_waste` limits, simulator wrapper, and DMA byte counter. |

The opening status section of `README.md` says levels 6 and 7 lack reference kernels. That status
predates the current files and is stale; the references and validators listed above are present.
