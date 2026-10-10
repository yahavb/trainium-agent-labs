# NKI optimization heuristics for Trainium2

What a kernel-writing agent (or a human) should know before it writes, and what a verifier can check
after it has written. Distilled on 2026-10-10 from five sources, each of which is cited per rule in
[APPENDIX.md](APPENDIX.md) and machine-readable in [rules.json](rules.json):

| source | what it contributed | records |
|---|---|---|
| NKI docs: architecture guide, performance guide, deep dives (DGE, DMA bandwidth, access patterns, dynamic loops), hardware pages | the physical limits and the pathologies | 165 |
| NKI API reference (docstrings of the installed `nki` 0.6.0, migration guide) | per-instruction constraints, the ones a model trained on older NKI breaks | 159 |
| NKI tutorials, how-tos and the NKI Library docs | before/after optimization steps with measured speedups; TKG vs CTE regime; LNC-2 sharding | 160 |
| AWS's own agent skills shipped in `neuron-agentic-development` (nki-writing, profiling, debugging, profile-querying) | the curated rules AWS gives its own agents, with profile-derived thresholds | 122 |
| production kernel source: `nkilib` (attention_tkg, mlp_tkg, rmsnorm, rope, allocator) and `aws-neuron/nki-samples` | patterns experts use that the docs never state | 120 |

Three deliverables sit next to this file:

* **`rules.json`** — 724 de-duplicated rules in one schema (`rule`, `why`, `trigger`, `fix`, `detectable`, `check_sketch`, `numbers`, `source`, `confidence`, `generation`, `tier`). 518 are statically detectable, 81 need a profile, 23 need the simulator.
* **`heuristics.py`** — the agent-facing API: `card()` (17 tier-0 rules, about 350 tokens, safe in every prompt), `retrieve(query)` (keyword retrieval over all rules), `for_error(text)` and `instructions_for_error(text)` (failure text to a "change exactly this" line with the rule id). Wired into `agent.py --heuristics {1,2}`, **off by default** because this repo has measured that every prompt change has to be A/B-tested with `--repeat` before it is believed.
* **Verifier checks in `nkibench.py`** — static: legacy namespace imports, removed `nl.load/store/par_dim/mgrid/arange/mask`, `dma_copy` on a PSUM tile, returning an SBUF/PSUM tile instead of a `shared_hbm` output. Simulator: a per-transfer DMA size histogram that reports "descriptor-bound" when most transfers move under 2 KiB per partition. All exercised by `--selftest`.

The rest of this document is the curated reading of the database: the rules that matter most, in the
order an agent meets them, with the numbers. Rule ids in parentheses point into the appendix.

---

## 0. First decide which regime you are in

Everything below depends on one number: arithmetic intensity, FLOPs per byte moved from HBM.

| Trainium2 per NeuronCore (physical) | value | source |
|---|---|---|
| DMA engines | 16, ~23 B/ns each | DMA bandwidth guide |
| aggregate DMA bandwidth | 368 GB/s | DMA bandwidth guide (`dma-peak-bandwidth-per-generation`) |
| measured attainable HBM bandwidth in a tuned kernel | ~435 GB/s (profile metadata) | profile-querying investigations (`hardware-peaks-trn2`) |
| Tensor Engine peak, bf16 | ~78.6 TFLOPS (profile metadata) | same |
| break-even intensity (compute = memory time), bf16 | ~181 FLOP/byte (78.6e12 / 435e9); the matmul tutorial quotes 222 | inferred / tutorial |
| logical core under LNC=2 | two physical cores: ~736 GB/s DMA, 2x PE | arch guide (`lnc-*`) |

Batch-1 LLM decode moves every weight byte once per token and does two FLOPs per weight element:
**intensity about 2 FLOP/byte, ninety times under the ridge.** In that regime nothing about the Tensor
Engine matters; only bytes moved, bytes moved twice, and the fixed cost per DMA matter
(`memory-vs-compute-bound-via-intensity`, `decode-intensity-memory-bound`). Prefill (hundreds of tokens)
is compute bound and the matmul rules in section 4 take over.

The verifier already prints this verdict (`MEMORY BOUND ... Nx more reuse needed`) before any timing exists.

## 1. Where custom kernels pay off, and where they do not

Annapurna's own guidance, confirmed by the docs and the library:

* **Dense matmul and convolution are already near peak through the compiler.** The graph compiler
  emits the same `nc_matmul` tiling an expert would; a hand-written kernel for a plain GEMM buys nothing
  unless it is *fusing* something around the matmul (norm, activation, residual, quantization) to
  remove an HBM round trip (`keep-intermediates-in-sbuf`, `activation-on-sbuf-not-hbm-roundtrip-after-matmul`).
* **Gather, scatter, interpolation, grid-sample style indirect access are where the compiler is weak**,
  because the generic lowering issues one small DMA per index. The NKI route is: put the irregular
  index pattern on the **HBM side** of a single DMA (`.ap(pattern, vector_offset=idx, indirect_dim=0)`),
  keep the SBUF side contiguous (`hbm-indexing-freedom`, `dma-vector-offset-rules`), generate index
  vectors on-chip with `nisa.iota` (`iota-for-index-generation`), and for in-SBUF gathers use
  `nc_n_gather` / `local_gather` rather than `dma_copy` (`dma-copy-sbuf-to-sbuf-prefer-compute-engine`).
  The bilinear/trilinear samples in nki-samples express 2x upsampling as a handful of affine-index
  gathers over a loaded tile instead of per-pixel loads (`interp-advanced-indexing-affine-mgrid`);
  pooling is a strided `.ap()` view plus one reduce (`pool-via-strided-ap-view-reduce`).
* **Indirect DMA has a hidden cost:** its descriptors must be generated at runtime by the GpSimd engine
  (software DGE, section 2), which is the "weak CPU" in the DMA path. Batch many indices per DMA
  (`batch-indirect-gathers-per-dma`), mark unused slots with -1 and `oob_mode.skip`
  (`indirect-dma-oob-skip-sentinel`), and never issue indirect DMAs inside an unrolled inner loop.
* **Framework ops with sparse engine activity and tiny DMA throughput in a profile are kernel candidates**
  (`framework-op-with-tiny-dma-utilization-is-kernel-candidate`, `choose-kernel-candidate-by-min-cut`).
  Fused decode blocks (norm + QKV + RoPE + attention + o_proj; norm + MLP) are the canonical example and
  AWS ships them in `nkilib` (`attention-block-tkg-keeps-everything-in-sbuf`).

## 2. DMA: the whole game when memory bound

The physical facts (DMA bandwidth guide, DGE deep dive):

* 16 DMA engines per core; **~1.3 us** cross-engine semaphore-to-start latency per transfer; a transfer
  touching only a few partitions uses only a few engines (`dma-engine-count-scales-with-partitions`).
* Bandwidth versus payload per partition: under 256 B severely overhead-bound; 256 B to 2 KiB improving;
  **2 KiB minimum recommended; 4 KiB saturates**. For bf16 that is a contiguous free dimension of at
  least 1024 elements per partition (`dma-free-dim-at-least-2kib-contiguous`, `dma-descriptor-at-least-4kib`).
  Measured on trn2 in the profiling investigations: 1 KiB to 4 KiB descriptors raised achieved
  bandwidth from 259 to 354 GB/s; row loads instead of tile loads from 152 to 246 GB/s
  (`row-loads-fewer-larger-transfers`).
* Descriptor generation (DGE) modes: `none` (host pre-computed, stored in HBM, lowest latency, static
  addresses only), `swdge` (GpSimd generates at runtime, the only mode for gather/scatter, competes with
  GpSimd compute), `hwdge` (dedicated block, ~600 ns per instruction, no indirect support, can be
  triggered from the Scalar engine to overlap with compute). Default `unknown` lets the compiler choose;
  override only after a profile shows descriptor stalls (`dma-dge-mode-selection`, `dge-mode-selection`).
* Strided DMA costs 1.2 to 1.5x a contiguous one and a two-DMA gather 2 to 2.5x; still cheaper than a
  contiguous load followed by an on-chip relayout (`strided-dma-cost-model`).
* The compiler splits trivial-dimension loads such as a `[1, H]` gamma into many 4-byte DMAs; reshape
  small vectors onto more partitions (`trivial-dim-dma-splitting`).

Rules:

1. Whole 128-partition tiles per `dma_copy`; never a DMA per row, element or partition
   (`no-element-wise-or-row-wise-dma`, `use-full-128-partitions`).
2. Coalesce along the free dimension: `[128, tiles*tile]` per call, not per 128x128 tile
   (`coalesce-dma-along-free-dim`, `coalesce-result-tiles-before-store`).
3. Load every input once; hoist loads out of the innermost loop; keep the residual and intermediates in
   SBUF (`load-once-reuse-in-sbuf`, `hoist-loads-out-of-k-loop`, `hoist-parameter-loads-out-of-inner-loops`).
4. Stream weights in large slabs through a ring of 2 to 4 SBUF tiles so the next slab's DMA overlaps the
   current matmul (`mlp-weight-tile-htile-ring-buffer`, `block-weight-loading-for-prefetch`).
5. Pre-lay-out weights offline in the layout the kernel wants (transposed `[K, M]`, unit-stride
   down-projection tiles) rather than transposing on device (`pretranspose-weights-offline`,
   `mlp-down-proj-optimized-weight-layout`, `down-proj-optimized-weight-layout-no-stride`).
6. Need a transposed load? `nisa.dma_transpose` in one step (2-byte dtypes, last dim <= 128, 16-row
   blocks for hwdge) beats `dma_copy` + `nc_transpose` (`dma-transpose-over-copy-plus-nc-transpose`,
   `dma-transpose-dtype-and-width`).
7. `dma_copy` cannot read or write PSUM (`dma-copy-no-psum-source`); `tensor_copy` to SBUF first.

The verifier's simulator layer counts transfers and bytes per transfer and now reports the descriptor-bound
case explicitly.

## 3. SBUF and PSUM budgets

| constant (`nl.tile_size`) | trn2 value |
|---|---|
| `pmax` | 128 partitions |
| SBUF | 24 MiB per core on trn2 (`sbuf_fmax_bytes` 212,984 B usable per partition; trn1 180,224; trn3 245,752) |
| PSUM | 16 KiB per partition = 8 banks x 2 KiB; `psum_bank_fmax` 512 fp32 per bank; 7 banks usable when `dma_transpose` lowers to a PE transpose |
| `gemm_stationary_fmax` / `gemm_moving_fmax` | 128 / 512 (trn3: moving up to 4096 fp32 dst, 8192 bf16 dst) |
| min alignment | 4 bytes |

Rules: keep live tiles under the per-partition budget (`sbuf-per-partition-capacity`); plan matmul
output columns per loop around 8 banks x 512 fp32 (`psum-bank-count-caps-I-per-loop`); pack several small
output tiles into one bank along the free axis for skinny decode matmuls (`down-proj-pack-multiple-h1-per-psum-bank`);
never give `tensor_tensor` two PSUM inputs and never give GpSimd a PSUM operand (`tt-inputs-not-both-psum`,
`gpsimd-cannot-access-psum`); use the library's `SbufManager` stack/heap discipline with
`interleave_degree` for multi-buffering (`sbufmanager-multibuffering`).

## 4. Tensor Engine matmul

`nisa.nc_matmul(dst, stationary, moving)` computes `dst = stationary.T @ moving`:

* Both operands in SBUF with the **contraction dimension on partitions (<= 128)**; `dst` in PSUM, fp32
  before trn3 (`mm-operands-sbuf-dst-psum`, `mm-contraction-on-partition-dim`, `mm-dst-dtype-fp32-pre-trn3`).
* Stationary free dim <= 128, moving free dim <= 512 on trn1/trn2 (`mm-stationary-free-le-128`, `mm-moving-free-le-512`).
* Accumulate across K tiles in **one PSUM tile allocated outside the K loop**; the first write must
  overwrite, never accumulate onto a memset PSUM (`psum-accumulate-over-k`, `mm-accumulate-first-write-must-overwrite`,
  `mm-no-accumulate-onto-memset-psum`). PSUM accumulation beats vector-engine adds (`psum-accumulation-beats-vector-accumulation`).
* Do not mix fp32/tf32 with bf16 operands in one matmul (`mm-supported-input-dtypes`).
* Skinny decode matmuls (T tokens, T < 128): make the **weight tile stationary** and the activation moving
  so the output lands with the next contraction dim already on partitions, no transpose
  (`tkg-lhs-rhs-swap-weight-stationary`); split PE columns when T < 128 (`pe-column-tiling-for-small-t`).
* Partition-axis reductions and broadcasts are matmuls against a ones tile (`ones-matmul-cross-partition-reduce`,
  `pe-broadcast-row-with-ones-matmul`); transposes use `nc_transpose` with the shared identity
  (`transpose-identity-reuse`), 128x128 on TensorE or 32x32 on VectorE.
* Every matmul is two profiler instructions, load-stationary then multiply; gaps between them are
  data starvation, not PE inefficiency (`matmul-is-two-hw-instructions`, `te-throttling-after-idle`).

## 5. Which engine runs what

TensorE: `nc_matmul`, `nc_transpose` (PSUM out). ScalarE: `activation`/`activate2` (exp, gelu, silu,
sigmoid, rsqrt, reciprocal; fp32 internally; fused `scale=` and `bias=` are free, so fold the preceding
multiply/add in: `fuse-scale-bias-into-activation`, `activation-scale-tensor-fuses-multiply-exp`).
VectorE: `tensor_tensor`, `tensor_scalar` (two ops for the price of one: `ts-two-ops-same-cost`),
`tensor_reduce`, `tensor_copy`, 32x32 transposes, `nisa.exponential` on trn3. GpSimd: `iota`,
`affine_select`, gathers, partition reductions, int32 math, software DGE; 4x slower than ScalarE for
rsqrt (`rsqrt-engine-tradeoff-and-psum`). Rules of thumb: `tensor_scalar` is 2x faster than
`tensor_tensor` in fp32 when one operand is a per-partition scalar; prefer fused reduce variants
(`activation_reduce`, `tensor_scalar_reduce`) over op + separate reduce; alternate PSUM evictions
between ScalarE and VectorE so neither becomes the bottleneck (`interleave-psum-evictions-across-engines`);
a residual add can ride on the DMA itself with `dma_compute` (`dma-compute-residual-add`).

## 6. Scheduling and pipelining

Write single-buffered correct code first; add multi-buffering only when a profile shows matmuls whose
last-finishing dependency is a DMA (`single-buffer-first-then-pipeline`, `matmul-last-finishing-dependency`).
Then: ring-buffer weight slabs with an offset start index to avoid write-after-read serialisation
(`weight-ring-offset-avoid-anti-dependency`); use `SbufManager.open_scope(interleave_degree=N)`; use
`Kernel.with_schedule` edges and address rotation for multi-buffered tensors when the compiler's order
is wrong (`scheduling-apis`); loops in 0.6.0 are fully unrolled (`affine_range`/`sequential_range`/
`static_range` are aliases of `range`), so long loops cost compile time and instruction memory, and
true runtime loops need `fori_loop`/`while_loop` (`loops-range-aliases-fully-unrolled`).

## 7. LNC=2: two physical cores per logical core

Launch with `kernel[2](...)` matching `NEURON_LOGICAL_NC_CONFIG` (`lnc-launch-bracket-matches-runtime`);
shard with `nl.program_id(0)` / `nl.num_programs(0)` and keep control flow identical on both cores.
Projections shard the hidden dimension so each core writes its slice with no collective and the next
kernel reads it contiguously via the transposed `[H0, n_prgs, H1_shard, T]` layout
(`transposed-io-layout-for-per-core-contiguous-dma`); partial contractions are exchanged as fp32 with
`nisa.sendrecv` (GpSimd DMA for <= 1 KiB per partition) (`lnc-h-shard-partial-sum-sendrecv`).
Decode attention at batch 1 cannot shard on batch, so it shards the prior sequence (needs
`s_prior >= 256`) and merges partial softmax statistics SBUF-to-SBUF (`attention-tkg-lnc2-sharding-rule`).
Tiny kernels (T <= 128) duplicate compute and shard only the store (`duplicate-compute-shard-store`).
Insert `nisa.core_barrier` after one core writes shared HBM the other will read (`lnc-core-barrier-before-cross-core-read`).

## 8. Numerics

Matmul inputs bf16, accumulation fp32 in PSUM; RMSNorm statistics via `activation_reduce(square, add)`
and a fused rsqrt with `scale=1/H, bias=eps` (`activation-reduce-for-sum-of-squares`, `rmsnorm-layout-by-T`);
softmax as one ScalarE exp with `bias=-rowmax` and a fused sum (`softmax-max-subtraction-as-negative-bias`);
fp8 e4m3 max is 240 on trn2 (OCP e4m3fn, max 448, is trn3 only); activation functions have valid input
ranges (sin has no range reduction outside [-pi, pi]; log, rsqrt, reciprocal have documented limits);
`range_select` needs an fp32 min fill or it produces NaN. Reduction registers are shared across
activation calls: reset them (`reset_reduce`) or results accumulate silently.

## 9. API constraints the agent breaks most (NKI 0.6.0)

Measured in this repo: a model trained on older NKI invents `nl.load`, `nl.store`, `nl.par_dim`,
`nl.mgrid`, `nl.arange`, `nl.mask`, `neuronxcc.nki` imports, `nki.benchmark`/`baremetal`/`profile`,
calls memory regions as functions, and returns SBUF tiles. All removed or never existed in 0.6.0
(`mig-neuronxcc-namespace-forbidden`, `mig-mask-mgrid-arange-removed`, `nl-load-store-are-experimental-wrappers`,
`mig-jit-kwargs-removed`, `output-tensors-shared-hbm`, `layout-partition-dim-first-no-block-dims`).
Every ISA call takes `dst=` first; buffers are objects (`nl.sbuf`, not strings); outputs are
`buffer=nl.shared_hbm`; every on-chip tile is at least 2-D with the partition dim first. The verifier's
static scan now names each of these with the fix.

## 10. Reading a profile

The bottleneck engine is the one with the largest interval-merged active time among Tensor, Vector,
Scalar, GpSimd and DMA (`bottleneck-engine-is-argmax-active-time`). For a DMA-bound kernel decompose the
gap into DMA idle, DMA inefficiency (descriptor size) and excess traffic (reloads and spills:
`excess_ratio = transferred / necessary`); fix reloads first because excess traffic is itself made of
inefficient transfers (`memory-family-gaps`, `investigate-reloads-before-dma-efficiency`). If the DMA
efficiency gap is under ~5%, stop tuning transfer sizes (`dma-efficiency-gap-threshold`). Bound any
single-engine optimisation by the next-slowest engine (`bottleneck-aware-speedup-cap`). The simulator
models no latency at all: never benchmark with it (`simulator-not-for-performance`). Compile the
profiled variant with `dge_mode` none or hwdge if you need DMA attribution to source lines.

## 11. Megakernel-specific rules learned while building the Qwen3-8B decode kernel

These came from `megakernel/` in this repo, measured on a Trainium2 workshop seat, and are not in the docs:

* AWS already ships a decode megakernel skeleton: `nkilib.experimental.transformer.transformer_tkg`
  (N layers per launch, attention block + MLP, residual optionally kept in SBUF). Build on it; do not
  rewrite attention or MLP.
* The standalone NumPy path interprets a Python **list** argument as one tensor per rank. Stack per-layer
  weights with a leading `[L, ...]` axis and index inside the kernel.
* `attention_tkg`'s pregenerated mask convention: the **last row of the `[S_ctx, B, q, S_tkg]` mask is
  the active token's own column**. Leave it 0 and the current token is silently ignored; the kernel
  still produces plausible numbers (30% error on real weights). Cache slot `S_ctx-1` is never read as
  prior.
* Keep the K cache transposed `[B, kv, d, S_max]` so QK^T needs no transpose (`write-k-cache-transposed-for-decode`).
* The torch HOP path (`libtorch_neuronx_lite.nki.nki_hop.wrap_nki(kernel)[2](...)` inside
  `torch.compile(backend="neuron_libtorch")`) keeps weights resident and propagates in-kernel cache
  writes through input/output aliasing; the executor is asynchronous with a shallow queue, so sync.
* Measured: per layer at Qwen3-8B shapes, the fused kernel runs at about 0.75 ms against a 0.53 ms
  HBM floor (70% of roofline) and 1.6x faster than per-op `torch.compile` of the same math; the
  per-op path's loss is exactly the HBM round trips between ops that section 2 describes.
