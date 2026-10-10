# Pitfalls, by symptom

Every entry cost real time on 2026-10-10 (trn2, Neuron 2.34, NKI 0.6.0, neuronx-cc 2.27, nkilib 0.0.0.0dev,
libtorch-neuronx-lite). Symptom → cause → fix.

## Wrong results that look right

| symptom | cause | fix |
|---|---|---|
| 30–52% error on real weights, no crash; error smaller when softmax is peaky | `attention_tkg` pregenerated-mask convention: the **last row of `[S_ctx, B, q, 1]` is the active token's own column**; left at 0 the new token never attends to itself | set it to 1; never use cache slot `S_ctx-1` as prior. Diagnose such bugs by fitting against deliberately wrong references |
| tiles 0..k-2 of every block off by ~1/√K, last tile exact | several `nc_matmul` accumulation groups (columns) in one PSUM tile | one PSUM tile per accumulation group. Note: nkiequiv proves the shared version equivalent; it is a hardware/compiler behaviour |
| free-running greedy text diverges from the framework after a few tokens | bf16 near-tie flips (margins of 0.03 logits) | compare teacher-forced; accept a disagreement only if margin < 3σ of logit error |

## Compile failures

| symptom | cause | fix |
|---|---|---|
| `NCC_EGCA111 Memory allocation failed ... Spilling is disabled` after fusing parts that each compile alone | an auto-allocating nkilib call (e.g. `output_projection_tkg`) next to manually placed buffers | pass a manual `BufferManager(0, 200 KB)` with `set_auto_alloc(False)`, `open_scope()` before `alloc` |
| `duplicate op name 'rope_pos_ids_load'` (or `L0_attn_...`) | a library op name without a per-call prefix inside a layer loop, or a loop body traced more than once | build that input yourself once per step (e.g. our `build_decode_mask`) |
| compile time ×K for a K-step loop | `nl.sequential_range` / `affine_range` with a constant trip count **unroll** in NKI 0.6 | `nl.dynamic_range` is a real device loop (traced once); its register can't index a DMA, so use a counter tensor + `vector_offset` |
| `NCC_IBVF059 tensor size must be more than 0` / walrus "empty MemoryLocationSet" | a full decode step inside `nl.dynamic_range` (compiler ICE, ~15 min even at 2 layers) | none found; don't build on it |
| `NKI does not support 'import' statements within a function`, `unsupported expression` (list comprehension), `failed to resolve name 'builtins.repr'` | the NKI tracer's Python subset | module-level imports, explicit loops, 3-D SBUF tensors instead of lists |

## Runtime failures

| symptom | cause | fix |
|---|---|---|
| `nrta status=1006`, "scatter/gather (indirect memory copy via scalar DGE) out-of-bound access" | an indirect DMA index is out of range (our tiled-LM-head step in the generate loop; unresolved) | read the neff instruction index with framework debug env vars; bisect by position/token |
| `nrt_init` failure when capturing a profile | the process that ran the kernel still holds the core | run the kernel in a child process, capture in the parent (`profile_mega.py`) |
| "Execution Queue Full" | torch path executes asynchronously with a shallow queue | sync each call (`.cpu()` of a small output) |

## Silent staleness (the worst kind)

| symptom | cause | fix |
|---|---|---|
| an edit to a helper changes nothing | libtorch's NKI trace cache (`~/.cache/neuron_libtorch/neuron/compile_cache/nki/`) keys on the **top-level kernel's source only** | delete the entry for that kernel (decode `dumped_config` → `func_name`), or set `NEURON_LIBTORCH_DISABLE_COMPILE_CACHE=1` |
| a runtime monkeypatch of an nkilib helper "has no effect" (or leaks into later runs) | same cache | disable the cache for patched runs |

## Performance traps

| symptom | cause | fix |
|---|---|---|
| a generate loop is 6 ms/token slower than the device time (36 layers; 0.7 at 2) | keeping the previous call's **output tensors alive** (or feeding them back as inputs) through `torch.compile(neuron_libtorch)` with an in-place KV cache | read the token to host (7 µs), re-upload (13 µs), drop the output tensors |
| a large weight stream at ~230 GB/s with GpSimd 100% busy | software DGE: one `DMA_DIRECT2D` per row block issued serially on GpSimd | hardware DGE (`dge_mode=hwdge`, alternate `engine=sync/scalar`), bigger contiguous per-partition chunks |
| DMA idle 35% of a layer while bytes = necessary | TensorE consuming weights as the moving operand slower than DMA delivers | weights stationary (nkilib `use_tkg_gate_up_proj_column_tiling=False`) |
| "keep the residual in SBUF" makes it slower | nkilib's `transposed_out`/`out_in_sb` o_proj path loads `W_out` in 2 serial transfers | measure before adopting documented "next steps" |

## Tooling

* `neuron-profile` is gone: use `neuron-explorer capture/view` (`--ingest-only` → parquet).
* `DmaPacket` is sampled; `DmaPacketAggregated` matches the Summary's byte totals.
* `pkill -f <pattern>` / `until ! pgrep -f <pattern>` from a shell whose own command line contains the
  pattern matches itself. Use PIDs.
