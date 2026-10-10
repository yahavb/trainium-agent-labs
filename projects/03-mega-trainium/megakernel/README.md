# Project 3 — A one-launch decode step for Qwen3-8B on Trainium2

**Token id in, next token id out, in one NKI kernel launch per token:** embedding, 36 decoder layers,
final norm, LM head and greedy argmax, on one logical NeuronCore, with weights and KV cache resident on
the device. Built and measured on 2026-10-10 on a workshop trn2 seat (LNC=2, Neuron 2.34, NKI 0.6.0).

> **Hand-in (the organizers' three deliverables):**
> 1. **The checker:** [`check.py`](check.py), with the reasoning in [`CHECKER.md`](CHECKER.md)
> 2. **The attempt log:** [`ATTEMPTS.md`](ATTEMPTS.md) (18 attempts with scores) and `results/check_log.jsonl`
> 3. **The one-page note:** [`NOTE.md`](NOTE.md)

| per token, batch 1, greedy | ms |
|---|---|
| per-op `torch.compile` of the same math, same core | 47.50 (3 runs: 47.50–47.51) |
| **one launch, device** | **27.74** (3 runs: 27.71–27.79) → **1.71× faster** |
| **one launch, end to end in a generation loop** | **27.55** (3 runs: 27.54–27.56), **~36 tokens/s** |
| HBM floor | 20.8 |
| vLLM server as deployed (TP=2) | 71.3 |

Correct on real Qwen3-8B weights: teacher-forced argmax matches HuggingFace float32 on **94 of 96 steps** (3 prompts ×
32 steps), and both misses are near-ties (0.028 and 0.041 logits apart in HF's own fp32 logits) that bf16
rounding explains. The checker accepts it and rejects two injected bugs, naming each. Details: `NOTE.md`.

```bash
NEURON_RT_VISIBLE_CORES=2 python check.py --resident                              # all five gates, final kernel
NEURON_RT_VISIBLE_CORES=2 python kernels/generate.py --layers 36 --new 32 --resident  # generate, compare with HF
NEURON_RT_VISIBLE_CORES=2 bash kernels/run_spread.sh                              # every run in NOTE.md
```

The sections below are the engineering record in the order it happened: the layer stack, then
profile-guided changes (Follow-up 1), then the full step (Follow-up 2) and the host-overhead hunt
(Follow-up 3).

## What is in the box

| file | what |
|---|---|
| `kernels/qwen3_megakernel.py` | the kernel: N layers per launch, built on `nkilib.experimental.transformer` (attention_block_tkg + mlp) with Qwen3's QK-norm wired in and stacked `[L, ...]` weights |
| `kernels/qwen3_ref.py` | the layout contract and a float32 torch reference (validated against HuggingFace to 1e-6); the same function in bf16 on the device is the per-op baseline |
| `kernels/test_tiny.py` | tiny-config test: reference vs HF, kernel vs reference, convention diagnostics (`--zero attn/mlp` attributes error per block) |
| `kernels/test_qwen3_8b.py` | real weights: kernel vs HF hidden state after layer L-1; with `--layers 36` also next-token / top-5 / logit agreement |
| `kernels/bench_mega.py` | megakernel vs per-op `torch.compile(backend="neuron_libtorch")`, same core, device-resident weights, batch 1 (`--kernel`, `--kw` select variants and compile-time options) |
| `check.py` | **the checker**: five gates (layout, tiny, head, real, speed), each with a reason that names the fix; `--control` injects known bugs to show it rejects them |
| `kernels/qwen3_decode_step.py` | **the full decode step in one launch**. `qwen3_decode_step_resident` (final): token id + position in, next token + position out; RoPE rows gathered from resident tables, decode mask built on the device. `qwen3_decode_step`: same with host-built RoPE and mask. `qwen3_decode_loop`: the persistent K-tokens-per-launch attempt (does not compile; see ATTEMPTS #15). `pack_lm_head` lays out the LM-head weights |
| `kernels/generate.py` | greedy generation with one launch per token (prompt prefilled through the same kernel), KV cache resident on the device, compared token-for-token with HF fp32 (`--teacher-force` compares every step on HF's context; `--resident` uses the final kernel) |
| `kernels/run_spread.sh` | the repeated runs behind `NOTE.md`'s spread |
| `kernels/bench_step.py` | full step, one launch vs per-op `torch.compile` of the same math (embedding + layers + norm + LM head + argmax) |
| `kernels/test_head.py` | device test of the embedding gather and LM head + argmax at Qwen3-8B vocabulary size |
| `kernels/profile_mega.py`, `analyze_profile.py` | `neuron-explorer` capture plus bounds analysis (AWS method, plus per-phase and DMA-idle attribution to source lines); `--head` profiles the embedding + LM head |
| `kernels/sweep_attn.py`, `sweep_mask.py`, `hyp_active.py` | the diagnostics that found the mask-convention bug (kept because the method matters) |
| `baseline/smoke_nc0.py`, `baseline/smoke_torch_compile.py` | prove a NKI kernel and the torch backend run standalone on a free core |
| `results/` | logs and JSONL from the runs described below |

Run everything with `NEURON_RT_VISIBLE_CORES=2` or `3`: those are the idle logical cores while the vLLM
server holds 0 and 1 (the README's "NC2/NC3 hold the model" is wrong for this pod; verified with
`neuron-monitor`).

## Why a megakernel, in numbers

Batch-1 decode is memory bound: every weight byte is read once per token and does two FLOPs, so the
arithmetic intensity is about 2 FLOP/byte against a ridge near 200. The only things that matter are
bytes moved and the fixed cost per DMA. A per-op graph pays an HBM round trip between every op (norm
out, QKV out, attention out, residual, MLP intermediates) and a launch gap per kernel; a fused layer
keeps all of that in SBUF and streams only weights and KV.

Measured on this pod (one logical core = two physical cores, 736 GB/s aggregate DMA):

| | ms per decode step | ms per layer | notes |
|---|---|---|---|
| vLLM server, Qwen3-8B, TP=2 over two logical cores | **71.3** | 2.0 | HTTP API, 256-token greedy run; includes sampling and host work |
| per-op `torch.compile(neuron_libtorch)`, 1 / 2 / 4 / **36** layers | 1.21 / 2.63 / 5.04 / **45.3** | 1.21–1.32 | same math as the reference, bf16, one core; 36-layer compile 242 s |
| megakernel, 1 / 2 / 4 / 8 layers | 0.91 / 1.66 / 3.12 / 6.01 | 0.91 → 0.75 | one launch; compile 18 s / 31 s / 64 s / 139 s |
| HBM floor (weights + KV once at 736 GB/s) | 0.53 per layer | 0.53 | the ceiling no kernel beats |
| **megakernel, 36 layers** | **26.5** | 0.74 | one launch; compile 563 s; HBM floor 19.1 ms (72% of roofline) |

So the fused stack runs at 72% of the HBM roofline and 1.7x faster than per-op compilation of the same
math on the same core (26.5 ms vs 45.3 ms per token on ONE logical core), against 71 ms for the vLLM
server on two. The vLLM number is not like-for-like (it carries
tensor-parallel collectives, the LM head, sampling and HTTP), so read it as "the stack as deployed",
not as the per-op baseline. The remaining 30% to the floor is the per-layer HBM residual round trip
(the library's `sbuf_residual_and_cc=True` path removes it), the descriptor cost of the KV-cache
indirect update, and the un-overlapped tail of each block; the heuristics document lists the fixes.
**Measured later, this guess was wrong:** the profile put the residual round trip at ~1% of the time, and the
`sbuf_residual_and_cc` path made the kernel 9.6% slower. See Follow-up 1.

## Correctness

| test | result |
|---|---|
| float32 reference vs HuggingFace Qwen3 (tiny config, 2 layers) | max rel diff 9e-7 |
| kernel vs float32 reference (tiny, 2 layers, bf16 inputs) | 1.0e-2 rel-L2, bf16 rounding alone 7e-3 |
| kernel vs HuggingFace fp32 hidden state, **real Qwen3-8B weights, 4 layers** | **4.7e-3 rel-L2** (the torch bf16 reference of the same math: 1.0e-2) |
| **36 layers, real weights**: hidden state vs HF fp32 | **9.9e-3 rel-L2**; next token `' Kernel'` = HF; top-5 overlap 5/5; logits rel-L2 9.9e-3 |

### The bug that took the afternoon, and how it was found

The first real-weights run was 52% off while the MLP-only path matched to bf16 precision. The
attribution went: zero the MLP (error stays) / zero attention (error vanishes) → attention; a shape
sweep (error present at every shape, smaller when the softmax is peakier) → not a layout bug but a
*missing contribution*; a hypothesis fit against references with the active token excluded / double
counted / un-rotated / un-normalised → **"active token excluded" matched to 3e-3 at every write position.**
Reading `gen_mask_tkg._load_active_mask` then showed why: in the pregenerated-mask convention the
**last row of the `[S_ctx, B, q, S_tkg]` mask is the active token's own column**, and my mask had zeros
there. One line in the mask builder fixed it. The kernel never crashed and always returned plausible
numbers, which is exactly the failure class the kernel-agent project is about.

## Design decisions

* **Build on `nkilib`, do not rewrite attention or MLP.** `transformer_tkg` is AWS's own N-layer decode
  megakernel; it lacked Qwen3's QK-norm and takes Python lists of weights, which the standalone path
  reads as "one tensor per rank". The kernel here is that loop with stacked `[L, ...]` weights and the
  QK-norm gammas wired to `rmsnorm_QK_pre_rope_*`.
* **Flat transposed KV cache** `[L, B, kv, d, S_max]` / `[L, B, kv, S_max, d]`, updated in-kernel at
  `pos_ids`; the mask gates prior slots `< pos` and sets the active column.
* **Torch integration through the HOP path** (`wrap_nki(kernel)[2](...)` inside
  `torch.compile(backend="neuron_libtorch")`), the same mechanism vLLM uses: weights stay resident and
  in-kernel cache writes propagate through input/output aliasing. The executor is asynchronous with a
  shallow queue, so the benchmark syncs each call (`y[0].cpu()`), which slightly favours the baseline.
* **One logical core, LNC=2.** The two physical cores shard the hidden dimension inside each block.
  Tensor parallelism across logical cores (collectives) is out of scope here; it is what vLLM does with
  TP=2 and is the next step if latency per token matters more than cores.

## Follow-up 1: profile first, then optimise (2026-10-10)

One layer profiled on the device with `neuron-explorer` (`kernels/profile_mega.py` captures,
`kernels/analyze_profile.py` computes AWS's bounds plus a per-phase and DMA-idle breakdown;
summaries in `results/profiles/*/bounds.json`).

| per layer, Qwen3-8B shapes, S_ctx 1024 | µs | what it says |
|---|---|---|
| kernel total | 818 | |
| DMA active (any queue busy) | 532 | the bottleneck engine is DMA |
| ideal DMA time for the bytes actually moved (435 GB/s per physical core) | 449 | |
| ideal DMA time for the necessary bytes (weights + KV read once) | 448 | **no excess traffic**: moved = necessary |
| DMA idle inside the kernel | 285 | **the dominant gap (35%)** |
| ...of which during the MLP down-projection | 172 | all weights already loaded; TensorE still on gate/up and down |
| ...of which in our residual / barrier glue | 6–10 | the HBM residual round trip is about 1% |

So the prediction above (that the residual round trip makes up most of the remaining 30%) was wrong. The
bytes are already minimal. Time goes where the TensorEngine can't keep up with DMA: gate/up streamed
the weights as the moving operand (1536 matmuls, free size 256, 372 ns each, ~280 GB/s per core of
weights), while the weights-stationary down projection reached ~375 GB/s.

What was tried, at L=4 (one launch, 100 iterations, nothing else on the device;
`results/variant_compare_L4.jsonl`):

| variant | ms (L=4) | vs before |
|---|---|---|
| before: HBM residual, nkilib default gate/up column tiling | 3.091 | |
| **HBM residual, `use_tkg_gate_up_proj_column_tiling=False` (weights stationary)** | **2.911** | **−5.8%** |
| SBUF residual (`qwen3_decode_layers_sbuf`, transformer_tkg's `sbuf_residual_and_cc` path, TP all-reduce dropped, hidden kept in SBUF across layers) | 3.389 | +9.6% |
| SBUF residual + weights-stationary gate/up | 3.082 | −0.3% |

The SBUF path saves ~8 µs per layer of glue but makes the attention block ~44 µs slower (profile
`sbuf_L2`: with `transposed_out=True, out_in_sb=True` the output projection loads `W_out` in 2 large
serialized transfers instead of 64, adding a 32 µs DMA-idle gap). It is correct (tiny test: same
1.0e-2 error as the HBM path) and kept for reference, but it is not the default. Down-projection
column tiling made no difference (sweep in `results/mlp_knob_sweep.jsonl`; that sweep overlapped other
device jobs, so only the clean table above is quoted).

**New default: weights-stationary gate/up. 36 layers: 26.5 → 25.0 ms per step (76% of the HBM
floor).** Real weights still match HF: hidden rel-L2 9.0e-3, next token and top-5 identical
(`results/correctness_8b_L36_gateup_stationary.log`).

## Follow-up 2: the full decode step in one launch (2026-10-10)

`kernels/qwen3_decode_step.py`: **token id in, next token id out, one NKI launch per token.**

| stage | how |
|---|---|
| embedding | one-row gather with an indirect DMA (`embed.ap(..., vector_offset=token, indirect_dim=0)`), written to HBM by core 0 |
| 36 layers | `qwen3_decode_layers` as above; the KV cache is updated in place at `pos_ids` |
| final RMSNorm | nkilib `rmsnorm_tkg`; the result stays in SBUF as `[128, 1, 32]` |
| LM head | nkilib `output_projection_tkg`, vocab (151,936, unpadded) sharded over the two physical cores; weight rows permuted on the host (`pack_lm_head`) to match the norm's SBUF layout, so no in-kernel transpose |
| sampling | greedy: nkilib `cascaded_max_core`, token id written by core 0 |

Two things that cost time: with its default auto-allocating buffer manager, `output_projection_tkg`
fails SBUF allocation once it sits behind the layer stack (`NCC_EGCA111`, although each part compiles
alone); a manual `BufferManager` over the region the MLP uses fixes it. And libtorch's NKI trace cache
keys on the top-level kernel's source only, so after editing a helper you must delete the stale entry
in `~/.cache/neuron_libtorch/neuron/compile_cache/nki/`, or the old trace is reused silently.

**Correctness, real Qwen3-8B weights, 36 layers, greedy, 21-token prompt** (`results/generate.jsonl`):

| check | result |
|---|---|
| free-running generation vs HF fp32 greedy | first 6 tokens identical; at token 6 the kernel picks HF's #2 (`' Each'` vs `' The'`, **0.028 apart in HF's own fp32 logits**), after which the texts differ as greedy texts do |
| teacher-forced (kernel fed HF's tokens, 32 steps) | **argmax agrees 31/32**; the one miss is that 0.028 near-tie; mean logits rel-L2 1.2e-2 |
| embedding gather / LM head alone (`test_head.py`, random weights) | row gathered bit-exactly; logits rel-L2 2.4e-3; argmax correct |
| 2 layers vs HF truncated to 2 layers | 16/16 tokens identical; logits rel-L2 5.6e-3 |

The near-tie flip is expected with bf16 weights: the kernel's logits differ from fp32 by ~1%, which
is larger than a 0.028 gap on logits of magnitude ~20.

The kernel's greedy text: *" use them all at once. Each NeuronCore has 128 32-bit MACs, and each MAC
can do a multiply and accumulate operation"*; HF fp32: *" use them all at once. The Neuron Kernel
Interface is a library that allows you to run your code on the NeuronCores. The Neuron Kernel"*.

**Speed** (same core, batch 1, S_ctx 1024, `results/bench_step.jsonl`):

| full decode step (embedding + 36 layers + norm + LM head + argmax) | ms per token |
|---|---|
| per-op `torch.compile(neuron_libtorch)` of the same math | 47.5 |
| **one launch** | **27.8** (1.71x; 75% of the floor) |
| HBM floor (layers + LM head + KV at 736 GB/s) | 20.77 |
| vLLM server as deployed (TP=2, includes HTTP, sampling, host work) | 71.3 |

In the generate loop (S_ctx 256) a step first took 27.5 ms on the device but 34.8 ms including the host.
Follow-up 3 found and fixed the gap.

## Follow-up 3: where the host time went, and the persistent-kernel attempt (2026-10-10)

The obvious explanation for the 7 ms gap was the per-step uploads (token, cos/sin, mask, position).
`qwen3_decode_step_resident` removed them all: RoPE rows gathered from resident tables, decode mask
built on the device (`build_decode_mask`, equal to the host mask at 4 positions), and position and token
in and out as two integers. It's correct (2 layers: 16/16 tokens, logits identical), but the gap stayed.
A diagnostic of call patterns at 36 layers settled it:

| call pattern (36 layers, one launch, sync on the token) | median ms |
|---|---|
| same inputs every call | 27.30 |
| fresh uploads every call (same values) | 27.33 |
| fresh uploads, position changing every call | 27.43 |
| fresh uploads, token changing every call | 27.44 |
| **the kernel's outputs fed back as the next inputs** | **33.73** |
| uploading the two integers / reading the token back | 0.013 / 0.007 |

The generate loop kept the previous step's output tensors alive while issuing the next call. Through
`torch.compile(neuron_libtorch)` with an in-place-mutated KV cache, that costs ~6.4 ms per call at 36
layers (0.7 ms at 2), growing with model size, consistent with a copy-on-write of the cache. Releasing
the outputs right after reading the token: **end to end 33.9 → 27.7 ms per token, equal to the device
time.** The token goes to the host (7 µs) and comes back (13 µs).

The persistent kernel (`qwen3_decode_loop`, K tokens per launch with the token fed back on chip) was the
other way to remove that cost. `nl.sequential_range` unrolls in NKI 0.6, multiplying compile time, so
it uses `nl.dynamic_range`, a real device loop traced once, with a counter-indexed scatter for the
output tokens. The mechanics work on a toy kernel. The full step inside the loop hits a neuronx-cc
internal error (`NCC_IBVF059`, "tensor size must be more than 0") after ~15 minutes even at 2 layers.
With the loop fixed, the remaining per-token host cost is ~0.05 ms, so it was left there.

LM head on hardware DGE (Sync and Activation queues instead of GpSimd software DGE, `--hwdge-oproj`):
2.71 → 2.53 ms for the head, no change in the layers. The limit is now nkilib's ~1.2 KB-per-partition
weight access pattern. Not adopted: a 0.18 ms gain doesn't justify replacing a library helper at
runtime and disabling the compile cache.

## What to do next

Measured, in order of payoff:

1. **LM head: about 0.5 ms per token, half done.** Our own LM-head matmul (`lm_head_tiled_argmax`: host-pre-tiled weights,
   32 KB per partition per DMA, weights stationary) is correct and takes the head from 2.71 to 2.23 ms on its own,
   and the full step from 4.28 to 3.71 ms at 2 layers at a fixed position. Inside the generate loop it hits an
   out-of-bounds indirect DMA on the second token (ATTEMPTS #18). Find that and adopt it.
2. **DMA efficiency inside the layers: about 220 µs per layer** (profile `v2_L1`: DMA busy 672 µs against 449 µs ideal;
   DMA-idle time already down from 285 to 67 µs). The weight streams run at 170–300 GB/s per queue
   (gate/up on one hardware-DGE queue, `W_out` and the KV cache on software DGE). Fixing it means changing nkilib's queue and
   DGE choices, not this kernel's structure.
3. Tensor parallelism over both free logical cores (2 and 3) roughly halves the per-token bytes per core,
   at the cost of one all-reduce per block. That's how vLLM runs, and it's the way to go well below 20 ms.

Tried and measured, not worth it on this build: the `sbuf_residual_and_cc` / `transposed_out` path
(+10% at L=4, Follow-up 1); a persistent multi-token kernel (compiler internal error; ~0.05 ms per token
left to gain after Follow-up 3); hardware DGE for the LM head (−0.18 ms).
