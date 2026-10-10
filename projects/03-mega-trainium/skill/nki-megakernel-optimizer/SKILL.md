---
name: nki-megakernel-optimizer
description: >
  Turn a model's decode step (or any chain of NKI kernels) into a verified, profile-optimized
  megakernel on AWS Trainium: one launch per step, correctness gated at every change, optimizations
  chosen from device performance counters rather than guessed. Use when asked to fuse layers into one
  NKI launch, speed up batch-1 decode on Trainium/Neuron, build a megakernel, or optimize an NKI kernel
  with profiling (neuron-explorer, bounds analysis). Worked example: Qwen3-8B, 36 layers, 47.5 -> 27.7 ms/token.
---

# NKI megakernel optimizer

A loop for an agent: **build → verify → measure → change one thing → verify again → log it**.
The kernel is never trusted because it runs. A wrong NKI kernel compiles, runs and returns
plausible numbers; our first Qwen3 megakernel was 52% wrong and looked fine. And no optimization
is chosen without a profile: the obvious next step in our case measured at 1% of the time and made
the kernel 10% slower.

Read `references/pitfalls.md` before the first compile. It lists every trap that cost hours, by symptom.
`references/case-study.md` is the full worked example with numbers; `../../examples/lm-head-walkthrough/`
takes one piece through the corpus, nkiequiv and the device gates, step by step.

## The rules corpus (use it in every phase)

`../../heuristics/` holds 724 de-duplicated NKI rules scraped from the NKI docs, AWS's nki-samples and
AWS's own agent skills, each with a trigger, a fix and its source (`NKI_HEURISTICS.md` is the curated
read; `APPENDIX.md` lists every rule; `cards/` has one page per topic). Query it rather than reading it:

```python
from heuristics import heuristics as H          # run from the repo root
H.card()                                        # ~350-token always-on card: put it in every kernel-writing prompt
H.retrieve("psum accumulate K loop", k=5)       # before writing or changing a piece
H.for_error(traceback_text)                     # on any compile/runtime failure: matching rules phrased as fixes
```

`references/pitfalls.md` adds what the corpus didn't have: the traps found building this megakernel.

## Phase 0: the floor, before any code

Batch-1 decode is memory bound: every weight byte is read once per token at ~2 FLOP/byte.
Compute the floor: `(weight bytes + KV bytes read) / DMA bandwidth` (Trainium2, one logical core at
LNC=2: ~736 GB/s; profile metadata reports 435 GB/s per physical core, and two cores share one HBM stack).
Every later number is reported as a fraction of this floor, so you always know what is left.

## Phase 1: the reference and the layout contract

1. Write a float32 reference of the exact math in torch (`qwen3_ref.py` pattern), in the layouts the kernel
   will use (packed QKV, transposed K cache, mask convention, RoPE convention).
2. Gate it against the framework model (HuggingFace) on a tiny config: **max rel diff < 1e-5**. If this
   fails, nothing downstream means anything.
3. Run the same reference in bf16 on the device through `torch.compile(backend="neuron_libtorch")`: that's
   the **per-op baseline** the megakernel must beat.

## Phase 2: build on library blocks, don't rewrite them

Check `nkilib` first (`nkilib.experimental.transformer.transformer_tkg`, `attention_block_tkg`, `mlp`,
`output_projection_tkg`, `rmsnorm_tkg`, `cascaded_max`). Write only the glue: the layer loop, stacked
`[L, ...]` weights (a Python list argument means "one tensor per rank" in the standalone path), model
specifics (e.g. QK-norm), and the pieces the library lacks (embedding gather, on-device mask, LM-head layout).

## Phase 3: the checker, with five gates, cheapest first

Template: `../../megakernel/check.py` (reasoning in `../../megakernel/CHECKER.md`). Every rejection must **name the change to make**.

| gate | accept when | on failure, say |
|---|---|---|
| layout | reference = framework model, < 1e-5 | fix the reference first |
| tiny | kernel max error ≤ 3× the bf16-rounding floor of the same math | which *deliberately wrong reference* the output matches best ("kernel implements active_token_excluded") |
| piece proofs (optional) | `nkiequiv` (github.com/forthoney/symnki) proves a self-contained piece EQUIVALENT to a NumPy spec | the counterexample's missing term. See `../../megakernel/symnki_poc/` |
| real | real weights, all layers, **teacher-forced**: every argmax disagreement has framework margin < 3σ of the kernel's logit error | the step, both tokens, margin vs 3σ: "a real error, not rounding" |
| speed | faster than per-op baseline | both times |

Include **negative controls** (`megakernel/check.py --control ...`): inject a known bug into the kernel's inputs and
confirm the gate rejects it *and names it*. A checker that has never rejected anything is unverified.

## Phase 4: profile, compute bounds, pick ONE change

```bash
NEURON_RT_VISIBLE_CORES=2 python megakernel/kernels/profile_mega.py --layers 1 --tag base_L1   # capture (child process holds the core)
python megakernel/kernels/analyze_profile.py base_L1                                          # bounds + phases + DMA-idle by source line
```

Read the result in this order (AWS's bounds method, `neuron-nki-profile-querying`):

| what the bounds say | what to do |
|---|---|
| bytes moved > necessary bytes | fix reloads/spills first (they are also inefficient transfers) |
| DMA idle is the largest gap | find what runs during the idle windows (`analyze_profile.py` attributes them to source lines); usually TensorE work not overlapped with weight streaming → change the matmul dataflow (weights stationary vs moving: `use_tkg_gate_up_proj_column_tiling=False` gave −5.8%) |
| DMA busy but far from ideal bandwidth | small/strided descriptors or a serial queue: check queue types (software DGE on GpSimd serializes), bytes per partition per DMA (aim ≥ 32 KB), transfers in flight |
| a block is far slower than its bytes | profile that block alone (`--head`, `--head-tiled`) |
| total ≈ floor | stop |

Use `DmaPacketAggregated` for byte counts (`DmaPacket` is sampled on this profiler build).

## Phase 5: change, re-gate, log

One change per attempt. Re-run `check.py` (at least `tiny` + `head`; `real` + `speed` before adopting).
Append to the attempt log (`../../megakernel/ATTEMPTS.md` format): attempt, correctness score, speed score, verdict,
**what it taught**. Log rejections with the same care as wins; they are what the next agent needs.

Timebox each attempt. An unresolved bug after the timebox is logged and set aside (our persistent kernel
hit a compiler ICE; our tiled LM head is 18% faster alone but faults in the loop; both logged, neither adopted).

## Phase 6: integration and the host loop

The device time is not the user's time. Measure the real generation loop, and diagnose any gap with
call-pattern experiments (same inputs / fresh uploads / changing values / outputs fed back) before
optimizing. In our case the 6 ms/token gap was holding the previous call's output tensors alive, not
the uploads (13 µs).

## Phase 7: report with spread

Repeat speed runs (≥ 3, median of ≥ 30 calls each) and correctness over ≥ 3 prompts
(`megakernel/kernels/run_spread.sh`). Report against the per-op baseline and the floor.
