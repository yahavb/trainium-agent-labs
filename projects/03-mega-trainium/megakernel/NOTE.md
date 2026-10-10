# One-page note: a one-launch decode step for Qwen3-8B on Trainium2

**What we ran.** A batch-1 greedy decode step for Qwen3-8B (36 layers, bf16) as **one NKI kernel launch
per token**. The launch covers the token id → embedding gather → 36 decoder layers (attention with
Qwen3's QK-norm and an in-place KV-cache update, MLP) → final RMSNorm → LM head → argmax → next token
id. It's built on AWS's `nkilib` blocks (`attention_block_tkg`, `mlp`, `output_projection_tkg`,
`cascaded_max`), with our own layer loop, stacked weights, QK-norm wiring, an on-device decode mask,
an embedding gather and the LM-head layout. Checked by `check.py` (see `CHECKER.md`); every attempt is
in `ATTEMPTS.md`.

**On what.** One trn2 workshop seat: one Trainium2 device, of which we used **one logical NeuronCore**
(two physical cores, LNC=2; 736 GB/s DMA). Two of the other three served vLLM. Neuron 2.34 runtime,
neuronx-cc 2.27, NKI 0.6.0, torch 2.11 with `torch.compile(backend="neuron_libtorch")`.

**By whom.** The agent is Claude (Opus 5.5) in Claude Code, with one of us steering: it wrote every attempt
in `ATTEMPTS.md`, and `check.py`'s verdict, with its reason, decided the next one.

**What came out.**

|                                                                                               | ms per token    | runs, spread      |
| --------------------------------------------------------------------------------------------- | --------------- | ----------------- |
| per-op `torch.compile` of the same math, same core (the baseline)                             | **47.50** | 3 runs × median of 30 calls: 47.50–47.51     |
| **one launch, device time** (36 layers, S_ctx 1024)                                           | **27.74** | 3 runs × median of 30: 27.71–27.79     |
| **one launch, end to end** in a greedy generation loop (host loop incl. read-back, S_ctx 256) | **27.55**  | 3 runs × median of 31 steps: 27.54–27.56      |
| HBM floor (all weights + KV read once at 736 GB/s)                                            | 20.8            | arithmetic        |
| vLLM server as deployed (TP=2 on two logical cores; HTTP, sampling, host work)                | 71.3            | 1 run, 256 tokens |

So the one-launch step is **1.71× faster than per-op compilation of the same math** and runs at
75% of the memory-bandwidth floor. Generation runs at ~36 tokens/s on one logical core.

**Correctness**, real Qwen3-8B weights against HuggingFace float32, teacher-forced (both models fed the
same context, so every step is an independent test): **94 of 96 steps agree (3 prompts × 32 steps: 31, 32, 31)**, mean logits rel-L2 1.0–1.2e-2. Both disagreements are near-ties that bf16
rounding explains: HF's own fp32 margin between the two tokens was 0.028 and 0.041 logits, below 3σ of the
kernel's logit error at those steps (0.071 and 0.081). Free-running, the first 6 generated tokens match HF exactly; then a 0.028-logit near-tie
flips and the greedy texts diverge, as greedy texts do. The checker also rejects two injected bugs, the
original mask-convention bug and identity RoPE, and names each one.

**What we learned, in the order it changed the work.**

1. *A kernel can be 52% wrong and look fine.* The first real-weights run never crashed and gave
   plausible numbers. Comparing against deliberately wrong references found it: the output matched
   "new token does not attend to itself" to 3e-3. That comparison is now part of the checker.
2. *Profile before optimising.* The documented next step (keep the residual in SBUF) was measured at
   ~1% of the time, and implementing it made the kernel 10% slower. The profile showed TensorE starving DMA in
   the MLP; one tiling flag gave −5.8%.
3. *The host overhead was not where we thought.* A 7 ms gap per token between device time and the
   generate loop was not input uploads (0.013 ms) or changing values. It was holding the previous
   call's output tensors alive into the next call, which costs ~6 ms per call at 36 layers through
   torch.compile and grows with model size. Releasing them closed the gap (33.9 → 27.7 ms per token).
4. *Not everything worked.* A persistent kernel (K tokens per launch) hit a compiler internal error. Our
   own LM head is correct and 18% faster on its own (2.71 → 2.23 ms), but faults with an out-of-bounds DMA
   inside the generate loop, so it isn't in the final kernel. The SBUF residual path was slower. All are
   logged with numbers.

**Left on the table** (measured): ~0.5 ms per token in the LM head, once the tiled version's integration bug is
found; DMA efficiency inside the layers costs ~220 µs per layer (nkilib's queue choices). Tensor parallelism over the
two free logical cores is the route well below the single-core 20.8 ms floor.

Reproduce: `NEURON_RT_VISIBLE_CORES=2 python check.py --resident` (all gates) and `bash kernels/run_spread.sh` (all runs in this note).
