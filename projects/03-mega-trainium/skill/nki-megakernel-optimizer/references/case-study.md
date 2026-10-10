# Case study: Qwen3-8B batch-1 decode, one launch per token (Trainium2, 2026-10-10)

One trn2 device, one logical NeuronCore (LNC=2), bf16, greedy. Full record: `../../../megakernel/ATTEMPTS.md`,
`../../../megakernel/NOTE.md`, `../../../megakernel/README.md`.

## Result

| per token | ms |
|---|---|
| vLLM as deployed (TP=2, two logical cores) | 71.3 |
| per-op `torch.compile(neuron_libtorch)`, same math, same core | 47.50 (3 runs) |
| **one launch, device** | **27.74** (3 runs: 27.71–27.79) |
| **one launch, end to end in a generation loop** | **27.55** (~36 tokens/s) |
| HBM floor | 20.8 |

Correctness: teacher-forced argmax = HuggingFace fp32 on 94/96 steps over 3 prompts; both misses are
near-ties inside bf16 noise.

## How each phase of the skill paid off

| phase | what happened | effect |
|---|---|---|
| 1. reference + layout gate | reference matched HF to 9e-7 before any kernel ran | every later error was the kernel's, not the reference's |
| 2. build on nkilib | `attention_block_tkg` + `mlp` with our layer loop, stacked weights, QK-norm | 36 layers in one launch on day one |
| 3. checker | first real-weights run 52% off; fitting against wrong references found the mask convention in an afternoon | the diagnostic became the `tiny` gate; negative controls prove it names the bug |
| 4. profile + bounds | bytes = necessary; DMA idle 285 µs/layer during the MLP; the documented next step (SBUF residual) was ~1% | one flag: −5.8%, 26.5 → 25.0 ms (36 layers) |
| 5. one change at a time | SBUF residual measured +9.6% and was rejected; LM head hardware DGE −0.18 ms not worth a monkeypatch; own LM head −0.48 ms alone but faults in the loop | 18 attempts logged; 5 changes in the final kernel (attempts 2, 4, 8, 11, 14) |
| 6. host loop | 33.9 → 27.7 ms/token by releasing device outputs between calls (uploads were 13 µs) | end to end = device time |
| 7. spread | 3 × 30-call runs; 3 prompts × 32 steps | numbers with ranges, not anecdotes |

## Time spent (one person-day with an agent)

Reference and layer stack: morning. Mask bug: ~3 h. Profile-guided change: ~1 h. Full step: ~1.5 h.
Host-overhead hunt: ~1 h. Checker, controls, spread, write-up: ~2 h. Most of the cost was compiles
(36 layers: ~10 min each); iterate at 1–4 layers and confirm at 36.
