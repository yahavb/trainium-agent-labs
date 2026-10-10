# mega-trainium

**A method, and the tools, for turning a model's decode step into a verified, profile-optimized
megakernel on AWS Trainium.** Worked out on Qwen3-8B during Hack the Chip (2026-10-10): one NKI
launch per token, **27.7 ms per token against 47.5 ms** for per-op `torch.compile` of the same math on
the same core, correct against HuggingFace on real weights.

```
$ cd megakernel && NEURON_RT_VISIBLE_CORES=2 python check.py --resident     # final kernel, 2026-10-10 20:30 UTC
[PASS] layout  reference matches HF  (17s)
[PASS] tiny    max error 1.02e-02 within 2.2e-02 (3x the 7.3e-03 that bf16 rounding alone gives)  (17s)
[PASS] head    embedding exact, logits rel-L2 2.4e-03, argmax consistent  (48s)
[PASS] real    teacher-forced argmax agrees on 31/32 steps; every disagreement is a near-tie inside bf16 noise (step 6: HF margin 0.028 < 3 sigma 0.0712)  (142s)
[PASS] speed   27.74 ms/token vs 47.60 per-op (1.72x), 75% of the HBM floor  (82s)
ACCEPTED
```

With the original mask bug injected (`--control mask_active_zero`), the same checker rejects and names it: "the output is closer to the 'active_token_excluded' variant (rel-L2 1.01e-02) than to the correct reference (5.61e-02): the kernel implements active_token_excluded" (`megakernel/results/check_log.jsonl`).

**The agent and its loop.** The agent is Claude (Opus 5.5) in Claude Code, with one of us steering. Each
attempt in [`megakernel/ATTEMPTS.md`](megakernel/ATTEMPTS.md) is a kernel the agent wrote; `check.py` judged it, and the
verdict with its reason, plus a device profile, decided the next attempt. The skill below is that loop
written down, so the next run starts from it.

A faster kernel for one model is a narrow product. What carries over to the next model and the next
chip is the loop that found it: library blocks as the base, a checker that names the bug, profiles
read as bounds, one measured change at a time, and every attempt logged. This repo packages that loop
as an agent skill, with the Qwen3-8B kernel as the worked example and a 724-rule NKI corpus the agent
can query.

## Results (Qwen3-8B, batch 1, greedy, one Trainium2 logical core)

| per token | ms |
|---|---|
| vLLM server as deployed (TP=2 on two logical cores) | 71.3 |
| per-op `torch.compile(neuron_libtorch)`, same math, same core | 47.50 (3 runs: 47.50–47.51) |
| **one launch, on the device** | **27.74** (3 runs: 27.71–27.79) |
| **one launch, end to end in a generation loop** | **27.55** (~36 tokens/s) |
| memory-bandwidth floor (all weights + KV once at 736 GB/s) | 20.8 |

Where the 7.0 ms above the floor are:

| part of the step | measured | floor at 736 GB/s | fraction of floor |
|---|---|---|---|
| 36 decoder layers alone | 25.00 ms (1 run) | 19.08 ms | 76% |
| embedding, final norm, LM head, argmax (by subtraction) | ~2.7 ms | 1.69 ms | ~62% |
| **whole step** | **27.74 ms** | **20.77 ms** | **75%**, 551 GB/s achieved |

About 5.9 ms of the gap is inside the layers (DMA efficiency within the nkilib blocks) and about 1.0 ms in
the LM head, where our own head would recover ~0.5 ms once its loop fault is fixed. Sources:
`megakernel/results/bench_8b_L36_v2.jsonl`, `spread_bench_step.jsonl`.

Correctness: teacher-forced argmax matches HuggingFace float32 on **94 of 96 steps** (3 prompts × 32);
both misses are near-ties (0.028 and 0.041 logits apart in HF's own logits) inside bf16 noise. The
checker accepts the kernel on all five gates and rejects two injected bugs, naming each.

## What's here

| path | what |
|---|---|
| [`skill/nki-megakernel-optimizer/`](skill/nki-megakernel-optimizer/SKILL.md) | **the method as a Claude Code skill**: phases, the bounds-to-action table, gates, and [`references/pitfalls.md`](skill/nki-megakernel-optimizer/references/pitfalls.md) (every trap we hit, by symptom). Copy it into `.claude/skills/` |
| [`megakernel/`](megakernel/README.md) | the Qwen3-8B one-launch decode step: kernels, the checker, tests, benchmarks, profiling tools, results |
| [`megakernel/check.py`](megakernel/check.py), [`CHECKER.md`](megakernel/CHECKER.md) | the checker: layout, tiny, head, real-weights (teacher-forced, 3σ rule), speed; `--control` injects known bugs |
| [`megakernel/ATTEMPTS.md`](megakernel/ATTEMPTS.md) | all 18 attempts with correctness and speed scores, including the failures |
| [`megakernel/NOTE.md`](megakernel/NOTE.md) | one-page note: what ran, on what, what came out, how many runs, the spread |
| [`megakernel/symnki_poc/`](megakernel/symnki_poc/README.md) | proof of concept: our LM head checked with [nkiequiv](https://github.com/forthoney/symnki), a symbolic equivalence checker for NKI |
| [`examples/lm-head-walkthrough/`](examples/lm-head-walkthrough/README.md) | **worked example with all three tools on one piece**: the LM head from nkilib's kernel (2.71 ms) to our own (2.23 ms), with the corpus queries and nkiequiv proofs run live and the device result at each step, including the bug only the device caught |
| [`heuristics/`](heuristics/NKI_HEURISTICS.md) | 724 NKI rules (docs, nki-samples, AWS agent skills), each with trigger, fix and source; `heuristics.py` gives the agent a card, retrieval and error-to-rule lookup |
| [`docs/summary.html`](docs/summary.html) | one-page visual summary |

## The loop

1. **Floor first.** Bytes per token ÷ bandwidth. Every number afterwards is a fraction of it.
2. **Reference and layout gate.** A float32 reference in the kernel's layouts, matched to the framework
   model to 1e-5 before any kernel runs; the same math in bf16 through `torch.compile` is the baseline.
3. **Build on library blocks** (`nkilib` attention, MLP, projections, norms); write only the glue.
4. **Gate every change**, cheapest first. When the checker rejects, it says which convention the kernel
   actually implements, or which step disagrees and whether bf16 noise explains it.
5. **Profile and compute bounds** (`neuron-explorer` → `analyze_profile.py`: memory, compute and pipeline
   bounds, plus DMA-idle time attributed to source lines). Change the one thing the bounds point at.
6. **Measure the real loop**, not just the device.
7. **Report with spread.**

What this caught on Qwen3-8B, in order: a mask-convention bug that made the kernel 52% wrong while looking
fine (found by fitting against deliberately wrong references); the documented next optimization being
1% of the time and 10% slower in practice (the profile pointed at the MLP dataflow instead: −5.8%); and a
6 ms-per-token host gap that turned out to be stale output tensors, not uploads.

## Two checkers, two kinds of bug

| | on the device (`check.py`, `kernels/test_head.py`) | [nkiequiv](https://github.com/forthoney/symnki) (symbolic) |
|---|---|---|
| runs on | Trainium2 | CPU, no device |
| cost per verdict | 41–48 s for the LM head; 142 s for 36 real layers | **0.3 s** at small tiles; 22 s at full Trainium2 tiles |
| covers | the whole step, real weights, real hardware | self-contained pieces, all inputs (proof over the reals) |
| caught | the mask bug (`check.py`); a PSUM interleaving fault that is a hardware/compiler behaviour (`test_head.py`) | a skipped k-tile, with the missing term named |
| misses | bugs on inputs it didn't sample | anything outside its NKI model (indirect DMA, the PSUM fault above) |

Use nkiequiv as the inner, cheap gate on rewrites of a piece; send survivors to the device gates.

## Running it

Needs a Trainium2 instance with Neuron 2.34+, NKI 0.6, `nkilib` and `libtorch-neuronx-lite`
(`torch.compile(backend="neuron_libtorch")`), and Qwen3-8B in the HuggingFace cache. On a shared machine,
pick a free logical core with `NEURON_RT_VISIBLE_CORES`.

```bash
cd megakernel
NEURON_RT_VISIBLE_CORES=2 python check.py --resident                                   # all five gates
NEURON_RT_VISIBLE_CORES=2 python kernels/generate.py --layers 36 --new 32 --resident   # generate and compare with HF
NEURON_RT_VISIBLE_CORES=2 bash kernels/run_spread.sh                                   # every run in NOTE.md
NEURON_RT_VISIBLE_CORES=2 python kernels/profile_mega.py --layers 1 --tag mine && python kernels/analyze_profile.py mine
```

## Status and open items

* Final kernel: `megakernel/kernels/qwen3_decode_step.py::qwen3_decode_step_resident`.
* Our own LM head is correct and 18% faster on its own (2.71 → 2.23 ms), but faults with an out-of-bounds
  indirect DMA inside the generate loop (ATTEMPTS #18): about 0.5 ms per token is waiting there.
* A persistent K-tokens-per-launch kernel hits a neuronx-cc internal error (ATTEMPTS #15).
* Single logical core; tensor parallelism across cores is the route below the 20.8 ms single-core floor.
* Not measured: a like-for-like vLLM comparison (the 71.3 ms is one TP=2 run over HTTP, with no log in this
  repo); how the step scales with context length (device timing at 1,024 tokens of context, generation at
  256); batch sizes above 1.

## Credits

Built at Hack the Chip on a Trainium2 workshop seat. The kernel builds on AWS's NKI Library (`nkilib`).
The symbolic checker is [forthoney/symnki](https://github.com/forthoney/symnki), from our teammate. The
profiling method follows AWS's `neuron-nki-profile-querying` bounds analysis.
