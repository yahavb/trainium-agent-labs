# CHIPBOOST: one-page note

*Hack the Chip, NYU × Annapurna Labs, Oct 10 2026. Final numbers, 17:40: 264 logged attempts from three
seats. Every speedup is against the start kernel timed in the same session, on the chip, unless labelled
[sim]. Nothing here is projected.*

**Question.** Can Qwen3-8B, served on one Trainium2 chip, make the matmul it is built from faster on that
same chip, under a referee strict enough that no speedup can be faked?

## What ran where

| Seat | Owner | Ran | Runs × budget |
|---|---|---|---|
| 100 | P1 | Pinned three-arm comparison on one referee process (core 3); multi-change feedback (core 2) | 3 × 8 per arm; 1 + 2 revised runs × 8 |
| 101 | P3 | Model arms on its own loop; named-error rules (v2); P1's instructions alone (v3); rules A–C (v4), A–D (v5) on them | 3 × 8 per arm; v2 3 × 8; v3, v4, v5 1 × 8 |
| 102 | P2 | Random search over the expert's block sizes; exhaustive sweep of all 62 settings | 3 × 24; 62 |

Shapes: Qwen3-8B per-core matmuls at tensor parallelism 2, 256 tokens (gate_up 4096×256×6144 plus q_proj
4096×256×2048; the score is their summed time). Start kernel: the NKI tutorial's tiled matmul, 960 µs.

## Findings

**1. The referee can be trusted** (`speedcheck.py`). Rules, then simulator correctness, then chip
correctness with hostile inputs, then interleaved A/B timing on the device clock, then three random
held-out shapes for any would-be "faster", then one instruction. Candidates run in a sandboxed child
process. Timer [chip]: 45 checks of the start kernel against itself never reported "faster" (spread
0.011%); against a deliberately slowed copy it measured 2.949× in all 18 runs (spread 0.04%); vLLM load
moved timings ≤ 0.04%; time scales linearly with work (R² = 0.9999998). Red team: **34 of 36 planted cheats caught, 9 of 9 honest kernels accepted**. Of the
two not caught, one gained no speedup and one (lower-precision accumulation) passed at 1.002×, also no
gain. Earlier rounds found and closed real escapes: a forged result record, a candidate patching the
referee, a shell from inside the kernel.

**2. AWS's published "fully optimised" NKI matmul loses precision at Qwen3's sizes.** It rounds each
K-block's partial sum to bf16; its own test (K=1024, one block) never takes that path. [sim] 5.3 bf16
ulps at K=2048 (limit 4), in the referee's simulator stage; at K=8192, the K of its own benchmark, 19.4
ulps: it fails its own correctness check (`tools/aws_matmul_bf16_repro.py --sim`, the file unmodified).
[chip] Wrong on the held-out down_proj shape (K=6144): 4.9 ulps. A one-line fp32 accumulator fixes it
(0.5 ulps) at no measurable speed cost. Its fixed blocking also asserts M % 2048 == 0, so it refuses
Qwen3's 256-token shapes outright.

**3. Tuning the expert's block sizes gains +35%, but only at the shape it was tuned on** [chip]. The
fixed expert is 2.49× the start kernel: AWS's design, not a discovery. Random search over its three
block sizes starts from it (a different claim from finding 4): 3 runs × 24 tries reach 1.325×, 1.352×,
1.372× the expert. An exhaustive sweep of all 62 legal settings ranks AWS's default #29, caps the gain at
1.375×, and ranks the runs' bests #4, #3, #1. The winner stays correct on 6 of 6 held-out shapes, but its
speed does not transfer: +84% at 128 tokens, −61% at kv_proj with 1024 tokens. Tune per shape.

**4. Feedback decides whether the model gets anywhere.**
- *Original feedback, 0 faster in 96 attempts (12 runs).* Model alone: 46 of 48 "no gain"; it rewrites the
  kernel without changing its speed. Model + referee: 45 of 48 wrong, two mistakes repeated: a `dma_copy`
  source 4× its tile (21) and a PSUM accumulator hoisted out of the output-tile loops (24). Root cause:
  after a crash the model was told "fix the error named in the referee message", a message it is never
  shown.
- *P1's multi-change feedback, 2 of 3 runs reached a correct kernel 1.517× faster* (on attempts 3 and 2).
  Instructions that name the change (one `nl.ndarray` of shape (TILE_K, K // TILE_K, TILE_N) instead of a
  Python list of tiles; one fresh accumulator per output tile), together with other agent changes: P1's
  runs label it an exploratory multi-change treatment. The second success is different code (sha1
  05be45c6c037): the model rediscovered the design rather than repeating an output. The first kernel
  (`kernels/qwen_v2_best.py`, sha1 57044ec26245, verbatim from the model's reply; no prompt contained a
  solution) re-timed twice at 1.517× and is **correct on 6 of 6 held-out shapes, faster than start on all
  6 (1.14–1.91×, geometric mean 1.52×)**; see `results_heldout_matmul.json`.
- *P3's agent, one rule at a time, reproduced it.* Naming the error (v2, 3 runs): 0 faster in 24 attempts,
  the same `dma_copy` crash 21 times. P1's instructions alone (v3): 8 of 8 crashed. Rules A–C on top
  (v4): 8 of 8 wrong, now wrong numbers rather than crashes. Rule D on top (v5): **1.517× on attempt 2**
  and again on attempt 3, from the start kernel, correct on 3 held-out shapes, no solution code in any
  prompt. Its attempt-2 code is byte-identical to P1's second success (sha1 05be45c6c037).

So Qwen3-8B reached the same 1.517× from the start kernel under two different agents (P1's: 2 of 3 runs;
P3's v5: 1 of 1), each time once the feedback named the change and the bugs it kept making. The earlier
feedback went 0 of 17 runs (the original 12, P3's v2 to v4). Which ingredient matters most is open. A
first result, not yet a rate.

## Also measured, not a referee verdict
Half of every NeuronCore sits idle under a plain launch: running the expert across both physical cores of
an LNC=2 core (`kernel[2]`) gives 1.50× more, 5.0× the start kernel, on two clocks. The referee launches
at LNC=1 and did not check held-out shapes for it, so it is not in any comparison above.

## Not claimed
No end-to-end Qwen3 speedup: no kernel was plugged into the served model. No share-of-runtime (Amdahl)
figure: the served model was not profiled. RMSNorm: start (518 µs) and copy floor (141 µs) measured at
2048 tokens; no arm ran on it.

## Deliverables
Checker: `speedcheck.py` and `REFEREE.md` (P1). Attempt logs: every seat's `attempts*.jsonl`, P1's
`experiments/`. Red team: `results_p1_throughput.json`, `redteam/`. Dashboard: `dashboard/index.html`
(everything) and `dashboard/results.html` (the summary; `dashboard/screenshots/`), built by
`python dashboard/collect.py` from all of them.
