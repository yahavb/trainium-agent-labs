# CHIPBOOST: one-page note

*Hack the Chip, NYU × Annapurna Labs, Oct 10 2026. Numbers as of 16:45; items marked **(pending)** are
filled in by 17:45. Every speedup is against the start kernel timed in the same session, on the chip,
unless labelled [sim]. Nothing here is projected.*

**Question.** Can Qwen3-8B, served on one Trainium2 chip, make the matmul it is built from faster on that
same chip, under a referee strict enough that no speedup can be faked?

## What ran where

| Seat | Owner | Ran | Runs × budget |
|---|---|---|---|
| 100 | P1 | Pinned three-arm comparison on one referee process (core 3); referee-side v2 (core 2) | 3 × 8 per arm; v2: 3 runs **(pending)** |
| 101 | P3 | Model arms on its own loop; agent-side v2 | 3 × 8 per arm; v2: 3 × 8 |
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
K-block's partial sum to bf16; its own test (K=1024, one block) never takes that path. [chip] 5.3 bf16
ulps at K=2048 (limit 4), and wrong on the held-out down_proj shape. [sim] At K=8192, the K of its own
benchmark, 19.4 ulps: it fails its own correctness check (`tools/aws_matmul_bf16_repro.py --sim`, the
file unmodified). A one-line fp32 accumulator fixes it (0.5 ulps) at no measurable speed cost. Its fixed
blocking also asserts M % 2048 == 0, so it refuses Qwen3's 256-token shapes outright.

**3. Tuning the expert's block sizes gains +35%, but only at the shape it was tuned on** [chip]. The
fixed expert is 2.49× the start kernel: AWS's design, not a discovery. Random search over its three
block sizes starts from it (a different claim from finding 4): 3 runs × 24 tries reach 1.325×, 1.352×,
1.372× the expert. An exhaustive sweep of all 62 legal settings ranks AWS's default #29, caps the gain at
1.375×, and ranks the runs' bests #4, #3, #1. The winner stays correct on 6 of 6 held-out shapes, but its
speed does not transfer: +84% at 128 tokens, −61% at kv_proj with 1024 tokens. Tune per shape.

**4. The model needs feedback that names the change.**
- *v1, 0 faster in 96 attempts.* Model alone: 46 of 48 "no gain"; it rewrites the kernel without
  changing its speed. Model + referee: 45 of 48 wrong, two mistakes repeated: a `dma_copy` source 4× its
  tile (21) and a PSUM accumulator hoisted out of the output-tile loops (24). Root cause: after a crash
  the model was told "fix the error named in the referee message" — a message it is never shown.
- *v2, P1, referee-side:* instructions that name the change (one `nl.ndarray` of shape
  (TILE_K, K // TILE_K, TILE_N) instead of a Python list of tiles; one fresh accumulator per output tile).
  First run: **a correct kernel 1.517× faster on the third attempt** (633.0 vs 960.4 µs), correct on 5
  held-out shapes; re-timed twice at 1.517×. The code is verbatim from the model's reply
  (`kernels/qwen_v2_best.py`, sha1 57044ec26245); no prompt contained a solution. Held-out grid for
  this kernel: **(pending)**. Later runs used a revised treatment ("Qwen + P1 fixes") and are reported
  separately: **(pending)**.
- *v2, P3, agent-side:* instructions that name the *error* ("your dma_copy moves 65536 elements into
  16384") and offer two generic fixes: 0 faster in 18 attempts, the same crash 15 times.

So far one success in 28 v2 attempts: a first result, not yet a rate.

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
`experiments/`. Red team: `results_p1.json`, `redteam/`. Dashboard: `python dashboard/collect.py`
builds `dashboard/index.html` from all of them.
