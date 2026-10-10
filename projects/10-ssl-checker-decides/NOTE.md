# The checker decides: what we ran on Trainium today

**Team:** Rishabh + teammates, seats 45, 48 and 49 (NYU × Annapurna Labs, Hack the Chip, 2026-10-10)
**Model:** Qwen3-8B on each seat's own Trainium chip (vLLM, TP=2). All grading in `nki.simulate` on CPU.
**Project:** 2, the kernel agent. Target: the linear layer `relu(W·x + b)`, starting with its matmul
half (level 4, tiled matmul).

## Headline results

| What we changed | Before | After | Runs |
|---|---|---|---|
| **Level 3 in the normal `agent.py` loop: step list + a fix message for PSUM→HBM DMA, written with the kernel's own names** | 0.30 every run (0/10 across 3 setups) | **2/2 solved**, stock `nkibench --level 3` clean, `stress.py` 28/28 | 2 (identical) |
| **Level 4 end to end: Qwen plans the tiling (step-list wording, honest feedback) and writes the body; the tool only expands the plan into loops** | 0/4 (0.62) | **5/5 solved**, stock `nkibench --level 4` 4/4 every run, `stress.py` 50/50 | 5 (identical) |
| Level 4: plan given, loop tool writes the loops, **explicit step list, one hole per prompt, freeze what passes** | 0/5 (0.62 baseline) | **5/5 solved**, stock checker 4/4, `stress.py` 50 hostile cases | 5 (identical, greedy) |
| Ablation of that result | loose wording: 0/5 staged or not | step list unstaged 2/5, staged 5/5 | 5 each |
| Speed up a correct matmul: feedback names ONE counted change ("64 nc_matmul where 16 would do; moving= is 128 wide, the limit is 512") | 330 µs | **109 µs (3.02×)**, beats the tutorial's 118 µs | 3 (identical) |
| Same, feedback = latency numbers only (control) | 330 µs | 321 µs (1.03×) | 3 |
| Same, generic roofline diagnosis | 330 µs | unchanged: 24/24 returned the kernel as-is | 1 |
| Error messages that map invented names to the real call (level 1) | invented `nisa.multiply`/`scalar_mul` in 80/160 attempts (50%) | 12/96 (12%); score unchanged at 0.30 | 2.75 |

## Final scoreboard: the 8-level ladder (graded by the stock `nkibench.py`)

| Level | Start of day | End of day | How |
|---|---|---|---|
| 1, average pooling | 0.30, 0/4 | 0.30, 0/4 more | not moved: stuck on `ap()` stride arithmetic |
| 2, transpose | solved 2/4 (noisy) | best unchanged (step list hurt: 0/5) | not moved |
| **3, matmul, single tile** | **0.30, 0/4** | **solved 2/2** | normal `agent.py` loop: step list + a named PSUM→SBUF fix |
| **4, matmul, tiled** | **0.62, 0/4** | **solved 5/5** | Qwen plans (step wording) + writes the body; the tool expands the plan into loops |
| 5, loads hoisted | not attempted | 0.88 (3/4 shapes; 2.00x bytes vs 1.60x) | the 109 µs kernel from the speed-up loop |
| 6, 7, blocked | not attempted | 0.75 | same kernel |
| **8, attention** | 0.30, 0/2 (original agent) | **solved 2/2** | staged: scores, softmax, P·V, each frozen once it passes; Qwen wrote the body. The stock level-8 grader crashes (KeyError 'M') on any correct kernel; graded with only those keys added |

**Three levels moved from unsolved to solved (3, 4 and 8).**

## What we learned about checkers (each one measured)

1. **Naming one specific, counted fix works, and a true but generic diagnosis freezes the model.**
   The roofline floor said how far off the kernel was. The instruction count said what to change.
2. **Qwen applies only the LATEST fix and drops earlier ones.** Message tuning clears each targeted
   error and the model moves to the next, so it cycles. What converged was keeping what already
   passed: freeze a hole once it passes, and ask for one piece at a time.
3. **Placeholders in a hint get copied literally** (`say i`, `t = …`, a fixed `[0:128, 0:128]`).
   Hints must use the kernel's own names and slices.
4. **Planning isn't the wall we thought.** Given the correct tiling plan, the whole-kernel score went
   DOWN (0.62 → 0.30): Qwen invented `nl.tile_index`. Underneath the tiling wall is an NKI-API wall.
   Asked for the whole plan with honest feedback, Qwen passes 0/5.
5. **The reward rewarded NaN.** "Runs" credit went to kernels that never wrote output, inflating
   reported loop-tool scores (every 0.50 was one). It didn't steer the loop, which repairs the latest
   attempt, but it would mislead a reader.
6. **Measurement hygiene:** sampling is effectively greedy on these seats (91 of 108 rounds returned 4
   identical samples), so repeats are near-identical and a rate there is about one trajectory. Level 4
   gives the same score on every run, so a single moved run is a real effect.

## Last hour (filled in as runs finish)

- **H1, Qwen plans the tiling itself (seat 49): planning SOLVED with the step-list wording.** Asked for
  the whole plan with step-list wording ("the largest tile the limits allow… tile m starts at m times
  the tile size", no formulas), and honest feedback: **5/5 plans pass**, against 0/5 before. Qwen's plan
  is the correct tiling (`M // 128`, `N // 512`, `K // 128`, accumulate over k). Asking one loop at a
  time (staging) made it WORSE: 0/5, because seeing m and n frozen led Qwen to list them in
  `accumulate_over`.
- **End to end, level 4 (Qwen plans + Qwen writes the body, the tool only expands the plan into
  loops): solved.** The stock `nkibench.py --level 4`: rules clean, 4/4 shapes; `stress.py`: 50/50.
  Checked independently. **Rate: 5/5**, each run planning from scratch: 2 plan rounds, 2 body rounds,
  stock checker 4/4 every run. All 5 runs followed one trajectory (identical plans and kernels), so
  this shows it's reproducible, not robust. Driver: `projects/02-kernel-agent/e2e_rate.py`.
- **H2, the step list in the normal agent, without the loop tool (seat 48): not supported.** Level 4
  went 0.62 → **0.30** (0/2 runs; with the plan given, 0.30, 0/1). The kernels follow the step names
  but drop the m/n output loops (0/52): the step list never gives loop counts, which the loop tool's
  skeleton supplied. The top failure (16/52) was a DMA of PSUM straight to HBM, an error with no fix
  message. That message has since been added.
- **H3, a "keep these fixes" list in repair prompts, against cycling (seat 45): not supported.** L4
  [0.62, 0.50], L1 [0.30]. Recurrence didn't drop: L4 1.00 vs 1.00, L1 0.43 vs 0.43. On L4 the list
  never fired (the error never changed). On L1 it fired and `nisa.multiply` came back anyway, while
  listed as "already fixed". **A prompt-only list doesn't replace freezing in code.**

## Honest limits

- Correctness is `nki.simulate`. Speed was optimized on the compiler's predicted latency; at the end
  we measured on the chip (seat 49, server stopped): the speed-up holds, **345.8 → 118.8 µs (2.91x)
  measured**. The compiler under-predicts by about 2x but ranks kernels correctly. Attention measures
  about 20 µs per call. See `results/2026-10-10-rishabh-chip-timing.md`.
- The 5/5 is with the plan given and the loops written by a tool, not "Qwen solves level 4".
- Bias + ReLU, the linear layer's second half, isn't built.

## Where everything is

Results: `results/2026-10-10-rishabh-*.md` (one file per experiment). Logs: `logs/`. Checkers:
`projects/02-kernel-agent/{nkibench,stress,plan_check,agent,loop_tool}.py`. Roofline tool, instruction
counter and speed-up loop: `~/nyu/hackathon/hack-the-chip-team/roofline/`.
