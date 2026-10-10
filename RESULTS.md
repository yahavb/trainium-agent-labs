# Results and ablations: what each change did

Team 24, project 2 (NKI kernel agent), 2026-10-10. This page is the short, readable version: which change
moved which number. The full write-up, with method, checker and taxonomy, is [SUBMISSION.md](SUBMISSION.md).

**Numbers as of 17:20.** The final run replaces the last column: [[TBD: final-run numbers]]. Level 2's final is
v8.3 if it holds at 17:45; on levels 1, 3 and 4 v8.3 sends byte-identical requests to v8.2, so it is the same agent there.

## The short version

The model never changed: Qwen3-8B, thinking off, served on the seat's Trainium2, 8,192 tokens of context.
Only what the agent showed it changed. Same model, same budget (8 rounds × 4 samples per run):

| level | operation | organizers' agent (baseline) | our v7 | our final, v8.2 |
|---|---|---|---|---|
| 1 | average pooling | 0/5 | 0/1 (best score 0.50) | **5/5** |
| 2 | 2-D transpose | 3/5 (a re-run: 2/5) | 0/3 | **5/9** (v8.3, in test: 5/6) |
| 3 | matmul, one tile | 0/5 | **5/5** | **5/5** |
| 4 | matmul, tiled | 0/5 | **5/5** | **5/5** (all five runs are one trajectory) |
| 5–7 | tiled matmul under an HBM traffic budget | not run | not run | [[TBD: WARM runs, 17:10–18:05]] |

Each cell is *runs solved / runs*. What did it, one line per level:

- **Level 3:** one worked example in the first prompt. Take it out and level 3 is not solved (0.30).
- **Level 4:** repair messages that give the fix as code. Hollow that code out and level 4 drops from 5/5 to 0/2.
- **Level 1:** solved by v8.2, whose main addition is fresh first attempts in every round. We have not
  separated which of v8.2's parts did it on level 1. What we did measure: the compiler gate's rewrite is what
  makes level-1 solves build for the real chip (0 → 4 of 10 runs, row 6).
- **Level 2:** v7's sampling settings had broken it (round-0 solves 2/20 → 0/20); mixing the original
  settings back in and adding fresh attempts brought it back to the baseline's level.

## How to read the numbers

- **Run:** the agent working on one level, up to 8 rounds. **Round:** 4 kernels (samples) from the model,
  each graded; the best one is repaired in the next round. Rounds count from 0: round 0 is the first try.
- **Solved:** some kernel scores 1.0, correct on every test shape in the NKI 0.6.0 simulator, and passes
  again when re-graded in a fresh process. The final version's solves also passed our held-out set (new
  shapes and hostile values).
- **"Round r":** the round of the first 1.0. Lower is better: round 2 means 3 rounds, 12 kernels.
- **Distinct trajectories:** the seat's model server returns the same text for the same request, so
  "5 runs" can be one run replayed five times. Where it matters we say how many runs actually differed.
- **Simulator vs chip:** unless a line says *full build* or *chip*, numbers come from the simulator.

## Every change, in order

Before → after is the number on the level the change was tested on. "Before" is the version the change
was built on: the baseline up to row 4, v7 from row 5 on (v8.2 for row 11). Each change was kept or dropped by rules written down before its results came in ([PLAN.md](PLAN.md)).

| # | change, in plain words | level | before | after | kept? |
|---|---|---|---|---|---|
| 0 | **Baseline:** the organizers' `agent.py` as given | 1 / 2 / 3 / 4 | | 0/5 · 3/5 · 0/5 · 0/5 (a re-run on another seat: 0/5 · 2/5 · 0/5 · 0/5) | reference |
| 1 | **E-A:** when the model calls an NKI function that doesn't exist, name the real one | 1 | 0/5 | 0/5; "invented function" errors 80 → 20, so the failures moved one layer deeper | kept as groundwork |
| 2 | **E-F:** for a wrong argument list, show the failing line and the real signature | 1 | 0/5 | 0/5; wrong-argument errors 8 → 3 per run | dropped |
| 3 | **v3:** repair messages that give the fix as code to paste | 4 | 0/5 | 1/2 (one late solve) | grew into v7 |
| 4 | **v7:** the full stack: an exact first prompt with real NKI calls, a worked example chosen by the operation's category, repair messages as code, the whole matmul tiling in one message, a prompt that allows restructuring, Qwen's sampling settings, a compiler gate | 3 | 0/5 | **5/5**, round 0 | **adopted** |
| | | 4 | 0/5 | **5/5**, round 2 (one trajectory) | |
| | | 1 | 0/5 | 0/1 (best 0.50) | |
| | | 2 | 3/5 | **0/3**: a regression, explained in row 8 | |
| 5 | **E-div:** tag samples 2–4 with one line each, so the server stops returning four identical answers | 3 / 4 | 5/5 · 5/5 | 5/5 · 5/5; solves unchanged, but level 4's five runs are now 5 distinct trajectories instead of 1 | kept |
| 6 | **Level-1 rule in the compiler gate:** a kernel that is correct but reduces one channel at a time (which the real compiler rejects) is held at 0.95 and given its own loop rewritten for all channels at once | 1 | 0/10 runs with a kernel that **builds** for trn2 | **4/10** (tested with the same model on a GPU; 0 of 30 such runs without the rule) | kept |
| 7 | **Skeleton feedback:** keep the code in repair messages but blank out slices and shapes, so the model must work them out | 4 | 5/5 | **0/2** (stuck at 0.62) | dropped |
| 8 | **Sampling check on level 2:** 20 first-round samples each | 2 | original settings: 2/20 | v7's settings: **0/20** | led to row 9 |
| 9 | **v8.1:** samples 1 and 3 use the original sampling, 2 and 4 use v7's; one sentence restating the level-2 task | 2 | 0/3 | the first level-2 solve in a repair round (1 run, then replaced by v8.2) | folded into v8.2 |
| 10 | **v8.2 (final):** in every repair round, sample 1 repairs the best kernel and samples 2–4 start over from the first prompt (about 27 fresh attempts per run instead of 4) | 1 | 0/1 | **5/5**, rounds 2, 2, 0, 4, 2; five different kernels | **final** |
| | | 2 | 0/3 | **5/9** | |
| | | 3 | 5/5 | 5/5, rounds 0, 2, 0, 0, 0 | |
| | | 4 | 5/5 | 5/5, all round 2 (one trajectory, the same kernel as v7) | |
| 11 | **v8.3:** on level 2, when a kernel runs but its numbers are wrong, say what it actually computed ("your output at row i*B+j holds x[i]") instead of only that it is wrong; levels 1, 3, 4 unchanged (byte-identical requests) | 2 | 5/9 | 5/6 so far | [[TBD: decided at 17:45]] |
| 12 | **WARM (levels 5–7 only):** levels 5–7 are level 4's matmul with a cap on HBM traffic; round 0 starts from the agent's own level-4 solve and the checker's verdict on it for the new level, instead of the first prompt. No reference kernel is shown | 5–7 | not run | [[TBD]] | [[TBD]] |

## Leave-one-out: which part of v7 did the work

Each row puts one v7 part back to the organizers' original and keeps everything else. One run per cell (the
server is deterministic, so one run is the trajectory). The number is the round of the first 1.0.

| part put back to the original | level 3 | level 4 |
|---|---|---|
| nothing (v7 as is) | round 0 | round 2 |
| first prompt | round 0 | round 1 (one round faster) |
| **worked example** | **not solved (0.30)** | round 4 (two rounds slower) |
| whole-tiling message for matmul | round 0 | round 4 (two rounds slower) |
| repair prompt | round 0 | round 2 |
| sampling settings | round 0 | round 2 (identical first rounds) |

Level 3 depends on exactly one part, the worked example. Level 4 depends on no single part: the example and
the tiling message each save two rounds, and v7's longer first prompt costs one. But level 4 does depend on
the repair code as a whole: row 7 above shows what happens when the code in the messages is blanked.

## Beyond the simulator: does it build, does it run on the chip?

The simulator accepts some kernels that the real trn2 compiler rejects, so we also checked with the full
compiler build (`check/compile_solves7.py`) and, for a set of kernels, on a NeuronCore of seat-115.

- **Level 1, before the rule (row 6):** every simulator solve reduced one channel at a time, and **none of
  them builds** for trn2.
- **The final agent:** every v8.2 and v8.3 solve on levels 1–4 builds in full for trn2 and matches in birsim,
  the compiler's instruction-level simulator: level 1 on two shapes, level 2 on four, level 3 on its one
  shape, level 4 on two. None was rejected.
- **Correct is not the same as fast:** the compiler's estimate shows one family of level-2 solves writing HBM
  one element at a time, predicted at 44–169 µs against about 2.6 µs for the others.
- **On the chip:** 63 kernel-shapes run on the NeuronCore, organizers' references included. The full build
  predicted every outcome: every kernel that builds ran correctly, every kernel that doesn't build failed,
  and the simulator alone would have passed all of them. Our level 1–4 candidates all passed on the chip
  ([analysis/seat115_chip_and_l1rule.md](analysis/seat115_chip_and_l1rule.md)).

## What hurt, kept because each one cost us time

- **Skeleton feedback (row 7):** level 4 went from 5/5 to 0/2. The model filled the blanks with wrong
  slices and stayed out of bounds for six rounds. The code in our messages is part of the solution.
- **v7's sampling settings (row 8):** level 2 went from 3/5 to 0/3. Fixed by mixing (rows 9–10).
- **E-F (row 2):** fewer wrong-argument errors, no solves; dropped.

## How far to trust these numbers

- **Small samples:** 1 to 10 runs per cell. Read 5/5 vs 0/5 as a real difference; 5/9 vs 3/5 as no difference.
- **Runs are not independent:** with a deterministic server, level 4's five v8.2 runs are one trajectory,
  one piece of evidence rather than five. The distinct-trajectory counts are in NOTES §0.
- **Different seats, same setup:** the runs are spread over seats 116–119, each with the same model and settings.
- **Row 6** was measured with the same model served on a GPU, because the seats were busy; its kernels
  were then built for trn2 and run on the chip.
- **Rounds count from 0 everywhere on this page.** NOTES §0 counts some of the older rows from 1.

## Where each number comes from

| what | source |
|---|---|
| baseline, E-A, E-F, v3, v7, E-div, v8, sampling check, v8.2 | [NOTES.md](NOTES.md) §0 (the experiment table), [SUBMISSION.md](SUBMISSION.md) §4 |
| the short-version table | [SUBMISSION.md](SUBMISSION.md) §1 |
| full builds of the final solves | [SUBMISSION.md](SUBMISSION.md) §5 and §10 |
| leave-one-out | [SUBMISSION.md](SUBMISSION.md) §3 |
| level-1 rule, full builds, chip | [projects/02-kernel-agent/V7.md](projects/02-kernel-agent/V7.md) "The level-1 fix", [analysis/seat115_chip_and_l1rule.md](analysis/seat115_chip_and_l1rule.md) |
| skeleton | [analysis/taxonomy_versions.md](analysis/taxonomy_versions.md) |
| v8.3 | [SUBMISSION.md](SUBMISSION.md) §4, [analysis/v83_l134_identity.md](analysis/v83_l134_identity.md) |
| how to run each version | [V7.md](projects/02-kernel-agent/V7.md), [V8.md](projects/02-kernel-agent/V8.md) |
