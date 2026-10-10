# The agent was stuck on the checker's words, not on tiling

**Team:** John (measurement, write-up), Achyuthan (checker), Perumal (agent and prompt).
Full evidence: `report/01-baseline-and-taxonomy.md` (baseline, taxonomy, hypotheses),
`report/03-run-sheet.md` (run sheet), `report/04-one-page-note.md` (long note), `results/RUNS.tsv` (every run,
seat, commit, flags).

## Claim

The organisers' agent sat at fixed scores: 0.30 on level 1, 0.30 on level 3, 0.62 on level 4. Their reading
was that the model does not understand tiling. On level 3 it was something else: **the checker described the
simulator's crash instead of the kernel's mistake, and twice told the model to do the wrong thing.** Changing
only what the checker says, never the answer, took level 3 from 0 of 15 runs to **16 of 17 completed runs**,
on four chips, every solve within the chip's rules. Level 4 did not truly move, level 1 stayed at 0.30 even
with the full recipe in the prompt, and nothing transferred to an untuned level (attention).

## What we ran

Qwen3-8B on each seat's server, CPU simulator (nki 0.6.0). Every scored run used the same settings (8 rounds,
4 samples, context 8192, 5 repeats); only the checker flags differ. Baseline: 15 runs on 3 seats, including
the organisers' untouched code. Levels 1, 3 and 4 never varied across them.

| Level | Baseline | Checker (`internal,origin`) | Checker + `ahead` |
|---|---|---|---|
| 1 avg pooling | 0/15 (0.30) | 0/5 (0.30) | not run |
| 2 transpose | 6/15 | 2/5 | 3/5 |
| 3 matmul, one tile | **0/15** (0.30) | **16/17**: 6/7, then 10/10 in a second batch | **10/10**, faster (4 rounds, not 6) |
| 4 matmul, tiled | 0.62 | 0.75 as scored; **0.62 with the 128-row rule enforced** | not adopted |
| 8 attention (untuned) | 0/5 (0.30) | 0/5 (0.30) | not run |

Level 3, both batches pooled because the level 3 feedback is identical between their commits on all 93 logged
kernels: seats 153 and 157 (6 of 7), then seats 150 and 158 (10 of 10, every solve in round 6). All 16 solves
keep every tile within 128 rows and return from HBM. An eighth early run stopped at round 3 with no result;
counted as unsolved, the figure is 16 of 18.

A two-sentence prompt hint on level 1 ("Average pooling is a per-window sum followed by a scale; it needs no
matrix multiplication. Use a strided view of pooling windows and nl.sum for the reduction.") scored 0/5 but **fixed the algorithm**: matrix multiply in 0 of 104 attempts (baseline 160 of 160), `nl.sum`
in 103 (the harness on that seat was not verified for this run; the finding is in the code, not the score). Given the full recipe instead (strided `.ap()` view, `nl.sum`, scale by `1/(p*p)`), it followed it in
100 of 100 attempts and still scored 0/5: 80 attempts failed copying the input into a fixed `(C, 128, 128)`
tile, the same tile-sizing error as level 3, answered with the same misleading `128x512` example. So level 1
is blocked by the error level 3's checker fixes, not only by the algorithm. With the checker added to the
recipe (16 rounds): still 0.30 in all 5 runs.

## What made level 3 move

| Checker flags | What the model is also told | Level 3 |
|---|---|---|
| `locate` | the failing line | 0/4 |
| `state` | shapes and buffers on that line, and the rule broken | 0/5 |
| `facts,origin` | the line that gave the wrong tile its shape; rule replaces all advice | 0/5 |
| `internal,origin` | the same, but advice dropped only for numpy errors inside the simulator | **4/5** |

Naming the failing line was not enough (0/4). What worked is `origin`: it names the wrong tile, which dimension
is off and which way, which operand it must match, and the line that allocated it; `internal` drops the old
advice for numpy errors from inside the simulator. The failures moved where that information pointed: the
baseline never got past loading the inputs (0 of 84 failed attempts); under the checker 64% got to the matmul
or later. **Demo:** `results/level3-recovery-transcript.md`, one run, every message verbatim. Round 0 still
carries the old "(1, N)" hint (the 1-D error is one of the simulator's own checks, so `internal` keeps it), and
the model follows it. From round 1 `origin`'s messages take the result tile from (64, 1) to (128, 512) to
(64, 512), the right shape, in two rounds, each step following the message; solved in round 5. `ahead`
replaces that hint with the tile's role (the result of the matmul on line 18, and what its two dimensions must
be): the hint appears in 12 of 12 one-dimension errors under the checker and 0 of 24 under `ahead`, which
removes the detour, and its solves take 4 rounds instead of 6.

## Holes in the scoring, reproduced on the real simulator

Six repro kernels in `../findings/` (that is, `projects/02-kernel-agent/findings/`), each run through the organisers' own
`nkibench.py --check` (output: `results/seat-150/findings.out`). Two pass that shouldn't: a level 4 kernel
with 256- and 512-row SBUF tiles (4 of 4 shapes), and a level 3 kernel returning its PSUM tile (1 of 1). The
rule scan catches a tile written `(256, M)` but not `(K, M)`. Two messages mislead: "cannot reshape" for a
kernel with no reshape, and a moving-operand limit when both operands are within limits. And the agent's
reused file path served a stale kernel's bytecode. Any level 4 solve should be re-checked with
`--features strict`. Not fixed, a finding only: about two thirds of all "`nki.isa` has no attribute" failures
are `multiply`, and the agent's hint says nothing similar exists; it is `nl.multiply`.

## Why it was stuck (taxonomy of all 416 baseline attempts)

- **Level 3:** the result tile given one dimension (`lhsT.shape[1:]`, 62 of 84). The checker's hint said "give
  a vector the shape (1, N)"; the model obeyed, and got `cannot reshape` errors in kernels that contain no
  reshape (0 of 16), then was told "Do not reshape". Another hint's example tile, (128, 512), was the exact
  wrong tile the model had made.
- **Level 4:** one kernel in 69 of 80 attempts, never tiling, resent unchanged in 60 of 60 repairs.
- **Level 1:** wrong algorithm in 160 of 160: each window matrix-multiplied by itself, times 0.5.

## Caveats, all measured

- **Level 4's 0.75 is a loophole.** 22 of 23 kernels scoring 0.75 allocate tiles of 256 or 512 rows and pass
  128-row slices; the harness checks only the slice. With the rule enforced: 0.62 in 5 of 6 runs, 0.50 in 1.
  What moved is the failure (the model now tiles K), not the score.
- **The model is near-deterministic:** five baseline runs gave identical kernels. So we credit a change only
  when the failure type also moves toward what the new message said.
- **Every flag set is reported**, including the losers: `pieces`, `pieces,shapes`, fixed `ahead`, `names`
  (level 3: 2 chip-valid solves in 8 runs, worse than the checker; level 4: 0.62 with the rule enforced), and `example` (a worked tiling example in the repair prompt, 16 rounds, with the 128-row rule
  enforced): 2 of 14 runs reached a chip-valid 0.75 on level 4 (K tiled correctly, then failing at M = 256).
  A flag is adopted only if it is better where it aims and no worse elsewhere.
- **Simulator only.** It also accepts a result returned from PSUM; we checked every solve, and all checker
  and first-`ahead` solves return from HBM.
- **Pooling across commits** only where a replay of every logged kernel gives identical feedback.
- **The bytecode bug is fixed** in the agent; no logged score changed. The harness is the organisers' copy,
  byte for byte.

## Conclusion

One wall fell cleanly. Level 3 went from 0 of 15 to 16 of 17 completed runs on four chips, every solve within
the chip's rules, by changing only what the checker says. Level 4's kernels learned to tile, but their higher score
depends on a tile the harness fails to reject, so honestly it is still 0.62. Level 1 needs the algorithm and then
the same tile fix as level 3: a two-sentence hint fixed the method, and the full recipe was
followed exactly, but both stalled on the tile-sizing error, and adding the checker did not get past it.
Nothing transferred to attention, which nobody tuned. The largest lesson: the existing
feedback was not just vague but misleading, and what fixed it was telling the model which tile is wrong, in
which dimension, which way, and where it was allocated; naming the failing line alone did nothing.
