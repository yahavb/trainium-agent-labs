# H3: a "keep these fixes" list in the repair prompt, levels 4 and 1

- **Who / seat:** Rishabh (+ Claude), seat 45, with its own vLLM server and no other jobs
- **Commit measured:** uncommitted working tree on top of 8f1ca41. md5s: `agent.py` bf14b521
  (`--keep-fixes`), `nkibench.py` 68e795d5
- **Hypothesis:** listing the errors that already went away ("Already fixed in earlier rounds -- keep
  these fixed, do not bring them back", last 5) does the job of freezing passed code in a prompt
  alone: it lowers recurrence and moves L4 above 0.62 and L1 above 0.30.
- **Change:** none by me. Ran the `--keep-fixes` flag (`_err_key`, `keep_note`, and the `keep` block
  in `solve()`).
- **Command:** `python agent.py --level N --keep-fixes --rounds 8 --samples 4 --context 8192 --repeat R`.
  **Cut for the 21:20 deadline: L4 2 runs (stopped at the start of run 3), L1 1 run.**
- **Logs:** logs/attempts-rishabh-keep-fixes-l4.jsonl (32 attempts), logs/attempts-rishabh-keep-fixes-l1.jsonl (32)
- **Comparison (same settings, no flag):** logs/attempts-rishabh-baseline.jsonl (L1 x5, L4 x5) and
  logs/attempts-rishabh-api-messages.jsonl (L1 x3, L4 x2). The api-messages run is the closer match,
  because its messages are the same as the ones agent.py now sends.

## Scores

```
no flag:     L4 0.62 x7 (5 baseline + 2 messages)   L1 0.30 x8
keep-fixes:  L4 [0.62, 0.50]                        L1 [0.30]
```

- **L4: the keep list never fired.** The top error never changed within a run, so the list stayed
  empty. Run 1 has the same prompt lengths as the no-flag runs, [2471, 1941, 2170, 2170]. Both runs
  stopped after 4 rounds on one identical failure: partition 256 > 128 in run 1, and in run 2 a 0.50
  NUMERICAL MISMATCH (8.5e4 x RMS, 100% of elements wrong).
- **Run 2's 0.50 is round-0 sampling, not the flag.** The round-0 prompt is the same with or without
  the flag. At L4, round 0 is *not* greedy: every L4 run, with or without the flag, has 3-4 distinct
  codes among its 4 samples. The 7 earlier runs all drew at least one 0.62 sample, and run 2 drew none.
  So "same score every run" holds for L4 only so far.
- **L1 is greedy.** Round 0 gives 1 distinct code, and the 5 baseline runs and 3 messages runs are each
  identical to each other. One run here is therefore informative. Rounds 0-2 are byte-identical to the
  messages run, and the runs diverge only when the keep list enters the prompt.

## Recurrence: the fraction of rounds 1..n-1 whose top error already occurred earlier in the run

The key is `_err_key(feedback)` of the round's best sample, the same key the keep list uses.
"Cycle-back" counts the rounds that return to an earlier error after a different one in between, the
failure H3 targets. "Coarse" strips numbers and quoted names.

| level | condition | runs | recurrence (exact) | cycle-back | recurrence (coarse) | cycle-back (coarse) |
|---|---|---|---|---|---|---|
| L4 | baseline | 5 | 14/14 = 1.00 | 0 | 1.00 | 0 |
| L4 | messages | 2 | 6/6 = 1.00 | 0 | 1.00 | 0 |
| L4 | keep-fixes | 2 | 6/6 = 1.00 | 0 | 1.00 | 0 |
| L1 | baseline | 5 | 15/35 = 0.43 (each run) | 0 | 0.43 | 0 |
| L1 | messages | 3 | 3/21 = 0.14 (each run) | 0 | 0.43 | 3 (1 per run) |
| L1 | keep-fixes | 1 | 3/7 = 0.43 | **1** | 0.43 | 1 |

In the whole-kernel agent, recurrence is almost all immediate repetition: the same error for 2-4
rounds, and then the give-up rule fires. True cycling back is rare, at most 1 per run. **The keep list
did not reduce recurrence. At L1 it went up, from 0.14 to 0.43 against the matching messages run.**

## What happened at L1 (top error per round, keep-fixes run)

```
r0-r1  dma_copy src/dst element-count mismatch (src=4, dst=16384)   [same as no-flag]
r2     nc_matmul(transpose_moving=)                                 [same as no-flag]
r3     dst must be in psum                     <- keep list now in the prompt (+207 chars = len(keep_note))
r4     nisa.multiply (invented)                                     [same as no-flag]
r5     stationary must be in sbuf, got psum    <- new; `tile` moved into psum alongside `tile2`
r6-r7  nisa.multiply again                     <- listed as "already fixed" in the r6 prompt
```

No flag (messages run): r5 `tensor_scalar() missing operand0`, r6 `missing data`, r7 the dma_copy
count mismatch again. Moving forward on `tensor_scalar` was the actual progress, and with the flag it
didn't happen.

**The keep list records masked errors as fixed.** The r5 code still calls `nisa.multiply`. The error
"went away" only because a new error earlier in the kernel (stationary in psum) stopped execution
before it. `solve()` counts "the error changed and the score didn't drop" as fixed, so `multiply` went
on the keep list while it was still in the code, and it surfaced again in r6. The r5 regression fits
an over-applied "dst must be in psum" keep line, but that's one instance and not proven.

Sample keep list exactly as it was appended to the round-6 prompt (rebuilt from the log with the
`solve()` logic; the r3 prompt-length check, +207 chars, matches `len(keep_note)` exactly):

```
Already fixed in earlier rounds -- keep these fixed, do not bring them back:
- AssertionError: dma_copy requires src and dst to have the same number of elements, got src=4, dst=16384 The tile you allocated
- TypeError: nc_matmul() got an unexpected keyword argument 'transpose_moving' Remove the `transpose_moving=` argument. The
- AssertionError: dst must be in ['psum'], got sbuf Allocate the `dst` tile with buffer=nl.psum instead of nl.sbuf. For nisa.nc_
- AttributeError: module 'nki.isa' has no attribute 'multiply' `nki.isa` has no `multiply`. To multiply a tile by a number or a
```

Round 6 then failed on exactly that last line. The keys carry the first words of the hint, cut off
mid-sentence, because `_err_key` keeps 110 characters.

## Verdict

**Not supported. Reverting is the default.** The keep list didn't lower recurrence at either level.
At L4 it never fired, because the agent is stuck on one error, not cycling. At L1 it fired and
recurrence was 0.43 against 0.14 without it. No score moved (L4 0.62/0.50 from round-0 sampling, L1
0.30). Only 2+1 runs, under deadline, but L1 is greedy and both levels gave the same score on every
earlier run. Two lessons. (1) A prompt-only constraint doesn't replace freezing in code: Qwen reproduced
a listed error. (2) The bookkeeping must check the code, not the error text. An error that disappears
because an earlier error masks it isn't fixed. Suggested change for the agent.py owner (not made):
add a key to `fixed` only when the offending construct is gone from the code (e.g. the invented
attribute no longer appears), or when the new error is raised *later* in execution (more shapes or
stages pass). Drop the hint text from the key, and keep only the error type and message.
