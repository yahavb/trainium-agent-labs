# Level 2: what the failing kernels actually compute

Status 16:40. Inputs: v8.2's level-2 logs (`runs/seat-116/v82/v82_L2_s116.jsonl`, three runs: solved once,
twice stuck at 0.50; `runs/seat-119/v82/v82_L2_s119.jsonl`, two runs, both solved), so **v8.2 is 3 of 5 on
level 2**, not 3 of 4. Also v8.1 (`runs/seat-116/v81_partial/v81_L2.jsonl`), v7 and the baseline + replica.
Scripts: `analysis/l2_deep/l2_what.py`, `analysis/l2_deep/l2_cat.py` (NKI 0.6.0, trn2, Docker paths).

**Method.** Every distinct level-2 kernel that runs but is wrong (13 kernels) was simulated on level 2's loop
shapes with `x = arange` (every element distinct), so each output value *is* the index of the input element it
came from. The output was then matched against named wrong transforms.

## What 0.50 kernels compute (34 attempts scoring 0.50; v8.1 and v8.2. v7 and the baseline have no 0.50 on level 2)

| what the kernel computes | 0.50 attempts | v8.1 | v8.2 | runs |
|---|---|---|---|---|
| **one loop variable dropped**: output position `i*B + j` takes `x[i]`, i.e. each source element repeated B times | **18** | 11 | 7 | 4 |
| **one loop variable dropped** (the other one): output position `k` takes `x[k mod A]` | **7** | 1 | 6 | 4 |
| **identity**: x returned unchanged | 4 | 1 | 3 | 3 |
| only part of the output written, the rest uninitialised (NaN) | 4 | - | 4 | 1 |
| raises on the (32, 12) as 3x4 case | 1 | - | 1 | 1 |

On shape (32, 12) as 3x4, row 0, the required output is `[0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]` and the most
common wrong kernel returns `[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]`. It loops over both block indices but builds the
source index from one of them only. None of the 0.50 kernels transposes all of x, swaps A and B, or shifts rows:
those were checked and match nothing.

## Why the failing runs never recover

The two failing v8.2 runs on seat-116, round by round. Sample 1 is the repair; samples 2-4 are E-mix fresh attempts
from the first prompt:

- **run 3**: from round 1 the repair kernel scores 0.50 every round. The message alternates between "NUMERICAL
  MISMATCH: worst error 4.21 of the output's RMS" and "... 3.54 ...", with the worst element's index and value
  for seven rounds. Those are two versions of the dropped-loop-variable kernel, swapping back and forth. The
  message gives size and place of the error and never says what the kernel did. The fresh samples score 0.30
  (they do not run) or occasionally 0.50.
- **run 1**: rounds 1, 5, 6, 7 carry the partial-write kernel: "NON-FINITE OUTPUT: 256 NaN ... Usually an
  uninitialised PSUM or SBUF tile". Rounds 2-4 detour through `nisa.fill` (does not exist) and `memset` /
  `tensor_copy` to HBM, trying to initialise the output instead of filling every position.

Of the fresh samples in these two runs (42), none solved. MIX did not rescue level 2 here.

## Recommendation: one change, level 2 only

**Error distillation for level 2's numeric mismatch.** When a level-2 kernel runs and is wrong, the checker runs
it once more on `arange` input, reads off where each output element came from, and says it in one sentence
without code. For example: "Your output at position `i*B + j` of each row holds `x[i]`: the source index uses
`i` only. Each row's A-by-B block must come out B-by-A: output position `j*A + i` must hold `x[i*B + j]`."
For identity: "Your output equals x unchanged: nothing was moved." For the partial write: "Only the first
B positions of each row were written."

Why this one: 29 of the 34 stuck attempts are one of three mechanically recognisable transforms, and the message
the model gets today describes none of them. The best-documented failure today (run 3) is exactly that: seven
rounds of an error size with no mechanism. It needs no code, so it gives no answer away: it restates what the
kernel did and what the level asks. It only fires on level 2's wrong-numbers path, so levels 1, 3 and 4 and
level 2's other messages are untouched. Implementation sketch: in `grade()` for level 2, after
`describe_mismatch`, simulate on `arange` (the kernel already loaded), compute `src = got - row*F`, and test the
three patterns above. That is about 30 lines, the same code as `l2_cat.py`'s `cat()`.

What it will not fix: the 0.30 fresh samples that do not run at all, and the NaN partial writes, which already
get a specific message.
