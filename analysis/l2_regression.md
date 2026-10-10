# Level 2 under v7: a regression, or one unlucky run?

Status 15:17. v7's level-2 log (`runs/seat-116/Ev7_L2/`) is not pulled yet; what we know of it is one run at
0.30. Everything below comes from the baseline and replica logs (`analysis/logs/`) and from the prompts v7 and
the baseline build, generated offline (NKI 0.6.0, trn2). Section 4 will take v7's own log when it arrives.

**In one line: one run at 0.30 is not evidence yet. Under the baseline's own rate a run fails about half the
time. If v7 is worse, the cause has to be in round 0, because that is the only round that has ever solved
level 2. Round 0 differs from the baseline in exactly two layers: PROMPT1 (the prompt) and SAMPLING (the
decoding). Test those with round-0-only runs, not full runs.**

## 1. Level 2 is solved in round 0 or not at all

| log | runs solved | solved in | round-0 samples correct | repair-round samples correct |
|---|---|---|---|---|
| baseline (seat-116) | 3/5 | round 0 each time (runs 1, 3, 5) | 4/20 | **0/48** |
| replica (seat-119) | 2/5 | round 0 each time (runs 1, 4) | 3/20 | **0/56** |
| together | 5/10 | round 0 | **7/40 = 17.5%** | **0/104** |

The runs that missed in round 0 never recovered. Their carried failures were `copy_size_mismatch` for the
first rounds, then `out_of_bounds` until the agent stopped (taxonomy modes; run-by-run in the logs). So under
the baseline, level 2 is a draw of four round-0 samples. A run fails when all four miss: 5 of 10 runs did.

**What one v7 run at 0.30 means.** If v7 were exactly as good as the baseline, a run would score 0.30 with
probability about 0.5. To call a regression, v7 needs more runs. 0 of 5 would happen by chance about 3% of the
time (0.5^5); 0 of 3, about 12%.

## 2. What differs in round 0

The baseline's first prompt for level 2, generated with the current `agent.first_prompt(2)`, is 2,673
characters. That is exactly the `prompt_chars` both logs recorded for every level-2 round-0 request (agent.py's
prompts are byte-identical to the organizers' since e7663a3).

v7 with V7.md's configuration builds a 3,911-character first prompt. The diff has two parts:

1. **PROMPT1=v2, the def line.** "Entry point: ... decorated with `@nki.jit`" becomes "... with exactly the
   reference's inputs:" followed by `@nki.jit` / `def tensor_transpose2D_kernel_(x, shape2D):`.
2. **PROMPT1=v2, a "More real functions" block** (about 1,200 characters, roughly 300 tokens):
   `nisa.activation` with nl.exp/log/sqrt/..., `nisa.tensor_tensor`, `nisa.tensor_scalar` with a per-row
   operand, `nisa.reciprocal`, `nl.max/sum/min/mean` with keepdims. **None of these is used by a transpose.**
   Level 2 needs element moves (`nisa.tensor_copy` or DMA), which the block does not mention.

`CARD=category` adds nothing for level 2 (`feedback_v5.category(2)` is "other").

The request also differs outside the prompt: **SAMPLING=qwen** sends temperature 0.7, top_p 0.8, top_k 20,
against the baseline's temperature 0.6, top_p 0.95.

**Checked: with `PROMPT1=theirs`, v7's level-2 first prompt is byte-identical to the baseline's** (2,673
characters, compared). So `PROMPT1=theirs` removes the prompt difference completely, and `SAMPLING=theirs` the
decoding one.

## 3. Which layer

| layer | acts in round 0? | suspect for level 2? |
|---|---|---|
| PROMPT1=v2 | **yes**: +1,238 characters of def line and functions a transpose does not use | **most likely**. It changes the only round that solves level 2, and adds a list the model may try to use. |
| SAMPLING=qwen | **yes**: lower top_p, top_k 20 | possible. Less diverse round-0 samples could hurt a level that lives on round-0 luck. |
| CARD=category | no change for level 2 | no |
| MESSAGES=v5 | no (repair rounds only) | unlikely. The baseline's repairs never solved level 2 either (0/104). |
| REPAIR_PROMPT=restructure | no (repair rounds only) | unlikely, same reason |
| GATE=static | only on a full score (holds it at 0.95) | only if v7's log shows a 0.95; the one run we know of scored 0.30 |

## 4. Suggested test: round 0 only, three configurations

A full run costs up to 8 rounds and says one bit (solved or not) with probability about 0.5 either way. Since level
2 is decided in round 0, measure round 0 directly. Same configuration as V7.md, level 2, one round, five repeats
(about 5 minutes each at ~60 s a round):

```bash
# a. v7 as configured
python3 feedback_v7.py --level 2 --rounds 1 --samples 4 --context 8192 --repeat 5 --log l2r0_v7.jsonl --verdicts l2r0_v7_v.jsonl
# b. the prompt back to the baseline's (byte-identical, checked)
PROMPT1=theirs python3 feedback_v7.py --level 2 --rounds 1 --samples 4 --context 8192 --repeat 5 --log l2r0_p1.jsonl --verdicts l2r0_p1_v.jsonl
# c. the decoding back to the baseline's
SAMPLING=theirs python3 feedback_v7.py --level 2 --rounds 1 --samples 4 --context 8192 --repeat 5 --log l2r0_s.jsonl --verdicts l2r0_s_v.jsonl
```

Each gives 20 round-0 samples to set against the baseline's 7/40 (17.5%). At the baseline's rate, 0 of 20
would happen by chance about 2% of the time (0.825^20). So if (a) scores near 0/20 and (b) or (c) comes back
near 3-4/20, the layer is found. If only one configuration can run, run (b): PROMPT1 is the bigger change.
`scripts/compare.py a=... b=... c=... --ref a` puts them in one table, and the per-sample counts are in the
attempts files (`round` 0, `reward` 1.0).

## 5. v7's level-2 log

*Pending: runs/seat-116/Ev7_L2/ not pulled at 15:17. To fill: taxonomy of its failures, its round-0 samples
correct, any 0.95 (gate), and the first prompt's length from `prompt_chars` (it should be 3,911).*
