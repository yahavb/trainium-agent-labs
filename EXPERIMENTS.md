# Kernel agent, level 1 (average pooling 2D): what we changed and what we measured

Team seat(s): 180–184 · Model: Qwen3-8B on the seat's vLLM (TP=2), 8192-token context · Branch: `Dev_Paul`

All runs below are **level 1 only**. Unless noted they use `--rounds 6 --samples 4 --context 8192`.
Reward: parses 0.1 + rules 0.2 + runs 0.2 + correct 0.5. A kernel that crashes scores 0.30; one that runs but gives wrong numbers scores 0.50.

## 1. What we changed in `agent.py`

Every change is behind a flag, so each one can be switched on or off as a single variable.

| Change | Flag (default) | Why |
|---|---|---|
| **API card in the repair prompt.** The list of available NKI functions is repeated in every repair prompt, not only the first one. | `--repair-card` (1) | In repair rounds the model kept inventing functions. |
| **Name checker.** When a name does not exist, we search `nl`, `nisa` and `nki`. If the name exists in another module, the feedback says exactly what to write, e.g. "write `nl.multiply`, not `nisa.multiply`". Otherwise it lists the closest real names. | always on | Vague errors ("has no attribute") stalled the model for several rounds. Exact ones were fixed in the next round. |
| **Missing-argument enrichment.** For "missing N required argument", the feedback lists the function's full argument list. | always on | Same reason: make the error actionable. |
| **Repair instruction.** "Keep the rest identical" is replaced by "Make only the changes that are necessary to fix this." | always on | The old wording, together with the failure ledger, contradicted itself and caused regressions. |
| **Static checker** (see §2). | `--static-check` (1) | The simulator stops at the first error. This finds several errors before running. |
| **Thinking modes.** `--think`: thinking on. `--plan`: think (2000 tokens) → summary → code with thinking off, so the long thought does not use up the 8k context. `--plan-merge`: N thoughts at different temperatures → one merged summary → N code samples. | `--plan`, `--plan-merge`, `--think-temps`, `--think-tokens` | Without thinking the model chose the wrong algorithm (see §3). |
| **Task as words.** The model first describes the NumPy reference in its own words. The prompt then gets that description instead of the code. No hand-written text. | `--spec words` (code) | Tests whether the reference code helps or anchors the model. |
| **History.** Earlier failed attempts (code + error only, no model-written diagnosis, deduplicated, capped at 6000 characters) go into the repair prompt. | `--history N` (0) | Stops the model returning to code that already failed. Diagnoses were left out because a wrong diagnosis would mislead it. |
| **Readable transcript.** Every run writes `<log>.txt` with the prompt, thinking, summary, each distinct reply and its feedback, for every round. | automatic | To see what the model actually saw and did. |

## 2. The checker and why it is built this way

`static_check(source)` runs on the model's code **before** the simulator and never changes the reward. It only adds text to the feedback when the kernel is not correct:

> Before running, a static check of your code found: (1) … (2) … When it ran: …

What it checks, and why:

1. **Every `nl.*` / `nisa.*` / `nki.*` name is looked up in the real modules.** Inventing or misplacing functions was the most common error in every run.
2. **Keyword arguments and missing required arguments are checked with `inspect.signature`** against the real function. Wrong argument names were the second most common error.
3. **Results that are thrown away.** `nl.sum(...)`, `.ap(...)` and `.reshape(...)` *return* a new value. Calling them on a line of their own does nothing. This came from a real run: our own feedback said "remove `dst=`", the model did so, and the output became all NaN. So the `dst=` hint now says "it RETURNS the result; write `result = nl.sum(...)`".
4. **At most 8 findings,** so the feedback stays short.

**False-positive check:** the checker reports zero findings on all four reference kernels (`reference_level1-4.py`). It is wrapped in try/except, so a bug in the checker can never stop a run.

**Known gap:** it checks names and arguments, not the *kind* of value. In `merge_static` round 3 the model wrote `data=1.0/(p*p)` where `data=` must be a tile, and the checker said nothing.

## 3. Runs

| Run (log) | Configuration | Best | Where it got stuck |
|---|---|---|---|
| `base_l1`, `card`, `names`, `v4` | no thinking; feedback changes added one at a time | 0.30 | Wrong algorithm: one `nc_matmul` per window. All 4 samples were identical, so best-of-4 gave nothing. Precise feedback fixed each error in one round, but a new one appeared every time. |
| `plan`, `plan6` | `--plan`, 2 samples | 0.30 | **Right algorithm from round 0** (strided view + reduce). Fixed one error per round, then reached the "tiles need 2 dimensions" wall and the reduce-axis wall. `plan6` stopped early because the tie-break kept the stuck sample and dropped the one that was progressing. |
| `merge_multi` | `--plan-merge`, temperatures 0.6/0.75/0.9/1.0, no static check | **0.50** (round 3) | **The only run where a kernel executed** (not replicated, see `merge_multi_rep`). The output was NaN because the result of `nl.sum` was thrown away. It got there with a reshape that is semantically wrong (it relabels the data instead of forming 2×2 windows). Rounds 4–5 regressed (it invented `nl.assign`). |
| `merge_single` | `--plan-merge`, all temperatures 0.6 | 0.30 | `.ap` partition stride, 5 rounds. |
| `merge_words` | `--plan-merge --spec words` | 0.30 | Chose a per-window loop and hit the 2-dimension wall 4 rounds in a row (give-up). |
| `merge_static` | `--plan-merge` + static check | 0.30 | API errors were all gone by round 3, then `.ap` partition stride for every remaining round. In rounds 1–4 all 4 samples were identical. |
| `merge_history` | + `--history 3` | 0.30 | Chose a per-window loop. **All 6 rounds** hit the 2-dimension wall: `nl.sum(view, axis=[1])` returns a 1-D `(C,)` tile, and the model never used `keepdims=` even though the feedback listed it. In round 3 it went back to `nisa.multiply`, although the history block showed round 1's code with that exact error. History did not prevent going back. |
| `merge_multi_rep` | same as `merge_multi` (replication) | 0.30 after 5 of 6 rounds | **The 0.50 did not replicate.** It hit a different wall almost every round: reduce axis `(2,4)` → 2-dimension wall → "partition dim must be preserved" → 2-dimension wall → `dma_copy` element count. *(round 5 running at deadline)* |
| `merge_t10` | static check, all temperatures 1.0 | 0.30 after 4 of 6 rounds | Same path as `merge_static`: rounds 0–2 fixing API errors flagged by the static check, then the `.ap` partition stride in round 3. *(running at deadline)* |
| `merge_tmix` | static check, 0.6–1.0 (replicates `merge_static`) | 0.30 after 4 of 6 rounds | `.ap` partition stride in rounds 1–3, bouncing between `[[1,32],[2,16],[2,16]]` and `[[2,16],[2,16],[1,32]]`. *(running at deadline)* |

Time per round with `--plan-merge` and 4 thoughts was about 430–455 s. With 2 thoughts it was about 155 s. The seat is saturated: running 4 at once costs about as much as running them one after another.

## 4. What we learned

1. **Precise feedback works; vague feedback stalls.** "Write `nl.multiply`" and "missing `operand0=`" were fixed in the next round. "float has no shape" was not fixed for several rounds.
2. **Thinking changes the algorithm.** Without thinking the model used matmul per window. With thinking it chose a strided view plus a reduction, which is the reference's approach.
3. **The walls are missing information, not a lack of search.** Across all runs, temperatures and feedback variants, the model hit the same three walls:
   - a reduction over axes that are not the last ones (`axis=[2,4]`);
   - SBUF/PSUM tiles that must have 2 dimensions;
   - the unit of the `.ap` stride.

   The error message says "Partition step 1 must equal tensor free dimension size 1024", yet the model never used 1024. Instead it removed dimensions from the pattern each round (5 → 3 → 2). The API card has only one line on `.ap`. More thinking and different temperatures did not get past any of these.
4. **Merging the thoughts collapses diversity.** After one shared summary, the 4 code samples are usually identical whatever the temperature. `--plan-merge` therefore behaves like *one* sample per round, and a wrong summary takes every sample down with it.
5. **Our own feedback can create bugs.** "Remove `dst=`" led to `nl.sum` being called with its result thrown away, and the output became NaN. We now word every hint around what the function *returns*.
6. **The reference code carries useful structure.** Replacing it with the model's own description led to a worse algorithm.
7. **Showing failed attempts was not enough to stop the model going back to them** (`merge_history`). An explicit, precise hint seems to matter more than memory.
8. **Variance is large; n is small.** Two runs with identical settings (seats 182 and 184 before history takes effect) hit different first errors at round 0. Our only 0.50 (`merge_multi`) **did not replicate** in `merge_multi_rep`. The three runs with the static check on (`merge_static`, `merge_tmix`, `merge_t10`) all ended at the same `.ap` wall, whether the temperatures were mixed or all 1.0. So the walls in finding 3 are reproducible, while the score is not. With 1–2 runs per configuration we report behaviour, and we **do not** rank configurations by score.

## 5. Next steps (not done)

- **Explain the `.ap` stride unit when that error appears.** Either attach the real NKI docstring, or restate the error with the numbers worked out for this tile ("one partition step = H·W = 1024 elements, one row = W, one column = 1"), in the style of finding 1.
- **A hint for reductions over non-last axes, and for the 2-dimension wall after a reduction** ("`nl.sum(..., keepdims=True)` keeps the tile 2-D").
- **A value-kind check in the static checker** (tile vs number).
- **Keep diversity through the merge:** one summary per thought, or break ties toward the sample whose error is *new*.
- **More replications per configuration** before comparing scores.

## 6. Reproduce

```bash
cd projects/02-kernel-agent
python agent.py --level 1 --plan-merge --rounds 6 --samples 4 --context 8192 --log merge_static.jsonl
# add --static-check 0 / --history 3 / --think-temps 1.0,1.0,1.0,1.0 / --spec words for the other rows
```

Each run writes `<log>.jsonl` (every attempt: prompt, thinking, summary, reply, reward, feedback; the file is appended to) and `<log>.txt` (a readable transcript, overwritten each run). These are the full attempt logs.