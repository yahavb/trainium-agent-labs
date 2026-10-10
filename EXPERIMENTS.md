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
| **Persona** (added after the first submission). One role sentence goes in front of every request: "You are a senior AWS Neuron kernel engineer with years of experience writing correct NKI kernels for Trainium." It adds no NKI facts, so any difference comes from the role framing alone. | `--persona` (off) | A common prompting trick; we test it as one variable. |
| **Decompose** (added after the first submission). The model writes NumPy steps `stage_1..stage_n`. The harness runs them on CPU on every test shape and accepts the plan only if the last stage equals the reference; a rejection shows each stage's output shape. The kernel is then built one step at a time: step *k* is graded against `stage_k`, and a passing step's kernel is the starting point of the next. | `--decompose` (off), `--plan-think` (1), `--plan-tries` (3), `--stage-rounds` | Targets regressions: fixing one thing kept breaking another. See §5 for the design. |

## 2. The static checker

`static_check(source)` runs on the model's code **before** the simulator and never changes the reward. It only adds text to the feedback when the kernel is not correct:

> Before running, a static check of your code found: (1) … (2) … When it ran: …

What it checks, and why:

1. **Every `nl.*` / `nisa.*` / `nki.*` name is looked up in the real modules.** Inventing or misplacing functions was the most common error in every run.
2. **Keyword arguments and missing required arguments are checked with `inspect.signature`** against the real function. Wrong argument names were the second most common error.
3. **Results that are thrown away.** `nl.sum(...)`, `.ap(...)` and `.reshape(...)` *return* a new value. Calling them on a line of their own does nothing. This came from a real run: our own feedback said "remove `dst=`", the model did so, and the output became all NaN. So the `dst=` hint now says "it RETURNS the result; write `result = nl.sum(...)`".
4. **At most 8 findings,** so the feedback stays short.

**False-positive check:** the checker reports zero findings on all four reference kernels (`reference_level1-4.py`). It is wrapped in try/except, so a bug in the checker can never stop a run.

**Known gap:** it checks names and arguments, not the *kind* of value. In `merge_static` round 3 the model wrote `data=1.0/(p*p)` where `data=` must be a tile, and the checker said nothing.

## 3. Trials

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

### After the first submission

**Persona, level 1** (`persona`: `--plan-merge --persona`, static check on; compare with `merge_static`, `merge_tmix`, `merge_t10`). Best **0.50**, in rounds 2–3. Round by round:

| Round | Best | What the code did |
|---|---|---|
| 0 | 0.30 | `reshape((C, H/p, p, W/p, p))`, then `nl.sum(..., axis=[2, 4], dst=...)`. The static check flagged `dst=`. |
| 1 | 0.30 | The reduce-axis wall: "for a 5D tensor, expected axis=(3, 4)". |
| 2 | **0.50** | Changed the reshape to `(C, H/p, W/p, p, p)` and summed `(3, 4)`. It runs, but a reshape only relabels the data, so it summed the wrong elements (98.7% wrong). **This is the same bug as `merge_multi`'s 0.50.** |
| 3 | 0.50 | The feedback said only "most elements are wrong". The model changed the scale to `op0=nl.divide, operand0=1/(p*p)`, which multiplies by p², and the error grew from 5.6× to 64× the RMS. |
| 4–5 | 0.30 | Diagnosed the layout correctly and switched to `.ap` to reorder the data (the reference's approach), but wrote the strides as `[[1,C],[1,H],[1,W],[p,H/p],[p,W/p]]`: the `.ap` stride-unit wall again. |

Two lessons beyond persona itself:

- **Both 0.50s came from the same wrong reshape.** It removes the error message but not the bug.
- **"Most elements are wrong" is a verdict, not an instruction**, and the model guessed at the wrong line. An expected-vs-actual comparison on a small block would show that it averages the wrong elements.

With one run, we do not credit the 0.50 to the persona: `merge_multi` reached 0.50 once without it, and that did not replicate.

**Matmul levels 5–7** (`l5_merge`, `l6_merge`, `l7_merge`: `--plan-merge`, static check on, one run each). Same reference as level 4; the levels add stricter limits on HBM traffic.

| Level | Best | Path |
|---|---|---|
| 5 (loads hoisted) | 0.62 in round 0 | Round 0 was a correct single-tile matmul: it passed K=M=128 and failed at 256 rows (no tiling). From round 1 on, every attempt to tile also broke the shape that had passed: 0.30 for 5 rounds. Off-by-one errors: `stationary[0]=127 != moving[0]=128`. |
| 6 (M, N blocked) | 0.62 in round 0 | Same round 0, then 0.30. Stuck 3 rounds on `'module' object is not callable`, an error with no enrichment rule that names neither the line nor the module. |
| 7 (M, N, K blocked) | 0.30 | Never passed the smallest shape. Element-count mismatches that flipped between rounds (`src=16384, dst=65536`, then `src=65536, dst=16384`), plus off-by-one bounds. |

- **The matmul walls are index arithmetic, not API knowledge.** The errors are bounds, element counts and off-by-one, not invented names or misunderstood API semantics as on level 1.
- **The best kernel came first and was never improved.** This is the regression pattern again and the strongest case for `--decompose`: on these levels "a single tile" is a step that already passes, and tiling is the next step.
- None of these runs can pass level 7. Passing needs every shape correct *and* HBM traffic within 1.05× the floor, so "passing the static checker" there means only that the API was used correctly.

**Still running when this was written:** `persona_l2` and `base_l2` (level 2, with and without `--persona`), and `dec_fast2` (`--decompose`, kernel rounds without thinking). The first `--decompose` run never produced a plan: three tries, the same `AxisError` each time. That led to the two changes in the table above: rejection feedback with every stage's output shape, and thinking on for the plan.

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

### Failure taxonomy (counts)

Counted over the 43 distinct kernels in the transcripts of `merge_static`, `merge_history`, `merge_tmix`, `merge_t10` and `merge_multi_rep`. Identical samples are counted once.

| Runtime wall (the error the simulator stopped on) | Kernels |
|---|---|
| `.ap` stride unit ("invalid partition stride") | 24 |
| Tile must be 2-D (usually after `nl.sum` without `keepdims`) | 15 |
| Reduce over non-last axes (`axis=(2,4)`) | 2 |
| Partition dim changed | 1 |
| `dma_copy` size mismatch | 1 |

API misuse caught by the static check in the same kernels (one kernel can have several): missing argument 12, wrong module (`nisa.multiply`) 10, `dst=` on a function that returns its result 9.

### Token instrumentation

Estimated input tokens of the task/repair prompt per round (characters ÷ 4, from the transcripts):

| Run | Round 0 → last |
|---|---|
| `merge_static` | 624, 995, 891, 882, 777, 956 |
| `merge_history` | 624, 875, 1430, 2038, 2531, 1928 |
| `merge_tmix` | 624, 860, 759, 759 |
| `merge_t10` | 624, 1037, 848, 849 |
| `merge_multi_rep` | 624, 729, 778, 720, 792 |

A repair prompt is roughly 40% API card, 40% the current kernel and 20% feedback. History adds about 500 tokens per remembered attempt. On top of this, each round sends the merge prompt (the task plus 4 clipped thoughts, sized to fit 8192 tokens) and 4 code prompts (task plus summary). The 8k budget was never exceeded.

## 5. Future Plan

- **Explain the `.ap` stride unit when that error appears.** Either attach the real NKI docstring, or restate the error with the numbers worked out for this tile ("one partition step = H·W = 1024 elements, one row = W, one column = 1"), in the style of finding 1.
- **A hint for reductions over non-last axes, and for the 2-dimension wall after a reduction** ("`nl.sum(..., keepdims=True)` keeps the tile 2-D").
- **A value-kind check in the static checker** (tile vs number).
- **Keep diversity through the merge:** one summary per thought, or break ties toward the sample whose error is *new*.
- **More replications per configuration** before comparing scores.
- **Tile methods in the name checker.** `nl.reshape` gets "nothing similar exists" because only the `nl`, `nisa` and `nki` modules are searched; `reshape` and `ap` are methods of a tile and should be suggested as `tile.reshape(...)`.
- **An enrichment rule for `'module' object is not callable`** that names the call and what it should be (level 6 stalled 3 rounds on it).
- **Decompose, then build in checked stages** (now implemented as `--decompose`; first results in §3). This targets the regressions we saw (fixing one thing broke another, as in `merge_multi` rounds 4–5 and `merge_history` round 3):
  1. The model writes its plan as a short sequence of NumPy steps (e.g. load `x` → window sums `(C, H/p, W/p)` → divide by p²).
  2. The harness runs the steps on CPU and checks that together they equal the reference, so a wrong plan is rejected in milliseconds. No hand-written answers are needed.
  3. The kernel is built in stages. Stage *k* must reproduce the output of the model's own NumPy step *k* before stage *k+1* may extend it. A stage that passes is frozen, so later fixes cannot break it, and the feedback always names the one stage that fails.

  We build stages on top of each other rather than writing the parts separately and joining them, because intermediate results live in SBUF tiles and a join would be a new place to fail. This does not supply missing knowledge (the `.ap` stride unit, `keepdims`), so it should be paired with those hints. Each should still be tested as a separate variable.

## 6. Reproduce

```bash
cd projects/02-kernel-agent
python agent.py --level 1 --plan-merge --rounds 6 --samples 4 --context 8192 --log merge_static.jsonl
# add --static-check 0 / --history 3 / --think-temps 1.0,1.0,1.0,1.0 / --spec words for the other rows
```

Each run writes `<log>.jsonl` (every attempt: prompt, thinking, summary, reply, reward, feedback; the file is appended to) and `<log>.txt` (a readable transcript, overwritten each run). These are the full attempt logs.