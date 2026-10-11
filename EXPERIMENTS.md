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
| `merge_multi_rep` | same as `merge_multi` (replication) | 0.30 (the run ended after 5 rounds) | **The 0.50 did not replicate.** It hit a different wall almost every round: reduce axis `(2,4)` → 2-dimension wall → "partition dim must be preserved" → 2-dimension wall → `dma_copy` element count. |
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

**Layout hints** (`--layout-hints 1`; static check on). With a matching error, the hint adds the NKI rule with this tile's own numbers: the `.ap` stride unit (one step along each dimension in elements, so the first pair must be `[1024, n]`), "reductions only take the last dimensions; reshape only relabels, so reorder with `.ap`", and `keepdims=True` after a reduction. It never gives the finished pattern. The runs were stopped by the deadline:

| Run | Mode | Rounds | Best | What happened |
|---|---|---|---|---|
| `hints` | `--plan-merge` (compare with `merge_static`, `merge_tmix`, `merge_t10`) | 5 | 0.30 | **Each hint was followed in the very next round.** R0: per-window loop, 2-D wall. R1: added `keepdims=True` (the hint), and met a new error, reducing the partition axis. R2: tried `.ap([[p, p], [1, 1]])` and got the stride hint. R3: **`tile.ap([[H*W, 1], [W, p], [1, p]])`, the first correct `.ap` strides in any run** (partition step H·W, row W, column 1), but then `tile / int`. R4: replaced `/` with `tensor_scalar`, then an index went out of bounds in its per-window loop. Still 0.30, but every round was a new error and none was a wall it had already hit. |
| `hints_plan` | `--plan` (each sample plans alone) | 4 | 0.30 | The hints never triggered. All 4 rounds swapped `data=` and `operand0=` in `tensor_scalar`: the value-kind gap of the static checker (§2), the same A → B → A as in `merge_static`. |
| `hints_think` | `--think --max-tokens 7000` | 1 | 0.30 | 3 of 4 samples used the whole budget thinking and returned no code (reward 0.0). Raw thinking does not fit an 8k context, as the organisers measured. |

The first `--decompose` run (`dec_fast`) never produced a plan: three tries, the same `AxisError` each time. That led to two changes: a rejection now shows every stage's output shape, and the plan is written with thinking on. The rerun was not finished by the deadline.

**Level 2 (2D transpose), with and without persona** (`persona_l2`, `base_l2`: `--plan-merge`, static check on). Both scored **0.30 after 6 rounds**. Both began with NumPy idioms (`nl.reshape`, `nl.transpose(axes=...)`):

- Without persona, it was stuck 3 rounds on a `dma_copy` element count (12 into 1), then on out-of-bounds indices.
- With persona, it invented `nisa.tile` and `nl.ndarray(ap=...)`, then tried to `dma_copy` an `.ap` view.

The organisers' plain agent (no thinking) solved level 2 in 4 of 5 runs, so on this level our thinking-and-merging setup did *worse* than no thinking at all. Settings differ (8 rounds, no static check) and n is small, but the direction is clear: thinking pulls the model toward NumPy idioms on this level too.

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

9. **(After the first submission) Rules stated with the tile's own numbers are followed immediately.** In `hints`, `keepdims=True` and the `.ap` stride unit were each applied in the round right after the hint. That run wrote the first correct `.ap` strides of the day, and it never went back to a wall it had already hit. The score stayed at 0.30 inside 5 rounds, so this is evidence about behaviour, not yet about outcome.

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

In priority order, after the evening runs.

1. **Run `--layout-hints` to the end, and repeat it.** These hints (the `.ap` stride unit with the tile's numbers, last-dims reductions, `keepdims`) are now implemented. In `hints` each one was followed in the next round, but the run had only 5 rounds. Next: 8+ rounds, 2–3 runs, compared with `merge_static`, `merge_tmix` and `merge_t10`.
2. **A value-kind check in the static checker** (a tile vs a number). This is now the most frequent wall where the hints do not apply: `merge_static` and `hints_plan` spent 4+ rounds swapping `data=` and `operand0=` in `nisa.tensor_scalar`. The check: `data=` must be a tile, and `operand0=` a number or a per-partition tile, possibly the same tile as `dst=`.
3. **Expected vs actual output on a small block** when a kernel runs but is wrong. "98.7% of elements are wrong" is a verdict. In `persona` round 3 the model changed the wrong line (multiplied by p² instead of dividing) and the error grew from 5.6× to 64× the RMS. A 4×4 block of input, expected output and actual output would show that the wrong elements are being averaged.
4. **Finish testing `--decompose`.** It is implemented, and the plan check now shows each stage's output shape and is written with thinking on. The rerun did not finish. The matmul levels are its best test case: there "a single tile" already passed in round 0 and was then broken by every attempt to tile.
5. **Tile methods in the name checker.** `nl.reshape` gets "nothing similar exists" because only the `nl`, `nisa` and `nki` modules are searched; `reshape` and `ap` are methods of a tile and should be suggested as `tile.reshape(...)`.
6. **An enrichment rule for `'module' object is not callable`** that names the call and what it should be (level 6 stalled 3 rounds on it).
7. **Keep diversity through the merge:** one summary per thought, or break ties toward the sample whose error is *new*.
8. **Level 2: thinking vs no thinking.** The organisers' plain agent solved it 4 times in 5; our `--plan-merge` runs scored 0.30 twice. Run both settings 3+ times each, to see whether thinking really hurts on simpler levels.
9. **More replications per configuration** before comparing scores.

**How `--decompose` works** (design notes for item 4). It targets the regressions we saw (fixing one thing broke another, as in `merge_multi` rounds 4–5, `merge_history` round 3 and levels 5–6):

1. The model writes its plan as a short sequence of NumPy steps (e.g. load `x` → window sums `(C, H/p, W/p)` → divide by p²).
2. The harness runs the steps on CPU and checks that the last one equals the reference, so a wrong plan is rejected in milliseconds. No hand-written answers are needed.
3. The kernel is built in stages. Stage *k* must reproduce the output of the model's own NumPy step *k*, and its kernel is the starting point of stage *k+1*, so later fixes do not have to rediscover it.

Stages are built on top of each other rather than written separately and joined, because intermediate results live in SBUF tiles and a join would be a new place to fail. Decomposition does not supply missing knowledge, so it should be paired with the layout hints, but each should first be tested as a separate variable.

## 6. Reproduce

```bash
cd projects/02-kernel-agent
python agent.py --level 1 --plan-merge --rounds 6 --samples 4 --context 8192 --log merge_static.jsonl
# add --static-check 0 / --history 3 / --think-temps 1.0,1.0,1.0,1.0 / --spec words for the other rows
```

Each run writes `<log>.jsonl` (every attempt: prompt, thinking, summary, reply, reward, feedback; the file is appended to) and `<log>.txt` (a readable transcript, overwritten each run). These are the full attempt logs.