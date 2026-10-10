# Demo script (about 6 minutes)

Every kernel, message and number below is from a logged run (`results/seat-130/`, code version
`e22ccab872`) or from `upstream-nki/`. Every command runs on a laptop with no model and no
accelerator; the model's own kernels are replayed from `demo/*.py` through the real harness.

Setup once: `uv sync`. Commands run from the repo root.

---

## 1. The problem (30 s)

> A small model (Qwen3-8B, 8192-token context) writes tiled NumPy kernels under accelerator rules:
> tiles of at most 128 x 512, explicit loops, no fancy indexing, no calling the op you implement,
> and the last tile is partial. Wrong kernels do not crash, they return plausible numbers. So the
> harness, not the model, has to know what is right, and the agent has to turn every verdict into
> one change the model can make.

Show the ladder and the deliverables: `README.md`, `TAXONOMY.md`, `EVALSET.md` (207 dev + 156
holdout cases, never shown to the model).

## 2. One failure and its recovery: row sum, three attempts (2 min)

Level 2, run 1 of `ours`. Three kernels the model wrote, replayed through the harness.

**Attempt 1: the obvious kernel, and it is illegal.**

```bash
uv run python -m kagent.harness 2 demo/l2_attempt1_np_sum.py --fixes
```

```
L2 row_sum: FAIL, 1 rule violation(s), 28/28 cases correct
- line 15 `result[i:end_row] += np.sum(tile, axis=1)`: np.sum [sum-like]
FIX line 15: Replace `np.sum` with an explicit python loop over the columns of the tile ...
```

> Numerically perfect, 28 of 28, and it scores zero, because `np.sum` hides the tiling. The agent
> does not send this report back. It sends the FIX line: one instruction, with the loop to write.

**Attempt 2: the instruction is followed, and the partial-tile trap fires.**

```bash
uv run python -m kagent.harness 2 demo/l2_attempt2_index_error.py --fixes
```

```
L2 row_sum: FAIL, 0 rule violation(s), 9/28 cases correct
Pattern: every failing case is 'multi-col-tile' and no passing case is.
Crash in 19 case(s) (e.g. 'normal 1x700'): IndexError: index 512 is out of bounds for axis 1
with size 188 at line 17 `acc += tile[:, col]`
```

> The rule is gone, but it indexes the tile with the global column `col`. On the second column
> tile, col is 512 and the tile is 188 wide. The harness says which cases fail, what they have in
> common (more than one column tile) and the exact line.

**Attempt 3: verified.**

```bash
uv run python -m kagent.harness 2 demo/l2_attempt3_pass.py
```

```
L2 row_sum: PASS, 0 rule violation(s), 28/28 cases correct
All 28 cases within the float32 error bound (worst uses 24% of it).
```

> The fix is one token: `tile[:, col - j]`. Then the agent re-checks on 21 holdout cases the model
> never saw, passes 21 of 21, and only then reports "verified", confidence 0.95. 773 input tokens
> for the whole level.

## 3. Why translate the verdict: naive vs ours on level 1 (1 min)

Same model, same task, same harness. The only difference: `naive` sends the checker's report back
verbatim; `ours` sends one instruction.

```bash
uv run python -m kagent.harness 1 demo/l1_naive_attempt1.py
```

```
- line 6 `x = np.asarray(x, dtype=np.float32)`: np.asarray [escape]
```

> Naive gets exactly this line back. Its next attempt changes `np.asarray` to `np.array`
> (`demo/l1_naive_attempt2.py`), the next one back to `np.asarray`, and the stopping rule fires.
> The report says what is wrong; it does not say what to write. Ours gets "Remove `np.asarray`;
> work directly on slices of the input arrays" and passes on the next attempt.

| level 1 (relu) | runs verified | attempts when verified |
|---|---|---|
| ours | 5/5 | 1.6 |
| naive | 3/5, the other 2 stuck on the same violation | 1.7 |

Across levels 1-4 (5 runs each, same harness version): **ours 15/20, naive 4/20**
(L1 5/5 vs 3/5, L2 4/5 vs 0/5, L3 5/5 vs 1/5, L4 1/5 vs 0/5).

(Source: notes E24, `scripts/analyze.py` on `results/seat-130/`.)

## 4. Does the agent know when it failed? (1 min)

> Level 4, RMSNorm, is our wall: 1 of 5 runs verified. The other four are reported as `failed` or
> `stuck`, confidence 0, never as done.

Show `TAXONOMY.md`: the first wall at every level is a banned reduction; `naive` stops on rule
violations, `ours` stops on deeper problems (crashes at partial tiles, float32 overflow,
accumulators). And the honest cost: 24 of 91 of our repairs fixed the error shown and broke
something else.

> One more calibration case: level 7 matmul passed dev but failed holdout by timeout. We reported it
> as `dev-only`, confidence 0.3, then re-checked with a long timeout: slow but correct, 10/10.
> Calibration has to separate slow from wrong (notes E21).

## 5. A failure the translator could not name, and the fix (30 s)

```bash
uv run python -m kagent.harness 3 kernels/broken/l3_overwrite_per_tile.py --fixes
```

```
Pattern: in all or nearly all 9 wrong cases, each wrong output equals the result over the last
column tile only.
```

> The model's own L3 bug: it assigned the per-tile max instead of keeping a running max, so each
> tile overwrote the last. Our first harness could only say "wrong numbers". We added a diagnostic
> that recomputes the reference per column tile; now it names the bug and the directive says
> "keep one running value per row across all column tiles". The deliberate-bug suite is a floor,
> the model's bugs extend it (notes E9).

## 6. Making verified kernels cheaper (30 s)

Show `notes/fig-opt2.png`.

> 8 of 15 verified kernels were per-element loops. The optimiser stage rewrites them and keeps only
> rewrites the harness still verifies: 5 of 11 accepted, up to 79x fewer instructions, no kernel
> made wrong (notes E19).

## 7. Stage B: the same lesson on real NKI (30 s)

`upstream-nki/FEEDBACK-EXPERIMENTS.md`, upstream NKI agent, NKI simulator, 5 runs per cell.

| NKI level | upstream baseline | with our feedback |
|---|---|---|
| L1 average pool | 0/5 (0.30) | **5/5** (v5) |
| L2 transpose | 5/5 | not changed in v6 (run pending) |
| L3 matmul, one tile | 0/5 (0.30) | 0/5 (0.30) |
| L4 matmul, tiled | 0/5 (0.62) | **1/5** (v2) |

> On NKI the failures are API misuse, not wrong numbers. The biggest single change was the same one
> as in Stage A: the stock repair prompt said "change exactly what the checker names and keep
> everything else identical" while the checker asked for a new loop structure; the model obeyed the
> first in 72 of 80 level-4 attempts. Asking for a rewrite solved level 4. Level 1 went from 0.30
> to 1.00 when feedback named each wrong `.ap()` stride. We also report the cost: that last
> feedback computes the answer per axis, and adding test shapes to an already-solved level made
> it worse, so we removed it there.

---

## Backup slides, if asked

- **Tokens:** largest prompt 706 tokens, 13% of the input budget (`notes/fig-tokens-run4.png`).
  In Stage A the budget never binds; in Stage B it would, because the model invents NKI APIs.
- **Runs are not independent:** the server is near-deterministic at temperature 0.6, so each repeat
  uses a different task wording (notes E10).
- **Reproduce without a model:**
  `uv run python -m kagent.agent --levels 1 2 --scripted tests/scripts/demo.json --out runs/demo.jsonl`
