# Plan given, whole kernel: does a correct tiling plan get level 4 past 0.62?

- **Who / seat:** Rishabh (+ Claude), seat 48
- **Commit measured:** uncommitted on `team-main` (base `8f1ca41`): `agent.py` md5 06805f2d…,
  `plan_check.py` md5 181b7a09…. Gate-2 grading and `enrich()` are unchanged from the baseline.
- **Hypothesis:** level 4 stalls because Qwen picks the tiling wrong. Hand it the correct tiling and
  it writes a passing kernel.
- **Change:** `agent.py --plan-file reference`. Gate 1 is skipped; the normal level-4 prompt gets
  "Implement this tiling plan exactly…" plus `plan_check.REFERENCE` (the plan of `reference_level4.py`).
- **Command:** `python agent.py --level 4 --plan-file reference --rounds 8 --samples 4 --context 8192 --repeat 1`
- **Log:** `logs/attempts-rishabh-plan-given.jsonl` (28 attempts)

## Scores

```
before (baseline, no plan):      level 4: solved 0/4   all = [0.62, 0.62, 0.62, 0.62]
after  (plan given, whole kernel): level 4: solved 0/1   all = [0.30]   stopped at round 7: same failure 4 rounds running
```

One run, because level 4 has zero spread across runs. Two `optimize.py` jobs shared the model server
(scores unaffected), and sampling is effectively greedy here. The loop-tool session ran its conditions
on seat 49 with `--samples 1`.

Failures across the 28 attempts. It never got far enough to run, hence 0.30, not 0.62:
- `shape mismatch: value array of shape (65536,) could not be broadcast…`: 8+
- invented `nl.tile_index` (7) and `nl.tile_offset` (1)
- `tensor_copy` with `dst` in `shared_hbm` (4)

## The level-4 picture with the loop-tool session's results

| Condition | Score | Stuck on |
|---|---|---|
| No plan (baseline) | 0.62 ×4 | Tiling: a tile with more than 128 rows |
| Plan given, Qwen writes the whole kernel (this file) | 0.30 ×1 | NKI API: invented tile-index helpers, wrong buffer for `tensor_copy` |
| Plan given, tool writes the loops (`results/2026-10-10-rishabh-loop-tool.md`) | 0.50 ×4, 0.30 ×1 | NKI API: indexing `nl.sbuf[0]`, then invented `fill`, NaN output |
| Plan from diagnostic feedback (`results/2026-10-10-rishabh-plan-diagnostic.md`) | 0/5 plans pass | Loop index confused with tile size |

## Verdict

**Reverted (not worth keeping as a level-4 improvement).** A correct plan made the score *worse*:
told to follow a plan, Qwen invented helpers to compute tile positions (`nl.tile_index`) instead of
writing the slices out. Planning is not the only wall. Under it is a second one, NKI API knowledge,
and that is where the next checker work should go. The held-back `enrich()` messages are next, as
one shared change measured against a same-code baseline: `nl.sbuf` indexing, "output never written"
instead of "uninitialised PSUM", and the level-1 invented-name mapping.
