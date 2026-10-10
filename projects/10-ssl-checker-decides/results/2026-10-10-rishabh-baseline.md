# Baseline, levels 1-4, Qwen3-8B (stopped at 4 of 5 runs)

- **Who / seat:** Rishabh (+ Claude), seat 48
- **Commit measured:** `8f1ca41` (workshop repo, unmodified agent.py and nkibench.py)
- **Hypothesis:** none; this is the reference the Part 1 changes compare against.
- **Change:** none.
- **Command:** `python agent.py --all --rounds 8 --samples 4 --context 8192 --repeat 5`
- **Log:** `logs/attempts-rishabh-baseline.jsonl` (432 attempts)

## Scores

```
level 1: solved 0/4   all = [0.30, 0.30, 0.30, 0.30]
level 2: solved 2/4   all = [1.00, 0.30, 0.30, 1.00]
level 3: solved 0/4   all = [0.30, 0.30, 0.30, 0.30]
level 4: solved 0/4   all = [0.62, 0.62, 0.62, 0.62]   (run 5 also 0.62 when stopped)
```

Matches README Part 2 on levels 1, 3 and 4 (zero spread). Level 2 is 2/4 here, against the README's
4/5: within this level's noise at these run counts.

## What the attempts fail on (228 attempts, grouped by feedback)

- **Level 1: invented API names, not tiling.** `nisa.multiply` (36), `nisa.scalar_mul` (12),
  `nc_matmul(transpose_moving=)` (12). The "did you mean" message lists real names alphabetically
  (`NkiInstruction, NkiValidationError, VirtualRegister, activate2…`), which doesn't help. The right
  call is `nisa.tensor_scalar(..., op0=nl.multiply)`, as in `reference_level1.py`. This is the
  cheapest message fix in Part 1.
- **Level 2:** copy-size mismatch (19), out-of-bounds index (32).
- **Level 3:** 1-D tile (27), reshape instead of slice (16).
- **Level 4:** tile with more than 128 rows (28 of 32).

## Verdict

Use this as the "before" line. The same feedback message repeating round after round ("same failure
again") is measurable: for each message type, count how often the next attempt returns the same
error.
