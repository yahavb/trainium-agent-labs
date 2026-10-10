# First verified improvement — 1.857x (from the 2.00x seed)

**Result:** `evidence/agent_kernel_verified_1p86x.py` is correct on all four benchmark shapes
and moves a worst-case **1.8571x** the byte floor (seed: 2.0000x) — about a **7% byte
reduction** — verified independently at evaluation seeds 0 and 1
(`agent_kernel_verified_1p86x_seed0.json`, `_seed1.json`). Three of the four shapes now also
pass the level-5 traffic gate (1.00x, 1.44x, 1.00x); the largest shape (1.86x) remains above
the 1.60x gate, so **level 5 is not cleared**.

## Lineage (who did what)

1. `evidence/agent_kernel_repair_rounds.jsonl`, round 0: the starting kernel is the model's own
   candidate from `pilot_v2` run 1 round 8 (written by Qwen3-8B during the normal agent loop).
   It measured 1.857x on bytes but computed wrong numbers on three of four shapes — the
   stationary operand never advanced past its first K-chunk (`src=lhsT[0*K:TILE_K, ...]`).
2. The checker's **behavioral diagnosis** named that fault from the candidate's own output
   (its numbers equal chunk-0-against-sum-of-rhs), not from a static hint.
3. Round 1: the model restacked the operand on its own. Shapes 1 and 3 became **fully accepted
   at exactly 1.00x**.
4. Rounds 1–3: the model repeated the same kernel, failing the cache-stride wall on the wide
   shapes (`same number of elements: src=32768, dst=16384`).
5. A **line-precise surgical instruction** (`surgical_instruction.txt`, three exact lines
   consistent with the model's own cache structure) then rode every prompt. The model applied
   it, and round 1 of the surgical run is the valid 1.857x kernel above.

## Disclosure — read this before quoting the number

- **The final repair was line-guided: the maximal assistance used all day.** The unguided loop
  (five frozen repeats, 80 attempts) produced **zero** valid improvements. Any write-up must
  state this; the 1.857x result is an agent-authored kernel with a line-guided final repair,
  not an unguided discovery.
- 1.857x on the largest shape is a byte measurement from the **simulator, with complete
  accounting** (inputs ≥ floor, no unmeasured transfers). No device execution; label
  `simulator_verified`.
- The level-5 gate is **not** met (largest shape 1.86x > 1.60x). Levels 6 and 7 are further.

## Files

| file | what it is |
|---|---|
| `agent_kernel_verified_1p86x.py` | the verified kernel (model-authored; final repair line-guided) |
| `agent_kernel_verified_1p86x_seed0.json` / `_seed1.json` | independent re-evaluations, two seeds |
| `agent_kernel_repair_rounds.jsonl` | every repair round: evaluation, diagnosis, tokens, outcome |
| `../surgical_instruction.txt` (commit `64dab85`) | the three-line instruction, for the record |
