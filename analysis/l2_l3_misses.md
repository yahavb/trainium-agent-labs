# Why L2 (v8.3, 7/9) and L3 (final_all) were not perfect

Logs (local copies of the seats' files): `runs/seat-117/v83/v83_L2c_s117*`, `runs/seat-119/v83/v83_L2_s119*` (run 2),
`runs/seat-116/final_all/final_all*` (96a9fc9, `--all --repeat 1`). Fresh vs repair per sample comes from the usage
log's `first` flag, joined on the code's sha1 (0 unmatched). Under MIX, round r >= 1 sends 1 repair (sample 1)
and 3 fresh first prompts.

## L2 miss 1: seat-117 `v83_L2c` (best 0.50, reached only in round 7)

| round | fresh samples | repair | the repair's checker message (first line) |
|---|---|---|---|
| 0 | 0.3 0.3 0.3 0.3 | - | (round 0: all fresh; 3 of 4 are the same kernel 84ac6b04: `dma_copy ... got src=384, dst=16384`) |
| 1 | 0.3 0.3 0.3 | 0.3 | Line 27 `local_tile.reshape((F1, F2))`: Partition dim size must be preserved, got 1 -> 3 |
| 2 | 0.1 0.3 0.3 | 0.3 | Line 26 `nisa.transpose(local_tile)`: module 'nki.isa' has no attribute 'transpose' |
| 3 | 0.3 0.3 0.3 | 0.3 | Line 26 `nl.transpose(local_tile)`: cannot reshape array of size 12 into shape (3,1) |
| 4 | 0.3 0.3 0.3 | 0.3 | Line 29 `transposed_tile[j, i] = local_tile[i, j]`: value array of shape (4,) could not be broadcast to (1,) |
| 5 | 0.3 0.3 0.3 | 0.3 | same as round 4 |
| 6 | 0.3 0.3 0.3 | 0.3 | same (line 31) |
| 7 | 0.3 0.5 0.3 | 0.3 | same (line 31) |

- L2CAT spoke once: on the round-7 fresh 0.5 ("Your output is x unchanged: no element moved. ..."), the last round, so
  nothing could act on it. It only fires on NUMERICAL MISMATCH / NON-FINITE, and no other attempt in this run ran.
- Fresh samples (rounds 1-7): 21, 12 distinct; the most common is 84ac6b04 (6 of 21), the round-0 kernel with the
  `dma_copy(dst=tile, src=x)` size bug. The same kernel is the most common fresh sample on seat-119 too (5 of 21).
- **Conclusion**: the kernel never ran. The repair chain moved from one runtime error to the next (reshape, a
  non-existent `nisa.transpose`, `nl.transpose`, then a broadcast on an element-wise copy for 4 rounds), and the
  fresh samples kept reproducing the round-0 bug. No quick fix: this is level 2's luck limit.

## L2 miss 2: seat-119 `v83_L2_s119` run 2 (best 0.50 from round 4)

| round | fresh samples | repair | the repair's checker message (first line) |
|---|---|---|---|
| 0 | 0.3 0.3 0.3 0.3 | - | |
| 1 | 0.3 0.3 0.1 | 0.3 | Line 34 `transposed_tile.reshape(F2*F1)`: 'int' object is not iterable |
| 2 | 0.3 0.3 0.3 | 0.3 | Line 33 `.reshape((F2, F1)).flatten()`: 'NkiTensor' has no attribute ... |
| 3 | 0.3 0.3 0.3 | 0.3 | Line 34 `nisa.reshape(...)`: module has no attribute ... |
| 4 | 0.3 0.5 0.3 | **0.5** | **WRONG SHAPE: returned (), reference is (32, 12). Check the output-size arithmetic, not the values.** |
| 5 | 0.3 0.3 0.3 | 0.5 | same |
| 6 | 0.3 0.3 0.3 | 0.5 | same |
| 7 | 0.3 0.3 0.3 | 0.5 | same |

- **The repair kernel from round 4 on has no `return` statement** (all 4 "returned ()" attempts; no other L2 attempt
  in the v8.x logs gets this message). The simulator returns nothing, the checker reports shape `()` and tells the
  model to check its "output-size arithmetic", which is the wrong lead: the sizes are right, the return is missing.
  The model never added it in 4 rounds.
- L2CAT spoke once: on the round-4 fresh 0.5 ("Your output is x unchanged"). It did not speak on the repair chain,
  because WRONG SHAPE is not one of its triggers.
- Checked in local nki (trn2 target): the round-4 and round-7 kernels with `return out` appended run, and are still
  wrong (NUMERICAL MISMATCH, 5.64 of RMS, 98.2% of elements; they index a 2-D tile as `tile[p, i, j]`). So the fix
  below does not solve this run by itself: it replaces a misleading message with one L2CAT can then explain.
- **Conclusion**: stuck on a misleading checker message (no return reported as a size problem), then on a wrong
  kernel underneath it.

## L3 miss: seat-116 final_all (0.30, stopped after round 3)

| round | fresh samples (sha) | repair (sha) | score |
|---|---|---|---|
| 0 | cc3c9aa6 70a5bcf6 596a588c 01f2d38f | - | all 0.3 |
| 1 | 596a588c 596a588c e7aa483f | 51fe68d6 | all 0.3 |
| 2 | 596a588c 596a588c 01f2d38f | 51fe68d6 | all 0.3 |
| 3 | 596a588c 596a588c 596a588c | 51fe68d6 | all 0.3 |

- Errors: the repair 51fe68d6 is `dma_copy(dst=lhsT, src=out_tile)`: src=32768, dst=8192, three rounds unchanged;
  the fresh 596a588c is `tensor_copy(dst=lhs_tile, src=psum_tile)`: shape (32768,) into (8192,).
- Round 0 against the per-level v8.2 L3 run on the same seat (`v82_L3_s116`, solved in round 0): the prompts have
  the same lengths (one at 4109 chars, three at 4134, in both runs), and the agent builds them from the level alone.
  Per-level round 0 got 596a588c, e7aa483f, **1d4dc712 (1.0)**, **a680640e (1.0)**. final_all got cc3c9aa6,
  70a5bcf6, 596a588c, 01f2d38f, all 0.3. Only 596a588c is in both. The same requests got different answers, so the
  difference is on the server side (batch composition, or prefix-cache state after L1 and L2 ran in the same
  process). I did not compare request bodies byte by byte; the lengths match.
- Early stop: it stopped because the repair failed identically 4 rounds running. The fresh samples were collapsing
  onto one wrong kernel (596a588c: 1 of 4 in round 0, 2 of 3 in rounds 1-2, 3 of 3 in round 3), so rounds 4-7 would
  most likely have repeated it. I cannot rule out a solve, though: 12 more fresh samples, and one of the per-level
  v8.2 L3 runs solved in round 2.
- **Conclusion**: an unlucky round 0 (the server returned different kernels for the same prompts), then the repair
  stuck on one dma_copy size error. Not fixing (per the plan).

## A one-place L2 fix (<10 min, L2 only)

In `feedback_v8.py`'s level-2 grade wrapper (`grade_cat`, under `L2CAT`): when the feedback contains
`WRONG SHAPE: returned ()`, append "Your kernel returns nothing: it never reaches a `return` statement. End it with
`return out` (the output tensor you allocated)." L1, L3 and L4 never take this path, so their requests stay unchanged.

Expected effect: it would have helped 1 of the 2 misses, and only partly. That run would move from the misleading
message to L2CAT's description of a wrong kernel (checked above); it would not be solved by this alone. The other
miss never had a running kernel. Across the v8.x L2 logs, "returned ()" occurs in this one run only (4 attempts).
Worth adding only if there is a run left to test it; otherwise, report it as a known checker weakness.
