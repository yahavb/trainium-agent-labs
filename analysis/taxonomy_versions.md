# Failure modes across versions

Failed attempts (reward < 1) per failure mode, by version and level. A version's runs per level and solves are in the last rows. Modes from `scripts/taxonomy.py`; cut-off answers come from USAGE_LOG.

| mode | baseline L1 | baseline L2 | baseline L3 | baseline L4 | v7 L1 | v7 L2 | v7 L3 | v7 L4 | v7+E-div L1 | v7+E-div L3 | v7+E-div L4 | v8 L1 | v8 L3 | v8 L4 | v8.1 L1 | v8.1 L2 | v8.1 L4 | total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `copy_size_mismatch` | 80 | 71 | 45 |  |  | 16 |  |  |  |  | 2 |  |  | 1 |  | 7 | 8 | 230 |
| `invented_name` | 160 |  |  | 2 | 16 | 30 |  |  | 18 |  |  |  |  |  |  |  |  | 226 |
| `partition_over_128` |  |  |  | 110 |  |  |  | 20 |  |  | 20 |  |  | 8 |  |  |  | 158 |
| `out_of_bounds` |  | 63 | 16 | 17 |  | 20 |  |  |  |  |  |  |  | 24 |  |  |  | 140 |
| `reshape` |  | 1 | 96 | 1 |  |  |  |  |  |  |  |  |  |  | 6 |  | 4 | 108 |
| `broadcast` |  |  |  | 18 |  |  |  | 20 |  | 12 | 18 |  | 6 | 7 | 5 |  | 7 | 93 |
| `tile_1d` |  |  | 55 |  | 2 |  |  |  | 5 | 1 |  | 2 |  |  |  |  | 1 | 66 |
| `wrong_signature` | 40 | 2 |  |  |  | 12 |  |  |  |  |  |  |  |  | 2 |  |  | 56 |
| `wrong_buffer` | 40 |  |  | 2 |  |  | 7 |  |  | 2 |  |  | 2 |  |  |  |  | 53 |
| `transpose_whole_input` |  |  |  |  |  | 30 |  |  |  |  |  |  |  |  |  |  |  | 30 |
| `truncated` |  |  |  |  | 12 |  |  |  |  |  |  | 2 |  |  | 2 |  |  | 16 |
| other modes |  |  | 4 | 14 | 10 |  |  |  | 29 |  |  |  |  |  | 5 | 13 |  | 75 |
| **failed / all attempts** | 320/320 | 137/144 | 216/216 | 164/164 | 40/40 | 108/108 | 7/20 | 40/60 | 52/52 | 15/20 | 40/60 | 4/8 | 8/12 | 40/40 | 20/20 | 20/24 | 20/20 | |
| **runs solved** | 0/10 | 5/10 | 0/10 | 0/10 | 0/2 | 0/4 | 5/5 | 5/5 | 0/2 | 5/5 | 5/5 | 1/1 | 1/1 | 0/2 | 0/2 | 1/2 | 0/1 | |

**Read with care: the columns are not the same size.** The baseline is 10 complete runs per level (seat-116 and
its replica on seat-119). The later versions are what had been pulled by 16:15, and several are snapshots of
runs still going: v7 L1 1 of 2 runs finished, v7 L2 3 of 4, v7+E-div L1 1 of 2, v8 L4 1 of 2, v8.1 none
finished (v8.1 L2 run 1 had solved). Compare which modes appear and disappear, not the raw counts.
Round-0-only experiments (`*_L2r0`), the A1-A5 ablations, E-A, E-F, E-v3 and the teammates' runs are left out.

Versions: **baseline** = organizers' agent.py; **v7** = feedback_v7 (V7.md); **v7+E-div** = v8 with every switch
off (a different prompt line per sample); **v8** = that + SKELETON + L1FIX + TRUNCFIX; **v8.1** = SKELETON off,
L1FIX + TRUNCFIX + MIXSAMP + L2HINT. Files: `runs/seat-*/` as listed in the command below.

## What moved

1. **Three walls fell, each to code-form feedback.** Level 3's baseline walls are reshaping (96), 1-D tiles (55)
   and copy-size mismatches (45), with no solve in 10 runs. Under v7 they are gone: what is left is 7
   wrong-memory attempts, and every run solved in round 0. Level 4's partition-over-128 wall (110, never
   passed) still appears under v7 in round 0. v5's message now hands over the whole three-loop tiling as code,
   and every run gets past it by round 2. Level 1's invented names (160 in the baseline, the model calling
   `nisa.multiply`) drop to 16-18.
2. **Failures moved rather than vanished.** On level 2, v7 traded the baseline's index errors (out of bounds 63,
   copy size 71) for two modes the baseline never showed: invented names (30: `dtype` 16, `transpose` 10, `reshape` 4,
   none of them from PROMPT1's list of functions), and transposing all of x instead of each row's block (30,
   `transpose_whole_input`). On level 1, cut-off answers (`truncated`, 12) appeared under v7, and took 3 of 8 rounds
   and 22 of 29 minutes of the run they hit. v8's TRUNCFIX turned them into a "write it shorter" instruction.
3. **One wall came back.** v8's level 4 reproduced v7+E-div's first two rounds exactly. Then SKELETON blanked
   the slices and loop bounds of v5's tiling code (`affine_range(<…>)`), and the model sat on out-of-bounds
   errors for six rounds (24). It solved 0 of 1 finished runs, against 5/5 without SKELETON.
4. **What is left at the end.** Level 4's round-0 broadcast mismatch is in every version (18, 20, 18, 7, 7). It is
   cleared by repair, not prevented. Level 1 is still the hardest: one solve in all of today's logs (v8,
   analysis/recovery_v8_L1.md), against 0 for the baseline, v7 and v7+E-div.

## How SKELETON broke level 4 (the v8 L4 column)

Round by round, the attempt each round carried forward (`runs/seat-118/v8_L4/v8_L4.jsonl` and
`runs/seat-118/Ediv_L4/Ediv_L4.jsonl`, same seat, same E-div sampling; v8 adds SKELETON, L1FIX, TRUNCFIX):

| version | run | rounds |
|---|---|---|
| v8 | 1 | r0 0.30 `broadcast` → r1 0.62 `partition_over_128` (message blanked) → r2 0.30 `out_of_bounds` → r3 0.30 `out_of_bounds` → r4 0.30 `out_of_bounds` → r5 0.30 `out_of_bounds` → r6 0.30 `out_of_bounds` → r7 0.30 `out_of_bounds` |
| v8 | 2 | r0 0.30 `broadcast` → r1 0.62 `partition_over_128` (message blanked) |
| v7+E-div | 1 | r0 0.30 `broadcast` → r1 0.62 `partition_over_128` → r2 1.00 `solved` |
| v7+E-div | 2 | r0 0.30 `broadcast` → r1 0.62 `partition_over_128` → r2 1.00 `solved` |
| v7+E-div | 3 | r0 0.30 `broadcast` → r1 0.62 `partition_over_128` → r2 1.00 `solved` |
| v7+E-div | 4 | r0 0.30 `broadcast` → r1 0.62 `partition_over_128` → r2 1.00 `solved` |
| v7+E-div | 5 | r0 0.30 `broadcast` → r1 0.62 `partition_over_128` → r2 1.00 `solved` |

Rounds 0 and 1 are the same in every run of both. The fork is the message sent after round 1, v5's "Tile all three
dimensions at once" code. With E-div alone it arrives whole:

```text
Replace everything between the def line and the return with this code, so the return stays `return out`:

    out = nl.ndarray((lhsT.shape[1], rhs.shape[1]), dtype=lhsT.dtype, buffer=nl.shared_hbm)
    TK = min(128, lhsT.shape[0])
    TM = min(128, lhsT.shape[1])
    TN = min(512, rhs.shape[1])
    for m in nl.affine_range(lhsT.shape[1] // TM):
        for n in nl.affine_range(rhs.shape[1] // TN):
            psum_tile = nl.ndarray((TM, TN), dtype=nl.float32, buffer=nl.psum)
            for k in nl.affine_range(lhs
```

Under SKELETON the same message arrives with every shape and loop bound blanked:

```text
Replace everything between the def line and the return with this code, so the return stays `return out`:

    out = nl.ndarray(<…>, dtype=lhsT.dtype, buffer=nl.shared_hbm)
    TK = min(128, lhsT.shape[0])
    TM = min(128, lhsT.shape[1])
    TN = min(512, rhs.shape[1])
    for m in nl.affine_range(<…>):
        for n in nl.affine_range(<…>):
            psum_tile = nl.ndarray(<…>, dtype=nl.float32, buffer=nl.psum)
            for k in nl.affine_range(<…>):
                lhs_tile = nl.ndarray(<…>, dtype=lhsT.dtype
```

v8's round-2 kernel shows what the model filled in. It wrote `nl.affine_range(0, lhsT.shape[1], TM)` as if it
were Python's `range(start, stop, step)` and used `m`, `n`, `k` as element offsets (`lhsT[k:k+TK, m:m+TM]`).
It also indexed `rhs` the wrong way round, `rhs[n:n+TN, k:k+TK]` on a (K, N) tensor. The result is
"Out-of-bound access ... on dimension 0: index range [0, 511]". The same error carried rounds 2 to 7 of run 1.
Our snapshot has run 1 complete and run 2 at round 1; executor 1 reports that both runs ended at 0.62. Without
SKELETON, all five runs solved in round 2. The final candidate, v8.2 (96a9fc9), runs with SKELETON=0.

## Reproduce

```bash
R=runs   # the pulled logs
python scripts/taxonomy_versions.py \
 baseline=$R/seat-116/latest/projects/02-kernel-agent/attempts.jsonl,$R/seat-119/replica/attempts.jsonl \
 v7=$R/seat-119/Ev7_L1/Ev7_L1.jsonl,$R/seat-116/Ev7_L2/Ev7_L2.jsonl,$R/seat-117/Ev7_L3/Ev7_L3.jsonl,$R/seat-118/Ev7_L4/Ev7_L4.jsonl \
 v7+E-div=$R/seat-117/Ediv_L1/Ediv_L1.jsonl,$R/seat-119/Ediv_L3/Ediv_L3.jsonl,$R/seat-118/Ediv_L4/Ediv_L4.jsonl \
 v8=$R/seat-116/v8_L1_s116/v8_L1_s116.jsonl,$R/seat-119/L2r0/v8_L3.jsonl,$R/seat-118/v8_L4/v8_L4.jsonl \
 v8.1=$R/seat-117/v81_partial/v81_L1_s117.jsonl,$R/seat-118/v81_partial/v81_L1_s118.jsonl,$R/seat-116/v81_partial/v81_L2.jsonl,$R/seat-119/v81_partial/v81_L4.jsonl \
 -o analysis/taxonomy_versions
```

`transpose_whole_input` counts a level-2 failure whose case label is `shape=(P, F) as AxB` and whose error is
either "Partition dim size must be preserved" or an index on axis 1 that runs to P-1. Executor 1's figure (81 of
134 level-2 repair failures in v7 + baseline) used feedback_v8's broader L2HINT triggers: any axis-1
out-of-bounds, a changed partition size, or any dma_copy element mismatch. On the files here that rule gives 67
of 140 (baseline seat-116 + v7), and the strict one 30 of 140 (all in v7). The denominators differ, most likely
because executor 1 used an earlier v7 snapshot; we could not reproduce 81/134 exactly.
