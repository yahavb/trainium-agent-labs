# H2: the loop tool's step list in the normal whole-kernel agent (level 4)

- **Who / seat:** Rishabh (+ Claude), seat 48
- **Commit measured:** uncommitted working tree on 8f1ca41. md5s on seat 48: `agent.py` bf14b521 (messages
  v2.1 + `--step-list`), `nkibench.py` 68e795d5, `plan_check.py` 181b7a09, `stress.py` 1461e099.
- **Hypothesis:** the step-list wording that took the loop tool from 0/5 (loose) to 2/5 (unstaged) and
  5/5 (staged) also helps whole-kernel `agent.py`, with no plan and with the plan given.
- **Change:** `agent.py --step-list` appends `agent.step_list_note()` to the level-4 first prompt and to
  every repair prompt (it goes in as `plan_note`). Exact text:

  > Write it in these steps. For each output tile, allocate one PSUM tile `acc` before the k loop; inside
  > the k loop, lhsT_slice and rhs_slice are this step's slices of lhsT and rhs, and out_slice is this
  > output tile's slice of the result. Inside the k loop: allocate SBUF tiles for lhsT_slice and
  > rhs_slice, dma_copy each slice into its tile, then accumulate their product into acc with
  > nisa.nc_matmul. Each allocation goes directly above its load. After the k loop: move acc out to
  > out_slice (acc is in PSUM; out_slice is in HBM).

- **Commands:**
  - A: `python agent.py --level 4 --step-list --rounds 8 --samples 4 --context 8192 --repeat 3 --log attempts-step-list.jsonl`
    (stopped after 2 runs, at the coordinator's request, to free the seat)
  - B: `python agent.py --level 4 --step-list --plan-file reference --rounds 8 --samples 4 --context 8192 --repeat 1 --log attempts-step-list-plan.jsonl`
- **Logs:** logs/attempts-rishabh-step-list.jsonl (A, 52 attempts), logs/attempts-rishabh-step-list-plan.jsonl (B, 32 attempts)

## Scores

```
A  no plan:    before (baseline, 8f1ca41)    level 4: solved 0/4  all = [0.62, 0.62, 0.62, 0.62]
               after  (--step-list)          level 4: solved 0/2  all = [0.30, 0.30]   (6 and 7 rounds, stopped on repeats)
B  plan given: before (plan given, 06805f2d) level 4: solved 0/1  all = [0.30]
               after  (--step-list)          level 4: solved 0/1  all = [0.30]         (all 8 rounds)
```

Runs: A 2, B 1 (cut from 3 each on the coordinator's deadline). Level 4 has given the same score on every
whole-kernel run so far, and A's two runs agree (0.30, 0.30). Sampling here is **not** fully greedy: in A,
round 0 returned 4 distinct kernels in run 1 and 3 in run 2, and the two runs' trajectories differ from
the first attempt on (9 of 13 rounds had 4 byte-identical samples). B had 6 of 8 rounds byte-identical.
No kernel scored above 0.30, so there was nothing to stress-test.

**Confound:** the "before" lines come from older agent.py builds (baseline: unmodified 8f1ca41;
plan-given: 06805f2d). This run's agent.py also carries messages v2.1. Messages v1 left level 4 at 0.62
(results/2026-10-10-rishabh-api-messages.md), but v2.1 has not been measured on whole-kernel level 4
without `--step-list`. So the A drop from 0.62 to 0.30 is "step list + v2.1", not the step list alone.
The failures below are prompt-shaped, though: the kernels use the step list's names and drop the
output-tile loops.

## Failure grouping (feedback per attempt, by error type)

**A, no plan (52 attempts, 2 runs).** Every attempt parses and passes the rules. None runs.
| n | error |
|---|---|
| 16 | `dma_copy requires HBM or SBUF tensors, got src=psum, dst=shared_hbm` (run 1, rounds 2-5: 4 rounds of one identical kernel) |
| 16 | `shape mismatch: value array of shape (65536,) … (16384,)` (run 2, rounds 3-6, identical) |
| 12 | out-of-bound access, dim 0, index range [1, 128] (k steps by 1 over K with `k:k+128` slices) |
| 7 | `tensor_copy dst must be in ['sbuf','psum'], got shared_hbm` |
| 1 | reshape |

Top failure: writing the whole `(M, N)` PSUM `acc` back with `dma_copy(dst=out, src=acc)`. That message
has **no hint attached** (no "tensor_copy psum→sbuf first"), and the identical kernel came back 4 rounds
running. Baseline level 4, for contrast: 63/76 "tile with more than 128 rows", 63/76 attempts run on
1 shape.

**B, plan given (32 attempts, 1 run).** Every attempt parses and passes the rules. None runs.
| n | error |
|---|---|
| 12 | `tensor_copy dst must be in ['sbuf','psum'], got shared_hbm` (writes `tensor_copy(dst=out_slice, src=acc)`) |
| 4 | `tensor_copy src … got shared_hbm` |
| 4 | `nl.sbuf … are buffer KINDS` (`tensor_copy(dst=nl.sbuf, src=acc); dma_copy(dst=out_slice, src=nl.sbuf)`) |
| 12 | invented names: `nl.index_map` 8, `nl.literal` 2, `nl.ceil_div` 1, `nl.symbolic` 1 |

Top failure: the write-back. "acc is in PSUM; out_slice is in HBM" says where the data lives but not that
it needs an SBUF tile in between, so the model goes straight from PSUM to HBM. The plan-given baseline's
failures were `nl.tile_index` (8) and the 65536-vs-16384 shape mismatch (16). With the step list, the
invented helpers change names but are still there (12).

## Did the kernels follow the steps? (static check of every attempt)

| | A step list | B step list + plan | baseline | plan given |
|---|---|---|---|---|
| has output-tile loop (`M //` or `N //`) | 0/52 | 8/32 | 0/76 | 28/28 |
| k loop steps by 128 (`K //`) | 0/52 | 8/32 | 0/76 | 28/28 |
| PSUM `acc` allocated per output tile, directly before the k loop | **0/52** | **29/32** | 0/76 | 0/28 |
| every SBUF allocation directly above its own load | **0/52** | **0/32** | 0/76 | 0/28 |
| uses `lhsT_slice`/`rhs_slice`/`acc` names | yes, all | yes, all | — | — |

- **A:** the model adopts the vocabulary (`acc`, `lhsT_slice`, `rhs_slice`, `out_slice`) and the k loop,
  but the step list never says how many output tiles there are or how long a k step is (the loop tool's
  skeleton supplied that). So the model writes one k loop `for k in nl.affine_range(K)` with `k:k+128`
  slices, no m/n loops, and one whole `(M, N)` PSUM `acc` at the top. That runs on no shape: 0.30, worse
  than the baseline's single-big-tile kernel that passes the 128x128x512 shape (0.62).
- **B:** with the plan, the structure is right in its last rounds: m/n loops, `acc = (128, 512)` PSUM
  per output tile before the k loop, the slices, dma loads, nc_matmul accumulating. The model always
  allocates both SBUF tiles first and then does both loads ("directly above its load" was never
  followed, in either condition, but the simulator doesn't care).
- **Counterfactual (one offline check, not a score):** B's final kernel, with only its write-back
  replaced by a named SBUF tile (`res = nl.ndarray((128,512), …, buffer=nl.sbuf)`; `tensor_copy(res, acc)`;
  `dma_copy(out_slice, res)`), passes **2/4 shapes** in `nkibench.py --check`. The other two fail on the
  model's `M, K = lhsT.shape` (lhsT is K×M), which only shows up when M ≠ K. So B was 2 bugs from solving.

## Verdict

**Not kept as an improvement: H2 is not supported.** The step list doesn't transfer on its own. Without
a plan it hurts (0.62 → 0.30 in 2/2 runs, confounded with messages v2.1): it brings in the k-loop vocabulary
without the loop counts, and the model drops the output tiling. With the plan given, the score is
unchanged (0.30), but the kernels are structurally much closer (29/32 per-tile PSUM acc, against 0/28), and the
remaining wall is the PSUM→HBM write-back. That's the same failure the staged loop-tool ablation hit with
the loose B wording.

Next (needs the agent.py/nkibench.py owner, not done here):
1. The B line should name the staging tile: "tensor_copy acc into one named SBUF tile, then dma_copy that
   tile to out_slice" (this is v2.2's queued "one named tile").
2. `dma_copy requires HBM or SBUF tensors, got src=psum` needs an `enrich()` hint (tensor_copy PSUM→SBUF
   first). It is bare today and repeated 4 rounds unchanged.
3. The `nl.sbuf … are buffer KINDS` hint ends with "Do the same for rhs_slice, out_slice": out_slice must
   be excluded (v2.2's queued item, seen again here).
4. The plan text doesn't stop `M, K = lhsT.shape`. A `--plan-file` note "lhsT is (K, M)" would cover it.
