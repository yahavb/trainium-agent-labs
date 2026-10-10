# Level 2: per-row 2D transpose

Level 2 takes two arguments: `x` with shape `(P, F)` and `shape2D=(F1, F2)`, where
`F = F1 * F2`. It transposes the two free dimensions within each partition row and returns a
`(P, F)` tensor with the input dtype. The partition axis is preserved.

## Current agent guidance

The level-specific prompt and repair feedback in `agent.py` require the two-argument entry point,
derive sizes and offsets from the arguments, and use separate input and output SBUF tiles plus a
returned shared-HBM output. The current prompt describes copying columns between SBUF views with
`nisa.tensor_copy`, then storing the completed tile with `nisa.dma_copy`. It also specifies Python
`min` for tile-row sizing; `nl.min` is a tensor reduction and is not an integer-bound helper.
Repair feedback covers the positional-argument, partition-stride, DMA-size, and scalar-index
failures encountered in earlier attempts.

These are generation and repair instructions, not a guarantee that a model-generated candidate
will pass. `agent.py --all` includes level 2, but the held-out tests and repeated model outcomes
are separate evidence.

## Recorded observations and validation

The original level-2 branch notes recorded a first baseline with 28 attempts scoring 0.30 or 0.50
and no complete pass. Later notes describe a baseline run that solved all four shapes without the
patch; that is evidence of run-to-run variance, not proof that the patch caused an improvement.
The notes also record failures from using `nl.min` as an integer helper, copying mismatched tile
shapes, dropping the required second function argument, and attempting an unsafe in-place
rectangular transpose.

The branch notes report that the shipped level-2 reference passed all four configured shapes and
that a diagnostic kernel using Python `min` passed those shapes in the pod simulator. They also
report replaying a saved run-5 candidate to confirm that its scalar-index error reached the new
feedback. These are recorded checks, not a current-device run or an end-to-end model solve-rate
measurement. No controlled before/after solve-rate result is available here.

Run the current checkout sequentially on the pod to collect fresh results:

```bash
python nkibench.py --selftest
python nkibench.py --level 2 --check reference_level2.py
python -u agent.py --level 2 --rounds 8 --samples 4 --context 8192 --repeat 5 \
  --log attempts-level2.jsonl
```

Compare the five-run solve rate using the resulting log and `analyze.py`. Do not run two agent
experiments concurrently in separate checkouts: `grade()` writes candidates to the shared
`/tmp/_agent_level2.py` path.
