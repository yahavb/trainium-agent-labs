# Level 3: single-tile matmul

The level-3 reference computes `lhsT.T @ rhs`. It accepts `lhsT` with shape `(K, M)` and `rhs`
with shape `(K, N)`, and returns a result with shape `(M, N)`. The left operand is already
transposed into the layout used by `nisa.nc_matmul`; the kernel must not transpose it again. The
configured level-3 shape fits one matmul tile.

## Current agent guidance

`agent.py` gives level 3 an explicit shape contract and a role-specific API card. It calls for
separate SBUF tiles for the two inputs, a float32 PSUM result, a separate result SBUF tile, and a
shared-HBM output. The intended data path is to load the operands, call `nisa.nc_matmul`, copy
PSUM to SBUF with `nisa.tensor_copy`, then store the result to HBM. Repair feedback repeats the
shape and buffer requirements when fixing allocation or dependent-copy errors.

This guidance addresses failures recorded in earlier runs: allocating the result as
`lhsT.shape[1:]` (which is `(M,)`), using incompatible copy shapes, and reusing one input tile for
both operands. It does not establish that a model-generated implementation is correct or faster.

## Recorded observations and validation

The original level-3 branch notes recorded a first baseline with 16 attempts scoring 0.30. A later
run-2 note describes the one-dimensional result allocation recurring; run 3 instead overwrote an
input tile while loading the second operand. The branch notes report that the shipped reference
passed its one configured shape and that a diagnostic kernel with a two-dimensional output
allocation passed the pod simulator. Those reports validate the reference and the described
diagnostic case; they are not a repeated end-to-end model evaluation.

The live solve-rate improvement remains unmeasured in the evidence recorded here. To measure it,
compare the same model, endpoint, prompt settings, sample count, rounds, and repeat count against
an unchanged baseline, then report the logs and solve counts.

```bash
python nkibench.py --selftest
python nkibench.py --level 3 --check reference_level3.py
python -u agent.py --level 3 --rounds 8 --samples 4 --context 8192 --repeat 5 \
  --log attempts-level3.jsonl
```

Run agent experiments sequentially. The harness writes candidate code to the shared
`/tmp/_agent_level3.py` path, so separate checkouts alone do not isolate concurrent runs.
