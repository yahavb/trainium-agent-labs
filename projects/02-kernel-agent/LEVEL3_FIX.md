# Level 3: single-tile matmul repair

Branch: `fix/kernel-agent-level3`, based on `master` at `8f1ca41`.

In seat 21's first baseline pass, all 16 attempts scored 0.30. The selected
candidate allocated its output as `lhsT.shape[1:]`, which is the one-dimensional
shape `(M,)`. Its PSUM allocation inherited that shape. Repairs changed the
already-correct input allocations, leaving the actual source of the error
untouched. Other samples copied inputs into tiles with incompatible shapes or
reused an input SBUF tile as the result.

Level 3 now gets an explicit shape contract: inputs `(K, M)` and `(K, N)`,
result `(M, N)`. Its API card assigns separate roles to input SBUF tiles,
float32 PSUM, result SBUF and output HBM. Dimension and DMA errors explain
which allocations to repair and how dependent copies must match. Other
levels keep their prompts and feedback; scoring and tolerances are unchanged.

Validation performed: Python compilation and the checker self-test passed.
The pod's NKI simulator reproduced the one-dimensional-output failure;
correcting that output to `(lhsT.shape[1], rhs.shape[1])` in the diagnostic
kernel passed the configured shape. The shipped reference also passed 1/1.
Non-target initial and repair prompts, representative error messages and
scoring weights were compared against `master` and remained unchanged.

Run this branch in a separate checkout after other agent runs have stopped:

```bash
python nkibench.py --selftest
python nkibench.py --level 3 --check reference_level3.py
nohup python -u agent.py --level 3 --rounds 8 --samples 4 --context 8192 --repeat 5 \
  --log attempts-level3-fix.jsonl > run-level3-fix.log 2>&1 < /dev/null &
tail -f run-level3-fix.log
```

The live solve-rate improvement is **not yet measured**. Compare the five-run
solve rate against an unchanged baseline. The harness uses a shared
`/tmp/_agent_level3.py` path, so separate working directories alone do not make
concurrent agent runs safe.
