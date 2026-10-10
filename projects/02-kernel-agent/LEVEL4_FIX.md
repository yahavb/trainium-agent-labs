# Level 4: tiled matmul

Level 4 computes `lhsT.T @ rhs`, where `lhsT` has shape `(K, M)`, `rhs` has shape `(K, N)`, and
the result has shape `(M, N)`. The left input is already transposed for `nisa.nc_matmul`. Unlike
level 3, this level's configured shapes require tiling across M, N, and K.

## Current agent guidance

The initial and repair prompts describe separate SBUF operand tiles, one float32 PSUM accumulator
per output tile, and a same-shaped SBUF tile for copying the completed PSUM before storing to the
shared-HBM output. The accumulator must persist across the K-tile loop; each output tile must be
written to its corresponding output slice.

For level 4, the agent adds a labeled three-axis loop scaffold to the first repair after a
nonempty candidate fails. The model is asked to complete or repair the scaffold's loads, matmul,
and stores; the ordinary checker still grades the returned candidate. This is prompting help, not
a reference-kernel fallback or a guarantee of success. The attempt log records whether the
scaffold was used.

This guidance addresses earlier reported failures including a PSUM-to-SBUF copy-size mismatch,
an output loop whose integer division yielded zero iterations, and wrong M/N tile origins. The
checker feedback now includes tile coverage and slice-origin guidance for relevant level-4
failures. It does not prove that all failures are covered.

## Recorded validation and open results

The original level-4 branch notes report local Python compilation, checker self-test, and a
prompt-structure check. They explicitly say the NKI simulation portion needed verification on the
pod. The notes do not report a successful model-generated level-4 run or a measured solve-rate
improvement. Treat the target of five successful repeats as an evaluation goal, not as an achieved
result.

Run the current checkout on the pod and retain the output log:

```bash
python nkibench.py --selftest
python nkibench.py --level 4 --check reference_level4.py
python -u agent.py --level 4 --rounds 5 --samples 4 --context 8192 --repeat 5 \
  --log attempts-level4.jsonl
```

Use `analyze.py` on the generated JSONL to report the actual per-run results. A clean reference
check validates the reference kernel, not the model's ability to generate a kernel. Run agent
experiments sequentially: candidates are written to the shared `/tmp/_agent_level4.py` path, so
separate checkouts alone do not make concurrent runs safe.
