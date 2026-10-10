# Exploratory pilot compiler audit

Read-only audit of `/tmp/p1-improvement-pilot-20261010-1`, seat-100 core 2, 2026-10-10. No hardware tests were launched and no running sources were edited. First two records copied to `.integration-prep/debug-pilot.jsonl`; retained worker records showed a third identical failure while this audit ran.

## Observed failure

Attempts 1 and 2 have identical candidate hash `de6dda41f3e0`. Both return verdict `wrong`, `sim_ok=false`, `chip_ok=false`, with no timing or speedup. Compiler diagnostic:

> error: failed to specialize NKI kernel: Collected 1 different diagnostics: - [x1] error: unsupported expression

The exact suspicious expression is:

```python
rhs_tiles = [nl.ndarray((TILE_K, TILE_N), dtype=rhs.dtype, buffer=nl.sbuf)
             for _ in range(K // TILE_K)]
```

It is followed by `rhs_tile = rhs_tiles[k]` under `nl.affine_range`. The list comprehension is the primary likely unsupported frontend expression. This is an inference from source and the generic diagnostic, not an isolated compiler proof. Other language constructs match the accepted starting kernel.

The code still uses `for m` then `for n`, allocating and loading RHS tiles inside both loops. Thus fixing the compilation alone does not establish an optimization: RHS still reloads for every m.

## Narrow repair instruction

When a compile-stage unsupported-expression error coincides with an AST ListComp allocating NKI tiles, emit a bounded trusted instruction along these lines:

`NKI_TILE_LIST: Replace the Python list comprehension of SBUF tiles with one nl.ndarray. Keep the partition axis first: use (TILE_K, K // TILE_K, TILE_N), and access a K tile as rhs_tiles[:, k, :]. Keep the K-dependent source slice and nc_matmul inside the K loop. Do not put the K-tile count on the partition axis.`

Restrict this diagnosis to detected list comprehensions rather than mapping all unsupported expressions to the same advice. It repairs the unsupported container representation without claiming speedup. A subsequent outer-loop reuse diagnosis should move n outside m and load the distinct K slots before m.

Existing `tests/fixtures/h4_rhs_hoist.py` uses exactly this single-buffer `(TILE_K, KT, TILE_N)` layout and was previously chip-validated. It is supporting local implementation evidence; no new proof of the model candidate's corrected version has been run.

## Diagnostic retention limitation

The worker retained only the same sanitized message in `/tmp/chipboost_worker_twqed0vi/results_cwqnum3i.jsonl`. No additional line-level compiler stderr was found in retained worker files. `speedcheck.py` captures exception text up to 300 characters and removes candidate scratch directories in `finally`, so the original child stderr is unavailable after completion.

Simulation reaching compilation is not a correctness pass. `run_child` executes simulation before compilation, but the parent returns immediately on the compile error, before comparing simulated outputs to references and setting sim_ok=true. Report these candidates only as compile failures.
