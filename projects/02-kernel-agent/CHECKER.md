# Our checker

The checker is the artifact the judges keep. This documents what it accepts, what it rejects, and
*why* — the reasoning is the point, not just the code.

## What it accepts / rejects, and why

| Stage | Rejects | Why (what would go wrong if we accepted it) | Cost |
|---|---|---|---|
| Parse | code that does not compile as Python | a parse error blamed on the model is really the extractor's bug | ms |
| Static rules | framework calls (**alias-resolved**: `np.mean`, `xp.mean`, `from numpy import mean`), partition dim > 128, missing `@nki.jit`, wrong entry name, `.T` on an arg, `@` operator | these hand the whole operation to a framework, or cannot run on the device | ms |
| Simulate (`nki.simulate`) | exceptions, wrong shape, NaN/Inf, mismatch > tol, input mutation, HW-hazard warnings | wrong numbers, or correct-on-CPU-wrong-on-device | s |
| Fresh seeds + ragged + hostile | kernels that pass the fixed inputs by luck, or break on the ragged edge / hostile values | the fixed inputs are seeded, so overfitting to them is possible | s |
| Traffic bar (L5–7) | `bytes > k × floor`; **unmeasured transfers fail closed** | levels 5–7 are graded on bytes, and an un-instrumented data path must not read as optimal | s |
| Compile gate (`calibrate.py`) | kernels that simulate but do not compile to a NEFF | the simulator does not model SBUF/PSUM capacity or restricted-Python compile errors | ~min |
| Device (optional) | mismatch on NC 0–1 | simulator numerics ≠ hardware | ~min |

## Tolerance and why
`max |got − want| / RMS(want) ≤ 2e-2`. Justification: the test inputs are float32; the ridge the
roofline quotes is the published **bf16** figure, so the tool labels float32 verdicts *indicative*.
`<fill in: did NKI_PRECISE_FP=0 vs 1 flip any verdict? state the eps you observed>`.

## Kill matrix (mutation tests)
Run `python mutants.py` (full) or `python mutants.py --rules-only` (no device). Paste the table:

| mutant | level | caught? | message class correct? |
|---|---|---|---|
| numpy_alias | 1 | ✅ | framework (alias-resolved) |
| from_import_matmul | 3 | ✅ | framework (alias-resolved) |
| drop_jit | 2 | ✅ | @nki.jit |
| rename_entry | 3 | ✅ | no function named |
| oversized_tile | 3 | ✅ | partition 256 > 128 |
| transpose_swap | 2 | NEEDS-DEVICE | numerics |
| pool_divisor | 1 | NEEDS-DEVICE | numerics |
| matmul_no_copyout | 3 | NEEDS-DEVICE | output zeros |

(The rules-only rows are verified off-device; the numeric rows must be run in the pod and the table
updated with the real result.)

## Feedback design: verdict → instruction
Three before/after examples with the measured effect. The level-3 reshape message is the worked one:

- **Before:** "Do not reshape. Work with the shapes you were given and slice them into tiles."
- **After:** names the `lhsT [K,M] / rhs [K,N]` transposed layout and the exact single-tile
  `nc_matmul → psum → tensor_copy → dma_copy` sequence.
- **Effect:** `<fill in from --repeat on level 3, before vs after>`.

`<add two more from your taxonomy>`

## Known gaps (honest)
- Latency (layers 2–3) is not measured; every intensity number is throughput reasoning.
- The device rung of the calibration ladder is scaffolded, not wired to `nrtpy` (see plan S2).
- Reference kernels for levels 5–7 are not shipped, so those traffic bars are untested end-to-end.
