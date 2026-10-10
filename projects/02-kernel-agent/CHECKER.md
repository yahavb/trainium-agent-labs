# What the kernel-agent checker actually verifies

This document describes the checks implemented in this repository and their limits. A passing
check is evidence for the scope listed below; it is not a claim that every kernel is correct or
fast on hardware.

## Checks and scope

| Check | What it does | Scope and limits |
|---|---|---|
| Python syntax | Compiles the generated source as Python before loading it. | A syntax failure can come from the model output or extraction; it does not by itself identify which one. |
| Static rules | Scans the AST for selected host-side framework operations (including recognized NumPy aliases and imported aliases), `.T` on an input, `@`, the required entry-point name and `@nki.jit`, and some literal oversized `nl.ndarray` partition dimensions. | This is a targeted rule set, not a proof that all prohibited calls, dynamic oversized tiles, or invalid NKI programs are absent. |
| Agent simulation grade | Loads the required kernel and runs it with `nki.simulate` on the registered shapes. Checks output shape and values, non-finite values, whether the input was modified, and selected simulator hardware-hazard warnings. | Covers the configured simulator, inputs, shapes, and warning strings. It does not establish correctness for every input or on physical hardware. |
| Traffic bar (levels 5–7) | Instruments recognized `nisa.dma_copy` calls and rejects a measured byte count above that level's configured multiple of the minimum traffic floor. It fails closed when the measured byte count or transfer count is zero. | Only recognized/instrumented transfers are counted; other data-movement paths may be missed. The bar gates bytes, not transfer count. The level thresholds overlap, so passing a byte threshold alone does not demonstrate a unique optimization at that level. |
| Fresh-seed calibration | `calibrate.py` runs a separate numerical check using seeds 101, 202, and 303 on that level's registered shapes. | This is a separate calibration command, not part of every normal agent round, and it checks numerical results rather than all grading gates. |
| Compile calibration | `calibrate.py` tries `nki.baremetal(kernel)` when available and otherwise falls back to `nki.simulate(kernel)`. | The current success label can say `COMPILED_OK` / `VERIFIED-COMPILED` after the simulation fallback. That label therefore does not always prove NEFF compilation. |
| Device validation | The `--device` path currently reports that an `nrtpy` run on NC 0–1 is the next step. | It is not wired to execute a device test or validate hardware results. |

`python nkibench.py --check <file>` is a useful standalone check, but it is not identical to the
agent's grading path: its report does not apply every agent gate, including the input-mutation,
hardware-warning, and level 5–7 traffic-bar checks. Also, `agent.py --all` runs levels 1–4; levels
5–7 and the held-out levels need their own invocations (`--level N` and `--heldout`, respectively).

## Numerical tolerance

The numerical comparison converts the result and reference to float64, computes
`scale = sqrt(mean(reference ** 2))` (using 1 when that value is zero), and accepts when
`max(abs(result - reference)) / scale <= 0.02`.

The repository does not establish this tolerance by a measured comparison of `NKI_PRECISE_FP=0`
and `NKI_PRECISE_FP=1`. The published bf16 roofline figure is separate from this float32 numerical
tolerance and is not evidence for why the tolerance is appropriate.

## Mutation checks

The Seat 20 output provided for `python mutants.py --rules-only` reports these static-rule results:

| Mutant | Level | Result in the supplied `--rules-only` output |
|---|---:|---|
| `numpy_alias` | 1 | Caught; message reported correct |
| `from_import_matmul` | 3 | Caught; message reported correct |
| `drop_jit` | 2 | Caught; message reported correct |
| `rename_entry` | 3 | Caught; message reported correct |
| `oversized_tile` | 3 | Caught; message reported correct |
| `transpose_swap` | 2 | Not run by `--rules-only`; numerical mutant |
| `pool_divisor` | 1 | Not run by `--rules-only`; numerical mutant |
| `matmul_no_copyout` | 3 | Not run by `--rules-only`; numerical mutant |

The supplied output proves the five static cases above passed. It does not prove the numeric
mutants were caught. Run `python mutants.py` in an environment with the NKI simulator available to
evaluate those cases; a physical accelerator is not inherently required for simulator mutations.

## Feedback design and evidence

Level 3 feedback gives the model the operand orientation and result shape for the transposed-left
matmul, then describes the intended tiled matmul, PSUM accumulation, and copy-out path. This is a
design description, not measured evidence that the feedback improves pass rate or runtime.

No controlled before/after `--repeat` results are included here. To claim an effect, compare the
same model, prompt settings, level, seeds, and repeat count before and after the feedback change,
and report the logs and pass counts. The same evidence standard applies to feedback changes at
other levels.

## Known gaps

- Physical-device correctness and latency have not been established by the current `--device` path.
- A successful simulation fallback in compile calibration must not be presented as proof of NEFF
  compilation.
- Traffic accounting recognizes instrumented DMA copies and may not include every possible data
  movement operation. The level 5–7 byte bars alone do not validate transfer efficiency or device
  performance.
- Levels 5–7 have reference kernels and validators in the repository, but that does not mean they
  have passed on hardware or in a full end-to-end agent run.
- A successful `nkibench.py --selftest` or `mutants.py --rules-only` validates only the checks
  included in those commands. Neither establishes that all seven levels solve cleanly across five
  repeated agent runs.
