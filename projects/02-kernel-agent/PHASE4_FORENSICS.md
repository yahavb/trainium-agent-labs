# Phase 4 failure forensics

Evidence: `/tmp/trainium-kernel-dev/projects/02-kernel-agent/runs/phase4-forensics-95fed3i7`. All 432 baseline candidate records across five inferred repetitions and all 64 pilot candidates were read; every unique source was parsed and inspected through allocations and operation calls. Original logs lack repetition IDs: boundaries are inferred from a return to Level 1, round 0. No model requests were made for this investigation.

| Level | Records | Unique source strings | Best reward | Best replay verified shapes | AST-changing selected repairs | Dominant feedback categories |
|---|---|---|---|---|---|---|
| 1 | 160 | 8 | 0.300 | 0/4 | 30/35 | {'DMA_SHAPE_MISMATCH': 40, 'INVALID_API_ARGUMENT': 20, 'INVALID_BUFFER_PLACEMENT': 20, 'INVALID_API_FUNCTION': 80} |
| 2 | 96 | 13 | 0.300 | 0/4 | 3/19 | {'DMA_SHAPE_MISMATCH': 62, 'INVALID_API_FUNCTION': 1, 'OUT_OF_BOUNDS': 33} |
| 3 | 88 | 7 | 0.300 | 0/1 | 4/17 | {'INVALID_TENSOR_DIMENSIONS': 56, 'DMA_SHAPE_MISMATCH': 32} |
| 4 | 88 | 11 | 0.625 | 1/4 | 1/17 | {'INVALID_TENSOR_DIMENSIONS': 83, 'NUMERICAL_MISMATCH': 5} |

## Actual algorithms and repair failures

- Level 1: the first highest-score tie loads each pool window into a fixed 128x128 tile, then tries a matrix product and constant 0.5 scale. This does not express an average reduction. Later repairs change tiles to `(p,p)` and API/buffer names while preserving the wrong computation and scalar output transfer. A DMA fix exposes invalid matmul arguments, then wrong buffers, then invented operator APIs. Eight equally scoring sources were preserved; score alone cannot identify a mathematically superior one.
- Level 2: input is `(P,F1*F2)`, but many candidates drop the P batch dimension, allocate `(128,F2)` or `(F1,F2)`, and transpose in place or mistake matmul for transpose. Fixing a load by slicing only a small prefix loses output coverage and exposes bounds errors. Only 3 of 19 selected repairs change AST structure.
- Level 3: candidates allocate output from `lhsT.shape[1:]`, yielding a rank-one result, or use a `(1,M)` PSUM destination for an `(M,N)` product. Other candidates reuse a `(K,M)` input SBUF tile for the `(M,N)` result. The mathematical operation is often appropriate but result shape and dataflow are wrong.
- Level 4: the best kernel has a valid single-tile stationary.T@moving path, a separate result-shaped SBUF tile, and output store. Oversized whole-matrix loading prevents larger cases from reaching matmul. A tied best candidate adds K slicing after the oversized DMA, so it cannot repair the first failure. Another lower-score candidate uses overlapping K windows, over-accumulating numerical results. Only 1 of 17 selected repairs changes AST structure.

## Exact Level 4 leader replay

Exact source: `/tmp/trainium-kernel-dev/projects/02-kernel-agent/runs/phase4-forensics-95fed3i7/level4/best.py`; all three distinct highest-score ties preserved as `level4/tied-best-*.py`. Private checker files are under this artifact directory. Score 0.625 = 0.1 parsing + 0.2 static rules + 0.2 at least one successful simulation + 0.5*(1/4) verified shapes. It does not mean half the shapes passed.

| Shape | Actual verification |
|---|---|
| K=128 M=128 N=512 | Numerical, input integrity, traffic gate and hazard gate passed |
| K=256 M=256 N=1024 | AssertionError: dma_copy dst partition dimension 256 exceeds maximum 128 |
| K=512 M=128 N=512 | AssertionError: dma_copy dst partition dimension 512 exceeds maximum 128 |
| K=256 M=512 N=1024 | AssertionError: dma_copy dst partition dimension 256 exceeds maximum 128 |

All three highest-scoring ties replay at 0.625 with one verified shape. The other shapes fail during simulation at DMA, before numerical comparison or memory-traffic validation. Level 4 has no excess-traffic score bar; traffic analysis is not what blocks this kernel. The algorithm generalizes only if loads, output loops, result allocation and contraction accumulation are tiled together. No historical kernel was used as an independent generated cold-start candidate.

## Pipeline and selection

Every baseline candidate parses and satisfies the static rules. Most failures are during simulated allocation/ISA execution, not import or traffic checking. Some Level 4 candidates reach numerical comparison and fail. Reward-only stable ordering picks the first highest-reward candidate; the controller repairs its latest selected kernel, even when that kernel regresses below the historical best. Repeated equal-score failures can therefore anchor an invalid dataflow. The exact checker feedback and code are already present; limited model knowledge and overly local repair scope both contribute.

The pilot preserves per-shape results: no candidate verifies any shape. All 64 pilot responses report finish_reason=stop and zero truncations. The old baseline has no endpoint finish reason/token usage fields; its complete source and console contain no observed truncation warning, but truncation cannot be conclusively excluded from those logs. AST changes show structural edits, not proof of successful repair. See `pilot-source-analysis.json` for selected source transitions and `findings.json` for categories/stages/cases.

## Independently verified targeted diagnostics

Installed NKI 0.6.0 `_math_funcs.py`: sum/max compute the reduced shape with keepdims=False by default and allocate SBUF from it. Reducing all free axes can therefore produce a prohibited rank-one tile even with no explicit 1D allocation. Keepdims=True preserves free singleton axes; choose this only when appropriate for consumers.

Installed simulator `matmul.py:219-223`: matmul produces an MxN result and then calls result.reshape(dst.view_shape). An invalid PSUM destination can produce a reshape error without a source reshape call. Shape-aware guidance identifies the actual matmul call and destination rather than repeating blanket "Do not reshape" advice.

Only input shapes from the actual checker-reported failed case are bound to the candidate entry arguments. Static literal arithmetic/slices are resolved conservatively; loop values, branches, arbitrary calls and unresolvable expressions are labeled unknown.

Reproduce forensics without inference: `NEURON_PLATFORM_TARGET_OVERRIDE=trn2 python -B phase4_forensics.py` (creates a new private directory).

The literal Level 3 error `cannot reshape array of size 32768 into shape (1,64)` is present in our baseline. Its actual source contains no explicit reshape. Binding only that reported checker case derives stationary `(128,64)`, moving `(128,512)`, PSUM `(1,64)` and required product `(64,512)` at source line 21. The returned HBM allocation is additionally `(64,)`, and the result copy reuses an input tile. Exact source/feedback and coordinated diagnosis are saved in `runs/phase4-forensics-95fed3i7/level3-reshape-diagnosis.json`.
