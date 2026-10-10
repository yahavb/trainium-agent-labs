# Checker Acceptance And Rejection Rules

This deterministic checker is the reusable artifact. It evaluates the stated equations, not an LLM's explanation of its own output. The original implementation is `source/math_tasks.py:check`; the top-level `checker.py` exposes grading and evidence replay without changing its gates.

## Correctness

Promote the exact stored FP32 inputs to FP64 before computing:

- Spring: `reference = -(k*x + c*v)`; `magnitude = abs(k*x) + abs(c*v)`.
- Net-force: `reference = sum(forces, axis=1)`; `magnitude = sum(abs(forces), axis=1)`.

Accept only when output shape matches the reference, dtype is FP32, every output is finite, execution input readback equals the original inputs, and every element satisfies:

```text
abs(actual - reference) <= 1e-6 + 2e-6 * magnitude
```

The bound scales with contributing terms rather than the final result, so cancelling contributions do not create an unrealistic near-zero relative-error requirement. It is the unchanged experiment gate, not a universal physics accuracy guarantee. Finite FP32 arithmetic can differ from FP64; exact equality is not required.

Reject wrong signs, missing terms, deviations beyond the fixed bound, nonfinite values, wrong shapes/dtypes and mutated inputs. Feedback records maximum absolute and normalized errors; model prompts also receive the previous proposal and equation context. A compilation/API error or duplicate proposal has no measured correctness score (`null`), rather than a physics failure.

## Performance

Correctness is required before and after timing; all public cases are rechecked. Benchmark the original before and after the candidate, with five repeats, 20 warmups and 200 device samples per repeat. More than 10% baseline drift invalidates the performance reward. Stop Qwen during timing. A correct but slower candidate is logged but does not replace the best verified kernel. Throughput reward is not a correctness gate and cannot compensate for wrong output.

## Replay And Trust Boundary

`python checker.py --replay` verifies package hashes and regrades recorded outputs without executing candidate code. It checks public equation outputs against their saved original inputs; public NPZ files alone cannot independently prove input preservation. Original execution-time reports retain that check. Frozen final outputs also include input readback and are checked against separately frozen input files and source hashes by the original evaluator.

Hashes detect inconsistencies with the supplied manifest, not deliberate forgery by whoever controls both evidence and manifest. Saved timing samples are auditable records, not cryptographic hardware attestations. Replaying correctness does not rerun device benchmarks or send held-out results back to Qwen.

The checker covers only these force primitives. It does not verify robot trajectories, frictional gripping, conservation properties of a complete engine or equivalence to MuJoCo. Broader physical validation in our engine PDF is reported separately.
