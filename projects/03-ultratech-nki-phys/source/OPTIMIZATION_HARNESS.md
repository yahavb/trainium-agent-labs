# Applicability And Selection Harness

The agent must distinguish two questions: can a transformation apply, and does
it improve measured throughput while preserving correctness? Applicability alone
does not establish a speedup. Keep the input if all tested candidates are slower,
incorrect, unmeasured, or invalid.

## Controlled Cases

| Input kernel | Copy removal | Scaling/update fusion | Candidates |
| --- | --- | --- | --- |
| Original baseline | Applicable | Applicable | copyfree, scale-fused |
| Copy already removed | Not applicable | Applicable | scale-fused |
| Copy removed and update fused | Not applicable | Not applicable | Keep input |
| Unknown program structure | Requires review | Requires review | No unattended execution |

These are three versions of the same contact solver, not new physics data.
The scaling candidate includes copy removal when applied to the original;
this is not an isolated scaling-only ablation. From a copy-free input it tests
the additional scaling fusion alone.

## Flow

1. Inspect the input against the finite human-reviewed AST catalog.
2. Save applicability decisions, reasons, and the input source hash.
3. Give Qwen the applicable reviewed transformation and checker feedback.
4. Check generated code, simulate, and require every physics gate to pass.
5. Measure on-device throughput against that exact input using paired timings.
6. Select the highest valid observed ratio above 1; otherwise retain the input.

`optimization_harness.py` provides the planner and measured-summary selector.
The full plane loop now saves `applicability.json`, filters its default search
through this planner, and includes the applicability reasons in Qwen's prompt.
Explicit legacy fused-subtraction experiments remain an opt-in reviewed path.
The loop still benchmarks against the original plane baseline; it does not yet
accept arbitrary input kernels. The standalone planner exports applicable source
candidates but does not run or benchmark them.

The selector accepts only trusted checker/controller summaries with matching
input hash, full correctness, stable paired timing, and an evidence location.
It is not a security boundary: a string pointing to evidence is not proof of
its integrity. Candidate code must never supply these summaries itself.

## Evaluation

Unit tests cover both techniques, scaling-only applicability, neither technique,
unknown-code rejection, fastest-correct selection, retaining a slower original,
and rejection of missing, mismatched, nonfinite, or failed evidence.
Their timing values are synthetic policy tests, not hardware performance claims.

Existing hardware evidence already shows why both decisions matter: scaling
fusion passed physics but achieved approximately 0.99703x the paired copy-free
throughput. In that comparison the correct choice is to retain copy-free,
despite scaling fusion being applicable.

Next hardware evaluation: benchmark each supported input against its own
applicable candidates, retain all attempts and raw timings, and report the
chosen kernel and gain per input. Do not reuse original-baseline ratios as
copy-free-input ratios. This catalog is a bounded agent harness, not general
optimization discovery, model weight training, or proof of global optimality.
