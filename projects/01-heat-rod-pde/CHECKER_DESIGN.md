# Checker interface alignment and optional diagnostics

**IMPLEMENTED — NOT TESTED.** No tests, selftests, compilation, benchmarks or
model inference were run for this revision. Static source comparisons only.

Maintained branch: `challenge1-runtime-resilience`. PR target: `master`.
Alignment reference: `b94759c18e91139a128ed260b6a7dd18ba59148d`.

The later implementation-only numerical/diagnostic refinement is documented in
[CHECKER_REFINEMENT.md](CHECKER_REFINEMENT.md). It preserves the public alignment
described here; protocol v2 belongs exclusively to the optional enhanced API.

## Public API: identical source to master

`pdecheck.py` and `checker_runtime.py` are restored in full from the reference
master revision, not merely renamed or filtered at their return boundaries.
Their signatures, return fields, feedback, parsing, numerical rules and exception
handling therefore use the same implementation as that master revision.

| Public API | Master contract retained |
| --- | --- |
| `pdecheck.extract(text)` | Original final-expression extraction |
| `pdecheck.parse(s)` | Original supported syntax and parser behavior |
| `pdecheck.check(problem, answer_text, n_points=24)` | `reward`, `parts`, `expr`, `start_error`, `feedback` |
| `checker_runtime.grade_candidate(problem, answer)` | Above fields plus `original_reward`, `original_parts`, `grading_policy`, `evaluation_error` |
| `prompt_of`, `verify`, mathematical helpers | Unchanged master implementation |

Ordinary missing/unparseable answers and unbound symbols retain the original
zero-score format and the available expression text. They are no longer mapped
to null merely because an enhanced parser marked them unscored. If the original
checker actually raises, `grade_candidate` retains master's null reward and
`evaluation_error={type, message}` shape. Its non-finite `start_error` normalization
also remains unchanged. New keys such as `scored`, `status`, `diagnostics`, or
`score_protocol` are not injected into either public API.

The existing original, improved and algorithm Agents, comparison programs, model
sampling and stopping rules remain on these master APIs. No caller migration or
new keyword argument is required. Weights, tolerances, problem definitions and
inference prompts remain unchanged.

## Why the previous version was incompatible

The prior consolidation replaced `pdecheck.check` with a subprocess adapter,
changed parser acceptance, introduced a deadline and new result fields, dropped
some original malformed-expression details, and translated unmeasured results to
null in `grade_candidate`. Matching function names alone did not preserve the
master contract. The present alignment removes those changes from the public path.

## Enhanced checks are an explicitly separate API

The useful implementation is retained rather than deleted:

- `validation.verify(problem, answer, validation_seed=9173)` explicitly requests
  supplementary physics validation.
- `validation.grade(problem, answer)` explicitly requests the optional combined
  scoring/validation report. Its result is an enhanced report, **not** the return
  format or acceptance policy of the master APIs.
- `checker_enhanced_runtime.py` owns the optional process deadline, public-field
  JSON transport, structured errors and expanded report. It does not implement
  or replace `grade_candidate`.
- `checker_scoring.py` contains the optional worker's numerical-protocol routine
  on an already safely parsed expression. It uses master mathematical helpers
  but never calls the original string parser for model text.
- `checker_worker.py`, `checker_syntax.py` and `checker_math.py` retain safe AST
  construction, bounded work, early-time residual sampling, actual left/right
  boundary conditions, jittered initial-profile quadrature and modal diagnostics.
- `checker_reference.py` remains an archival copy; it is never a runtime import.

The optional worker does not receive hidden exact solutions or generator
callbacks. It derives modal information from public boundary conditions. Current
problem generators are unchanged; bounded finite Sum syntax remains opt-in for
external problems that explicitly allow it. No unbound index is deleted.

Supplementary residual/BC checks keep the 1e-6 threshold and initial conditions
keep the problem's relative L2 tolerance. Multiple grids and early-time slices
reduce particular sampling risks. Failure records include category, condition,
severity, evidence, location and correction. Coefficient directions are derived
from projections of the public initial function, not hidden answers.

## Reporting rules

Official comparisons must obtain original checker scores from the public
`grade_candidate` API and report any explicitly requested supplementary result
separately. Optional worker reports retain their own numerical-protocol score and
validation acceptance; do not assume the safe-parser path is bit-for-bit equivalent
to master for all inputs. Do not substitute optional `validation.grade().reward`
for official scores or change an Agent's stopping rule implicitly.

No enhancements execute automatically from a public checker call. Existing
algorithm summaries that say additional validation was not run remain accurate.

## Constructed feedback examples, not test results

- A measured residual failure plus an identifiable inconsistent exponential can
  suggest re-pairing its spatial frequency and `k*frequency**2` decay.
- A right Neumann failure identifies the right slope condition; it does not
  invent a left-end failure or impose a zero-temperature condition there.
- Missing inspected initial-profile modes with passing PDE/boundaries can suggest
  additional terms, while wrong retained projections suggest fixing coefficients.
- An unbound index requests explicit binding/expansion; an optional-worker timeout
  is inconclusive, not a claim of mathematical correctness.

## Static review and remaining limitations

Static review compares both public files to the pinned master Git blobs, checks
import routing and Agent call sites, and inspects the diff for whitespace and
conflict markers. It does not execute Python or establish runtime compatibility
of the optional enhancements. Historical tests do not validate this revision.

Restoring exact master behavior also retains its existing string-parser security
limitations and unbounded in-process symbolic work. The new safe parser and
process deadline protect only explicitly invoked enhanced APIs; this alignment
must not be described as hardening the default public path.

Enhanced checks remain finite-sampling heuristics, not symbolic proofs. They can
miss localized or nonlinear features, inspect only finitely many modes, and may
reject hard expressions as inconclusive. Subprocess startup costs, numerical
precision behavior and eight-second deadlines remain unmeasured. Windows has no
hard memory cap here. Additional testing requires renewed owner authorization.

Source-identical public APIs are restored; the optional implementation is retained.
No merge into master or new performance claim is made. Overall submission status
remains NOT READY / ENGINEERING REVIEW REQUIRED for the previously recorded gaps.
