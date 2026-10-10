# Checker implementation note

**Status: IMPLEMENTED — NOT TESTED.** No tests, selftests, compilation checks,
benchmarks, inference calls, or Trainium experiments were executed for this change.
Examples below are constructed illustrations of code paths, not observed results.

Current maintained branch: `challenge1-runtime-resilience`. Checker implementation
was originally developed on `challenge1-improvements`, starting at revision:
`c3c16f65f5e05eb3af81eccd612d45f779baf592`.

## Consolidation into the maintained branch

The owner requested one development branch and PRs targeting `master`.
`challenge1-improvements` is merged into `challenge1-runtime-resilience` and is
retired after its commits are preserved. Reuse the existing master-targeting PR.

The current resilience and algorithm Agents call `checker_runtime.grade_candidate`,
which is retained with nullable unmeasured scores and `evaluation_error` reporting.
It uses the **original numerical score only**, preserving their stopping rule.
The supplementary `validation.grade/verify` APIs remain available explicitly;
they are **not automatically enabled** in these Agents. The table below describes
those optional APIs, not a change in the primary Agent's grading policy.

The shared `pdecheck.check` does gain safe parsing and process limits. Any
unmeasured result is translated to `reward=None` for the resilience API rather
than masquerading as a measured zero. The historical source snapshot under
`evidence/decay-repair/live-three-seeds/source/validation.py` is left untouched.
Neither this consolidation nor the Checker implementation was tested; historical
test evidence does not certify the combined revision.

## A. Architecture and scoring boundary

Before: `pdecheck.check` parsed model text with SymPy's string parser and evaluated
four weighted conditions. `validation.grade` guarded Python syntax, called this
checker, and ran a second sampling protocol only on a score of 1.0. A failed
second check overwrote the score without retaining the original score. Exceptions
had little diagnostic structure. Symbolic work had no process deadline.

After: the same public entrypoints send public problem fields and answer text to
a worker. An allowlisted AST builder constructs mathematical expressions without
evaluating model-generated Python. The worker checkpoints the historical
numerical score, performs supplementary checks, and returns structured evidence.
The parent terminates work exceeding eight seconds. A completed original score
survives a later supplementary timeout.

The numerical scoring routine is retained, including the original seeded 24
space/time samples, 801-point initial grid, normalization, strict `< 1e-6` PDE/BC
threshold, problem-provided initial relative L2 tolerance, and weights
`0.4/0.2/0.2/0.2`. `prompt_of` and problem generators are unchanged.

The pre-change file is preserved as `checker_reference.py`. It is an archival
reference, **not a safe entrypoint for untrusted answers**, and is never imported
by the new runtime. Its original string parser must not be used on model text.

| Field/API | Meaning |
| --- | --- |
| `pdecheck.check(...).reward` | Original numerical protocol; does not certify enhanced validation |
| `grade(...).original_reward` | Original numerical score, or null if no score was computed |
| `original_parts`, `original_feedback` | Original condition results and available historical feedback |
| `validation.accepted` | All supplementary conditions passed |
| `validation.conditions` | Per-condition pass/fail/inconclusive, measurements and locations |
| `verified_reward` | Weighted sum of supplementary conditions actually established as passing |
| `grade(...).reward` | Existing Agent-facing gate: retain partial baseline score; baseline 1.0 requires supplementary acceptance |
| `solved` | Baseline 1.0 **and** supplementary acceptance |
| `scored` | Whether the original numerical score was computed; false distinguishes an API fallback zero |

Neither a null original score nor a fallback zero should be reported as a measured
mathematical failure. Aggregate comparisons must report **original score** and
**supplementary acceptance** separately and count inconclusive/timeout outcomes
separately. The unchanged benchmark program does not yet aggregate these new
fields; they are included in the unchanged Agent's nested attempt-grade logs.
Do not compare historical and new `reward == 1.0` totals as equivalent protocols.

This implementation does strengthen the **supplementary** gate, while retaining
its existing role. No original weight or tolerance was relaxed. Parser restrictions,
construction order and resource limits can change outcomes for unsupported or
numerically fragile inputs. This is numerical-protocol compatibility, not a
claim of bit-for-bit equivalence to the old executable; equivalence is untested.

## B. Changed files

| File | Change |
| --- | --- |
| `pdecheck.py` | Replace string evaluation parser; isolate public check; keep original numerical routine and inference prompt |
| `validation.py` | Preserve `checked_expression`, `verify`, `grade` signatures; delegate protected work |
| `checker_syntax.py` | Token/AST whitelist; node, depth, literal and finite-sum budgets; incomplete/unbound expression handling |
| `checker_runtime.py` | Public-field-only JSON transport, wall deadline, original-score preservation, compatible grading and JSON-safe failures |
| `checker_worker.py` | Isolated parse/evaluate lifecycle; optional POSIX CPU/address-space caps; checkpoint before enhancement |
| `checker_math.py` | Crossed residual samples, boundary evidence, initial-profile anti-aliasing, modal diagnostics and concise feedback |
| `checker_reference.py` | Unmodified pre-change checker for review; no runtime import |
| `CHECKER_DESIGN.md` | Architecture, score semantics, examples, compatibility and limitations |

Agent sampling, prompts, model configuration, calculator policy, problem generators,
test files, baseline logs and benchmark code were not changed.

## C. Verification and diagnostics

**PDE.** Retain the previous supplementary samples and add crossed spatial slices
at twelve times from `1e-12 L²/k` to `0.1 L²/k`. Inspect candidate temperatures as
well as `u_t-k*u_xx`. Small expressions receive limited algebraic expansion and
factoring; no unrestricted `simplify`, integration or `doit` is used. A failing or
near-threshold worst sample is also evaluated at 50-digit precision. Conflicting
precision is inconclusive. This does not establish a global mathematical proof.

**Boundaries.** Inspect both sides separately using each actual Dirichlet value
or Neumann slope condition. Neumann deviations retain the historical L scaling;
normalized error and representative `(x,t)` are recorded. The project schema
supports homogeneous Dirichlet/Neumann conditions. Robin or nonzero boundary data
are not implemented and must not be silently mapped to this schema.

**Initial condition.** Keep the 1601-point supplementary uniform grid, then add
independently jittered 1601- and 3201-point grids. Use the problem's relative L2
tolerance with the historical denominator fallback for zero profiles. Report
the worst local discrepancy without introducing a new pointwise acceptance rule.
Disagreeing/nonconvergent quadrature is inconclusive. Resolved finite Fourier
approximations are accepted at the stated tolerance; an infinite series is not
required. A conservative linear-frequency check avoids certifying underresolved
oscillations solely from favorable samples.

**Fourier diagnostics.** Derive bases from public L and left/right conditions,
including DD, DN, ND and NN (constant NN mode uses its actual norm). Compare at
most eight initial-profile projections on two grids. Give qualitative directions,
not target coefficient vectors. Distinguish wrong retained projections from
missing projections; only suggest truncation as the primary issue when inspected
retained projections match and the PDE/boundaries pass. This limited inspection
cannot prove that every uninspected coefficient is correct.

Explicit separated sine/cosine terms can receive a rate-versus-`k*frequency²`
diagnostic when the full residual fails. An incompatible individual spatial
factor is a warning only when a full boundary also fails: other terms might
cancel that factor's boundary contribution. General mixed modes and algebraic
identities fall back to whole-expression condition evidence instead of guessed
root causes. No string comparison to a target answer is used.

**Syntax.** Attribute access, arbitrary calls, indexing, comprehensions and Python
statements are excluded. Unknown free symbols, including `n`, are reported and
never deleted. Integral/Derivative and unfinished COMPUTE-only responses require
completion. Current problem prompts require explicit terms. An external problem
may opt into bounded `Sum(body,(n,lo,hi))` with public `allow_symbolic_sum=True`;
only increasing finite integer bounds within the expansion budget are handled.
No existing generator enables this option. Infinite, reversed, or oversized sums
are not silently rewritten.

Diagnostics include category, severity, condition, evidence, location and
correction. Categories include `PARSE_ERROR`, `INCOMPLETE_EXPRESSION`,
`UNBOUND_SYMBOL`, `PDE_RESIDUAL_ERROR`, `LEFT_BOUNDARY_ERROR`,
`RIGHT_BOUNDARY_ERROR`, `INITIAL_CONDITION_ERROR`, `FOURIER_COEFFICIENT_ERROR`,
`DECAY_RATE_ERROR`, `TRUNCATION_ERROR`, `SPATIAL_BASIS_MISMATCH`,
`NUMERICAL_VALIDATION_INCONCLUSIVE`, `DIAGNOSTIC_UNAVAILABLE`, `VALID_SOLUTION`.

## D. Illustrative feedback — not test results

- If the full residual fails and an explicit mode has an inconsistent temporal
  exponent: “The displayed sin(...) term uses a decay rate inconsistent with
  k*frequency**2. Reassemble paired spatial and temporal factors; the full PDE
  residual also fails.” Passing boundaries are listed for preservation.
- If only the insulated right end fails: “Correct the right neumann boundary:
  slope u_x must be zero. Choose modes compatible with both stated boundaries,
  including the actual rod length.” The record carries the measured deviation
  and time; no failure at the left end is invented.
- If PDE and boundaries pass but initial projections are absent while inspected
  retained coefficients match: “Compute additional coefficients until the stated
  relative L2 tolerance is met; an infinite sum is not required.”
- If a free `n` remains: “Evaluate coefficients and expand or bind each index;
  do not delete an unbound index.” No manufactured replacement answer is produced.
- If the process deadline is reached: “Checker exceeded its wall-time budget;
  simplify the expression and resubmit.” Status is inconclusive, acceptance false,
  and any checkpointed original score is retained.

## E. Compatibility assessment and static review

Public argument lists and Agent-required result keys (`reward`, `parts`, `expr`,
`start_error`, `feedback`, `validation` on a full grade) remain available. Rewards
remain numeric because the Agent ranks them directly. Extra diagnostics are
JSON-compatible and non-finite values are serialized as null. The existing
Model → Checker → Feedback → Retry loop is unchanged.

Static review covered return paths, score gating, the hidden-field transport
allowlist, unchanged weights/thresholds, and unchanged Agent/problem files. No
program execution or performance validation was performed. The original
feedback receives no generator callbacks or hidden exact-answer metadata;
its modal helpers are reconstructed from public boundary conditions, as are the
new supplementary modal diagnostics.

## F. Remaining limitations and costs

- Finite deterministic sampling can miss localized spikes, singularities, or
  adversarial nonlinear/high-frequency structures. Jittering reduces a specific
  aliasing risk; it does not eliminate numerical false positives or negatives.
- Higher precision is used on representative residual/BC samples, not every
  initial-profile quadrature point. Roundoff, underflow, and symbolic construction
  order can still affect decisions. Near-threshold behavior needs later validation.
- Algebraically equivalent forms may exceed syntax or execution budgets. Safe
  supported syntax is narrower than arbitrary SymPy. Standalone parser helpers
  have structural limits only; use public grade/verify for process isolation.
- Windows gets a wall timeout but no enforced memory cap here. POSIX caps are
  best-effort. This is not a complete hostile-code or operating-system sandbox.
- One subprocess is started per grade, and supplementary diagnostics now run on
  partial scores too. This adds CPU, startup latency, and feedback tokens. No
  model requests are added, but latency and resource costs are unmeasured.
- An eight-second cap includes imports, original scoring and enhancement. A cold
  environment or expensive diagnostic can time out. Original checkpoints survive,
  but incomplete enhancement cannot yield a validated full score.
- The original Agent/benchmark summary treats unsuccessful checks as unsolved;
  downstream analysis must use nested status/evidence to distinguish checker
  inconclusive from mathematical rejection. Summary aggregation was not changed.
- Full regression compatibility, official selftests, actual Qwen behavior,
  pass rates and hardware behavior are unverified. No accuracy or speed gain is
  claimed. Prior benchmark numbers are historical and do not evaluate this code.

## G. Delivery status

Code and this note are intended for a branch commit only; no merge to master/main.
The delivery receipt supplies the actual commit hash after push. This note is an
implementation artifact, not a new benchmark report or a completed submission
readiness certificate. Engineering acceptance remains outstanding.

**IMPLEMENTED — NOT TESTED**
