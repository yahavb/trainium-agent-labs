# Checker numerical evidence and diagnostic refinement

**IMPLEMENTED — NOT TESTED.** This revision implements Checker-only changes
under the owner's explicit prohibition on tests, inference and benchmarks.
No Python imports, compilation, constructed-case execution or model calls were
performed. Benefits below are intended behavior established by code inspection,
not measured runtime or correctness improvements.

Branch: `challenge1-runtime-resilience`; PR: #5 targeting `master`.
Starting revision: `a2a1888dff991b80bcdd88f64934f4d27a40935b`.
Public compatibility reference: `b94759c18e91139a128ed260b6a7dd18ba59148d`.

## A. Architecture and compatibility

The public route remains Agent → `checker_runtime.grade_candidate` →
`pdecheck.check`. Both public source files remain identical to the pinned master.
No problem generator, model, prompt, calculator, sampling, attempt budget,
acceptance threshold, public return schema or default stopping rule is changed.
The original score weights remain equation=.4, each boundary=.2, IC=.2.

Enhancement is explicit: `validation.verify` → bounded subprocess → safe AST
parser → four condition checks → flushed core result → optional modal diagnostics.
`validation.grade` remains a separate opt-in combined report; its supplementary
veto policy is not an official score. Obtain actual original Checker scores from
the public API and report extra validation acceptance separately. The worker's
safe-parser numerical-protocol score is not guaranteed identical to the public
string-parser score for every input.

Core records include `validation_complete`; optional analysis completion is
reported independently as `diagnostics_complete`. On a late worker timeout or
failure, retain a fully emitted core record and attach `DIAGNOSTIC_UNAVAILABLE`
and `worker_issue`. If required checks never complete, return inconclusive,
never an inferred pass. Failed or inconclusive conditions remain such regardless
of diagnostic availability. This preserves existing completed mathematical
evidence; it does not relax any condition.

## B. Static root causes and changes

| File | Observed weakness | Implemented change | Cost / possible regression |
|---|---|---|---|
| `checker_math.py` | Floating near-zero residuals skipped precision review | Up to three 50-digit point audits, including worst sample and endpoints of the sample array; disagreement is inconclusive | Additional bounded symbolic work; some expressions may reach the existing deadline |
| `checker_math.py` | Boundary residual coordinates rounded L before evaluation | Substitute exact public 0/L into boundary expression first; use actual Dirichlet value or L-scaled Neumann derivative | Temperature-domain sampling still uses floating coordinates |
| `checker_math.py` | An exception while constructing derivatives erased other condition evidence | Construct and guard each PDE/BC expression independently; keep exception details | Public f/scale failures can still prevent all checks |
| `checker_math.py` | Squaring large initial profiles could overflow | Scaled relative-L2 norm computation; explicitly reject non-finite subtraction as inconclusive | Floating quadrature remains approximate |
| `checker_math.py` | Near-tolerance or aliased IC results could appear decisive | Record grid-disagreement estimate; overlapping threshold is inconclusive; conservative oscillation estimate includes products/powers; unresolved nonlinear oscillation cannot certify an unequal IC | More inconclusive answers; disagreement is an estimate, not a rigorous error bound |
| `checker_math.py` | Largest IC discrepancy only reported at existing nodes | Refine 17 points in the worst sampled neighborhood | Diagnostic only; no additional L-infinity acceptance criterion; unseen narrow spikes remain possible |
| `checker_math.py` | Fixed 2% projection tolerance could label incorrect coefficients matched | Use tolerance-aware diagnostic precision; classify unstable projections unresolved; withhold truncation-only suggestion if any inspected projection is unresolved | Stricter diagnosis may report uncertainty rather than a modal cause |
| `checker_math.py` | Explicit cancelled terms could confuse decay diagnosis | Group identical spatial/exponential factors and omit zero combined amplitude; report observed multiple rates separately | Does not resolve all equivalent trigonometric identities or establish a unique cause |
| `checker_math.py` | Repeated numerical compilation and one shared modal exception guard | Worker-local 32-entry compilation cache; independently guard basis/decay/projection diagnostics; include locations in feedback | No measured speed benefit; cache lifetime is one worker only |
| `checker_worker.py` | Optional diagnostic failure could erase completed physics | Emit flushed `validation_core` before diagnostics; isolate late exception records | Additional small JSON checkpoint |
| `checker_enhanced_runtime.py` | Worker failure superseded all validation output; escaped Unicode could exceed input cap | Preserve only explicitly completed core checkpoints; expose worker issue; consistent Unicode JSON and 100000-character transport cap | A core pass may coexist with an optional-analysis timeout, explicitly reported |
| `checker_syntax.py` | Numeric constructor strings and signed/composite exponents bypassed literal bounds | Bound decimal magnitude/negative exponent, numeric literal nesting, and constant power exponents; permit simple fractional exponent denominators | Unsupported expensive numeric exponent forms return inconclusive; standalone parser is not a process sandbox |

Public `pdecheck.py`, `checker_runtime.py`, `checker_scoring.py`, `validation.py`,
Agents and historical experiment artifacts are unchanged in this revision.
`CHECKER_DESIGN.md` links this supplement; `submission/READINESS.md` records the
implementation-only status. Prior unrelated local documentation drafts are not
included in this commit.

## C. Mathematical acceptance and bounded work

Supplementary PDE residual is `u_t-k*u_xx`; normalization retains the public
initial-profile scale. Residual and boundary threshold remains strictly `<1e-6`.
Each boundary uses its actual public type. IC uses the problem's original
relative-L2 tolerance, on uniform 1601, jittered 1601 and jittered 3201 grids;
when f is zero the absolute-L2 convention remains. A finite Fourier approximation
can pass without requiring an infinite series if its sampled IC error meets the
stated tolerance and other conditions pass. Uncertainty does not become success.

The optional worker still has the existing eight-second wall deadline and
best-effort POSIX seven-second CPU / 2 GiB address-space caps. No longer timeout
is introduced. Windows has no equivalent hard memory limit in this implementation.
Parser bounds remain 12000 source characters, 1800 AST nodes, nesting 48,
64 finite Sum terms. Finite Sum is supported only with the public opt-in flag;
unbound n is rejected without deletion. Unfinished Integral/Derivative is not
an assembled final expression. No raw model-text eval or parse_expr is used in
the optional parser. Original master parser limitations remain on the public path.

All modal evidence derives from public f, L, k and BCs. No hidden exact solution,
series_answer, generator basis callback or benchmark lookup is supplied to the
worker. Only the first eight projections are inspected; no all-mode proof or
complete coefficient vector is claimed.

## D. Structured feedback taxonomy

Each diagnostic has category, condition, severity, evidence, location and
correction. Categories retained include PDE_RESIDUAL_ERROR,
LEFT_BOUNDARY_ERROR, RIGHT_BOUNDARY_ERROR, INITIAL_CONDITION_ERROR,
FOURIER_COEFFICIENT_ERROR, TRUNCATION_ERROR, SPATIAL_BASIS_MISMATCH,
DECAY_RATE_ERROR, UNBOUND_SYMBOL, INCOMPLETE_EXPRESSION, PARSE_ERROR,
NUMERICAL_VALIDATION_INCONCLUSIVE, DIAGNOSTIC_UNAVAILABLE and VALID_SOLUTION.
Newly distinguished observations: FOURIER_DIAGNOSIS_INCONCLUSIVE and
MIXED_MODE_ERROR. Modal direction/basis warnings are scoped observations,
not assertions of a unique root cause. Passing conditions are named so a repair
can preserve them. Failure locations appear in textual and structured feedback.

## E. Constructed examples — NOT TEST RESULTS

These examples illustrate intended reasoning only; none was executed.
For L=k=1, DD boundaries and public f=sin(pi*x):

| Candidate / situation | Intended evidence and correction |
|---|---|
| `sin(pi*x)*exp(-pi**2*t)` | A mathematically consistent separated candidate; execution would still need all sampled checks to establish acceptance |
| `sin(pi*x)*exp(-2*pi**2*t)` | Nonzero PDE residual; explicit decay observation suggests pairing rate with k*frequency**2 |
| `0.99*sin(pi*x)*exp(-pi**2*t)`, tol=0.005 | PDE/BCs are mathematically satisfied but IC relative L2 is 0.01; correct amplitude rather than simply append modes |
| `sin(2*pi*x)*exp(-4*pi**2*t)` | Wrong initial projection although PDE/BCs are satisfied; missing/spurious projections are not proof of truncation alone |
| Public DN boundary and integer sine mode | Right Neumann slope must be checked; do not replace it with a right zero-temperature requirement |
| Bare unbound n, unfinished Integral or incomplete parentheses | Structural rejection/repair guidance, without silently deleting indices |
| Disagreeing quadrature estimates straddle tol | Inconclusive numerical validation; no fabricated pass or uniquely diagnosed coefficient cause |
| Completed core result followed by modal timeout | Preserve core condition outcomes; expose incomplete diagnostics and worker timeout separately |

## F. Evidence and limitations

This revision has **zero test executions and zero real-model calls**. Static
review checks source diffs, public-file identity and unchanged call routing.
No new solve rate, latency reduction, aliasing guarantee, regression result or
statistical improvement is claimed. Earlier real-model runs on a4b58a5-equivalent
Python code did not invoke this optional Checker and cannot validate this change.

Finite sampling can miss localized singularities or oscillations, including
nontrigonometric high-frequency features. Three precision points are not a
domain-wide proof. Grid disagreement is only a numerical uncertainty estimate.
The frequency estimate is conservative for inspected trigonometric structures,
not a universal spectral bound. Projection precision can suppress useful modal
diagnosis. Optional analysis can exceed its budget; required checks can too,
in which case no complete core record exists and acceptance is withheld.
Computational cost and platform behavior remain unmeasured.

## G. Delivery status

Checker task: **IMPLEMENTED — NOT TESTED**. Existing master PR #5 is updated,
not merged. Overall Project 1 remains **NOT READY / ENGINEERING REVIEW REQUIRED**
because historical required feedback logs are incomplete and the current optional
implementation has no authorized runtime verification. The Git commit containing
this document is the implementation revision; the handoff provides its hash.
Next authorized step should exercise parser limits, exact BCs, precision
disagreement, projection uncertainty and both pre-core/post-core worker timeouts
before connecting any supplementary result to Agent feedback.
