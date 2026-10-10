# Physics-Feedback Qwen Experiment

## Observed Baseline

Seat-260 NKI simulator run: data/grip-sim-smoke-v1.
16 certified public gripping snapshots; batch 2; 32 updates; one execution repeat.
Trusted grade grading-793899db230c467aa8c34b1d2efe23e9: 0/16 physics passes.
The saved simulator outputs exactly match independent CPU fixed-update outputs
(maximum absolute difference 0). All inputs unchanged; padded rows zero;
nonnegative edge coefficients; before/after outputs identical.
Normalized projected residual range: 0.0082506384..0.782813523 (limit 1e-4).
Normalized edge-force error range: 0.0556245409..0.973530269 (limit 1e-4).
This supports insufficient convergence at 32 updates, not an NKI simulator
implementation mismatch. It does not prove actual device correctness.
No hardware throughput is eligible for this failing baseline.

The earlier CPU 8192-update baseline passed 8/16 distinct cases, five repeats
each (40/80). This is a different update budget; preserve both records rather
than replacing the weaker baseline or confusing repeated runs with new data.

## Feedback Preparation

The original next-prompt.txt was 30,797 characters, with duplicate before/after
diagnostics. Compact feedback summarizes every observed metric's worst value per
case across repeats/phases, counts gate failures, and retains three representative
failed cases. Full per-case reports are unchanged and remain available in the
grading logs. Suggested causes are hypotheses, not automatic diagnoses.
The formatter is deterministic CPU code, not an LLM and not physics authority.
Old grading artifacts are preserved; regenerate grading to obtain a new compact
next-prompt.txt. This change does not alter the frozen physics gates or scores.

## First Generation

First actual Qwen attempt: qwen-grip-e710085905c7472382496ca630c9cc80.
Generation took 52.0199 seconds; prompt 1536 tokens, completion 726, finish reason
stop. Candidate SHA256 fb0b13349ef9448b4c5725bf1e568aabc8f14caf989e64252ffffe5280fb5a83.
Manual static review rejected an HBM arithmetic operand and an invalid
tensor_scalar call. Stage score 0; physics score remains unmeasured. It retains
the original projected-gradient recurrence, so there is no convergence gain yet.
candidate.py and generation-attempt.json are untouched; static-review.json and
revision-feedback.txt record findings separately. SDK execution has not occurred.
The earlier connection-refused attempt is infrastructure failure, not a model
physics failure. Preserve it separately from the generated-code attempt.

For revision, qwen_grip.py accepts --previous candidate.py and uses both the
trusted baseline and prior candidate as context. The previous code is never
executed by generation, and its hash is saved to link the attempt lineage.

Second actual attempt: qwen-grip-43604247fef94b6db66eedcd55033ec5, linked by
previous_candidate_sha256. Generation 41.2404 seconds; prompt 2085 tokens,
completion 571; finish reason stop. Static review approved the exact candidate
hash 359cc19b08af4a2360a12da1341558b17da374db4c57391aa1216791030dbf1d
for simulator smoke testing: both earlier API issues are corrected. Projected
gradient and its budget are unchanged; no physics/convergence/speed gain yet.
The runner accepts --candidate plus --candidate-sha256, preserves a source copy,
and loads the reviewed copy rather than silently running the baseline. This is
operator-reviewed execution, NOT an isolation mechanism for arbitrary code.

Simulator evaluation of the second attempt has now completed on seat-260:
data/qwen-revision1-sim, batch 2, 32 updates, one repeat. Trusted grade
grading-68c360ecc3584fb8ac1cc8ff2b20a028: 0/16 physics passes, throughput
ineligible. Saved outputs match the same-budget baseline exactly (max absolute
difference 0), with unchanged inputs and zero padding. This demonstrates an
API-usage repair after feedback, not improved convergence. A separate
physics-evaluation.json links this result to the candidate hash, leaving the
generation and pre-execution static-review records unchanged. The next prompt
explicitly requests a mathematical convergence change and preserves all gates.

Third generated attempt qwen-grip-fa0faa75c9ba42b8a6c80001087dccf7 returned
byte-identical source to attempt two (same SHA256). Generation 41.2446 seconds;
prompt 2072 tokens, completion 571. duplicate-review.json records revision score
0; no new simulator evaluation, physics score remains unmeasured for this attempt.
The prior same-code 0/16 result applies only to the same workload/settings.
Generation now detects unchanged ASTs (including comment/format-only changes)
and records duplicate_previous without executing code. This is not general
semantic equivalence detection. --temperature is explicitly configurable/logged.

The next prompt is a human-guided accelerated projected-gradient experiment,
with distinct iterate/history/extrapolated state and a tunable fixed momentum.
This must be disclosed: it is not autonomous algorithm discovery, and success
would still require measured physics scores and matched-accuracy device timings.

Fourth generated attempt qwen-grip-c6039970cf1f4059a74e0d11bdf7be11 changed
source (SHA256 d88c5772a874464d74fed453fefef6d7cb9d7b41c3387afc8095481fd7702e03),
but failed mathematical review. It omits alpha, updates with
x_new=max(0,x+.9*(previous-gradient)), and sets extrapolated=x_new without
extrapolation. At positive fixed points it requires gradient(x)=x rather than
the QP's gradient(x)=0. Stage score 0, physics unmeasured, not executed.
Generation 50.8760 seconds; 2011 prompt and 710 completion tokens.
The next feedback supplies the exact fixed-momentum projected recurrence.
Any later success is human-guided NKI implementation plus checker feedback,
not independent discovery of that mathematical method by Qwen.

Fifth generated attempt qwen-grip-6074b673bb6348ce932b631a7a079f4c implements
the supplied recurrence correctly on static inspection: rate is used, gradient
is evaluated at y, x_new is projected, and y_new=x_new+.9*(x_new-x) is computed
before overwriting x. Candidate SHA256
23fd2f7c0580a17561f9a320d255d08979077e31c372df6af35b14703b0f06b9.
Generation 63.3061 seconds; prompt 2133 tokens, completion 880, finish reason stop.
Static approval for simulator execution only. Physics and timing are unmeasured;
compare first at the baseline's 32-update budget. beta=.9 is an experimental
choice, not a claimed optimum or a guarantee that every case converges.

The fifth candidate has now executed in the seat simulator at 32 updates,
batch 2, one repeat. Trusted grade grading-7b3daddcb7e04fa39668cb79881ea2ce
under data/qwen-accelerated-sim32: 0/16 physics passes. Compared with the
32-update baseline, projected residual, edge-force error and contact-wrench error
decreased on all 16 cases; acceleration error decreased on eight and increased
on eight. Worst residual fell from .782813523 to .546554950, still above 1e-4.
This is partial numerical progress at equal budget, not acceptance or a speedup.
physics-evaluation.json records the comparison and links full scored reports.
Next diagnostic: both methods at 256 updates with unchanged gates. This changes
the experimental budget, not the candidate algorithm, and remains public-only.

Matched-budget 256-update simulator comparison is now graded: baseline 0/16
(grading-18f798ca24384b918e8e209b7025feb4); accelerated candidate 8/16
(grading-c0a848a7fe514dd28d1aabec06cfd461). Every friction=.8 case passes,
every friction=.2 case fails. Failing residual range .00362952.. .13883326;
failing force-error range .02939747.. .23761097, each against limit 1e-4.
comparison-sim256.json links full evidence. This is a public correctness gain
under the same update count; no hardware throughput metric is eligible yet.
Mass-scaled pairs have identical normalized error profiles, limiting numerical
diversity. This is not a 16-case claim of unrelated task generalization.
Next generation feedback targets poorly conditioned cases, must retain existing
passes, and is evaluated first at the same 256-update budget with unchanged gates.

Sixth generated attempt qwen-grip-c73d95723c2e475997b373aa9b88a8d2 changes
only beta from .9 to .85 (plus a comment). Candidate SHA256
d6b0edb0efc8bde9b2a4b90ec2a5e7baec23fff50bc8f126fdd2a1421e011d78.
Generation 65.0912 seconds; prompt 2339 tokens, completion 905. Static review
approves this exact code for simulator execution. The comment claiming improved
stability is an untested hypothesis, not measured feedback evidence. This is
parameter tuning within the human-specified method, not a new algorithm.
Compare all 16 cases at 256 updates, checking both remaining failures and any
regression of the preceding eight passes. Physics and throughput remain pending.

The beta=.85 candidate is now graded: data/qwen-beta085-sim256,
grading-179a4800bcd8416391a8ea10f4101380. At 256 updates it passes the same
eight cases as beta=.9, with no lost/new passes. All eight remaining cases have
larger edge-force errors (range .0894295.. .386580 vs .0293975.. .237611).
Worst failing residual rises from .138833 to .225873. Preserve this negative
result; beta=.9 remains the stronger convergence candidate at this budget.
No all-case acceptance or hardware timing. Next is a disclosed human-proposed
beta=.95 experiment, branched from beta=.9 to isolate this single parameter.

Seventh generated attempt qwen-grip-0507b9b809d8419fa3b6d32d1d64d9d3 uses
the documentation-backed card and captured seat signatures from nki
0.6.0+31049202112.g85070674. Code/request/card/SDK artifact hashes verify.
Candidate SHA256 39b36a1dd464e48c626d5f04fbfe6b797807dbe984dd1cae2e263aa98ec290d3.
Only functional change from beta=.9 is the human-requested beta=.95. Generation
63.9375 seconds; 2596 prompt tokens, 889 completion tokens, finish reason stop.
Static approval for simulator execution at 256 updates. All-case physics and
hardware throughput remain unmeasured; this does not isolate any causal benefit
from the documentation card, because the algorithm parameter also changed.

Seventh candidate at 256 updates is now scored: data/qwen-beta095-sim256,
grading-44378d000ae849388f1f76261602b3b7. It passes 4/16 vs beta=.9's 8/16,
with no new passes and regressions on grip-006/007/014/015. Force error decreases
on all eight low-friction cases but they still fail. Higher momentum is a trade-
off at this endpoint, not an overall accepted-solution gain. Do not infer
oscillation, instability or FP32 stagnation from this single endpoint alone.
physics-evaluation.json preserves this negative all-gate result and its partial
numerical benefits. Next diagnostic compares beta=.9 and beta=.95 at 1024
updates, keeping all cases and gates fixed. This is an increased compute budget,
not a new model-generated algorithm or device throughput result.

qwen_grip.py sends the baseline source, input/output contract and compact public
feedback to an OpenAI-compatible Qwen endpoint. It saves request/response,
baseline, feedback, candidate source, code hashes, generation duration, finish
reason and usage when returned. No oracle force arrays or private cases are sent.
It does not execute generated code or assign a physics score from syntax alone.
generation-attempt.json has physics_score=null until independently evaluated.
Truncated output and wrong entry-point signatures are recorded as generation errors.

Prepare without contacting a model:

```bash
python qwen_grip.py --feedback <new-grading-directory>/next-prompt.txt --dry-run
```

On a seat with the model server running, copy qwen_grip.py and only the compact
feedback file, then use its known local endpoint (confirm the server is actually
running; stopping it for hardware benchmarking also stops generation):

```bash
python qwen_grip.py --feedback gripping-feedback.txt --base-url http://localhost:8000/v1
```

Default model is Qwen/Qwen3-8B, disable-thinking, temperature .2, max output 1800
tokens. Adjust context/output limits explicitly if the server reports truncation.
Generated code requires review and isolated simulator execution before any device
benchmark. The current trusted baseline harness is not an untrusted-code sandbox.
The generation helper is one attempt, not yet an automatic revision controller.

## What a Correct Qwen Output Would Establish

An independently verified 16/16 public result would establish improved public
correctness over the 32-update baseline's 0/16. If the update budget changes,
report it; do not attribute the gain to an algorithm change without evidence.
It is not evidence of fine-tuning, private generalization, successful sustained
gripping, or throughput speedup by itself. Record the candidate hash, prompt,
model, sampling settings, update budget, failed attempts and each accepted score.

For a throughput benefit, compare candidate and baseline that BOTH pass the
same physics gates, on the same device allocation, all the same public cases,
batch sizes and repeated timing protocol. Include preprocessing/transfer costs
in the declared measurement boundary. Report spread, compilation separately,
and accepted problems per second. A failing baseline cannot support a matched-
accuracy speedup ratio. Preserve data/results for independent score reproduction.

To claim physics feedback itself helped, later compare rich feedback with
generic failure-only feedback under matched generation budgets and repeated
runs. One successful revision illustrates the loop but does not isolate causality.
Freeze unseen gripping cases before final evaluation; do not return private-case
feedback for more revisions. The existing private set covers frictionless cases,
not gripping. All model-improvement and device-performance claims remain pending.

## Scope Pivot Due to the Hackathon Time Limit

On October 10, 2026, we shifted the active deliverable to simpler frictionless
plane/pair contact problems because of the hackathon's time limit. The gripping
fixed-momentum feedback loop plateaued at 10/16; a human-written adaptive-
restart CPU reference also passed 10/16 in FP32 at 1024 and 4096 updates.
Higher-precision working arithmetic passed 16/16 on the same exported inputs,
suggesting a numerical issue that needs more investigation than the remaining
hackathon time permits. This is not a claim that FP32 gripping cannot work.

The simpler cases provide a passing baseline for measuring correctness-gated
Trainium throughput. We have not yet demonstrated a new Qwen-generated speedup
on that scope. Earlier gripping attempts and failures remain in the record;
we neither hide them nor relax the checker to turn failures into passes.
Detailed diagnostics, artifact locations and limitations are documented in
ADAPTIVE_RESTART.md. The checker, complete attempt log and one-page run note
remain the required final deliverables.
