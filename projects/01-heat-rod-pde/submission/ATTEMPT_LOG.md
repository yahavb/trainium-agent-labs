# Formal experiment log

This log records the team's selected experiment attempts. An experiment attempt
is a run of the configured problem set; each problem can contain several rounds
and candidate answers. Detailed candidate traces remain a separate deliverable.

## Initial baseline — Original agent

This is the user-designated earliest attempt, separate from the approximately
three planned optimization attempts. Date: 10 October 2026. Reported environment:
seat-86, Qwen3-8B (based on the supplied conversation report). Entry: original
`agent.py`, before subsequent optimization. Commit and other unreported settings
remain unknown. Four candidates were generated per round; Level 1 allowed four
rounds. A calculator and the original checker feedback were enabled.

| Problem | Status | Best reward | Rounds | Sum of round durations (s) |
|---|---|---:|---:|---:|
| level0.1 | Solved | 1.0 | 1 | 73.8 |
| level0.2 | Solved | 1.0 | 1 | 159.3 |
| level0.3 | Solved | 1.0 | 1 | 110.3 |
| level1.1 | Solved | 1.0 | 1 | 215.3 |
| level1.2 | Unsolved | 0.8 | 4 | 872.1 |
| level1.3 | Solved | 1.0 | 2 | 320.6 |

Level 0: 3/3 solved, 343.4 seconds. Level 1: 2/3 solved, 1,408.0 seconds.
Overall: 5/6 solved, 1,751.4 seconds. These totals sum round durations, counting
each parallel candidate round once; they are not full process wall times.

### Supplied candidate scores and feedback

| Problem | Round | Candidate scores | Reported issue or outcome |
|---|---:|---|---|
| level1.2 | 1 | [0.0, 0.0, 0.0, 0.0] | Unparseable LaTeX |
| level1.2 | 2 | [0.0, 0.0, 0.6, 0.0] | Correct waves and amplitudes, wrong decay |
| level1.2 | 3 | [0.0, 0.0, 0.0, 0.0] | Unparseable LaTeX again |
| level1.2 | 4 | [0.0, 0.8, 0.0, 0.0] | Zero solution; initial-shape error 100% |
| level1.3 | 1 | [0.6, 0.6, 0.6, 0.6] | Correct coefficients, decay too fast |
| level1.3 | 2 | Full array unknown; one full-score candidate reported | Corrected decay, four nonzero Fourier terms |

The zero solution satisfies the PDE and homogeneous boundaries, so 0.8 does not
mean solved. The original loop used the previous round's best answer in its next
prompt: the third round's malformed output displaced the useful second-round
answer. This exposes output-format, decay-calculation and answer-retention issues.
Problem 1.3 demonstrates a successful feedback correction in this baseline run.

Twenty explicitly supplied baseline candidate scores are transcribed in
`reported_candidate_scores.jsonl`. The second-round candidate indices and other
scores for level1.3, and all candidate arrays for Level 0 and level1.1, were not
supplied and are not invented. The original `attempts.jsonl` and later
`legacy-records.jsonl` are mentioned in the report without a verified location;
neither has been located locally for this selected baseline. Project 2 commands
without accompanying results are excluded.

## Optimization Attempt 1 — Selected four-candidate rerun

The user designated this rerun as the formal Attempt 1 result. The earlier
two-candidate run is retained below as preliminary evidence, not renumbered as
Attempt 2. This supplied run is separate from the stopped seat-87 tests performed
in this chat; their source hashes and settings must not be attributed to it.

Date: 10 October 2026. Level 1, all three problems, problem seed 0, four candidates
per round. Calculator and decay repair are evidenced by the console. Original
checker acceptance is explicitly reported. Commit, seat/model identity, entrypoint,
workers, maximum round budget, token limit, tool exchange limit, retries and timeout
were not supplied. Actual rounds used: one per problem. The intended four-round
budget cannot be verified from three first-round successes alone.

| Problem | Best score | Candidate scores (round 0) | Mean | Rounds | Model requests | Seconds | Accepted decay repairs |
|---|---:|---|---:|---:|---:|---:|---:|
| level1.1 | 1.0 | [0.6, 0.0, 1.0, 0.0] | 0.40 | 1 | 6 | 186.7939924270031 | 0 |
| level1.2 | 1.0 | [0.0, 1.0, 0.0, 0.0] | 0.25 | 1 | 5 | 185.48773628599884 | 0 |
| level1.3 | 1.0 | [1.0, 1.0, 1.0, 1.0] | 1.00 | 1 | 8 | 38.36541318098898 | 4 |
| Total | 3/3 solved | 12 candidates; 6 full-score | 0.55 | 3 | 19 | 410.6471418939909 | 4 |

The total sums the supplied per-problem durations, not full batch process wall
time. Calculator requests (individual expressions) were 3/2/20, total 25; these
are distinct from model API requests. All reported request and evaluation errors
were zero. Candidate score range: 0.0–1.0; best score range across problems: 1.0–1.0;
per-problem time range: 38.3654–186.7940 seconds. These are across-problem and
within-run variation, not estimates of repeated-run variability.

### Reported winning answers

```text
level1.1: u(x, t) = exp(- (pi**2/4)*t) * sin(pi*x/2)
level1.2: u(x, t) = exp(-pi**2*t/24)*sin(pi*x/6) + 2*exp(-25*pi**2*t/24)*sin(5*pi*x/6)
level1.3: u(x, t) = 32*exp(-pi**2*t/2)*sin(pi*x/2)/pi**3 + 32*exp(-9*pi**2*t/2)*sin(3*pi*x/2)/(27*pi**3) + 32*exp(-25*pi**2*t/2)*sin(5*pi*x/2)/(125*pi**3)
```

### Logic, evidence and next directions

The optimized design retains the best answer across rounds and applies symbolic
decay repair before grading with the original checker. Repair changes decay rates,
not wave amplitudes. The rerun explicitly reports four accepted repairs on 1.3;
its four full scores must be attributed to the tool-assisted agent. Raw pre-repair
scores and exact edits are absent, so do not invent a model-only pass rate. Problems
1.1 and 1.2 had no accepted decay repair, despite calculator use. Full-score selection
solved all three problems, while six other candidates remained below full marks.

The preliminary run used two candidates/round; this rerun used four. No exact source
diff or other settings for the rerun were supplied. Do not claim that a particular
new algorithm or increased sampling caused the observed change from 2/3 to 3/3.
Runtime was 410.65 versus 400.32 seconds in these unmatched runs. Baseline Level 1
was 2/3 in 1,408.0 seconds with a different timing definition and incomplete settings.
The current observation supports a successful run, not a stable success probability
or controlled speedup. All problems ended on round 0, so this rerun does not test
whether retaining best answers improves later-round recovery.

Future work: improve parseable scalar answer output and coefficient calculation;
repeat fixed seeds/configurations; compare repair enabled/disabled and log both
original and executed grades. These are recommendations only; this reporting task
does not change agent or checker code.

Reported artifact directory (relative path; host and absolute location unknown):

```text
results/full-test-level1/20261010T194251Z-e8af09c9
```

`reported_rerun_summary.json` transcribes the supplied per-problem JSON;
`reported_candidate_scores.jsonl` transcribes all twelve supplied scores. These
are not downloaded original traces. All unprovided model answers, per-candidate
executed answers and traces remain null. Preserve and verify the raw `attempts.jsonl`,
settings and `summary.json` before final evidence hand-in.

## Historical preliminary Attempt 1 — Two candidates

- Date: 10 October 2026.
- Environment: seat-86, Qwen3-8B on AWS Trainium.
- Tested commit: `6ea9f98583c1c01db92705903c9cab167096b882` (`6ea9f98`).
- Entry point: `improved_agent.py`.
- This was the master revision at execution; runtime-fix merge `b94759c` occurred
  afterward and is not the tested version for this record.
- Settings: all Level 1 problems, problem seed 0, two candidates/round, two workers,
  at most three rounds, 1200 output tokens/request, calculator enabled, at most one
  calculator exchange/candidate, decay repair enabled, zero request retries.
- Acceptance: original `pdecheck.check` reward 1.0; no extra physics validation.
- Provenance: user-supplied run report. Original JSONL and summary are not yet
  downloaded or independently checked. Unknown fields are not reconstructed.

| Problem | Status | Best reward | Rounds | Requests | Seconds | Request errors |
|---|---|---:|---:|---:|---:|---:|
| level1.1 | Solved | 1.0 | 2 | 5 | 186.01 | 0 |
| level1.2 | Unsolved | 0.8 | 3 | 8 | 195.05 | 0 |
| level1.3 | Solved | 1.0 | 1 | 4 | 19.26 | 0 |
| Total | 2/3 solved | — | 6 | 17 | 400.32 | 0 |

The total is the sum of reported per-problem elapsed times, not an independently
measured end-to-end batch time. It includes the unsuccessful problem.

| Problem | Round 1 candidates | Round 2 candidates | Round 3 candidates |
|---|---|---|---|
| level1.1 | [0.6, 0.0] | [1.0, 0.0] | — |
| level1.2 | [0.0, 0.0] | [0.8, 0.8] | [0.0, 0.0] |
| level1.3 | [1.0, 1.0] | — | — |

Twelve preliminary Attempt 1 candidate scores are transcribed in `reported_candidate_scores.jsonl`.
That file is explicitly a score transcription, not the original attempt trace.
Its round/sample indices start at zero; this table displays rounds starting at one.

### Failure and tool attribution

Reported best answer for level1.2:

```text
u(x, t) = 0.5*exp(-pi**2*t/24)*sin(pi*x/6) + 1.0*exp(-25*pi**2*t/24)*sin(5*pi*x/6)
```

Required initial shape:

```text
u(x, 0) = sin(pi*x/6) + 2*sin(5*pi*x/6)
```

Both amplitudes are half the required values. The equation and boundary conditions
pass, giving 0.8; the initial-shape condition fails. The final two candidates
received zero, while the summary retained the previous best answer. The decay
tool does not adjust amplitudes.

Full marks for level1.3 do not establish whether the raw model answer or a repaired
answer earned the score. Inspect `model_grade`, `executed_answer`, and the
`decay_repair` trace before assigning credit to the tool.

### Raw evidence locations

These paths are on seat-86, not on the local laptop:

```text
/workspace/project1-master-6ea9f98/master-run-1.log
/workspace/project1-master-6ea9f98/results/master-run-1/20261010T190437Z-ddda37f3/attempts.jsonl
/workspace/project1-master-6ea9f98/results/master-run-1/20261010T190437Z-ddda37f3/summary.json
```

### Interpretation

One run of each of three problems solved 2/3. Across these problems, best rewards
range from 0.8 to 1.0 and times from 19.26 to 195.05 seconds. These ranges do not
estimate repeated-run variability or a stable success probability.

The initial baseline and preliminary Attempt 1 both solved 2/3 Level 1 problems
with best rewards 1.0/0.8/1.0. The baseline reported 1,408.0 seconds (sum of round
durations); Optimization Attempt 1 reported 400.32 seconds (sum of per-problem
elapsed times). The baseline used four candidates and four rounds; the optimized
run used two and three. Baseline seed, token budget and remaining settings are
unknown. These are unmatched single observations: no causal speedup or stable
improvement/regression in success probability is established. Level 0 has no
corresponding result in Optimization Attempt 1 and is excluded from that comparison.

## Historical runtime run — Previously designated Attempt 2

Superseded by the user-selected complete cd16f7b algorithm-agent run below. The
486.08-second result remains evidence, not the formal Attempt 2 entry.

### Runtime-labeled six-problem record

Date: 10 October 2026. Artifact directory contains version label `a4b58a5`; the
tested full commit has not been independently verified. Formal evaluation covers
Level 1; Level 0 results below are warm-up evidence. All problems use seed 0 and
four candidates per round. Original checker policy is explicit. Machine/model
identity, entrypoint, full settings and exact source changes were not supplied.
Do not infer them from the artifact name or earlier runs. Observed rounds are not
the maximum permitted round budget.

| Problem | Best score | Rounds | Model requests | Seconds | Request / evaluation errors |
|---|---:|---:|---:|---:|---|
| level0.1 | 1.0 | 1 | 7 | 91.39270988600038 | 0 / 0 |
| level0.2 | 1.0 | 1 | 4 | 185.17332292899664 | 0 / 0 |
| level0.3 | 1.0 | 1 | 7 | 110.6850118690054 | 0 / 0 |
| Level 0 warm-up | 3/3 | 3 | 18 | 387.2510446840024 | 0 / 0 |
| level1.1 | 1.0 | 2 | 12 | 264.3794697919948 | 0 / 0 |
| level1.2 | 1.0 | 1 | 5 | 183.30194744600158 | 0 / 0 |
| level1.3 | 1.0 | 1 | 8 | 38.40301401499892 | 0 / 0 |
| Level 1 formal | 3/3 | 4 | 25 | 486.0844312529953 | 0 / 0 |
| Both levels | 6/6 | 7 | 43 | 873.3354759369977 | 0 / 0 |

All totals sum the reported per-problem durations, not full batch process wall
time. They include every completed problem, not just the fastest candidates.

| Problem | Round 0 candidates | Round 1 candidates |
|---|---|---|
| level0.1 | [1.0, 1.0, 1.0, 1.0] | — |
| level0.2 | [0.0, 1.0, 0.0, 0.0] | — |
| level0.3 | [1.0, 0.0, 1.0, 1.0] | — |
| level1.1 | [0.0, 0.0, 0.0, 0.0] | [1.0, 1.0, 0.8, 0.8] |
| level1.2 | [0.0, 1.0, 0.0, 0.0] | — |
| level1.3 | [1.0, 1.0, 1.0, 1.0] | — |

### Interpretation and future directions

Level 1: 7/16 candidates earned full marks, while selection solved all three
problems. Mean candidate reward was 0.5375; candidate scores ranged 0.0–1.0 and
best problem scores were all 1.0. Per-problem durations ranged 38.40–264.38 seconds.
Level 0: 8/12 candidates earned full marks, with 3/3 problems solved. These describe
within-run and across-problem variation, not repeated-run reliability.

Compared with selected Attempt 1, Level 1 completion stayed 3/3, while total time
increased by 75.44 seconds (410.65 to 486.08) and model requests by six (19 to 25).
Problem 1.1 needed a second round: its time increased by 77.59 seconds, whereas
1.2 was 2.19 seconds faster and 1.3 nearly unchanged. Feedback preceded recovery
from four zero scores to two full-score and two 0.8 candidates. The report does
not give causes for the zero/0.8 scores or establish that feedback alone caused
recovery. It also does not identify a code change responsible for these outcomes.

Repair counts and pre-repair candidate scores were not supplied. Despite an
identical final 1.3 expression, do not inherit Attempt 1's repair attribution.
Full source/settings and raw traces are needed to distinguish implementation,
sampling and infrastructure effects. One slower run does not establish stable
regression. Continue work on output validity and coefficient accuracy; compare
repair on/off with fixed budgets and repeated seeds. These are recommendations,
not code changes made by this reporting task.

### Reported evidence locations

Relative artifact paths; host and absolute locations unverified:

```text
results/runtime-a4b58a5-20261010-204834/level0/20261010T204834Z-4777af32
results/runtime-a4b58a5-20261010-204834/level1/20261010T205502Z-f1890680
```

`reported_attempt2_summary.json` transcribes all six supplied summaries;
`reported_candidate_scores.jsonl` adds all 28 supplied scores (12 warm-up, 16
formal). Missing candidate answers, raw model grades and traces remain null.
This material is not the original `attempts.jsonl`; preserve original files and
verify settings before final hand-in.

## Optimization Attempt 2 — Selected latest-branch algorithm test

The supplied console follows a replaced symlink and concatenates three batches.
Do not combine them: master b94759c Level 0 was stopped; the a4b58a5 algorithm run
was stopped after two completed Level 1 problems; cd16f7b completed all three.
The owner designates the complete final batch as formal Attempt 2. The previously
designated 486.08s runtime run is retained above as historical evidence.

Tested commit: `cd16f7b57a878661bb371f4c1ef09bd6864ff356`, branch
`challenge1-runtime-resilience`, PR #5. Entry: `algorithm_agent.py`.
Environment: seat-87, requested model `Qwen/Qwen3-8B` on the Trainium service.
Level 1 only; seed 0, four candidates and workers, at most four rounds, 1200-token
request ceiling, one calculator exchange, request timeout 900s and per-problem
deadline 900s. Features: cache, final, feedback, adaptive. Adaptive stages may use
smaller caps, recorded in traces. Enhanced validation: not run. Source comparison
confirmed public checker files match master b94759c. The existing improved_agent
also matches master; this test exercises the new optional entrypoint instead.

| Problem | Round 0 candidates | Round 1 candidates | Rounds | Model requests | Seconds |
|---|---|---|---:|---:|---:|
| level1.1 | [1.0, 1.0, 1.0, 1.0] | — | 1 | 8 | 13.840276973001892 |
| level1.2 | [1.0, 0.0, 0.0, 0.0] | — | 1 | 8 | 73.8715014319896 |
| level1.3 | [0.6, 0.6, 0.6, 0.6] | [1.0, 1.0, 1.0, 1.0] | 2 | 15 | 140.52721492199635 |
| Total | 3/3 solved | 9/16 full-score candidates | 4 | 31 | 228.23899332698784 |

Times sum the per-problem measurements; the separate wrapper measured 229.8681s.
No candidate transport errors or evaluation errors appear in the downloaded logs.
Mean candidate reward: 0.7125. Best problem scores all 1.0; individual scores range
0.0–1.0. Per-problem time range: 13.84–140.53s. These are within-run/across-problem
differences, not repeated-run success estimates.

### What changed and what the evidence shows

The new optional agent caches identical model-directed calculator expressions,
repairs answers through further model requests, provides targeted feedback and
adapts request budgets. It retains the historical best answer. It does not enable
improved_agent's symbolic decay-rate replacement or the enhanced checker gate.
The seven model-repair requests are distinct from seven actual calculator executions.
Calculator requests/executions/cache hits were 4/1/3, 0/0/0 and 24/6/18 by problem,
total 28/7/21. There were 4494 completion tokens and 19613 prompt tokens.

### Decay correction: exact evidence, mathematical reason and risks

This is an answer correction, not a checker update or model-weight update. In
level1.3, all four first-round candidates used the same three coefficients and
waves as the four second-round candidates. Only their temporal decay rates changed
(up to equivalent expression formatting):

| Mode n | Spatial frequency mu | Round 0 rate lambda | Round 1 rate lambda |
|---|---|---|---|
| 1 | pi/2 | 2*pi**2 | pi**2/2 |
| 3 | 3*pi/2 | 18*pi**2 | 9*pi**2/2 |
| 5 | 5*pi/2 | 50*pi**2 | 25*pi**2/2 |

For a term `u=A*exp(-lambda*t)*sin(mu*x)`, differentiation gives
`u_t=-lambda*u` and `u_xx=-mu**2*u`. Thus `u_t=k*u_xx` requires
`lambda=k*mu**2`. Here k=2 and mu=n*pi/2, so lambda=n**2*pi**2/2.
The first-round expressions used 2*n**2*pi**2, omitting division by L**2=4.
Dividing these particular rates by four corrects that mathematical error.

The original checker recorded equation=False and both boundaries/start_shape=True
for every round-0 candidate (reward 0.6). Round 1 recorded all four components=True
(reward 1.0). Initial-shape error stayed 0.003364, or 0.3364%, below 0.5%.
Thus the observed 0.4-point gain is specifically the equation component, with no
relaxation of acceptance. The correction was generated by the model in the feedback
loop; improved_agent's automatic symbolic rate replacement was not used.

Hypothesis: localized feedback lets the model retain already-correct Fourier
coefficients and waves and fix their exponent frequencies, avoiding unnecessary
regeneration. The before/after algebra explains why the resulting expressions
satisfy the PDE, but this single sampled loop does not isolate feedback's causal
effect on model behavior. No coefficient, boundary or truncation fix was needed
for this particular candidate set. Problem 1.2 still had three zero-score
candidates; structural diagnostics are not an extra physics acceptance gate.

Risks: “divide by four” is specific to the omitted L**2 factor here, not a general
repair rule. Derive k*mu**2 again when diffusivity, rod length or boundary conditions
change. Decay correction cannot fix wrong amplitudes or missing Fourier modes;
all checker components must still pass. Duplicate candidate answers also limit
sampling diversity. Preserve raw pre/post answers and test new problems/seeds.
Only rates changed within this case's graded answers; the complete Attempt 2
implementation also changed caching, feedback, repair and budget logic relative
to Attempt 1. Do not attribute the whole-suite timing difference to rate correction.

Compared with selected Attempt 1, completion stayed 3/3, total time fell by 182.41s
(410.65 to 228.24), but model requests increased from 19 to 31. The run is promising,
not a controlled speedup: Attempt 1's exact source, concurrency, stage budgets,
timeouts and model-service conditions remain unverified. Full commit/source and
settings for this selected Attempt 2 are available. Do not inherit the earlier
486.08s result's settings or repair attribution.

### Interrupted-run variability and retained evidence

The a4b58a5 and cd16f7b agent/checker sources are identical; cd16f7b adds experiment
evidence only. The interrupted a4 run solved 1.1 in 17.30s and 1.2 in 164.33s over
four rounds; 1.3 was not completed. Its 1.2 round scores were all zeros, then
[0.6,0.6,0.0,0.6], [0.6,0.6,0.4,0.4], and [0.8,0.6,1.0,1.0]. Retain this slower
recovery alongside the final run, without reporting a complete three-case rate
for the interrupted batch or selecting only the faster observation as reliability.

Official selftests passed on a4b58a5. They were not repeated on the unchanged
source at cd16f7b; the prior log and source identity are recorded. Optional enhanced
checker behavior and complete regression coverage are not established by these
selftests or this live-model run.

Original evidence is in `raw/attempt2-cd16f7b`; interrupted evidence is in
`raw/interrupted-a4b58a5`. `raw/supplied-mixed-console.txt` preserves the attachment.
`attempt2_original_records.jsonl` combines the 16 full candidate records with source
file references; original per-case files remain intact. The summary transcription
and machine-readable register are linked to these originals.

Server batch: `/workspace/trainium-team87/projects/01-heat-rod-pde/results/team18-pr5-latest-20261010T211750Z-ba55c1`.

Future work: repeat a fixed configuration and run a matched master control;
compare cache/feedback/adaptive features separately; improve scalar formatting and
mode-specific decay calculations. No code was changed during this test. Automatic
merge criteria remain unmet until a matched comparison and integration evidence
are established; no merge was performed.

## Optimization Attempt 3

Merge decisions follow UPDATE_LOG.md: a matched Level 1 improvement on the tested
PR commit may be merged into the team fork without another owner confirmation.
The earlier 486.08s runtime run did not meet the rule. Selected Attempt 2 is
shorter than Attempt 1, but full comparison settings remain unmatched.

Pending. Planned budget: four candidates per round and at most four rounds on
the three Level 1 problems. Record tested commit, seat/model, full runtime settings,
per-round scores, timing, request/evaluation errors and repair attribution. Match
settings for any direct comparison. Preserve raw logs. No pending result is inferred.
