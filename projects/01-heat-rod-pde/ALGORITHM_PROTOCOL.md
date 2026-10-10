# Algorithm study: predeclared protocol

Runtime starting point: a943cff on challenge1-runtime-resilience. The official checker,
problem files, original agent and previous improved agent remain unchanged.
Prior exploratory research is development evidence only; this study reruns real Qwen3-8B
with the current original-checker scoring policy. Historical 3/8 vs 3/8 validated results
remain separate because their extra validation policy differed.

## Hypotheses and risks

- A (cache): concurrent per-problem deduplication reduces repeated calculator execution.
  It preserves the original calculator's numeric semantics. Keys normalize syntax only;
  equivalent spellings can safely miss the cache. It may save time without improving solves.
- B (final): advisory structure checks and at most one reserved continuation repair may
  recover truncated/unexpanded answers. It never lowers original checker scores or deletes
  a free index; the model chooses coefficients and explicit finite terms. A repair uses
  the existing second-call slot and may waste that slot on an unrecoverable answer.
- C (feedback): public-boundary method selection, compact coefficient/assembly stages,
  different sample checklists, reuse of computed coefficients, and feedback linking each
  model-requested normalized projection to its own wave should reduce method/assembly errors.
  These subcomponents are a bundled reasoning policy, not separately attributable causes.
- D (adaptive): reduce requested tokens only near the case deadline, reserve save time,
  and shorten repeated calculator context. Smaller replies can also worsen correctness.
- E: initial combined configuration cache,final,feedback,adaptive. Diagnose first; record
  any selection/change before the formal benchmark. No modification during a frozen suite.

Common new-controller infrastructure: a 3-second per-calculation subprocess cap, original
calculator implementation, per-response/per-calculation journal, global request ceiling,
and incremental candidate logging. The feature-disabled core control measures this
infrastructure separately from A–D. Subprocess startup can increase CPU time.

## Fixed diagnostic and formal experiments

Diagnostics: Level 1.1 seed 0 and Level 1.3 seed 0; variants core,A,B,C,D,E.
The formal set is the historical eight configurations:
[[0,1,0],[0,2,0],[0,3,0],[1,1,0],[1,2,0],[1,3,0],[1,3,1],[1,3,2]].
Formal variants: baseline (original agent), previous (current improved agent), E (selected
algorithm configuration). All cases are retained regardless of outcome. The diagnostic
cases overlap the fixed benchmark; this is not an unseen generalization study.

All variants: real Qwen/Qwen3-8B, samples=2, rounds=3, max_tokens=512 per request,
tool_steps=1, workers=2, outer deadline=180 seconds, temperature=.6, top_p=.95, thinking
disabled. No model RNG seed is sent (Neuron incompatibility). Problem seeds are fixed.
Previous-agent transport retries are explicitly zero, matching the new controller's budget;
the historical reference used retry defaults, so this difference must be reported.
New controller internal deadline=177 seconds. At most 12 model requests and 6144 requested
completion tokens/case. Cache hits still consume requested calculator-expression slots.

Variants rotate by case/repeat index and run sequentially at suite level; two samples
may overlap within a case. Source files are snapshotted before a suite, with hashes,
declared source commit, complete commands and telemetry. Completed response usage and
unknown in-flight/error usage are reported separately.

Primary metric: original-checker accepted solves in completed runs; Level 1 is primary
within that metric. Timeout, service failure, evaluation error and unsolved are separate.
A full-score candidate in a timeout is disclosed separately and does not turn the case
into a completed solve. Additional validation is not run and is explicitly labeled so.
No grammar check, held-out validator or hidden solution changes acceptance.

Planned resource ceiling before optional follow-up: 12 diagnostic case-runs and
24 formal case-runs, each at most 180 seconds. If healthy and useful, one additional
full eight-case E repeat may assess variability. Actual requests, tokens, tool executions,
cache hits and time are reported; these are caps, not promises of resource use or success.
If a revision is needed after diagnostics, preserve the first suite and label a new version.

Stop the affected live suite on unavailable inference, permission failure or resource
exhaustion; preserve partial evidence. Do not substitute offline outputs or selectively
drop failures. Official selftests and all regression tests run before live inference.
No main/master merge is authorized by this study.
