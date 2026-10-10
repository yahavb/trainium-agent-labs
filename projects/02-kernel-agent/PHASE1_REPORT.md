# Phase 1: deterministic diagnostics and equal-reward selection

Development worktree: `/tmp/trainium-kernel-dev`. Branch: `feat/nki-diagnostic-selection`.
Base/control revision: `8f1ca41`. This is an implementation and offline validation
report, not evidence of improved agent correctness. No model inference was run.

## Files and architecture

- Created `failure_selection.py`: pure classifier, AST fingerprints, diagnostic
  evidence, maximum-reward selector, and additive JSONL metadata.
- Modified `agent.py`: imports, optional `solve()` selection/logging branch, and
  `--selection-policy {reward,diagnostic}` (default `reward`).
- Created `tests/test_failure_selection.py`: classification, selection, mocked
  integration, original-revision parity, CLI, and offline replay tests.
- Created `replay_selection.py`: read-only recorded-choice replay, with no NKI
  execution or model calls.
- Created this report and carried forward root `AGENTS.md` and
  `HACKATHON_CONTEXT.md` with the approved Phase 1 scope.

Classification runs after grading. It never changes reward, code, feedback, the
checker, or the repair prompt. Diagnostic selection determines which existing
candidate supplies the existing next prompt. Failure history is local to one
solve and contains all failed candidates, grouped by round. No adaptive planner,
new sampling strategy, or additional LLM call is included.

## Classification

Implemented: `NO_CODE`, `SYNTAX_ERROR`, `STATIC_RULE_VIOLATION`,
`INVALID_API_FUNCTION`, `INVALID_API_ARGUMENT`, `DMA_SHAPE_MISMATCH`,
`INVALID_BUFFER_PLACEMENT`, `INVALID_TENSOR_DIMENSIONS`, `OUT_OF_BOUNDS`,
`NUMERICAL_MISMATCH`, `INCOMPLETE_OUTPUT`, `MEMORY_TRAFFIC_EXCESS`,
`HARDWARE_CORRECTNESS_HAZARD`, and `UNKNOWN`.

Each frozen diagnostic preserves original feedback, category, normalized
signature, classification confidence, and reason. Exact checker verdicts and
specific assertions receive confidence 0.99; missing API/member/callability
patterns receive 0.95; unsupported feedback receives 0.0. These are heuristic
pattern-strength labels, not calibrated probabilities or kernel confidence.
Successful feedback without a failure pattern is also `UNKNOWN`; reward and
the original checker remain the authority on success.

Signatures remove numeric values and addresses while retaining API/operand names.
Case labels and most appended advice are excluded by matching the actual symptom.
Different sizes in the DMA assertion therefore share a signature; different
missing API names remain distinguishable. Assignment/broadcast size errors map
to tensor dimensions, not DMA unless the feedback explicitly identifies DMA.

## Selection algorithm

1. Find the exact maximum reward. All lower-reward candidates are excluded.
2. In reward mode, retain the first candidate at that reward, as before.
3. In diagnostic mode, rank only maximum-reward candidates lexicographically by:
   - repair-target strength: 0 none, 1 identified symbol/location, 2 explicit
     operand/symbol constraint or targeted corrective action;
   - localization strength: 0 global, 1 named API/operand, 2 source line or output
     index (the input test-case label does not count);
   - structural code novelty relative to previously failed candidates;
   - fewer previous rounds with the same normalized failure signature.
4. Retain original candidate order if these criteria still tie.

No fixed category-difficulty ordering is used. Classes determine which explicit
evidence patterns to inspect, not a claim that one failure is easier to repair.
Target and localization scores are unverified repairability hypotheses. AST
fingerprints ignore formatting and comments; they do not prove semantic novelty.
Repeated signatures are counted once per round, not once per duplicate sample.

Every candidate has a reason and evidence tuple. Diagnostic JSONL rows preserve
all eight original fields and add `failure_category`, `normalized_signature`,
`classification_confidence`, `classification_reason`, `candidate_index`,
`selected`, `selection_policy`, `selection_reason`, and `selection_evidence`.
Original feedback remains in the legacy `feedback` field. Default reward-mode
rows retain the exact original schema.

## Validation

Run from `projects/02-kernel-agent`:

```bash
python -B -m unittest discover -s tests -v
python -B nkibench.py --selftest
python -B kernelbench.py --selftest
```

28 deterministic tests passed. Both existing checker self-tests passed. Syntax
compilation and diff whitespace checks passed. Integration tests mock both
generation and grading, use only in-memory logs, and never access the shared
`/tmp/_agent_levelN.py` paths. A regression test executes the original revision's
agent definitions and compares default results, prompt calls, JSONL, and console
output against the modified agent under identical mocks, including empty answers.
Checker, scoring, generation/configuration, prompt, and stopping logic remain
unchanged; the selector is enabled only by the explicit flag.

## Recorded baseline replay

A read-only snapshot at `2026-10-10T16:17:57.620167Z` captured 144 complete attempt
rows, forming 36 four-candidate rounds across more than one repeat. Baseline
files were read, not modified. Copies and results live in the ignored directory:

`runs/phase1-offline-20261010T161757.620167Z/`

The snapshot manifest records byte counts and hashes. The final replay is
`selection-replay-final.json`. To reproduce its choices without inference:

```bash
python -B replay_selection.py runs/phase1-offline-20261010T161757.620167Z/attempts.jsonl --samples 4
```

| Level | Replayed rounds | Changed choices |
| --- | ---: | ---: |
| 1 | 16 | 0 |
| 2 | 11 | 0 |
| 3 | 5 | 1 |
| 4 | 4 | 0 |

The changed choice was the first repeat's Level 3 round 0: baseline chose
candidate 0 with the generic SBUF/PSUM rank assertion; diagnostic selection chose
candidate 1 with DMA `src=8192, dst=65536`. Both rewards were exactly
`0.30000000000000004`. The latter diagnostic names both operands and a size
invariant, while the former lacks a source location. The new candidate is not
known to be better code or easier to repair.

Actual Level 1 errors classify as DMA mismatch (`src=4, dst=16384`), invalid API
argument (`transpose_moving`), invalid buffer (`dst` needs PSUM), and invalid API
function (`multiply`, then `scalar_mul`). Level 2's `src=384, dst=512` and
`src=384, dst=12` DMA errors share a normalized signature. No candidate changes
were observed on those levels in this snapshot.

Replay is reliable for individual choice comparison given the inspected
single-writer, four-sample run. Legacy logs lack explicit repeat/run IDs; the tool
infers solve boundaries from level changes and round resets and rejects incomplete
or ambiguous batches. Later rows remain baseline-generated even after a changed
choice, so replay does not estimate the improved policy's subsequent trajectory,
solve rate, token consumption, or runtime. Limited diversity leaves little room
for this intervention; the one changed choice is not a correctness gain.

## Limitations and risks

- Regex classification can miss new SDK wording; those errors remain `UNKNOWN`.
- Precise errors may describe superficial defects in a fundamentally wrong algorithm.
- Structural novelty can reward a variable rename without meaningful progress.
- Priority of precision over novelty may retain a recurring but well-described failure.
- Confidence is uncalibrated and is not used as a correctness score or tie-breaker.
- Existing checker discrepancies, temporary-path sharing, stopping rules, context
  estimation, and missing token usage remain unchanged by design.
- Diagnostic log metadata adds CPU work and log volume; wall time must be measured.

## Next controlled experiment — pending approval, not executed

After the protected baseline and any other NKI grading process have finished,
run reward and diagnostic policies sequentially on identical Level 1 cases. Project
1 may share model capacity, so record its activity when comparing wall time.
Use unique outputs; never start this command while the protected NKI run is active.

```bash
cd /tmp/trainium-kernel-dev/projects/02-kernel-agent
mkdir -p runs
phase1_run_dir=$(mktemp -d "$PWD/runs/phase1-level1-XXXXXXXX")
for policy in reward diagnostic; do
  python -u agent.py --level 1 --rounds 8 --samples 4 --context 8192 \
    --max-tokens 2500 --give-up-after 4 --terse 0 --repeat 5 \
    --base http://localhost:8000/v1 --model Qwen/Qwen3-8B \
    --selection-policy "$policy" --log "$phase1_run_dir/$policy-attempts.jsonl" \
    > "$phase1_run_dir/$policy-console.log" 2>&1
done
```

Compare solve counts, rounds/generated candidates to success, per-attempt reward,
final-best reward, numerical/checker acceptance, failure classes, and changed
choices. Because no Level 1 choices changed offline, also evaluate Level 3 under
the same controlled settings before drawing a broad conclusion. Current logs
still do not record actual token usage or full per-candidate timing; neither can
be honestly reported as exact without separately approved instrumentation.
