# Phases 2 and 3: implementation and validation

Implemented 2026-10-10 in `/tmp/trainium-kernel-dev`, branch
`feat/nki-diagnostic-selection`, baseline revision `8f1ca41`. Phase 1 remains
intact. No model inference was run: the original baseline PID 14656 was still
active when the guarded pilot launcher was invoked, and launch was refused.

## 1. Architecture

The existing controller still generates, grades, selects and repairs. New opt-in
policies act at two points: candidate prompt construction before the existing
parallel calls, and local documentation enrichment before repair generation.
Numerical reward remains primary; diagnostic selection only breaks exact ties.
The existing checker, NumPy references, scoring weights and stopping logic remain
unchanged. Default controller behavior is checked against the actual original
revision, including an instrumented standard configuration.

## 2. Files created and modified

Created:
- `candidate_diversity.py`: deterministic perspectives and exact/AST hashes.
- `nki_knowledge.py`: 12 compact cards, source-aware retrieval, installed SDK
  compatibility checks and local token budgeting.
- `experiment_metrics.py`: endpoint usage, truncation and context-local grading
  instrumentation, without inferred usage counts.
- `run_controlled.py`: preparation and guarded sequential A-D pilots, exclusive
  artifacts, manifests and aggregation.
- `tests/test_candidate_diversity.py`, `tests/test_nki_knowledge.py`,
  `tests/test_experiment_metrics.py`, `tests/test_nki_card_simulation.py`,
  `tests/test_controlled_runner.py`.
- This report.

Modified development `agent.py`, the existing Phase 1 regression test,
root `AGENTS.md`, and `HACKATHON_CONTEXT.md`. No original application/log files,
Project 1 files, checker implementations, reference kernels or SDK were modified.

## 3. CLI

| Option | Values/default | Effect |
|---|---|---|
| `--candidate-policy` | `standard` / `diverse`; standard | Same prompt or deterministic perspectives |
| `--selection-policy` | `reward` / `diagnostic`; reward | Existing Phase 1 tie selection |
| `--repair-policy` | `standard` / `grounded`; standard | Existing repair or local reference cards |
| `--instrument` | off by default | Add experimental telemetry to a control run |
| `--grade-dir` | unset by default | Private candidate-code directory |

Diverse or grounded policies enable experimental telemetry automatically.
All-standard reward mode without instrumentation retains the exact eight-field
legacy JSONL schema. Experimental fields are additive. Diagnostic-only Phase 1
logging continues to work. Standard model request parameters remain temperature
0.6, top_p 0.95, thinking off by default, and the existing answer-budget policy.

## 4. Candidate diversity

Candidate 0 uses the input prompt unchanged. Other perspectives examine simplest
legal operations, shapes/buffers/transfers, and tiling/boundary/output coverage.
They apply to initial generation and repairs without embedding benchmark kernels.
Repairs retain the selected source and exact checker message. Initially the
variants ask for localized corrections; only the simple perspective can relax
the minimal-change instruction after two repeated failures, and only for the
failing operation and its dependent shapes. Unrelated code remains protected.

`--samples 4` makes exactly four model requests, with results retained in input
candidate order regardless of completion order. Standard mode makes the same
requests as the original. Additional sample counts are supported deterministically.

Each experimental round records total candidates, unique exact-source and AST
hashes, duplicate fraction, prompt strategy, full prompt, selected candidate,
selection reason, reward and category. Hashes describe extracted source, not raw
response text. Duplicate fraction is `1 - unique_AST / total`; four identical
sources therefore have fraction 0.75. AST uniqueness is not semantic novelty or
correctness; empty or invalid sources can still contribute source hashes.

## 5. AWS documentation cards

Cards cover DMA, matmul, scalar arithmetic, binary tile arithmetic, reduction,
on-chip copy, allocation, tensor rank, memory regions, tiling/bounds, PSUM
accumulation and simulation limitations. Combined cards cover all requested
concepts. Each stores categories, signature/parameter names where applicable,
buffers, allowed/invalid patterns, concise guidance, a version-pinned official
AWS URL, hardware/SDK compatibility and explicit verification status.

Attribution: AWS Neuron documentation v2.32.0 and the reviewed
[AWS agentic development repository](https://github.com/aws-neuron/neuron-agentic-development/tree/ee25e45b9ace4aaefd8b8913bf5fa9b68adf6674).
Cards link individually to their relevant API/guide. They contain general
invariants, not complete kernels or benchmark-specific answers.

## 6. Retrieval and budget

Canonical category narrows the catalog. The original error identifies the
offending API/constraint before unrelated source calls. AST inspection resolves
import aliases and records actual source call expressions/line numbers; it does
not invent an error location. Invented multiply/scalar_mul calls retrieve legal
scalar or binary operation cards, with keyword context informing ordering.
Unknown/unmapped APIs fall back without unrelated documentation.

Compatibility filtering precedes deterministic ranking. Retrieve at most two
cards under 500 documentation tokens. A cached local Qwen tokenizer is used
without downloads or endpoint calls. If unavailable, a conservative UTF-8 byte
ceiling is used for budgeting and explicitly labeled; it is never logged as
endpoint token usage. The answer budget and a context reserve constrain optional
cards. Cards are omitted before sacrificing original code/error context.

Grounded prompts retain the original repair prompt, selected code and exact
feedback, then add the category, relevant API/constraint and attributed guidance.
The original repair policy otherwise stays unchanged, allowing retrieval to be
evaluated separately from a full adaptive planner.

## 7. SDK compatibility

All API card parameter lists were checked against installed
`0.6.0+31049202112.g85070674` signatures. Cards fail closed for an unverified SDK
build, unsupported target or signature drift. Target tags admit Trainium2 only;
explicit target overrides are respected. No Trainium3 compute indirection,
expanded matmul destination rules or DMA priority use is recommended.

Installed source/docstrings were inspected, including binary arithmetic's rule
that both operands cannot be PSUM. Tiny CPU exercises validate representative
DMA, scalar/binary arithmetic, reduction, copy, allocation and matmul calls,
first-write/subsequent accumulation behavior, plus rank/DMA rejection. This does
not prove all card constraints or device compilation/correctness. No hardware
test or SDK upgrade occurred.

## 8. Tests and validation

53 deterministic tests passed, retaining all 28 Phase 1 tests. Coverage includes
default behavior parity, exact call count/order, diversity on initial/repair
prompts, duplicate detection, selection integration, all seven observed failure
routes, attribution, compatibility filtering, budget exhaustion, unknown APIs,
telemetry, unchanged HTTP request settings, per-shape reward parity, private
grading paths, runner guards, unique preparation and metric aggregation.

Both existing `nkibench.py --selftest` and `kernelbench.py --selftest` passed.
Syntax compilation passed for 20 Python files. All ten original tracked Project 2
files match `8f1ca41` byte-for-byte; development checker/reference files match
the original. Baseline prompt/enrichment/extraction functions remain AST-identical.
No test called the live model or the shared baseline grading path. Existing
baseline logs continue to be owned by their running process, not modified here.

## 9. Recorded repair examples

Read-only construction from actual immutable Phase 1 snapshot candidates:

| Observed feedback | Category | Retrieved cards | Documentation tokens |
|---|---|---|---:|
| DMA src=4, dst=16384 | DMA_SHAPE_MISMATCH | dma, tiling | 229 |
| DMA src=384, dst=512 | DMA_SHAPE_MISMATCH | dma, tiling | 229 |
| nc_matmul transpose_moving unsupported | INVALID_API_ARGUMENT | matmul | 177 |
| Matmul destination SBUF, requires PSUM | INVALID_BUFFER_PLACEMENT | matmul, memory | 276 |
| nki.isa.multiply absent | INVALID_API_FUNCTION | scalar, binary | 279 |
| nki.isa.scalar_mul absent | INVALID_API_FUNCTION | scalar, binary | 279 |
| On-chip tensor rank below two | INVALID_TENSOR_DIMENSIONS | rank, allocation | 209 |

Examples preserve the full actual source and feedback. The guidance asks to match
DMA slice counts, use legal matmul layout/buffers, choose actual scalar/binary
instructions, or preserve two-dimensional on-chip layout. No corrected complete
kernel was inserted or generated. Counts above are local documentation counts,
not endpoint usage or evidence of successful repair.

## 10. Experiments and telemetry

No live result is available. At validation, the actual pilot launcher exited 2
because original baseline PID 14656 was active. This is a deliberate guard result,
not an implementation failure. It does not wait indefinitely or terminate work.

Prepared eight sequential entries: Levels 1 and 3, each with A standard/reward/
standard; B diverse/reward/standard; C diverse/diagnostic/standard; D diverse/
diagnostic/grounded. Pilot configuration: 2 rounds, 4 samples, 1 repeat, Qwen3-8B,
8192 context, existing sampling, checker, shapes and stopping. Each child gets
private grade files, attempt log and console log. Artifacts are exclusively
created within a new unique directory. Manifests include revision, source hashes
(including uncommitted code), settings, SDK and explicit trn2 target. Each
preparation also saves all project Python source files; actual child experiments
execute that saved snapshot, preserving the prepared code through later edits.

JSONL records endpoint prompt/completion/total usage exactly when reported; absent
values are null. Finish reason, truncation, per-request and round timing, checker
timing, per-shape numerical/traffic/hazard results, policies and selection are
recorded. Unexecuted shape checks are explicitly unevaluated; they are not passes.
Round timings exclude JSONL serialization; the runner also records full child
wall time. Summaries count each round once, not once per duplicate log row.
Trials report solve rate, best/mean candidate reward, rounds and generated
candidates to success, diversity, recurrence, usage and time. Failed attempts
remain censored with null attempts-to-success rather than disappearing from totals.

Saved preparation:
`runs/controlled-20261010T170602-ogi36ruu/manifest.json` and its `source/` snapshot.
Saved audit, seven complete recorded repair prompts and validation summary:
`runs/phase2-3-validation-20261010T170255-_3fid91i/`.
Raw evidence directories remain ignored and are never overwritten.

## 11. Remaining weaknesses

Distinct prompts can still yield identical or wrong kernels. Structural hashes
do not establish a changed algorithm. Retrieval is intentionally small and
heuristic; ambiguous operations may receive two alternatives. Full shape dataflow
analysis and global algorithm replanning are not implemented. Repeated errors may
still stop before a better algorithm appears. The controller retains its legacy
character-based answer-budget estimate; local tokenizer checks bound added cards
but do not replace server accounting. Hardware correctness/performance remains
unverified. Shared endpoint contention can confound wall-time comparisons.

The fixed A-D pilot order is for integration checks; larger studies should vary
arm order and add retrieval-only/diagnostic-only arms to separate interactions.
User-reported baseline Runs 1 and 2 solved 0/4 levels; these are not new measured
results of this implementation. No superiority claim is made.

## 12. Next recommended improvement

Run the guarded small pilot once the baseline finishes. Inspect truncation,
actual unique-kernel counts, per-shape results and repair transitions before
expanding to 8-round, 5-repeat studies. If diversity remains low, revise generic
perspectives using recorded evidence; if it improves without correctness, inspect
the wrong-algorithm/minimal-repair limitation before building a broader planner.

## 13. Reproduction

From `/tmp/trainium-kernel-dev/projects/02-kernel-agent`:

```bash
python -B -m unittest discover -s tests -v
python -B nkibench.py --selftest
python -B kernelbench.py --selftest
python -B run_controlled.py --rounds 2 --repeat 1 --levels 1 3
```

The last command only prepares a fresh manifest. Once the baseline is finished:

```bash
python -B run_controlled.py --run --rounds 2 --repeat 1 --levels 1 3
```

The launcher independently checks active original agents before launching and
between configurations. It does not poll/wait or bypass the guard. After a clean
pilot, the same runner supports `--rounds 8 --repeat 5 --levels 1 3` for repeated
evaluation. All-standard control and improved arms share instrumentation and
private paths. Existing uninstrumented baseline artifacts remain separate.

## 14. Diff summary

This phase adds four focused modules, five test files and this report, modifies
the development controller and Phase 1 regression test, and updates two root
knowledge files. Phase 1 classifier/replay remain unchanged. Checker/reference
and original files remain unchanged. See `git diff --stat` for the cumulative
Phase 1-3 worktree diff; changes are uncommitted and source hashes are retained
in the prepared manifest for reproducibility.

Final cumulative review scope: 18 changed files, approximately 2.5k insertions
and 12 deletions, including the earlier Phase 1 and AWS review artifacts.

## Completed controlled pilot: 2026-10-10

The protected original baseline finished naturally. Its final console summary reports 0/5 solves for each of Levels 1-4. The automatic monitor then launched the authorized sequential A-D pilot. All eight children exited with code 0.

Pilot artifacts: `/tmp/trainium-kernel-dev/projects/02-kernel-agent/runs/controlled-20261010T173044-tvi8dlbf`. Each configuration used two rounds, four candidates per round, one repeat, the same model and checker, private grading files, and a frozen source snapshot.

| Level | Arm | Solved | Best reward | Mean AST-unique kernels/round | Duplicate fraction | Endpoint total tokens | Wall seconds |
|---|---|---|---|---|---|---|---|
| 1 | A_baseline | False | 0.30 | 1.0 | 0.750 | 7240 | 108.2 |
| 1 | B_diversity | False | 0.30 | 3.5 | 0.125 | 7643 | 125.8 |
| 1 | C_diversity_diagnostic | False | 0.30 | 3.5 | 0.125 | 7686 | 124.1 |
| 1 | D_full | False | 0.30 | 2.5 | 0.375 | 8532 | 114.1 |
| 3 | A_baseline | False | 0.30 | 2.0 | 0.500 | 6380 | 83.9 |
| 3 | B_diversity | False | 0.30 | 3.0 | 0.250 | 6733 | 95.4 |
| 3 | C_diversity_diagnostic | False | 0.30 | 2.5 | 0.375 | 7881 | 137.6 |
| 3 | D_full | False | 0.30 | 2.0 | 0.500 | 8848 | 118.7 |

All eight configurations remained at reward 0.30 with zero verified solves. Diversity-only increased mean AST-unique kernels from 1.0 to 3.5 on Level 1 and from 2.0 to 3.0 on Level 3. This is a diversity observation, not a correctness improvement or evidence of superiority. Grounding did not yield a verified solve in this short pilot and consumed more reported tokens than baseline. Diagnostic selection avoided repeated selected failure signatures in some trials, without improving reward.

Remaining failures included DMA shape mismatches, invalid tensor dimensions, invalid API arguments, and invalid buffer placement. The two-round budget permits only one repair round; repeated, longer controlled runs are needed before evaluating repair effectiveness. No device performance was measured.

Reproduce the pilot (creates fresh artifacts, guards the original baseline):

```bash
cd /tmp/trainium-kernel-dev/projects/02-kernel-agent
python -B run_controlled.py --run --rounds 2 --repeat 1 --levels 1 3
```
