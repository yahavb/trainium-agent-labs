# Context Agent Notes

## Baseline harness validation

Commands run on seat-61 in `/workspace/projects/02-kernel-agent`.

- `python nkibench.py --selftest` -> passed.
- `python nkibench.py --level 1 --check reference_level1.py` -> passed 4/4 shapes.
- `python nkibench.py --level 2 --check reference_level2.py` -> passed 4/4 shapes.
- `python nkibench.py --level 3 --check reference_level3.py` -> passed 1/1 shape.

Saved logs:

- `harness-step1-selftest.log`
- `harness-step1-level1.log`
- `harness-step1-level2.log`
- `harness-step1-level3.log`

## First context-router smoke test

Command:

```bash
python agent.py --level 1 --rounds 2 --samples 1 --context 8192 --log attempts-smoke-level1.jsonl
```

Result:

- Best reward: 0.30.
- Solved: no.
- Selected cards were logged and printed.
- Failure repeated: `dma_copy requires src and dst to have the same number of elements`.

Conclusion:

- The context-router mechanics work, but this first version did not improve Level 1 in the smoke test.
- Next direction should follow the challenge recommendation: validate harness with deliberate bugs, then build a verdict-to-instruction translator.

## Current implementation change

`agent.py` now has:

- General context cards.
- Failure-category context selection.
- Compact failure ledger.
- Context card logging in attempts JSONL.


## Step 2: deliberate Level 1 break

Created `broken_level1_wrong_scale.py` from `reference_level1.py` and changed the avgpool scale from `1/(pool_size*pool_size)` to `1/pool_size`.

Command:

```bash
python nkibench.py --level 1 --check broken_level1_wrong_scale.py
```

Result:

- Harness failed it as expected: status 1.
- Rules were clean.
- Numerics failed 0/4 shapes.
- Message correctly said most elements were wrong and not a ragged-edge issue.
- Message did not identify the likely scale-factor error directly.

Takeaway:

- Harness catches the bug, but the failure message is still more verdict than instruction.
- Good next improvement: add a scale-ratio diagnosis to `describe_mismatch`, so if `got/want` is nearly constant, it says to check scalar scaling/division.

Saved log:

- `harness-step2-broken-level1.log`

## Harness improvement: scale-ratio diagnosis

Changed `nkibench.py` / `describe_mismatch` to detect broad, consistent scale errors.

Validation:

- `python nkibench.py --selftest` -> passed after patch.
- `python nkibench.py --level 1 --check reference_level1.py` -> passed 4/4 after patch.
- `python nkibench.py --level 1 --check broken_level1_wrong_scale.py` -> still failed, now with a surgical message:
  - output about 2x/4x/3x reference depending on pool size
  - check scalar scaling after reduction
  - divide by full reduction count/area

Saved logs:

- `harness-after-scale-selftest.log`
- `harness-after-scale-level1.log`
- `harness-step2-broken-level1-after-scale-diagnosis.log`

Files changed:

- `nkibench.py`: added consistent-scale-error diagnosis before generic numerical mismatch.

## Verifier audit against challenge checklist

Ran deliberate and direct checks to see whether the harness gives actionable feedback.

Checks performed:

- Static banned framework op: `broken_level1_banned_mean.py` using `np.mean`.
  - Result: statically rejected with rule violation.
- Missing final output copy: `broken_level1_no_output_copy.py`.
  - Result: failed numerics with non-finite output message.
  - Improved message now points to unfilled returned `shared_hbm` output and final `nisa.dma_copy` path.
- Prime/dimension-1 shapes for Level 1 reference:
  - `(7, 13, 13), pool=1`
  - `(1, 17, 17), pool=1`
  - Result: reference passed 2/2.
- Direct ragged-edge mismatch:
  - Result: message identifies final partial partition/free tile.
- Direct mostly-zero output:
  - Result: message points to missing copy/write path.
- Direct exact-index mismatch:
  - Result: message names exact index and says full-tile accumulation/core arithmetic, not edge.
- Direct consistent-scale mismatch:
  - Result: message points to scalar scaling/reduction count.

Tolerance:

- Current numeric tolerance is `tol=2e-2` in `describe_mismatch` / verifier path.
- We have not yet written a detailed tolerance justification; note this as remaining write-up work.

Remaining gaps:

- Need failure taxonomy counts from actual agent attempts after verifier improvements.

Saved logs:

- `harness-audit-banned-op.log`
- `harness-audit-no-output-copy.log`
- `harness-audit-no-output-copy-v2.log`
- `harness-audit-direct.log`
- `harness-audit-final-selftest.log`

## Gap fixes: tolerance and hostile-value selftest

Updated `nkibench.py --selftest` so the verifier repeatedly checks:

- hostile attention reference values with very large positive/negative scores, zeros, repeated rows, and negative values;
- missing-output diagnostics;
- consistent-scale-error diagnostics.

Tolerance justification:

- Current verifier tolerance is `tol=2e-2`, interpreted as error relative to the reference output RMS.
- This is intentionally loose for the NKI simulator loop because generated kernels may use float32/bfloat16-like accumulation and different reduction order.
- It is tight enough to reject structural bugs: wrong scale, missing copies, wrong tile bounds, wrong layout, NaNs/Infs, and ragged-edge failures all exceed it by a large margin in our audits.
- For final device timing/production-level numerical claims, the tolerance should be re-stated per operation and dtype. For this hackathon inner loop, `2e-2` is a pragmatic correctness gate, not a precision benchmark.

Remaining gap:

- Need failure taxonomy counts from actual agent attempts after verifier improvements.

## Agent improvement: reduction API routing

The Level 1 run moved from a missing `dtype` failure to an NKI tensor `.mean()` failure. That is
progress, but the old router treated `.mean()` as a generic scale error.

Updated `agent.py` so:

- `.mean()` / `has no attribute 'mean'` is categorized as `reduction_api`;
- Level 1 reduction failures receive a compact `avgpool_reduction` card;
- prompt accounting and logs now report the more precise category.

Local validation:

- `python3 -m py_compile projects/02-kernel-agent/agent.py projects/02-kernel-agent/nkibench.py`
- Direct `distill_failure(...)` check returns `reduction_api` and cards
  `['avgpool_reduction', 'reductions', 'signatures']`.

## Agent instrumentation

Added two no-model utilities to `agent.py`:

- `--audit-context`: prints selected cards and estimated token budget for representative failures.
- `--summarize-log attempts.jsonl`: summarizes failure categories, cards, and prompt budget from a run.

Validation:

- `python3 projects/02-kernel-agent/agent.py --level 1 --audit-context`
- `python3 projects/02-kernel-agent/agent.py --all --audit-context`
- fake JSONL summary test for `--summarize-log`

Observed Level 1 context sizes:

- first prompt: about 122 tokens;
- `.mean()` repair: about 315 tokens with `avgpool_reduction`, `reductions`, `signatures`;
- DMA-shape repair: about 276 tokens.

## Agent improvement: reduction axis routing

Real Level 1 run found the next failure after `.mean()`:

- `tensor_reduce axis must be the last contiguous dim(s) ... Got axis=(2, 4)`

Updated `agent.py` so this maps to `reduction_axis` and sends a focused card explaining that
`nl.sum` can only reduce trailing contiguous dimensions, so the avgpool access-pattern view must
place pool dimensions last and reduce `axis=[3, 4]`.

## Prompt/reply inspection

Added debug tooling to inspect exactly what the agent sends to the model:

- `--dump-prompts`: stores full prompt and raw model reply in the JSONL log.
- `--show-attempt PATH --show-round N`: prints prompt, raw reply, extracted code, and checker
  feedback for one attempt.

This is for auditing whether failures are due to missing context, bad model reasoning, or bad
extraction/checking.

## Agent improvement: broader base context and multi-fix repair

Prompt inspection showed Round 0 invented NumPy-like NKI calls:

- `.reshape()`
- `.mean()`
- `.copy_from()`
- missing `dtype`

Round 1 followed the prompt too literally: it fixed only `dtype` and kept the known-invalid calls.

Updated `agent.py` so:

- the base NKI card is still small but explicitly says NKI tensors are not NumPy arrays;
- repair prompts say to fix the current error plus known invalid API patterns from previous
  failures;
- known invalid patterns are deduced from the unique failure ledger.

Observed audit sizes remain small:

- first prompt: about 188 tokens;
- `.mean()` repair: about 344 tokens;
- reduction-axis repair: about 415 tokens.
