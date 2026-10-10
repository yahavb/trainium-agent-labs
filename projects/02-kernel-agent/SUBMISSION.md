# Project 2 Submission Notes

This is the final hand-in guide for the kernel agent.

Team: 13

Submitted by: Gyanasri Konda

## Final Results

Final command used:

```bash
python agent.py --all --rounds 4 --samples 1 --context 8192 --prompt-style reference --dump-prompts --log submission-reference.jsonl
```

Result summary:

- Verifier self-test: `SELFTEST PASSED`
- Levels solved: `2/4`
- Level 1 average pooling 2D: reward `0.30`, not solved after 4 rounds.
- Level 2 2D transpose: reward `0.30`, not solved after 4 rounds.
- Level 3 single-tile matmul: reward `1.00`, solved on round 0.
- Level 4 tiled matmul: reward `1.00`, solved on round 0.

Observed solved kernels:

- Level 3: correct on every shape; reported memory-bound at `19.7 Flops/Byte`, `11.3x` more reuse needed.
- Level 4: correct on every shape; reported memory-bound at `36.6 Flops/Byte`, `6.1x` more reuse needed.

Failure taxonomy from `submission-reference.jsonl`:

- `generic`: 7 attempts
- `dma_shape`: 3 attempts

Failure keys:

- `dma.shape_mismatch`: 3
- `generic.0_of_4_shapes_passed_on_c_h_w_32_32_32_pool_2_raised_a`: 2
- `generic.the_code_does_not_parse_invalid_syntax_on_line_3_send_one`: 2
- `generic.0_of_4_shapes_passed_on_c_h_w_32_32_32_pool_2_raised_n`: 1
- `generic.correct_on_every_shape_memory_bound_19_7_flops_byte_agains`: 1
- `generic.correct_on_every_shape_memory_bound_36_6_flops_byte_agains`: 1

Token/context examples from the final run:

- Level 1 round 0: `prompt~5775`, `limit=8192`, `docs~4983`, `code~675`
- Level 2 round 0: `prompt~5880`, `limit=8192`, `docs~5071`, `code~693`
- Level 3 round 0: prompt was about `5890` tokens and solved.
- Level 4 round 0: prompt was about `6023` tokens and solved.

## What To Hand In

- `agent.py`: the agent loop, context manager, retry logic, confidence-by-verification behavior, token accounting, and failure taxonomy logging.
- `nkibench.py`: the verification harness.
- `reference_level1.py` through `reference_level4.py`: reference-style kernels used by the optional `--prompt-style reference` mode.
- Final run logs:
  - `submission-reference.jsonl`
  - `submission-reference-summary.txt`
  - `submission-context-audit.txt`
  - `submission-verifier-selftest.txt`
- This reproduction note.

## Final Run Commands

Run from inside the seat container:

```bash
cd /workspace/trainium-agent-labs-fork
git pull
cd projects/02-kernel-agent

python nkibench.py --selftest | tee submission-verifier-selftest.txt

python agent.py --all --rounds 4 --samples 1 --context 8192 --prompt-style reference --dump-prompts --log submission-reference.jsonl

python agent.py --summarize-log submission-reference.jsonl | tee submission-reference-summary.txt

python agent.py --all --audit-context | tee submission-context-audit.txt
```

Optional: also run the generic version for honesty/comparison:

```bash
python agent.py --all --rounds 4 --samples 1 --context 8192 --prompt-style full-docs --dump-prompts --log submission-generic.jsonl

python agent.py --summarize-log submission-generic.jsonl | tee submission-generic-summary.txt
```

To inspect a specific failure/recovery for the demo:

```bash
python agent.py --show-attempt submission-reference.jsonl --show-round 0 > demo-round0.txt
python agent.py --show-attempt submission-reference.jsonl --show-round 1 > demo-round1.txt
```

## Verification Harness

The harness is `nkibench.py`.

It checks:

- parsing and import validity;
- static rule violations, including illegal framework shortcuts and invalid API patterns;
- execution against NumPy references;
- hostile and uneven shapes, including dimensions that do not divide cleanly by tile sizes;
- useful failure descriptions, including shape mismatch, out-of-bounds, nonfinite output, scale errors, partial-tile hints, and traffic hints.

Tolerance:

- Default tolerance is `tol=2e-2`, measured relative to the reference output RMS.
- This is intentionally loose enough for simulator/kernel differences and float32/bfloat16-like accumulation order.
- It is still strict enough to catch structural bugs: wrong scale, wrong copy shape, missing output writes, NaN/Inf, wrong tile bounds, and ragged-edge errors.

Self-test:

```bash
python nkibench.py --selftest
```

## Eval Set

The eval set is embedded in `nkibench.py` under `LEVELS`.

It includes:

- Level 1: 2D average pooling across multiple `C,H,W,pool` shapes.
- Level 2: 2D transpose with reshaped/tiled layouts.
- Level 3: single-tile matmul.
- Level 4: tiled matmul.

The harness also tests hostile conditions such as:

- uneven dimensions;
- final partial tiles;
- dimensions of size `1`;
- large/negative/zero/repeated values where relevant;
- all-zero/missing-output detection;
- nonfinite output detection;
- consistent-scale mismatch detection.

## Failure Taxonomy

`agent.py --summarize-log LOG.jsonl` prints the taxonomy.

The main failure groups are:

- `signature`: invalid NKI API call, wrong namespace, unsupported keyword, missing dtype, or called memory region.
- `dma_shape`: `nisa.dma_copy` source and destination element counts do not match.
- `tile_rank`: illegal 1D SBUF/PSUM tile.
- `reduction_api`: NumPy-like reductions such as `.mean()` instead of NKI reductions.
- `reduction_axis`: reduction axes are not trailing contiguous axes.
- `rule`: static rule violation or forbidden framework shortcut.
- `numeric`: wrong values, nonfinite output, missing output write, or scale mismatch.
- `traffic`: correct result but excessive HBM traffic or issue-bound behavior.
- `generic`: verifier result did not match a known bucket; inspect the checker text.

For the write-up, use the counts from:

```bash
python agent.py --summarize-log submission-reference.jsonl
```

## Token Instrumentation

Every logged attempt includes:

- `prompt~...`: estimated input tokens used;
- `limit=8192`: configured input-token budget;
- `docs~...`: docs/cards/reference spend;
- `evidence~...`: verifier feedback and ledger spend;
- `code~...`: current candidate/reference-code spend;
- `cards=...`: context cards selected.

This is printed by:

```bash
python agent.py --summarize-log submission-reference.jsonl
```

## Confidence And Honesty

The agent does not report success unless the verifier returns full reward for that level.

Interpretation:

- reward `1.00`: verified on the local harness for that level;
- reward below `1.00`: not verified; report as failed or partially successful;
- parse/rule failures are not hidden; they are logged and counted.

This is important for calibration: a failed level is reported honestly instead of being called done.

## Demo Suggestion

Show:

1. One failed attempt from the JSONL log.
2. The verifier message explaining what failed.
3. The next prompt containing distilled feedback, ledger, and token accounting.
4. The final summary showing which levels passed and which failure modes remained.

Use:

```bash
python agent.py --show-attempt submission-reference.jsonl --show-round 0
python agent.py --show-attempt submission-reference.jsonl --show-round 1
```
