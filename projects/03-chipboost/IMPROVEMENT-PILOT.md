# Recovery controller pilot

This exploratory treatment combines the latest P1 outer-loop and DMA feedback with
`experimental_agent.py`. It is not the original model-alone/referee comparison or
the DMA-only v2 ablation. No expert optimized kernel is given to the model.

Three parallel debugging audits identified repeated AST-identical programs, an API
reference omitted after the initial request, and incorrect operand axes/K slices.
The controller retains the API card and explicit operand indexing contract, records
the last four attempt outcomes and program identities, and returns to the best
correct kernel after two consecutive invalid candidates. Duplicates still consume
the normal evaluation budget. The model-alone arm is explicitly rejected by this
experimental entry point.

Seat-100 run `/tmp/p1-improvement-pilot-20261010-1` was launched as PID 844147 on
core 2. It uses one persistent referee worker, a fresh baseline A/A acceptance gate,
eight evaluations, Qwen/Qwen3-8B, temperature 0.6, top_p 0.95, thinking disabled,
2500 answer tokens, and the original reference_level4.py baseline. Source hashes
are recorded in its state file. No success is claimed until validated records exist.

The original comparison continues on core 3; DMA-only v2 remains queued behind it.
The pilot shares the model server, so inference latency and wall-clock throughput
are not controlled comparisons. Candidate speedups use the same-session baseline
on the pilot's separate chip core. Replicated controlled runs are needed before
attributing any gain to one of these changes.

Validation: `python tests/test_experimental_agent.py` passes three CPU tests,
including recovery after repeated failures and charging duplicate candidates to
the evaluation budget. Existing final integrated P1 regression passed all 33 cases
before this pilot was launched. Full-K staging can exhaust SBUF at sufficiently
large K; a generated optimization must still pass correctness and held-out checks.
