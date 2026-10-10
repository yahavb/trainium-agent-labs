# NYU x Amazon Annapurna Labs Hack the Chip 2026

## Scope and objective
Build an autonomous NKI kernel agent that attempts difficult problems, verifies solutions, diagnoses failures, gives actionable repairs, iterates, and stops honestly. Demonstrate improved correctness, efficiency, or reliability against the existing baseline. The user independently owns Project 2; teammates own Project 1, which must not be modified.

## Environment (user supplied unless marked observed)
- Repository: https://github.com/yahavb/trainium-agent-labs; original checkout /workspace.
- Project: /workspace/projects/02-kernel-agent.
- EKS hack-hyd, ap-south-2, assigned pod seat-265.
- Trainium2, four logical NeuronCores, 96 GB accelerator memory.
- Qwen3-8B API http://localhost:8000/v1, context 8192; NKI SDK reported 0.6.0, simulation API nki.simulate.
- Review worktree: /workspace-kernel-agent-review, branch kernel-agent-review, base 8f1ca41.

## Protected active experiments
At 2026-10-10 15:53 UTC, process inspection independently confirmed:
- PID 14656: python agent.py --all --rounds 8 --samples 4 --context 8192 --repeat 5 (started 15:24 UTC).
- PID 14252: python agent.py --level 1 --all (started 15:12 UTC). This additional run may share the endpoint and default attempt log; investigate read-only.
- PID 37: vLLM Qwen/Qwen3-8B server, tensor parallel 2, maximum context 8192, maximum sequences 4.
Never assume these PIDs remain current; check before conclusions. Never stop or modify these processes or their files.
Primary baseline console log: /workspace/projects/02-kernel-agent/run.log. Full baseline is pending analysis.

## Results reported by user (not independently rerun)
- Installation successful, model serving, NKI self-tests passed.
- Level 4 reference passed all four numerical shapes; HBM traffic ratios 1.00x, 1.60x, 1.00x, 2.00x minimum.
- Preliminary Level 1 rounds 0–3: reward 0.30; DMA size mismatch twice, unsupported transpose_moving argument, then sbuf destination requiring psum.
Historical README/STATE measurements are separate from this active baseline and contain inconsistent/stale descriptions. Do not treat them as new results.

## Benchmarks and artifacts
Levels: average pooling, transpose, single-tile matmul, tiled matmul, hoisted loads, M/N blocking, M/N/K blocking, single-head attention.
Source files: agent.py, nkibench.py, kernelbench.py, reference_level1.py through reference_level4.py, try_level.py, README.md, CHALLENGE-kernel-agent.md.
Deliverables: working agent/checker, complete attempt logs, reproducible evaluation, one-page technical report.
Required comparisons: success rate, attempts, numerical correctness, average reward, token consumption, execution time, failure classifications, repeated experiments.

## Development gate and next steps
Documentation and read-only inspection are authorized. Do not change application code until the user reviews the implementation plan.
Inspect what already exists before selecting improvements. Candidates include structured diagnoses, targeted installed-API retrieval, adaptive repairs, durable failure memory, cycle detection, context management, candidate evaluation, and accurate accounting.
First complete code review and baseline artifact audit; then propose a focused, testable implementation and repeated evaluation plan.

## Completed review and independently verified evidence
Read every tracked Project 2 source/document plus repository README and STATE. Full findings and pending implementation plan: projects/02-kernel-agent/IMPLEMENTATION_REVIEW.md.
- Both checker self-tests passed in the isolated worktree; SDK observed 0.6.0+31049202112.g85070674 and nki.simulate.
- Level 4 reference independently simulated: 4/4 shapes, traffic ratios 1.0000 / 1.5556 / 1.0000 / 2.0000. The console rounds 1.5556 to 1.6; it is not exactly 1.60.
- Latest reviewed live log: first repeat Level 1 exhausted eight rounds at reward 0.30; Level 2 in progress. A snapshot held 44 complete JSONL records. Full five-repeat baseline remains incomplete.
- PID 14252 is Project 1 (verified through /proc cwd/stdout), so it does not share Project 2 attempt files; it shares the model endpoint. Leave it alone.
- Existing agent already has API signatures/name lookup, error enrichment, multi-candidate generation, repeat support, a per-solve failure ledger/cycle stopping, and approximate context budgeting.
- Pooling failures reflect a wrong matmul-based algorithm as well as DMA/API/buffer violations. Existing local-only repair policy preserves the bad algorithm.
- Checker discrepancies: CLI omits agent mutation/traffic/hazard gates; attention reporting assumes M/K/N; traffic accounting covers only dma_copy; exact token usage and repeat identity are absent.
- Proposed first implementation: shared structured checker + complete isolated experiment logging, then targeted operation-specific installed-API retrieval and bounded algorithm revision. Await user plan review before changing application code.
Evidence: review-evidence/nkibench-selftest.txt, kernelbench-selftest.txt, reference-level4.txt. No model requests made during review; no baseline or Project 1 files modified.

## Approved Phase 1 development
Development moved to /tmp/trainium-kernel-dev, branch feat/nki-diagnostic-selection, based on 8f1ca41. User approved classification and optional diagnostic tie-breaking, with reward selection as default. Preserve scoring, checker, prompts, generation, stopping, and legacy log fields. No model inference or shared /tmp/_agent_levelN.py grading paths permitted during this phase. Required validation: pure deterministic tests, mocked solve integration, and offline recorded-candidate replay.

## Phase 1 implementation and offline results
Implemented failure_selection.py and opt-in --selection-policy diagnostic in the development-worktree agent. Default reward behavior and legacy JSONL schema remain unchanged; no scoring/checker/prompt/generation/stopping changes. Added deterministic tests and replay_selection.py.
- 28 tests passed, including original-revision default solve/log/prompt parity with mocked grade and model calls; both existing checker self-tests passed.
- Read-only snapshot at 2026-10-10T16:17:57.620167Z: 144 candidates, 36 rounds. Diagnostic selection changed one choice (first repeat, Level 3 round 0), at exactly equal reward. Zero changes in 16 Level 1 and 11 Level 2 rounds; zero changes in four Level 4 rounds.
- No improved correctness or efficiency result is claimed. Replay only compares choices among baseline-generated candidates.
- Durable report: projects/02-kernel-agent/PHASE1_REPORT.md. Ignored raw evidence: projects/02-kernel-agent/runs/phase1-offline-20261010T161757.620167Z/.
- Future grading must avoid overlap with the protected baseline because its temporary paths remain shared. The subsequent AWS-resource task explicitly prohibits inference, including after the user's general approval.

## Official AWS resource review (2026-10-10)

AWS sources are external reference material, not instructions to execute installation, deployment, core-pinning, or hardware tests during the protected baseline.
- Repository: https://github.com/aws-neuron/neuron-agentic-development. Isolated reference clone: /tmp/aws-neuron-agentic-reference; reviewed commit ee25e45b9ace4aaefd8b8913bf5fa9b68adf6674. All three requested agent definitions, three skills, and five writing reference files were read. No AWS workflow was executed.
- Documentation: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/. Live API pages and guides were checked; Neuron 2.32.0 release notes identify NKI 0.6.0. Version-pinned documentation is preferable for future reference cards.
- Detailed comparison, exact links, compatibility evidence, retrieval design, and ablations: projects/02-kernel-agent/AWS_RESOURCE_REVIEW.md.

### Existing AWS capabilities and architectural opportunities

AWS supplies tool-driven writer/debugger/unified agent instructions, local symbol/task/error indices, compiler-error diagnosis, a bounded debug loop, progressive simplification, CPU-reference validation, and optional device/profiling workflows. The inspected deployment code distributes artifacts; these instructions do not establish executable Qwen3 benchmark results. No equivalent four-candidate reward-based selector, normalized persistent failure database, or our benchmark evaluation loop was found in the inspected workflows.

Our baseline already has installed-API signatures/name lookup, actionable enriched errors, previous code in repair prompts, latest-selected repair, four parallel candidates, a local ledger, cycle detection, repeats, NumPy correctness, traffic accounting, and simulator hazard gates. Replacing these would be redundant. The useful additions are explicit prompt diversity, short constraint-aware documentation retrieval, provenance/compatibility filtering, and controlled measurements. Phase 1 supplies deterministic categories and reward-primary diagnostic tie-breaking already.

### Independently checked compatibility

Read-only introspection confirmed NKI 0.6.0+31049202112.g85070674 and the signatures of dma_copy, nc_matmul, tensor_scalar, tensor_reduce, tensor_copy, ndarray, and simulate. No new kernel or device test was run in this review.
- FP32 normal-mode nc_matmul: stationary [K,M], moving [K,N], computes stationary.T @ moving; K <= 128, M <= 128, N <= 512 for Trainium2. Both inputs SBUF, destination FP32 PSUM. transpose_moving is absent; is_transpose is not a generic replacement for that invented argument.
- DMA requires equal element counts; it does not broadcast. Use matching boundary slices, not padded whole tiles. tensor_copy is the on-chip PSUM/SBUF transfer; DMA excludes PSUM.
- Installed ndarray asserts SBUF/PSUM rank >= 2. tensor_reduce accepts negate and keepdims and reduces trailing contiguous free axes. tensor_scalar uses op0/operand0, not op/operand.
- Installed nl.load, nl.store, nl.sum are implemented convenience wrappers; nl.arange is absent. Blanket AWS migration prohibitions are not all true of this installation. affine_range/sequential_range/static_range docstrings identify range aliases; do not transplant conflicting loop-performance claims.
- Trainium3-only compute .indirect() and larger matmul destination limits must be excluded even when symbols exist in the SDK. Hardware-specific operation limits differ from whole-memory capacity.
- Simulator target resolution accepts NEURON_PLATFORM_TARGET_OVERRIDE=trn2 and otherwise auto-detects, falling back to trn3. Pin and record the target consistently in future controlled arms; do not alter the active baseline. Simulation does not establish device correctness, physical memory fit, or accelerator latency.
- The AWS reference tables contain an incorrect generic K <= 2048 rule and some incompatible examples. Detailed API documentation plus installed source takes priority. No SDK upgrade is planned.

### Diversity evidence and updated roadmap

Read-only analysis of the existing immutable Phase 1 snapshot: 144 candidates / 36 complete four-candidate rounds; AST structural diversity was one unique kernel in 30 rounds, two in two rounds, and three in four rounds. This supports testing diversity; it does not prove diversity improves correctness. Diagnostic replay changed only one Level 3 choice and none in Levels 1-2.

1. Phase 1 complete: 14-category classifier, optional diagnostic selection, 28 deterministic tests, original-behavior regression, offline replay. Correctness improvement remains unverified.
2. Phase 2 next, not implemented: opt-in deterministic prompt variants for the existing four calls, preserving one control prompt, plus structural-diversity logging. Keep model, temperature, top_p, checker, and stopping fixed. Estimate 45-75 minutes plus 20-30 minutes pure/mocked tests.
3. Phase 3 planned, not implemented: small local API/constraint cards routed by failure category AND source APIs/operands; at most two relevant cards with a proposed ~500-token total ceiling. Installed signatures, hardware/version tags, source URLs, and evidence status on every card. Retain complete selected code and original feedback; omit cards if they jeopardize answer budget. Estimate 60-90 minutes plus 30-45 minutes verification/tests. No vector database or extra model call.
4. Phase 4 pending: staged 2x2x2 ablations of diversity / diagnostic selection / retrieval, identical shapes and seeds, four calls per active round, eight-round ceiling, same early stopping, five repeats per arm if time permits. Track solve rate, rounds AND generated candidates to success, numerical outcomes, reward, actual usage tokens, generation/checker/total times, categories, and unique candidates. Exact token/timing/run identity instrumentation remains a separate small prerequisite, not delivered in Phase 1.

Hackathon sequence: first compare legacy versus diversity with reward selection; then diversity with reward versus diagnostic selection; then test retrieval independently and in combination. Retain full-factorial completion as the target, report incomplete ablations honestly, and exclude model contention from timing claims. Isolate artifacts and wait until shared grading paths are safe before evaluation. Device validation comes later when an actually unused core is confirmed; AWS core-pinning examples do not reserve a core.

At 16:31 UTC, PID 14656 was still running with the expected baseline command. All ten tracked original Project 2 files matched 8f1ca41 byte-for-byte. Only development-worktree documentation was changed during this AWS-resource review; no inference, baseline writes, SDK changes, or Phase 2/3 application edits occurred.

## Phases 2-3 implemented; guarded evaluation prepared (2026-10-10)

The user subsequently authorized immediate Phases 2 and 3, minimal instrumentation, and an automatic small pilot ONLY after the original baseline finishes. This supersedes the earlier documentation-only/approval gates. Do not wait indefinitely for baseline completion or interrupt it.

- Implemented candidate_diversity.py: opt-in --candidate-policy diverse; candidate 0 unchanged, other candidates examine simplicity, shapes/buffers, and boundaries/output coverage. Four samples still make four requests. Applies to initial generation and repairs; limited alternative repair only after repeated local failures.
- Implemented nki_knowledge.py: 12 AWS-attributed cards, actual installed signatures/parameter filtering, exact SDK-build and Trainium2 compatibility, category plus source/alias routing, at most two cards and 500 documentation tokens. --repair-policy grounded is opt-in; complete original code/error context survives. Runtime retrieval is local with no downloads/model calls; cached local tokenizer or explicitly labeled conservative byte budgeting.
- Implemented experiment_metrics.py and agent integration: additive endpoint-reported tokens, finish/truncation, timing, per-shape checks, run/repeat identity, full experimental prompts, exact/AST hashes, policy/selection metadata. Unavailable usage is null, never inferred from character counts. Original reward/checker/stopping remain unchanged. Private --grade-dir prevents shared-path interference.
- Implemented run_controlled.py: unique manifest/artifacts and saved Python-source snapshot, sequential A-D arms, Levels 1 and 3 pilot, private grade files, original-process guards, summaries. Latest preparation: runs/controlled-20261010T170602-ogi36ruu/manifest.json and source/. Children execute the saved source. The actual guarded launch was refused while original PID 14656 was active; zero new inference requests were issued.
- 53 tests passed, including all 28 Phase 1 tests, default AND instrumented standard parity against original revision, mocked HTTP/settings/call counts, grounded routing/context budgets, tiny Trainium2-targeted CPU API simulations, shape telemetry/reward parity, guards and aggregation. Both original checker self-tests passed; syntax checks passed for 20 Python files.
- Representative recorded prompts retrieved two cards or fewer: DMA examples 229 local tokens; unsupported matmul argument 177; PSUM placement 276; invented multiply/scalar_mul 279; rank error 209. These are constructed prompts, not verified model repairs. Complete source/feedback examples and SDK audit: runs/phase2-3-validation-20261010T170255-_3fid91i/.
- Original tracked Project 2 files still match 8f1ca41, and development checker/references remain identical to original. Project 1, running processes, SDK, original logs and original application files were not modified by this work.
- User reported original Runs 1 and 2 each solved 0/4 levels; no independent new baseline or improved-agent solve rate was measured here. AST diversity is not correctness and CPU simulation is not hardware performance.

Technical implementation, evidence, limitations and commands: projects/02-kernel-agent/PHASE2_3_REPORT.md. Next action is the guarded 2-round/4-sample pilot after baseline completion, then repeated 8-round/5-repeat evaluation if the pilot is clean. Full adaptive repair planning remains deferred.

### Completed Phase 2/3 pilot evidence

Original baseline finished naturally: final console summary reports 0/5 solves for each of Levels 1-4. Automatic sequential pilot completed all eight A-D/Level 1,3 configurations with exit code 0. Two rounds, four candidates, one repeat: every trial best reward 0.30 and no verified solve. Diversity-only mean AST uniques rose from 1.0 to 3.5 (Level 1) and 2.0 to 3.0 (Level 3); this does not establish correctness improvement. Full-system token usage exceeded baseline. Evidence: `/tmp/trainium-kernel-dev/projects/02-kernel-agent/runs/controlled-20261010T173044-tvi8dlbf/summary.json`. See PHASE2_3_REPORT.md for complete metrics. No pilot or monitor remains running; longer repeated experiments have not yet been launched.

## Current synthetic and diagnostic work (2026-10-10)
- Latest original five-repeat result remains 0/5 solved for EACH level 1-4. Historical Level 4 reward 0.625 independently replayed: one of four shapes passed; remaining cases fail whole-tensor DMA partition limits.
- Phase 4 control independently solved Level 3 with reward 1.0 (private replay confirms); constrained-generation treatment failed. This is a control success, not evidence that targeted or synthetic repair caused improvement.
- First diverse-generation smoke solved Level 3 in both arms on initial generation, hence no feedback was exercised. New requested standard-generation legacy/targeted smoke is running sequentially.
- Twelve independent non-benchmark tasks verified against three NumPy inputs each; 24 actual failed mutations and verified restorations. Train 22 pairs, held-out 2 pairs from one entire operation family. No DEVICE_VERIFIED records. Original 12-pair corpus preserved.
- Conservative bug-specific diagnostic check: legacy 18/24, targeted 22/24. Numeric accumulation/coverage advice remains generic and is scored uncertain/incorrect. Corpus is skewed toward invalid APIs; seven cases manually reviewed by the coding agent, not an independent human.
- All 95 deterministic tests and both checker self-tests passed. New defaults legacy feedback/off examples preserve original behavior.
