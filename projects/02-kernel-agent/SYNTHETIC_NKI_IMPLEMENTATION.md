# Synthetic NKI implementation

Development only: `/tmp/trainium-kernel-dev`, branch `feat/nki-diagnostic-selection`. Original checkout, checker, reference kernels, tolerances, reward weights, model server and SDK remain protected. The system changes inference-time prompting and memory; it does not fine-tune Qwen or train model weights.

## Architecture

`shape_repair.py` reuses conservative AST inspection, checker-case argument binding and installed SDK constraints. `--feedback-policy targeted` adds one precise explanation and permits dependent lines to change together. `--feedback-policy legacy` is the default. Original code and checker error remain in the prompt. If targeted feedback is selected, its evidence replaces additional generic grounded cards rather than appending several unrelated cards. Existing `standard`, `grounded` and `shape-aware` repair flags remain supported.

`synthetic_nki/generate.py` constructs 12 independent tiny tasks; `mutations.py` introduces the primary error per task; `generate_pairs.py` adds one single-call invalid-DMA-API variant per task; `verify.py` checks syntax, imports/decorator, actual CPU execution, untouched inputs, independent NumPy numerics and simulator hardware warnings. `retrieval.py` indexes only compatible, verified, structurally unique training records and inserts up to two short before/after statements. It never inserts a complete kernel. `--example-policy synthetic` enables this; `off` is the default.

`repair_history.py` stores six signatures/source hashes and a separate best verified candidate. Optional `--adaptive-repair` escalates repeated corrections to dependent dataflow changes, preserves the best verified source when a new selection regresses, and caps added history to available context. A changed first failure is explicitly a hypothesis, not proof that its hidden constraint is repaired.

`run_controlled.py --synthetic-pilot diagnosis-smoke|diagnosis-ablation|iteration` uses process/endpoint guards, sequential configurations, exclusive directories, private grading and frozen Python/dataset snapshots with SHA-256 manifests. Generation still uses exactly four calls per round with samples=4. No retrieval/controller model calls are added. `analyze_experiments.py` reads every attempt and reports per-shape verification, repair transitions, actual endpoint usage and wall/checker/available simulation timing.

## Corpus and verification

12 AST-distinct clean kernels, 24 independently observed repair pairs over those 12 kernels, 22 train and 2 held-out records. The entire elementwise arithmetic family is held out, and held-out files are never indexed. Families cover bounded DMA, scalar/binary arithmetic, sum/max reductions, on-chip regions, small matrix products with an offset, disjoint K accumulation, partition copies and column output coverage. They are not official benchmark entry points or reference solutions. Matrix-product tasks include a separate offset and tiny constrained layouts; retrieval exposes only the changed invariant.

Each clean and restored task passed three seed-varied float32 input cases against independent NumPy operations at rtol=atol=2e-5. All mutations failed with the intended observed category. Current corpus: 1 DMA mismatch, 5 tensor-dimension failures, 14 invalid API functions, 1 invalid API argument, 2 numerical failures and 1 buffer-placement failure. No mutations were rejected in the initial build. Unit tests also inject a non-failing mutation and confirm rejection. The 2/2 held-out restored-pair verification is a data-quality check, not held-out LLM accuracy. Device-verified examples: zero.

Limitations: only float32 so far; tiny fixed layouts; one held-out family/kernel with two injected bugs; finite input tests are not formal proof. Uninitialized missing-output behavior is caught by numerical comparison, not attributed to a new official checker category. Static checks are deliberately limited and simulator execution supplies the hardware-layout gate. Generated-code licensing is unspecified by this repository; no external implementation was copied. Source, broken result, actual error, repaired result, seed/SDK/target and provenance are retained in JSONL.

## Compatibility and source attribution

All executable patterns were tested with NKI `0.6.0+31049202112.g85070674`, target Trainium2. SDK/source inspection and tiny simulator tests confirmed reduction-created rank loss and matmul's internal result reshape. Wan2 signature conflicts and device-planning limits are in `WAN2_REUSE_REVIEW.md`; no Wan dependencies were installed.

- [AWS NKI Synthesizer](https://github.com/aws-neuron/nki-synthesizer): layout-aware sketches, loop kinds, CEGIS counterexamples and equality-saturation ideas. Its Rosette/Egglog infrastructure was inspected but not installed.
- [AWS NKI Library](https://github.com/aws-neuron/nki-library): tile-grid, callback and independent-test organization; current repository explicitly warns that compiler compatibility is revision-dependent. No complex allocator or reference kernel imported.
- [AWS NKI Samples](https://github.com/aws-neuron/nki-samples): matching tile loads, destination-style arithmetic and output slices; tutorial divisibility assumptions are not general arbitrary-shape guarantees.
- [Amazon kernel-writing tutorial](https://neuron-science.github.io/llm_kernel_writing/): verification, cheating detection, inference scaling and execution-feedback learning motivate strict validation/attribution.
- [DRTriton](https://arxiv.org/abs/2603.21465): controlled synthetic operation difficulty and execution-verified learning motivate a small validated corpus; this project does not reproduce its RL training or reported performance.
- [Kevin](https://arxiv.org/abs/2507.11948): multi-turn execution feedback and serial refinement motivate bounded failure history; CUDA implementations/results are not NKI evidence.
- [KernelLLM](https://huggingface.co/facebook/KernelLLM): externally curated PyTorch/Triton pairs motivate avoiding benchmark-solution leakage. Its model was not downloaded or served.

Read-only reference revisions: NKI Synthesizer `967838d771d4ba21b023a1f05cb86584cbcd78b7`, Library `92d11f63a9a8ec1ade34e6e1a3b8db66ef31307e`, Samples `62c300da468b34655aa1fc98a70d807c8dad4a72`. External code is not assumed compatible merely because an API exists.

## Reproduction

```bash
cd /tmp/trainium-kernel-dev/projects/02-kernel-agent
python -B -m unittest discover -s tests -q
python -B nkibench.py --selftest
python -B kernelbench.py --selftest
# Fresh corpus directory; generator refuses to overwrite existing data.
artifact_dir=$(mktemp -d "$PWD/runs/synthetic-rebuild.XXXXXX")
python -B -m synthetic_nki.generate --output "$artifact_dir/data" --seed 2026
python -B run_controlled.py --run --synthetic-pilot diagnosis-smoke --rounds 2 --samples 4 --repeat 1 --levels 3
python -B run_controlled.py --run --synthetic-pilot diagnosis-ablation --rounds 4 --samples 4 --repeat 1 --levels 3
```

All original default-behavior and equal-reward selection regressions remain in the test suite. Live results and final validation count are recorded separately in `SYNTHETIC_NKI_EXPERIMENTS.md`.

## Diagnostic assessment and updated isolation

The preserved initial corpus is `synthetic_nki/data/`; default retrieval now uses `data_v2/`. Pair deduplication keys correct structure plus cause so two distinct actual bugs in the same clean kernel remain useful, while duplicate cause variants are removed. All held-out-family variants are excluded. `diagnostics.py` applies predeclared bug-specific semantic checks to added legacy/targeted guidance, excluding raw errors. Conservative score is 18/24 legacy versus 22/24 targeted; generic numerical guidance does not count as a confirmed cause. The coding agent inspected seven source/error/diagnosis pairs; no independent human annotation was obtained. Full per-category evidence is in SYNTHETIC_DIAGNOSTIC_EVALUATION.md. This evaluates diagnosis coverage, not Qwen correctness or population accuracy.

Updated A–C live arms hold standard generation, diagnostic selection and standard repair constant. A uses legacy feedback; B uses targeted feedback; C adds synthetic cards. D adds diverse generation and adaptive history to targeted/cards. D is a bundle, so its effects cannot be attributed to one intervention. The earlier smoke used grounded repair and is labeled separately. SDK verification confirmed nl.load and nl.store exist. 95 tests and both unmodified checker self-tests passed before the ablation.

Rebuild 24 pairs after creating a fresh base corpus:
```bash
python -B -m synthetic_nki.generate_pairs --base "$artifact_dir/data" --output "$artifact_dir/data_v2"
```

## Verified curriculum extension

Original datasets remain intact. Default retrieval now uses data_v4: 16 distinct clean kernels, 28 verified repair pairs, 26 train and 2 held-out pairs. Added independent offset transpose, grouped sum, row-output tiling and ragged K accumulation. The initial extension rejected one boundary mutation with a wrongly anticipated DMA category; actual OUT_OF_BOUNDS feedback was retained in data_v3/summary.json and the expectation was corrected/reverified. No additional model calls and no device validation.

Final 28-pair guidance check: legacy 20/28, targeted 25/28, targeted plus symbolic 25/28. Symbolic constraints detect nine injected shape/layout causes, with zero false-positive violations on sixteen verified clean kernels. The three numerical bugs remain unconfirmed specifically. All emitted source locations refer to actual AST calls/allocations; this is not independent runtime localization proof. Current concrete corpus has no unresolved supported-shape analyses after bounded loop reasoning; API and numerical legality remain outside this status. Exact timings and records: synthetic_nki/data_v4/symbolic_evaluation_final.json and diagnostic_evaluation.json.


## Operation-aware primitive and LoRA update (2026-10-10)

Latest evidence is summarized in LEVEL_SCORECARD.md, PRIMITIVE_CORRECTNESS_SPRINT.md, LORA_TRAINING_REPORT.md and LORA_COMPARISON.md. Full untuned L1/L2 remained .30, zero numerical cases. Current cold L1 planner/legalizer has a provisional .50 execution-only candidate, zero shapes; no cold solve claimed. A generic deterministic instruction-legalizer warm replay reached L1 1.00, 4/4 cases, separately labeled and excluded from cold-start rates/training. L3 remains 1.00; L4 .625 (1/4). No hardware performance measured.

Corpus data_v8: 21 independently simulator/NumPy-verified clean kernels, 38 verified error/restoration pairs, 36 training/retrieval candidates and two held-out records. Existing retrieval and actual LoRA training remain on frozen data_v4; later data did not enter the running training. Diagnosis rubric on 38 actual injected failures: legacy 21 correct, targeted 30, targeted+SymPy 30; ten supported symbolic causes, four UNKNOWN, zero clean static false-positive kernels among 21. These are task-specific offline checks, not general diagnostic accuracy or proof of live improvement.

LoRA has completed 82 optimizer steps on 41 examples (15 generation, 26 repair), with three held-out examples from one completely excluded family. Rank 8/alpha 16, q_proj/v_proj, lr1e-4, completion-only loss, batch1; CPU training avoided Neuron device resources. Held-out mean example loss .9951169491 -> .1274786662 is not correctness evidence. Adapter saved in runs/qwen3-nki-lora-mdzxuq1k/training/adapter. The preserved full-adapter evaluation waits for active cold-run completion; a matched original/LoRA x legacy/planner study follows sequentially on the same isolated CPU backend. New held-out repairs are queued afterwards. Shared Qwen/Neuron vLLM was not restarted or modified.

SDK-verified miniature compositions exercise access-pattern reduction/scalar scaling, free-axis permutation, partition/free transpose and explicit K accumulation. Optional --primitive-policy legalize corrects narrowly verified instruction/opcode namespaces and known HBM scalar-result staging; it never supplies an algorithm or benchmark-specific code. Original generated source and transformed source are both recorded. Original grading/references/shapes/tolerances/hazards remain unchanged.
