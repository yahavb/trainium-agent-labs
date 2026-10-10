# Synthetic NKI implementation

Development only: `/tmp/trainium-kernel-dev`, branch `feat/nki-diagnostic-selection`. Original checkout, checker, reference kernels, tolerances, reward weights, model server and SDK remain protected. The system changes inference-time prompting and memory; it does not fine-tune Qwen or train model weights.

## Architecture

`shape_repair.py` reuses conservative AST inspection, checker-case argument binding and installed SDK constraints. `--feedback-policy targeted` adds one precise explanation and permits dependent lines to change together. `--feedback-policy legacy` is the default. Original code and checker error remain in the prompt. If targeted feedback is selected, its evidence replaces additional generic grounded cards rather than appending several unrelated cards. Existing `standard`, `grounded` and `shape-aware` repair flags remain supported.

`synthetic_nki/generate.py` constructs 12 independent tiny tasks; `mutations.py` introduces one error per task; `verify.py` checks syntax, imports/decorator, actual CPU execution, untouched inputs, independent NumPy numerics and simulator hardware warnings. `retrieval.py` indexes only compatible, verified, structurally unique training records and inserts up to two short before/after statements. It never inserts a complete kernel. `--example-policy synthetic` enables this; `off` is the default.

`repair_history.py` stores six signatures/source hashes and a separate best verified candidate. Optional `--adaptive-repair` escalates repeated corrections to dependent dataflow changes, preserves the best verified source when a new selection regresses, and caps added history to available context. A changed first failure is explicitly a hypothesis, not proof that its hidden constraint is repaired.

`run_controlled.py --synthetic-pilot smoke|ablation|iteration` uses process/endpoint guards, sequential configurations, exclusive directories, private grading and frozen Python/dataset snapshots with SHA-256 manifests. Generation still uses exactly four calls per round with samples=4. No retrieval/controller model calls are added. `analyze_experiments.py` reads every attempt and reports per-shape verification, repair transitions, actual endpoint usage and wall/checker/available simulation timing.

## Corpus and verification

12 AST-distinct clean kernels, 12 independently observed repair pairs, 11 train and 1 held-out record. The entire elementwise arithmetic family is held out, and held-out files are never indexed. Families cover bounded DMA, scalar/binary arithmetic, sum/max reductions, on-chip regions, small matrix products with an offset, disjoint K accumulation, partition copies and column output coverage. They are not official benchmark entry points or reference solutions. Matrix-product tasks include a separate offset and tiny constrained layouts; retrieval exposes only the changed invariant.

Each clean and restored task passed three seed-varied float32 input cases against independent NumPy operations at rtol=atol=2e-5. All mutations failed with the intended observed category. Current corpus: 1 DMA mismatch, 5 tensor-dimension failures, 2 invalid API functions, 1 invalid API argument, 2 numerical failures and 1 buffer-placement failure. No mutations were rejected in the initial build. Unit tests also inject a non-failing mutation and confirm rejection. The 1/1 held-out restored-kernel verification is a data-quality check, not held-out LLM accuracy. Device-verified examples: zero.

Limitations: only float32 so far; tiny fixed layouts; one held-out family/example; finite input tests are not formal proof. Uninitialized missing-output behavior is caught by numerical comparison, not attributed to a new official checker category. Static checks are deliberately limited and simulator execution supplies the hardware-layout gate. Generated-code licensing is unspecified by this repository; no external implementation was copied. Source, broken result, actual error, repaired result, seed/SDK/target and provenance are retained in JSONL.

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
python -B run_controlled.py --run --synthetic-pilot smoke --rounds 2 --samples 4 --repeat 1 --levels 3
python -B run_controlled.py --run --synthetic-pilot ablation --rounds 4 --samples 4 --repeat 1 --levels 3
```

All original default-behavior and equal-reward selection regressions remain in the test suite. Live results and final validation count are recorded separately in `SYNTHETIC_NKI_EXPERIMENTS.md`.
