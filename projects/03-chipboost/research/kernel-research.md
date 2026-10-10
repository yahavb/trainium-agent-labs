# Research-to-experiment notes: NKI kernel agent

Researched 2026-10-10. These are proposed experiments, not measured improvements in our system. CUDA/Triton results do not establish NKI capability. Preserve all original comparison results and distinguish feedback-only changes from stronger-model, prompt, and template changes.

## 1. Diagnose at the failing stage, with concrete evidence

KernelBench evaluates iterative refinement using previous code, execution/compiler errors, and profiler information; its experiments find feedback useful, but improvements depend on model capability. More sampling alone fails on some difficult task families. [KernelBench, sections 5.1–5.2](https://arxiv.org/html/2502.10517v1)

Implementation: return a structured record with stage, named error, candidate line, observed source/destination shapes, and one valid edit. Include the actual compiler diagnostic and candidate line for unsupported expressions. Keep simulator success distinct from device compilation success. For our DMA error, explain that `[128,512]` and `[128,128]` differ in element count, then identify whether the bad operand is stationary or moving before proposing the replacement. Do not infer operand identity solely from the number 65536.

A recent compiler-grounded study on Ascend advocates escalating from profiling symptoms to compiler/IR evidence before rewriting; this supports inspecting the exact compiler failure rather than guessing from a truncated message. Its performance results are specific to converted Ascend workloads. [Compiler-Grounded Hierarchical Diagnosis](https://arxiv.org/abs/2607.23089)

## 2. Escape repeated failures, preserve the best valid candidate

GEAK explicitly describes a debugging trap, caps repair attempts per snippet, then starts a fresh strategy. Its optimizer receives historical code/performance; independent diverse trajectories complement serial repair. The paper also reports that examples can improve correctness without always improving speed. [GEAK, section 4 and section 5.1](https://arxiv.org/html/2507.23194v1)

Implementation: cap the same normalized error at two repairs; after that restart from the best correct candidate with a different optimization hypothesis. Track normalized AST hashes to detect unchanged code, but still count all model calls and duplicate proposals in the experimental budget. Skip redundant device execution only with an explicit duplicate/cached-result record. Keep four short history fields: attempted change, validity, latency, failure signature. Split an eight-generation pilot into two independent four-turn trajectories rather than eight retries of one broken kernel. This allocation is our hypothesis, not a paper-prescribed optimum.

## 3. Ground every turn in the installed NKI contract

The AWS tutorial teaches matmul through tiling and blocking, with transposed LHS `[K,M]`, RHS `[K,N]`, and Tensor Engine contraction along the partition dimension. The ISA reference defines stationary-transpose times moving, with SBUF inputs and PSUM output. [AWS matmul tutorial](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.28.0/nki/guides/tutorials/matrix_multiplication.html), [nc_matmul reference](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.26.0/general/nki/api/generated/nki.isa.nc_matmul.html)

Implementation: use an API card extracted from our installed, tested examples on every generation and repair turn: stationary `[Ktile,Mtile]`, moving `[Ktile,Ntile]`, accumulator `[Mtile,Ntile]`; each DMA source and destination must match; K slices must be preserved when staging reusable tiles. Include one minimal valid copy/matmul snippet in the exact installed API syntax. The linked versions differ; do not copy a signature from an arbitrary documentation version into our environment. Ask for one local transformation per attempt: loop reorder, RHS reuse across M, LHS reuse across N, or bounded blocking. This reduces the space of unsupported constructs.

## 4. Audit decoding and truncation before paying for another batch

The official Qwen3-8B model card distinguishes thinking and nonthinking modes. It advises against greedy decoding in thinking mode because of repetitions and degradation. Suggested settings are temperature 0.6/top-p 0.95/top-k 20 for thinking and 0.7/0.8/20 for nonthinking. [Qwen3-8B model card](https://huggingface.co/Qwen/Qwen3-8B)

Implementation: log the actual request parameters, enabled chat-template mode, completion token count and finish reason. Test extraction against both thinking and nonthinking responses; reject truncated code as a generation failure with a targeted retry. Compare a modest thinking pilot to the existing configuration using the same candidate budget; report latency/token cost separately. Do not assume temperature or thinking mode alone solves NKI knowledge gaps.

## 5. Screen an alternate coding model, keeping the model ablation clean

Qwen offers Qwen3-Coder-30B-A3B-Instruct, a coding-specialized model with 30.5B total / 3.3B activated parameters and nonthinking output. Active parameter count does not mean 3B storage requirements. Official documentation does not demonstrate NKI competence or guarantee our Neuron serving stack supports it. [Official model card](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct), [Qwen3-Coder repository](https://github.com/QwenLM/Qwen3-Coder)

Implementation: first enumerate already available authorized endpoints and available server memory/model support. Screen any candidate on four identical small tasks: fix the observed DMA mismatch, preserve the K reduction, remove the observed unsupported construct, and propose one reuse optimization. Run all outputs through the unchanged evaluator. For candidates passing the screen, compare current Qwen3-8B and alternate model with the same prompts, baseline, budget and feedback. Keep this separate from DMA-only v2. A coding model is a plausible candidate, not a promised upgrade.

Kevin demonstrates that multi-turn CUDA-specific training can improve a QwQ-32B base and that serial refinement benefits its trained model. That is evidence for evaluating repair capability, not evidence that a CUDA-trained checkpoint transfers to NKI. Training our own model is a later project, not the immediate remedy. [Kevin](https://arxiv.org/abs/2507.11948)

## 6. Separate a practical improvement arm from the feedback thesis

For the fastest route to useful kernels, introduce a clearly labeled template-assisted arm: start from a valid expert template and let the model propose constrained reuse/blocking changes. Compare it against template random search from exactly the same initial templates and parameter space. The original model-only arm remains the clean baseline. This is our experimental design, motivated by our observed template advantage; it is not evidence that model feedback already wins.

Keep hidden correctness cases, input isolation, unchanged timing, and actual chip execution. Report compile rate, correctness rate, duplicate rate, repeated-error rate, best validated speedup per independent run, and wall-clock/token cost. Do not select only successful trajectories. Sakana's own post-mortem describes inflated results caused by benchmark-memory access and inadequate checks, underscoring why evaluation integrity must remain fixed while optimizing the agent. [Sakana first-party post-mortem](https://sakana.ai/ai-cuda-engineer-post-mortem/)

## Proposed order

1. Inspect the current pilot's full unsupported-expression diagnostic and ensure decoding is not truncated.
2. Run one repaired candidate through simulator, device compiler, correctness and timing.
3. Run a small controller/NKI-card pilot; stop repeated-error loops early but log every generation.
4. Screen an available alternate coding model without interrupting pinned original experiments.
5. Compare model and controller effects separately; expand only configurations that produce valid novel kernels.
6. Run template-assisted search as a separately labeled practical arm, not a retrofit of the original thesis.
