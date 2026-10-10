# Research-driven follow-up, October 10, 2026

## Evidence and design

- [AccelOpt, MLSys 2026](https://arxiv.org/html/2511.15915v2) studies NKI optimization directly.
  It retains multiple correct candidates and explores distinct plans. Our small two-candidate,
  two-direction search tests this idea; it does not reproduce the paper's larger budget.
- [KernelBench](https://openreview.net/pdf?id=yeoN1iQT1x) motivates execution and profiling
  feedback, while showing that repeated sampling cannot reliably rescue every model/task pair.
- [NKI-Agent](https://arxiv.org/html/2607.04395v1) supports compiler-guided repair and explicit
  tensor-rank guidance. Its correctness benchmark does not establish speed gains here.
- [GEAK](https://arxiv.org/abs/2507.23194) motivates abandoning repeated failed repairs.
  Its Triton/GPU results are not directly transferable performance claims for Trainium.

## Preserved treatments

1. Original 72-evaluation comparison: unchanged P1/P2/P3 snapshots on core 3.
2. DMA-only v2: queued behind that comparison, unchanged model/settings, 24 evaluations.
3. Recovery v3: completed eight attempts, zero faster; preserve this negative result.
4. Astra coding-agent trial: separately generated candidates, with baseline/API guidance and
   parent-suggested optimization directions. This is assisted generation, not a matched Qwen
   API experiment. Public-referee records and repeat measurements are retained separately.
5. Research beam v4 reasoning screen: four evaluations, two distinct branches per round,
   two best AST-distinct correct parents, persistent NKI API and shape guidance, corrected
   compiler feedback, Qwen3-8B thinking enabled, 4096-token answer cap and 8192-token context.
   Duplicates still count. Multi-change exploratory treatment; not a causal ablation.

Kernel checks run sequentially on core 2 with the original reference baseline; model inference
shares the existing server. Do not compare generation wall-clock time as a controlled result.
No correctness threshold, hidden-test requirement, timing gate, or baseline is weakened.

The target is reproducible chip speedup for correct generated kernels, not a more favorable
label or a larger number of attempts. Record failed experiments alongside improvements.
