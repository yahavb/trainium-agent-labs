# Project 3 — Fused token scoring on Trainium2

**Compute a selected token's log-probability and the distribution's entropy
together, sharing the work over a large vocabulary.**

> **STATUS: RUNNABLE AND BENCHMARKED on Trainium2, seat-256.**
>
> **20.7% lower median device latency** on 512 positions × 32768 vocabulary
> entries versus the fastest measured separate NKI implementation; **2.14x
> faster** than the measured AWS cross-entropy-plus-entropy composition.
> All 111 hardware correctness comparisons pass with the original accuracy gate.
> The 20% target was met on workload B and missed on A; all results are reported.

## The problem and the contribution

A language model emits a score, called a logit, for every possible next token.
Scoring and reinforcement-learning workflows can need both the probability of
a chosen token and a measure of uncertainty, called entropy. Computing those
separately repeats vocabulary reads, normalization and exponentiation.

This project supplies a standalone **forward NKI kernel**, a stable FP32
reference, two complete separate-operation baselines, numerical validation and
reproducible Trainium2 measurements. The optimization keeps bounded vocabulary
tiles on chip and reuses exponentials and normalization for both outputs. It
writes two vectors instead of a vocabulary-sized probability intermediate.
An on-chip gather selects one score per row without building a full-vocabulary
selection mask; our own separate baseline receives the same improvement.
Scalar Engine produces exponentials and their sum together while the original
logits remain available for the gather and the shifted entropy calculation.

Fusion already exists in [Hugging Face TRL's GPU scoring kernel](https://github.com/huggingface/trl/blob/v1.14.0/trl/kernels/logprob_entropy.py).
AWS also provides [NKI cross entropy](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/loss/cross_entropy.py),
which computes loss with streaming log-sum-exp. Our contribution is the NKI
joint scoring implementation, its stable entropy extension and an auditable
Trainium comparison. See [prior work](NOTICE.md).

## Measured impact

Real Trainium2 execution, BF16 inputs and FP32 outputs, on October 10, 2026.
Each cell summarizes 300 measured calls across three rounds. The comparison is
the **fastest correct tuned separate baseline**, which was `separate_score` for
all three workloads. Lower latency is better.

| Workload | Positions × vocabulary | Before p50 (ms) | Fused p50 (ms) | Fused p95 (ms) | Speedup | Latency reduction |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| A: small | 128 × 8192 | 0.073176 | 0.065931 | 0.066444 | 1.11x | **9.9%** |
| B: larger | 512 × 32768 | 0.406309 | 0.322317 | 0.323087 | 1.26x | **20.7%** |
| C: partial tiles | 129 × 8193 | 0.077624 | 0.067073 | 0.067860 | 1.16x | **13.6%** |

On B, the improvement saves **0.084 ms per scoring call** against the strongest
baseline. The AWS-library composition takes 0.688173 ms for both outputs versus
0.322317 ms for fusion, a **53.2% reduction**, but that slower baseline is not
used for the headline gain. Device medians remain consistent across rounds:
0.322226–0.322400 ms for fusion and 0.406286–0.406357 ms for the best separate path.

Host execution timing, with device inputs and outputs already resident:

| Workload | Separate host p50 (ms) | Fused host p50 (ms) | Reduction |
| --- | ---: | ---: | ---: |
| A | 0.124943 | 0.117902 | 5.6% |
| B | 0.457228 | 0.372783 | 18.5% |
| C | 0.129239 | 0.128378 | 0.7% |

The host measurements include costs beyond the device execution interval. On C,
the host benefit is much smaller than the device benefit. These are scoring
primitive results; a full model or training-step speedup remains unmeasured.

**Accuracy:** 37 cases × 3 implementations = **111 passing hardware comparisons**,
plus 10 invalid-input rejection checks. Worst fused absolute error is
**0.0000458 for log-probability** and **0.0000429 for entropy**. No tolerance was
relaxed. The selected benchmark configurations also pass the uniform, peaked,
wide-distribution and large-offset cases, not just the benchmark's random inputs.

**Goal check:** the proposal targeted at least 20% lower median device latency on
both A and B. It succeeded on B and missed A. C also improves, with a 13.6% gain.
The broader two-workload target is therefore **partially met**.

The optional 512 × 151936 vocabulary experiment did not complete and supplies
no reported correctness or performance result. The submission's measured scope
is A, B and C above.

Evidence: [summary and per-round statistics](results/summary.json),
[individual timing samples](results/raw_samples.json),
[physical-core trace events](results/traces/),
[tuning measurements](results/tuning.json),
[hardware validation](results/validation-tuned.json), and
[environment and source hashes](results/environment.json).
`python audit_results.py` verifies the reported statistics, trace grouping,
fastest-baseline selection and unchanged source files without requiring Neuron.

## What is measured

All implementations return **both** FP32 outputs from identical BF16 logits and
int32 indices, with the same core allocation. The separate NKI baseline performs
two independent normalization passes inside one compiled dispatch; it avoids a
full probability tensor and uses the same streaming arithmetic as the fused
kernel. A second baseline uses the installed AWS cross-entropy kernel in FP32,
negates its loss and computes entropy separately, also inside one dispatch.
The fastest correct measured separate baseline is the comparison for each shape.

Device timing uses `nrtpy` execution trace events. LNC=2 creates two physical-core
intervals per execution; the reported duration spans the earliest start to the
latest completion for that `exec_id`, using synchronized trace timestamps to
account for the cores' clock offsets. Compilation, input generation, transfers
and host validation are excluded. Host execution timing is reported separately
with inputs and outputs already resident on the device; it is not a full
application latency measurement.

Before final measurement, every implementation gets the same tile search on
a separate deterministic input seed: four configurations for each of three
implementations on three workloads, with all 36 candidates passing correctness.
The final run rotates implementation order
across three rounds, using ten warmups and one hundred measured calls per round.
It retains individual samples, core trace events, percentiles, variation,
environment details and code hashes. The predefined goal is at least **20% lower
median device latency on workloads A and B**; this is our project target.

## Interface and numerical behavior

```python
from runtime import score_tokens
logprobs, entropy = score_tokens(logits, selected_indices)
```

- `logits`: contiguous NumPy BF16 `[T,V]`, finite values with `abs(x) <= 10000`.
- `selected_indices`: contiguous NumPy int32 `[T]`, indices from zero through `V-1`.
- Outputs: two FP32 arrays `[T]`, using natural logarithms, temperature one.
- `T >= 1`; `1 <= V <= 2**24`. The vocabulary bound keeps FP32 local index arithmetic
  exact in this SDK. Partial row and vocabulary tiles are supported.
- Fixed accuracy gate: every element satisfies `abs(error) <= 1e-3 + 1e-4 * abs(reference)`.
  The reference uses the same quantized BF16 input, promoted to FP32.

For a row, subtract its maximum `m`, compute `s = sum(exp(x-m))` and
`u = sum(exp(x-m)*(x-m))`, then return
`logprob = x[selected]-m-log(s)` and `entropy = log(s)-u/s`.
When a later tile raises the running maximum, both sums are rescaled; the
weighted sum also receives the required shift correction. Only valid slices
are loaded, so padded infinities cannot contaminate the entropy.

Uniform scores, vocabulary size one, random scores, peaked distributions,
positive/negative common offsets, first/middle/last indices, increasing tile
maxima and ragged shapes exercise the hardware correctness suite. Host checks
reject invalid layout, dtype, shape, values and indices before device execution.

## Run it on the workshop pod

This implementation targets the installed workshop NKI 0.6 snapshot on Trainium2.
`nki`, `nkilib`, `nrtpy`, NumPy and `ml_dtypes` are already available there.
The standalone compiler adapter uses snapshot-specific APIs; inspect the recorded
versions before porting it to another SDK.

From a terminal with the workshop Kubernetes credentials:

```bash
kubectl exec -it seat-256 -- bash
cd /workspace/projects/03-fused-token-scoring
export NEURON_PLATFORM_TARGET_OVERRIDE=trn2
export NEURON_LOGICAL_NC_CONFIG=2
export NEURON_RT_VISIBLE_CORES=0,1
python runtime.py
python validate.py
python benchmark.py --tune
python validate.py --tuning results/tuning.json --output results/validation-tuned.json
python benchmark.py
python audit_results.py
```

For this development checkout, replace the `cd` line with
`cd /workspace/fused-token-scoring-shubham/projects/03-fused-token-scoring`.

The entire sequence can survive a dropped terminal connection:

```bash
nohup bash run.sh > run.log 2>&1 < /dev/null &
tail -f run.log
# Completion status: cat results/run.exit (0 means success).
```

Use your assigned pod and available cores. An idle-looking core can still be
reserved by another runtime process. Coordinate the model server with your team
before pausing it; preserve its settings and restore it after benchmarking.
This implementation's development runs use the isolated directory
`/workspace/fused-token-scoring-shubham/projects/03-fused-token-scoring`, preserving
the active checkout and other project files in `/workspace`.

CPU simulation is useful before hardware access, but supplies no performance
result:

```bash
python validate.py --simulate --quick --output results/simulator-quick.json
```

The submitted evidence can also be checked locally without Trainium or NumPy,
from the repository root:

```bash
python3 projects/03-fused-token-scoring/audit_results.py \
  --results projects/03-fused-token-scoring/results
```

For repeated device calls, construct `runtime.LoadedKernel` once and reuse its
device inputs/outputs. The convenience `score_tokens` call includes validation,
compilation/loading and transfers, and is deliberately excluded from kernel
timing. Compiler artifacts are cached under ignored `build/` directories.

## Files and implementation steps

| File | Purpose |
| --- | --- |
| [IMPLEMENTATION.md](IMPLEMENTATION.md) | Ordered checkpoints and frozen acceptance criteria |
| [kernel.py](kernel.py) | Fused kernel and independent-pass NKI baseline |
| [baseline.py](baseline.py) | Complete baseline using installed AWS cross entropy |
| [reference.py](reference.py) | BF16 input contract, stable FP32 reference and accuracy gate |
| [runtime.py](runtime.py) | Standalone compile/load, device I/O and trace timing |
| [validate.py](validate.py) | Hardware correctness and host rejection checks |
| [benchmark.py](benchmark.py) | Equal-budget tuning, repeated measurement and saved evidence |
| [run.sh](run.sh) | Sequential pipeline that records a final exit status |
| [audit_results.py](audit_results.py) | Offline check of hashes, raw timings, trace grouping and reported gains |
| `results/` | Actual validation, tuning, samples, traces and environment metadata |

## Scope of the result

This is a forward scoring primitive for materialized logits. It does not include
the language-model head, gradients, optimizer steps, framework/autograd
integration, multi-device execution or a full GRPO training run. Kernel speedup
can reduce the cost of this operation; the application benefit depends on how
much of its total time it spends scoring. Any full training-speed claim needs a
separate integration benchmark. Memory traffic reduction is an algorithmic
property here, not a measured device memory-usage result.
