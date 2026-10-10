# Fused Token Scoring on Trainium2

**Joint selected-token log-probability and entropy using the AWS Neuron Kernel Interface (NKI).**

> **Status: implemented, validated and benchmarked on Trainium2, seat-256.**
> On the larger workload, fusion reduces median device latency by **20.7%**
> (**1.26× speedup**) against the fastest measured separate baseline.
> All **111 hardware correctness comparisons** pass the original accuracy gate.

## Abstract

Language-model scoring and reinforcement-learning workflows can require both a
selected token's log-probability and the entropy of its probability distribution.
Separate calculations repeat vocabulary reads, normalization and exponentiation.
We implement a tiled NKI kernel that shares these calculations while maintaining
stable FP32 state over BF16 inputs. We compare it with two complete, independently
tuned separate-operation baselines on real Trainium2 hardware. Median device
latency decreases by **9.9%, 20.7% and 13.6%** on three predefined workloads,
with all numerical comparisons passing. The proposed 20% reduction on both
primary workloads is **partially achieved**: the larger workload passes and the
smaller one misses. The result is a reproducible forward scoring primitive;
full-model and training-step acceleration remain to be measured after integration.

## 1. Problem statement

### 1.1 Application need

Consider training a customer-support language model to generate useful answers.
A PPO/GRPO-style workflow can score the tokens in generated responses to determine
how likely those tokens are under the model. It may also compute entropy to
monitor uncertainty or encourage exploration through its training objective.

The model supplies a matrix of raw scores, called **logits**, with shape `[T,V]`:
`T` is the number of token positions being scored and `V` is the vocabulary size.
The operation returns two FP32 vectors of length `T`:

- **Selected-token log-probability:** the logarithm of the probability assigned
  to the chosen token at each position.
- **Entropy:** how spread out the probability distribution is at each position.
  It measures distribution uncertainty, not answer correctness.

This application motivates the optimization. Our experiments use deterministic
synthetic logits and measure the scoring primitive on Trainium2.

### 1.2 Optimization opportunity and contribution

Both outputs need the same vocabulary normalization and exponentials. Calculating
them through independent passes repeats work over a potentially large input.
The project evaluates whether sharing that work reduces latency while preserving
numerical correctness and support for uneven input shapes.

The contribution consists of a **joint NKI forward kernel**, a stable streaming
entropy calculation, an on-chip selected-score gather, complete baselines,
hardware validation and auditable measurements. Temporary vocabulary tiles remain
on chip, and the outputs are two vectors rather than a full probability matrix.
Our own separate baseline also avoids a full probability intermediate; the
comparison therefore measures the benefit of sharing work between the outputs.

Fusion is an established technique. [Hugging Face TRL provides GPU token scoring](https://github.com/huggingface/trl/blob/v1.14.0/trl/kernels/logprob_entropy.py),
and [AWS NKI Library provides streaming cross entropy](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/loss/cross_entropy.py).
This project contributes an NKI joint implementation and a measured Trainium
comparison; it makes no claim to invent fusion. See [NOTICE.md](NOTICE.md).

## 2. Objectives

| Objective | Acceptance criterion | Outcome |
| --- | --- | --- |
| Preserve correctness | Every output element passes the fixed FP32-reference accuracy gate | **111/111 hardware comparisons pass** |
| Support realistic boundaries | Handle singleton vocabularies, extreme offsets, boundary indices and partial tiles | All 37 numerical cases pass across three implementations |
| Reduce device latency | At least 20% lower median latency on both A and B versus the fastest correct separate baseline | **Partially met:** B achieves 20.7%; A achieves 9.9% |
| Make evaluation reproducible | Retain tuning, raw samples, physical-core traces, versions and source hashes | Evidence committed; offline audit passes |

The 20% threshold is our predefined project objective, not an organizer scoring
rule. Workload C adds boundary coverage beyond the two primary workloads.

## 3. Methodology

### 3.1 Input contract and stable computation

Inputs are contiguous NumPy BF16 logits `[T,V]` and contiguous int32 selected
indices `[T]`. Outputs are FP32 log-probability and entropy vectors `[T]`, using
natural logarithms and temperature one. Inputs must be finite with
`abs(logit) <= 10000`, `T >= 1`, valid indices, and `1 <= V <= 2**24`.
The vocabulary bound keeps local FP32 index arithmetic exact in this SDK.

For each row, the kernel maintains a running maximum and two sums:

```text
m = maximum logit
s = sum(exp(x - m))
u = sum(exp(x - m) * (x - m))

logprob = x[selected] - m - log(s)
entropy = log(s) - u / s
```

Rows and vocabulary entries are processed in tiles using fast on-chip memory.
When a later tile increases the maximum, the previous sums must be corrected
before adding that tile's contribution:

```text
delta = old_m - new_m
scale = exp(delta)
u = scale * (old_u + delta * old_s)
s = scale * old_s
```

The entropy correction uses the **old** normalization sum. An on-chip gather
selects one score per row without a vocabulary-sized selection mask. Scalar
Engine produces exponentials and their sum together; valid slices handle partial
tiles. Token positions are divided across two physical cores. The installed
Neuron compiler builds the kernel, and `nrtpy` loads and executes it.

### 3.2 Complete and comparable baselines

| Implementation | Computation | Role |
| --- | --- | --- |
| `fused_score` | Shared vocabulary processing for both outputs | Proposed implementation |
| `separate_score` | Independent log-probability and entropy passes with the same streaming arithmetic and gather optimization | Strong separate NKI baseline |
| `aws_separate_score` | Installed AWS cross entropy in FP32, negated to obtain log-probability, plus independently computed entropy | AWS-library composition baseline |

All three return **both required outputs**, use identical inputs and the same
core allocation, and execute in **one compiled dispatch**. The primary comparison
uses the fastest correct tuned separate implementation for each workload.

### 3.3 Workloads and environment

| Workload | Positions `T` | Vocabulary `V` | Input scores | Purpose |
| --- | ---: | ---: | ---: | --- |
| A: small | 128 | 8192 | 1,048,576 | Compact scoring batch; primary workload |
| B: larger | 512 | 32768 | 16,777,216 | 16× the input scores of A; primary workload |
| C: partial / uneven | 129 | 8193 | 1,056,897 | One additional row and vocabulary entry relative to A; exercises uneven boundaries |

“Small” and “larger” describe **input matrix sizes**, not language-model sizes.
“Partial” refers to valid slices when shards or tiles contain fewer entries than
a full block. Diagonal or square matrices are not required.

Measurements were collected on **October 10, 2026**, in workshop pod `seat-256`
on Trainium2, with logical NeuronCore configuration **LNC=2**, two physical-core
execution intervals per scoring call, and `NEURON_RT_VISIBLE_CORES=0,1`.
The image uses Python 3.13.7, NKI `0.6.0+31049202112.g85070674` and
Neuron compiler `2.27.5334.0+f702b353`. Exact versions and source identifiers
are recorded in [environment.json](results/environment.json).

### 3.4 Correctness and benchmark protocol

The reference promotes the **same quantized BF16 inputs** to FP32. Every element
must satisfy the unchanged accuracy gate:

```text
abs(actual - reference) <= 1e-3 + 1e-4 * abs(reference)
```

The 37-case suite covers random and wide distributions, uniform scores, peaked
distributions, vocabulary size one, positive and negative common offsets,
first/middle/last indices, increasing tile maxima and ragged shapes. It also
checks numeric invariants and ten invalid host-input rejections.

Performance evaluation follows this sequence:

1. **Tune equally.** Each implementation receives four configurations:
   `(row_tile, vocab_tile)` = `(64,4096)`, `(128,4096)`, `(128,8192)` and
   `(128,16384)`. Tuning uses seed 2027, five warmups and 30 device samples
   per candidate. All **36 candidates** pass correctness.
2. **Validate selected configurations.** The final tile choices undergo the
   full hardware correctness suite, including distributions outside the
   tuning input.
3. **Measure independently.** Final inputs use seed 2026. Implementation order
   rotates over three rounds, with ten warmups and 100 measured calls per
   round, implementation, workload and timing mode.
4. **Retain and audit.** The run saves **5,400 timing samples**, **27 trace
   files**, percentiles, variation, per-round statistics and compiled-program
   hashes. The offline audit reconstructs timings and checks source hashes.

**Device latency** spans the earliest physical-core start to the latest completion
for one `exec_id`, using synchronized trace timestamps. Two physical-core
intervals are grouped as one execution; raw clocks from different cores are not
mixed. Compilation, input generation, transfers and host validation are excluded.
**Resident host latency** is measured separately with device inputs and outputs
already allocated. It excludes transfers and full application integration.

### 3.5 Reproduction and submission artifacts

The full project is contained in this folder. On the supplied Trainium2 image:

```bash
kubectl exec -it seat-256 -- bash
cd /workspace/projects/03-fused-token-scoring
export NEURON_PLATFORM_TARGET_OVERRIDE=trn2
export NEURON_LOGICAL_NC_CONFIG=2
export NEURON_RT_VISIBLE_CORES=0,1
bash run.sh
# Completion status: cat results/run.exit
```

The existing isolated development copy uses
`/workspace/fused-token-scoring-shubham/projects/03-fused-token-scoring`.
Use available cores and coordinate model-server reservations with the team.
The standalone compiler adapter targets the recorded NKI snapshot; another SDK
may need adapter changes. CPU simulation provides correctness checks only.

The committed evidence can be audited locally **without Trainium or NumPy**, from
the repository root:

```bash
python3 projects/03-fused-token-scoring/audit_results.py \
  --results projects/03-fused-token-scoring/results
```

| Artifact | Purpose |
| --- | --- |
| [kernel.py](kernel.py), [baseline.py](baseline.py) | Proposed kernel and both complete baselines |
| [reference.py](reference.py), [validate.py](validate.py) | Input contract, FP32 oracle and hardware correctness suite |
| [runtime.py](runtime.py) | Compile/load, device I/O and execution trace timing |
| [benchmark.py](benchmark.py), [run.sh](run.sh) | Equal-budget tuning and repeated benchmark pipeline |
| [audit_results.py](audit_results.py) | Offline checks of source hashes, samples, traces and reported gains |
| [summary.json](results/summary.json), [RESULTS.md](results/RESULTS.md) | Aggregated measurements and per-round statistics |
| [raw_samples.json](results/raw_samples.json), [traces/](results/traces/) | Individual timings and physical-core event evidence |
| [tuning.json](results/tuning.json), [validation-tuned.json](results/validation-tuned.json) | Candidate measurements and selected-configuration correctness |
| [environment.json](results/environment.json) | SDK versions, baseline/source hashes and hardware metadata |
| [IMPLEMENTATION.md](IMPLEMENTATION.md) | Implementation checkpoints and frozen acceptance criteria |

The convenience interface is
`runtime.score_tokens(logits, selected_indices)`. Repeated application calls
should reuse `runtime.LoadedKernel`; the convenience call includes validation,
compilation/loading and transfers and is excluded from the kernel timing.

## 4. Results and metrics

### 4.1 Device latency: comparison with the strongest baseline

Each statistic summarizes **300 calls across three rounds**. `p50` is the median;
`p95` is the 95th percentile. The fastest correct separate baseline was
`separate_score` for every measured workload. Lower latency is better.

| Workload | Separate p50 (ms) | Fused p50 (ms) | Fused p95 (ms) | Speedup | Latency reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| A: small | 0.073176 | 0.065931 | 0.066444 | 1.11× | **9.9%** |
| B: larger | 0.406309 | 0.322317 | 0.323087 | 1.26× | **20.7%** |
| C: partial / uneven | 0.077624 | 0.067073 | 0.067860 | 1.16× | **13.6%** |

Speedup is `separate_p50 / fused_p50`; latency reduction is
`100 * (1 - fused_p50 / separate_p50)`. On B, fusion saves approximately
**0.084 ms (84 microseconds) per scoring call**. Its three round medians range
from 0.322226 to 0.322400 ms, compared with 0.406286 to 0.406357 ms for the
strongest separate path. Raw samples and complete statistics are retained.

The AWS-library composition produces both outputs in 0.130962, 0.688173 and
0.126745 ms on A, B and C, respectively. On B, fusion is **2.14× faster** than
that composition, with **53.2% lower latency**. The primary 20.7% gain uses the
stronger separate NKI baseline.

### 4.2 Resident host execution latency

| Workload | Separate host p50 (ms) | Fused host p50 (ms) | Reduction |
| --- | ---: | ---: | ---: |
| A | 0.124943 | 0.117902 | 5.6% |
| B | 0.457228 | 0.372783 | **18.5%** |
| C | 0.129239 | 0.128378 | 0.7% |

Host measurements include costs beyond the device execution interval. The
smaller benefit on C illustrates how those costs can dilute a device improvement.
These values describe resident execution, not full application latency.

### 4.3 Numerical correctness

All **37 cases × 3 implementations = 111 hardware comparisons** pass, together
with **10 invalid-input rejection checks**. No tolerance was relaxed.

| Implementation | Worst log-probability absolute error | Worst entropy absolute error | Hardware cases passed |
| --- | ---: | ---: | ---: |
| Fused NKI | 0.0000458 | 0.0000429 | 37/37 |
| Separate NKI | 0.0000458 | 0.0000429 | 37/37 |
| AWS-library composition | 0.0004826 | 0.0000429 | 37/37 |

Errors are measured against the FP32 reference on the same BF16 inputs. Passing
this numerical gate establishes correctness for the tested operation and cases;
it is not a model-quality evaluation.

### 4.4 Interpretation and limitations

The observations support sharing vocabulary processing as a useful optimization
for this scoring primitive. The predefined **20% target is partially met**:
B passes, A misses, and C provides an additional 13.6% improvement.

The measured scope is one workshop pod, the recorded SDK, three input shapes
and deterministic synthetic distributions. Results should be remeasured for a
new device, SDK, shape or integration. The optional `512 × 151936` experiment
did not complete; no correctness or performance result is claimed for it.

This implementation computes forward scores from materialized logits.
Language-model-head fusion, gradients, optimizer steps, framework/autograd
integration, multi-device execution and a complete GRPO training run remain
outside the measured contribution. Reduced vocabulary processing follows from
the algorithm; device memory usage and energy savings were not measured.

Application gains depend on the fraction of total time spent scoring. For
example, **hypothetically**, if scoring takes one second of a ten-second training
step, a 20.7% scoring reduction would save about 0.207 seconds, or 2.1% of total
step time, assuming other work is unchanged. Full-step gains require an actual
integration benchmark.

## 5. Conclusion

We deliver a runnable NKI implementation that computes selected-token
log-probability and entropy jointly on Trainium2. Against an optimized separate
baseline with comparable precision, tiling and core allocation, it achieves
**20.7% lower median device latency on the larger workload**, with additional
**9.9% and 13.6% reductions** on the small and uneven workloads. All **111
hardware correctness comparisons** pass, and the saved evidence is independently
auditable without accelerator access.

The practical contribution is a faster, validated building block for Trainium
workflows that repeatedly need both token likelihood and entropy. The next step
is integration into a real scoring or training pipeline, gradients where required,
and measurement of complete application throughput. The benchmark supports the
kernel improvement; the broader two-workload target and full training impact
remain explicit limits of the result.
