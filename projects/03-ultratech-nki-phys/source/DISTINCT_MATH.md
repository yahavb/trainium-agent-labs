# Different Math, Not Just Different Parameters

The objective is an evidence-guided optimization agent, not model weight training
or a task-name-to-template lookup. The evaluation harness is necessary infrastructure,
but is not itself proof that the agent can reason about unfamiliar kernels.

## Three Simple Workloads

| Task | Mathematics | Computational structure |
| --- | --- | --- |
| Existing frictionless contacts | x_next = max(0, x - alpha * (A*x + b)) | Iterative matrix-vector solver |
| Spring-damper force | F = -(k*x + c*v) | Elementwise expression, no solver iterations |
| Net force on a body | F_net = sum(F_i) | Reduction across contributing forces |

The new tasks are intentionally small physics computations, not full simulators.
They have different equations and computational structures. Varying seeds within
one task changes values only, and must not be described as a new algorithm.

## Evaluation Foundation Implemented

`math_tasks.py` defines two new equations, 16 deterministic input cases each,
FP64 references from frozen FP32 inputs, and equation checkers. Cases include
zero displacement, zero velocity, zero damping, cancellation, and mixed magnitudes.
Shape, finite FP32 output and input preservation are also required. Error limits
are fixed before optimization: 1e-6 + 2e-6 * sum of absolute contributing terms.
This accounts for FP32 cancellation without using a near-zero result as a scale.

Spring candidates compare separate arithmetic with sign fusion. Reduction
candidates compare serial additions with a native reduction. All are human-supplied
templates; installed-SDK compatibility and hardware benefits remain unmeasured.

`math_harness.py` evaluates both forms, retains inputs/outputs/sources/hashes,
checks all 16 cases, and saves attempt logs plus next-feedback text. Device mode
uses existing SpikeModel timing support, five repeats, paired baseline timings,
and correctness before/after timing. Only seed 3 is timed; all seeds are correctness
tests. It reports speedups per task rather than pooling unrelated rates.
CPU mode does not execute NKI or establish a speedup. The runner does not itself
call Qwen; the new math_agent_loop.py controller supplies model-generated proposals.

## Model-Driven Loop Implemented

Run locally from the repository:

```sh
.venv-physics-current/bin/python projects/03-contact-physics/math_agent_loop.py \
  --cores 0,1 --minutes 30 --attempts 4
```

The controller uploads dependencies to seat-260, starts Qwen for generation,
alternates spring and net-force tasks, and feeds each task its own prior feedback.
Four attempts provide two proposals per task; this is a budget, not a guaranteed
number of successful benchmarks. A caller can select one task with --tasks spring.
Never run this beside another controller using the same seat.

Qwen receives equations, input shapes, baseline NKI and feedback. It supplies
a hypothesis and an operation graph, not an exact requested template. Trusted
math_program.py validates and lowers the graph to NKI. Up to 16 operations are
supported: multiply, add, subtract, negate, fused multiply/negate, copy, native
or serial reduction, with auto or Vector Engine placement. Shape/layout checks
reject invalid graphs before compilation. Generated Python is never executed.
This is a finite code-generation language, not arbitrary NKI optimization or
general algorithm discovery. It does not cover the contact solver yet.

Each novel graph gets all-case simulation checks, then device validation and
paired original-baseline timing. Qwen is stopped for timing. Lowered source
hashes, requests, replies, every failure, numerical deviations and raw timings
are retained. Correctness failures with valid outputs get a measured pass fraction;
generation/compilation failures remain unmeasured. Slower proposals do not replace
the incumbent. All unseen-task generalization and hardware benefits remain unproven.

## What Makes The Next Stage An Agent

Give Qwen the task equation, input shapes, baseline source, installed API constraints,
documentation and prior feedback. Ask it to identify the computational structure,
propose a technique with a falsifiable reason, and produce code. Do not prescribe
the target technique based on the task name. Independently review and execute
proposals; return exact correctness failures and measured throughput. Let Qwen
revise, change technique, or retain the input. Keep every rejected proposal.

Evaluation should compare model-selected proposals against the fixed-template
script under the same attempt budget. Report full-correctness rate, best valid
speedup per task, time/attempts to improvement, and regressions avoided. Freeze
unseen task variations before the final assessment and do not feed their failures
back into development. Public fixture feedback alone is not a generalization test.

The earlier four contact-kernel experiments remain separate: hoisted bias,
explicit Vector Engine placement, round-robin worlds, and contraction tiles of 32.
They are now source templates in the reviewed catalog, not measured improvements.
Profiling remains needed to establish stalls or actual engine overlap. No kernel
change or cross-task optimization benefit is claimed without device evidence.

References: [NKI ISA](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/_modules/nki/isa.html)
for scalar/vector operands, engine restrictions and native free-axis reduction;
[performance guide](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.26.0/general/nki/nki_perf_guide.html)
for fusion, layout and pipelining hypotheses. Installed SDK signatures take
precedence over documentation from other versions.
