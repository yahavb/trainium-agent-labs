# Documentation-Backed NKI Feedback

Reviewed official documentation on 2026-10-10:

- API signatures and operand memory: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/_modules/nki/isa.html
- Trainium2 layout/engines: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/guides/architecture/trainium2_arch.html
- Performance methodology (historical guide): https://awsdocs-neuron.readthedocs-hosted.com/en/v2.26.0/general/nki/nki_perf_guide.html

The guide is historical; it is not used as authority for current import paths,
benchmarker interfaces or multicore restrictions. The installed seat SDK and
successful simulation/device execution remain authoritative for compatibility.

## Feedback Layers

1. API failure: identify the exact unsupported call or keyword; provide the
installed signature and a supported memory/layout pattern, not just "invalid".
2. Physics failure: report the actual failed gate, tolerance, case coverage,
worst errors and changes versus the previous candidate. API changes alone do
not establish convergence. Do not declare instability solely from slow progress.
3. Performance: only after all-case accuracy, report actual device/host timing
and profile evidence, then propose a targeted kernel change. Our small matvecs
and serial-world loop are reasons to investigate utilization, not measurements.

## Implementation

nki_guidance.py provides a compact prompt card explaining HBM staging, transpose
convention, legal arithmetic memory, 2D rate/state shapes and precision discipline.
It inspects installed nki.isa signatures for the five operations used by the
baseline without importing candidate code. Detailed signatures/version/provenance
are saved in sdk-context.json; selected signatures enter the prompt. Missing
SDK is explicitly recorded, never treated as compatibility proof. Inspection
of signatures establishes API presence only, not device correctness.

qwen_grip.py now saves nki-guidance.txt and its hash for every new generation.
Copy BOTH qwen_grip.py and nki_guidance.py to the seat before running it.
The additional card uses model context; reduce output budget or explicitly
adjust server context if needed. Truncated candidates are still rejected.
All previous request/response/code/score artifacts remain unchanged.

The existing deterministic physics feedback supplies errors and hypotheses.
An extra LLM is not required and never decides physics acceptance. The card
is guidance, not a comprehensive static API checker or execution sandbox.

## Candidate-Specific Advice

At 256 updates beta=.85 passes the same eight cases as beta=.9 but makes the
remaining force errors worse. Therefore a statement that lowering beta improved
stability is unsupported. Retain measured results, not the model's comment.
Momentum selection/conditioning are numerical questions, not prescribed by NKI
documentation. Repacking, fusion and fewer DMA copies cannot by themselves fix
a mathematically unconverged output at the same recurrence/update budget.

Documented tensor_scalar supports up to two arithmetic operations with optional
second operands, which may permit fusion once installed-signature compatibility
is checked. This is a later performance experiment, not a correctness guarantee.
Do not force arbitrary matrices into one shared matmul by treating worlds as
identical: each has its own A and b. Any packed independent-world strategy must
preserve that distinction, layout bounds and the original objective.

Next feedback should report stagnation versus continued progress across update
budgets, detect same-source duplicates, and distinguish convergence limitations
from potential FP32 floors. These diagnoses need measured trajectories; they
must not be invented from one endpoint. No profile or trajectory capture is
claimed by the documentation card itself.
