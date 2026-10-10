# Documentation-Backed Optimization Backlog

Priority is widening the agent's evidence-guided choices, not training Qwen's
weights. Every technique below is a hypothesis until it passes physics and
repeated device timing. New forms require review before unattended execution.

## 1. Actually Fuse Scaling and the Impulse Update

Current copy-removal iteration forms gradient=product+b, scaled=gradient*rate,
updated=impulses-scaled, then projects updated to nonnegative values. Precompute
negative_rate=-rate once per world, then combine the middle two instructions:
updated=gradient*negative_rate+impulses. Keep projection separate.

This removes an arithmetic instruction per update rather than merely expressing
subtraction as multiply/add, which was the slower previously tested variant.
Preserve the per-world step size and test changed FP32 rounding. Actual engine
selection and runtime cost still require installed-SDK simulation and profiling.
Basis: [tensor_scalar ISA reference](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/_modules/nki/isa.html).

## 2. Hoist Invariant Arithmetic

Try computing bias*rate once per world rather than every update, if an equivalent
instruction schedule benefits from that form. The algebra is
x-rate*(A*x+b)=x-rate*(A*x)-rate*b. Matrix, bias and rate are constant throughout
the solve, but impulses are not. Reassociation can change FP32 errors; preserve
checker thresholds. Do not assume hoisting helps if it adds more instructions.
Basis: [temporal locality and fusion guide](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.26.0/general/nki/nki_perf_guide.html).

## 3. Choose the Right Engine for Tiny Elementwise Tiles

Compare supported explicit engine choices with compiler-selected placement for
scalar operations. tensor_tensor's supported engine choices are not the same as
tensor_scalar's; never mechanically force every operation onto Scalar Engine.
Our state tiles have a free dimension of one, suggesting instruction overhead
may matter, but no device profile currently proves a bottleneck.
Basis: [ISA engine restrictions](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/_modules/nki/isa.html).

## 4. Interleave Independent Worlds

Investigate loading multiple worlds' state and scheduling work across them so
Tensor, Vector and Scalar engines can overlap independent operations. Iterations
within one world depend on the previous result and cannot simply be parallelized.
Worlds have different matrices: do not replace eight independent matvecs with a
shared-matrix GEMM unless the layout mathematically preserves every problem.
Memory pressure and synchronization can erase any gain.
Basis: [engine pipelining guide](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.26.0/general/nki/nki_perf_guide.html).

## 5. Better Tiles or a Different Matvec Mapping

Compare legal tile layouts for these small matrix-vector operations. A vector
multiply/reduce formulation may avoid Tensor Engine setup, but can cost more
work or introduce transposes. Cropping known padded zeros is another hypothesis
for the fixed 33-contact task; output shape/padding and benchmark workload must
remain unchanged. Do not assume padding reduction or more occupied lanes wins.
Basis: [tile efficiency guide](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.26.0/general/nki/nki_perf_guide.html).

## 6. Profile Before Buffer and DMA Changes

Check for engine idle gaps, semaphores, intermediate spills and redundant loads.
Our baseline already keeps matrix/state on chip; we should not claim another
reuse win without identifying a redundant transfer. Source-level ndarray calls
inside the loop do not automatically imply repeated physical allocation costs.
Buffer reuse or double buffering is useful only if the trace identifies a reason.
The installed SDK lacks the old nki.benchmark export, so do not assume older
nki.profile examples work unchanged. Inspect supported CLI/runtime interfaces.
Basis: [official profiling workflow](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.26.0/general/nki/neuron_profile_for_nki.html).

## Reference Projects

[AWS NKI Samples](https://github.com/aws-neuron/nki-samples) and
[AWS NKI Library](https://github.com/aws-neuron/nki-library) provide real kernel
patterns, including fusion and tiling. Their workloads and SDK versions may
differ; borrow verified patterns rather than claiming their reported benefits
transfer to our physics kernel.

## Next Experiment Protocol

First replicate copy-removal's 1.354% observed gain in an independent run.
Then add only technique 1 as a newly reviewed template, retrieve its source-linked
guidance for Qwen, and compare it with both the original and copy-removal baselines.
Require all plane/pair gates, unchanged updates/workloads, five repeats and raw
timing samples. Log rejections and slowdowns. Do not advertise gains in advance.

## Prepared Next Run

The new reviewed scale-fused form is implemented in nki_contact_scale_fused.py.
The full loop now requests it after copy-removal replication, and compares it
with both paired original and paired copy-removal baselines. Its source math
has a CPU arithmetic test; installed NKI simulation/device results are pending.
No new speedup is claimed. Use seat-260 only while other team seats are busy.

parallel_plane_runs.py is prepared for future independent experiments, but no
other seats have been used. It requires authorized seat IDs and confirmed cores,
rejects inventories exposing the same physical chip, and keeps baselines paired
within each seat. Raw rates across different chips are not one pooled speedup.
