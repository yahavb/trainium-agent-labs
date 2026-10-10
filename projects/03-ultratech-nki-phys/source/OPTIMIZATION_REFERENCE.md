# Documentation-Guided Throughput Loop

This is a small retrieval-augmented generation layer: keyword retrieval over
curated paraphrases of official Neuron documentation, not live browsing or an
embedding database. optimization_knowledge.py selects three relevant cards from
the checker/timing feedback. Each Qwen request saves retrieval.json, source
links, query, selected cards and content hashes. Installed SDK signatures remain
separate evidence; online documentation alone does not validate compilation.

## Optimization Candidates

| Idea | Documentation Basis | Evaluation Requirement |
|---|---|---|
| Remove intermediate copy | tensor_tensor accepts mixed PSUM/SBUF operands | Same update and physics gates; measure device time |
| Fuse supported scalar operations | tensor_scalar supports two operations with scalar/per-partition operands | Preserve alpha; verify installed API; measure device time |
| Preserve correct matmul layout | nc_matmul computes stationary.T @ moving | Packed matrix already contains A.T |
| Profile before broader changes | Performance guide discusses reuse and engine utilization | No bottleneck claim without a profile |

Sources: [NKI ISA reference](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/_modules/nki/isa.html)
and [historical performance guide](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.26.0/general/nki/nki_perf_guide.html).
The latter is used for principles, not current API signatures.

## Full Measured Loop

Run full_plane_loop.py --cores 0,1 --minutes 30 --attempts 4 from the laptop.
The cores must belong exclusively to this seat. The controller manages the
seat-local Qwen server; it may leave it stopped after the final benchmark.

Each proposal undergoes AST review and CPU NKI simulation on plane and pair
scenes. All required cases must pass. Then stop Qwen and refuse device timing
if Neuron processes still own the device. Measure baseline-before, candidate,
baseline-after with identical workloads: eight worlds per scene, 33 contacts,
32 updates, five repeats, 20 warmups and 200 measurements per repeat/mode.
Pre/post physics checks are mandatory. Report device throughput as total worlds
divided by total measured time, not the mean of per-launch rates.

More than 10% baseline mean-latency drift invalidates the performance reward.
This threshold is an experimental guard, not statistical significance. Saved
reports include every timing sample, repeat spread, compilation context, output
and source snapshot. next-feedback.txt provides measured ratios to the next Qwen
request. The fastest valid observed candidate is retained; the baseline remains
best if none exceeds it. A slower correct candidate is not hidden.

Only two changed, human-reviewed forms currently execute unattended: copy
removal and the permitted multiply/add subtraction. The search stops after
both are tested, or the budget is exhausted. Broader documentation-inspired
optimizations need review or actual isolation. This is not unrestricted discovery,
weight training or a proof of global optimality. Model restarts consume the
wall-clock budget. Remote execution has not been verified from the agent session
because AWS credentials are unavailable here; use the user's working terminal.

## First Device Result

Full run fb1b0f0aa75c4645ae36731a27f7181a attempt 0 passed both scenes but
achieved 21,211.84 worlds/second versus the paired baseline's 26,218.27:
0.809048x throughput. Baseline drift was below 0.1% in each scene. The baseline
remains best. Subsequent proposals repeated the fused form and were rejected
as duplicates, with unmeasured physics/timing. Prompts now explicitly select the
untested reviewed form instead of offering both forms again. Original logs and
negative results are retained. No measured improvement has been demonstrated.

## Copy-Removal Device Result

Subsequent run ca51b1815ec0403caf90df1234bb9aa6 tested the saved copy-removal
proposal e9331d05e5529337a7ddfd8a854f325ff3d7b4b547e79466059e487e85ce376d.
All plane/pair physics gates passed before and after device timing. The combined
device throughput was 26,573.52 worlds/second versus 26,218.53 for the paired
baseline: 1.013540x, or an observed 1.354% improvement. Plane and pairs measured
1.312% and 1.396% improvements respectively.

Five candidate timing repeats per scene, 200 samples per repeat: plane repeat
mean batch latency ranged 0.30108247..0.30109759 ms; pairs ranged
0.300942665..0.301067 ms. Baseline-before/after mean drift was 0.0130% for plane
and 0.0403% for pairs. The measured gain exceeds these observed within-run
variations, but one experiment is not a statistical significance assessment or
independent-run replication. Timing excludes compilation, packing, transfers
and readback as described above. No full-simulator speedup is claimed.

The candidate was Qwen-produced from explicit human copy-removal guidance and
reviewed templates; do not present the optimization as autonomous discovery.
The earlier fused slowdown and duplicate/rejected attempts remain in the log.

## Replication and Scaling/Update Fusion

Run 9f23118d8ff945a9a99c892d07271528 independently repeated copy-removal at
1.013489x (26,572.31 versus 26,218.65 worlds/s). Its next Qwen proposal matched
the newly reviewed scaling/update-fusion form and passed all physics gates.
It measured 1.010582x versus the paired original baseline, but 0.997028x versus
the paired copy-removal reference. Copy-removal remains best. Both comparisons
passed the baseline-drift guard. These are observed ratios, not a significance
test or optimality claim. Each result, including the non-improvement, is preserved.
