# Kernels run on the real Trainium chip (seat-115, 2026-10-10)

**What was run.** `daykit/agent/nki/check/device_check.py` (the script is `run_device.sh`), with the model server
stopped so the NeuronCores were free.
- **Settings:** `NEURON_PLATFORM_TARGET_OVERRIDE=trn2`, `NEURON_LOGICAL_NC_CONFIG=1`.
- **Inputs:** each kernel ran on every graded shape of its level, with `nkibench.make_inputs` (the
  grader's own inputs and seed).
- **Two runs per shape:** first in the NKI CPU simulator, as the agent was graded; then on a NeuronCore.
  In nki 0.6.0, calling a `@nki.jit` kernel with NumPy arrays compiles it with neuronx-cc and runs it
  standalone on the device.
- **Pass/fail:** both outputs were compared with the NumPy reference at nkibench's tolerance: worst
  error ≤ 2% of the reference's RMS.

**The kernels:**
- **Positive controls:** the organizers' `reference_level1/3/4.py`.
- **Kernels written by Qwen3-8B:** the hand-in pick per level from the 4090 rehearsal (task 13,
  `kernels_tested/`).
- **Negative controls:** the picks for levels 1 and 10, which our compile check had marked REJECTED BY
  COMPILER.

| kernel | written by | our verdict before | simulator | real chip |
|---|---|---|---|---|
| reference level 1 (pooling) | organizers | – | 4/4 shapes | **4/4** (worst error 2–5e-7 of RMS) |
| reference level 3 (matmul, 1 tile) | organizers | – | 1/1 | **1/1** |
| reference level 4 (matmul, tiled) | organizers | – | 4/4 | **4/4** |
| level 2, transpose | Qwen3-8B | VERIFIED, compiles | 4/4 | **4/4** (exact) |
| level 3, matmul, 1 tile | Qwen3-8B | VERIFIED, compiles | 1/1 | **1/1** |
| level 4, matmul, tiled | Qwen3-8B | VERIFIED, compiles | 4/4 | **4/4** |
| level 9, row softmax (made under the compiler gate) | Qwen3-8B | VERIFIED, compiles | 3/3 | **3/3** (2–5e-5 of RMS) |
| level 11, gated SiLU | Qwen3-8B | VERIFIED, compiles | 3/3 | **3/3** |
| level 1, pooling | Qwen3-8B | REJECTED BY COMPILER | 4/4 | **0/4**: neuronx-cc fails (exit 70) |
| level 10, layer norm (`op=nl.divide`) | Qwen3-8B | REJECTED BY COMPILER | 3/3 | **0/3**: MLIR verification fails |

**Results:**
- **All 15 shapes of the 5 model-written kernels our pipeline calls VERIFIED and compiling run correctly
  on the chip.**
- **All 7 shapes of the 2 kernels it calls REJECTED BY COMPILER fail to build for the chip,** although the
  simulator passes every one of them.
- **The controls validate the harness:** the organizers' references pass all 9 of their shapes.

So the compile check (`seat.sh compile`, the verdicts, the gate) predicted on-chip behaviour exactly for
these 7 kernels (22 kernel-shapes): 15 of 15 correct, 7 of 7 not buildable. The CPU simulator alone would
have called all 22 correct.

**Timing is not measured here.** Each standalone call takes about 1.2 s whatever the kernel. That is
the launch (loading the compiled program each call), not the kernel's run time. The exception is the
model's level-2 transpose: about 6.8 s per call on the (128, 64) and (64, 128) shapes, which suggests many
small transfers. Real kernel latency needs the profiler (`neuron-explorer capture` on each kernel's NEFF).

Files:
- `device.log`: one line per kernel and shape;
- `device.json`: the same, as data;
- `device_reps20.log`: the first try, with 20 timed calls per shape, stopped after level 1;
- `run_device.sh`;
- `kernels_tested/`: the exact kernels that were run.
