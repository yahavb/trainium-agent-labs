# Level 8 repair on seat 97 — October 10, 2026

## Start here

This is the execution follow-up to [LEVEL8_READONLY_ANALYSIS.md](LEVEL8_READONLY_ANALYSIS.md). That earlier audit inspected seat 95 without changing or running its project. The user then authorized step-by-step repair and checks on **seat 97**.

**Result: Level 8 generation with the explicit buffer/stage plan solved 2/2 runs, both on round 0. All four sampled candidates passed the official three shapes. Each run's selected kernel also passed 24/24 independent cases.** The saved model-generated solution is [level08_attention.py](projects/20-kernel-agent/solved/level08_attention.py); its [provenance](projects/20-kernel-agent/solved/level08_attention.provenance.json) identifies run 1, sample 0. It is separate from the hand-written fixture. Two runs are a smoke evaluation, not a broad solve-rate guarantee.

**Hardware follow-up: the saved model-generated kernel also passed 3/3 official cases on seat 97's actual Trainium2 device, each called twice.** Repeated outputs were bitwise equal and inputs were unchanged. The Qwen server stayed running throughout; a subsequent inference returned HTTP 200. Start with [device-summary.json](projects/20-kernel-agent/results/seat97-repair/device-summary.json) and Step 5 below for the hardware command and evidence. The 24 independent cases above were CPU simulation; only the three official shapes were tested on the device.

Local code is on `team20-kernel-agent`, under `projects/20-kernel-agent`. Seat 97 initially had an older project under `/workspace/projects/02-kernel-agent` and no project 20 directory. We created an isolated working directory:

```text
/workspace/level8-repair-shubham-97
```

The original seat 97 project was left unchanged. Copies of its original `agent.py` and `nkibench.py` are also preserved in the repair directory's `original-seat97/`. Seat 95 remains unchanged by this repair work. The connection uses Kubernetes exec into the `seat-97` pod, not a public SSH host.

The **latest controller version** is in a separate snapshot, `/workspace/level8-repair-shubham-97-planned`. Its core source hash is `d86182cbbed9` and matches the current local core files byte for byte. The first directory retains the earlier experiment unchanged.

Never put AWS keys or session tokens in this document or the repo. No credential files were created for this work.

## Step 1: establish the environment

- Pod: `seat-97`; namespace: `default`; EKS cluster: `hack-hyd`; region: `ap-south-2`.
- Python: 3.13.7. NKI SDK is installed.
- Existing model server responds at `http://localhost:8000/v1` with `Qwen/Qwen3-8B` and an 8192-token context limit.
- No kernel-agent process was running when inspected. The existing model server was preserved.
- The current local controller, harness, and linter were copied into the isolated directory; they are newer than seat 97's original project.

## Step 2: verify a hand-written attention fixture

File: [attention_kernel.py](projects/20-kernel-agent/examples/attention_kernel.py).

This fixture was written explicitly as a correctness baseline. It is **not a model-generated solve**, and its complete source is not passed to the model in the fresh-generation experiment.

The task is:

```text
Q, K, V: (N, D)
S = Q @ K.T / sqrt(D)                 # (N, N)
E = exp(S - row_max(S))
P = E / row_sum(E)                    # (N, N)
O = P @ V                            # (N, D)
```

NKI `nc_matmul` computes `stationary.T @ moving`; it reads SBUF and writes PSUM. Therefore scores use actual Q.T and K.T, while the final product uses actual P.T and V. The final PSUM result goes through SBUF before its HBM store. This agrees with the [official nc_matmul API](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.isa.nc_matmul.html) and [kernel optimization tutorial](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/nki/guides/tutorials/kernel-optimization.html). The installed SDK and executed simulator were used to verify the calls in this fixture.

Executed on seat 97:

```bash
cd /workspace/level8-repair-shubham-97
python3 -B nkibench.py --level 8 --check examples/attention_kernel.py
```

Result: **3/3 official shapes passed**, rules clean.

| Shape (N,D) | Simulated HBM bytes | Harness minimum |
|---|---:|---:|
| (128,64) | 131072 | 131072 |
| (64,128) | 131072 | 131072 |
| (96,32) | 49152 | 49152 |

These counts equal reading Q/K/V once and storing O once. This is simulated traffic, not measured Trainium latency, a compiler success claim, or a throughput speedup.

## Step 3: strengthen controller checks and test independently

Changes in [agent.py](projects/20-kernel-agent/agent.py):

- All Level 8 first-prompt modes explain that inputs are not pretransposed and pin the formula, intermediate layouts, output shape, stable softmax, scalar scaling, and memory flow.
- Level 8 repair prompts retain that contract and allow fixing dependent stages together.
- With `--lint`, equal official rewards are ranked by passing the static gate, then fewer static issues, then novelty. Official reward weights are unchanged.
- The linter receives the level so it can check the Level 8 return shape without imposing it on other levels.

Changes in [lint.py](projects/20-kernel-agent/lint.py):

- Reject calls to the SDK dtype identifiers `nl.float32`, `nl.float16`, and `nl.bfloat16`.
- Reject nonexistent `nisa.sum`, with the valid row-reduction form in feedback.
- Catch PSUM DMA sources/destinations.
- Track allocation shapes supplied through `shape=` as well as positional arguments.
- Check a directly known Level 8 output shape against Q's shape.
- Copy feedback preserves the task's output contract instead of blindly changing output allocation.
- Treat a discarded `_` dimension as unknown. This avoids a false positive on an existing Level 3 kernel when recognizing keyword allocation shapes.
- Reject the unsupported `nc_matmul(..., transpose_moving=...)` keyword observed in the first seat-97 generation experiment.

Regression tests: [test_attention_repairs.py](projects/20-kernel-agent/tests/test_attention_repairs.py). **All eight pass locally and in the latest seat-97 snapshot**. Reviewing all nine existing saved kernels found no newly introduced lint findings after the discarded-dimension correction.

Independent SDK cases: [check_attention.py](projects/20-kernel-agent/tests/check_attention.py).

```bash
python3 -B tests/check_attention.py examples/attention_kernel.py \
  --json results/fixture-independent.json
```

Result on seat 97: **24/24 passed**. These include three fresh seeds for each official shape, zero queries (output equals mean V), constant V, large finite logits, joint K/V permutations (output invariant), and shapes (32,48), (16,16), and (1,8). Each case also checks input preservation and the harness's recorded hardware-hazard warnings. Numeric acceptance uses the existing harness threshold: max absolute error divided by RMS reference at most 0.02. These tests do not establish arbitrary sequence-length or dtype support.

## Step 4: evaluate the actual model agent

The initial two-run experiment was launched with:

```bash
python3 -u -B agent.py --level 8 --rounds 4 --samples 2 --repeat 2 \
  --context 8192 --max-tokens 3000 \
  --base http://localhost:8000/v1 --model Qwen/Qwen3-8B \
  --lint --level-hints --portfolio --echo-check \
  --log results/agent-repair.jsonl
```

It uses no `--seed-from`, no `--blocks`, and no complete hand-written attention solution. Result: **0/2 runs solved, 16 candidates, no successful executions or correct candidates, best reward 0.30**. Selected static-issue counts were 6,6,5,5 in each run. The repairs changed some code but retained HBM compute inputs, conflicting layouts, and invented API usage. This shorter contract/API guidance did not solve the generation problem.

The experiment source hash is `628ce7219feb` (SHA256 over agent.py, nkibench.py, lint.py). The later discarded-dimension correction is not part of this completed experiment; do not silently mix source versions.

Two simple, distinct marker prompts to the model endpoint both returned exactly the requested marker with HTTP 200. These probes establish basic request/response behavior, not general coding capability or model quality.

### Second experiment: explicit buffer and stage plan

The latest controller adds a named buffer inventory and 13-stage attention plan. This gives substantially more algorithmic guidance than the first experiment, including separate Q/K transpose buffers, NxN scores/probabilities, input DMA, and the final HBM allocation/store. The model still writes the complete Python kernel; it does not receive the complete hand-written fixture. Report any success as **generation with an explicit attention plan**, not unconstrained discovery of the algorithm.

The second snapshot also includes the discarded-dimension correction and the unsupported transpose-keyword check. It used the same flags and limits as above, from the `-planned` directory, with `--log results/agent-planned.jsonl`. Hash: `d86182cbbed9`. **Result: 2/2 runs solved on round 0; all four candidates scored 1.00.** The prompt carried 1441 tokens, and completions ranged from 1023 to 1051 tokens. Each selected kernel passed the independent 24-case suite, with maximum normalized error 6.40e-6 and minimum simulated HBM traffic in every case.

Separately, the controller's `grade()` was executed on the hand-written fixture in this snapshot: reward **1.00**, all four parts true, feedback `Correct on every shape.` This confirms that the repaired controller accepts a correct attention implementation; it is not a model-generation result.

The 24 independent fixture cases had maximum normalized error **6.40e-6**, below the harness limit 0.02. All 24 simulated traffic counts equalled their input-plus-output floor.

### Experiment comparison and interpretation

| Configuration | Controller hash | Candidates | Correct candidates | Runs solved | Selected independent checks |
|---|---|---:|---:|---:|---|
| Contract and API guidance | `628ce7219feb` | 16 | 0 | 0/2 | No passing candidate |
| Contract plus explicit buffer/stage plan | `d86182cbbed9` | 4 | 4 | 2/2 | 24/24 for each selected kernel |

The first configuration used 25570 prompt tokens and 10021 completion tokens across its 16 requests. The successful configuration used 5764 prompt tokens and 4167 completion tokens across four requests. These are totals across each experiment, not single context windows or latency measurements. The maximum prompt-plus-completion count in either configuration was 2492, well inside the server's 8192-token limit; every completion ended with `stop`. These results support fixing layout guidance and generation composition before spending effort on context compression for this particular failure.

The successful configuration also added two checker corrections; this is not a controlled ablation isolating the stage plan as the sole cause. The recorded candidates and stronger static checks make the outcome reviewable. The attention reference, its tolerance, and official reward weights were not relaxed.

## Saved evidence and how to rerun

Local evidence directory: [projects/20-kernel-agent/results/seat97-repair](projects/20-kernel-agent/results/seat97-repair).

- [summary.json](projects/20-kernel-agent/results/seat97-repair/summary.json): counts, token totals, error/traffic summaries, source hash, file hashes, and confirmation that the original seat-97 files still match their preserved copies.
- [agent-planned.jsonl](projects/20-kernel-agent/results/seat97-repair/agent-planned.jsonl): all four successful model candidates, selection, grades, telemetry, flags, and provenance.
- [agent-planned.log](projects/20-kernel-agent/results/seat97-repair/agent-planned.log): completed two-run console output.
- [agent-contract.jsonl](projects/20-kernel-agent/results/seat97-repair/agent-contract.jsonl): all 16 failed candidates from the shorter-guidance configuration.
- [generated-run1-independent.json](projects/20-kernel-agent/results/seat97-repair/generated-run1-independent.json): the saved solution's 24 numerical/invariant/input-preservation/traffic results.
- The directory also contains both generated kernels, run 0's independent checks, the fixture checks, endpoint probes, and regression output.

To check the saved model-generated kernel on seat 97:

```bash
cd /workspace/level8-repair-shubham-97-planned
python3 -B nkibench.py --level 8 --check solved/level08_attention.py
python3 -B tests/check_attention.py solved/level08_attention.py
```

To generate again, use the Step 4 command in this directory with a **new log filename** such as `results/agent-next.jsonl`. Do not seed from the saved Level 8 solution when measuring fresh generation. On the local laptop, run the CPU-only regression suite with:

```bash
python3 -B -m unittest discover -s projects/20-kernel-agent/tests -v
```

The local laptop lacks the NKI SDK, so the actual kernel simulation checks ran on seat 97. All model and test jobs launched for these two experiments completed. Existing model-serving infrastructure was not restarted.

## Step 5: execute on actual Trainium2 hardware

Execution took place in `/workspace/level8-repair-shubham-97-planned` on the `seat-97` pod, hosted on `trn2.48xlarge`, using NKI `0.6.0+31049202112.g85070674`. The saved solution was unchanged: SHA256 `50840792fec8c59fec2d34a3f29b9f92e86db7ac943f61ad4c6c859728a070e3`.

The runner [check_attention_device.py](projects/20-kernel-agent/tests/check_attention_device.py) calls `kernel[2](*args)` through `nki.jit` with NumPy inputs. This invokes standalone compilation and Neuron Runtime device execution. It does not invoke the CPU simulator. It compares against the NumPy reference, checks input preservation, and calls each shape twice to check repeatability.

Explicit attempts to pin logical core 0 or 1 failed during runtime initialization with `NRT_FAILURE` and a cores-busy message, before numerical execution. Memory usage alone did not establish core availability. Requesting automatic allocation of one free logical core succeeded. The selected core ID was not captured. The [Neuron Runtime configuration guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-runtime/guides/configuration-guide.html) documents `NEURON_RT_NUM_CORES` for allocating a requested number of available cores.

Successful command (use a new JSON filename when repeating):

```bash
cd /workspace/level8-repair-shubham-97-planned
env -u NEURON_RT_VISIBLE_CORES \
  NEURON_RT_NUM_CORES=1 NEURON_LOGICAL_NC_CONFIG=2 \
  NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \
  python3 -u -B tests/check_attention_device.py solved/level08_attention.py \
  --lnc 2 --suite official --json results/device-auto.json --stop-on-error
```

| Official shape `(N,D)` | Normalized maximum error | Device result |
|---|---:|---|
| `(128,64)` | `1.19955e-5` | Passed twice; repeatable; inputs unchanged |
| `(64,128)` | `1.05806e-5` | Passed twice; repeatable; inputs unchanged |
| `(96,32)` | `1.22213e-5` | Passed twice; repeatable; inputs unchanged |

All three met the existing normalized-error threshold of `0.02`. The runner exited successfully. This establishes numerical correctness for these inputs on hardware; recorded wall times include host, compilation, runtime, and transfer overhead and do not establish isolated kernel latency or a speedup.

The model server was **never stopped or restarted**. User preference was to keep it running, and automatic core allocation made the test possible without interrupting it. PID 103 still existed afterward, and a fresh inference returned `SEAT97_SERVER_KEPT_RUNNING` with HTTP 200.

Hardware evidence is preserved locally under `projects/20-kernel-agent/results/seat97-repair/`:

- [device-auto.json](projects/20-kernel-agent/results/seat97-repair/device-auto.json) and [device-auto.log](projects/20-kernel-agent/results/seat97-repair/device-auto.log): actual execution, per-shape errors, repeatability, input preservation, and call wall times.
- [device-summary.json](projects/20-kernel-agent/results/seat97-repair/device-summary.json): execution environment, kernel and runner hashes, evidence hashes, and server health.
- [device-compile-only.json](projects/20-kernel-agent/results/seat97-repair/device-compile-only.json): a separate successful compilation of all three shapes to NEFF, with artifact sizes and SHA256 hashes. Those binaries remain in the remote `results/device-compile/` directories. This compile-only check did not execute the device; the `device-auto` records establish device execution.
- `device-core0.*` and `device-core1.*`: preserved failed explicit-allocation attempts. `device-environment.json` and `device-server-health.json`: environment and post-test inference evidence.

## Resume without reanalyzing

Read this document and the final results below first. Inspect only the relevant saved candidate and feedback if a new failure appears. Keep the original seat project intact. Run fresh checks from the isolated directory or copy the verified files into a deliberately chosen working directory.

The whole-tile fixture covers the workshop's small float32 cases. Sequence length or feature dimension above 128 requires a tiled design. Do not use its current shape coverage as evidence for long-context attention.

Actual hardware compilation and execution now pass for the three official shapes. Isolated latency benchmarking, profiling, broader hardware input coverage, a paired old/new controller comparison, and a larger solve-rate study remain separate work. No kernel latency benchmark was performed.
