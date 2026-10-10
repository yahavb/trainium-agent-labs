# CHIPBOOST status: P1 (`referee-timing`)

*Updated Oct 10 2026. Historical measurements below are from seat-100 (trn2.48xlarge node,
1 Trainium2 device, 4 logical cores, LNC=2, 96 GB). Current acceptance results are recorded in P1-HANDOFF.md.*

## Summary

| Item | State |
|---|---|
| 13:30 gate: on-chip timing works | **PASSED** |
| `timing.py`: device-side timer | Done, selftest passes |
| `speedcheck.py`: the referee | Done, verified on the start kernel, a faster case and a cheat |
| Hardened referee regression | Historical results saved in `results_p1.json` (see below) |
| Throughput implementation | Done in `76b2227`: worker, parallel compilation, threaded inputs |
| Current signoff | See `P1-HANDOFF.md` for scope, evidence, and remaining limitations |
| Original 72-evaluation comparison | Complete, strict report passed: neither Qwen arm found a speedup; template-control median best 3.125919x |
| Recovery pilot v3 | Complete: 8 attempts, all `wrong`; no verified improvement |
| Canceled treatments | Reasoning run: zero graded attempts; queued DMA-only v2 superseded, no comparison results |
| Consolidated Qwen-only v2 | Running on core 2, budget 8; attempt 3 reported approximately 1.517x `faster`, provisional pending final results/replay |

See [RESULTS-SUMMARY.md](RESULTS-SUMMARY.md) for exact measurements, provenance, and cohort limitations.
Original v1: referee 22 `wrong` + 2 `no_gain`; model alone 2 `wrong` + 22 `no_gain`;
template control 22 `faster` + 2 `slower`. Each arm has 24 attempts across three runs.
Full evidence and the strict report are in `experiments/qwen-v1-comparison/`; one infrastructure
interruption is excluded from these counts. The control uses an expert-template prior, not Qwen generation.
The deliverable is Qwen-only (plus random-search control); earlier non-Qwen side trials are retained
only in the archive appendix and excluded from deliverable aggregates. Original comparison results remain
pinned to earlier feedback and must not be presented as validation of subsequent feedback fixes.

### Feedback correction

The outer-loop reuse diagnosis now lives in `speedcheck.py`'s `instruction_given`, rather than only
in the simulator fallback used by P3. Recognized excess rhs traffic is tied to reuse across `m`;
remaining lhsT traffic is tied to reuse across `n`. The instruction explicitly preserves distinct
K tiles and the contraction loop. Conservative AST guards avoid guessing for unfamiliar kernels.
CPU coverage: `python projects/03-chipboost/tests/test_feedback.py`.

Hardware verification on seat-100/core 2 passed all four representative cases: baseline, reordered
loops, lhsT already hoisted, and rhs already hoisted. Each remained correct on the chip and returned
the expected diagnosis through `instruction_given`; records and source hash are in `feedback_validation.json`.
Repeat with `CHIPBOOST_SEAT=100 python verify_feedback.py --core 2 --out /tmp/feedback-validation.json`
from this directory after checking that core 2 is free. These are feedback acceptance checks, not model
optimization results.

The comparison launched from `434e5f9` remains pinned to the earlier feedback and cannot establish
whether this correction improves model outcomes. Its results must be labeled accordingly.

### Subsequent failure feedback and canceled DMA-only treatment

The first targeted simulator DMA instruction was replayed against the observed
`src=65536, dst=16384` failure. Subsequent work generalizes DMA mismatch guidance and adds
conservative fallback and PSUM/compiler repair advice. Candidate-controlled exception text remains
data, and these feedback changes do not turn a failed candidate into an accepted verdict.
The fixes are pushed in `7da33ee`; local `python -m unittest discover -s projects/03-chipboost/tests -p 'test_*.py'`
passed 36 tests. This is not a measured model improvement. Final seat/core-2 checks in
`research/final_feedback_validation.json` verified DMA, NKI tile-list, and PSUM instructions;
the corresponding failures remained `wrong`, and the baseline returned `no_gain`.

The queued DMA-only v2 comparison was canceled as superseded and produced no comparison results.
The separate full-feedback/repair-controller v3 pilot completed eight evaluations, all `wrong`.
The reasoning experiment was canceled for the deadline with zero graded evaluations. Cancellations
are operational events, not kernel failures and not part of the original comparison's attempt count.
The consolidated Qwen-only v2 run is a separate launched treatment, not the canceled DMA-only queue:
PID `884773`, core 2, output `/tmp/p1-qwen-v2-20261010-1`, eight planned evaluations.
The real failing candidate was replayed through the sandboxed referee on seat-100/core 2:
it remained `wrong` and returned the named instruction. This is a simulator rejection before
device timing, not a speed measurement. Evidence and the isolated source diff are in
`experiments/dma-feedback-v2/`; protocol and limitations are in `REFEREE-V2.md`.

## Environment facts (measured, not assumed)

| Fact | Value | Why it matters |
|---|---|---|
| Cores used by vLLM (Qwen3-8B, TP=2) | **0-1** (the repo assumed 2-3) | Kernels are timed on **core 2** (fallback 3) |
| NKI version | 0.6.0, neuronx-cc 2.27, SDK 2.32 | `nki.benchmark` is gone |
| Timing API used | `SpikeModel.benchmark(mode="device")`, NeuronCore trace | Excludes compile and host overhead |
| Compile time per kernel | ~2-2.6 s | Hundreds of candidates per hour are affordable |
| Host-side timing vs device timing | 70-77 us vs 24.8 us (small matmul) | Timing from Python is ~3x wrong |
| Fixed launch cost | ~17 us | Toy shapes measure overhead; use Qwen3 shapes |
| Interference from vLLM under load | **0.0%** (43 samples, 4 requests in flight) | Timing beside the model server is safe |
| `neuron-profile`, `neuron-bench` | Installed at `/opt/aws/neuron/bin` | Available for profiles later |

## Timer measurements (`python timing.py --selftest`)

Start kernel = `reference_level4.py` (tiled matmul), bf16 inputs.

| Shape (K, M, N) | Median | Noise (IQR) | Throughput | Correct |
|---|---|---|---|---|
| 512, 256, 1024 | 24.8 us | 2.3-2.9% | 10.8 TFLOP/s | yes |
| 512, 256, 2048 | 32.5 us | 2.4% | 16.5 TFLOP/s | yes |
| **4096, 256, 2048** (Qwen3 q_proj per core) | **268.4 us** | **0.2%** | 16.0 TFLOP/s | yes |
| **4096, 256, 6144** (Qwen3 gate/up per core) | **691.7 us** | **0.0%** | 18.6 TFLOP/s | yes |

- **Scaling:** 3x the work took 2.58x the time at Qwen3 shapes (1.31x for 2x at toy shapes, because of the fixed cost).
- **A/A:** the same kernel against itself, interleaved, gives 1.000.
- **Headroom:** the start kernel reaches 16-19 TFLOP/s; the agent has room to improve.

## Referee (`speedcheck.py`)

Current stages, stopping at the first failure: rules -> simulator (bytes over **all** DMA ops, inputs untouched)
-> chip correctness at the timing shapes -> interleaved timing vs baseline -> held-out with hostile values
only for a would-be `faster` -> one named change. The current noise floor is 1%, with `no_gain` inside the
band. The tables below retain the original measurements and verdict names from before that change.

| Test | Result |
|---|---|
| Start kernel vs itself | 1.000x -> **slower** (below the 1.05 noise threshold), correct ✓ |
| Start kernel vs a narrow-tile (TILE_N=128) baseline | **2.95x -> faster** ✓ (2.74x and 3.03x per shape) |
| Cheat: never writes its output | **wrong** at the simulator: 100% NaN ✓ |
| Honest bf16 output error | 0.50 ulps (limit 4) ✓ |
| JSON record against `schema.py` | valid ✓ |
| `check_isolated` (fresh process per candidate) | works ✓ |
| Wall time per full check | **~32 s** (6 compiles, 2 timing shapes x 3 interleaved rounds) |

**Shapes:**
- Timing: (4096, 256, 2048) q_proj and (4096, 256, 6144) gate/up, Qwen3-8B per core under tensor parallel 2.
- Held-out: (6144, 256, 4096) down_proj, (2048, 256, 4096) o_proj, (4096, 512, 2048), (4096, 128, 6144), with hostile values.
- Simulator: (256, 512, 1024), (512, 256, 2048).

**Precision:** judged in bf16 ulps against an fp32 reference, limit 4. The old relative-to-RMS bar (2e-2) was
nearly failed by the start kernel itself (1.6e-2 on hostile inputs).

## Findings the team should know

1. **Redundant DMAs can be free on the chip.** Loading every rhs tile twice changed nothing (960.2 us both
   ways); the compiler removed or overlapped it. The simulator's byte count ("1.41x the floor") is a hint,
   not a cost. Only device timing decides.
2. **vLLM holds cores 0-1** on the seat pods, not 2-3 as the repo's docs say.
3. **Never time from Python:** host timing reads ~3x the real kernel time.
4. **Matmul held-out shapes must be tile multiples:** `reference_level4.py` asserts K, M % 128 and N % 512.

## How to use it

```bash
cd /workspace/projects/03-chipboost
python timing.py --selftest --shapes qwen
python speedcheck.py --op matmul --check <kernel.py>                    # human readable
python speedcheck.py --op matmul --check <kernel.py> --json --log attempts.jsonl
```

From Python (the agent): `speedcheck.check_isolated(path, op="matmul")` returns one schema record.

## Load and speed test (core 3, original referee, no code changes)

**Verdict: the timer and the referee can be trusted, idle or under vLLM load.**

| Test | n | Result |
|---|---|---|
| A/A (start kernel vs itself), idle + loaded, in-process + isolated | 45 | **all `slower`, never a false `faster`**; speedup 0.99971-1.00021 (sd 0.011%) |
| vs slow baseline (TILE_N=128) | 18 | all `faster`, **2.9488-2.9501x** (sd 0.04%) |
| Device time, idle vs vLLM under load (4 x 1520-token requests) | | medians moved <= 0.04%; IQR unchanged |
| vLLM throughput, load only vs load + referee | 71 samples | 22.0 vs 21.9 tok/s: **< 1% impact**, 0 errors |
| 20 in-process checks in a row | | device memory peaks 130 MB per check, back to 5 MB between; no leak, no slowdown |
| Infinite loop / compile blow-up / near-hang on device | | `wrong` at the timeout; core usable immediately after |
| SIGKILL mid-benchmark with a model loaded | 2 | core usable; `timing.py --selftest` passes afterwards |
| Timer linearity (same NEFF repeated 1-32x) | | 10.56 us + 260.54 us/repeat, **R^2 = 0.9999998** |

**Where a check's time goes** (in-process, 24.4 s): **compile 19.9 s (81%)**, input generation 2.2 s,
simulator 0.7 s, reference + checks 0.8 s, **timing 0.7 s (3%)**. Each `check_isolated` adds a 6-13 s runtime
start. About 109 candidates/hour today. Compiles parallelise: 8 compiles take 25.0 s serially, 8.5 s with 8 threads.

**Threshold finding:** max(5%, 2xIQR) always resolves to 5%, about 170x the measured A/A noise. So a real
2-4% speedup is reported `slower`. Use max(1%, 2xIQR) plus `no_gain` (master's schema already has it).

**Speedups, ranked (historical estimate together: ~400-600 candidates/hour, not a measured throughput claim):**
1. Held-out only for a would-be `faster`. **Done** in the hardened referee.
2. Persistent referee worker with a watchdog: **implemented**, runtime start paid once per worker.
3. Baseline NEFF cached on disk. Persistent baseline models/input caching remains a possible optimization.
4. Compile in parallel threads: **implemented**, up to 8; input generation is threaded too.
5. Fail fast on a candidate over 3x slower: **implemented**. Child wall timeout remains 600 s, not 180 s.

## Hardening: what the verification agents found, and what changed

Three independent checks attacked the first referee. Every hole below let a wrong or unfair kernel score.

**Red-team round 1** (25 cheat kernels, 4 honest variants). The four honest variants were never rejected.

| Hole | Cheat | Old result | Fix |
|---|---|---|---|
| C1 | computes only when its output still holds the NaN poison, then skips work in timed runs | **FASTER 26x** | every timed run starts from fresh random garbage and is verified |
| C2 | module-level code monkeypatches the referee it is imported into | **FASTER 23x** | the referee never imports the candidate; a child process does |
| C3 | `atexit` prints a forged "faster" record | **FASTER 9.6x** | the record comes back through a parent-named file and must match the exit code |
| M1 | correct only on the 8 published shapes | accepted | held-out shapes are drawn at random from every tile multiple, per check |
| M2 | `--no-heldout` gave an unmarked accept | accepted | the flag is removed; held-out always runs before a "faster" |
| L1 | clamps its output only at the timing shapes | accepted | timing shapes are also checked with hostile inputs |
| L2 | `dma_compute` (positional), aliased imports, `nl.load`/`store` | 0 bytes counted | every DMA entry point is hooked; aliased imports are a rule violation |

**Red-team round 2** (against the child-process design, while the child ran as root):
- `np._core._add_newdocs_scalars.os.system(...)` gave root shell from inside the kernel body, with no forbidden name in sight.
- A `setsid` survivor outlived the process-group kill.
- `np.tofile` wrote any file.
- The baseline NEFF cache had a predictable name, so a slower baseline could be planted.
- New files in the referee folder were never compared.
- Exception text from the candidate reached the agent's instruction (prompt injection).

**Lesson:** a source scan cannot contain Python. The boundary has to be the operating system.

**Code review** (read-only):
- Timed runs reused one input set, so a kernel could return a result cached from an earlier call.
- Candidate errors could crash the referee instead of producing a verdict.
- Infrastructure failures, such as a busy core, were graded "wrong".
- An explicit baseline could be silently replaced.
- Plus leaked temp directories, an exposed seed, and the byte floor computed at the wrong dtype.

**The referee now:**
- **Sandboxed child.** The child runs under a fresh unprivileged uid of its own, with memory, CPU, file-size and process-count limits and no NeuronCore. Every process of that uid is killed afterwards.
- **Tamper checks.** The whole referee tree is snapshotted, so added, removed or changed files are caught. The baseline cache is root-only, and world-writable parent directories are locked down.
- **Unpredictable inputs.** The seed never leaves the referee. Every timed run gets a different input set in an unpredictable order and is verified, and inputs are read back after timing.
- **Stricter "faster".** It requires beating the noise on the total, no regression on any shape, and passing random held-out shapes.
- **Clean failures.** Candidate error text is quoted as data, never turned into an instruction. Infrastructure failures raise `RefereeError`, which is never logged as a verdict.

### Final verification of the hardened referee (core 2, full regression on the final code)

**Every test gives the expected result.** Rows are in `results_p1.json`.

| Group | Result |
|---|---|
| Honest kernels (8) | all accepted: h1 1.000x; **h2 1.541x, h3 1.203x, c5a 1.857x, c5b 1.858x, c5c 1.319x FASTER** with held-out passed; c5e, h4 1.000x |
| Round-1 cheats (21) | all rejected (rules / simulator / chip) |
| `c3a` (wrong only off the timing shapes, not faster) | `slower` by design: held-out is checked before any `faster` |
| Round-2 attacks a1-a4 + a new `np.f2py.os.posix_spawnp` shell | all rejected, **no side effects**, no survivors |
| Sandbox probes (rules bypassed, fed straight to the child) | runs as a fresh uid with **zero capabilities, NoNewPrivs**; PermissionError on /workspace, the referee, the cache and site-packages; fork bomb stopped at 2047 by RLIMIT_NPROC; 60 GB allocation -> MemoryError; setsid daemon and a pid-hopper both killed; device init fails |
| Busy core (`CHIPBOOST_CORE=0`) | `check_isolated` -> `None` (referee error, not a verdict) |
| Infinite loop at trace time | `wrong` at the timeout, with how it ended |

**Bugs that verification found and fixed:**
1. Tamper-snapshot noise from `__pycache__` turned honest kernels into `rules`.
2. `nl.load`/`nl.store` were rejected as file operations.
3. The child could swap its workdir for a symlink. Now the workdir is root-owned and only `out/` is writable.
4. A malformed `result.json` could crash the referee.
5. Tensor names from the NEFF could inject text into `instruction_given`.

Plus: the kill runs as the sandbox uid (atomic against forks), the referee reaps its orphans, and the NKI compile cache is no longer world-writable.

**Still open:**
- `/dev/neuron0` is 0666. The child's lack of a core depends on `NEURON_RT_VISIBLE_CORES` plus vLLM holding cores 0-1; nothing enforces it.
- c5a/c5b (`dma_compute` loads) at ~1.86x deserve a human look at why.
- A `check_isolated` timeout kills the referee but leaves its child bounded only by RLIMIT_CPU.

## Handoff

P2 (`origin/kernels-search` at `919c6be`) already includes the current referee, kernels/shapes, and uses
`RefereeWorker`. P3 (`origin/redteam-agent` at `2ce9416`) already includes the current referee and uses
`check_isolated`, retries infrastructure failures, and sends only `instruction_given` to the model.
See `P1-HANDOFF.md` for current validation evidence and the command to repeat acceptance on a seat.
