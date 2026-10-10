# CHIPBOOST status: P1 (`referee-timing`)

*Updated Oct 10 2026, measured on seat-100 (trn2.48xlarge node, 1 Trainium2 device, 4 logical cores, LNC=2, 96 GB).*

## Summary

| Item | State |
|---|---|
| 13:30 gate: on-chip timing works | **PASSED** |
| `timing.py`: device-side timer | Done, selftest passes |
| `speedcheck.py`: the referee | Done, verified on the start kernel, a faster case and a cheat |
| Red-team suite (P3), full validation | In progress (verification agents) |

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

Stages, stopping at the first failure: rules -> simulator (bytes over **all** DMA ops, inputs untouched)
-> chip correctness at the timing shapes -> held-out shapes with hostile values -> interleaved timing vs the
baseline -> one named change.

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

## Next

- Verification agents: red-team cheats, false-reject tests, load and speed tests (results to be added here).
- Hand to P2: put `kernels/matmul_start.py` and `shapes.py` in place (the referee falls back to
  `reference_level4.py` and its built-in matmul spec until then).
- Hand to P3: wire `agent.py` grading to `speedcheck.check_isolated`.
