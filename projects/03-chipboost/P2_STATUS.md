# CHIPBOOST status: P2 (`kernels-search`)

*Updated Oct 10 2026, about 13:30. Measured on seat-102 with P1's referee (`speedcheck.py`: device clock,
interleaved A/B), branch `kernels-search` at 2fb343c. Every number below is from the chip unless marked.*

## Summary

| Item | State |
|---|---|
| `shapes.py`: dev / timing / held-out shapes, in the referee's spec format | Done. The referee runs `--op matmul, rmsnorm, copy, swiglu` |
| `kernels/matmul_start.py` (the baseline) | Done. Correct on the chip |
| `kernels/matmul_expert.py` (the expert ceiling) | Done. **2.49x faster** than start; correct on 8 chip shapes, worst 0.50 ulps |
| `kernels/matmul_expert_aws.py` (AWS as published) | Done. **Fails the referee**: 5.3 bf16 ulps (limit 4) |
| `kernels/rmsnorm_start.py` | Done. Correct on 7 chip shapes: ragged rows, the eps trap, a 1-token decode |
| `kernels/copy_tiled.py`, `kernels/copy_floor.py` (RMSNorm's floor) | Tiled floor measured; packed floor written, not yet run |
| `search.py` (arm c: random search, no AI) | In progress: second agent, branch `p2-tools` |
| `results_p2.json` (for the dashboard) | Done, from the numbers below |

## Measured

**Matmul**, Qwen3-8B per-core shapes at 256 prompt tokens:

| Kernel | gate_up 4096x256x6144 | q_proj 4096x256x2048 | Sum | Verdict |
|---|---|---|---|---|
| `matmul_start` | 691.7 us (18.6 TFLOP/s) | 268.8 us (16.0 TFLOP/s) | 960.5 us | baseline |
| `matmul_expert` | **280.4 us (45.9 TFLOP/s)** | **104.7 us (41.0 TFLOP/s)** | **385.1 us** | **faster, 2.494x** |
| `matmul_expert_aws` | not timed | not timed | not timed | **wrong: 5.3 bf16 ulps** |

**RMSNorm** against its floor (a copy of the same bytes):

| Kernel | input_layernorm 256x4096 | q_norm 4096x128 | Sum |
|---|---|---|---|
| `rmsnorm_start` | 46.8 us | 67.2 us | 114.3 us |
| `copy_tiled` (floor) | 20.5 us | 51.6 us | 72.1 us |
| **Room left** | **2.28x** | **1.30x** | 1.59x |

- **Reproducible across seats:** the start kernel measured 691.7 / 268.8 us here and 691.7 / 268.4 us on P1's
  seat-100.
- **Correctness was checked everywhere, not only where the timing was taken.** That covers the 6 held-out
  matmul shapes with hostile values: odd 10/5/5 tile counts, a single N-tile, K=6144 in 6 blocks. For RMSNorm,
  the 5 held-out shapes are 127, 129 and 1 rows, a ragged 1000x128, and 2048 rows, all with quiet, loud and
  silent rows.

## Findings worth telling the room

1. **AWS's published fully optimised matmul loses precision at Qwen3's sizes.**
   - The SDK 2.32 tutorial adds each K-block's result into a tile of the *output* dtype. In bf16, every
     output is rounded once per block.
   - Its own test uses K=1024, a single block, so that path is never exercised.
   - The referee measured **5.3 bf16 ulps at K=2048** (limit 4, honest rounding is about 1).
   - Emulated on the laptop, it reaches **9 to 13 ulps at K=4096 and 6144**. Our fp32-accumulating version
     measured 0.50.
   - The fix costs SBUF, not speed. Our version is the 2.49x one.
2. **Hostile inputs hide precision loss.** Under P1's hostile pattern (large magnitudes, zeros, sign flips),
   the same bf16 accumulation measured only 3.4 to 3.5 ulps at K=4096 (emulated), under the limit. A
   referee must judge precision on ordinary inputs too. Ours does, at the timing shapes.
3. **The precision catch happened in the simulator** because the dev shapes include K=2048, which is two
   K-blocks. Without that shape it would have been found only on the chip.
4. **The byte counter overstated dtype-converting DMAs by 2x.**
   - A float32 SBUF -> bf16 HBM store was counted at its float32 size, so the expert matmul looked like
     1.29x the floor while moving exactly the floor.
   - Fixed in nkibench (count `min(src, dst)`). The referee's own counter still counts `src`.
5. **Narrow rows starve the DMAs.**
   - Copying 4096x128 bf16 (2 MB) took 51.6 us; 256x4096 (4 MB) took 20.5 us.
   - q_norm's rows are 256 bytes, so a 128-row tile moves 32 KB per DMA.
   - Packing 32 rows per partition is the obvious fix. It is the packed `copy_floor`, and later an RMSNorm
     move.
6. **A float32 tolerance rejects correct bf16 kernels.** A correct bf16 matmul one rounding step from the
   reference measured 2.2% of the output's RMS, over the old 2e-2 bar. Levels 9-12 carry their own
   tolerance, and the referee judges bf16 in ulps.

## Reproduce (in the seat pod)

```bash
cd /workspace/chipboost/projects/03-chipboost && export CHIPBOOST_SEAT=102
python tools/probe_nki.py --sim-only                                           # every kernel, simulator
python speedcheck.py --op matmul  --check kernels/matmul_expert.py             # 2.49x, correct
python speedcheck.py --op matmul  --check kernels/matmul_expert_aws.py         # wrong: precision loss
python speedcheck.py --op rmsnorm --check kernels/rmsnorm_start.py             # start time (vs itself)
python speedcheck.py --op copy --check kernels/copy_tiled.py --baseline kernels/copy_tiled.py
python speedcheck.py --op copy --check kernels/copy_floor.py --baseline kernels/copy_tiled.py
```

## Not claimed

- **No matmul floor and no end-to-end Qwen number.** The expert is measured against our start kernel, not
  against the production kernels vLLM-Neuron uses.
- **The Amdahl share of matmul in a Qwen3 layer needs a profile of the served model, which we have not
  taken.** Any end-to-end figure we quote will be labelled a projection.

## Next

1. Run the packed `copy_floor` on the chip and update `results_p2.json`.
2. Merge `p2-tools` and run `search.py` against the referee on the same budget as the agent arms.
3. Stretch: `kernels/swiglu_start.py`.
