# Roofline tool for Trainium2

Given an operation and a shape, `roofline.py` works out the fastest the hardware could possibly run it
(the **floor**) and what sets that floor: the matrix engine (TensorE), the vector engines (VectorE,
ScalarE), or HBM bandwidth. Compare a kernel's predicted latency against the floor, and you get a
message the model writing the kernel can act on:

> softmax over last axis 128x4096 float32: memory (HBM bandwidth) bound. Hardware floor 11.18 us …
> Your kernel is predicted at 28.58 us: 2.6x the hardware floor. A perfect kernel would be predicted at
> about 14.00 us, so 2.0x is yours to remove. It moves 6144 KiB, 1.50x the minimum -- some input is
> loaded more than once; keep it in SBUF.

`roofline.py` itself is pure Python and needs no Neuron SDK. Everything else runs in a seat pod.

## Run it

On your laptop, with kubectl working (Part 1 of the workshop repo's README). Use your own seat number:

```bash
kubectl cp roofline seat-42:/root/roofline -c app
kubectl exec seat-42 -c app -- bash -c 'cd /root/roofline && python roofline.py'
```

| Script | What it does | Needs |
|---|---|---|
| `roofline.py` | Floors for softmax, the AlexNet conv block, matmul, avgpool and transpose. `feedback()` builds the message | Nothing |
| `calibrate.py` | Simulates and compiles the reference kernels (levels 1–4), a slower level-4 variant and three softmax kernels, and compares each prediction with its floor | Seat pod (reads `/workspace/projects/02-kernel-agent`) |
| `predict.py` | Compiles a kernel for trn2 with no device and returns the predicted latency and engine utilization | Seat pod |
| `softmax_kernels.py`, `test_softmax.py` | Three correct softmax kernels (fused, input loaded twice, extra passes), checked with `nki.simulate` | Seat pod |
| `peaks_probe.py`, `peaks_probe2.py` | Measure the rates the compiler's latency model charges, by comparing predicted times | Seat pod |
| `lnc2_check.py` | Checks that predictions are the same at LNC 1 and 2 for unsharded kernels | Seat pod |

The probe and calibration scripts write only to `/tmp`. `results/` holds the output from our run on
2026-10-10.

Predicted latency comes from the compiler, with no chip needed, in about 0.15 s per kernel:

```python
from predict import predict
predict(kernel, {"lhsT": a, "rhs": b})   # -> {"ns": ..., "util": {"DMA": ..., "Tensor": ..., "Vector": ...}, ...}
```

## Two floors

The compiler's latency model charges different rates than the hardware docs, so the tool reports both:

- **HW floor**: from the documented peaks, assuming perfect overlap and no fixed costs. A true lower bound.
- **Model floor**: what a perfect kernel would be *predicted* at. `predicted / model floor` is the part
  of the gap the kernel's author can still remove.

| Resource | Hardware docs | What the compiler's model charges |
|---|---|---|
| TensorE | bf16 78.6 TFLOP/s, fp32 20 TFLOP/s | 78.6 TFLOP/s for any dtype, plus about 231 ns per matmul |
| VectorE | 256 elem/cycle fp32, 512 bf16, at 0.96 GHz | 128 elem/cycle for any dtype, plus about 430 ns per instruction |
| ScalarE | 128 elem/cycle at 1.2 GHz | Same, plus about 320 ns per instruction |
| HBM | 3 TB/s shared by 8 cores, so 375 GB/s per core | 368 GB/s, plus 2.6 µs for a load followed by a store |

Sources: the Neuron docs shipped in every seat pod, under
`/opt/conda/lib/python3.13/site-packages/neuron_agentic_development/artifacts/skills/neuron-nki-docs/references/`
(`architecture/trainium2_arch.md` lines 8, 10, 124–132; `optimization/nki_perf_guide.md`;
`programming/tutorials/matrix_multiplication.md`), and `nki/compiler/ncc_driver.py` lines 490–503.
Every number in `roofline.py` cites its line.

**Ridge point** (FLOPs per HBM byte where compute takes over from memory): about **210** for bf16 and
**53** for fp32 on trn2. The 222 in `nkibench.py` is the trn1 figure.

## Results

| Operation | Floor | Limited by |
|---|---|---|
| Softmax 128×4096, fp32 | 11.2 µs | HBM |
| Softmax 128×4096, bf16 | 5.6 µs | HBM |
| AlexNet conv2 block (64→192, 5×5) on 27×27, fp32 | 31.1 µs | TensorE |
| Same, bf16 | 7.9 µs | TensorE |
| conv1-like block (3→64, 11×11, stride 4) on 63×63, fp32 | 1.11 µs | TensorE, array only 47% occupied |
| Matmul 512×256×1024, fp32 | 13.4 µs | TensorE |

Calibration, predicted latency ÷ model floor (lower is better):

| Kernel | Ratio |
|---|---|
| Level 4 reference, 4 shapes | 1.33–5.24× |
| Level 4 with `TILE_N=128` (correct, slower) | 3.66–14.63× |
| Levels 1–3 references | 1.15–1.72× |
| Softmax fused / input loaded twice / extra passes, 128×4096 | 1.97× / 2.04× / 2.95× |

## Caveats

1. **Not checked against the chip.** On-chip runs fail with "cores busy" while `serve.sh` runs, because
   the model server holds all four logical cores.
2. **The compiler's model ignores dtype.** For fp32 matmul the docs say compute bound and the model says
   memory bound. Running in bf16 avoids the disagreement.
3. **Small kernels are dominated by the 2.6 µs fixed cost**, so for levels 1–2 use the model-floor ratio,
   not the hardware one.
4. **The prediction misses some waste.** It didn't notice the softmax that loads its input twice at
   512×1024. Keep the simulator's byte count (`nkibench.simulate_and_count`) alongside it.
5. **Assumed bandwidth share.** One core is assumed to get 1/8 of the 3 TB/s. If a lone core can use
   more, the floors of memory-bound operations drop.
6. The conv floor assumes the im2col patches are formed on-chip. `nl.divide` simulates but doesn't compile.
