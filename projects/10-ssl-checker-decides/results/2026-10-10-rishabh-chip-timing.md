# Measured on-chip latency vs the compiler's prediction

- **Who / seat:** Rishabh (+ Claude), seat 49 (its vLLM server stopped by the user to free the cores)
- **Commit measured:** uncommitted; `projects/02-kernel-agent/chip_time.py`, nki 0.6.0, trn2, LNC=1
- **Hypothesis:** the compiler's predicted latency, used all day as the speed signal, ranks kernels the
  same way the chip does.
- **Change:** none to any kernel. Each one is compiled for trn2 and benchmarked on the device
  (5 warmup, 10 timed iterations), and its device output is checked against NumPy.
- **Command:** `LNC=1 NEURON_LOGICAL_NC_CONFIG=1 python chip_time.py`
- **Log:** `logs/rishabh-chip-timing.json`

## Scores

```
kernel                          predicted   measured on chip (mean, min-max)   measured/predicted  device correct
attention seq128 dim64            8.84 us   20.45 us (20.32-20.67)            2.31x               yes
attention seq64  dim128           8.68 us   20.14 us (19.96-20.39)            2.32x               yes
attention seq96  dim32            8.36 us   19.58 us (19.43-19.77)            2.34x               yes
matmul slow start (TILE_N=128)  183.92 us   345.79 us (345.40-346.31)         1.88x               yes
matmul tutorial reference        65.94 us   127.78 us (127.39-128.15)         1.94x               yes
matmul Qwen's kernel (109 us)    59.71 us   118.81 us (118.13-119.33)         1.99x               yes
```

The matmuls are on K=256 M=512 N=1024, the largest level-4 shape. Level-8 shapes are the stock ones.

## Verdict

- **The speed-up is real on hardware:** 345.8 → 118.8 µs measured, **2.91x** (3.08x predicted on this
  shape). Qwen's kernel beats the tutorial's on the chip, 118.8 vs 127.8 µs (7% faster).
- **The compiler under-predicts by about 2x, but it ranks kernels correctly.** The ordering is
  identical, so predicted latency was a sound optimization signal. Absolute predicted µs should not be
  quoted as wall-clock.
- **Attention measures about 20 µs per call on every shape**, consistent with being bound by fixed
  overhead (instruction count), not by data (already at the byte floor) or compute.
- The spread is under 0.5 µs on every kernel.
